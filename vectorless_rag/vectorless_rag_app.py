"""
Vectorless RAG — standalone application mirroring the experience-generation-server
orchestration/rag/ implementation.

This app demonstrates the same PageIndex RAG pattern from the server
(pageindex_rag.py + retrieval.py + schema.py) but runs self-contained with
only openai + pydantic as dependencies.

Pipeline:
  1. Build tree  — LLM reads markdown/text and produces a hierarchical outline
  2. Store index — Pydantic models (StoredPresentationIndex) hold the result
  3. Retrieve    — LLM reasons over the outline to select relevant node_ids
  4. Answer      — LLM generates a cited, grounded answer from retrieved sections

No vector DB. No embeddings. No chunking.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema — mirrors orchestration/rag/schema.py
# ---------------------------------------------------------------------------


class DocumentStructureEntry(BaseModel):
    doc_id: str
    name: str
    structure: Dict[str, Any] = Field(default_factory=dict)


class IndexSegment(BaseModel):
    node_id: str
    doc_id: str
    title: str
    summary: str = ""
    text: str = ""


class StoredPresentationIndex(BaseModel):
    """Root schema — mirrors orchestration/rag/schema.py:StoredPresentationIndex."""

    documents: List[DocumentStructureEntry] = Field(default_factory=list)
    segments: List[IndexSegment] = Field(default_factory=list)


class RetrievalOutput(BaseModel):
    """Structured output from the retrieval LLM call."""

    node_ids: List[str] = Field(default_factory=list)


class TreeNode(BaseModel):
    title: str = ""
    summary: str = ""
    text: str = ""
    children: List["TreeNode"] = Field(default_factory=list)
    nodes: Optional[List["TreeNode"]] = None


class TreeRoot(BaseModel):
    root: Optional[TreeNode] = None
    nodes: Optional[List[TreeNode]] = None


# ---------------------------------------------------------------------------
# Prompts — mirrors _TREE_PROMPT and _RETRIEVAL_PROMPT from the server
# ---------------------------------------------------------------------------

_TREE_PROMPT = """Act as a Document Architect. Analyze the following document and produce a clear, table-of-contents style hierarchical outline.

Requirements:
- The output must be a single JSON object with either a "root" key (single top-level node) or a "nodes" array (list of top-level nodes).
- Each node must have: "title" (string), "summary" (string, one-sentence summary of that section), "text" (string, content of that section).
- Each node may have "children" or "nodes" (array of child nodes with the same shape).

Example shape: {{"nodes": [{{"title": "Introduction", "summary": "One-sentence summary here.", "text": "...", "children": [...]}}]}}

Document content:
---
{content}
---
Return only the JSON object."""

_RETRIEVAL_PROMPT = """You are given a document outline (list of sections with node_id, title, summary) and a query.

Task: Which sections are most relevant to the query? Reason about which sections best address the query; do not rely only on keyword overlap. Return the node_ids in order of relevance (most relevant first), with at most {top_k} ids.

Document outline:
{outline}

Query: {query}

Return a JSON object with a single key "node_ids" containing an ordered list of node_id strings (at most {top_k})."""

_ANSWER_PROMPT = """You are an expert document analyst.
Answer the question using ONLY the provided context.
For every claim you make, cite the section title in parentheses.
Be concise and precise.

Question: {query}

Context:
{context}

Answer:"""


# ---------------------------------------------------------------------------
# Tree builder — mirrors orchestration/rag/pageindex_rag.py
# ---------------------------------------------------------------------------


def _normalize_tree(parsed: Any, doc_id: str):
    """Normalize tree to consistent {root: [...]} shape. Mirrors _normalize_tree."""
    if isinstance(parsed, list):
        root = parsed
    elif isinstance(parsed, dict):
        if "root" in parsed:
            root = parsed["root"]
        elif "nodes" in parsed:
            root = parsed["nodes"]
        else:
            root = parsed
    else:
        root = parsed

    def normalize_node(n: Any) -> Any:
        if not isinstance(n, dict):
            return n
        out = dict(n)
        if "nodes" in out and "children" not in out:
            out["children"] = out.pop("nodes")
        if "children" in out:
            out["children"] = [normalize_node(c) for c in out["children"]]
        return out

    if isinstance(root, list):
        root = [normalize_node(n) for n in root]
    else:
        root = normalize_node(root)

    structure = {"root": root}
    return root, structure


def _flatten_tree_to_segments(
    node: Any, doc_id: str, prefix: str, segments: List[Dict] | None = None
) -> List[Dict]:
    """Recursively flatten tree to segment dicts. Mirrors _flatten_tree_to_segments."""
    if segments is None:
        segments = []
    if isinstance(node, dict):
        title = node.get("title") or node.get("name") or ""
        summary = node.get("summary") or ""
        text = node.get("text") or node.get("content") or ""
        segments.append(
            {
                "node_id": prefix,
                "doc_id": doc_id,
                "title": str(title),
                "summary": str(summary),
                "text": str(text),
            }
        )
        children = (
            node.get("children") or node.get("child_nodes") or node.get("nodes") or []
        )
        for i, child in enumerate(children):
            _flatten_tree_to_segments(child, doc_id, f"{prefix}_{i}", segments)
    elif isinstance(node, list):
        for i, child in enumerate(node):
            _flatten_tree_to_segments(child, doc_id, f"{prefix}_{i}", segments)
    return segments


def build_index_from_text(
    doc_id: str,
    doc_name: str,
    content: str,
    client: OpenAI,
    model: str = "gpt-4o",
) -> StoredPresentationIndex:
    """Build a StoredPresentationIndex from raw text using the LLM tree builder.

    Mirrors build_index_from_documents() in orchestration/rag/pageindex_rag.py,
    but accepts raw text instead of DocumentInfo objects so no server deps are needed.
    """
    prompt = _TREE_PROMPT.format(content=content[:100_000] or "(empty document)")

    logger.info("Building tree for doc_id=%s doc_name=%r ...", doc_id, doc_name)

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=TreeRoot,
    )
    parsed: TreeRoot = response.choices[0].message.parsed

    dumped = parsed.model_dump()
    if parsed.root is None and parsed.nodes is not None:
        raw = {"nodes": dumped["nodes"]}
    else:
        raw = dumped

    root, structure = _normalize_tree(raw, doc_id)

    if not root or (isinstance(root, list) and len(root) == 0):
        raise ValueError(f"Model returned an empty tree for document {doc_name}")

    segment_dicts = _flatten_tree_to_segments(root, doc_id, prefix=doc_id)
    if not segment_dicts:
        raise ValueError(f"Tree produced no segments for document {doc_name}")

    logger.info("Tree built: %s segments for doc_id=%s", len(segment_dicts), doc_id)

    doc_entry = DocumentStructureEntry(doc_id=doc_id, name=doc_name, structure=structure)
    segments = [IndexSegment(**s) for s in segment_dicts]

    return StoredPresentationIndex(documents=[doc_entry], segments=segments)


# ---------------------------------------------------------------------------
# Retrieval — mirrors orchestration/rag/retrieval.py
# ---------------------------------------------------------------------------

MAX_TOP_K = 20


def _build_outline(index: StoredPresentationIndex) -> str:
    lines = []
    for seg in index.segments:
        lines.append(
            f"  - node_id: {seg.node_id!r}  title: {seg.title!r}  summary: {seg.summary!r}"
        )
    return "\n".join(lines) if lines else "(no sections)"


def retrieve_for_query(
    index: StoredPresentationIndex,
    query: str,
    top_k: int = 5,
    client: OpenAI = None,
    model: str = "gpt-4o",
) -> List[IndexSegment]:
    """Return the most relevant segments for the query using LLM reasoning.

    Mirrors retrieve_for_query() in orchestration/rag/retrieval.py.
    """
    top_k = min(max(1, top_k), MAX_TOP_K)
    if not index.segments or not query:
        return []

    outline = _build_outline(index)
    prompt = _RETRIEVAL_PROMPT.format(outline=outline, query=query, top_k=top_k)

    logger.info("Retrieving top_%s segments for query=%r ...", top_k, query[:80])

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=RetrievalOutput,
    )
    result: RetrievalOutput = response.choices[0].message.parsed

    node_ids = [str(x) for x in (result.node_ids or [])][:top_k]
    segment_map = {seg.node_id: seg for seg in index.segments}

    retrieved = []
    for nid in node_ids:
        seg = segment_map.get(nid)
        if seg is None:
            logger.warning("Model returned unknown node_id %r — skipping", nid)
            continue
        retrieved.append(seg)

    logger.info(
        "Retrieved %s segments: %s",
        len(retrieved),
        [s.title for s in retrieved],
    )
    return retrieved


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------


def generate_answer(
    query: str,
    segments: List[IndexSegment],
    client: OpenAI,
    model: str = "gpt-4o",
) -> str:
    """Generate a grounded, cited answer from the retrieved segments."""
    if not segments:
        return "No relevant sections found in the document."

    context_parts = []
    for seg in segments:
        context_parts.append(
            f"[Section: '{seg.title}']\n{seg.text or seg.summary or '(no content)'}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = _ANSWER_PROMPT.format(query=query, context=context)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Expert-guided retrieval — mirrors the expert_rag pattern from the notebook
# ---------------------------------------------------------------------------

_EXPERT_RETRIEVAL_PROMPT = """You are a domain expert analyzing a document.
Find all node IDs that most likely contain the answer to the query.
Use the expert routing rules below to guide your reasoning.

Query: {query}

Document outline:
{outline}

Expert routing rules (follow these carefully):
{rules}

Return a JSON object with a single key "node_ids" containing an ordered list of node_id strings (at most {top_k})."""


def retrieve_with_expert_rules(
    index: StoredPresentationIndex,
    query: str,
    expert_rules: str,
    top_k: int = 5,
    client: OpenAI = None,
    model: str = "gpt-4o",
) -> List[IndexSegment]:
    """Expert-guided retrieval: inject domain routing rules into the retrieval prompt."""
    top_k = min(max(1, top_k), MAX_TOP_K)
    if not index.segments or not query:
        return []

    outline = _build_outline(index)
    prompt = _EXPERT_RETRIEVAL_PROMPT.format(
        query=query, outline=outline, rules=expert_rules, top_k=top_k
    )

    logger.info(
        "Expert-guided retrieval top_%s for query=%r ...", top_k, query[:80]
    )

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=RetrievalOutput,
    )
    result: RetrievalOutput = response.choices[0].message.parsed

    node_ids = [str(x) for x in (result.node_ids or [])][:top_k]
    segment_map = {seg.node_id: seg for seg in index.segments}

    retrieved = []
    for nid in node_ids:
        seg = segment_map.get(nid)
        if seg is None:
            logger.warning("Model returned unknown node_id %r — skipping", nid)
            continue
        retrieved.append(seg)

    logger.info(
        "Expert retrieval returned %s segments: %s",
        len(retrieved),
        [s.title for s in retrieved],
    )
    return retrieved


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def vectorless_rag(
    query: str,
    index: StoredPresentationIndex,
    client: OpenAI,
    model: str = "gpt-4o",
    top_k: int = 5,
    expert_rules: str | None = None,
) -> str:
    """End-to-end vectorless RAG pipeline.

    Step 1: Retrieve — LLM reasons over outline to select node_ids
    Step 2: Fetch    — collect matching IndexSegment objects
    Step 3: Answer   — LLM generates cited answer from segment text

    Optionally inject expert_rules to guide retrieval without embedding fine-tuning.
    """
    if expert_rules:
        segments = retrieve_with_expert_rules(
            index, query, expert_rules, top_k=top_k, client=client, model=model
        )
    else:
        segments = retrieve_for_query(
            index, query, top_k=top_k, client=client, model=model
        )
    return generate_answer(query, segments, client=client, model=model)


# ---------------------------------------------------------------------------
# Index persistence (JSON file — mirrors storage.py without blob dependency)
# ---------------------------------------------------------------------------


def save_index(index: StoredPresentationIndex, path: str) -> None:
    """Serialize index to a local JSON file (replaces blob store for local use)."""
    Path(path).write_text(json.dumps(index.model_dump(mode="json"), indent=2))
    logger.info("Index saved to %s (%s segments)", path, len(index.segments))


def load_index(path: str) -> StoredPresentationIndex:
    """Load index from a local JSON file (replaces blob store for local use)."""
    data = json.loads(Path(path).read_text())
    index = StoredPresentationIndex.model_validate(data)
    logger.info("Index loaded from %s (%s segments)", path, len(index.segments))
    return index


# ---------------------------------------------------------------------------
# Demo — runs when executed directly
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENT = """
# Python Concurrency Guide

## 1. Introduction
Python offers several concurrency models for CPU-bound and I/O-bound workloads.
Choosing the right model is critical for performance.

## 2. Threading
The `threading` module creates OS threads. Because of the Global Interpreter Lock (GIL),
threads cannot execute Python bytecode in parallel, but they excel at I/O-bound tasks
such as network requests, file reads, and database queries.

### 2.1 Thread Pools
`concurrent.futures.ThreadPoolExecutor` manages a pool of worker threads.
Use it when you have many short I/O tasks to parallelise.

## 3. Multiprocessing
`multiprocessing` spawns separate OS processes, each with its own interpreter and memory.
The GIL does not apply, so CPU-bound tasks — image processing, number crunching, ML inference —
can scale across all available cores.

### 3.1 Process Pools
`concurrent.futures.ProcessPoolExecutor` provides a high-level interface identical to
ThreadPoolExecutor but backed by processes.

## 4. Asyncio
`asyncio` is Python's built-in event loop for cooperative multitasking. A single thread
handles thousands of concurrent I/O operations using async/await syntax.
It is the preferred model for high-throughput network services (HTTP servers, WebSockets).

### 4.1 async/await syntax
Coroutines declared with `async def` and awaited with `await` are the building blocks.
`asyncio.gather()` runs multiple coroutines concurrently in the same event loop.

## 5. Choosing the Right Model
- CPU-bound work  → multiprocessing
- I/O-bound work  → asyncio (preferred) or threading
- Legacy blocking I/O → threading (no async support)
- Mixed workloads → asyncio + thread/process pool executors via `loop.run_in_executor`

## 6. Common Pitfalls
- Shared mutable state across threads without locks → race conditions
- Blocking calls inside an asyncio event loop → event loop starvation
- Spawning too many processes → memory overhead
- Forgetting to join/close pools → resource leaks
"""

EXPERT_RULES = """
Routing rules for this Python concurrency document:
- GIL, threads, ThreadPoolExecutor → section 2 (Threading / Thread Pools)
- CPU-bound, processes, ProcessPoolExecutor → section 3 (Multiprocessing)
- asyncio, event loop, async/await, gather → section 4 (Asyncio)
- "which model to use", "when to use" → section 5 (Choosing the Right Model)
- bugs, race conditions, starvation, leaks → section 6 (Pitfalls)
"""


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o")

    index_path = Path(__file__).parent / "demo_index.json"

    # Build (or load cached) index
    if index_path.exists():
        print(f"\nLoading cached index from {index_path.name} ...")
        index = load_index(str(index_path))
    else:
        print("\nBuilding tree index from sample document ...")
        index = build_index_from_text(
            doc_id="python-concurrency",
            doc_name="Python Concurrency Guide",
            content=SAMPLE_DOCUMENT,
            client=client,
            model=model,
        )
        save_index(index, str(index_path))

    # Print tree overview
    print(f"\n{'='*60}")
    print(f"Index: {len(index.documents)} document(s), {len(index.segments)} segments")
    print("Segments:")
    for seg in index.segments:
        print(f"  [{seg.node_id}] {seg.title}")

    # Demo queries
    queries = [
        ("What should I use for CPU-bound tasks?", None),
        ("How does asyncio handle concurrency?", None),
        ("What are the common bugs in concurrent Python code?", EXPERT_RULES),
    ]

    for query, rules in queries:
        print(f"\n{'='*60}")
        print(f"Query : {query}")
        if rules:
            print("(with expert rules)")
        print("-" * 60)
        answer = vectorless_rag(
            query=query,
            index=index,
            client=client,
            model=model,
            top_k=3,
            expert_rules=rules,
        )
        print(answer)

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
