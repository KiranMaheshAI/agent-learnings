# RAG Evaluation — Directory Index

Three documents covering evaluation of Retrieval-Augmented Generation (RAG) systems:
metrics, tools, and a framework for production-grade grounding evaluation.

---

## Reading Order

```
1-rag_evaluation_guide.md       ← Concepts: RAG Triad, retrieval ranking, failure modes
         ↓
1-rag_evaluation_metrics.ipynb  ← Runnable code: faithfulness, answer relevancy, context relevance
         ↓
2-rag_evaluation_tools.md       ← Tool reference: RAGAS, TruLens, DeepEval, Arize, Galileo
```

---

## Relationship to Agent Evaluation

RAG evaluation is a **subset** of agent evaluation — it applies whenever an agent
retrieves context before generating an answer. The two sets of docs are complementary:

| Agent Evaluation (`../agents/`) | RAG Evaluation (this folder) |
|---------------------------------|------------------------------|
| Why agent eval is different | Why RAG eval is different from output-only eval |
| Component error rates | Per-component RAG Triad diagnosis |
| LLM-as-judge patterns | Faithfulness / groundedness (QAG pattern) |
| Trajectory metrics | Retrieval ranking metrics (MRR, NDCG) |
| Production eval architecture | RAG-specific production monitoring signals |

**Start with `../agents/`** if you are new to evaluation entirely.
**Start here** if you already understand agent eval and need RAG-specific depth.

---

## Document Summaries

### `1-rag_evaluation_guide.md`
**What it covers:** The RAG Triad (context relevance, groundedness, answer relevance) —
what each metric measures, how it fails, and how to diagnose which component is broken.
Retrieval ranking metrics (Precision@K, Recall@K, MRR, NDCG). Context neglect as a
distinct failure mode from hallucination. Context window realism. Additional metrics
(context precision/recall, answer correctness, summarisation score). Implementations
with RAGAS, TruLens, and DeepEval.

---

### `1-rag_evaluation_metrics.ipynb`
**What it covers:** Runnable code for the full RAG Triad without a live API:
- Faithfulness (QAG pattern) — claim-by-claim verification against retrieved context
- Answer relevancy — keyword-overlap proxy with production upgrade path
- Context relevance — per-chunk relevance scoring
- `evaluate_rag_triad()` — runs all three and returns a diagnosis

Replace `simulate_judge()` with a real LLM call for production use.

---

### `2-rag_evaluation_tools.md`
**What it covers:** Tools specifically for RAG evaluation — RAGAS, TruLens, DeepEval
FaithfulnessMetric, Arize Phoenix, Galileo, Openlayer — with a comparison table and
guidance on when to use each.

---

## Key Concepts

| Concept | Location |
|---------|----------|
| RAG Triad definition | `1-rag_evaluation_guide.md` §The RAG Triad |
| Context relevance metric | `1-rag_evaluation_guide.md` §Metric Definitions |
| Groundedness / faithfulness | `1-rag_evaluation_guide.md` §Metric Definitions |
| Answer relevance metric | `1-rag_evaluation_guide.md` §Metric Definitions |
| Diagnosing which component broke | `1-rag_evaluation_guide.md` §Diagnosing Failures |
| MRR, NDCG, Precision@K | `1-rag_evaluation_guide.md` §Retrieval Ranking Metrics |
| Context neglect vs hallucination | `1-rag_evaluation_guide.md` §Context Neglect |
| Context window realism | `1-rag_evaluation_guide.md` §Context Window Realism |
| Faithfulness code (QAG) | `1-rag_evaluation_metrics.ipynb` §1 |
| Answer relevancy code | `1-rag_evaluation_metrics.ipynb` §1 |
| Context relevance code | `1-rag_evaluation_metrics.ipynb` §1 |
| RAGAS implementation | `1-rag_evaluation_guide.md` §Implementation with RAGAS |
| TruLens implementation | `1-rag_evaluation_guide.md` §Implementation with TruLens |
| Tool selection | `2-rag_evaluation_tools.md` |
