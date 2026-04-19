# RAG Evaluation Guide

How to evaluate Retrieval-Augmented Generation (RAG) systems — covering the RAG Triad,
retrieval ranking metrics, failure mode diagnosis, and implementation patterns.

**Prerequisites:** Familiarity with LLM-as-judge evaluation and component error analysis
from `../agents/1-agent_evaluation_guide.md`.

---

## Why RAG Evaluation Is Different

Standard agent evaluation asks: "Is the output correct?"

RAG evaluation asks three separate questions — one for each system component:

```
                    ┌─────────────┐
                    │    Query    │
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │       Retriever         │
              │  (finds context chunks) │
              └────────────┬────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
  Context Relevance   Groundedness      Answer Relevance
  ─────────────────   ────────────      ────────────────
  "Does the retrieved  "Is the answer   "Does the answer
  context actually     only based on    actually address
  contain what is      the context?     the user's
  needed?"             (no halluc.)"    question?"

  Retriever metric    Generator metric  End-to-end metric
```

Each leg of this triad can fail independently. A high end-to-end score can hide a
broken retriever (the generator got lucky with parametric memory). Each must be
measured separately.

---

## The RAG Triad

### Metric Definitions

#### Context Relevance (Retrieval Quality)
- **What**: Fraction of retrieved chunks that are relevant to the query
- **Failure mode**: Retriever returns topically related but not specifically useful content
- **Formula**: `relevant_chunks / total_retrieved_chunks`
- **Target**: ≥ 0.7

#### Groundedness / Faithfulness (Hallucination Detection)
- **What**: Every claim in the answer can be traced back to a retrieved chunk
- **Failure mode**: Model adds facts not present in context ("hallucination")
- **How measured**: LLM judge checks each claim against context; returns pass/fail per claim
- **Formula**: `grounded_claims / total_claims_in_answer`
- **Target**: ≥ 0.8 (higher for regulated domains)

#### Answer Relevance (End-to-End Quality)
- **What**: Does the final answer directly address what the user asked?
- **Failure mode**: Answer is factually correct but doesn't actually answer the question
- **How measured**: LLM judge with query + answer, no context needed
- **Formula**: Cosine similarity between question embedding and answer embedding, or LLM score 0-1
- **Target**: ≥ 0.8

---

## Diagnosing Failures with the RAG Triad

Use this table to identify which component to fix:

| Context Relevance | Groundedness | Answer Relevance | Root Cause |
|:-:|:-:|:-:|---|
| Low | Any | Any | **Retriever** — wrong chunks fetched. Fix: embedding model, chunk size, top-k |
| High | Low | Any | **Generator** — hallucinating despite good context. Fix: prompt constraints, temperature |
| High | High | Low | **Alignment** — answer is factual but off-topic. Fix: system prompt, query reformulation |
| High | High | High | System working correctly |

---

## Retrieval Ranking Metrics

Beyond relevance, measure the *quality of ranking* — relevant chunks should appear early.
A system that retrieves the right chunk at rank 10 is far worse than one that retrieves
it at rank 1.

| Metric | What it measures | Formula | Target |
|--------|-----------------|---------|--------|
| **Precision@K** | Fraction of top-K results that are relevant | relevant in top-K / K | ≥ 0.7 |
| **Recall@K** | Fraction of all relevant docs retrieved in top-K | relevant retrieved / total relevant | ≥ 0.8 |
| **MRR** (Mean Reciprocal Rank) | How quickly the first relevant result appears | average of 1/rank for first relevant doc | ≥ 0.8 means relevant doc is rank 1-2 |
| **NDCG** (Normalised Discounted Cumulative Gain) | Relevance weighted by rank position — penalises relevant docs buried lower | logarithmic position discount | ≥ 0.8 |

---

## Context Neglect — A Distinct Failure Mode

Hallucination and context neglect are different problems with different fixes:

```
Hallucination:       Context is present but model invents facts not in it
                     Fix: lower temperature, stricter prompt, faithfulness threshold

Context Neglect:     Context is present and correct, but model ignores it
                     and answers from parametric (training) memory instead
                     Fix: stronger grounding instruction in system prompt,
                          RAG-specific prompt template, temperature tuning
```

Context neglect is confirmed by Google DeepMind research. Detect it by checking whether
the answer would change if you removed the retrieved context entirely — if it would not,
the model is ignoring the retrieval.

---

## Context Window Realism

Always evaluate with the same context window constraints you use in production.
Evaluating with full un-truncated context overestimates real performance by 25-30%.

```python
# Bad: eval with 50 chunks (never happens in prod)
eval_context = all_retrieved_chunks[:50]

# Good: eval with the same truncation logic production uses
MAX_CONTEXT_TOKENS = 8000
eval_context = truncate_to_token_limit(retrieved_chunks, MAX_CONTEXT_TOKENS)
```

---

## Additional RAG Metrics

Beyond the core triad:

| Metric | What it measures |
|--------|-----------------|
| **Context Precision** | Of retrieved chunks, what fraction are truly relevant (precision not recall) |
| **Context Recall** | Were all relevant chunks actually retrieved? Requires ground-truth context |
| **Answer Correctness** | Answer matches expected answer (requires golden set) |
| **Summarisation Score** | How well the answer synthesises multiple chunks |

---

## Implementation with RAGAS

RAGAS covers the full triad plus context precision/recall in one library.

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,           # groundedness
    answer_relevancy,
    context_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

eval_dataset = Dataset.from_list([
    {
        "question":     "What is SAP AI Core?",
        "answer":       "<agent output>",
        "contexts":     ["<retrieved chunk 1>", "<retrieved chunk 2>"],
        "ground_truth": "<expected answer from golden set>"
    }
])

results = evaluate(eval_dataset, metrics=[
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_precision,
    context_recall,
])
print(results)
# {'faithfulness': 0.85, 'answer_relevancy': 0.91, ...}
```

---

## Implementation with TruLens (RAG Triad)

TruLens provides chain-of-thought reasoning explanations alongside each score.

```python
from trulens.apps.langchain import TruChain
from trulens.providers.openai import OpenAI as TruOpenAI

provider = TruOpenAI()

f_groundedness      = Feedback(provider.groundedness_measure_with_cot_reasons)
f_answer_relevance  = Feedback(provider.relevance_with_cot_reasons)
f_context_relevance = Feedback(provider.context_relevance_with_cot_reasons)

tru_recorder = TruChain(
    your_rag_chain,
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance]
)
```

---

## Implementation with DeepEval

DeepEval's `FaithfulnessMetric` uses the QAG pattern (claim extraction + yes/no
verification) which is more reliable than direct LLM scoring.

```python
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="What is SAP AI Core?",
    actual_output="<agent answer>",
    retrieval_context=["<chunk 1>", "<chunk 2>"]
)

faithfulness_metric = FaithfulnessMetric(threshold=0.8)
relevancy_metric    = AnswerRelevancyMetric(threshold=0.8)

faithfulness_metric.measure(test_case)
print(faithfulness_metric.score)    # 0.0 to 1.0
print(faithfulness_metric.reason)
```

---

## References

- [RAGAS documentation](https://docs.ragas.io)
- [TruLens documentation](https://www.trulens.org/docs)
- [DeepEval FaithfulnessMetric](https://deepeval.com/docs/metrics-faithfulness)
- [Maxim AI: Complete Guide to RAG Evaluation Metrics 2025](https://www.getmaxim.ai/articles/complete-guide-to-rag-evaluation-metrics-methods-and-best-practices-for-2025/)
- [Redis: RAG System Evaluation](https://redis.io/blog/rag-system-evaluation/)
- [Openlayer: Measuring RAG Groundedness](https://www.openlayer.com/blog/post/measuring-rag-groundedness-complete-evaluation-guide)
- [Galileo: Top Metrics to Monitor RAG Performance](https://galileo.ai/blog/top-metrics-to-monitor-and-improve-rag-performance)
