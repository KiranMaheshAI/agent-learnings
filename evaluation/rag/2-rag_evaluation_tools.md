# RAG Evaluation Tools

Tools specifically for evaluating Retrieval-Augmented Generation (RAG) systems —
covering the RAG Triad (context relevance, groundedness, answer relevance), retrieval
ranking, and production RAG monitoring.

For full metric definitions and code examples, see `1-rag_evaluation_guide.md`.

---

## Tool Comparison

| Tool | Type | Best for |
|------|------|----------|
| **RAGAS** | Open-source Python | Full RAG Triad + context precision/recall in one library |
| **TruLens** | Open-source Python | RAG Triad with chain-of-thought reasoning explanations |
| **DeepEval FaithfulnessMetric** | Open-source Python | Claim-level groundedness via QAG — most reliable faithfulness scorer |
| **Arize Phoenix** | Open-source | RAG trace visualisation with per-chunk relevance scores |
| **Galileo** | Cloud | RAG monitoring with hallucination alerts in production |
| **Openlayer** | Cloud | Groundedness scoring integrated into CI/CD |

---

## When to Use Each

### RAGAS
Use when you want a single library that covers the full triad plus context
precision/recall. Best for offline eval on a golden dataset before deployment.

```python
# Install: pip install ragas datasets
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy
```

**Strengths:** Comprehensive metric coverage, actively maintained, dataset integration.  
**Limitations:** Requires ground-truth answers for some metrics (context_recall, answer_correctness).

---

### TruLens
Use when you need human-readable explanations alongside scores — each metric
returns a chain-of-thought reasoning trace, not just a number.

```python
# Install: pip install trulens-eval
from trulens.providers.openai import OpenAI as TruOpenAI
```

**Strengths:** CoT explanations make failures debuggable; works with LangChain apps.  
**Limitations:** OpenAI dependency for the provider; less flexible metric customisation.

---

### DeepEval FaithfulnessMetric
Use when faithfulness / hallucination detection is your primary concern.
Uses the QAG (Question-Answer Generation) pattern: extracts atomic claims from the
answer, then verifies each claim individually against retrieved context.

```python
# Install: pip install deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric
```

**Strengths:** QAG is more reliable than direct LLM scoring; composable with other
DeepEval metrics (GEval, AnswerRelevancy, HallucinationMetric).  
**Limitations:** Can be slow on long answers with many claims.

---

### Arize Phoenix
Use for local, visual RAG debugging — visualises the full span tree with per-chunk
relevance scores alongside latency and token counts.

**Strengths:** Open-source, runs locally, excellent for trace-level debugging.  
**Limitations:** Not designed for large-scale automated eval; better as a debugging tool.

---

### Galileo
Use for production RAG monitoring — alerts when hallucination rate spikes above
threshold on live traffic without manual trace inspection.

**Strengths:** Automated alerting, production-grade throughput.  
**Limitations:** Cloud-only, paid; overkill for early-stage systems.

---

### Openlayer
Use when you need groundedness scoring gated into CI/CD — scores every PR
against a dataset and fails the pipeline if faithfulness drops below threshold.

**Strengths:** CI/CD-native; combines RAG eval with deployment gating.  
**Limitations:** Cloud-only.

---

## Quick Decision Guide

```
Offline eval on a golden dataset      → RAGAS
Need explanations alongside scores    → TruLens
Faithfulness / hallucination focus    → DeepEval FaithfulnessMetric
Local debugging of individual traces  → Arize Phoenix
Production monitoring / alerting      → Galileo
CI/CD gating on groundedness          → Openlayer
Already using DeepEval for agents     → DeepEval FaithfulnessMetric (consistent toolchain)
```

---

## References

- [RAGAS documentation](https://docs.ragas.io)
- [TruLens documentation](https://www.trulens.org/docs)
- [DeepEval documentation](https://docs.confident-ai.com)
- [Arize Phoenix documentation](https://docs.arize.com/phoenix)
- [Galileo documentation](https://docs.galileo.ai)
- [Openlayer documentation](https://docs.openlayer.com)
