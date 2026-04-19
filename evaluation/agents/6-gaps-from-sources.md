Data Privacy Integration# Gap Analysis — What the External Sources Added

Cross-referencing the four external sources against the existing evaluation docs
revealed the following missing areas. All gaps have been patched into the relevant
documents (noted in the "Added to" column).

Sources reviewed:
1. arXiv 2507.21504 — Comprehensive survey on LLM agent evaluation
2. Confident AI / DeepEval — LLM evaluation metrics guide
3. Maxim AI — Complete RAG evaluation guide 2025
4. Redis — RAG system evaluation

---

## Gap Table

| # | Missing Topic | Source | Added to |
|---|---------------|--------|----------|
| 1 | **Retrieval ranking metrics** — MRR, NDCG, Precision@K never mentioned | Maxim AI, Redis | `6-establishing-evaluation-framework.md` §Pillar 2 |
| 2 | **Semantic cache hit rate** — cost saving from cached queries is unmeasured | Redis | `6-establishing-evaluation-framework.md` §Pillar 5 |
| 3 | **Context neglect** — model ignores retrieved context and uses parametric memory instead | Maxim AI | `6-establishing-evaluation-framework.md` §Pillar 2 |
| 4 | **Scoring method taxonomy** — G-Eval vs DAG vs QAG vs SelfCheckGPT not explained | DeepEval | `6-establishing-evaluation-framework.md` §Scoring Methods |
| 5 | **QAG scorer** — claim extraction + yes/no verification pattern for faithfulness | DeepEval | `6-establishing-evaluation-framework.md` §Pillar 2 |
| 6 | **DAG (Deep Acyclic Graph)** metric — decision-tree eval for deterministic criteria | DeepEval | `6-establishing-evaluation-framework.md` §Scoring Methods |
| 7 | **SelfCheckGPT** — reference-less hallucination via sampling consistency | DeepEval | `6-establishing-evaluation-framework.md` §Scoring Methods |
| 8 | **The 5-Metric Rule** — avoid over-instrumentation (1-2 custom + 2-3 generic) | DeepEval | `6-establishing-evaluation-framework.md` §Metric Selection |
| 9 | **Multi-turn / conversational metrics** — turn faithfulness, turn relevancy, knowledge retention | DeepEval | `6-establishing-evaluation-framework.md` §Conversational Eval |
| 10 | **Memory & context retention** — factual recall across 600+ turn conversations | arXiv survey | `6-establishing-evaluation-framework.md` §Agent Capabilities |
| 11 | **Multi-agent collaboration** — evaluating information-sharing between agents | arXiv survey | `6-establishing-evaluation-framework.md` §Agent Capabilities |
| 12 | **Robustness testing** — paraphrased instructions, typos, misleading context | arXiv survey | `6-establishing-evaluation-framework.md` §Agent Capabilities |
| 13 | **Consistency / pass^k metric** — success across ALL k attempts, not just one | arXiv survey | `6-establishing-evaluation-framework.md` §Agent Capabilities |
| 14 | **Policy & compliance evaluation** — RBAC, GDPR, HIPAA, approval workflows | arXiv survey | `6-establishing-evaluation-framework.md` §Agent Capabilities |
| 15 | **Agent-as-a-Judge** — multi-agent refinement of evaluations (beyond LLM-as-judge) | arXiv survey | `6-establishing-evaluation-framework.md` §Scoring Methods |
| 16 | **Failure pattern segmentation** — slice metrics by query category, not just aggregate | Maxim AI | `6-establishing-evaluation-framework.md` §Pillar 1 |
| 17 | **Context window realism** — evaluate with truncated context, not ideal full context | Maxim AI | `6-establishing-evaluation-framework.md` §Pillar 2 |
| 18 | **Statistical scorers** — BLEU, ROUGE, METEOR, BERTScore definitions and when to use | DeepEval | `6-establishing-evaluation-framework.md` §Scoring Methods |
| 19 | **Safety benchmarks for agents** — AgentBench, AgentDojo, SafeAgentBench, AgentHarm | arXiv survey | `6-establishing-evaluation-framework.md` §Pillar 3 |
| 20 | **Prompt alignment metric** — checks each instruction individually, not the whole prompt | DeepEval | `6-establishing-evaluation-framework.md` §Agent Capabilities |

---

## Summary of Themes

**From arXiv survey (most novel additions):**
- Agent evaluation needs a *two-dimensional taxonomy*: what to assess (behaviour, capabilities,
  reliability, safety) × how to assess (datasets, metrics, contexts)
- Memory and long-horizon testing is a distinct dimension entirely absent from prior docs
- Multi-agent collaboration evaluation is a separate problem from single-agent evaluation
- Robustness (perturbation testing) and consistency (pass^k) are distinct from accuracy
- Policy compliance is a production requirement for enterprise deployments

**From DeepEval blog (scoring method depth):**
- The QAG pattern (question-answer generation for claim verification) is more reliable than
  direct LLM scoring for faithfulness
- DAG metrics provide deterministic paths for clear success/failure criteria
- The 5-metric rule prevents eval noise from over-instrumentation
- Multi-turn metrics are a separate family from single-turn metrics

**From Maxim AI / Redis (RAG depth):**
- Retrieval ranking quality (MRR, NDCG) is separate from retrieval relevance
- Context neglect — where the model ignores retrieved context — is a distinct failure mode
  from hallucination (making something up) and needs its own metric
- Semantic cache hit rate ties quality to cost: caching reduces cost but must be validated
  for quality (cached answer still relevant to slightly different query)
- Failure pattern segmentation by query type reveals systematic weaknesses that aggregate
  metrics hide (simple queries score 0.9, complex synthesis queries score 0.5)
