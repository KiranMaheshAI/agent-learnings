# Agent Evaluation — Directory Index

Six documents covering agent evaluation end-to-end: from first principles through
production monitoring, advanced frameworks, and CI/CD gating.

---

## Two Paths Through This Material

### Learning Path — read in order

```
1-agent_evaluation_guide.md
  Why agent eval is different · Four autonomy levels · Two axes · Andrew Ng method
  Error analysis loop · Evaluation maturity levels 0-5 · Evaluation flywheel
         ↓
2-evaluation_metrics.ipynb  (run §2-9 now; §10-13 after reading 3 and 5)
  Objective metrics · LLM-as-judge · Trajectory · Component error rates
  Score aggregation · Hyperparameter sweep
         ↓
3-production_evaluation_pattern.md
  How this codebase runs evals · Langfuse + Cassandra · Golden dataset pipeline
  CI/CD quality gate integration
         ↓  (then return to notebook §10-13)
4-establishing-evaluation-framework.md
  Scoring methods (G-Eval/DAG/QAG) · Golden Sets · RAG Triad · Safety
  A/B testing · Latency & cost · Multi-turn · Multi-agent · Robustness · Compliance
```

### Reference Path — jump to what you need

```
5-evaluation_tools.md           ← Pick a tool for any pipeline stage
6-gaps-from-sources.md          ← Research provenance and sourcing notes
0-README.md (this file)         ← Key concepts cross-reference table
```

`5-evaluation_tools.md` is a **reference catalogue**, not a learning step. Open it
when you need to choose or add a tool; do not read it sequentially.

---

## Document Summaries

### `1-agent_evaluation_guide.md`
**What it covers:** Why agent evaluation differs from traditional LLM eval. The four
autonomy levels (generator → tool-calling → planning → autonomous). The two evaluation
axes (objective vs LLM-as-judge × ground truth vs no ground truth). The Andrew Ng
method: gold standard dataset, component error rate tracking, error analysis loop.
Evaluation maturity levels 0–5. Cost awareness. The evaluation flywheel.

**Ends with:** A table mapping each concept to the specific notebook section that
implements it in code.

---

### `2-evaluation_metrics.ipynb`
**What it covers:** 13 runnable sections, no live API required.

**Run §2–9 after reading doc 1:**

| Section | Topic |
|---------|-------|
| §2 | Date extraction accuracy — objective metric with ground truth |
| §3 | Source quality F1-score — research agent sourcing |
| §4 | LLM-as-judge: talking points coverage (Andrew Ng pattern) |
| §5 | LLM-as-judge: rubric scoring (mirrors codebase EvaluationWorkflow) |
| §6 | Trajectory: tool call precision / recall / path efficiency |
| §7 | Component error rate tracking — the "counting up errors" method |
| §8 | Score aggregation — mirrors evaluation_runner.py logic |
| §9 | Hyperparameter sweep — find best search config empirically |

**Run §10–13 after reading docs 3 and 5:**

| Section | Topic |
|---------|-------|
| §10 | RAG Triad: faithfulness (QAG), answer relevancy, context relevance |
| §11 | Consistency / pass^k — mission-critical agent testing |
| §12 | Multi-turn eval: knowledge retention, turn relevancy, completeness |
| §13 | CI/CD quality gate: golden dataset regression check, pass/fail verdict |

---

### `3-production_evaluation_pattern.md`
**What it covers:** The codebase's evaluation architecture end-to-end:
`evaluation_runner.py` → `EvaluationWorkflow` → Langfuse + Cassandra.
Data models, canonical score names, evaluation type registry, production monitoring
thresholds, status lifecycle. The **Langfuse → Golden Dataset Pipeline**: pull
low-scoring traces, annotate, promote to golden dataset, run regression check.

**Ends with:** Guidance on which notebook sections to run next and when to read doc 4 vs doc 5.

---

### `5-evaluation_tools.md` _(reference — consult as needed)_
**What it covers:** 12 tool categories organised by pipeline stage — observability,
offline eval, LLM-as-judge, component testing, trajectory analysis, production
monitoring, human evaluation, A/B testing, RAG/grounding, toxicity/safety, latency/cost,
and golden dataset management. Bold entries are already in use in this codebase.

**Use it:** When choosing a new tool, comparing alternatives, or looking up what's
available for a specific stage. Quick-reference table at the bottom maps every stage
to its top tool options.

---

### `4-establishing-evaluation-framework.md`
**Prerequisites:** Doc 1 (concepts), Doc 2 §2–9 (code), Doc 3 (codebase architecture).

**What it covers — in order:**

1. **Scoring Method Taxonomy** — G-Eval, DAG, QAG, SelfCheckGPT, BLEU/ROUGE, the 5-Metric Rule
2. **Pillar 1: Golden Sets** — silver→gold promotion, adversarial subsets, failure pattern segmentation
3. **Pillar 2: RAG & Grounding** — RAG Triad, retrieval ranking (MRR/NDCG), context neglect, context window realism
4. **Pillar 3: Toxicity & Safety** — dimensions, prompt injection types, red-teaming, agent-specific benchmarks
5. **Pillar 4: A/B Testing** — design, statistical significance, shadow mode, online drift detection
6. **Pillar 5: Latency & Cost** — per-component benchmarks, cost formulas, semantic cache hit rate
7. **Conversational & Multi-Turn Evaluation** — turn faithfulness, knowledge retention, role adherence
8. **Agent Capabilities** — memory/long-horizon, multi-agent collaboration (with codebase `EvaluationWorkflow` examples), robustness testing, pass^k, policy & compliance

---

### `6-gaps-from-sources.md` _(reference)_
**What it covers:** Cross-reference of four external research sources against the prior
docs, listing every topic that was missing and which document it was added to.

**Use it:** When adding new content and checking whether a topic has already been
sourced, or to understand why a specific approach was chosen over alternatives.

---

## Key Concepts Cross-Reference

| Concept | Primary location | Runnable code |
|---------|-----------------|---------------|
| Why agent eval differs | `1` §Why Agent Evaluation Is Different | — |
| Four autonomy levels | `1` §The Four Autonomy Levels | — |
| Two axes of evaluation | `1` §The Two Axes of Evaluation | — |
| Gold standard dataset | `1` §Build a Gold Standard Dataset | `2` §2–3 |
| Component error rates | `1` §Error Analysis | `2` §7 |
| Evaluation maturity levels 0–5 | `1` §Evaluation Maturity Levels | — |
| Evaluation flywheel | `1` §The Evaluation Flywheel | — |
| LLM-as-judge patterns | `1` §Two Axes + `4` §Scoring Methods | `2` §4–5 |
| Trajectory metrics | `1` §Four Autonomy Levels | `2` §6 |
| Hyperparameter sweep | `1` §Improving Performance | `2` §9 |
| Codebase eval architecture | `3` §Codebase Evaluation Architecture | `3` §Key Files |
| Langfuse → Golden Dataset | `3` §Langfuse → Golden Dataset Pipeline | `3` full section |
| CI/CD quality gate | `3` §Step 4 | `2` §13 |
| Scoring methods (G-Eval/DAG/QAG) | `4` §Scoring Method Taxonomy | `2` §4–5, `4` §code examples |
| Golden Sets (silver→gold) | `4` §Pillar 1 | `3` §Langfuse → Golden Dataset |
| RAG Triad | `4` §Pillar 2 | `2` §10 |
| Hallucination / faithfulness (QAG) | `4` §Pillar 2 | `2` §10 |
| Safety / toxicity | `4` §Pillar 3 | `5` §10 |
| Red-teaming | `4` §Pillar 3 | `5` §10 |
| A/B testing + shadow mode | `4` §Pillar 4 | `4` §A/B Test Design |
| Latency & cost benchmarks | `4` §Pillar 5 | `4` §Benchmarking Code |
| Multi-turn eval | `4` §Conversational Eval | `2` §12 |
| pass^k consistency | `4` §Agent Capabilities | `2` §11 |
| Multi-agent collaboration | `4` §Multi-Agent Collaboration | `5` §Codebase Example |
| Robustness testing | `4` §Robustness Testing | — |
| Policy & compliance eval | `4` §Policy & Compliance | — |
| Tool selection by stage | `5` §Quick Reference | — |
