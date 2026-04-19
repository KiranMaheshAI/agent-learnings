# Evaluation Tools Reference

Tools organized by evaluation process stage — from trace capture through production monitoring.
Bold entries indicate tools already in use in the experience-generation-server codebase.

---

## 1. Observability — Capturing Traces, Runs, and Threads

These tools capture what the agent actually did so you have data to evaluate against.
They implement the three observability primitives: Run → Trace → Thread.

| Tool | Type | Best for |
|------|------|----------|
| **Langfuse** | Open-source / Cloud | Trace capture, score ingestion, dashboards — **used in this codebase** |
| LangSmith | Cloud (LangChain) | Deep LangChain/LangGraph integration, trace visualizer |
| Weights & Biases Weave | Cloud | ML-first teams, experiment tracking + agent traces |
| Arize Phoenix | Open-source | Local-first, great for RAG + LLM trace visualization |
| OpenTelemetry | Open standard | Vendor-neutral span/trace export, works with any backend |
| Helicone | Cloud | Lightweight proxy-based logging, zero code change |

### Primitive Mapping

```
Run   (single LLM call)  → captured as a span
Trace (one execution)    → captured as a trace
Thread (full session)    → captured as a session
```

### How Langfuse Is Used in This Codebase

Scores are written to the **original report generation trace** so generation and evaluation
appear together in one trace view:

```
Langfuse Trace (report generation)
├── span: research_phase
├── span: writing_phase
└── span: report_evaluation          ← attached by evaluation_runner.py
    ├── span: completeness_evaluator
    ├── span: structure_evaluator
    ├── span: relevance_evaluator
    └── span: overall_quality_evaluator

Scores on trace:
  report.completeness         = 4.0
  report.structure            = 5.0
  report.overall              = 4.1  (average)
```

Entry point: `core/common/telemetry/langfuse_scores.py :: write_trace_scores()`

---

## 2. Offline Evaluation — Testing Against a Gold Standard Dataset

Tools for running structured evals before deployment, against a curated dataset.

| Tool | Type | Best for |
|------|------|----------|
| **DeepEval** | Open-source Python | GEval, RAG metrics, LLM-as-judge — **used in this codebase** |
| RAGAS | Open-source Python | RAG-specific: faithfulness, answer relevancy, context recall |
| LangChain Evals | Open-source Python | Built-in criteria evaluators, pairwise comparison |
| OpenAI Evals | Open-source | OpenAI-specific, good for structured output evals |
| Promptfoo | Open-source CLI | YAML-driven eval configs, easy CI/CD integration |
| Braintrust | Cloud | Dataset management + eval runs with human annotation |
| Maxim AI | Cloud | End-to-end eval platform with dataset builder |

### DeepEval Metrics Available Out of the Box

```python
from deepeval.metrics import (
    GEval,                    # custom rubric-based LLM judge  ← used in codebase
    AnswerRelevancyMetric,
    FaithfulnessMetric,       # hallucination detection
    ContextualRecallMetric,
    HallucinationMetric,
    ToxicityMetric,
    BiasMetric,
)
```

### Codebase Usage

`evaluations/metrics.py` defines three GEval metrics via `MetricsManager`:
- KPIs and Success Metrics (3-tier rubric: 0-3, 4-7, 8-10)
- Competitive Positioning
- Market Analysis Depth

`evaluations/evaluation_registry.py` registers evaluation type configurations with GEval
thresholds (0.5 to 0.8) and criteria for each evaluation type.

---

## 3. LLM-as-Judge — Subjective Quality Scoring

Tools specifically for using an LLM to score another LLM's output.
Used when there is no programmatic ground truth.

| Tool | Type | Best for |
|------|------|----------|
| **DeepEval GEval** | Open-source | Rubric-based scoring with custom criteria — **used in this codebase** |
| LangChain CriteriaEvalChain | Open-source | Drop-in judge for conciseness, correctness, helpfulness |
| Prometheus-2 | Open-source model | Fine-tuned judge model, no OpenAI dependency |
| MT-Bench / FastChat | Open-source | Pairwise comparison ("which response is better?") |
| Confident AI | Cloud (DeepEval) | Hosted judge with calibration and inter-rater agreement |

### Judge Prompt Pattern (Andrew Ng Course)

```
Determine how many of the N gold-standard talking points
are present in the provided essay.

Original Prompt: {original_prompt}
Essay to Evaluate: {essay_text}
Gold Standard Talking Points: {gold_standard_points}

Output Format:
Return a JSON object with two keys:
- "score": integer between 0 and N
- "explanation": string listing the talking points present
```

### Mitigating LLM Judge Bias

LLM judges have positional bias and verbosity bias. To reduce this:
- Use structured output (`{"score": int, "reasoning": str}`)
- Average scores from 2-3 judge runs for high-stakes evals
- Calibrate the judge against human labels on 20-30 examples before trusting it

---

## 4. Component-Level Evaluation — Isolating Pipeline Stages

Tools for testing individual agents and tools in isolation, without running the full pipeline.

| Tool | Type | Best for |
|------|------|----------|
| **pytest** | Open-source | Unit/integration tests for agent components — **used in this codebase** |
| **pytest-asyncio** | Open-source | Async agent method testing — **used in this codebase** (`asyncio_mode = "auto"`) |
| **DeepEval** | Open-source | Component-level test cases with `assert_test()` |
| unittest.mock / pytest-mock | Open-source | Mock external tools (web search, APIs) for deterministic tests |
| Hypothesis | Open-source | Property-based testing — finds edge cases automatically |

### Codebase Test Structure

```
tests/
├── orchestration/        component tests for agents and workflows
├── core/                 core logic tests
├── apis/                 API endpoint tests
│   └── gql/
│       └── test_evaluation_resolvers.py   20+ test cases for eval GraphQL API
└── data_access/
    └── cassandra/store/
        └── test_evaluation_run_store.py   CRUD tests for evaluation runs
```

### Component Eval Pattern

```python
# Test a single evaluator node in isolation
async def test_completeness_evaluator_scores_above_threshold():
    workflow = EvaluationWorkflow()
    result = await workflow.get_graph().ainvoke({
        "report": SAMPLE_REPORT,
        "outline": SAMPLE_OUTLINE,
        "user_question": "What are SAP's AI capabilities?"
    })
    scores = {e["name"]: e["score"] for e in result["evaluations"]}
    assert scores["completeness"] >= 3
```

---

## 5. Trajectory Evaluation — Did the Agent Take the Right Steps?

Tools that evaluate the sequence of tool calls and reasoning steps, not just the final output.
Two agents with the same final score can differ drastically in efficiency and safety.

| Tool | Type | Best for |
|------|------|----------|
| LangSmith | Cloud | Visual trajectory comparison, step-by-step diff |
| **Langfuse** | Open-source / Cloud | Full span tree showing every tool call in order — **used in this codebase** |
| Arize Phoenix | Open-source | Span-level analysis with tool call inspection |
| AgentEval (AutoGen) | Open-source | Task-specific trajectory scoring framework |
| Custom step precision/recall | DIY | See `3-evaluation_metrics.ipynb` cell 6 |

### What to Look for in Trajectory Data

```
Good trajectory (3 steps, all necessary):
  user_question → web_search → synthesize → write_section → END

Bad trajectory (7 steps, agent is looping):
  user_question → web_search → web_search → web_search
               → calculator → synthesize → write_section → END
```

### Trajectory Metrics

| Metric | Formula | What it reveals |
|--------|---------|-----------------|
| Step Precision | correct tools called / total tools called | Are the agent's steps all necessary? |
| Step Recall | correct tools called / expected tools | Did it miss any required steps? |
| Path Efficiency | optimal steps / actual steps taken | How wasteful was the path? |

---

## 6. Production / Online Monitoring — Continuous Scoring

Tools for running evals automatically on live traffic, not just pre-deployment datasets.

| Tool | Type | Best for |
|------|------|----------|
| **Langfuse** | Open-source / Cloud | Score ingestion per trace, dashboards, alerting — **used in this codebase** |
| LangSmith | Cloud | Online evaluators that trigger automatically per trace |
| Arize | Cloud (enterprise) | ML monitoring with concept drift detection |
| WhyLabs | Cloud | Statistical monitoring, data quality alerts |
| Grafana + custom metrics | Open-source | Infrastructure-level monitoring alongside eval scores |
| PagerDuty / Slack webhooks | Cloud | Alert routing when scores drop below threshold |

### Production Score Thresholds (Recommended)

| Score | Healthy | Alert condition |
|-------|---------|-----------------|
| `report.overall` | ≥ 3.5 / 5 | < 3.0 for 5 consecutive runs |
| `report.completeness` | ≥ 3.5 / 5 | Drops > 0.5 vs 7-day average |
| `report.overall_quality.source_quality` | ≥ 3.0 / 5 | < 2.5 — research component issue |
| Evaluation duration | < 30s | > 60s — model latency or loop issue |
| Eval job failure rate | < 5% | > 10% — infrastructure issue |

### This Codebase's Production Pipeline

```
Every report generation
  → schedule_report_evaluation_job()       async, non-blocking
  → EvaluationWorkflow (9 dimensions)
  → write_trace_scores() to Langfuse       scores attached to original trace
  → update EvaluationRun in Cassandra      for internal API querying
```

---

## 7. Human Evaluation — Ground Truth and Calibration

Tools for collecting human labels. Human judgment is the gold standard that all
automated evals calibrate against. Teams incorporating weekly human feedback
achieve significantly higher user satisfaction than those reviewing quarterly.

| Tool | Type | Best for |
|------|------|----------|
| Label Studio | Open-source | Flexible annotation UI, self-hosted |
| Argilla | Open-source | LLM-specific feedback collection, preference labeling |
| Scale AI / Surge | Cloud | High-volume human annotation at scale |
| Braintrust | Cloud | Side-by-side human review built into eval workflow |
| **Langfuse** | Open-source / Cloud | Human annotations on traces — **available in this codebase** |
| Google Sheets + survey | DIY | Quick calibration for 20-50 examples to start |

### When to Use Human Eval

- Calibrating your LLM judge: compare its scores to human scores on 30 examples
- Evaluating dimensions with no clear programmatic metric (tone, brand alignment)
- Investigating failures that automated evals did not catch
- Generating the initial gold standard dataset for a new evaluation dimension

---

## 8. A/B Testing and Experimentation — Comparing Agent Versions

Tools for running controlled experiments between agent configurations — prompt versions,
model changes, or hyperparameter tweaks.

| Tool | Type | Best for |
|------|------|----------|
| LangSmith Experiments | Cloud | Side-by-side comparison of prompt or model changes |
| Langfuse Experiments | Cloud | Dataset-based A/B eval runs |
| Braintrust | Cloud | Experiment comparison with statistical significance |
| GrowthBook | Open-source | Feature flags for routing % of live traffic to new agent version |
| Shadow mode (custom) | DIY | Run old and new agent in parallel, compare scores silently |

### Hyperparameter Sweep Pattern

Vary one parameter at a time across your eval dataset and track the effect on scores.
See `3-evaluation_metrics.ipynb` cell 9 for a working example.

```
Parameters to sweep for a research agent:
  search_engine:       google | bing | tavily
  num_results:         5 | 10 | 20
  date_range_days:     30 | 90 | 365

Track per configuration:
  source_quality_f1, synthesis_coverage, avg_latency_ms, cost_per_run_usd

Pick the config that maximizes quality within your latency and cost budget.
```

---

## 9. RAG & Grounding Evaluation

Tools specifically for the RAG Triad: context relevance, groundedness, and answer relevance.
See `6-establishing-evaluation-framework.md` for full metric definitions and code examples.

| Tool | Type | Best for |
|------|------|----------|
| RAGAS | Open-source Python | Full RAG Triad + context precision/recall in one library |
| TruLens | Open-source Python | RAG Triad with chain-of-thought reasoning explanations |
| DeepEval FaithfulnessMetric | Open-source Python | Claim-level groundedness / hallucination detection |
| Arize Phoenix | Open-source | RAG trace visualisation with per-chunk relevance scores |
| Galileo | Cloud | RAG monitoring with hallucination alerts in production |
| Openlayer | Cloud | Groundedness scoring integrated into CI/CD |

---

## 10. Toxicity & Safety Evaluation

Tools for detecting harmful, biased, or unsafe agent outputs.
See `6-establishing-evaluation-framework.md` for thresholds and red-teaming methodology.

| Tool | Type | Best for |
|------|------|----------|
| DeepEval ToxicityMetric / BiasMetric | Open-source | Drop-in toxicity and bias scoring |
| Perspective API (Google) | Cloud API | Fine-grained toxicity: hate, threat, insult, identity attack |
| Llama Guard (Meta) | Open-source model | Local safety classifier — no external API dependency |
| Garak | Open-source | Automated red-teaming and jailbreak resistance testing |
| Microsoft PyRIT | Open-source | Adversarial prompt generation for red-team workflows |
| Presidio (Microsoft) | Open-source | PII detection and anonymisation in agent outputs |

**Safety datasets:**

| Dataset | What it tests |
|---------|--------------|
| RealToxicityPrompts | 100k prompts graded for toxicity risk |
| ToxiGen | Implicit bias and hate speech |
| AdvBench | 500 adversarial behaviours for jailbreak resistance |
| WildGuard | 92k examples across 13 risk categories |

---

## 11. Latency & Cost Benchmarking

Tools for tracking operational performance alongside quality metrics.
See `6-establishing-evaluation-framework.md` for target benchmarks and cost formulas.

| Tool | Type | Best for |
|------|------|----------|
| **Langfuse** | Open-source / Cloud | Per-span latency, token counts per trace — **used in this codebase** |
| LangSmith | Cloud | Cost and latency dashboards per run and per prompt |
| OpenLLMetry | Open-source | OpenTelemetry-based LLM cost + latency instrumentation |
| Helicone | Cloud | Proxy-based cost tracking, zero code change |
| PromptLayer | Cloud | Token usage and cost logging per prompt version |
| Grafana + Prometheus | Open-source | Custom infra dashboards alongside LLM metrics |

---

## 12. Golden Dataset Management

Tools for building, versioning, and maintaining evaluation datasets.
See `6-establishing-evaluation-framework.md` for the silver → gold promotion process.

| Tool | Type | Best for |
|------|------|----------|
| Braintrust | Cloud | Dataset versioning, eval runs, side-by-side annotation |
| Argilla | Open-source | LLM-specific feedback collection, preference labelling |
| Label Studio | Open-source | Flexible self-hosted annotation for SME review |
| **Langfuse Datasets** | Open-source / Cloud | Curate production traces directly into eval datasets |
| Maxim AI | Cloud | Dataset builder with synthetic generation support |

---

## Quick Reference: Tool by Stage

```
Capture traces          → Langfuse*, LangSmith, Arize Phoenix, W&B Weave
─────────────────────────────────────────────────────────────────────────
Offline eval dataset    → DeepEval*, RAGAS, Promptfoo, Braintrust
─────────────────────────────────────────────────────────────────────────
LLM-as-judge scoring    → DeepEval GEval*, LangChain Criteria, Prometheus-2
─────────────────────────────────────────────────────────────────────────
Component unit tests    → pytest*, pytest-asyncio*, DeepEval assert_test
─────────────────────────────────────────────────────────────────────────
Trajectory analysis     → Langfuse spans*, LangSmith, Arize Phoenix
─────────────────────────────────────────────────────────────────────────
RAG / grounding         → RAGAS, TruLens, DeepEval Faithfulness
─────────────────────────────────────────────────────────────────────────
Toxicity & safety       → DeepEval Toxicity*, Perspective API, Llama Guard
─────────────────────────────────────────────────────────────────────────
Red-teaming             → Garak, PyRIT, manual adversarial testing
─────────────────────────────────────────────────────────────────────────
Production monitoring   → Langfuse online scores*, LangSmith online evals
─────────────────────────────────────────────────────────────────────────
Latency & cost          → Langfuse spans*, OpenLLMetry, Helicone
─────────────────────────────────────────────────────────────────────────
Human labeling          → Langfuse annotations*, Label Studio, Argilla
─────────────────────────────────────────────────────────────────────────
A/B experiments         → LangSmith Experiments, Langfuse Experiments
─────────────────────────────────────────────────────────────────────────
Golden dataset mgmt     → Langfuse Datasets*, Braintrust, Argilla
─────────────────────────────────────────────────────────────────────────

* = already used or available in this codebase
```

---

## Recommended Next Step for This Codebase

The cheapest path to adding a CI/CD quality gate without rebuilding infrastructure:

1. Activate **Langfuse Experiments** or **LangSmith Experiments**
2. Export the lowest-scoring 20-30 production traces as a regression dataset
3. Run that dataset against every significant prompt or model change before merging
4. Fail the PR if `report.overall` drops more than 0.3 points versus baseline

This reuses the existing `EvaluationWorkflow`, Langfuse integration, and DeepEval
metrics — no new tooling required.

---

## References

- `orchestration/evaluation/evaluation_runner.py` — production evaluation entry point
- `evaluations/metrics.py` — DeepEval GEval metric definitions
- `core/common/telemetry/langfuse_scores.py` — Langfuse score writer
- [DeepEval documentation](https://docs.confident-ai.com)
- [Langfuse documentation](https://langfuse.com/docs)
- [RAGAS documentation](https://docs.ragas.io)
