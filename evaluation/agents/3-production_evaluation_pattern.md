# Production Evaluation Pattern

How to run continuous evaluation on AI agents in production — covering architecture,
observability, monitoring, and the evaluation system used in this codebase.

---

## Architecture Overview

The production evaluation pattern has three phases that run continuously:

```
┌─────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────────┐
│  1. Pre-Deployment      │    │  2. Production (Online)  │    │  3. Continuous Improve   │
│─────────────────────────│    │──────────────────────────│    │──────────────────────────│
│  Offline eval dataset   │───►│  Observability (traces)  │───►│  Error analysis          │
│  Component unit tests   │    │  Async scoring per run   │    │  Dataset curation        │
│  Simulation scenarios   │    │  Real-time alerting      │    │  Regression suite update │
│  CI/CD quality gate     │    │  Langfuse dashboard      │    │  Hyperparameter tuning   │
└─────────────────────────┘    └──────────────────────────┘    └──────────────────────────┘
```

Simulation-based pre-deployment testing finds approximately 85% of critical issues before
production. The remaining 15% require production data to surface.

---

## Codebase Evaluation Architecture

The experience-generation-server evaluation system:

```
User request triggers ReportGenerationWorkflow
              │
              │ workflow completes
              ▼
┌─────────────────────────────────────────┐
│  schedule_report_evaluation_job()       │  ← non-blocking asyncio.Task
│  orchestration/evaluation/              │
│  evaluation_runner.py                   │
└──────────────────┬──────────────────────┘
                   │
                   │ ReportEvaluationJobInput:
                   │   execution_id, workflow_id, org_id, user_id
                   │   report_content, outline, user_question
                   │   langfuse_trace_id, langfuse_session_id
                   ▼
┌─────────────────────────────────────────┐
│  EvaluationWorkflow (LangGraph)         │
│  orchestration/workflows/               │
│  evaluation_workflow.py                 │
│                                         │
│  [completeness_evaluator]  → score 1-5  │
│         ↓                               │
│  [structure_evaluator]     → score 1-5  │
│         ↓                               │
│  [relevance_evaluator]     → score 1-5  │
│         ↓                               │
│  [overall_quality_evaluator]            │
│    → research_depth        score 1-5    │
│    → source_quality        score 1-5    │
│    → analytical_rigor      score 1-5    │
│    → practical_value       score 1-5    │
│    → balance_objectivity   score 1-5    │
│    → writing_quality       score 1-5    │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────────┐
       ▼                    ▼
┌─────────────┐    ┌────────────────────┐
│  Cassandra  │    │  Langfuse          │
│             │    │                    │
│ evaluation_ │    │  Scores attached   │
│ runs table  │    │  to original trace │
│ (status,    │    │  (report.overall,  │
│  scores,    │    │   report.complete  │
│  details)   │    │   report.structure │
│             │    │   etc.)            │
└─────────────┘    └────────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `orchestration/evaluation/evaluation_runner.py` | Entry point — schedules and executes the eval job |
| `orchestration/workflows/evaluation_workflow.py` | LangGraph pipeline with 4 sequential evaluator agents |
| `orchestration/resources/evaluation_types.py` | Pydantic input/output types for all evaluators |
| `orchestration/resources/evaluation_states.py` | LangGraph state — accumulates `EvaluationResult` list |
| `evaluations/evaluation_registry.py` | Registry of evaluation type configurations (GEval thresholds, criteria) |
| `evaluations/metrics.py` | DeepEval GEval metric definitions with 3-tier rubrics |
| `evaluations/evaluation_model.py` | DeepEvalBaseLLM wrapper for the LLM provider |
| `core/common/telemetry/langfuse_score_schema.py` | Canonical score names and Langfuse telemetry helpers |
| `data_access/entities/evaluation_run_entity_handler.py` | CRUD for evaluation run records (Cassandra) |
| `data_access/entities/evaluation_entity_handler.py` | CRUD for evaluation configurations (PostgreSQL) |

---

## Data Models

### EvaluationEntity (PostgreSQL)
Stores evaluation configuration blueprints — what to evaluate and how.

```
EvaluationEntity
├── id                UUID
├── organization_id   str         (multi-tenant)
├── evaluation_type   str         (REPORT_COMPREHENSIVE_ASSESSMENT, etc.)
├── display_name      str
├── description       str
├── kind              str         (STANDARD | CUSTOM)
├── method            str         (auto_llm)
├── criteria          dict        (GEval metric configs with thresholds)
└── configuration     dict        (executor, target_type)
```

### EvaluationRun (Cassandra)
Tracks each individual evaluation execution with status lifecycle.

```
EvaluationRun
├── evaluation_run_id    UUID
├── organization_id      str
├── user_id              str
├── evaluation_id        UUID        → links to EvaluationEntity
├── evaluation_type      str
├── chat_id              str | None
├── target_artifact_ids  list[str]
├── target_run_ids       list[str]
├── status               PENDING → RUNNING → COMPLETED | FAILED
├── overall_score        float | None
├── results_summary      dict        (canonical_name → score)
├── results_details      dict        (canonical_name → {score, reasoning})
├── duration_ms          int
└── error_message        str | None
```

### Canonical Score Names (Langfuse)

All scores are written to Langfuse under standardized names:

| Evaluator output name                    | Langfuse canonical name                       |
|------------------------------------------|-----------------------------------------------|
| `completeness`                           | `report.completeness`                         |
| `structure`                              | `report.structure`                            |
| `relevance`                              | `report.relevance`                            |
| `overall_quality_research_depth`         | `report.overall_quality.research_depth`       |
| `overall_quality_source_quality`         | `report.overall_quality.source_quality`       |
| `overall_quality_analytical_rigor`       | `report.overall_quality.analytical_rigor`     |
| `overall_quality_practical_value`        | `report.overall_quality.practical_value`      |
| `overall_quality_balance_and_objectivity`| `report.overall_quality.balance_and_objectivity` |
| `overall_quality_writing_quality`        | `report.overall_quality.writing_quality`      |
| _(computed average)_                     | `report.overall`                              |

---

## Evaluation Types Registry

Three built-in evaluation type configurations in `evaluations/evaluation_registry.py`:

### REPORT_COMPREHENSIVE_ASSESSMENT
The default type. Runs 4 GEval metrics via DeepEval:

| Metric | Threshold | What it checks |
|--------|-----------|----------------|
| Financial Performance | 0.5 | Revenue trends, margins, growth rates |
| KPI Coverage | 0.5 | Key performance indicators in the report |
| Factual Correctness | 0.7 | Factual accuracy against context |
| Competitive Landscape | 0.5 | Market positioning, competitor analysis |

Each metric uses a 3-tier rubric:
- Score 0-3: Missing or minimal coverage
- Score 4-7: Adequate coverage
- Score 8-10: Comprehensive and insightful

### GENERAL_TEXT_COHERENCE_CHECK
- 1 GEval metric: Coherence (threshold 0.7)
- Checks logical flow, clarity, and readability

### FINANCIAL_PERFORMANCE_METRIC
- 1 GEval metric: Financial Metric Accuracy (threshold 0.8)
- Stricter threshold — used for finance-specific reports

---

## Production Monitoring Signals

### Metrics to Track Per Run

| Metric | Healthy range | Alert if |
|--------|---------------|----------|
| `report.overall` | ≥ 3.5/5 | < 3.0 for 5 consecutive runs |
| `report.completeness` | ≥ 3.5/5 | Drops > 0.5 points vs 7-day avg |
| `report.overall_quality.source_quality` | ≥ 3.0/5 | < 2.5 — research component issue |
| Evaluation duration | < 30s | > 60s — model latency or loop issue |
| Eval job failure rate | < 5% | > 10% — infrastructure issue |

### Evaluator Run Status Lifecycle

```
              create_evaluation_run()
                      │
                      ▼
                 [ PENDING ]
                      │
     update_status(RUNNING)
                      │
                      ▼
                 [ RUNNING ]
                      │
          ┌───────────┴────────────┐
   success│                  error │
          ▼                        ▼
     [ COMPLETED ]           [ FAILED ]
      overall_score           error_message
      results_summary         duration_ms
      results_details
```

---

## Observability: Traces, Spans, and Langfuse

Every evaluation job is linked to the **original report generation trace** in Langfuse.
This allows you to see the full picture: generation + evaluation in one trace.

```
Langfuse Trace (report generation)
├── span: research_phase
│   ├── span: web_search
│   └── span: synthesize_research
├── span: writing_phase
│   └── span: write_section (x N)
└── span: report_evaluation          ← attached by evaluation_runner.py
    ├── span: completeness_evaluator
    ├── span: structure_evaluator
    ├── span: relevance_evaluator
    └── span: overall_quality_evaluator

Scores written to trace:
  report.completeness         = 4.0
  report.structure            = 5.0
  report.relevance            = 4.0
  report.overall_quality.*    = 3.0 - 5.0
  report.overall              = 4.1  (average)
```

Tags applied per evaluation trace:
- `evaluation:report`
- `score_category:report_quality`
- `workflow:<workflow_id>`
- `execution_id:<execution_id>`

---

## Continuous Improvement Loop

```
Week 1-2: Establish baseline
  ├── Enable evaluation for all production runs
  ├── Let Langfuse collect 50-100 scored traces
  └── Compute average scores per dimension

Week 3: First error analysis
  ├── Pull 20 lowest-scoring traces from Langfuse
  ├── Categorize: which dimension is lowest?
  ├── Drill into that component's traces
  └── Identify the root cause pattern

Week 4: Fix and measure
  ├── Improve the failing component (prompt, model, tool config)
  ├── Re-run the eval dataset
  └── Compare before/after scores

Ongoing: Regression guard
  ├── Curate failures into regression dataset
  ├── Run dataset on every significant change
  └── Alert if any metric drops > 0.3 points
```

---

## Langfuse → Golden Dataset Pipeline

The continuous improvement loop described above requires pulling low-scoring traces
from Langfuse and promoting them into the regression eval dataset. This section makes
that step concrete.

### Why This Step Is Critical

Without it, the eval dataset stays frozen at the examples you wrote on day one.
Production failures — the most valuable test cases — never enter the dataset,
so the regression suite does not guard against real failure modes.

```
Production trace (low score)
        │
        │  pull from Langfuse
        ▼
Silver record — agent output + metadata
        │
        │  SME review — is this a real failure?
        │  annotate correct expected output
        ▼
Gold record — added to regression dataset
        │
        │  dataset runs on every PR
        ▼
CI/CD gate — merge blocked if score drops > 0.3 vs baseline
```

### Step 1 — Query Low-Scoring Traces from Langfuse

```python
# Pull the N lowest-scoring traces from a date window.
# Requires: pip install langfuse

from langfuse import Langfuse
from datetime import datetime, timedelta

langfuse = Langfuse()   # reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY from env

def fetch_low_scoring_traces(
    score_name: str = "report.overall",
    max_score:  float = 3.0,
    limit:      int = 20,
    days_back:  int = 7,
) -> list[dict]:
    """
    Fetch traces where report.overall < max_score over the last N days.
    Returns a list of dicts ready for SME review.
    """
    cutoff = datetime.utcnow() - timedelta(days=days_back)

    # Langfuse Python SDK: fetch traces with score filter
    traces = langfuse.fetch_traces(
        tags=["evaluation:report"],          # only report evaluation traces
        from_timestamp=cutoff,
        limit=limit * 5,                     # over-fetch; we filter by score below
    )

    low_scoring = []
    for trace in traces.data:
        # Scores are attached to the trace object
        scores = {s.name: s.value for s in (trace.scores or [])}
        overall = scores.get(score_name)
        if overall is not None and overall < max_score:
            low_scoring.append({
                "trace_id":      trace.id,
                "session_id":    trace.session_id,
                "timestamp":     trace.timestamp.isoformat(),
                "user_question": (trace.input or {}).get("user_question", ""),
                "report":        (trace.output or {}).get("report_content", ""),
                "scores":        scores,
                "overall_score": overall,
            })
        if len(low_scoring) >= limit:
            break

    low_scoring.sort(key=lambda x: x["overall_score"])
    print(f"Fetched {len(low_scoring)} traces with {score_name} < {max_score}")
    return low_scoring
```

### Step 2 — Review and Annotate

```python
import json
from pathlib import Path

def export_for_sme_review(traces: list[dict], output_path: str = "silver_candidates.json") -> None:
    """
    Export low-scoring traces as a reviewable JSON file.
    SME opens this file, reads each entry, and fills in `expected_output`
    and `failure_category` fields.
    """
    candidates = []
    for t in traces:
        candidates.append({
            "trace_id":          t["trace_id"],
            "timestamp":         t["timestamp"],
            "user_question":     t["user_question"],
            "agent_output":      t["report"][:500] + "..." if len(t["report"]) > 500 else t["report"],
            "scores":            t["scores"],
            # SME fills these in:
            "expected_output":   "",      # what a correct answer looks like
            "failure_category":  "",      # e.g. hallucination | missing_section | wrong_source
            "promote_to_gold":   False,   # SME sets True to include in dataset
            "difficulty":        "medium" # easy | medium | hard | adversarial
        })

    Path(output_path).write_text(json.dumps(candidates, indent=2))
    print(f"Exported {len(candidates)} candidates to {output_path}")
    print("Next: open the file, annotate, set promote_to_gold=true, then run promote_to_golden_dataset()")
```

### Step 3 — Promote Reviewed Records to the Golden Dataset

```python
def promote_to_golden_dataset(
    reviewed_path: str = "silver_candidates.json",
    dataset_path:  str = "golden_dataset.json",
) -> int:
    """
    Load SME-reviewed candidates and append approved records to the golden dataset.
    Returns the count of records promoted.
    """
    candidates = json.loads(Path(reviewed_path).read_text())
    approved   = [c for c in candidates if c.get("promote_to_gold") is True]

    if not approved:
        print("No records marked promote_to_gold=True. Nothing promoted.")
        return 0

    # Load or initialise the golden dataset
    if Path(dataset_path).exists():
        golden = json.loads(Path(dataset_path).read_text())
    else:
        golden = []

    existing_trace_ids = {r["trace_id"] for r in golden}

    new_records = []
    for record in approved:
        if record["trace_id"] in existing_trace_ids:
            continue   # already in the dataset

        new_records.append({
            "trace_id":         record["trace_id"],
            "query":            record["user_question"],
            "expected_output":  record["expected_output"],
            "failure_category": record["failure_category"],
            "difficulty":       record["difficulty"],
            "source":           "production_failure",
            "promoted_at":      datetime.utcnow().isoformat(),
            "original_scores":  record["scores"],
        })

    golden.extend(new_records)
    Path(dataset_path).write_text(json.dumps(golden, indent=2))
    print(f"Promoted {len(new_records)} new records → {dataset_path}")
    print(f"Golden dataset now has {len(golden)} records total.")
    return len(new_records)
```

### Step 4 — Run the Dataset as a Regression Check

```python
def run_regression_check(
    dataset_path: str = "golden_dataset.json",
    baseline_score: float = 3.5,
    alert_threshold: float = 0.3,
) -> dict:
    """
    Run the EvaluationWorkflow over every record in the golden dataset
    and compare against the baseline score.

    In CI/CD: fail the pipeline if any metric drops > alert_threshold vs baseline.
    """
    golden = json.loads(Path(dataset_path).read_text())
    if not golden:
        print("Golden dataset is empty — nothing to check.")
        return {}

    results = []
    for record in golden:
        # In production: call EvaluationWorkflow.run(record["query"])
        # Here we simulate with a random score near baseline
        import random
        simulated_score = round(random.gauss(baseline_score, 0.4), 2)
        results.append({
            "trace_id":     record["trace_id"],
            "query":        record["query"][:60],
            "score":        simulated_score,
            "delta":        round(simulated_score - baseline_score, 2),
            "passed":       simulated_score >= baseline_score - alert_threshold,
        })

    passed  = sum(1 for r in results if r["passed"])
    failed  = len(results) - passed
    avg_score = sum(r["score"] for r in results) / len(results)

    print(f"\nRegression Results  ({len(results)} records)")
    print(f"  Passed:    {passed}/{len(results)}")
    print(f"  Failed:    {failed}/{len(results)}")
    print(f"  Avg score: {avg_score:.2f}  (baseline: {baseline_score})")
    print()
    for r in sorted(results, key=lambda x: x["score"])[:5]:
        flag = "FAIL" if not r["passed"] else "    "
        print(f"  {flag}  score={r['score']}  Δ={r['delta']:+.2f}  {r['query']}")

    ci_decision = "BLOCK MERGE" if failed > 0 else "PASS — safe to merge"
    print(f"\nCI/CD decision: {ci_decision}")
    return {"passed": passed, "failed": failed, "avg_score": avg_score, "decision": ci_decision}


# ── Weekly workflow ────────────────────────────────────────────────────────────
# Run this script once a week. The first three steps are SME-assisted (manual review);
# step 4 runs automatically on every PR via CI/CD.

# Week N workflow:
#   traces    = fetch_low_scoring_traces(max_score=3.0, limit=20)
#   export_for_sme_review(traces)
#   # ... SME annotates silver_candidates.json ...
#   promoted  = promote_to_golden_dataset()
#   result    = run_regression_check()
```

### Langfuse Dataset API (Alternative)

Langfuse has a native Datasets feature that can replace the file-based approach above.
It stores the dataset server-side and lets you run eval jobs directly from the UI.

```python
# Create or update a Langfuse dataset from low-scoring traces
def promote_to_langfuse_dataset(
    traces: list[dict],
    dataset_name: str = "report-regression-set",
) -> None:
    """
    Add approved traces directly to a Langfuse dataset.
    Langfuse stores these server-side — no local JSON file needed.
    """
    # Create the dataset if it doesn't exist
    langfuse.create_dataset(name=dataset_name)

    for trace in traces:
        if not trace.get("promote_to_gold"):
            continue
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input={"user_question": trace["user_question"]},
            expected_output=trace.get("expected_output", ""),
            metadata={
                "trace_id":         trace["trace_id"],
                "failure_category": trace.get("failure_category", ""),
                "difficulty":       trace.get("difficulty", "medium"),
                "source":           "production_failure",
            }
        )
    print(f"Items promoted to Langfuse dataset '{dataset_name}'")
    print("Run experiments against this dataset in the Langfuse UI or via SDK.")
```

---



1. **Add the enum value** to `EvaluationType` in `evaluations/evaluation_registry.py`
2. **Add the configuration entry** to `EVALUATION_DEFINITIONS_REGISTRY` with GEval metrics, thresholds, and criteria
3. **Add Pydantic types** if the evaluator needs new input/output shapes in `orchestration/resources/evaluation_types.py`
4. **Add an evaluator node** to `EvaluationWorkflow` in `orchestration/workflows/evaluation_workflow.py`
5. **Register the canonical score name** mapping in `core/common/telemetry/langfuse_score_schema.py`
6. **Create the evaluation entity** in the database for each relevant organization

---

## Data Usage Permission

The evaluation runner is gated behind `@require_data_usage_permission(DataUsagePurpose.CONTINUOUS_IMPROVEMENT)`.
This ensures evaluations (which read report content) only run when the organization has consented
to using their data for system improvement. If an org has not granted this permission, the evaluation
job silently skips — no error, no data access.

This is a GDPR/data governance control. Do not remove or bypass this decorator.

---

## References

- `orchestration/evaluation/evaluation_runner.py` — production entry point
- `orchestration/workflows/evaluation_workflow.py` — the 4-agent eval pipeline
- `evaluations/evaluation_registry.py` — evaluation type definitions
- `core/common/telemetry/langfuse_score_schema.py` — canonical naming
- [LangChain: Agent Observability Powers Agent Evaluation](https://www.langchain.com/conceptual-guides/agent-observability-powers-agent-evaluation)
- [Maxim AI: Testing and Evaluating AI Agents in Production](https://www.getmaxim.ai/articles/a-comprehensive-guide-to-testing-and-evaluating-ai-agents-in-production)

---

## What to Do Next

You now understand how the codebase runs evaluation in production and how to build and
maintain a golden dataset.

**Run the advanced notebook sections:**
- `2-evaluation_metrics.ipynb` §10 — RAG Triad (faithfulness, answer relevancy, context relevance)
- `2-evaluation_metrics.ipynb` §11 — pass^k consistency testing
- `2-evaluation_metrics.ipynb` §12 — Multi-turn evaluation
- `2-evaluation_metrics.ipynb` §13 — CI/CD quality gate (uses the golden_dataset.json you just built)

**Read next:** `4-establishing-evaluation-framework.md` — adds RAG evaluation depth,
safety gates, A/B experiments, latency/cost benchmarks, and agent capability metrics.

**For tool selection:** `5-evaluation_tools.md` is a reference catalogue — consult it
when you need to pick or add a tool for a specific stage, not as a reading-order step.
