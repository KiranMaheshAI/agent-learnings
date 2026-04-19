# Agent Evaluation Guide

A practical guide to evaluating AI agents — from first principles through production monitoring.
Based on Andrew Ng's Agentic AI course and industry best practices.

---

## Why Agent Evaluation Is Different

Traditional ML evaluation asks: **"Is this output correct?"**

Agent evaluation asks: **"Did the agent take the right path, use the right tools, and reach the right outcome — efficiently and safely?"**

Two agents can produce the same final answer while one used 3 steps and one used 17. Trajectory matters.

```
Traditional LLM         Agent System
─────────────────       ──────────────────────────────────
Input → Output          Input → [LLM → Tool → LLM → Tool → LLM] → Output
  ↓                                ↑
Evaluate once               Evaluate EACH step + overall trajectory
```

---

## The Two Axes of Evaluation

Every eval sits on two axes. Pick the cell that matches what you are measuring.

|                           | Evaluate with Code (Objective)                        | LLM-as-Judge (Subjective)                             |
|---------------------------|-------------------------------------------------------|-------------------------------------------------------|
| **Per-example ground truth**  | `if extracted_date == actual_date: num_correct += 1`  | Count gold-standard talking points mentioned in output |
| **No per-example ground truth** | `if len(text) <= 10: num_correct += 1`              | Grade with rubric (clear axes labels, depth, etc.)     |

Most production agents need all four cells across different metrics.

---

## The Four Autonomy Levels

Before choosing metrics, identify what kind of agent you are evaluating:

| Level | Agent Type       | What to Evaluate                                        |
|-------|------------------|---------------------------------------------------------|
| 1     | Generator        | Output quality, factual correctness                     |
| 2     | Tool-calling     | Router accuracy, parameter extraction                   |
| 3     | Planning         | Trajectory analysis, step sequencing                    |
| 4     | Autonomous       | Long-horizon goal satisfaction, safety constraint checks |

A multi-step research pipeline (research → synthesize → write) sits at Level 3-4. That demands trajectory evaluation, not just output scoring.

---

## The Three Observability Primitives

You need three layers of captured data to evaluate anything:

```
┌─────────────────────────────────────────────┐
│  Thread (full user session / conversation)  │
│  ┌───────────────────────────────────────┐  │
│  │  Trace (one agent execution)          │  │
│  │  ┌─────────────────────────────────┐  │  │
│  │  │  Run / Span (single LLM call)   │  │  │
│  │  │  - Full prompt context          │  │  │
│  │  │  - Tool arguments & results     │  │  │
│  │  │  - Latency, token count         │  │  │
│  │  └─────────────────────────────────┘  │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

**Observability and evaluation are inseparable.** Production traces serve triple duty:
1. Manual debugging of individual failures
2. Source of ground-truth data for offline eval datasets
3. Continuous online evaluation on every request

---

## End-to-End Evaluation: The Andrew Ng Method

### Step 1 — Start Quick and Dirty

Do not wait for a perfect harness. Start with a script that checks 10 outputs manually.

> "Quick and dirty is ok to start! As you find places where your evals fail to capture human judgement, use that as an opportunity to improve the metric."

### Step 2 — Build a Gold Standard Dataset

Manually annotate 10-20 representative examples first. Scale from there.

```python
gold_standard = [
    {
        "prompt": "Recent developments in SAP CX AI",
        "gold_websites": ["sap.com", "sapinsider.org"],
        "gold_talking_points": ["AI Core", "CX portfolio", "GenAI Hub"],
        "expected_date_range": "2024-2026"
    },
    # 10-20 examples to start, 50-100 for reliable signal
]
```

**Dataset size guidelines:**
- 10-20 examples: enough to get started and spot obvious failures
- 50-100 examples: reliable signal for trend tracking
- 200+ examples: statistical significance across edge cases

### Step 3 — Measure Each Component Separately

End-to-end evals are expensive and noisy. Component evals isolate the failure point.

```
End-to-End:  Prompt → [Full Pipeline] → Report
                                            ↑
                             Score 3/5 — but WHICH step failed?

Component:   Prompt → [Web Search]   → 4/5 (fine)
             Prompt → [Synthesis]    → 2/5 (PROBLEM)
             Prompt → [Writing]      → 4/5 (fine)
```

### Step 4 — Track Error Rates Per Component

This is the "counting up the errors" method. Categorize failures by component and find the biggest bucket.

| Prompt                       | Search terms | Search results                   | Picking 5 best |
|------------------------------|--------------|----------------------------------|----------------|
| Black holes                  | OK           | Too many blog posts              | OK             |
| Renting vs buying in Seattle | OK           | OK                               | Missed key blog|
| Robotics for harvesting fruit| Terms generic| Website for elementary students  | OK             |
| Batteries for electric cars  | OK           | Only US-based companies          | Missed magazine|
| **Error rate**               | **5%**       | **45%**                          | **10%**        |

Fix the 45% bucket first. That is the entire discipline of error analysis.

---

## Error Analysis — The Most Important Habit

> "One of the good indicators of a team is how they do disciplined error analysis to tell you where to focus."

### The Error Analysis Loop

```
1. Sample 20-50 failed or low-scoring traces
         ↓
2. Categorize each failure by component
         ↓
3. Count failures per category
         ↓
4. Attack the largest bucket first
         ↓
5. Re-measure — did the numbers move?
         ↓
   Repeat
```

### Common Error Categories for Research Agents

| Category            | Description                                              |
|---------------------|----------------------------------------------------------|
| `tool_selection`    | Agent chose the wrong tool for the task                  |
| `search_quality`    | Web search returned irrelevant or low-quality results    |
| `source_authority`  | Sources cited are unreliable or not authoritative        |
| `synthesis_gaps`    | Key information was present in sources but not surfaced  |
| `hallucination`     | Agent stated facts not present in source material        |
| `format_failure`    | Output did not match the required structure              |
| `loop_excessive`    | Agent took far more steps than necessary                 |

---

## Improving Performance After Error Analysis

Use this decision tree after identifying the failing component:

```
Error in LLM component?
├── YES
│   ├── Missing instructions       → Improve prompt (add explicit rules)
│   ├── Needs examples             → Add few-shot examples to prompt
│   ├── Model capacity issue       → Try a stronger model
│   ├── Task too complex           → Split into smaller sub-agents
│   └── Consistent data pattern    → Fine-tune on your data
│
└── NO — Error in non-LLM component?
    ├── Web search quality         → Tune: date range, result count, engine
    ├── RAG retrieval              → Tune: chunk size, similarity threshold, top-k
    └── Persistent low quality     → Replace the component entirely
```

---

## Evaluation Timing

| When         | What it is                                         | Use for                              |
|--------------|----------------------------------------------------|--------------------------------------|
| **Offline**  | Pre-deployment against curated dataset             | Regression gating before shipping    |
| **Online**   | Continuous scoring of every production trace       | Detecting quality drift over time    |
| **Ad-hoc**   | AI-assisted analysis of production data samples    | Deep-dive investigation of incidents |

---

## Evaluation Maturity Levels

Work through these progressively. Do not try to jump to Level 5 immediately.

| Level | What you have                          | What you can answer                       |
|-------|----------------------------------------|-------------------------------------------|
| 0     | Nothing / vibes                        | "Feels ok?"                               |
| 1     | Manual sampling (10 traces/week)       | "What failed this week?"                  |
| 2     | Automated scores on every run          | "Is quality trending up or down?"         |
| 3     | Per-component metrics                  | "Which component is the bottleneck?"      |
| 4     | Regression dataset from past failures  | "Does my change break anything?"          |
| 5     | Evals gating every CI/CD deployment    | "Ship only when metrics pass"             |

---

## Cost Awareness

Every evaluation has a cost. Be deliberate.

| Cost type                 | Driver                  | Optimization                                    |
|---------------------------|-------------------------|-------------------------------------------------|
| LLM evaluation steps      | Tokens consumed         | Cache prompts; use smaller models for bulk evals|
| API-calling tools         | Per-call pricing        | Batch requests; raise quota limits              |
| Compute                   | Server capacity/time    | Run evals async and off-peak                    |

**Practical rule:** Use a fast/cheap model (e.g., Haiku) for high-volume component evals. Reserve the strongest model (Opus) for final quality gate evals where accuracy matters most.

---

## The Evaluation Flywheel

```
          ┌──────────────────┐
     ┌───►│  Build / Improve │◄────────────────────┐
     │    └────────┬─────────┘                     │
     │             │ Deploy                        │
     │             ▼                               │
     │    ┌──────────────────┐         Prioritized │
     │    │    Production    │         next steps  │
     │    │  (Observability) │                     │
     │    └────────┬─────────┘                     │
     │             │ Traces                        │
     │             ▼                               │
     │    ┌──────────────────┐                     │
     │    │  Error Analysis  ├─────────────────────┘
     │    │  (Find root      │
     └────┤   cause)         │
Improved  └────────┬─────────┘
metrics            │ Curate failures
                   ▼
          ┌──────────────────┐
          │  Eval Dataset    │
          │  (Gold standard) │
          └──────────────────┘
```

The discipline is: **measure → find the biggest error bucket → fix it → measure again**. Teams that do this rigorously ship noticeably better agents than teams that iterate on instinct.

---

## References

- Andrew Ng — Agentic AI Course (evaluation and error analysis modules)
- [LangChain: Agent Observability Powers Agent Evaluation](https://www.langchain.com/conceptual-guides/agent-observability-powers-agent-evaluation)
- [Maxim AI: How to Evaluate AI Agents and Agentic Workflows](https://www.getmaxim.ai/articles/how-to-evaluate-ai-agents-and-agentic-workflows-a-comprehensive-guide/)
- [Maxim AI: Testing and Evaluating AI Agents in Production](https://www.getmaxim.ai/articles/a-comprehensive-guide-to-testing-and-evaluating-ai-agents-in-production)

---

## What to Do Next

You now understand the *why* and *what* of agent evaluation. The next step is running
the code patterns so the concepts become concrete.

Open `2-evaluation_metrics.ipynb` and run these sections in order:

| Notebook section | Concept from this guide |
|-----------------|------------------------|
| §2 Date extraction accuracy | Objective metric with ground truth (Two Axes, top-left cell) |
| §3 Source quality F1 | Objective metric with ground truth (research agents) |
| §4 Talking points coverage | LLM-as-judge with ground truth (Two Axes, top-right cell) |
| §5 Rubric-based scoring | LLM-as-judge without ground truth (codebase EvaluationWorkflow) |
| §6 Trajectory metric | Level 2-3 agent evaluation — tool call precision/recall |
| §7 Component error rates | The "counting up errors" method (Error Analysis section above) |
| §8 Score aggregation | How evaluation_runner.py computes report.overall |
| §9 Hyperparameter sweep | Improving performance after error analysis |

After §9, continue to §10-13 once you have read `3-production_evaluation_pattern.md`.
