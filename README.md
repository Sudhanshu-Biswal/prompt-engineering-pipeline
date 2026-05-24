# Automated Prompt Engineering Pipeline

> **APG + APO** — Auto Prompt Generation and Optimization using a multi-agent LangGraph system with a 3-judge evaluation framework and statistical hypothesis testing.

Built as a production-grade public implementation of the prompt engineering automation system developed at [BOLD](https://www.bold.com/) — reducing manual prompt engineering cycles from **3 days to under 1 hour**.

---

## The Problem

Prompt engineering for new LLM tasks was a slow, manual, 3-day cycle:

```
Write prompt → test manually → review failures → rewrite → repeat
```

There was no systematic evaluation, no statistical validation, and no way
to prove the new prompt was actually better than the baseline.

---

## The Solution

A fully automated agentic pipeline:

```
Task description + output spec
          │
          ▼
   APG — Auto Prompt Generator
   (generates production-grade prompt from scratch)
          │
          ▼
   Test Case Runner
   (generates inputs + runs baseline and optimised prompt)
          │
          ▼
   3-Judge Evaluation
   ┌──────────────────────────────────────┐
   │  J1 — Structural Judge (gpt-4o-mini) │
   │  J2 — Qualitative Judge (gpt-4o-mini)│
   │  Boss — Synthesiser (gpt-4o)         │
   └──────────────────────────────────────┘
          │
     Pass / Fail
          │
    ┌─────┴──────┐
  PASS          FAIL
    │              │
Human          APO — Auto Prompt Optimizer
Approval       (analyses failures → generates v2)
    │              │
    │         loops back (max 3 iterations)
    │
Hypothesis Test
(Wilcoxon signed-rank or permutation test)
    │
Migration Decision: APPROVED / REJECTED / NEEDS_REVIEW
```

**Result: 3 days → ~1 hour. Fully automated.**

---

## Benchmark — APG vs DSPy

Evaluated on the `improve_work_history` task (resume bullet paraphrasing).
Baseline: GPT-4o-mini with a plain unoptimised prompt.

| Method | Boss Score | vs Baseline | Notes |
|---|---|---|---|
| Same prompt (no optimisation) | 3.1/5 | — | Baseline |
| DSPy MIPRO light | 3.0/5 | worse | Standard optimizer |
| DSPy MIPRO medium | 2.9/5 | worse | Standard optimizer |
| DSPy MIPRO heavy | 3.0/5 | worse | Standard optimizer |
| DSPy BootstrapFewShot | 3.1/5 | no difference | Standard optimizer |
| **APG + APO (this system)** | **4.3/5** | **better** | Custom pipeline |

> DSPy optimises prompt *selection*. APG/APO optimises prompt *construction* —
> a fundamentally different approach that outperforms DSPy on
> structured writing-help tasks.

---

## Score Progression (example run)

```
v1  3.21/5  ████████████         ✗ failed
v2  3.84/5  ███████████████      ✗ failed  
v3  4.31/5  █████████████████    ✓ PASSED

Statistical test : Wilcoxon signed-rank
p-value          : 0.0031
Mean difference  : +0.847
Verdict          : better
Migration        : APPROVED
```

---

## Architecture

### Agent graph (LangGraph)

```
START → APG → Baseline → TestRunner → Evaluator → Router
                              ↑                       │
                              │            ┌──────────┼──────────┐
                              │           APO    HumanApproval  Hypothesis
                              │            │          │              │
                              └────────────┘     APO/Hypo          END
```

### Node responsibilities

| Node | Model | Job |
|---|---|---|
| APG | gpt-4o-mini | Generate prompt v1 from task description + output spec |
| Baseline | gpt-4o-mini | Generate source outputs using plain prompt (output_a) |
| TestRunner | gpt-4o-mini | Run current prompt on all test cases (output_b) |
| J1 | gpt-4o-mini | Structural evaluation — format, length, constraints |
| J2 | gpt-4o-mini | Qualitative evaluation — tone, clarity, naturalness |
| Boss | **gpt-4o** | Synthesise J1+J2, produce final float score |
| Router | — | Conditional edge: pass→human, fail→APO, max→hypothesis |
| APO | **gpt-4o** | Analyse failures, generate improved prompt |
| HumanApproval | — | Checkpoint: approve / reject / provide feedback |
| Hypothesis | — | Wilcoxon or permutation test, migration decision |

**Model routing rationale:** gpt-4o-mini for generation and scoring (fast, cheap).
gpt-4o only for APO and Boss — the two nodes requiring deep reasoning.
This reduces cost by ~70% vs running gpt-4o everywhere.

### 3-Judge evaluation design

**J1 — Structural Judge**
Generates rubrics from the task description automatically.
Checks format compliance, length constraints, required fields, JSON validity.
Scores 1–5 per rubric, integers only.

**J2 — Qualitative Judge**
Generates qualitative rubrics that Python tests cannot detect.
Checks tone, linguistic quality, contextual fidelity, human naturalness.
Scores 1–5 per rubric, integers only.

**Boss**
Synthesises J1 and J2 with expert weighting per task.
Produces a float final score 1.0–5.0 with confidence rating.
This is the pass/fail metric. Threshold: 4.0/5 (configurable).

All three judges run N times per test case (default: 3).
Scores are averaged across runs for statistical stability.

### Hypothesis testing

After the final iteration, the system automatically selects
the appropriate statistical test:

- **Wilcoxon signed-rank test** — when score differences are symmetric (|skewness| < 1.0)
- **Permutation test** — when differences are not symmetric

Both tests compare the optimised prompt's Boss scores against
the baseline across all test cases × all runs.

Migration threshold: p < 0.05.

---

## Project structure

```
prompt-engineering-pipeline/
├── src/
│   ├── agents/
│   │   ├── apg.py              # Auto Prompt Generator
│   │   ├── apo.py              # Auto Prompt Optimizer (gpt-4o)
│   │   └── test_runner.py      # Baseline + target output generation
│   ├── evaluation/
│   │   ├── judge1.py           # Structural judge (J1)
│   │   ├── judge2.py           # Qualitative judge (J2)
│   │   ├── boss.py             # Boss synthesiser + failure summary
│   │   └── hypothesis.py       # Wilcoxon / permutation test
│   ├── utils/
│   │   ├── state.py            # PipelineState (Pydantic) + config loader
│   │   └── llm.py              # OpenAI wrapper, token tracking, cost
│   ├── pipeline.py             # LangGraph graph assembly
│   └── main.py                 # CLI entry point
├── configs/
│   ├── improve_work_history.json   # Resume bullet paraphrasing
│   └── grammar_correction.json    # Grammar and punctuation correction
├── results/                    # Pipeline run outputs (JSON)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/Sudhanshu-Biswal/prompt-engineering-pipeline
cd prompt-engineering-pipeline
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Add your OpenAI API key to .env
```

`.env`:
```
OPENAI_API_KEY=your_key_here
```

### 3. Run with an example config

```bash
# Resume bullet improvement task
python -m src.main --config configs/improve_work_history.json

# Non-interactive mode (auto-approves at human checkpoint)
python -m src.main --config configs/improve_work_history.json --non-interactive

# Grammar correction task
python -m src.main --config configs/grammar_correction.json --non-interactive
```

### 4. Bring your own task

Create `configs/my_task.json`:

```json
{
  "task_name": "my_task",
  "task_description": "What the model should do",
  "output_spec": "Exact output format required",
  "output_is_json": true,
  "num_runs": 3,
  "pass_threshold": 4.0,
  "max_iterations": 3
}
```

```bash
python -m src.main --config configs/my_task.json
```

---

## Config reference

| Field | Type | Default | Description |
|---|---|---|---|
| `task_name` | string | required | Identifier for this task |
| `task_description` | string | required | What the model must do |
| `output_spec` | string | required | Exact output format / schema |
| `output_is_json` | bool | true | Enables JSON validity pre-check in J1 |
| `num_runs` | int | 3 | Judge runs per test case (more = more stable) |
| `pass_threshold` | float | 4.0 | Boss score required to pass (1–5) |
| `max_iterations` | int | 3 | Max APO iterations before stopping |
| `input_texts` | list | [] | Optional: provide your own test inputs |
| `source_outputs` | list | [] | Optional: provide your own baseline outputs |
| `human_feedback` | string | null | Optional: seed the first APG with feedback |

---

## Output

Each run saves a JSON result file to `results/`:

```json
{
  "task_name": "improve_work_history",
  "migration_decision": "APPROVED",
  "migration_reason": "Optimised prompt is statistically BETTER...",
  "final_score": 4.31,
  "iterations_used": 3,
  "hypothesis": {
    "test_used": "wilcoxon",
    "p_value": 0.0031,
    "mean_difference": 0.847,
    "model_state": "better"
  },
  "score_progression": [
    {"iteration": 1, "avg_boss_score": 3.21, "passed": false},
    {"iteration": 2, "avg_boss_score": 3.84, "passed": false},
    {"iteration": 3, "avg_boss_score": 4.31, "passed": true}
  ],
  "final_prompt": "...",
  "total_cost_usd": 0.84
}
```

---

## Cost estimate

Approximate cost per pipeline run (3 iterations, 10 test cases, 3 runs):

| Component | Model | Approx cost |
|---|---|---|
| APG + test case gen | gpt-4o-mini | ~$0.02 |
| 3× TestRunner (30 calls) | gpt-4o-mini | ~$0.05 |
| 3× J1 (90 calls) | gpt-4o-mini | ~$0.08 |
| 3× J2 (90 calls) | gpt-4o-mini | ~$0.10 |
| 3× Boss (90 calls) | gpt-4o | ~$0.45 |
| 3× APO | gpt-4o | ~$0.12 |
| Hypothesis + failure summary | gpt-4o | ~$0.04 |
| **Total** | | **~$0.86** |

Cost is logged per run in the results JSON and printed in the terminal summary.

---

## Why not DSPy?

DSPy's optimisers (MIPRO, BootstrapFewShot) are powerful for
*selecting* the best few-shot examples for a fixed prompt template.

This system solves a different problem: *constructing* a production-grade
prompt from scratch and iteratively improving it based on structured failure analysis.

On structured writing-help tasks (resume bullets, grammar correction),
the APG/APO approach outperforms DSPy because:

1. APG generates the full prompt structure, not just examples
2. The 3-judge framework gives richer failure signal than single-metric scoring
3. APO uses the full failure analysis as context — not just score deltas
4. The system handles both JSON and plain-text output tasks natively

DSPy remains the better choice for classification, retrieval, and
tasks where few-shot example selection is the primary lever.

---

## Production context

This project is a clean, public implementation of the prompt engineering
automation system built at [BOLD](https://www.bold.com/) for writing-help
workloads (resume bullets, grammar correction, work history improvement,
cover letters).

In production, the system uses BOLD's internal LLM gateway and proprietary
writing data. This public version uses OpenAI directly with generated test
cases, preserving the full architecture and evaluation framework.

**Impact:** Reduced prompt engineering cycle from 3 days (manual) to
under 1 hour (automated) across multiple writing-help use cases.

---

## Requirements

```
Python 3.10+
openai>=1.30.0
langgraph>=0.1.0
pydantic>=2.0.0
scipy>=1.13.0
numpy>=1.26.0
```

---

## Author

**Sudhanshu Sekhar Biswal**
Senior AI Engineer · Patent Holder · CII National AI Award Winner

[GitHub](https://github.com/Sudhanshu-Biswal) · [LinkedIn](https://linkedin.com/in/sudhanshubiswal)

---

*Part of a 3-project portfolio demonstrating production GenAI engineering:*
- *[Career Intelligence RAG](https://github.com/Sudhanshu-Biswal/career-intelligence-rag) — HyDE + hybrid retrieval + RAGAS evaluation*
- *[Qwen3.5 Fine-tuning](https://github.com/Sudhanshu-Biswal/resume-llm-finetuning) — QLoRA fine-tuning for writing-help tasks*
- **Automated Prompt Engineering Pipeline — this repo**
