# Automated Prompt Engineering Pipeline

> **APG + APO** — Auto Prompt Generation and Optimization using a
> LangGraph workflow with a tool-using APO agent, 3-judge evaluation
> framework, and statistical hypothesis testing.

Built as a production-grade public implementation of the prompt engineering
automation system developed at [BOLD](https://www.bold.com/) — reducing
manual prompt engineering cycles from **3 days to under 1 hour**.

---

## The Problem

Prompt engineering for new LLM tasks was a slow, manual, 3-day cycle:

```
Write prompt → test manually → review failures → rewrite → repeat
```

No systematic evaluation. No statistical validation. No way to prove
the new prompt was actually better than the baseline.

---

## The Solution

A fully automated pipeline with a tool-using APO agent at its core:

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
   ┌──────────────────────────────────────────┐
   │  J1 — Structural Judge  (gpt-5.4-nano)   │
   │  J2 — Qualitative Judge (gpt-5.4-nano)   │
   │  Boss — Synthesiser     (gpt-5.4-mini)   │
   └──────────────────────────────────────────┘
          │
     Pass / Fail
          │
    ┌─────┴──────┐
  PASS          FAIL
    │              │
Human          APO Agent — Tool-using Auto Prompt Optimizer
Approval       │
    │          ├── search_prompt_history (ChromaDB)
    │          ├── get_rubric_failures (structured data)
    │          └── generates optimised prompt v{n+1}
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

> DSPy optimises prompt *selection* — picking the best few-shot examples
> for a fixed template. APG/APO optimises prompt *construction* —
> generating and iteratively improving the full prompt structure.
> Fundamentally different approaches. APG/APO wins on structured
> writing-help tasks.

---

## Score Progression (example run)

```
v1  3.21/5  ████████████         ✗ failed
             APO agent: called get_rubric_failures
                        called search_prompt_history (2 results)
             
v2  3.84/5  ███████████████      ✗ failed
             APO agent: called get_rubric_failures
             
v3  4.31/5  █████████████████    ✓ PASSED

Statistical test : Wilcoxon signed-rank
p-value          : 0.0031
Mean difference  : +0.847
Verdict          : better
Migration        : APPROVED
```

---

## Architecture

### LangGraph workflow

```
START → APG → Baseline → TestRunner → Evaluator → Router
                              ↑                       │
                              │            ┌──────────┼──────────┐
                              │         APO Agent  HumanApproval  Hypothesis
                              │            │          │              │
                              └────────────┘     APO/Hypo          END
```

### Node responsibilities

| Node | Model | Job |
|---|---|---|
| APG | gpt-5.4-nano | Generate prompt v1 from task description + output spec |
| Baseline | gpt-5.4-nano | Generate source outputs using plain prompt (output_a) |
| TestRunner | gpt-5.4-nano | Run current prompt on all test cases (output_b) |
| J1 | gpt-5.4-nano | Structural evaluation — format, length, constraints |
| J2 | gpt-5.4-nano | Qualitative evaluation — tone, clarity, naturalness |
| Boss | **gpt-5.4-mini** | Synthesise J1+J2, produce final float score |
| Router | — | Conditional edge: pass→human, fail→APO, max→hypothesis |
| **APO Agent** | **gpt-5.4-mini** | Tool-using agent — retrieves context, optimises prompt |
| HumanApproval | — | Checkpoint: approve / reject / provide feedback |
| Hypothesis | — | Wilcoxon or permutation test, migration decision |

**Model routing rationale:** gpt-5.4-nano for generation and scoring
(fast, cheap — $0.20/$1.25 per 1M tokens). gpt-5.4-mini only for APO
Agent and Boss — the two nodes requiring deeper reasoning
($0.75/$4.50 per 1M tokens). Reduces cost by ~75% vs running
gpt-5.4-mini everywhere.

---

## APO — Tool-using Agent

The Auto Prompt Optimizer is a **tool-using agent**, not a single LLM call.

### The problem with a single LLM call

Without tools, APO sees only a text summary of failures:

```
"The prompt failed because sentences were too long and 
the tone was too robotic in some cases."
```

Vague. Unstructured. The LLM has to guess what to fix.

### With tools, APO reasons before optimising

```
APO Agent starts
  
  "The failure summary mentions length issues.
   Let me get the exact rubric data first."
  → calls get_rubric_failures
     Returns: J1 length compliance 2.1/5 in 7/10 cases
              J2 robotic tone 2.8/5 in 4/10 cases
  
  "Length is the bigger issue. Let me find prompts
   that solved this for similar tasks."
  → calls search_prompt_history
     query="prompt that enforces 8-13 word length constraint"
     Returns: 2 passing prompts from previous iterations
              showing how explicit word count enforcement worked

  → DONE
  → writes prompt v2 using retrieved context
```

### Tools available

| Tool | When APO calls it | What it returns |
|---|---|---|
| `search_prompt_history` | Structural failures — needs reference examples | Similar passing prompts from ChromaDB vector store |
| `get_rubric_failures` | Vague failure summary — needs precise data | Per-rubric scores, fail counts, worst test cases |

### Agent loop

```
Observe → failure analysis + score history
Orient  → call tools (max 4 calls per iteration)
Decide  → DONE
Act     → write optimised prompt using retrieved context
```

Fallback: if agent loop produces no valid prompt,
single-call APO runs as safety net.

### Why this is agentic

The agent **decides** which tools to call based on what it reads
in the failure analysis. It doesn't follow a fixed sequence.
Two different failures → two different tool call patterns →
two different optimisation strategies. That's agency.

**What this is NOT:** A multi-agent system. There is one agent
(APO), one LangGraph graph, and one state object. Other nodes
are deterministic functions. This is positioned accurately as
a tool-using agent inside a workflow — not overclaimed as
a multi-agent system.

---

## 3-Judge Evaluation Design

**J1 — Structural Judge (gpt-5.4-nano)**
Auto-generates rubrics from the task description.
Checks: format compliance, length constraints, required fields,
JSON validity. Scores 1–5 per rubric, integers only.

**J2 — Qualitative Judge (gpt-5.4-nano)**
Auto-generates qualitative rubrics Python tests cannot detect.
Checks: tone, linguistic quality, contextual fidelity, naturalness.
Scores 1–5 per rubric, integers only.

**Boss (gpt-5.4-mini)**
Synthesises J1 + J2 with expert weighting per task type.
Produces float score 1.0–5.0 with confidence rating.
This is the pass/fail metric. Threshold: 4.0/5 (configurable).

All three judges run N times per test case (default: 3).
Scores averaged across runs for statistical stability.

---

## Hypothesis Testing

After the final iteration, the system automatically selects
the appropriate statistical test based on score distribution:

**Wilcoxon signed-rank test** — when score differences are
symmetric (|skewness| < 1.0). Non-parametric — no normality
assumption required.

**Permutation test** — when differences are not symmetric.
10,000 permutations, two-tailed. More conservative.

Both compare the optimised prompt's Boss scores against
the baseline across all test cases × all runs.

Migration threshold: p < 0.05.

---

## Project Structure

```
prompt-engineering-pipeline/
├── src/
│   ├── agents/
│   │   ├── apg.py              # Auto Prompt Generator
│   │   ├── apo.py              # APO tool-using agent loop
│   │   ├── tools.py            # search_prompt_history + get_rubric_failures
│   │   ├── test_runner.py      # Baseline + target output generation
│   │   └── router.py           # Pass/fail routing logic
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
│   ├── improve_work_history.json
│   └── grammar_correction.json
├── results/
├── .env.example
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Sudhanshu-Biswal/prompt-engineering-pipeline
cd prompt-engineering-pipeline
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
```

`.env`:
```
OPENAI_API_KEY=your_key_here
JUDGE_MODEL=gpt-5.4-nano
APO_MODEL=gpt-5.4-mini
BOSS_MODEL=gpt-5.4-mini
```

### 3. Run with an example config

```bash
# Resume bullet improvement task
python -m src.main --config configs/improve_work_history.json

# Non-interactive (auto-approves at human checkpoint)
python -m src.main --config configs/improve_work_history.json --non-interactive

# Grammar correction task
python -m src.main --config configs/grammar_correction.json --non-interactive
```

### 4. Bring your own task

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

## Config Reference

| Field | Type | Default | Description |
|---|---|---|---|
| `task_name` | string | required | Identifier for this task |
| `task_description` | string | required | What the model must do |
| `output_spec` | string | required | Exact output format / schema |
| `output_is_json` | bool | true | Enables JSON validity pre-check in J1 |
| `num_runs` | int | 3 | Judge runs per test case |
| `pass_threshold` | float | 4.0 | Boss score required to pass (1–5) |
| `max_iterations` | int | 3 | Max APO iterations before stopping |
| `input_texts` | list | [] | Optional: provide your own test inputs |
| `source_outputs` | list | [] | Optional: provide your own baseline outputs |
| `human_feedback` | string | null | Optional: seed APG with prior observations |

---

## Output

Each run saves to `results/`:

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

## Cost Estimate

Approximate cost per pipeline run (3 iterations, 10 test cases, 3 runs):

| Component | Model | Pricing | Approx cost |
|---|---|---|---|
| APG + test case gen | gpt-5.4-nano | $0.20/$1.25 per 1M | ~$0.01 |
| 3× TestRunner (30 calls) | gpt-5.4-nano | $0.20/$1.25 per 1M | ~$0.02 |
| 3× J1 (90 calls) | gpt-5.4-nano | $0.20/$1.25 per 1M | ~$0.03 |
| 3× J2 (90 calls) | gpt-5.4-nano | $0.20/$1.25 per 1M | ~$0.04 |
| 3× Boss (90 calls) | gpt-5.4-mini | $0.75/$4.50 per 1M | ~$0.35 |
| 3× APO Agent | gpt-5.4-mini | $0.75/$4.50 per 1M | ~$0.18 |
| Hypothesis + failure summary | gpt-5.4-mini | $0.75/$4.50 per 1M | ~$0.04 |
| **Total** | | | **~$0.67** |

Cost logged per run in results JSON and terminal summary.

---

## Why Not DSPy?

DSPy's optimisers (MIPRO, BootstrapFewShot) select the best
few-shot examples for a fixed prompt template.

This system solves a different problem: constructing a prompt
from scratch and iteratively improving it based on structured
failure analysis from a 3-judge framework.

| Dimension | DSPy | APG + APO |
|---|---|---|
| What it optimises | Example selection | Full prompt construction |
| Failure signal | Single metric | J1 + J2 + Boss (3 dimensions) |
| Tool use | No | Yes (ChromaDB + rubric data) |
| Statistical validation | No | Wilcoxon / permutation test |
| Output | Best few-shot config | Production-ready prompt |

On structured writing-help tasks: APG/APO wins.
On classification/retrieval with few-shot examples: DSPy wins.

---

## Production Context

This is a public implementation of the prompt engineering automation
system built at [BOLD](https://www.bold.com/) for writing-help
workloads — resume bullets, grammar correction, work history
improvement, cover letters.

In production: BOLD's internal LLM gateway, proprietary writing data,
proprietary evaluation test cases.

This public version: OpenAI directly, generated test cases,
same architecture and evaluation framework.

**Impact:** Reduced prompt engineering cycle from 3 days (manual)
to under 1 hour (automated) across multiple writing-help use cases.

---

## Requirements

```
Python 3.10+
openai>=1.30.0
langgraph>=0.1.0
pydantic>=2.0.0
scipy>=1.13.0
numpy>=1.26.0
chromadb>=0.5.0
sentence-transformers>=2.7.0
```

---

## Author

**Sudhanshu Sekhar Biswal**
Senior AI Engineer · Patent Holder · CII National AI Award Winner

[GitHub](https://github.com/Sudhanshu-Biswal) ·
[LinkedIn](https://linkedin.com/in/sudhanshubiswal)

---

*Part of a 3-project portfolio demonstrating production GenAI engineering:*
- *[Career Intelligence RAG](https://github.com/Sudhanshu-Biswal/career-intelligence-rag) — HyDE + hybrid retrieval + RAGAS evaluation*
- *[Resume Writing LLM Fine-tuning](https://github.com/Sudhanshu-Biswal/resume-llm-finetuning) — QLoRA fine-tuning for writing-help tasks*
- **Automated Prompt Engineering Pipeline — this repo**
