import logging
from src.utils.llm import call_mini, parse_json
from src.utils.state import (
    PipelineState, TestCaseResult,
    SingleRunResult, AveragedResult,
)

log = logging.getLogger(__name__)


# ── Rubric generator ──────────────────────────────────────────────────────────
J1_RUBRIC_PROMPT = """You are an expert AI evaluator specialising in structural output validation.

A language model was given the task prompt below and produced an output.
Your job is to generate structural rubric questions to grade whether that
model's OUTPUT correctly follows the task prompt's explicit requirements.

IMPORTANT RULES:
- Generate ONE rubric question per distinct structural requirement in the prompt.
- Minimum 3 questions, maximum 7 questions.
- Do NOT pad with invented requirements.
- Each question must be answerable with a score of 1-5 by inspecting the output directly.
- Do NOT generate meta-questions about rubrics or evaluation criteria themselves.

WHAT TO LOOK FOR:
- Output format: Is a specific format required? (JSON, plain text, numbered list, specific keys)
- Output length: Is a specific count or limit required?
- Content rules: Are there forbidden or required elements?
- Punctuation rules: Are there explicit punctuation requirements?
- Faithfulness: Must output stay close to input?

Task Prompt:
{task_description}

Output Specification:
{output_spec}

Return ONLY a valid JSON array of strings. No preamble, no markdown fences.
"""


# ── Grading prompt ────────────────────────────────────────────────────────────
J1_GRADING_PROMPT = """You are a strict structural evaluator.

Task Description:
{task_description}

Output A (Baseline — unoptimised prompt):
{output_a}

Output B (Target — optimised prompt):
{output_b}

Structural Rubric Questions:
{rubrics}

Grade EACH output against EACH rubric question:
- Scores MUST be integers only: 1, 2, 3, 4, or 5. No decimals.
- 5: Fully meets this structural requirement
- 3: Partially meets it
- 1: Completely fails this requirement

Be strict. Do not give 5 unless the requirement is perfectly met.
overall_score is the average of all rubric scores.

Return ONLY valid JSON in this exact format:
{{
  "output_a": {{
    "scores": {{"<rubric_question>": <integer>, ...}},
    "overall_score": <integer>,
    "reasoning": "<concise explanation for Output A>"
  }},
  "output_b": {{
    "scores": {{"<rubric_question>": <integer>, ...}},
    "overall_score": <integer>,
    "reasoning": "<concise explanation for Output B>"
  }}
}}
"""


def get_j1_rubrics(state: PipelineState) -> list[str]:
    """
    Generate structural rubrics once per pipeline run.
    Cached in state.j1_rubrics after first call.
    """
    if state.j1_rubrics:
        log.info("[J1] Using cached structural rubrics")
        return state.j1_rubrics

    log.info("[J1] Generating structural rubrics")
    raw = call_mini(
        prompt=J1_RUBRIC_PROMPT.format(
            task_description=state.task_description,
            output_spec=state.output_spec,
        ),
        temperature=0.0,
        max_tokens=800,
        call_type="j1_rubric_gen",
    )

    try:
        rubrics = parse_json(raw)
        assert isinstance(rubrics, list) and len(rubrics) >= 2
        log.info(f"[J1] Generated {len(rubrics)} rubrics")
        return rubrics
    except Exception as e:
        log.warning(f"[J1] Rubric parsing failed: {e} — using defaults")
        return [
            "Does the output follow the required format exactly?",
            "Does the output contain all required fields or sections?",
            "Does the output respect length or count constraints?",
        ]


def _run_j1_once(
    tc: TestCaseResult,
    rubrics: list[str],
    state: PipelineState,
) -> SingleRunResult:

    # If output should be JSON, pre-check validity
    # Invalid JSON immediately scores 1 — mirrors production behaviour
    if state.output_is_json:
        import json as _json
        def _is_valid(text):
            try:
                _json.loads(text)
                return True
            except Exception:
                return False

        a_valid = _is_valid(tc.output_a)
        b_valid = _is_valid(tc.output_b)

        # If both invalid — skip LLM call entirely, force score 1
        if not a_valid and not b_valid:
            log.info(f"[J1] tc={tc.index} both outputs invalid JSON — forcing score 1")
            return SingleRunResult(
                score_a=1.0, score_b=1.0,
                reasoning_a="Invalid JSON — structural evaluation skipped.",
                reasoning_b="Invalid JSON — structural evaluation skipped.",
            )
    else:
        a_valid = True
        b_valid = True

    prompt = J1_GRADING_PROMPT.format(
        task_description=state.task_description,
        output_a=tc.output_a,
        output_b=tc.output_b,
        rubrics="\n".join(f"- {r}" for r in rubrics),
    )

    raw = call_mini(
        prompt=prompt,
        temperature=0.0,
        max_tokens=1000,
        call_type=f"j1_grade_iter{state.iteration}_tc{tc.index}",
    )

    r = parse_json(raw)
    ra = r["output_a"]
    rb = r["output_b"]

    # Override invalid JSON outputs after the LLM call
    if state.output_is_json and not a_valid:
        ra = {"overall_score": 1, "scores": {}, 
              "reasoning": "Invalid JSON — score forced to 1."}
    if state.output_is_json and not b_valid:
        rb = {"overall_score": 1, "scores": {},
              "reasoning": "Invalid JSON — score forced to 1."}

    return SingleRunResult(
        score_a=float(ra.get("overall_score", 1)),
        score_b=float(rb.get("overall_score", 1)),
        reasoning_a=ra.get("reasoning", ""),
        reasoning_b=rb.get("reasoning", ""),
        raw_a={k: float(v) for k, v in ra.get("scores", {}).items()},
        raw_b={k: float(v) for k, v in rb.get("scores", {}).items()},
    )

def run_judge1(
    test_cases: list[TestCaseResult],
    rubrics: list[str],
    state: PipelineState,
) -> list[TestCaseResult]:
    """
    Run J1 structural judge for num_runs across all test cases.
    Averages scores across runs and stores in tc.judge1.
    """
    log.info(
        f"[J1] iter={state.iteration} "
        f"{len(test_cases)} cases × {state.num_runs} runs"
    )

    for tc in test_cases:
        if tc.error:
            log.warning(f"[J1] Skipping tc={tc.index} — has error")
            continue

        runs = []
        for run_idx in range(state.num_runs):
            try:
                result = _run_j1_once(tc, rubrics, state)
                runs.append(result)
                log.debug(
                    f"[J1] tc={tc.index} run={run_idx+1} "
                    f"A={result.score_a} B={result.score_b}"
                )
            except Exception as e:
                log.error(f"[J1] tc={tc.index} run={run_idx+1} failed: {e}")

        if not runs:
            continue

        # Average across runs
        scores_a = [r.score_a for r in runs]
        scores_b = [r.score_b for r in runs]

        all_keys = set()
        for r in runs:
            all_keys.update(r.raw_a.keys())
            all_keys.update(r.raw_b.keys())

        per_rubric_a = {
            k: round(sum(r.raw_a.get(k, 0) for r in runs) / len(runs), 3)
            for k in all_keys
        }
        per_rubric_b = {
            k: round(sum(r.raw_b.get(k, 0) for r in runs) / len(runs), 3)
            for k in all_keys
        }

        tc.judge1 = AveragedResult(
            score_a=round(sum(scores_a) / len(scores_a), 3),
            score_b=round(sum(scores_b) / len(scores_b), 3),
            reasoning_a=runs[0].reasoning_a,
            reasoning_b=runs[0].reasoning_b,
            per_rubric_a=per_rubric_a,
            per_rubric_b=per_rubric_b,
            all_scores_a=scores_a,
            all_scores_b=scores_b,
            runs=runs,
        )

        log.info(
            f"[J1] tc={tc.index} avg "
            f"A={tc.judge1.score_a} B={tc.judge1.score_b}"
        )

    return test_cases