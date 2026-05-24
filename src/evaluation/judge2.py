import logging
from src.utils.llm import call_mini, parse_json
from src.utils.state import (
    PipelineState, TestCaseResult,
    SingleRunResult, AveragedResult,
)

log = logging.getLogger(__name__)


J2_RUBRIC_PROMPT = """You are a Lead AI Quality Assurance Architect specialising in
forensic prompt analysis and qualitative evaluation.

Your job is to generate qualitative rubric questions that a judge LLM will use
to evaluate model outputs. These questions must cover aspects that Python tests
CANNOT detect — they must require human-level judgment.

Task Description:
{task_description}

Output Specification:
{output_spec}

WHAT TO COVER:
1. Linguistic excellence — stylistic consistency, vocabulary, absence of AI-typical
   repetitive phrasing
2. Contextual fidelity — does output stay within scope? No irrelevant fluff?
3. Tone and persona alignment — is the tone appropriate for the task?
4. Clarity of intent — does the output actually solve the user's underlying need?
5. Human naturalness — does the response feel human in its reasoning and flow?

RULES:
- Maximum 5 questions. Focus on the most important aspects.
- Each question must be specific, not generic.
- No questions that could be answered by a Python test (structure, length, format).
- Do not use exact examples from the task in your questions.

Return ONLY a valid JSON array of strings. No preamble, no markdown fences.
"""


J2_GRADING_PROMPT = """You are an Elite AI Quality Auditor specialising in
forensic output analysis for production deployment decisions.

Task Description:
{task_description}

Qualitative Rubrics:
{rubrics}

Output A (Baseline — unoptimised prompt):
{output_a}

Output B (Target — optimised prompt):
{output_b}

GRADING SCALE (integers only: 1, 2, 3, 4, 5):
- 5 Exceptional: Perfect alignment. Flawless quality. Exceeds expectations.
- 4 Strong: Meets all criteria. Minor non-critical room for improvement.
- 3 Mediocre: Functional but flawed. Lacks depth or has tonal issues.
- 2 Poor: Significant failures. Ignores persona or has poor logical flow.
- 1 Fail: Completely misses the mark or violates the core task intent.

For each rubric:
1. Compare BOTH outputs against that rubric in context of the task.
2. Find specific phrases to justify your score.
3. Be critical — do not give 5 unless truly exceptional.

Return ONLY valid JSON:
{{
  "output_a": {{
    "scores": {{"<rubric_question>": <integer>, ...}},
    "overall_score": <integer>,
    "reasoning": "<detailed reasoning for Output A>"
  }},
  "output_b": {{
    "scores": {{"<rubric_question>": <integer>, ...}},
    "overall_score": <integer>,
    "reasoning": "<detailed reasoning for Output B>"
  }}
}}
"""


def get_j2_rubrics(state: PipelineState) -> list[str]:
    """Generate qualitative rubrics once. Cached in state.j2_rubrics."""
    if state.j2_rubrics:
        log.info("[J2] Using cached qualitative rubrics")
        return state.j2_rubrics

    log.info("[J2] Generating qualitative rubrics")
    raw = call_mini(
        prompt=J2_RUBRIC_PROMPT.format(
            task_description=state.task_description,
            output_spec=state.output_spec,
        ),
        temperature=0.0,
        max_tokens=800,
        call_type="j2_rubric_gen",
    )

    try:
        rubrics = parse_json(raw)
        assert isinstance(rubrics, list) and len(rubrics) >= 2
        log.info(f"[J2] Generated {len(rubrics)} rubrics")
        return rubrics
    except Exception as e:
        log.warning(f"[J2] Rubric parsing failed: {e} — using defaults")
        return [
            "Does the output maintain an appropriate professional tone?",
            "Is the output free from unnecessary filler or AI-typical phrasing?",
            "Does the output clearly address the user's underlying need?",
        ]


def _run_j2_once(
    tc: TestCaseResult,
    rubrics: list[str],
    state: PipelineState,
) -> SingleRunResult:
    prompt = J2_GRADING_PROMPT.format(
        task_description=state.task_description,
        output_a=tc.output_a,
        output_b=tc.output_b,
        rubrics="\n".join(f"- {r}" for r in rubrics),
    )

    raw = call_mini(
        prompt=prompt,
        temperature=0.2,
        max_tokens=1500,
        call_type=f"j2_grade_iter{state.iteration}_tc{tc.index}",
    )

    r = parse_json(raw)
    ra = r["output_a"]
    rb = r["output_b"]

    return SingleRunResult(
        score_a=float(ra.get("overall_score", 1)),
        score_b=float(rb.get("overall_score", 1)),
        reasoning_a=ra.get("reasoning", ""),
        reasoning_b=rb.get("reasoning", ""),
        raw_a={k: float(v) for k, v in ra.get("scores", {}).items()},
        raw_b={k: float(v) for k, v in rb.get("scores", {}).items()},
    )


def run_judge2(
    test_cases: list[TestCaseResult],
    rubrics: list[str],
    state: PipelineState,
) -> list[TestCaseResult]:
    """Run J2 qualitative judge for num_runs across all test cases."""
    log.info(
        f"[J2] iter={state.iteration} "
        f"{len(test_cases)} cases × {state.num_runs} runs"
    )

    for tc in test_cases:
        if tc.error:
            continue

        runs = []
        for run_idx in range(state.num_runs):
            try:
                result = _run_j2_once(tc, rubrics, state)
                runs.append(result)
                log.debug(
                    f"[J2] tc={tc.index} run={run_idx+1} "
                    f"A={result.score_a} B={result.score_b}"
                )
            except Exception as e:
                log.error(f"[J2] tc={tc.index} run={run_idx+1} failed: {e}")

        if not runs:
            continue

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

        tc.judge2 = AveragedResult(
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
            f"[J2] tc={tc.index} avg "
            f"A={tc.judge2.score_a} B={tc.judge2.score_b}"
        )

    return test_cases