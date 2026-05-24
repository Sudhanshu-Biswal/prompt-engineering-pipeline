import logging
from src.utils.llm import call_full, parse_json
from src.utils.state import (
    PipelineState, TestCaseResult,
    SingleRunResult, AveragedResult,
)

log = logging.getLogger(__name__)


BOSS_PROMPT = """You are the final evaluation authority for an LLM prompt optimisation decision.

Your job is to synthesise scores from two specialist judges and produce a
final score for each model output.

Task Description:
{task_description}

Output A (Baseline — unoptimised prompt):
{output_a}

Output B (Target — optimised prompt):
{output_b}

Judge 1 — Structural Analysis
Rubrics: {j1_rubrics}
Output A: {j1_score_a:.2f}/5 | Reasoning: {j1_reasoning_a}
Output B: {j1_score_b:.2f}/5 | Reasoning: {j1_reasoning_b}

Judge 2 — Qualitative Analysis
Rubrics: {j2_rubrics}
Output A: {j2_score_a:.2f}/5 | Reasoning: {j2_reasoning_a}
Output B: {j2_score_b:.2f}/5 | Reasoning: {j2_reasoning_b}

YOUR TASK:
1. Review both judge scores and reasoning carefully.
2. Decide how to weight structural vs qualitative scores based on what
   matters most for this specific task. Use expert judgment — no fixed formula.
3. Produce a final score for each output as a float between 1.0 and 5.0.
4. Provide separate reasoning for each output.
5. Rate your overall confidence: high, medium, or low.

Do NOT declare a winner. Do NOT compare A vs B.
Score each output independently on its own merit.

Return ONLY valid JSON:
{{
  "final_score_a": <float 1.0-5.0>,
  "reasoning_a": "<what drove Output A's score>",
  "final_score_b": <float 1.0-5.0>,
  "reasoning_b": "<what drove Output B's score>",
  "confidence": "<high|medium|low>"
}}
"""


FAILURE_SUMMARY_PROMPT = """You are an expert prompt engineering analyst.

A prompt was tested against {n_cases} test cases and scored below the
passing threshold of {threshold}/5.

Boss Judge Final Scores:
- Output A (baseline) average: {avg_a:.3f}/5
- Output B (optimised prompt) average: {avg_b:.3f}/5

Per-test-case results:
{case_details}

J1 Structural Rubrics used:
{j1_rubrics}

J2 Qualitative Rubrics used:
{j2_rubrics}

YOUR TASK:
Produce a concise, actionable failure analysis for the prompt engineer.

Include:
1. PRIMARY FAILURES — what specific requirements did Output B consistently fail?
2. PATTERNS — are there recurring failure modes across test cases?
3. SPECIFIC FIXES — what exact changes to the prompt would address each failure?
4. EDGE CASES — which test case types caused the most failures?

Be specific. Reference actual rubric questions and score patterns.
This analysis will be fed directly into the Auto Prompt Optimizer (APO).

Return plain text only. No JSON, no markdown headers. Max 300 words.
"""


def _run_boss_once(
    tc: TestCaseResult,
    j1_rubrics: list[str],
    j2_rubrics: list[str],
    state: PipelineState,
) -> SingleRunResult:
    """Run Boss judge once for a single test case."""

    # Need J1 and J2 results to exist
    if not tc.judge1 or not tc.judge2:
        raise ValueError(f"tc={tc.index} missing J1 or J2 results")

    prompt = BOSS_PROMPT.format(
        task_description=state.task_description,
        output_a=tc.output_a,
        output_b=tc.output_b,
        j1_rubrics=", ".join(j1_rubrics),
        j1_score_a=tc.judge1.score_a,
        j1_reasoning_a=tc.judge1.reasoning_a,
        j1_score_b=tc.judge1.score_b,
        j1_reasoning_b=tc.judge1.reasoning_b,
        j2_rubrics=", ".join(j2_rubrics),
        j2_score_a=tc.judge2.score_a,
        j2_reasoning_a=tc.judge2.reasoning_a,
        j2_score_b=tc.judge2.score_b,
        j2_reasoning_b=tc.judge2.reasoning_b,
    )

    raw = call_full(
        prompt=prompt,
        temperature=0.3,
        max_tokens=800,
        call_type=f"boss_iter{state.iteration}_tc{tc.index}",
    )

    r = parse_json(raw)

    return SingleRunResult(
        score_a=float(r["final_score_a"]),
        score_b=float(r["final_score_b"]),
        reasoning_a=r.get("reasoning_a", ""),
        reasoning_b=r.get("reasoning_b", ""),
        raw_a={},
        raw_b={},
    )


def run_boss(
    test_cases: list[TestCaseResult],
    j1_rubrics: list[str],
    j2_rubrics: list[str],
    state: PipelineState,
) -> list[TestCaseResult]:
    """
    Run Boss judge for num_runs across all test cases.
    Boss uses gpt-4o — only judge that does.
    """
    log.info(
        f"[Boss] iter={state.iteration} "
        f"{len(test_cases)} cases × {state.num_runs} runs"
    )

    for tc in test_cases:
        if tc.error or not tc.judge1 or not tc.judge2:
            log.warning(f"[Boss] Skipping tc={tc.index}")
            continue

        runs = []
        for run_idx in range(state.num_runs):
            try:
                result = _run_boss_once(tc, j1_rubrics, j2_rubrics, state)
                runs.append(result)
                log.debug(
                    f"[Boss] tc={tc.index} run={run_idx+1} "
                    f"A={result.score_a:.2f} B={result.score_b:.2f}"
                )
            except Exception as e:
                log.error(
                    f"[Boss] tc={tc.index} run={run_idx+1} failed: {e}"
                )

        if not runs:
            continue

        scores_a = [r.score_a for r in runs]
        scores_b = [r.score_b for r in runs]

        tc.boss = AveragedResult(
            score_a=round(sum(scores_a) / len(scores_a), 3),
            score_b=round(sum(scores_b) / len(scores_b), 3),
            reasoning_a=runs[0].reasoning_a,
            reasoning_b=runs[0].reasoning_b,
            per_rubric_a={},
            per_rubric_b={},
            all_scores_a=scores_a,
            all_scores_b=scores_b,
            runs=runs,
        )

        log.info(
            f"[Boss] tc={tc.index} avg "
            f"A={tc.boss.score_a} B={tc.boss.score_b}"
        )

    return test_cases


def compute_avg_boss_score(test_cases: list[TestCaseResult]) -> float:
    """Average Boss score_b across all test cases — this is the pass/fail metric."""
    scores = [
        tc.boss.score_b
        for tc in test_cases
        if tc.boss is not None
    ]
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 3)


def generate_failure_summary(
    test_cases: list[TestCaseResult],
    j1_rubrics: list[str],
    j2_rubrics: list[str],
    state: PipelineState,
) -> str:
    """
    Generates a concise failure analysis after a failed iteration.
    This is fed directly into APO as the primary failure signal.
    """
    log.info(f"[Boss] Generating failure summary for iter={state.iteration}")

    # Build per-case detail string
    case_details_lines = []
    for tc in test_cases:
        if not tc.boss:
            continue
        line = (
            f"Case {tc.index}: "
            f"Boss A={tc.boss.score_a} B={tc.boss.score_b} | "
            f"J1 B={tc.judge1.score_b if tc.judge1 else 'N/A'} | "
            f"J2 B={tc.judge2.score_b if tc.judge2 else 'N/A'} | "
            f"Input: {tc.input_text[:80]!r}"
        )
        case_details_lines.append(line)

    avg_a = sum(
        tc.boss.score_a for tc in test_cases if tc.boss
    ) / max(1, sum(1 for tc in test_cases if tc.boss))

    avg_b = compute_avg_boss_score(test_cases)

    prompt = FAILURE_SUMMARY_PROMPT.format(
        n_cases=len(test_cases),
        threshold=state.pass_threshold,
        avg_a=avg_a,
        avg_b=avg_b,
        case_details="\n".join(case_details_lines),
        j1_rubrics="\n".join(f"- {r}" for r in j1_rubrics),
        j2_rubrics="\n".join(f"- {r}" for r in j2_rubrics),
    )

    summary = call_full(
        prompt=prompt,
        temperature=0.2,
        max_tokens=600,
        call_type=f"failure_summary_iter{state.iteration}",
    )

    log.info(f"[Boss] Failure summary: {summary[:120]}...")
    return summary.strip()