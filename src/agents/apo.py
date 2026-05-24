import logging
from src.utils.llm import call_full, parse_json
from src.utils.state import PipelineState

log = logging.getLogger(__name__)


APO_PROMPT = """You are an expert prompt engineer specialising in iterative
prompt optimisation for production LLM systems.

Your job is to improve an existing prompt that has failed quality evaluation.
You will receive the current prompt, a failure analysis, and optional human feedback.

## CURRENT PROMPT (version {iteration})
{current_prompt}

## TASK DESCRIPTION
{task_description}

## OUTPUT SPECIFICATION
{output_spec}

## FAILURE ANALYSIS
{failure_summary}

{human_feedback_section}

## SCORE HISTORY
{score_history}

## YOUR TASK

Analyse the failures carefully. Then rewrite the prompt to fix them.

RULES FOR OPTIMISATION:
1. Keep what is working — do not remove constraints that are passing
2. Fix what is failing — address every failure point in the analysis
3. Be more explicit — vague instructions cause inconsistent outputs
4. Add examples only if they would prevent a recurring failure pattern
5. Add explicit edge case handling for any edge cases that failed
6. Do not make the prompt longer than necessary — precision over verbosity
7. The optimised prompt must be immediately usable as a system/user message

## WHAT TO CHECK BEFORE FINISHING
- Does the prompt explicitly address every failure in the analysis?
- Are all structural requirements stated as hard rules not suggestions?
- Are edge cases explicitly handled?
- Is the output format specified with zero ambiguity?

## OUTPUT
Return the optimised prompt as plain text only.
No preamble, no labels, no markdown fences.
The first word of your response must be the first word of the prompt.
"""


def run_apo(state: PipelineState) -> PipelineState:
    """
    APO node — Auto Prompt Optimizer.

    Takes:
    - state.current_prompt       (prompt that failed)
    - state.history[-1].failure_summary  (what went wrong)
    - state.human_feedback       (optional, from human approval node)
    - state.iteration            (current version number)

    Produces:
    - state.current_prompt       (optimised prompt v{n+1})
    - state.iteration            (incremented)

    This is the core of the optimisation loop.
    In production at BOLD this reduced the manual prompt engineering
    cycle from 3 days to under 1 hour.
    """
    log.info(
        f"[APO] Optimising prompt "
        f"iter={state.iteration} → {state.iteration + 1}"
    )

    # ── Get failure summary from last iteration ───────────────────────────────
    failure_summary = "No failure summary available."
    if state.history:
        last = state.history[-1]
        if last.failure_summary:
            failure_summary = last.failure_summary

    # ── Human feedback section ────────────────────────────────────────────────
    if state.human_feedback and state.human_feedback.strip():
        human_feedback_section = (
            f"## HUMAN FEEDBACK\n"
            f"The following feedback was provided by a human reviewer:\n"
            f"{state.human_feedback.strip()}\n"
            f"This feedback takes priority over automated failure analysis."
        )
    else:
        human_feedback_section = (
            "## HUMAN FEEDBACK\nNone provided. "
            "Rely on the failure analysis above."
        )

    # ── Score history ─────────────────────────────────────────────────────────
    score_lines = []
    for record in state.history:
        score_lines.append(
            f"  Iteration {record.iteration}: "
            f"Boss avg = {record.avg_boss_score:.3f}/5 "
            f"({'PASSED' if record.passed else 'FAILED'})"
        )
    score_history = (
        "\n".join(score_lines)
        if score_lines
        else "  No prior iterations."
    )

    # ── Build APO prompt ──────────────────────────────────────────────────────
    apo_prompt = APO_PROMPT.format(
        iteration=state.iteration,
        current_prompt=state.current_prompt,
        task_description=state.task_description,
        output_spec=state.output_spec,
        failure_summary=failure_summary,
        human_feedback_section=human_feedback_section,
        score_history=score_history,
    )

    # ── Call gpt-4o — only node that needs deep reasoning ────────────────────
    raw = call_full(
        prompt=apo_prompt,
        temperature=0.3,
        max_tokens=2000,
        call_type=f"apo_iter{state.iteration + 1}",
    )

    optimised_prompt = raw.strip()
    new_iteration    = state.iteration + 1

    log.info(
        f"[APO] Generated prompt v{new_iteration} "
        f"({len(optimised_prompt)} chars)"
    )

    # ── Log prompt diff summary ───────────────────────────────────────────────
    old_len = len(state.current_prompt)
    new_len = len(optimised_prompt)
    log.info(
        f"[APO] Prompt length: {old_len} → {new_len} chars "
        f"({'grew' if new_len > old_len else 'shrunk'})"
    )

    return state.model_copy(update={
        "current_prompt": optimised_prompt,
        "iteration":      new_iteration,
        "human_feedback": None,   # clear after use
    })