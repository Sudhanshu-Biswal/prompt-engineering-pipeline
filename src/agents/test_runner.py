import logging
from src.utils.llm import call_mini, parse_json
from src.utils.state import PipelineState, TestCaseResult

log = logging.getLogger(__name__)


# ── Source output generator ───────────────────────────────────────────────────
# In production at BOLD, source_outputs were GPT-4o-mini baseline outputs
# collected before the new prompt was introduced.
# In this public version we generate them once using a plain prompt
# (no system prompt optimisation) as the baseline to compare against.

BASELINE_PROMPT = """You are a helpful assistant. Complete the following task.

Task: {task_description}

Input:
{input_text}

Respond with the output only. No preamble."""


def generate_source_outputs(state: PipelineState) -> PipelineState:
    """
    Generates baseline (source model) outputs using a plain unoptimised prompt.
    This runs ONCE at the start of the pipeline — these become output_a in judges.
    Skipped if source_outputs already provided.
    """
    if state.source_outputs:
        log.info("[TestRunner] source_outputs already provided — skipping baseline gen")
        return state

    log.info(f"[TestRunner] Generating {len(state.input_texts)} baseline outputs")

    source_outputs = []
    for i, input_text in enumerate(state.input_texts):
        prompt = BASELINE_PROMPT.format(
            task_description=state.task_description,
            input_text=input_text,
        )
        try:
            output = call_mini(
                prompt=prompt,
                temperature=0.0,
                max_tokens=1024,
                call_type=f"baseline_{i}",
            )
            source_outputs.append(output.strip())
            log.debug(f"[TestRunner] Baseline {i}: {output[:80]}...")
        except Exception as e:
            log.error(f"[TestRunner] Baseline {i} failed: {e}")
            source_outputs.append("")

    log.info(f"[TestRunner] Generated {len(source_outputs)} baseline outputs")
    return state.model_copy(update={"source_outputs": source_outputs})


# ── Target output generator ───────────────────────────────────────────────────
def run_test_cases(state: PipelineState) -> list[TestCaseResult]:
    """
    Runs the current_prompt against all input_texts.
    Returns TestCaseResult list with output_a (baseline) and output_b (current prompt).
    """
    log.info(
        f"[TestRunner] iter={state.iteration} "
        f"running {len(state.input_texts)} test cases"
    )

    results = []
    for i, (input_text, source_output) in enumerate(
        zip(state.input_texts, state.source_outputs)
    ):
        # Build the prompt — current_prompt is the system context,
        # input_text is the user message
        full_prompt = f"{state.current_prompt}\n\nInput:\n{input_text}"

        try:
            target_output = call_mini(
                prompt=full_prompt,
                temperature=0.0,    # deterministic for evaluation
                max_tokens=1024,
                call_type=f"test_run_iter{state.iteration}_{i}",
            )
            target_output = target_output.strip()
            error = None
        except Exception as e:
            log.error(f"[TestRunner] test case {i} failed: {e}")
            target_output = ""
            error = str(e)

        results.append(TestCaseResult(
            index=i,
            input_text=input_text,
            output_a=source_output,     # baseline — GPT-4o-mini plain prompt
            output_b=target_output,     # current optimised prompt output
            error=error,
        ))

        log.debug(
            f"[TestRunner] case {i} "
            f"A={source_output[:60]!r} "
            f"B={target_output[:60]!r}"
        )

    log.info(f"[TestRunner] Completed {len(results)} test cases")
    return results