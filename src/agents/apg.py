import logging
from src.utils.llm import call_mini, parse_json
from src.utils.state import PipelineState

log = logging.getLogger(__name__)

# ── APG Meta-Prompt ───────────────────────────────────────────────────────────
# This is the core of APG — it takes a task description + output spec
# and generates a production-grade prompt from scratch.
# Based on the actual APG system built at BOLD for writing-help workloads.

APG_META_PROMPT = """You are an expert prompt engineer specialising in production LLM systems.

Your job is to generate a high-quality, production-ready prompt for a language model given:
1. A task description — what the model must do
2. An output specification — the exact format the model must return
3. Optional human feedback — prior observations about what went wrong

## WHAT MAKES A PRODUCTION PROMPT

A production prompt must have ALL of the following:

### ROLE
Assign the model a precise expert role relevant to the task.
Do not use generic roles like "helpful assistant."

### TASK DEFINITION
State exactly what the model must do. Be explicit about:
- What the input is
- What transformation or generation is required
- What the output must contain

### CONSTRAINTS
List hard rules the model must follow. Examples:
- Output format (JSON keys, structure, types)
- Length limits (word count, character count, number of items)
- What to NEVER do (add pronouns, rephrase beyond the fix, hallucinate)
- Faithfulness rules (stay close to input, do not invent)

### OUTPUT FORMAT
Provide the exact output structure. If JSON — show the schema with key names and types.
If plain text — describe the exact format precisely.

### EXAMPLES (if helpful)
Include 1-2 input/output examples only if they significantly clarify edge cases.
Do not pad with trivial examples.

### EDGE CASES
Explicitly handle what the model should do when:
- Input is empty or malformed
- The task cannot be completed (e.g. nothing to fix)
- Output would exceed length limits

## RULES FOR PROMPT GENERATION
- Write the prompt in second person ("You are...", "Your task is...")
- Be explicit, not suggestive. "Return ONLY valid JSON" not "Try to return JSON"
- Every constraint must be specific and testable
- Do not include meta-commentary about the prompt itself
- Do not wrap the prompt in markdown fences or labels
- The prompt must be immediately usable as a system/user message

## INPUTS

Task Description:
{task_description}

Output Specification:
{output_spec}

{feedback_section}

## OUTPUT
Return the generated prompt as plain text only.
No preamble, no labels, no markdown fences.
The first word of your response should be the first word of the generated prompt.
"""

# ── Test Case Generator ───────────────────────────────────────────────────────
TEST_CASE_GEN_PROMPT = """You are an expert QA engineer for LLM systems.

Your job is to generate diverse, realistic test cases for the following task.

Task Description:
{task_description}

Output Specification:
{output_spec}

## REQUIREMENTS
- Generate exactly {n} test cases
- Each test case must be a realistic input the model would receive in production
- Cover a range of difficulty: simple cases, edge cases, tricky cases
- Edge cases to include: empty input, very short input, input with errors,
  input that is already correct (nothing to fix), unusual formatting

## OUTPUT FORMAT
Return ONLY a valid JSON array of objects. Each object must have:
{{
  "index": <integer starting at 0>,
  "input_text": "<the input the model will process>",
  "description": "<one sentence describing what makes this test case interesting>"
}}

No preamble, no markdown fences. JSON array only.
"""


def run_apg(state: PipelineState) -> PipelineState:
    """
    APG node — Auto Prompt Generator.

    Takes: task_description, output_spec, optional human_feedback
    Produces: current_prompt (v1), input_texts (test cases if not provided)

    This is always the first node in the pipeline.
    """
    log.info(f"[APG] Generating prompt for task: {state.task_name}")

    # ── Build feedback section ────────────────────────────────────────────────
    if state.human_feedback and state.human_feedback.strip():
        feedback_section = (
            f"Human Feedback (observations from prior runs):\n"
            f"{state.human_feedback.strip()}"
        )
    else:
        feedback_section = (
            "Human Feedback: None provided. "
            "Generate the best possible prompt from the task description alone."
        )

    # ── Generate prompt ───────────────────────────────────────────────────────
    apg_prompt = APG_META_PROMPT.format(
        task_description=state.task_description,
        output_spec=state.output_spec,
        feedback_section=feedback_section,
    )

    raw = call_mini(
        prompt=apg_prompt,
        temperature=0.2,
        max_tokens=2000,
        call_type="apg_generate",
    )

    generated_prompt = raw.strip()
    log.info(f"[APG] Prompt generated ({len(generated_prompt)} chars)")

    # ── Generate test cases if not provided ───────────────────────────────────
    if not state.input_texts:
        log.info("[APG] No input_texts provided — generating test cases")

        tc_prompt = TEST_CASE_GEN_PROMPT.format(
            task_description=state.task_description,
            output_spec=state.output_spec,
            n=10,
        )

        raw_tc = call_mini(
            prompt=tc_prompt,
            temperature=0.3,
            max_tokens=2000,
            call_type="apg_test_cases",
        )

        try:
            test_cases = parse_json(raw_tc)
            input_texts = [tc["input_text"] for tc in test_cases]
            log.info(f"[APG] Generated {len(input_texts)} test cases")
        except Exception as e:
            log.warning(f"[APG] Test case parsing failed: {e} — using fallback")
            # Fallback: use raw lines as test cases
            input_texts = [
                line.strip()
                for line in raw_tc.split("\n")
                if line.strip() and len(line.strip()) > 10
            ][:10]

        state = state.model_copy(update={"input_texts": input_texts})

    return state.model_copy(update={
        "current_prompt": generated_prompt,
        "iteration":      1,
    })