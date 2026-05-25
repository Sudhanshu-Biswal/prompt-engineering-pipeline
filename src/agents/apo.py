import logging
import json
from src.utils.llm import call_full, call_mini
from src.utils.state import PipelineState
from src.agents.tools import (
    search_prompt_history,
    get_rubric_failures,
)

log = logging.getLogger(__name__)


# ── Tool definitions for the agent ───────────────────────────────────────────
# These are passed to the LLM so it knows what tools are available
# and when to call them.

TOOLS = [
    {
        "name": "search_prompt_history",
        "description": (
            "Search for similar prompts that scored well on this task type. "
            "Use this when you need reference examples of what worked before. "
            "Returns prompts with their boss scores and iteration numbers. "
            "Call this first when the failure analysis mentions structural issues "
            "that might be solved by looking at passing examples."
        ),
        "parameters": {
            "query": "str — describe what you're looking for (e.g. 'prompt that enforces 8-13 word length constraint')",
            "task_type": "str — the task type to filter by",
            "top_k": "int — number of results (default 3)",
            "min_score": "float — minimum boss score to include (default 3.5)",
        },
    },
    {
        "name": "get_rubric_failures",
        "description": (
            "Get structured breakdown of which rubrics failed most frequently "
            "in the last evaluation. Returns per-rubric scores, fail counts, "
            "and the worst-performing test cases. "
            "Use this when you need precise data on what specifically failed "
            "rather than relying on the text failure summary alone. "
            "Call this when the failure summary is vague or when you want "
            "to prioritise which issues to fix first."
        ),
        "parameters": {
            "top_n": "int — number of worst rubrics/cases to return (default 5)",
        },
    },
]

TOOLS_DESC = json.dumps(TOOLS, indent=2)


# ── Agent system prompt ───────────────────────────────────────────────────────

APO_AGENT_SYSTEM = """You are an expert prompt engineering agent specialising
in iterative prompt optimisation for production LLM systems.

You have access to two tools:

{tools}

## HOW TO USE TOOLS

Respond with tool calls in this exact format when you want to use a tool:

TOOL_CALL: search_prompt_history
ARGS: {{"query": "...", "task_type": "...", "top_k": 3, "min_score": 3.5}}

TOOL_CALL: get_rubric_failures
ARGS: {{"top_n": 5}}

You can call tools multiple times. Call DONE when you have enough
information to write the optimised prompt.

DONE

## WHEN TO CALL TOOLS

Call search_prompt_history when:
- The failure analysis mentions structural issues (format, length, constraints)
- You want to see what worked for similar tasks before
- You're unsure how to fix a specific failure pattern

Call get_rubric_failures when:
- The failure summary is vague or high-level
- You want to know which specific rubrics failed most
- You want to prioritise which issues to fix first

## AFTER TOOLS — WRITE THE OPTIMISED PROMPT

Once you have called DONE, write the optimised prompt.
Rules for the optimised prompt:
1. Keep what is working — don't remove passing constraints
2. Fix what is failing — address every failure in the analysis
3. Be explicit — vague instructions cause inconsistent outputs
4. Use concrete examples only if they prevent a recurring failure
5. No preamble, no labels, no markdown fences
6. The first word must be the first word of the prompt
"""


# ── Agent user prompt ─────────────────────────────────────────────────────────

APO_AGENT_USER = """You are optimising a prompt that failed evaluation.

## TASK
Task name: {task_name}
Task type: {task_type}
Task description: {task_description}
Output spec: {output_spec}

## CURRENT PROMPT (version {iteration})
{current_prompt}

## FAILURE ANALYSIS
{failure_summary}

## SCORE HISTORY
{score_history}

{human_feedback_section}

## YOUR JOB
1. Use tools to gather more information if needed
2. Call DONE when ready
3. Write the optimised prompt

Start now."""


# ── Tool executor ─────────────────────────────────────────────────────────────

def _execute_tool(
    tool_name: str,
    args: dict,
    state: PipelineState,
) -> str:
    """Execute a tool call and return result as string."""

    if tool_name == "search_prompt_history":
        results = search_prompt_history(
            query=args.get("query", state.task_description),
            task_type=args.get("task_type", state.task_name),
            top_k=args.get("top_k", 3),
            min_score=args.get("min_score", 3.5),
        )
        if not results:
            return "No similar passing prompts found in history yet."
        lines = []
        for r in results:
            lines.append(
                f"[Score: {r['boss_score']:.2f} | "
                f"Iter: {r['iteration']} | "
                f"Similarity: {r['similarity']}]\n"
                f"{r['prompt'][:300]}..."
            )
        return "\n\n---\n\n".join(lines)

    if tool_name == "get_rubric_failures":
        result = get_rubric_failures(
            state=state,
            top_n=args.get("top_n", 5),
        )
        if "error" in result:
            return f"Error: {result['error']}"
        return json.dumps(result, indent=2)

    return f"Unknown tool: {tool_name}"


def _parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    """
    Parse tool calls from agent response.
    Returns list of (tool_name, args) tuples.
    """
    import re
    calls = []

    pattern = r"TOOL_CALL:\s*(\w+)\s*\nARGS:\s*(\{.*?\})"
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        tool_name = match.group(1).strip()
        try:
            args = json.loads(match.group(2))
        except json.JSONDecodeError:
            args = {}
        calls.append((tool_name, args))

    return calls


def _extract_final_prompt(text: str) -> str:
    """
    Extract the final optimised prompt after DONE marker.
    Falls back to full text if no DONE marker found.
    """
    if "DONE" in text:
        parts = text.split("DONE", 1)
        if len(parts) > 1:
            prompt = parts[1].strip()
            if len(prompt) > 50:
                return prompt

    # Fallback — use full response if no clear DONE marker
    # Remove any TOOL_CALL sections
    import re
    cleaned = re.sub(
        r"TOOL_CALL:.*?(?=TOOL_CALL:|DONE|$)",
        "", text, flags=re.DOTALL
    ).strip()

    return cleaned if len(cleaned) > 50 else text.strip()


# ── Main APO agent ────────────────────────────────────────────────────────────

def run_apo(state: PipelineState) -> PipelineState:
    """
    APO node — Tool-using Auto Prompt Optimizer agent.

    Unlike a simple LLM call, this agent:
    1. Receives failure analysis and score history
    2. Decides which tools to call based on the failures
    3. Calls search_prompt_history to find passing examples
    4. Calls get_rubric_failures to get structured failure data
    5. Uses retrieved context to write a better prompt

    The agent loop runs until it calls DONE or hits max_tool_calls.
    This is what makes APO genuinely agentic — it reasons about
    what information it needs before optimising.

    Tool-using agent pattern:
    Observe (failure analysis) → Orient (tool calls) →
    Decide (DONE) → Act (write optimised prompt)
    """
    log.info(
        f"[APO Agent] Optimising prompt "
        f"iter={state.iteration} → {state.iteration + 1}"
    )

    # ── Build context ─────────────────────────────────────────────────────────
    failure_summary = "No failure summary available."
    if state.history:
        last = state.history[-1]
        if last.failure_summary:
            failure_summary = last.failure_summary

    human_feedback_section = (
        f"## HUMAN FEEDBACK\n{state.human_feedback.strip()}\n"
        f"This feedback takes priority over automated failure analysis."
        if state.human_feedback and state.human_feedback.strip()
        else "## HUMAN FEEDBACK\nNone provided."
    )

    score_lines = [
        f"  v{r.iteration}: {r.avg_boss_score:.3f}/5 "
        f"({'PASSED' if r.passed else 'FAILED'})"
        for r in state.history
    ]
    score_history = "\n".join(score_lines) if score_lines else "  No prior iterations."

    # ── Agent loop ────────────────────────────────────────────────────────────
    system_prompt = APO_AGENT_SYSTEM.format(tools=TOOLS_DESC)

    user_prompt = APO_AGENT_USER.format(
        task_name=state.task_name,
        task_type=state.task_name,
        task_description=state.task_description,
        output_spec=state.output_spec,
        iteration=state.iteration,
        current_prompt=state.current_prompt,
        failure_summary=failure_summary,
        score_history=score_history,
        human_feedback_section=human_feedback_section,
    )

    # Conversation history for the agent loop
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    max_tool_calls = 4   # prevent infinite loops
    tool_calls_made = 0
    tool_results = []

    log.info("[APO Agent] Starting agent loop...")

    while tool_calls_made < max_tool_calls:
        # Call the agent
        from openai import OpenAI
        import os
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model=os.getenv("APO_MODEL", "gpt-4o"),
            temperature=0.3,
            max_tokens=2000,
            messages=messages,
        )

        agent_text = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": agent_text})

        # Parse tool calls
        calls = _parse_tool_calls(agent_text)

        if not calls:
            # No tool calls — agent is done
            log.info(f"[APO Agent] No tool calls — agent complete")
            break

        # Check for DONE marker
        if "DONE" in agent_text and not calls:
            log.info("[APO Agent] DONE marker found")
            break

        # Execute tool calls
        tool_results_text = []
        for tool_name, args in calls:
            log.info(f"[APO Agent] Calling tool: {tool_name} args={args}")
            result = _execute_tool(tool_name, args, state)
            tool_calls_made += 1
            tool_results.append({
                "tool": tool_name,
                "args": args,
                "result_preview": result[:200],
            })
            tool_results_text.append(
                f"Tool: {tool_name}\nResult:\n{result}"
            )
            log.info(
                f"[APO Agent] {tool_name} returned "
                f"{len(result)} chars"
            )

        # Add tool results to conversation
        if tool_results_text:
            messages.append({
                "role": "user",
                "content": (
                    "Tool results:\n\n" +
                    "\n\n---\n\n".join(tool_results_text) +
                    "\n\nContinue. Call more tools if needed, or call DONE and write the optimised prompt."
                ),
            })

        # If DONE is in the response with tool calls, we're done after executing
        if "DONE" in agent_text:
            break

    # ── Extract final prompt ──────────────────────────────────────────────────
    # Get the last assistant message
    last_assistant = next(
        (m["content"] for m in reversed(messages) if m["role"] == "assistant"),
        ""
    )

    optimised_prompt = _extract_final_prompt(last_assistant)

    if len(optimised_prompt) < 50:
        log.warning(
            "[APO Agent] Extracted prompt too short — "
            "falling back to simple APO"
        )
        optimised_prompt = _simple_apo_fallback(state, failure_summary)

    new_iteration = state.iteration + 1

    log.info(
        f"[APO Agent] Complete — "
        f"tool_calls={tool_calls_made} "
        f"prompt_len={len(optimised_prompt)} "
        f"iter={new_iteration}"
    )

    if tool_results:
        log.info(f"[APO Agent] Tools used: {[t['tool'] for t in tool_results]}")

    return state.model_copy(update={
        "current_prompt": optimised_prompt,
        "iteration":      new_iteration,
        "human_feedback": None,
    })


# ── Simple fallback ───────────────────────────────────────────────────────────

def _simple_apo_fallback(
    state: PipelineState,
    failure_summary: str,
) -> str:
    """
    Fallback to simple single-call APO if agent loop fails.
    Ensures APO always produces a valid prompt.
    """
    log.info("[APO Agent] Using simple fallback")

    from src.utils.llm import call_full

    prompt = f"""You are an expert prompt engineer.

Improve this prompt that failed evaluation.

Current prompt:
{state.current_prompt}

Failure analysis:
{failure_summary}

Task description:
{state.task_description}

Write the improved prompt only. No preamble."""

    return call_full(
        prompt=prompt,
        temperature=0.3,
        max_tokens=2000,
        call_type="apo_fallback",
    ).strip()