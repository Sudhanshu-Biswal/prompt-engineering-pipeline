import logging
from typing import Literal
from langgraph.graph import StateGraph, END
from src.utils.state import PipelineState, IterationRecord
from src.agents.apg import run_apg
from src.agents.apo import run_apo
from src.agents.test_runner import generate_source_outputs, run_test_cases
from src.evaluation.judge1 import get_j1_rubrics, run_judge1
from src.evaluation.judge2 import get_j2_rubrics, run_judge2
from src.evaluation.boss import (
    run_boss, compute_avg_boss_score, generate_failure_summary
)
from src.evaluation.hypothesis import run_hypothesis

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# NODE FUNCTIONS
# Each node takes PipelineState, does one job, returns updated PipelineState
# ─────────────────────────────────────────────────────────────────────────────

def node_apg(state: PipelineState) -> PipelineState:
    """
    Node 1 — Auto Prompt Generator.
    Generates prompt v1 and test cases (if not provided).
    Always the entry point.
    """
    log.info("━━━ NODE: APG ━━━")
    return run_apg(state)


def node_baseline(state: PipelineState) -> PipelineState:
    """
    Node 2 — Baseline generator.
    Generates source outputs (output_a) using a plain unoptimised prompt.
    Runs ONCE — skipped on subsequent iterations because
    source_outputs is already populated in state.
    """
    log.info("━━━ NODE: Baseline ━━━")
    return generate_source_outputs(state)


def node_test_runner(state: PipelineState) -> PipelineState:
    """
    Node 3 — Test runner.
    Runs current_prompt against all input_texts.
    Produces output_b for each test case.
    Loops back here after every APO iteration.
    """
    log.info(f"━━━ NODE: TestRunner (iter={state.iteration}) ━━━")
    test_results = run_test_cases(state)

    # Store results temporarily in state for evaluator
    return state.model_copy(update={"_test_results": test_results})


def node_evaluator(state: PipelineState) -> PipelineState:
    """
    Node 4 — Full evaluation.
    Runs J1 + J2 + Boss across all test cases × num_runs.
    Computes average Boss score.
    Generates failure summary if score below threshold.
    Records full IterationRecord in history.
    """
    log.info(f"━━━ NODE: Evaluator (iter={state.iteration}) ━━━")

    # Get rubrics — cached after first call
    j1_rubrics = get_j1_rubrics(state)
    j2_rubrics = get_j2_rubrics(state)

    # Update rubric cache in state
    state = state.model_copy(update={
        "j1_rubrics": j1_rubrics,
        "j2_rubrics": j2_rubrics,
    })

    # Retrieve test results from state
    test_results = getattr(state, "_test_results", [])

    # Run judges
    test_results = run_judge1(test_results, j1_rubrics, state)
    test_results = run_judge2(test_results, j2_rubrics, state)
    test_results = run_boss(test_results, j1_rubrics, j2_rubrics, state)

    # Compute pass/fail
    avg_boss = compute_avg_boss_score(test_results)
    passed   = avg_boss >= state.pass_threshold

    log.info(
        f"[Evaluator] iter={state.iteration} "
        f"avg_boss={avg_boss:.3f} "
        f"threshold={state.pass_threshold} "
        f"passed={passed}"
    )

    # Generate failure summary for APO (only if failed)
    failure_summary = None
    if not passed:
        failure_summary = generate_failure_summary(
            test_results, j1_rubrics, j2_rubrics, state
        )

    # Record this iteration
    record = IterationRecord(
        iteration=state.iteration,
        prompt=state.current_prompt,
        test_results=test_results,
        avg_boss_score=avg_boss,
        passed=passed,
        failure_summary=failure_summary,
    )

    updated_history = list(state.history) + [record]

    return state.model_copy(update={
        "history":      updated_history,
        "passed":       passed,
        "final_prompt": state.current_prompt if passed else state.final_prompt,
        "final_score":  avg_boss if passed else state.final_score,
        "_test_results": None,   # clear from state
    })
    # ── Index prompt for APO tool search ─────────────────────────────────────
    # After every iteration, index the prompt and its score
    # so future APO calls can search for passing examples
    from src.agents.tools import index_prompt
    index_prompt(
        prompt=state.current_prompt,
        task_name=state.task_name,
        task_type=state.task_name,
        iteration=state.iteration,
        boss_score=avg_boss,
        passed=passed,
    )

    return state.model_copy(update={
        "history":       updated_history,
        "passed":        passed,
        "final_prompt":  state.current_prompt if passed else state.final_prompt,
        "final_score":   avg_boss if passed else state.final_score,
        "_test_results": None,
    })


def node_apo(state: PipelineState) -> PipelineState:
    """
    Node 5 — Auto Prompt Optimizer.
    Analyses failures and generates improved prompt.
    Only reached when evaluation fails and iterations remain.
    """
    log.info(f"━━━ NODE: APO (iter={state.iteration}) ━━━")
    return run_apo(state)


def node_human_approval(state: PipelineState) -> PipelineState:
    """
    Node 6 — Human approval checkpoint.
    In automated mode: auto-approves and moves to hypothesis.
    In interactive mode: prompts for human input.

    In production at BOLD, human approval happened via a review UI.
    In this public version we expose it as an interrupt point.
    """
    log.info("━━━ NODE: HumanApproval ━━━")

    print("\n" + "=" * 60)
    print("  HUMAN APPROVAL CHECKPOINT")
    print("=" * 60)
    print(f"  Final prompt score: {state.final_score:.3f}/5")
    print(f"  Iterations used:    {state.iteration}")
    print(f"\n  FINAL PROMPT:\n")
    print(state.final_prompt)
    print("\n" + "=" * 60)
    print("  Options:")
    print("  [1] Approve — proceed to statistical test")
    print("  [2] Reject  — stop pipeline")
    print("  [3] Provide feedback — run one more APO iteration")
    print("=" * 60)

    try:
        choice = input("\n  Your choice (1/2/3) [default=1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        # Non-interactive mode — auto approve
        choice = "1"

    if choice == "2":
        log.info("[HumanApproval] Rejected by human")
        return state.model_copy(update={
            "stop":              True,
            "migration_decision": "REJECTED",
            "migration_reason":  "Rejected by human reviewer.",
        })

    elif choice == "3":
        feedback = input(
            "  Enter feedback for APO: "
        ).strip()
        log.info(f"[HumanApproval] Human feedback: {feedback[:80]}")
        return state.model_copy(update={
            "human_feedback": feedback,
            "passed":         False,   # re-trigger APO
            "awaiting_human": False,
        })

    else:
        # Approve
        log.info("[HumanApproval] Approved by human")
        return state.model_copy(update={"awaiting_human": False})


def node_hypothesis(state: PipelineState) -> PipelineState:
    """
    Node 7 — Statistical hypothesis testing.
    Wilcoxon signed-rank or permutation test.
    Produces migration decision: APPROVED / REJECTED / NEEDS_REVIEW.
    """
    log.info("━━━ NODE: Hypothesis ━━━")
    return run_hypothesis(state)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER — the brain of the agent
# Conditional edges branch based on state
# ─────────────────────────────────────────────────────────────────────────────

def router(state: PipelineState) -> Literal[
    "apo", "human_approval", "hypothesis", "__end__"
]:
    """
    Router node — decides next step after evaluation.

    Decision logic:
    1. If stop=True (human rejected) → end
    2. If passed=True → human approval
    3. If iteration >= max_iterations → hypothesis (best effort)
    4. Otherwise → APO (optimise and retry)

    This conditional edge is what makes the system agentic.
    Without it: a pipeline. With it: an agent that loops and decides.
    """
    if state.stop:
        log.info("[Router] stop=True → END")
        return "__end__"

    if state.passed:
        log.info("[Router] passed=True → human_approval")
        return "human_approval"

    if state.iteration >= state.max_iterations:
        log.info(
            f"[Router] max_iterations={state.max_iterations} reached "
            f"→ hypothesis (best effort)"
        )
        # Use best scoring prompt from history
        if state.history:
            best = max(state.history, key=lambda r: r.avg_boss_score)
            log.info(
                f"[Router] Best iteration: {best.iteration} "
                f"score={best.avg_boss_score:.3f}"
            )
        return "hypothesis"

    log.info(
        f"[Router] iter={state.iteration} "
        f"score={state.final_score} → APO"
    )
    return "apo"


def human_approval_router(state: PipelineState) -> Literal[
    "apo", "hypothesis", "__end__"
]:
    """
    Router after human approval node.

    - stop=True (rejected) → end
    - passed=False (feedback given) → APO one more time
    - Otherwise → hypothesis
    """
    if state.stop:
        return "__end__"
    if not state.passed:
        return "apo"
    return "hypothesis"


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    """
    Assembles the LangGraph agent.

    Graph structure:
    APG → Baseline → TestRunner → Evaluator → Router
                         ↑                        │
                         │              ┌─────────┼──────────┐
                         │           APO      HumanApproval  Hypothesis
                         │              │         │               │
                         └──────────────┘    ┌────┤              END
                                             APO  Hypothesis
    """
    # PipelineState must be dict-serialisable for LangGraph
    # We use model_dump / model_validate for this
    builder = StateGraph(dict)

    # ── Add nodes ─────────────────────────────────────────────────────────────
    builder.add_node("apg",            _wrap(node_apg))
    builder.add_node("baseline",       _wrap(node_baseline))
    builder.add_node("test_runner",    _wrap(node_test_runner))
    builder.add_node("evaluator",      _wrap(node_evaluator))
    builder.add_node("apo",            _wrap(node_apo))
    builder.add_node("human_approval", _wrap(node_human_approval))
    builder.add_node("hypothesis",     _wrap(node_hypothesis))

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.set_entry_point("apg")

    # ── Normal edges (always go A → B) ───────────────────────────────────────
    builder.add_edge("apg",         "baseline")
    builder.add_edge("baseline",    "test_runner")
    builder.add_edge("test_runner", "evaluator")
    builder.add_edge("hypothesis",  END)

    # ── Conditional edges (router decides) ───────────────────────────────────
    builder.add_conditional_edges(
        "evaluator",
        _wrap_router(router),
        {
            "apo":            "apo",
            "human_approval": "human_approval",
            "hypothesis":     "hypothesis",
            "__end__":        END,
        }
    )

    builder.add_conditional_edges(
        "human_approval",
        _wrap_router(human_approval_router),
        {
            "apo":        "apo",
            "hypothesis": "hypothesis",
            "__end__":    END,
        }
    )

    # APO always loops back to test_runner
    builder.add_edge("apo", "test_runner")

    return builder.compile()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — LangGraph uses dicts, we use Pydantic models
# These wrappers handle the conversion transparently
# ─────────────────────────────────────────────────────────────────────────────

def _wrap(node_fn):
    """
    Wraps a node function that takes/returns PipelineState
    so LangGraph can pass/receive plain dicts.
    """
    def wrapped(state_dict: dict) -> dict:
        state  = PipelineState.model_validate(state_dict)
        result = node_fn(state)
        return result.model_dump()
    wrapped.__name__ = node_fn.__name__
    return wrapped


def _wrap_router(router_fn):
    """
    Wraps a router function that takes PipelineState
    so LangGraph can pass a plain dict.
    """
    def wrapped(state_dict: dict) -> str:
        state = PipelineState.model_validate(state_dict)
        return router_fn(state)
    wrapped.__name__ = router_fn.__name__
    return wrapped


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(state: PipelineState) -> PipelineState:
    """
    Run the full pipeline from a PipelineState.
    Returns the final PipelineState with all results.
    """
    pipeline = build_pipeline()
    result   = pipeline.invoke(state.model_dump())
    return PipelineState.model_validate(result)