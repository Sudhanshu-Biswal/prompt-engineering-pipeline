import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from src.utils.state import PipelineState, load_config
from src.utils.llm import print_token_summary, get_total_cost
from src.pipeline import run_pipeline


def print_final_report(state: PipelineState):
    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Task:              {state.task_name}")
    print(f"  Iterations used:   {state.iteration} / {state.max_iterations}")
    print(f"  Final score:       {state.final_score:.3f}/5" if state.final_score else "  Final score:       N/A")
    print(f"  Migration:         {state.migration_decision}")
    print(f"  Reason:            {state.migration_reason}")

    if state.hypothesis:
        print(f"\n  Statistical test:  {state.hypothesis.test_used}")
        print(f"  p-value:           {state.hypothesis.p_value:.6f}")
        print(f"  Mean difference:   {state.hypothesis.mean_difference:+.4f}")
        print(f"  Verdict:           {state.hypothesis.model_state}")

    print("\n  SCORE PROGRESSION:")
    for record in state.history:
        bar = "█" * int(record.avg_boss_score * 4)
        print(
            f"    v{record.iteration}  {record.avg_boss_score:.3f}/5  "
            f"{bar:<20}  "
            f"{'✓ PASSED' if record.passed else '✗ failed'}"
        )

    print("\n" + "=" * 65)
    print("  FINAL PROMPT:")
    print("=" * 65)
    print(state.final_prompt or state.current_prompt)
    print("=" * 65)


def save_results(state: PipelineState, output_dir: str):
    """Save full results to JSON for analysis and the README benchmark table."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"{state.task_name}_{timestamp}.json"

    output = {
        "task_name":          state.task_name,
        "migration_decision": state.migration_decision,
        "migration_reason":   state.migration_reason,
        "final_score":        state.final_score,
        "iterations_used":    state.iteration,
        "hypothesis": {
            "test_used":       state.hypothesis.test_used,
            "p_value":         state.hypothesis.p_value,
            "mean_difference": state.hypothesis.mean_difference,
            "model_state":     state.hypothesis.model_state,
        } if state.hypothesis else None,
        "score_progression": [
            {
                "iteration":     r.iteration,
                "avg_boss_score": r.avg_boss_score,
                "passed":        r.passed,
                "prompt":        r.prompt,
            }
            for r in state.history
        ],
        "final_prompt": state.final_prompt,
        "total_cost_usd": get_total_cost(),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    log.info(f"Results saved → {path}")
    return str(path)


def main():
    parser = argparse.ArgumentParser(
        description="Automated Prompt Engineering Pipeline — APG + APO"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config JSON (e.g. configs/improve_work_history.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save results (default: results/)",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip human approval prompt — auto-approve",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        return

    if args.non_interactive:
        os.environ["NON_INTERACTIVE"] = "1"

    # Load config into PipelineState
    log.info(f"Loading config: {args.config}")
    state = load_config(args.config)
    log.info(f"Task: {state.task_name}")
    log.info(f"Max iterations: {state.max_iterations}")
    log.info(f"Pass threshold: {state.pass_threshold}/5")
    log.info(f"Num runs: {state.num_runs}")

    # Run
    final_state = run_pipeline(state)

    # Report
    print_final_report(final_state)
    print_token_summary()

    # Save
    output_path = save_results(final_state, args.output_dir)
    print(f"\n  Full results → {output_path}\n")


if __name__ == "__main__":
    main()