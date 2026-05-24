import logging
import numpy as np
from scipy import stats
from src.utils.state import PipelineState, HypothesisResult, IterationRecord

log = logging.getLogger(__name__)


def _get_score_pairs(
    history: list[IterationRecord],
) -> tuple[list[float], list[float]]:
    """
    Extract score_a (baseline) and score_b (optimised)
    from the final passing iteration's test results.
    Uses all runs for statistical power — mirrors production pipeline.
    """
    if not history:
        return [], []

    # Use final iteration
    final = history[-1]

    scores_a, scores_b = [], []
    for tc in final.test_results:
        if tc.boss:
            scores_a.extend(tc.boss.all_scores_a)
            scores_b.extend(tc.boss.all_scores_b)

    return scores_a, scores_b


def _check_symmetry(differences: list[float]) -> bool:
    """
    Check if differences are symmetric around zero.
    Wilcoxon requires symmetry — if not symmetric, use permutation test.
    Mirrors the production hypothesis testing logic exactly.
    """
    if len(differences) < 3:
        return False
    skewness = float(stats.skew(differences))
    is_symmetric = abs(skewness) < 1.0
    log.info(f"[Hypothesis] Skewness={skewness:.3f} symmetric={is_symmetric}")
    return is_symmetric


def _wilcoxon_test(
    scores_a: list[float],
    scores_b: list[float],
) -> HypothesisResult:
    """Wilcoxon signed-rank test — used when differences are symmetric."""
    statistic, p_value = stats.wilcoxon(scores_b, scores_a)
    mean_diff = float(np.mean(np.array(scores_b) - np.array(scores_a)))

    if p_value < 0.05 and mean_diff > 0:
        model_state = "better"
    elif p_value < 0.05 and mean_diff <= 0:
        model_state = "worse"
    else:
        model_state = "no_significant_difference"

    log.info(
        f"[Hypothesis] Wilcoxon stat={statistic:.3f} "
        f"p={p_value:.6f} mean_diff={mean_diff:.4f} → {model_state}"
    )

    return HypothesisResult(
        test_used="wilcoxon",
        statistic=float(statistic),
        p_value=float(p_value),
        mean_difference=mean_diff,
        model_state=model_state,
    )


def _permutation_test(
    scores_a: list[float],
    scores_b: list[float],
    n_permutations: int = 10_000,
) -> HypothesisResult:
    """Permutation test — used when differences are not symmetric."""
    a = np.array(scores_a)
    b = np.array(scores_b)
    observed_diff = float(np.mean(b) - np.mean(a))
    combined = np.concatenate([b, a])

    perm_diffs = []
    rng = np.random.default_rng(seed=42)
    for _ in range(n_permutations):
        perm = rng.permutation(combined)
        perm_diffs.append(
            np.mean(perm[:len(b)]) - np.mean(perm[len(b):])
        )

    p_value = float(
        np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    )

    if p_value < 0.05 and observed_diff > 0:
        model_state = "better"
    elif p_value < 0.05 and observed_diff <= 0:
        model_state = "worse"
    else:
        model_state = "no_significant_difference"

    log.info(
        f"[Hypothesis] Permutation p={p_value:.6f} "
        f"mean_diff={observed_diff:.4f} → {model_state}"
    )

    return HypothesisResult(
        test_used="permutation",
        statistic=None,
        p_value=p_value,
        mean_difference=observed_diff,
        model_state=model_state,
    )


def run_hypothesis(state: PipelineState) -> PipelineState:
    """
    Hypothesis testing node.

    Automatically selects Wilcoxon or permutation test based on
    symmetry of score differences — exactly as the production pipeline does.

    Produces:
    - state.hypothesis (HypothesisResult)
    - state.migration_decision: APPROVED | REJECTED | NEEDS_REVIEW
    - state.migration_reason
    """
    log.info("[Hypothesis] Running statistical test on final iteration scores")

    scores_a, scores_b = _get_score_pairs(state.history)

    if len(scores_a) < 3 or len(scores_b) < 3:
        log.warning("[Hypothesis] Insufficient data for statistical test")
        return state.model_copy(update={
            "migration_decision": "NEEDS_REVIEW",
            "migration_reason": (
                "Insufficient test cases for statistical significance. "
                "Run with more input_texts for a reliable verdict."
            ),
        })

    # Select test based on symmetry
    differences = [b - a for a, b in zip(scores_a, scores_b)]

    if _check_symmetry(differences):
        result = _wilcoxon_test(scores_a, scores_b)
    else:
        result = _permutation_test(scores_a, scores_b)

    # Migration decision
    if result.model_state == "better":
        decision = "APPROVED"
        reason = (
            f"Optimised prompt is statistically BETTER than baseline "
            f"(p={result.p_value:.4f}, mean_diff=+{result.mean_difference:.3f}). "
            f"Safe to deploy."
        )
    elif result.model_state == "worse":
        decision = "REJECTED"
        reason = (
            f"Optimised prompt is statistically WORSE than baseline "
            f"(p={result.p_value:.4f}, mean_diff={result.mean_difference:.3f}). "
            f"Do NOT deploy."
        )
    else:
        decision = "NEEDS_REVIEW"
        reason = (
            f"No statistically significant difference detected "
            f"(p={result.p_value:.4f}, mean_diff={result.mean_difference:.3f}). "
            f"Human review recommended before deployment."
        )

    log.info(f"[Hypothesis] Decision: {decision} — {reason}")

    return state.model_copy(update={
        "hypothesis":          result,
        "migration_decision":  decision,
        "migration_reason":    reason,
    })