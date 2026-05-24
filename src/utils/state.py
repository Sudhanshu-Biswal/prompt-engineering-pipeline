from typing import Optional
from pydantic import BaseModel, Field
import json
from pathlib import Path


# ── Judge result for a single run ────────────────────────────────────────────
class SingleRunResult(BaseModel):
    score_a: float
    score_b: float
    reasoning_a: str
    reasoning_b: str
    raw_a: dict[str, float] = Field(default_factory=dict)  # rubric -> score
    raw_b: dict[str, float] = Field(default_factory=dict)


# ── Averaged result across N runs ────────────────────────────────────────────
class AveragedResult(BaseModel):
    score_a: float
    score_b: float
    reasoning_a: str
    reasoning_b: str
    per_rubric_a: dict[str, float] = Field(default_factory=dict)
    per_rubric_b: dict[str, float] = Field(default_factory=dict)
    all_scores_a: list[float] = Field(default_factory=list)
    all_scores_b: list[float] = Field(default_factory=list)
    runs: list[SingleRunResult] = Field(default_factory=list)


# ── One test case + its evaluation ───────────────────────────────────────────
class TestCaseResult(BaseModel):
    index: int
    input_text: str
    output_a: str                        # source model output
    output_b: str                        # target model (current prompt) output
    judge1: Optional[AveragedResult] = None
    judge2: Optional[AveragedResult] = None
    boss: Optional[AveragedResult] = None
    error: Optional[str] = None


# ── One full iteration record ─────────────────────────────────────────────────
class IterationRecord(BaseModel):
    iteration: int
    prompt: str
    test_results: list[TestCaseResult] = Field(default_factory=list)
    avg_boss_score: float = 0.0
    passed: bool = False
    failure_summary: Optional[str] = None  # fed into APO


# ── Hypothesis test result ────────────────────────────────────────────────────
class HypothesisResult(BaseModel):
    test_used: str                   # "wilcoxon" or "permutation"
    statistic: Optional[float] = None
    p_value: float = 0.0
    mean_difference: float = 0.0
    model_state: str                 # "better" | "worse" | "no_significant_difference"


# ── Main pipeline state ───────────────────────────────────────────────────────
class PipelineState(BaseModel):

    # ── Inputs ────────────────────────────────────────────────────────────────
    task_name: str                           # e.g. "resume_bullets"
    task_description: str                    # what the LLM should do
    output_spec: str                         # plain text or JSON schema string
    source_outputs: list[str] = Field(default_factory=list)
    # source_outputs: GPT-4o-mini baseline outputs — used as output_a in judges
    input_texts: list[str] = Field(default_factory=list)
    # input_texts: raw inputs the prompt will process
    human_feedback: Optional[str] = None    # optional, passed to APO
    num_runs: int = 3                        # judge runs per iteration
    pass_threshold: float = 4.0            # boss score to pass
    output_is_json: bool = True   # True = JSON output, False = plain text

    # ── Generated once ────────────────────────────────────────────────────────
    j1_rubrics: list[str] = Field(default_factory=list)
    j2_rubrics: list[str] = Field(default_factory=list)

    # ── Prompt versioning ─────────────────────────────────────────────────────
    current_prompt: str = ""
    iteration: int = 0
    max_iterations: int = 3
    history: list[IterationRecord] = Field(default_factory=list)

    # ── Routing flags ─────────────────────────────────────────────────────────
    passed: bool = False
    stop: bool = False
    awaiting_human: bool = False

    # ── Final outputs ─────────────────────────────────────────────────────────
    final_prompt: Optional[str] = None
    final_score: Optional[float] = None
    hypothesis: Optional[HypothesisResult] = None
    migration_decision: Optional[str] = None   # "APPROVED" | "REJECTED" | "NEEDS_REVIEW"
    migration_reason: Optional[str] = None

def load_config(path: str) -> PipelineState:
    """
    Load a pipeline run from a JSON config file.
    This is the primary way to start a pipeline run —
    mirrors the config-driven design of the production system.
    """
    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    return PipelineState(
        task_name=d["task_name"],
        task_description=d["task_description"],
        output_spec=d["output_spec"],
        output_is_json=d.get("output_is_json", True),
        num_runs=d.get("num_runs", 3),
        pass_threshold=d.get("pass_threshold", 4.0),
        max_iterations=d.get("max_iterations", 3),
        human_feedback=d.get("human_feedback"),
        input_texts=d.get("input_texts", []),
        source_outputs=d.get("source_outputs", []),
    )