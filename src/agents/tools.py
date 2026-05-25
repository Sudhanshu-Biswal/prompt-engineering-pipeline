import logging
import json
import chromadb
from chromadb.utils import embedding_functions
from src.utils.state import PipelineState, IterationRecord

log = logging.getLogger(__name__)

# ── ChromaDB setup ────────────────────────────────────────────────────────────
# In-memory for dev/demo — persists to disk in production
# Zero setup — no account, no cloud, works anywhere

_chroma_client: chromadb.Client = None
_collection = None

COLLECTION_NAME = "prompt_history"


def get_collection():
    """
    Lazy singleton ChromaDB collection.
    Uses sentence-transformers for embeddings — no API key needed.
    Falls back to default embeddings if sentence-transformers unavailable.
    """
    global _chroma_client, _collection

    if _collection is not None:
        return _collection

    _chroma_client = chromadb.Client()  # in-memory

    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    except Exception:
        log.warning("[Tools] sentence-transformers unavailable — using default embeddings")
        ef = embedding_functions.DefaultEmbeddingFunction()

    _collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    log.info(f"[Tools] ChromaDB collection ready: {COLLECTION_NAME}")
    return _collection


# ── Tool 1 — search_prompt_history ───────────────────────────────────────────

def index_prompt(
    prompt: str,
    task_name: str,
    task_type: str,
    iteration: int,
    boss_score: float,
    passed: bool,
):
    """
    Index a prompt into ChromaDB after each iteration.
    Called by the evaluator node after scoring.
    Builds the searchable history that APO queries.
    """
    collection = get_collection()

    doc_id = f"{task_name}_iter{iteration}_{int(boss_score * 100)}"

    try:
        collection.upsert(
            ids=[doc_id],
            documents=[prompt],
            metadatas=[{
                "task_name":  task_name,
                "task_type":  task_type,
                "iteration":  iteration,
                "boss_score": boss_score,
                "passed":     str(passed),
            }],
        )
        log.debug(f"[Tools] Indexed prompt: {doc_id} score={boss_score:.3f}")
    except Exception as e:
        log.warning(f"[Tools] Failed to index prompt: {e}")


def search_prompt_history(
    query: str,
    task_type: str,
    top_k: int = 3,
    min_score: float = 3.5,
) -> list[dict]:
    """
    Tool 1 — Search for similar passing prompts.

    APO calls this when it needs reference examples of
    prompts that worked for similar tasks.

    Args:
        query:     failure description or task description
        task_type: filter to same task type only
        top_k:     number of results to return
        min_score: only return prompts above this boss score

    Returns:
        List of {prompt, boss_score, iteration, passed}
    """
    collection = get_collection()

    if collection.count() == 0:
        log.info("[Tools] search_prompt_history: collection empty")
        return []

    try:
        # Filter to same task type and minimum score
        where = {
            "$and": [
                {"task_type": {"$eq": task_type}},
                {"boss_score": {"$gte": min_score}},
            ]
        }

        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"][0]:
            log.info("[Tools] search_prompt_history: no results found")
            return []

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "prompt":     doc,
                "boss_score": meta["boss_score"],
                "iteration":  meta["iteration"],
                "passed":     meta["passed"] == "True",
                "similarity": round(1 - dist, 3),
            })

        log.info(
            f"[Tools] search_prompt_history: "
            f"{len(output)} results for task_type={task_type}"
        )
        return output

    except Exception as e:
        log.warning(f"[Tools] search_prompt_history failed: {e}")
        return []


# ── Tool 2 — get_rubric_failures ─────────────────────────────────────────────

def get_rubric_failures(
    state: PipelineState,
    top_n: int = 5,
) -> dict:
    """
    Tool 2 — Structured rubric failure breakdown.

    APO calls this to get precise data on which rubrics
    failed most frequently and on which test cases.

    Instead of a text summary (which loses structure),
    this returns queryable data APO can reason about.

    Returns:
    {
        j1_failures: [{rubric, avg_score, fail_count, worst_cases}]
        j2_failures: [{rubric, avg_score, fail_count, worst_cases}]
        overall_pass_rate: float
        worst_test_cases: [input_text]
        total_cases: int
    }
    """
    if not state.history:
        return {"error": "No iteration history available"}

    last = state.history[-1]
    test_results = last.test_results

    if not test_results:
        return {"error": "No test results in last iteration"}

    # ── J1 rubric failures ────────────────────────────────────────────────────
    j1_rubric_scores: dict[str, list[float]] = {}
    j1_rubric_cases:  dict[str, list[str]]   = {}

    for tc in test_results:
        if not tc.judge1:
            continue
        for rubric, score in tc.judge1.per_rubric_b.items():
            if rubric not in j1_rubric_scores:
                j1_rubric_scores[rubric] = []
                j1_rubric_cases[rubric]  = []
            j1_rubric_scores[rubric].append(score)
            if score < 3.0:
                j1_rubric_cases[rubric].append(tc.input_text[:100])

    j1_failures = []
    for rubric, scores in j1_rubric_scores.items():
        avg  = round(sum(scores) / len(scores), 3)
        fail = sum(1 for s in scores if s < 3.0)
        if avg < 4.0:   # only surface rubrics that are failing
            j1_failures.append({
                "rubric":      rubric,
                "avg_score":   avg,
                "fail_count":  fail,
                "total_cases": len(scores),
                "worst_cases": j1_rubric_cases[rubric][:3],
            })

    j1_failures.sort(key=lambda x: x["avg_score"])

    # ── J2 rubric failures ────────────────────────────────────────────────────
    j2_rubric_scores: dict[str, list[float]] = {}
    j2_rubric_cases:  dict[str, list[str]]   = {}

    for tc in test_results:
        if not tc.judge2:
            continue
        for rubric, score in tc.judge2.per_rubric_b.items():
            if rubric not in j2_rubric_scores:
                j2_rubric_scores[rubric] = []
                j2_rubric_cases[rubric]  = []
            j2_rubric_scores[rubric].append(score)
            if score < 3.0:
                j2_rubric_cases[rubric].append(tc.input_text[:100])

    j2_failures = []
    for rubric, scores in j2_rubric_scores.items():
        avg  = round(sum(scores) / len(scores), 3)
        fail = sum(1 for s in scores if s < 3.0)
        if avg < 4.0:
            j2_failures.append({
                "rubric":      rubric,
                "avg_score":   avg,
                "fail_count":  fail,
                "total_cases": len(scores),
                "worst_cases": j2_rubric_cases[rubric][:3],
            })

    j2_failures.sort(key=lambda x: x["avg_score"])

    # ── Worst test cases ──────────────────────────────────────────────────────
    cases_with_scores = [
        (tc.input_text[:100], tc.boss.score_b if tc.boss else 0.0)
        for tc in test_results
    ]
    cases_with_scores.sort(key=lambda x: x[1])
    worst_cases = [c[0] for c in cases_with_scores[:top_n]]

    # ── Overall pass rate ─────────────────────────────────────────────────────
    passed = sum(
        1 for tc in test_results
        if tc.boss and tc.boss.score_b >= state.pass_threshold
    )
    total = len(test_results)

    result = {
        "j1_failures":      j1_failures[:top_n],
        "j2_failures":      j2_failures[:top_n],
        "overall_pass_rate": round(passed / max(1, total), 3),
        "worst_test_cases": worst_cases,
        "total_cases":      total,
        "iteration":        last.iteration,
        "avg_boss_score":   last.avg_boss_score,
    }

    log.info(
        f"[Tools] get_rubric_failures: "
        f"{len(j1_failures)} J1 failures, "
        f"{len(j2_failures)} J2 failures, "
        f"pass_rate={result['overall_pass_rate']}"
    )

    return result