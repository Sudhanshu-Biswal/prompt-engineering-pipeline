import os
import json
import re
import time
import logging
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ── Models ────────────────────────────────────────────────────────────────────
MINI  = os.getenv("JUDGE_MODEL", "gpt-4o-mini")  # APG, test gen, rubric gen, PTF gen, J1, J2
FULL  = os.getenv("APO_MODEL", "gpt-4o")       # APO, Boss — needs deep reasoning

# ── Client ────────────────────────────────────────────────────────────────────
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Token tracking ────────────────────────────────────────────────────────────
_token_log: list[dict] = []

PRICING = {
    MINI: {"input": 0.00015,  "output": 0.0006},
    FULL: {"input": 0.0025,   "output": 0.01},
}


def get_token_log() -> list[dict]:
    return _token_log


def get_total_cost() -> float:
    return round(sum(r["cost_usd"] for r in _token_log), 6)


def clear_token_log():
    _token_log.clear()


def _compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    if model not in PRICING:
        return 0.0
    p = PRICING[model]
    return round(
        (prompt_tokens  / 1_000_000 * p["input"] * 1000) +
        (completion_tokens / 1_000_000 * p["output"] * 1000),
        6,
    )


# ── Core call ─────────────────────────────────────────────────────────────────
def call_llm(
    prompt: str,
    model: str = MINI,
    temperature: float = 0.2,
    max_tokens: int = 1500,
    call_type: str = "unknown",
    max_retries: int = 3,
) -> str:
    """
    Single synchronous LLM call with retry logic and token tracking.
    Returns the content string.
    """
    for attempt in range(max_retries):
        try:
            response = _client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.choices[0].message.content or ""
            usage   = response.usage

            # Track tokens
            pt   = usage.prompt_tokens
            ct   = usage.completion_tokens
            cost = _compute_cost(model, pt, ct)

            _token_log.append({
                "call_type":         call_type,
                "model":             model,
                "prompt_tokens":     pt,
                "completion_tokens": ct,
                "total_tokens":      pt + ct,
                "cost_usd":          cost,
            })

            log.debug(
                f"[{call_type}] model={model} "
                f"pt={pt} ct={ct} cost=${cost:.6f}"
            )

            return content

        except Exception as e:
            wait = 2 ** attempt
            log.warning(
                f"[{call_type}] attempt {attempt+1}/{max_retries} "
                f"failed: {e} — retrying in {wait}s"
            )
            time.sleep(wait)

    raise RuntimeError(
        f"[{call_type}] LLM call failed after {max_retries} attempts"
    )


# ── JSON parser ───────────────────────────────────────────────────────────────
def parse_json(text: str) -> dict | list:
    """
    Robust JSON parser — strips markdown fences, handles common
    LLM formatting issues. Raises ValueError if unparseable.
    """
    text = text.strip()
    text = text.replace("\\'", "'")

    # Strip ```json ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$",          "", text).strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract first [...] or {...} block
    for pattern in (r"(\[.*\])", r"(\{.*\})"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    # Numbered list fallback — return as list of strings
    numbered = re.findall(r"^\d+\.\s*(.+)$", text, re.MULTILINE)
    if len(numbered) >= 2:
        log.info(f"[parse_json] Parsed {len(numbered)} items from numbered list")
        return numbered

    raise ValueError(
        f"Could not parse JSON from LLM response. "
        f"First 300 chars: {repr(text[:300])}"
    )


# ── Convenience wrappers ──────────────────────────────────────────────────────
def call_mini(
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 1500,
    call_type: str = "unknown",
) -> str:
    """gpt-4o-mini — APG, rubric gen, PTF gen, J1, J2, test case gen."""
    return call_llm(
        prompt,
        model=MINI,
        temperature=temperature,
        max_tokens=max_tokens,
        call_type=call_type,
    )


def call_full(
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 1500,
    call_type: str = "unknown",
) -> str:
    """gpt-4o — APO and Boss only."""
    return call_llm(
        prompt,
        model=FULL,
        temperature=temperature,
        max_tokens=max_tokens,
        call_type=call_type,
    )


# ── Token summary printer ─────────────────────────────────────────────────────
def print_token_summary():
    if not _token_log:
        print("No token usage recorded.")
        return

    model_totals: dict[str, dict] = {}
    for entry in _token_log:
        m = entry["model"]
        if m not in model_totals:
            model_totals[m] = {
                "prompt": 0, "completion": 0,
                "total": 0,  "cost": 0.0,
            }
        model_totals[m]["prompt"]     += entry["prompt_tokens"]
        model_totals[m]["completion"] += entry["completion_tokens"]
        model_totals[m]["total"]      += entry["total_tokens"]
        model_totals[m]["cost"]       += entry["cost_usd"]

    print("\n" + "=" * 55)
    print("  TOKEN USAGE SUMMARY")
    print("=" * 55)
    print(f"  {'Model':<20} {'Tokens':>10} {'Cost (USD)':>12}")
    print(f"  {'-'*44}")
    for model, t in model_totals.items():
        print(
            f"  {model:<20} {t['total']:>10,} "
            f"${t['cost']:>11.6f}"
        )
    print(f"  {'-'*44}")
    print(f"  {'TOTAL':<20} "
          f"{sum(t['total'] for t in model_totals.values()):>10,} "
          f"${get_total_cost():>11.6f}")
    print("=" * 55 + "\n")