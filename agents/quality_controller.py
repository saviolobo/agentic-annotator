"""Quality Controller Agent — consistency check and drift detection.

Model: llama-3.1-8b-instant via Groq (fast, cheap)
Input: final label + original query + batch statistics
Output: {approved, flags, batch_quality_score}
"""

import json

from dotenv import load_dotenv
from groq import Groq, RateLimitError
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()

_EXPECTED_INTENTS = 77

_SYSTEM = """You are a quality control agent in a banking intent annotation pipeline.

For each submitted annotation perform two semantic checks and return a JSON verdict.

CHECK 1 — Semantic coherence
  Does the assigned label match the customer query semantically?
  A label is WRONG if the query clearly belongs to a different intent family.
  Example mismatch: label="change_pin" for query="my card was lost"

CHECK 2 — Label plausibility
  Is this label plausible for this text? Plausible = you would not be surprised
  to see it correctly annotated this way in a banking dataset.

You will also receive pre-computed drift alerts (computed externally via statistics).
Incorporate any drift alerts into your flags and lower the batch_quality_score accordingly.

Respond with ONLY this JSON (no markdown):
{"approved": true, "flags": [], "batch_quality_score": 0.95}

Rules:
- approved=false only if the annotation is semantically wrong (CHECK 1 or 2 fails)
- flags: list of short concern strings (empty list if none)
- batch_quality_score: 0.0–1.0; lower it when drift alerts are present"""


class QCOutput(BaseModel):
    approved: bool
    flags: list[str] = Field(default_factory=list)
    batch_quality_score: float = Field(ge=0.0, le=1.0)


_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq()
    return _client


def _detect_drift(label: str, batch_stats: dict[str, int]) -> list[str]:
    """Python-side drift detection — more reliable than asking an LLM to do arithmetic."""
    if not batch_stats:
        return []
    total = sum(batch_stats.values())
    expected = total / _EXPECTED_INTENTS
    alerts = []
    for intent, count in batch_stats.items():
        if count > 3 * expected:
            alerts.append(
                f"Drift: '{intent}' appears {count}x (expected ≈{expected:.1f}, "
                f"threshold={3 * expected:.1f})"
            )
    return alerts


def _build_prompt(query: str, label: str, batch_stats: dict[str, int]) -> str:
    drift_alerts = _detect_drift(label, batch_stats)
    drift_section = (
        "Drift alerts (pre-computed):\n" + "\n".join(f"  - {a}" for a in drift_alerts)
        if drift_alerts
        else "Drift alerts: none"
    )
    return f"Query: {query}\nAssigned label: {label}\n\n{drift_section}"


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
)
def quality_check(
    query: str,
    label: str,
    batch_stats: dict[str, int] | None = None,
) -> QCOutput:
    """Check annotation quality and detect batch-level drift."""
    stats = batch_stats or {}
    response = _get_client().chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _build_prompt(query, label, stats)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = json.loads(response.choices[0].message.content)
    if raw.get("flags") is None:
        raw["flags"] = []
    output = QCOutput(**raw)
    # Merge Python-detected drift flags so they're always present regardless of LLM
    drift_flags = _detect_drift(label, stats)
    for flag in drift_flags:
        if flag not in output.flags:
            output.flags.append(flag)
    if drift_flags:
        output.batch_quality_score = min(output.batch_quality_score, 0.6)
    return output
