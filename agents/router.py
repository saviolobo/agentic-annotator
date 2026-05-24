"""Router Agent — classifies query complexity for pipeline routing.

Model: llama-3.1-8b-instant via Groq (fast, cheap)
Output: SIMPLE (clear single intent) or COMPLEX (ambiguous, needs deep reasoning)
"""

import json
from typing import Literal

from dotenv import load_dotenv
from groq import Groq, RateLimitError
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()

_SYSTEM = """You are a routing agent for a banking intent classification pipeline with 77 fine-grained labels.

Your task: decide whether classifying the customer query requires SIMPLE or COMPLEX routing.

SIMPLE — one label is unambiguously correct; two experienced annotators would always agree.
  Examples:
  - "My card was eaten by the ATM"  → only card_swallowed fits
  - "I need to change my PIN"       → only change_pin fits
  - "Please delete my account"      → only terminate_account fits

COMPLEX — two or more labels are plausible; annotators might reasonably disagree.
  Always route COMPLEX for these patterns:
  - Refund/return language       → Refund_not_showing_up vs request_refund vs reverted_card_payment
  - Transfer timing questions    → pending_transfer vs transfer_timing vs transfer_not_received
  - Identity/verify language     → unable_to_verify_identity vs verify_my_identity vs why_verify_identity
  - Reluctance toward identity verification (not wanting to do it, asking if optional) → why_verify_identity vs unable_to_verify_identity
  - Unrecognised/strange charges → card_payment_not_recognised vs extra_charge_on_statement vs compromised_card

You MUST respond with ONLY this JSON (no markdown, no extra keys):
{"complexity": "SIMPLE", "reasoning": "one sentence", "confidence": 0.9}
  or
{"complexity": "COMPLEX", "reasoning": "one sentence", "confidence": 0.8}

The "complexity" value MUST be exactly the word SIMPLE or the word COMPLEX."""


class RouterDecision(BaseModel):
    route: Literal["SIMPLE", "COMPLEX"]
    reasoning: str = Field(description="One sentence explaining the routing decision")
    confidence: float = Field(ge=0.0, le=1.0)


_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq()
    return _client


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
)
def route(query: str) -> RouterDecision:
    """Route a banking query to SIMPLE or COMPLEX annotation path."""
    response = _get_client().chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Customer query: {query}"},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = json.loads(response.choices[0].message.content)
    # Model outputs "complexity" key per the prompt; map to our field name
    complexity = raw.get("complexity", raw.get("route", "")).upper().strip()
    if complexity not in ("SIMPLE", "COMPLEX"):
        complexity = "COMPLEX"  # default to caution on parse failure
    return RouterDecision(
        route=complexity,
        reasoning=raw.get("reasoning", ""),
        confidence=float(raw.get("confidence", 0.5)),
    )
