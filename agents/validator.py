"""Validator Agent — independent second-pass annotation (blind to Primary).

Model: gpt-oss-120b via Cerebras
Key constraint: receives the same query + guidelines as Primary, but NEVER sees
Primary's label or reasoning. Agreement → high-confidence annotation.
Disagreement → routes to Arbitrator.
Output: {label, confidence, reasoning}
"""

import json

from cerebras.cloud.sdk import Cerebras
from cerebras.cloud.sdk._exceptions import RateLimitError
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agents.primary_annotator import AnnotatorOutput, build_annotation_context

load_dotenv()

_SYSTEM = """You are an independent banking customer service intent classifier.

IMPORTANT: You are performing a blind second annotation. You have not seen any prior
classification of this query and must form your own independent judgment.

You will be given:
1. A customer query
2. The list of all 77 valid Banking77 intent labels
3. Similar labeled examples retrieved from the training set
4. Detailed guidelines for the most likely candidate intents

Your job: assign EXACTLY ONE label from the 77 valid intents independently.

Rules:
- The label must be copied exactly from the valid intents list (including casing and punctuation)
- Confidence 0.9+ means you are certain; below 0.6 means the case is genuinely ambiguous
- Reasoning: 2-3 sentences explaining your choice and why you rejected close alternatives

Respond with ONLY this JSON (no markdown):
{"label": "...", "confidence": 0.85, "reasoning": "...", "evidence": "..."}"""

_client: Cerebras | None = None


def _get_client() -> Cerebras:
    global _client
    if _client is None:
        _client = Cerebras()
    return _client


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
)
def validate(query: str, use_mcp: bool = True) -> AnnotatorOutput:
    """Independently assign an intent label — blind to any prior annotation."""
    response = _get_client().chat.completions.create(
        model="gpt-oss-120b",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": build_annotation_context(query, use_mcp=use_mcp)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = json.loads(response.choices[0].message.content)
    return AnnotatorOutput(**raw)


def agrees_with(validator_output: AnnotatorOutput, primary_label: str) -> bool:
    """Return True if Validator's label matches Primary's label."""
    return validator_output.label == primary_label
