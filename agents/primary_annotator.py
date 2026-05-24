"""Primary Annotator Agent — first-pass intent classification.

Model: qwen-3-235b-a22b-instruct-2507 via Cerebras (llama-3.3-70b unavailable on this account)
Tools used (called directly as Python functions):
  - guidelines-mcp: search_similar_examples, get_guidelines_for_intent
  - label-schema-mcp: list_valid_intents, validate_intent
Output: {label, confidence, reasoning, evidence}
"""

import json

from cerebras.cloud.sdk import Cerebras
from cerebras.cloud.sdk._exceptions import RateLimitError
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from mcp_servers.guidelines_mcp.server import get_guidelines_for_intent, search_similar_examples
from mcp_servers.label_schema_mcp.server import list_valid_intents

load_dotenv()

_VALID_INTENTS: set[str] = set()

_SYSTEM = """You are an expert banking customer service intent classifier.

You will be given:
1. A customer query
2. The list of all 77 valid Banking77 intent labels
3. Similar labeled examples retrieved from the training set
4. Detailed guidelines for the most likely candidate intents

Your job: assign EXACTLY ONE label from the 77 valid intents.

Rules:
- The label must be copied exactly from the valid intents list (including casing and punctuation)
- Confidence 0.9+ means you are certain; below 0.6 means the case is genuinely ambiguous
- Reasoning: 2-3 sentences explaining your choice and why you rejected close alternatives
- Evidence: the specific phrase(s) in the query that most support your label

Respond with ONLY this JSON (no markdown):
{"label": "...", "confidence": 0.85, "reasoning": "...", "evidence": "..."}"""


class AnnotatorOutput(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    evidence: str

    @field_validator("label")
    @classmethod
    def label_must_be_valid(cls, v: str) -> str:
        global _VALID_INTENTS
        if not _VALID_INTENTS:
            _VALID_INTENTS = {e["intent_name"] for e in list_valid_intents()}
        if v not in _VALID_INTENTS:
            raise ValueError(f"'{v}' is not a valid Banking77 intent")
        return v


def build_annotation_context(query: str, use_mcp: bool = True) -> str:
    """Build the shared user prompt used by Primary and Validator agents."""
    valid = list_valid_intents()
    intent_list = "\n".join(f"- {e['intent_name']}" for e in valid)

    if not use_mcp:
        return (
            f"Customer query: {query}\n\nValid intent labels (choose exactly one):\n{intent_list}"
        )

    similar = search_similar_examples(query, k=5)
    examples_block = "\n".join(f'  "{ex["text"]}" → {ex["intent_name"]}' for ex in similar)

    # Guidelines for top-3 unique intents from similar examples
    seen: list[str] = []
    for ex in similar:
        if ex["intent_name"] not in seen:
            seen.append(ex["intent_name"])
        if len(seen) == 3:
            break

    guidelines_block = ""
    for intent_name in seen:
        g = get_guidelines_for_intent(intent_name)
        if "error" not in g:
            guidelines_block += (
                f"\n### {g['intent_name']}\n"
                f"Definition: {g['definition']}\n"
                f"Edge cases: {g['edge_cases']}\n"
                f"Commonly confused with: {', '.join(g['commonly_confused_with'])}\n"
            )

    return (
        f"Customer query: {query}\n\n"
        f"Valid intent labels (choose exactly one):\n{intent_list}\n\n"
        f"Similar training examples:\n{examples_block}\n\n"
        f"Guidelines for top candidate intents:{guidelines_block}"
    )


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
def annotate(query: str, use_mcp: bool = True) -> AnnotatorOutput:
    """Assign an intent label to a banking customer query."""
    user_prompt = build_annotation_context(query, use_mcp=use_mcp)
    response = _get_client().chat.completions.create(
        model="qwen-3-235b-a22b-instruct-2507",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = json.loads(response.choices[0].message.content)
    return AnnotatorOutput(**raw)
