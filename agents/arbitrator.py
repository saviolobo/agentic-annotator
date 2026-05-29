"""Arbitrator Agent — resolves Primary/Validator disagreements.

Model: gpt-oss-120b via Cerebras
Only invoked when Primary and Validator labels differ.
If final confidence < CONFIDENCE_THRESHOLD → sets route_to_human=True.
Output: {final_label, confidence, explanation, route_to_human}
"""

import json

from cerebras.cloud.sdk import Cerebras
from cerebras.cloud.sdk._exceptions import RateLimitError
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agents.primary_annotator import AnnotatorOutput, build_annotation_context
from mcp_servers.label_schema_mcp.server import list_valid_intents

load_dotenv()

CONFIDENCE_THRESHOLD = 0.65

_SYSTEM = """You are the Arbitrator in a banking intent classification pipeline.
Two independent annotators have disagreed on the intent label for a customer query.
Your job: review both annotations, reason carefully, and make the final call.

You will receive:
- The customer query
- Annotator A's label, confidence, and reasoning
- Annotator B's label, confidence, and reasoning
- Similar training examples and intent guidelines for context

Rules:
- Carefully weigh both annotations against the guidelines and examples
- Choose the label that best fits the query semantics and the Banking77 definitions
- If you are still genuinely uncertain after careful review, assign a confidence below 0.65
  (this will trigger escalation to human review)
- The label must be copied exactly from the valid intents list

Respond with ONLY this JSON (no markdown):
{"final_label": "...", "confidence": 0.80, "explanation": "2-3 sentence explanation of the decision and why the rejected label was wrong"}"""

_VALID_INTENTS: set[str] = set()


class ArbitratorOutput(BaseModel):
    final_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    route_to_human: bool = False

    @field_validator("final_label")
    @classmethod
    def label_must_be_valid(cls, v: str) -> str:
        global _VALID_INTENTS
        if not _VALID_INTENTS:
            _VALID_INTENTS = {e["intent_name"] for e in list_valid_intents()}
        if v not in _VALID_INTENTS:
            raise ValueError(f"'{v}' is not a valid Banking77 intent")
        return v


_client: Cerebras | None = None


def _get_client() -> Cerebras:
    global _client
    if _client is None:
        _client = Cerebras()
    return _client


def _build_arbitration_prompt(
    query: str,
    primary: AnnotatorOutput,
    validator: AnnotatorOutput,
) -> str:
    context = build_annotation_context(query)
    disagreement = (
        f"DISAGREEMENT TO RESOLVE:\n"
        f"Annotator A: label={primary.label!r}, confidence={primary.confidence:.2f}\n"
        f"  Reasoning: {primary.reasoning}\n\n"
        f"Annotator B: label={validator.label!r}, confidence={validator.confidence:.2f}\n"
        f"  Reasoning: {validator.reasoning}\n"
    )
    return f"{disagreement}\n\n{context}"


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
)
def arbitrate(
    query: str,
    primary: AnnotatorOutput,
    validator: AnnotatorOutput,
) -> ArbitratorOutput:
    """Resolve a Primary/Validator disagreement and return the final label."""
    response = _get_client().chat.completions.create(
        model="gpt-oss-120b",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _build_arbitration_prompt(query, primary, validator)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = json.loads(response.choices[0].message.content)
    output = ArbitratorOutput(**raw)
    output.route_to_human = output.confidence < CONFIDENCE_THRESHOLD
    return output
