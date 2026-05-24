"""Primary Annotator Agent tests — runs against live Cerebras + Redis APIs.

Skipped automatically if CEREBRAS_API_KEY is not set.
Simple fixtures: assert exact label match.
Ambiguous fixtures: assert label is valid (LLM may disagree with our difficulty labeling).
"""

import json
import os
from pathlib import Path

import pytest

requires_cerebras = pytest.mark.skipif(
    not os.getenv("CEREBRAS_API_KEY"), reason="CEREBRAS_API_KEY not set"
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "items"


def _load_fixtures(difficulty: str | None = None) -> list[dict]:
    items = []
    for f in sorted(FIXTURES_DIR.glob("item_*.json")):
        item = json.loads(f.read_text())
        if difficulty is None or item["difficulty"] == difficulty:
            items.append(item)
    return items


SIMPLE_ITEMS = _load_fixtures("simple")
AMBIGUOUS_ITEMS = _load_fixtures("ambiguous")
ALL_ITEMS = _load_fixtures()


@requires_cerebras
@pytest.mark.parametrize("item", SIMPLE_ITEMS, ids=[i["true_intent"] for i in SIMPLE_ITEMS])
def test_simple_items_get_correct_label(item: dict) -> None:
    from agents.primary_annotator import annotate

    result = annotate(item["text"])
    assert result.label == item["true_intent"], (
        f"text: {item['text']!r}\n"
        f"expected: {item['true_intent']}\n"
        f"got:      {result.label}\n"
        f"reasoning: {result.reasoning}"
    )
    assert result.confidence >= 0.7, "Simple items should have high confidence"


@requires_cerebras
@pytest.mark.parametrize("item", AMBIGUOUS_ITEMS, ids=[i["true_intent"] for i in AMBIGUOUS_ITEMS])
def test_ambiguous_items_return_valid_label(item: dict) -> None:
    from agents.primary_annotator import AnnotatorOutput, annotate
    from mcp_servers.label_schema_mcp.server import list_valid_intents

    valid = {e["intent_name"] for e in list_valid_intents()}
    result = annotate(item["text"])

    assert isinstance(result, AnnotatorOutput)
    assert result.label in valid, f"Returned invalid label: {result.label!r}"
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.reasoning) > 10
    assert len(result.evidence) > 0


@requires_cerebras
def test_output_structure() -> None:
    from agents.primary_annotator import AnnotatorOutput, annotate

    result = annotate("My card got lost.")
    assert isinstance(result, AnnotatorOutput)
    assert result.label == "lost_or_stolen_card"
    assert len(result.reasoning) > 10
    assert len(result.evidence) > 0
    assert 0.0 <= result.confidence <= 1.0


@requires_cerebras
def test_invalid_label_raises() -> None:
    """Validator rejects labels not in the Banking77 taxonomy."""
    from agents.primary_annotator import AnnotatorOutput
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        AnnotatorOutput(
            label="not_a_real_intent",
            confidence=0.9,
            reasoning="test",
            evidence="test",
        )
