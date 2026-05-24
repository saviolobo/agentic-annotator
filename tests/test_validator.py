"""Validator Agent tests — runs against live Cerebras + Redis APIs.

Skipped automatically if CEREBRAS_API_KEY is not set.
"""

import json
import os
from pathlib import Path

import pytest

requires_cerebras = pytest.mark.skipif(
    not os.getenv("CEREBRAS_API_KEY"), reason="CEREBRAS_API_KEY not set"
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "items"


def _load_fixtures(difficulty: str) -> list[dict]:
    return [
        json.loads(f.read_text())
        for f in sorted(FIXTURES_DIR.glob("item_*.json"))
        if json.loads(f.read_text())["difficulty"] == difficulty
    ]


SIMPLE_ITEMS = _load_fixtures("simple")
AMBIGUOUS_ITEMS = _load_fixtures("ambiguous")


@requires_cerebras
@pytest.mark.parametrize("item", SIMPLE_ITEMS, ids=[i["true_intent"] for i in SIMPLE_ITEMS])
def test_simple_items_get_correct_label(item: dict) -> None:
    from agents.validator import validate

    result = validate(item["text"])
    assert result.label == item["true_intent"], (
        f"text: {item['text']!r}\n"
        f"expected: {item['true_intent']}, got: {result.label}\n"
        f"reasoning: {result.reasoning}"
    )
    assert result.confidence >= 0.7


@requires_cerebras
@pytest.mark.parametrize("item", AMBIGUOUS_ITEMS, ids=[i["true_intent"] for i in AMBIGUOUS_ITEMS])
def test_ambiguous_items_return_valid_label(item: dict) -> None:
    from agents.validator import validate
    from mcp_servers.label_schema_mcp.server import list_valid_intents

    valid = {e["intent_name"] for e in list_valid_intents()}
    result = validate(item["text"])
    assert result.label in valid
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.reasoning) > 10


@requires_cerebras
def test_agrees_with_helper() -> None:
    from agents.validator import agrees_with, validate

    result = validate("My card got lost.")
    assert result.label == "lost_or_stolen_card"
    assert agrees_with(result, "lost_or_stolen_card") is True
    assert agrees_with(result, "compromised_card") is False


@requires_cerebras
def test_validator_agrees_with_primary_on_simple_query() -> None:
    """Both agents should independently reach the same label on a clear query."""
    from agents.primary_annotator import annotate
    from agents.validator import agrees_with, validate

    query = "I need to change my PIN."
    primary = annotate(query)
    validator = validate(query)
    assert agrees_with(validator, primary.label), (
        f"Primary: {primary.label}, Validator: {validator.label} — "
        "expected agreement on an unambiguous query"
    )
