"""Router Agent unit tests — runs against live Groq API.

Skipped automatically if GROQ_API_KEY is not set.
"""

import json
import os
from pathlib import Path

import pytest

requires_groq = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "items"


def _load_fixtures(difficulty: str) -> list[dict]:
    items = []
    for f in sorted(FIXTURES_DIR.glob("item_*.json")):
        item = json.loads(f.read_text())
        if item["difficulty"] == difficulty:
            items.append(item)
    return items


SIMPLE_ITEMS = _load_fixtures("simple")
AMBIGUOUS_ITEMS = _load_fixtures("ambiguous")


@requires_groq
@pytest.mark.parametrize("item", SIMPLE_ITEMS, ids=[i["true_intent"] for i in SIMPLE_ITEMS])
def test_simple_items_route_to_simple(item: dict) -> None:
    from agents.router import route

    decision = route(item["text"])
    assert decision.route == "SIMPLE", (
        f"Expected SIMPLE for '{item['text']}'\n"
        f"Got {decision.route}: {decision.reasoning}"
    )
    assert 0.0 <= decision.confidence <= 1.0


@requires_groq
@pytest.mark.parametrize("item", AMBIGUOUS_ITEMS, ids=[i["true_intent"] for i in AMBIGUOUS_ITEMS])
def test_ambiguous_items_route_to_complex(item: dict) -> None:
    from agents.router import route

    decision = route(item["text"])
    assert decision.route == "COMPLEX", (
        f"Expected COMPLEX for '{item['text']}'\n"
        f"Got {decision.route}: {decision.reasoning}"
    )
    assert 0.0 <= decision.confidence <= 1.0


@requires_groq
def test_router_returns_valid_decision_structure() -> None:
    from agents.router import RouterDecision, route

    decision = route("How do I change my PIN?")
    assert isinstance(decision, RouterDecision)
    assert decision.route in ("SIMPLE", "COMPLEX")
    assert len(decision.reasoning) > 5
    assert 0.0 <= decision.confidence <= 1.0
