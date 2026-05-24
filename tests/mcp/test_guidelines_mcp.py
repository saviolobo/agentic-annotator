"""Smoke tests for guidelines-mcp.

JSON-backed tools run always.
Redis-backed search_similar_examples is skipped if Redis is unavailable.
"""

import pytest
import redis as redis_lib

from mcp_servers.guidelines_mcp.server import (
    get_edge_cases_for_intent,
    get_guidelines_for_intent,
    search_similar_examples,
)


def _redis_available() -> bool:
    try:
        r = redis_lib.Redis(host="localhost", port=6379)
        r.ping()
        # Also check the index exists
        r.ft("examples").info()
        return True
    except Exception:
        return False


requires_redis = pytest.mark.skipif(
    not _redis_available(), reason="Redis Stack not running or examples index missing"
)

# --- JSON-backed tools (always run) ---


def test_get_guidelines_known_intent() -> None:
    result = get_guidelines_for_intent("card_arrival")
    assert result["intent_name"] == "card_arrival"
    assert len(result["definition"]) > 10
    assert len(result["example_queries"]) == 3
    assert len(result["edge_cases"]) > 10
    assert len(result["commonly_confused_with"]) > 0


def test_get_guidelines_unknown_intent() -> None:
    result = get_guidelines_for_intent("not_a_real_intent")
    assert "error" in result


def test_get_edge_cases_known_intent() -> None:
    result = get_edge_cases_for_intent("unable_to_verify_identity")
    assert result["intent_name"] == "unable_to_verify_identity"
    assert "verify_my_identity" in result["commonly_confused_with"]


def test_get_edge_cases_unknown_intent() -> None:
    result = get_edge_cases_for_intent("bogus")
    assert "error" in result


# --- Redis-backed tool ---


@requires_redis
def test_search_similar_examples_returns_k_results() -> None:
    results = search_similar_examples("I am waiting for my card", k=5)
    assert len(results) == 5


@requires_redis
def test_search_similar_examples_fields() -> None:
    results = search_similar_examples("how do I change my PIN", k=3)
    for r in results:
        assert "text" in r
        assert "intent_name" in r
        assert "score" in r
        assert isinstance(r["score"], float)


@requires_redis
def test_search_similar_examples_relevance() -> None:
    results = search_similar_examples("my card was swallowed by the ATM", k=5)
    intents = [r["intent_name"] for r in results]
    assert "card_swallowed" in intents
