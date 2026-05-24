"""Smoke tests for label-schema-mcp — calls tool functions directly."""

from mcp_servers.label_schema_mcp.server import (
    get_similar_intents,
    list_valid_intents,
    validate_intent,
)


def test_list_returns_77_intents() -> None:
    intents = list_valid_intents()
    assert len(intents) == 77


def test_list_entries_have_required_fields() -> None:
    for entry in list_valid_intents():
        assert "intent_name" in entry
        assert "definition" in entry
        assert len(entry["definition"]) > 10


def test_get_similar_known_intent() -> None:
    similar = get_similar_intents("unable_to_verify_identity")
    assert len(similar) > 0
    # Should surface the tight verification cluster
    assert any(s in similar for s in ("verify_my_identity", "why_verify_identity"))


def test_get_similar_unknown_intent() -> None:
    assert get_similar_intents("not_a_real_intent") == []


def test_validate_intent_valid() -> None:
    result = validate_intent("card_arrival")
    assert result["valid"] is True
    assert result["intent_name"] == "card_arrival"


def test_validate_intent_invalid() -> None:
    result = validate_intent("make_it_rain")
    assert result["valid"] is False


def test_validate_intent_with_casing_quirk() -> None:
    # Refund_not_showing_up has capital R — must be accepted exactly as-is
    result = validate_intent("Refund_not_showing_up")
    assert result["valid"] is True
