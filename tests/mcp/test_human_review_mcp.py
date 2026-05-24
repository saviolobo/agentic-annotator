"""Smoke tests for human-review-mcp (SQLite-backed queue)."""

import json
import sqlite3

import pytest

from mcp_servers.human_review_mcp.server import (
    add_to_review_queue,
    get_completed_reviews,
    get_pending_count,
)


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """Route all tool calls to a fresh temp DB for each test."""
    monkeypatch.setenv("HUMAN_REVIEW_DB_PATH", str(tmp_path / "test_review.db"))


def test_empty_queue_count() -> None:
    result = get_pending_count()
    assert result["pending_count"] == 0


def test_add_single_item() -> None:
    result = add_to_review_queue(
        query_id="test_00001",
        text="I am having trouble verifying my identity.",
        reason="Primary and Validator disagreed",
        agent_outputs={"primary": "unable_to_verify_identity", "validator": "verify_my_identity"},
    )
    assert result["status"] == "queued"
    assert result["query_id"] == "test_00001"
    assert isinstance(result["id"], int)


def test_pending_count_increments() -> None:
    add_to_review_queue("q1", "text one", "reason", {"primary": "card_arrival"})
    add_to_review_queue("q2", "text two", "reason", {"primary": "change_pin"})
    assert get_pending_count()["pending_count"] == 2


def test_completed_reviews_empty_when_none_done() -> None:
    add_to_review_queue("q1", "some text", "reason", {})
    reviews = get_completed_reviews("2000-01-01T00:00:00+00:00")
    assert reviews == []


def test_completed_reviews_returned_after_manual_update(tmp_path, monkeypatch) -> None:
    """Simulate a human marking an item complete directly in SQLite."""
    import os

    db_path = os.environ["HUMAN_REVIEW_DB_PATH"]

    add_to_review_queue(
        "q1", "My card got lost.", "low confidence", {"primary": "lost_or_stolen_card"}
    )

    # Simulate a human reviewer marking it complete
    conn = sqlite3.connect(db_path)
    conn.execute(
        """UPDATE review_queue
           SET status = 'completed', human_decision = 'lost_or_stolen_card',
               reviewed_at = '2099-01-01T12:00:00+00:00'
           WHERE query_id = 'q1'"""
    )
    conn.commit()
    conn.close()

    reviews = get_completed_reviews("2099-01-01T00:00:00+00:00")
    assert len(reviews) == 1
    assert reviews[0]["human_decision"] == "lost_or_stolen_card"
    assert reviews[0]["query_id"] == "q1"


def test_agent_outputs_stored_as_json(tmp_path, monkeypatch) -> None:
    import os

    db_path = os.environ["HUMAN_REVIEW_DB_PATH"]
    outputs = {"primary": "card_arrival", "validator": "card_delivery_estimate", "confidence": 0.42}

    add_to_review_queue("q1", "Where is my card?", "disagreement", outputs)

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT agent_outputs FROM review_queue WHERE query_id='q1'").fetchone()
    conn.close()

    stored = json.loads(row[0])
    assert stored == outputs
