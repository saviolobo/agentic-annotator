"""Integration tests — full pipeline against live APIs.

Runs all 10 fixture items through the LangGraph pipeline end-to-end.
Requires both GROQ_API_KEY and CEREBRAS_API_KEY.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

requires_both = pytest.mark.skipif(
    not (os.getenv("GROQ_API_KEY") and os.getenv("CEREBRAS_API_KEY")),
    reason="GROQ_API_KEY and CEREBRAS_API_KEY both required",
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "items"


def _load_fixtures() -> list[dict]:
    return [json.loads(f.read_text()) for f in sorted(FIXTURES_DIR.glob("item_*.json"))]


ALL_ITEMS = _load_fixtures()


@requires_both
@pytest.mark.parametrize("item", ALL_ITEMS, ids=[i["query_id"] for i in ALL_ITEMS])
def test_pipeline_produces_valid_label(item: dict) -> None:
    """Every fixture item should exit the pipeline with a valid Banking77 label."""
    from graph.pipeline import build_pipeline
    from mcp_servers.label_schema_mcp.server import list_valid_intents

    valid_intents = {e["intent_name"] for e in list_valid_intents()}
    pipeline = build_pipeline()

    result = pipeline.invoke(
        {"query_id": item["query_id"], "query": item["text"], "batch_stats": {}}
    )

    assert result.get("final_label"), f"No final_label produced for {item['query_id']}"
    assert result["final_label"] in valid_intents, (
        f"Label '{result['final_label']}' is not a valid Banking77 intent "
        f"(query_id={item['query_id']})"
    )
    assert result.get("qc_output") is not None, f"QC skipped for {item['query_id']}"


@requires_both
def test_pipeline_simple_path_skips_validator() -> None:
    """A clear, unambiguous query should take the SIMPLE path — no validator output."""
    from graph.pipeline import build_pipeline

    pipeline = build_pipeline()
    result = pipeline.invoke(
        {"query_id": "test_simple", "query": "My card got lost.", "batch_stats": {}}
    )

    assert result.get("route") == "SIMPLE", (
        f"Expected SIMPLE route, got {result.get('route')}"
    )
    assert result.get("validator_output") is None, (
        "Validator should not run on a SIMPLE item"
    )
    assert result.get("final_label") == "lost_or_stolen_card", (
        f"Expected lost_or_stolen_card, got {result.get('final_label')}"
    )
    assert result.get("arbitrator_output") is None


@requires_both
def test_pipeline_state_structure() -> None:
    """Verify the state TypedDict fields are present after a full run."""
    from graph.pipeline import build_pipeline

    pipeline = build_pipeline()
    result = pipeline.invoke(
        {"query_id": "test_struct", "query": "Please delete my account.", "batch_stats": {}}
    )

    assert result.get("route") in ("SIMPLE", "COMPLEX")
    assert result.get("primary_output") is not None
    assert result.get("final_label")
    assert result.get("qc_output") is not None
    assert isinstance(result.get("route_to_human", False), bool)
    assert 0.0 <= result["qc_output"].batch_quality_score <= 1.0


@requires_both
def test_pipeline_with_checkpointer() -> None:
    """Pipeline should work with SqliteSaver checkpointing enabled."""
    from graph.pipeline import build_pipeline, make_checkpointer

    checkpointer = make_checkpointer(":memory:")
    pipeline = build_pipeline(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "test-thread-001"}}

    result = pipeline.invoke(
        {"query_id": "chk_01", "query": "I lost my card.", "batch_stats": {}},
        config=config,
    )

    assert result.get("final_label") is not None
    assert result.get("qc_output") is not None
