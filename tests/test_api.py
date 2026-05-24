"""API tests — no live API calls required for basic route checks."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from api.app import create_app

client = TestClient(create_app())


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_eval_results_unavailable(tmp_path, monkeypatch) -> None:
    """When no eval_results.json exists the endpoint returns available=False."""
    import api.routes as routes_mod

    monkeypatch.setattr(routes_mod, "_EVAL_RESULTS_PATH", tmp_path / "missing.json")
    resp = client.get("/eval/results")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False
    assert data["summaries"] == []


def test_eval_results_available(tmp_path, monkeypatch) -> None:
    """When eval_results.json exists the endpoint returns summaries."""
    import json

    import api.routes as routes_mod

    results_file = tmp_path / "eval_results.json"
    results_file.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "config_name": "Config 3: Full pipeline (MCP)",
                        "n_items": 10,
                        "agreement_rate": 0.9,
                        "macro_f1": 0.88,
                        "human_review_rate": 0.1,
                        "items_per_hour": 8.0,
                        "total_elapsed_seconds": 4500.0,
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(routes_mod, "_EVAL_RESULTS_PATH", results_file)
    resp = client.get("/eval/results")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert len(data["summaries"]) == 1
    assert data["summaries"][0]["agreement_rate"] == 0.9


def test_annotate_validation_rejects_empty_text() -> None:
    resp = client.post("/annotate", json={"text": ""})
    assert resp.status_code == 422


def test_annotate_calls_pipeline(monkeypatch) -> None:
    """POST /annotate wires through to the pipeline and returns the right shape."""
    import api.routes as routes_mod
    from agents.quality_controller import QCOutput

    fake_state = {
        "route": "SIMPLE",
        "final_label": "lost_or_stolen_card",
        "route_to_human": False,
        "qc_output": QCOutput(approved=True, flags=[], batch_quality_score=0.95),
    }
    mock_pipeline = MagicMock()
    mock_pipeline.invoke.return_value = fake_state
    monkeypatch.setattr(routes_mod, "_pipeline", mock_pipeline)

    resp = client.post("/annotate", json={"text": "My card got lost."})
    assert resp.status_code == 200
    data = resp.json()
    assert data["final_label"] == "lost_or_stolen_card"
    assert data["route"] == "SIMPLE"
    assert data["route_to_human"] is False
    assert data["qc_approved"] is True
    assert "elapsed_seconds" in data
