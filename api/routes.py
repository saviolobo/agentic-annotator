"""FastAPI route handlers."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.schemas import (
    AnnotateRequest,
    AnnotateResponse,
    EvalResultsResponse,
    EvalSummaryItem,
    QueueStats,
)

router = APIRouter()

_EVAL_RESULTS_PATH = Path("eval_results.json")

# Lazy-loaded pipeline singleton
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from graph.pipeline import build_pipeline

        _pipeline = build_pipeline()
    return _pipeline


# ---------- routes ----------


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "agentic-annotator"}


@router.post("/annotate", response_model=AnnotateResponse)
def annotate(req: AnnotateRequest) -> AnnotateResponse:
    """Run the full annotation pipeline on a single customer query."""
    query_id = req.query_id or str(uuid.uuid4())
    t0 = time.perf_counter()
    try:
        state = _get_pipeline().invoke(
            {"query_id": query_id, "query": req.text, "batch_stats": req.batch_stats}
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.perf_counter() - t0
    qc = state.get("qc_output")
    return AnnotateResponse(
        query_id=query_id,
        route=state.get("route", "UNKNOWN"),
        final_label=state.get("final_label", ""),
        route_to_human=bool(state.get("route_to_human")),
        qc_approved=qc.approved if qc else False,
        qc_flags=qc.flags if qc else [],
        elapsed_seconds=round(elapsed, 2),
    )


@router.get("/queue/pending", response_model=QueueStats)
def queue_pending() -> QueueStats:
    """Return the number of items currently awaiting human review."""
    from mcp_servers.human_review_mcp.server import get_pending_count

    result = get_pending_count()
    return QueueStats(pending_count=result.get("pending_count", 0))


@router.get("/eval/results", response_model=EvalResultsResponse)
def eval_results() -> EvalResultsResponse:
    """Return the most recent cached eval results, if available.

    Run `python -m eval.run_eval --output eval_results.json` to populate.
    """
    if not _EVAL_RESULTS_PATH.exists():
        return EvalResultsResponse(available=False)

    data = json.loads(_EVAL_RESULTS_PATH.read_text())
    summaries = [EvalSummaryItem(**s) for s in data.get("summaries", [])]
    return EvalResultsResponse(available=True, summaries=summaries)
