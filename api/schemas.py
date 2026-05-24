"""Pydantic request / response models for the REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnnotateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The customer query to annotate")
    query_id: str = Field(default="", description="Optional caller-supplied ID")
    batch_stats: dict[str, int] = Field(
        default_factory=dict,
        description="Running label counts for drift detection",
    )


class AnnotateResponse(BaseModel):
    query_id: str
    route: str
    final_label: str
    route_to_human: bool
    qc_approved: bool
    qc_flags: list[str]
    elapsed_seconds: float


class QueueStats(BaseModel):
    pending_count: int


class EvalSummaryItem(BaseModel):
    config_name: str
    n_items: int
    agreement_rate: float
    macro_f1: float
    human_review_rate: float
    items_per_hour: float
    total_elapsed_seconds: float


class EvalResultsResponse(BaseModel):
    available: bool
    summaries: list[EvalSummaryItem] = []
