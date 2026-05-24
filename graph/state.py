from __future__ import annotations

from typing import TypedDict

from agents.arbitrator import ArbitratorOutput
from agents.primary_annotator import AnnotatorOutput
from agents.quality_controller import QCOutput


class AnnotationState(TypedDict, total=False):
    query_id: str
    query: str
    route: str  # "SIMPLE" | "COMPLEX"
    primary_output: AnnotatorOutput | None
    validator_output: AnnotatorOutput | None
    arbitrator_output: ArbitratorOutput | None
    qc_output: QCOutput | None
    final_label: str
    route_to_human: bool
    batch_stats: dict[str, int]
    error: str | None
