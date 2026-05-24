"""LangGraph annotation pipeline.

Flow:
  START → route_query
    SIMPLE  → run_primary → check_agreement → run_qc → END
    COMPLEX → run_primary ─┐
              run_validator ┘ (parallel via Send API)
              → check_agreement
                  agree      → run_qc → END
                  disagree   → run_arbitrator
                                  confident → run_qc → END
                                  uncertain → human_review → END
"""

from __future__ import annotations

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.arbitrator import arbitrate
from agents.primary_annotator import annotate
from agents.quality_controller import quality_check
from agents.router import route as router_route
from agents.validator import agrees_with, validate
from mcp_servers.human_review_mcp.server import add_to_review_queue

from .state import AnnotationState

# ---------- nodes ----------


def route_node(state: AnnotationState) -> dict:
    decision = router_route(state["query"])
    return {"route": decision.route}


def primary_node(state: AnnotationState) -> dict:
    return {"primary_output": annotate(state["query"])}


def validator_node(state: AnnotationState) -> dict:
    return {"validator_output": validate(state["query"])}


def check_agreement_node(state: AnnotationState) -> dict:
    primary = state.get("primary_output")
    if primary is None:
        return {"error": "primary_output missing"}
    validator = state.get("validator_output")
    if validator is None or agrees_with(validator, primary.label):
        return {"final_label": primary.label}
    return {}


def arbitrator_node(state: AnnotationState) -> dict:
    result = arbitrate(
        state["query"],
        state["primary_output"],  # type: ignore[arg-type]
        state["validator_output"],  # type: ignore[arg-type]
    )
    return {
        "arbitrator_output": result,
        "final_label": result.final_label,
        "route_to_human": result.route_to_human,
    }


def qc_node(state: AnnotationState) -> dict:
    result = quality_check(
        state["query"],
        state["final_label"],  # type: ignore[arg-type]
        state.get("batch_stats", {}),
    )
    return {"qc_output": result}


def human_review_node(state: AnnotationState) -> dict:
    agent_outputs: dict = {}
    if p := state.get("primary_output"):
        agent_outputs["primary"] = p.model_dump()
    if v := state.get("validator_output"):
        agent_outputs["validator"] = v.model_dump()
    if a := state.get("arbitrator_output"):
        agent_outputs["arbitrator"] = a.model_dump()

    add_to_review_queue(
        query_id=state.get("query_id", "unknown"),
        text=state["query"],
        reason="Low arbitrator confidence",
        agent_outputs=agent_outputs,
    )
    return {"route_to_human": True}


# ---------- routing functions ----------


def dispatch_by_route(state: AnnotationState) -> list[Send] | str:
    if state.get("route") == "SIMPLE":
        return "run_primary"
    return [Send("run_primary", state), Send("run_validator", state)]


def route_after_agreement(state: AnnotationState) -> str:
    return "run_qc" if state.get("final_label") else "run_arbitrator"


def route_after_arbitration(state: AnnotationState) -> str:
    return "human_review" if state.get("route_to_human") else "run_qc"


# ---------- build ----------


def build_pipeline(checkpointer=None):
    """Compile the annotation pipeline.

    Pass a SqliteSaver (or MemorySaver) to enable state checkpointing.
    Without a checkpointer the graph still runs; interrupts/resume won't work.
    """
    workflow = StateGraph(AnnotationState)

    workflow.add_node("route_query", route_node)
    workflow.add_node("run_primary", primary_node)
    workflow.add_node("run_validator", validator_node)
    workflow.add_node("check_agreement", check_agreement_node)
    workflow.add_node("run_arbitrator", arbitrator_node)
    workflow.add_node("run_qc", qc_node)
    workflow.add_node("human_review", human_review_node)

    workflow.add_edge(START, "route_query")
    workflow.add_conditional_edges("route_query", dispatch_by_route)

    # both parallel branches (and the SIMPLE branch) fan-in here
    workflow.add_edge("run_primary", "check_agreement")
    workflow.add_edge("run_validator", "check_agreement")

    workflow.add_conditional_edges("check_agreement", route_after_agreement)
    workflow.add_conditional_edges("run_arbitrator", route_after_arbitration)

    workflow.add_edge("run_qc", END)
    workflow.add_edge("human_review", END)

    return workflow.compile(checkpointer=checkpointer)


def make_checkpointer(db_path: str = ".checkpoints.db") -> SqliteSaver:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver
