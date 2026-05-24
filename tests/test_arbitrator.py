"""Arbitrator Agent tests — runs against live Cerebras + Redis APIs.

Uses synthetic disagreements so we control the inputs precisely.
Skipped automatically if CEREBRAS_API_KEY is not set.
"""

import os

import pytest

from agents.primary_annotator import AnnotatorOutput

requires_cerebras = pytest.mark.skipif(
    not os.getenv("CEREBRAS_API_KEY"), reason="CEREBRAS_API_KEY not set"
)


def _make_output(label: str, confidence: float, reasoning: str) -> AnnotatorOutput:
    return AnnotatorOutput(
        label=label, confidence=confidence, reasoning=reasoning, evidence="synthetic"
    )


# --- Synthetic disagreement scenarios ---


@requires_cerebras
def test_arbitrator_picks_correct_label_on_clear_disagreement() -> None:
    """unable_to_verify vs verify_my_identity on fixture item_006."""
    from agents.arbitrator import ArbitratorOutput, arbitrate

    query = "I am having trouble verifying my identity."
    primary = _make_output(
        "unable_to_verify_identity",
        0.75,
        "Customer says they are having trouble, which implies a technical difficulty.",
    )
    validator = _make_output(
        "verify_my_identity", 0.65, "Customer is asking about the identity verification process."
    )

    result = arbitrate(query, primary, validator)

    assert isinstance(result, ArbitratorOutput)
    assert result.final_label in ("unable_to_verify_identity", "verify_my_identity")
    assert result.final_label == "unable_to_verify_identity", (
        f"Expected unable_to_verify_identity (trouble = failing), got {result.final_label}\n"
        f"Explanation: {result.explanation}"
    )
    assert len(result.explanation) > 10


@requires_cerebras
def test_arbitrator_resolves_refund_boundary() -> None:
    """Refund_not_showing_up vs request_refund — refund already initiated."""
    from agents.arbitrator import arbitrate

    query = "I requested a refund from a store but it hasn't arrived."
    primary = _make_output(
        "Refund_not_showing_up", 0.72, "The refund was already requested and is not showing up."
    )
    validator = _make_output("request_refund", 0.68, "Customer wants a refund from a store.")

    result = arbitrate(query, primary, validator)
    assert result.final_label in ("Refund_not_showing_up", "request_refund")
    assert result.final_label == "Refund_not_showing_up", (
        f"Expected Refund_not_showing_up (refund already requested), got {result.final_label}\n"
        f"Explanation: {result.explanation}"
    )


@requires_cerebras
def test_arbitrator_routes_to_human_on_low_confidence() -> None:
    """When both annotations are weak, final confidence should fall below threshold."""
    from agents.arbitrator import CONFIDENCE_THRESHOLD, arbitrate

    # Deeply ambiguous query with conflicting weak signals
    query = "There is an issue with my account that I need help with."
    primary = _make_output("card_not_working", 0.40, "Possible card issue.")
    validator = _make_output(
        "balance_not_updated_after_bank_transfer", 0.35, "Could be balance issue."
    )

    result = arbitrate(query, primary, validator)
    assert result.route_to_human is True or result.confidence < CONFIDENCE_THRESHOLD, (
        f"Expected low confidence or human routing for a vague query, "
        f"got confidence={result.confidence:.2f}, route_to_human={result.route_to_human}"
    )


@requires_cerebras
def test_arbitrator_output_structure() -> None:
    from agents.arbitrator import ArbitratorOutput, arbitrate

    query = "My card got lost."
    primary = _make_output("lost_or_stolen_card", 0.95, "Card is missing.")
    # Fake disagreement — Validator returns wrong label
    validator = _make_output("compromised_card", 0.55, "Possible fraud on card.")

    result = arbitrate(query, primary, validator)
    assert isinstance(result, ArbitratorOutput)
    assert result.final_label == "lost_or_stolen_card"
    assert isinstance(result.route_to_human, bool)
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.explanation) > 10
