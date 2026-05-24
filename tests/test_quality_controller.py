"""Quality Controller Agent tests — runs against live Groq API.

Skipped automatically if GROQ_API_KEY is not set.
"""

import os

import pytest

requires_groq = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
)


@requires_groq
def test_qc_approves_correct_annotation() -> None:
    from agents.quality_controller import QCOutput, quality_check

    result = quality_check(
        query="What do I do if the ATM ate my card?",
        label="card_swallowed",
        batch_stats={"card_swallowed": 3, "change_pin": 5, "card_arrival": 4},
    )
    assert isinstance(result, QCOutput)
    assert result.approved is True
    assert result.batch_quality_score >= 0.5


@requires_groq
def test_qc_flags_obvious_label_mismatch() -> None:
    from agents.quality_controller import quality_check

    result = quality_check(
        query="I need to change my PIN.",
        label="transfer_timing",  # clearly wrong
        batch_stats={"transfer_timing": 2, "change_pin": 3},
    )
    assert result.approved is False, (
        f"Expected QC to reject 'transfer_timing' for PIN-change query, "
        f"flags={result.flags}"
    )
    assert len(result.flags) > 0


@requires_groq
def test_qc_detects_label_drift() -> None:
    from agents.quality_controller import quality_check

    # card_arrival appears 60 times in a 77-item batch — clear drift
    stats = {"card_arrival": 60}
    stats.update({f"intent_{i}": 1 for i in range(17)})  # fill to 77 items total

    result = quality_check(
        query="I am still waiting on my card?",
        label="card_arrival",
        batch_stats=stats,
    )
    # Drift should lower quality score and/or add a flag, even if item is approved
    assert result.batch_quality_score < 0.85 or len(result.flags) > 0, (
        f"Expected drift to be flagged: score={result.batch_quality_score}, "
        f"flags={result.flags}"
    )


@requires_groq
def test_qc_works_with_empty_batch_stats() -> None:
    from agents.quality_controller import quality_check

    result = quality_check(
        query="My card got lost.",
        label="lost_or_stolen_card",
        batch_stats={},
    )
    assert result.approved is True
    assert isinstance(result.flags, list)
    assert 0.0 <= result.batch_quality_score <= 1.0


@requires_groq
def test_qc_output_structure() -> None:
    from agents.quality_controller import QCOutput, quality_check

    result = quality_check("Please delete my account.", "terminate_account")
    assert isinstance(result, QCOutput)
    assert isinstance(result.approved, bool)
    assert isinstance(result.flags, list)
    assert 0.0 <= result.batch_quality_score <= 1.0
