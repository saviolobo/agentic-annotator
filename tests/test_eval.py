"""Unit tests for eval/evaluators.py — no API calls required."""

from __future__ import annotations

import pytest

from eval.evaluators import EvalResult, compute_summary, print_comparison


def _r(pred: str, true: str, human: bool = False, elapsed: float = 1.0) -> EvalResult:
    return EvalResult("qid", pred, true, route_to_human=human, elapsed_seconds=elapsed)


def test_perfect_agreement():
    results = [_r("card_swallowed", "card_swallowed"), _r("change_pin", "change_pin")]
    s = compute_summary("test", results, 2.0)
    assert s.agreement_rate == 1.0
    assert s.human_review_rate == 0.0
    assert s.n_items == 2


def test_zero_agreement():
    results = [_r("wrong", "card_swallowed"), _r("also_wrong", "change_pin")]
    s = compute_summary("test", results, 2.0)
    assert s.agreement_rate == 0.0


def test_partial_agreement():
    results = [_r("card_swallowed", "card_swallowed"), _r("wrong", "change_pin")]
    s = compute_summary("test", results, 2.0)
    assert s.agreement_rate == pytest.approx(0.5)


def test_human_routed_counts_as_wrong():
    # Human-routed item has a label (from arbitrator) but should count as wrong
    results = [
        _r("card_swallowed", "card_swallowed", human=False),
        _r("change_pin", "change_pin", human=True),  # routed despite matching label
    ]
    s = compute_summary("test", results, 2.0)
    assert s.agreement_rate == pytest.approx(0.5)
    assert s.human_review_rate == pytest.approx(0.5)


def test_all_human_routed():
    results = [_r("a", "a", human=True), _r("b", "b", human=True)]
    s = compute_summary("test", results, 2.0)
    assert s.agreement_rate == 0.0
    assert s.human_review_rate == 1.0


def test_throughput_calculation():
    results = [_r("a", "a"), _r("b", "b")]
    s = compute_summary("test", results, 1.0)  # 2 items in 1 second
    assert s.items_per_hour == pytest.approx(7200.0)


def test_macro_f1_is_float():
    results = [_r("card_swallowed", "card_swallowed"), _r("change_pin", "change_pin")]
    s = compute_summary("test", results, 1.0)
    assert isinstance(s.macro_f1, float)
    assert 0.0 <= s.macro_f1 <= 1.0


def test_empty_results_raises():
    with pytest.raises(ValueError, match="No results"):
        compute_summary("test", [], 1.0)


def test_print_comparison_output(capsys: pytest.CaptureFixture) -> None:
    results = [_r("card_swallowed", "card_swallowed")]
    s = compute_summary("Config 1: Baseline", results, 1.0)
    print_comparison([s])
    out = capsys.readouterr().out
    assert "Config 1: Baseline" in out
    assert "Agree" in out
    assert "MAFA" in out


def test_summary_fields_present():
    results = [_r("card_swallowed", "card_swallowed")]
    s = compute_summary("cfg", results, 1.0)
    assert s.config_name == "cfg"
    assert s.n_items == 1
    assert hasattr(s, "macro_f1")
    assert hasattr(s, "items_per_hour")
    assert hasattr(s, "total_elapsed_seconds")
