"""Evaluation metrics for the annotation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import f1_score


@dataclass
class EvalResult:
    query_id: str
    predicted_label: str
    true_label: str
    route_to_human: bool = False
    elapsed_seconds: float = 0.0


@dataclass
class EvalSummary:
    config_name: str
    n_items: int
    agreement_rate: float
    macro_f1: float
    human_review_rate: float
    items_per_hour: float
    total_elapsed_seconds: float


_ERROR_SENTINELS = {"__error__", "__none__", "__invalid__"}


def compute_summary(
    config_name: str,
    results: list[EvalResult],
    total_elapsed: float,
) -> EvalSummary:
    n_total = len(results)
    if n_total == 0:
        raise ValueError("No results to summarize")

    # Exclude API/network errors from accuracy metrics
    valid = [r for r in results if r.predicted_label not in _ERROR_SENTINELS]
    n = len(valid)

    if n == 0:
        return EvalSummary(
            config_name=config_name,
            n_items=n_total,
            agreement_rate=0.0,
            macro_f1=0.0,
            human_review_rate=0.0,
            items_per_hour=n_total / total_elapsed * 3600 if total_elapsed > 0 else 0.0,
            total_elapsed_seconds=total_elapsed,
        )

    human_count = sum(1 for r in valid if r.route_to_human)
    correct = sum(1 for r in valid if not r.route_to_human and r.predicted_label == r.true_label)

    preds = ["__human__" if r.route_to_human else r.predicted_label for r in valid]
    trues = [r.true_label for r in valid]
    mf1 = float(f1_score(trues, preds, average="macro", zero_division=0))

    return EvalSummary(
        config_name=config_name,
        n_items=n,
        agreement_rate=correct / n,
        macro_f1=mf1,
        human_review_rate=human_count / n,
        items_per_hour=n_total / total_elapsed * 3600 if total_elapsed > 0 else 0.0,
        total_elapsed_seconds=total_elapsed,
    )


def print_comparison(summaries: list[EvalSummary]) -> None:
    w = 35
    header = (
        f"{'Config':<{w}} {'N':>6} {'Agree%':>8} {'Macro-F1':>10} {'Human%':>8} {'items/hr':>10}"
    )
    print()
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s.config_name:<{w}} {s.n_items:>6} "
            f"{s.agreement_rate * 100:>7.1f}% "
            f"{s.macro_f1:>10.4f} "
            f"{s.human_review_rate * 100:>7.1f}% "
            f"{s.items_per_hour:>10.1f}"
        )
    print()
    print("Reference baselines:")
    print("  JP Morgan MAFA (AAAI 2026)   86.0%  agreement")
    print("  Best supervised ML           87.35% Macro-F1")
    print("  Manual annotation cost       ~$0.15/item")
    print()
