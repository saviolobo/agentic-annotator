"""Tests for eval/datasets.py — run once; holdout is written to disk."""

import json
from pathlib import Path

import pandas as pd
import pytest

SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"


@pytest.fixture(scope="module")
def splits() -> dict[str, pd.DataFrame]:
    from eval.datasets import build_splits
    return build_splits(seed=42, holdout_n=500)


def test_train_size(splits: dict[str, pd.DataFrame]) -> None:
    assert len(splits["train"]) == 10_003


def test_holdout_size(splits: dict[str, pd.DataFrame]) -> None:
    assert len(splits["holdout"]) == 500


def test_no_overlap(splits: dict[str, pd.DataFrame]) -> None:
    # train and holdout come from different source splits so no overlap possible,
    # but guard against accidental id collision anyway
    train_ids = set(splits["train"]["query_id"])
    holdout_ids = set(splits["holdout"]["query_id"])
    assert train_ids.isdisjoint(holdout_ids), "query_id overlap between train and holdout"


def test_holdout_all_intents(splits: dict[str, pd.DataFrame]) -> None:
    assert splits["holdout"]["category"].nunique() == 77


def test_parquets_written() -> None:
    assert (SPLITS_DIR / "train.parquet").exists()
    assert (SPLITS_DIR / "holdout.parquet").exists()


def test_indices_written_and_match(splits: dict[str, pd.DataFrame]) -> None:
    idx_path = SPLITS_DIR / "holdout_indices.json"
    assert idx_path.exists()
    indices = json.loads(idx_path.read_text())
    assert len(indices) == 500
    assert len(set(indices)) == 500  # no duplicates


def test_parquet_readable() -> None:
    train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    holdout = pd.read_parquet(SPLITS_DIR / "holdout.parquet")
    assert list(train.columns) == ["query_id", "text", "category"]
    assert list(holdout.columns) == ["query_id", "text", "category"]
