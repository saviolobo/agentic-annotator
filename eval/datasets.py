"""Build and persist train / holdout splits for Banking77.

Usage:
    uv run python eval/datasets.py
"""

import io
import json
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"


def _fetch_csv(filename: str) -> pd.DataFrame:
    url = f"{BASE}/{filename}"
    with urllib.request.urlopen(url) as resp:  # noqa: S310
        df = pd.read_csv(io.BytesIO(resp.read()))
    df.columns = [c.strip() for c in df.columns]
    return df


def build_splits(seed: int = 42, holdout_n: int = 500) -> dict[str, pd.DataFrame]:
    """Load Banking77, build stratified holdout, persist parquets + index.

    Returns dict with keys 'train' and 'holdout'.
    """
    train_df = _fetch_csv("train.csv")
    test_df = _fetch_csv("test.csv")

    # Add a stable integer id to the full test split
    test_df = test_df.reset_index(drop=True)
    test_df.insert(0, "query_id", [f"test_{i:05d}" for i in test_df.index])

    # Stratified sample of holdout_n items from the 3,080-item test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=holdout_n, random_state=seed)
    _, holdout_idx = next(sss.split(test_df, test_df["category"]))
    holdout_idx_sorted = sorted(holdout_idx.tolist())

    holdout_df = test_df.iloc[holdout_idx_sorted].reset_index(drop=True)

    # Also persist train with query_ids
    train_df = train_df.reset_index(drop=True)
    train_df.insert(0, "query_id", [f"train_{i:05d}" for i in train_df.index])

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(SPLITS_DIR / "train.parquet", index=False)
    holdout_df.to_parquet(SPLITS_DIR / "holdout.parquet", index=False)
    (SPLITS_DIR / "holdout_indices.json").write_text(
        json.dumps(holdout_idx_sorted, indent=2)
    )

    print(f"train:   {len(train_df):,} rows → data/splits/train.parquet")
    print(f"holdout: {len(holdout_df):,} rows → data/splits/holdout.parquet")
    print(f"indices: {len(holdout_idx_sorted)} → data/splits/holdout_indices.json")
    print(f"holdout intents: {holdout_df['category'].nunique()} unique")

    return {"train": train_df, "holdout": holdout_df}


if __name__ == "__main__":
    build_splits()
