"""Banking77 dataset exploration script.

Loads data directly from PolyAI's GitHub repo (raw CSV) — no remote code execution.
Same canonical split (10,003 train / 3,080 test) used by the MAFA paper (AAAI 2026).
"""

import io
import urllib.request

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"


def fetch_csv(filename: str) -> pd.DataFrame:
    url = f"{BASE}/{filename}"
    with urllib.request.urlopen(url) as resp:  # noqa: S310
        return pd.read_csv(io.BytesIO(resp.read()))


def main() -> None:
    print("=" * 60)
    print("Loading Banking77 from PolyAI GitHub (raw CSV)...")
    print("=" * 60)

    train = fetch_csv("train.csv")
    test = fetch_csv("test.csv")

    # Normalise column names — PolyAI CSVs use 'text' and 'category'
    train.columns = [c.strip() for c in train.columns]
    test.columns = [c.strip() for c in test.columns]

    print(f"\nTrain split: {len(train):,} items")
    print(f"Test split:  {len(test):,} items")
    print(f"Total:       {len(train) + len(test):,} items")
    print(f"\nTrain columns: {list(train.columns)}")
    print(f"Sample row:    {dict(train.iloc[0])}")

    # --- Intent names ---
    label_col = "category" if "category" in train.columns else train.columns[-1]
    intent_names: list[str] = sorted(train[label_col].unique().tolist())

    print(f"\n{'=' * 60}")
    print(f"All {len(intent_names)} intents:")
    print("=" * 60)
    for i, name in enumerate(intent_names):
        print(f"  {i:3d}. {name}")

    # --- Value counts (balance check) ---
    print(f"\n{'=' * 60}")
    print("Intent distribution — train split (are intents balanced?)")
    print("=" * 60)
    counts = train[label_col].value_counts()
    print(
        f"  Min count: {counts.min()}  Max count: {counts.max()}  "
        f"Mean: {counts.mean():.1f}  Std: {counts.std():.1f}"
    )
    print("\n  Top 5 most frequent:")
    for intent, cnt in counts.head(5).items():
        print(f"    {intent:<50} {cnt}")
    print("\n  Bottom 5 least frequent:")
    for intent, cnt in counts.tail(5).items():
        print(f"    {intent:<50} {cnt}")

    # --- 5 sample items ---
    text_col = "text" if "text" in train.columns else train.columns[0]
    print(f"\n{'=' * 60}")
    print("5 sample items (text + intent label):")
    print("=" * 60)
    for i, row in train.head(5).iterrows():
        print(f"\n  [{i + 1}] intent: {row[label_col]}")
        print(f"       text:  {row[text_col]}")

    # --- Top 5 most confused intent pairs via embedding similarity ---
    print(f"\n{'=' * 60}")
    print("Top 5 most confused intent pairs (by intent-name embedding similarity):")
    print("=" * 60)
    print("  Loading sentence-transformers/all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    readable = [name.replace("_", " ") for name in intent_names]
    embeddings: np.ndarray = model.encode(readable, normalize_embeddings=True)

    sim_matrix: np.ndarray = embeddings @ embeddings.T
    np.fill_diagonal(sim_matrix, -1.0)

    n = len(intent_names)
    pairs: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((float(sim_matrix[i, j]), i, j))

    pairs.sort(reverse=True)
    print()
    for rank, (score, i, j) in enumerate(pairs[:5], 1):
        print(f"  {rank}. [{score:.4f}]  {intent_names[i]}  ↔  {intent_names[j]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
