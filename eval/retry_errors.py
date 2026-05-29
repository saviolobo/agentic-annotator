"""Retry error items from a completed eval run.

Reads eval_results.json, finds items with __error__ / __none__ predicted labels,
reruns them through Config 3 with a longer sleep, and patches the file in place.

Usage:
  uv run python eval/retry_errors.py --input eval_results.json --sleep 60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

HOLDOUT_PATH = Path(__file__).parent.parent / "data" / "splits" / "holdout.parquet"
_ERROR_SENTINELS = {"__error__", "__none__", "__invalid__"}


def _load_query_text(query_ids: set[str]) -> dict[str, dict]:
    import pandas as pd

    df = pd.read_parquet(HOLDOUT_PATH)
    rows = df[df["query_id"].isin(query_ids)]
    return {
        row.query_id: {"query_id": row.query_id, "text": row.text, "true_intent": row.category}
        for row in rows.itertuples()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="eval_results.json")
    parser.add_argument("--config", default="3", choices=["1", "2", "3"])
    parser.add_argument("--sleep", type=int, default=60, help="Seconds between items")
    args = parser.parse_args()

    path = Path(args.input)
    data = json.loads(path.read_text())
    cfg = args.config
    results = data["per_config_results"].get(cfg, [])

    errors = [r for r in results if r["predicted_label"] in _ERROR_SENTINELS]
    if not errors:
        print(f"No errors found in config {cfg}. Nothing to retry.")
        return

    print(
        f"Found {len(errors)} error items to retry (config {cfg}), sleep={args.sleep}s between items"
    )
    error_ids = {r["query_id"] for r in errors}
    items_map = _load_query_text(error_ids)

    if cfg == "3":
        from graph.pipeline import build_pipeline

        pipeline = build_pipeline()

        patched = 0
        for i, err_result in enumerate(errors, 1):
            if i > 1:
                time.sleep(args.sleep)
            qid = err_result["query_id"]
            item = items_map.get(qid)
            if not item:
                print(f"  [{i}/{len(errors)}] {qid} — not found in holdout, skipping")
                continue

            print(f"  [{i}/{len(errors)}] {qid}", end=" ", flush=True)
            t0 = time.perf_counter()
            try:
                state = pipeline.invoke({"query_id": qid, "query": item["text"], "batch_stats": {}})
                elapsed = time.perf_counter() - t0
                final_label = state.get("final_label", "__none__")
                route_to_human = bool(state.get("route_to_human"))
                status = "→human" if route_to_human else f"→ {final_label}"
                print(f"{state.get('route', '?')} {status} ({elapsed:.1f}s)")
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                print(f"[ERROR: {exc}] ({elapsed:.1f}s)")
                final_label = "__error__"
                route_to_human = False

            # Patch result in place
            for r in data["per_config_results"][cfg]:
                if r["query_id"] == qid:
                    r["predicted_label"] = final_label
                    r["route_to_human"] = route_to_human
                    r["elapsed_seconds"] = elapsed
                    break
            patched += 1

            # Save after every item
            path.write_text(json.dumps(data, indent=2))

    remaining = sum(
        1 for r in data["per_config_results"][cfg] if r["predicted_label"] in _ERROR_SENTINELS
    )
    print(f"\nDone. Patched {patched} items. Remaining errors: {remaining}")


if __name__ == "__main__":
    main()
