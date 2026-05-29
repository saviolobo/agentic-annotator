"""Ablation runner — 3-config evaluation of the annotation pipeline.

Config 1: Single LLM call, no multi-agent, no MCP (baseline)
Config 2: Multi-agent without MCP (structural benefit only)
Config 3: Full pipeline — multi-agent + MCP guidelines + Redis examples

Usage:
  uv run python eval/run_eval.py --smoke --n 10
  uv run python eval/run_eval.py --configs 3 --n 50
  uv run python eval/run_eval.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "items"
HOLDOUT_PATH = Path(__file__).parent.parent / "data" / "splits" / "holdout.parquet"


# ---------- item loading ----------


def _load_smoke_items(n: int) -> list[dict]:
    items = [json.loads(f.read_text()) for f in sorted(FIXTURES_DIR.glob("item_*.json"))]
    return items[:n]


def _load_holdout_items(n: int | None) -> list[dict]:
    if not HOLDOUT_PATH.exists():
        print(
            f"ERROR: {HOLDOUT_PATH} not found. Run eval/datasets.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    import pandas as pd

    df = pd.read_parquet(HOLDOUT_PATH)
    if n:
        df = df.head(n)
    return [
        {"query_id": row.query_id, "text": row.text, "true_intent": row.category}
        for row in df.itertuples()
    ]


# ---------- Config 1: single LLM ----------

_SINGLE_LLM_SYSTEM = """You are a banking customer service intent classifier.
Assign exactly one intent label from the provided list.
Respond with ONLY valid JSON: {"label": "<intent_name>"}"""


def _run_config1(
    items: list[dict], out_path: Path | None = None, all_results: dict | None = None
) -> tuple[list, float]:
    from groq import Groq

    from eval.evaluators import EvalResult
    from mcp_servers.label_schema_mcp.server import list_valid_intents

    client = Groq()
    valid = {e["intent_name"] for e in list_valid_intents()}
    intent_list = "\n".join(f"- {n}" for n in sorted(valid))

    done_ids = {r["query_id"] for r in (all_results or {}).get("1", [])}
    results: list[EvalResult] = [EvalResult(**r) for r in (all_results or {}).get("1", [])]
    if done_ids:
        print(f"  Resuming: {len(done_ids)} items already done, skipping.")

    t_start = time.perf_counter()

    for i, item in enumerate(items, 1):
        if item["query_id"] in done_ids:
            continue
        print(f"  [1/{len(items)}→{i}] {item['query_id']}", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": _SINGLE_LLM_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Customer query: {item['text']}\n\nValid intents:\n{intent_list}"
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw = json.loads(resp.choices[0].message.content)
            label = raw.get("label", "").strip()
            if label not in valid:
                label = "__invalid__"
        except Exception as exc:
            print(f"[ERROR: {exc}]", end=" ")
            label = "__error__"

        elapsed = time.perf_counter() - t0
        print(f"→ {label} ({elapsed:.1f}s)")
        r = EvalResult(
            query_id=item["query_id"],
            predicted_label=label,
            true_label=item["true_intent"],
            elapsed_seconds=elapsed,
        )
        results.append(r)
        if out_path and all_results is not None:
            all_results["1"] = [vars(x) for x in results]
            out_path.write_text(json.dumps({"per_config_results": all_results}, indent=2))

    return results, time.perf_counter() - t_start


# ---------- Config 2: multi-agent, no MCP ----------


def _run_config2(
    items: list[dict], out_path: Path | None = None, all_results: dict | None = None
) -> tuple[list, float]:
    from agents.arbitrator import CONFIDENCE_THRESHOLD, arbitrate
    from agents.primary_annotator import annotate
    from agents.router import route as router_route
    from agents.validator import agrees_with, validate
    from eval.evaluators import EvalResult

    done_ids = {r["query_id"] for r in (all_results or {}).get("2", [])}
    results: list[EvalResult] = [EvalResult(**r) for r in (all_results or {}).get("2", [])]
    if done_ids:
        print(f"  Resuming: {len(done_ids)} items already done, skipping.")

    t_start = time.perf_counter()
    first_new = True

    for i, item in enumerate(items, 1):
        if item["query_id"] in done_ids:
            continue
        if not first_new:
            time.sleep(3)  # avoid Cerebras rate limits between items
        first_new = False
        print(f"  [2/{len(items)}→{i}] {item['query_id']}", end=" ", flush=True)
        t0 = time.perf_counter()
        query = item["text"]

        try:
            route_decision = router_route(query)
            primary = annotate(query, use_mcp=False)
            final_label = primary.label
            route_to_human = False

            if route_decision.route == "COMPLEX":
                validator = validate(query, use_mcp=False)
                if not agrees_with(validator, primary.label):
                    arb = arbitrate(query, primary, validator)
                    final_label = arb.final_label
                    route_to_human = arb.confidence < CONFIDENCE_THRESHOLD

            elapsed = time.perf_counter() - t0
            status = "→human" if route_to_human else f"→ {final_label}"
            print(f"{route_decision.route} {status} ({elapsed:.1f}s)")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"[ERROR: {exc}] ({elapsed:.1f}s)")
            final_label = "__error__"
            route_to_human = False

        r = EvalResult(
            query_id=item["query_id"],
            predicted_label=final_label,
            true_label=item["true_intent"],
            route_to_human=route_to_human,
            elapsed_seconds=elapsed,
        )
        results.append(r)
        if out_path and all_results is not None:
            all_results["2"] = [vars(x) for x in results]
            out_path.write_text(json.dumps({"per_config_results": all_results}, indent=2))

    return results, time.perf_counter() - t_start


# ---------- Config 3: full pipeline ----------


def _run_config3(
    items: list[dict], out_path: Path | None = None, all_results: dict | None = None
) -> tuple[list, float]:
    from eval.evaluators import EvalResult
    from graph.pipeline import build_pipeline

    pipeline = build_pipeline()

    done_ids = {r["query_id"] for r in (all_results or {}).get("3", [])}
    results: list[EvalResult] = [EvalResult(**r) for r in (all_results or {}).get("3", [])]
    if done_ids:
        print(f"  Resuming: {len(done_ids)} items already done, skipping.")

    t_start = time.perf_counter()
    first_new = True

    for i, item in enumerate(items, 1):
        if item["query_id"] in done_ids:
            continue
        if not first_new:
            time.sleep(10)  # avoid Cerebras rate limits between items
        first_new = False
        print(f"  [3/{len(items)}→{i}] {item['query_id']}", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            state = pipeline.invoke(
                {
                    "query_id": item["query_id"],
                    "query": item["text"],
                    "batch_stats": {},
                }
            )
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

        r = EvalResult(
            query_id=item["query_id"],
            predicted_label=final_label,
            true_label=item["true_intent"],
            route_to_human=route_to_human,
            elapsed_seconds=elapsed,
        )
        results.append(r)
        if out_path and all_results is not None:
            all_results["3"] = [vars(x) for x in results]
            out_path.write_text(json.dumps({"per_config_results": all_results}, indent=2))

    return results, time.perf_counter() - t_start


# ---------- main ----------

_CONFIG_NAMES = {
    "1": "Config 1: Single LLM (baseline)",
    "2": "Config 2: Multi-agent, no MCP",
    "3": "Config 3: Full pipeline (MCP)",
}

_CONFIG_FNS = {
    "1": _run_config1,
    "2": _run_config2,
    "3": _run_config3,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation eval on the annotation pipeline")
    parser.add_argument("--smoke", action="store_true", help="Use fixture items instead of holdout")
    parser.add_argument("--n", type=int, default=10, help="Number of items to evaluate")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=["1", "2", "3"],
        default=["1", "2", "3"],
        metavar="N",
        help="Which configs to run (default: 1 2 3)",
    )
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    if args.smoke:
        items = _load_smoke_items(args.n)
        print(f"Smoke mode: {len(items)} fixture items")
    else:
        items = _load_holdout_items(args.n)
        print(f"Holdout mode: {len(items)} items")

    if not items:
        print("No items to evaluate.", file=sys.stderr)
        sys.exit(1)

    from eval.evaluators import compute_summary, print_comparison

    out_path = Path(args.output) if args.output else None

    # Load existing results so we can merge, not overwrite
    existing: dict = {}
    if out_path and out_path.exists():
        existing = json.loads(out_path.read_text())

    all_results: dict[str, list] = existing.get("per_config_results", {})
    all_summaries = []

    for cfg in args.configs:
        print(f"\nRunning {_CONFIG_NAMES[cfg]} ...")
        results, elapsed = _CONFIG_FNS[cfg](items, out_path=out_path, all_results=all_results)
        summary = compute_summary(_CONFIG_NAMES[cfg], results, elapsed)
        all_results[cfg] = [vars(r) for r in results]
        all_summaries.append(summary)

    # Rebuild summaries for all configs present in the file
    from eval.evaluators import EvalResult
    for cfg, rows in all_results.items():
        if cfg not in args.configs:
            results_obj = [EvalResult(**r) for r in rows]
            elapsed = sum(r["elapsed_seconds"] for r in rows)
            all_summaries.append(compute_summary(_CONFIG_NAMES[cfg], results_obj, elapsed))

    all_summaries.sort(key=lambda s: s.config_name)
    print_comparison(all_summaries)

    if out_path:
        payload = {
            "summaries": [vars(s) for s in all_summaries],
            "per_config_results": all_results,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
