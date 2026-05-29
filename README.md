# Multi-Agent Banking Annotation System

A production-grade multi-agent pipeline for automated intent classification
on the Banking77 benchmark (13,083 queries, 77 intents), inspired by
JP Morgan's MAFA system (AAAI 2026).

## Results

| | This project | JP Morgan MAFA (AAAI 2026) |
|---|---|---|
| Agreement rate | **95.8%** | 86.0% |
| Macro-F1 | **89.6%** | — |
| Cost per item | **~$0.003** | — |
| Manual annotation | — | ~$0.15 |

> Evaluated on a 333-item holdout sample from the Banking77 test split.
> Full pipeline uses 5 agents with MCP-backed guideline retrieval and
> Redis vector search for similar-example lookup.

## Architecture

Five specialized agents replace manual annotation for high-confidence items
while routing ambiguous cases to a human review queue.

![Architecture](docs/architecture.png)

**Key pattern:** LangGraph Send API runs Primary Annotator and Validator
in parallel on COMPLEX queries, cutting latency on the most expensive path.

| Agent | Model | Role |
|---|---|---|
| Router | Groq llama-3.1-8b-instant | Classify query complexity |
| Primary Annotator | Cerebras gpt-oss-120b | First-pass label + confidence |
| Validator | Cerebras gpt-oss-120b | Independent second annotation |
| Arbitrator | Cerebras gpt-oss-120b | Resolve disagreements |
| Quality Controller | Groq llama-3.1-8b-instant | Consistency + drift detection |

## MCP Servers

Three FastMCP servers provide tools to the agents at inference time:

- **guidelines-mcp** — intent definitions + Redis vector search for similar labeled examples
- **label-schema-mcp** — 77 valid intent labels + taxonomy
- **human-review-mcp** — SQLite-backed queue for low-confidence items

## Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph + SqliteSaver checkpointing |
| MCP servers | FastMCP |
| Vector search | Redis Stack + sentence-transformers |
| API | FastAPI |
| Frontend | Next.js 15 + shadcn/ui + Recharts |
| Dataset | Banking77 — PolyAI (CC-BY 4.0) |

## Setup

```bash
cp .env.example .env        # add GROQ_API_KEY, CEREBRAS_API_KEY
uv sync
docker compose up -d        # Langfuse tracing
colima start && docker run -d -p 6379:6379 redis/redis-stack:latest
uv run python mcp_servers/guidelines_mcp/indexer.py
uv run python main.py       # API on :8000
cd frontend && npm run dev  # Dashboard on :3000
```

## Eval

```bash
# Smoke test — 10 fixture items
make eval-smoke

# Full ablation — 500 items, saves progress after every item
caffeinate -i uv run python -m eval.run_eval \
  --configs 1 2 3 --n 500 --output eval_results.json

# Retry failed items (rate limits etc.)
caffeinate -i uv run python -m eval.retry_errors \
  --input eval_results.json --sleep 60
```
