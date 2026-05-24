# Agentic Annotator — Project Context for Claude Code

## What we're building
A multi-agent data annotation pipeline that automates intent classification
on the Banking77 dataset (13,083 banking customer queries, 77 intents).
Five specialized agents replace manual human annotation for high-confidence
items while routing ambiguous cases to a human review queue.

Goal: portfolio project demonstrating production-grade multi-agent AI with
measurable business impact — automating a process that costs enterprises
$0.15/item manually, at ~$0.003/item via agents.

## The business problem
Every AI team needs labeled training data. Manual annotation is slow,
expensive, and inconsistent at scale. JP Morgan Chase published a
multi-agent annotation system (MAFA, AAAI 2026) that eliminated a
1 million utterance backlog saving 5,000+ hours/year. Inspired by JP Morgan's published MAFA architecture (AAAI 2026), this project
independently builds a similar multi-agent annotation system on the public
Banking77 benchmark, enabling reproducible comparison against their published
86% agreement rate.

## Stack (locked — do not substitute without asking)
- Python 3.11+, uv for package management
- LangGraph for orchestration (with SqliteSaver checkpointing)
- FastMCP for the 3 MCP servers
- Langfuse (self-hosted via Docker Compose) for tracing and evals
- Groq API: llama-3.1-8b-instant (Router, Quality Controller)
- Cerebras API: llama-3.3-70b (Primary Annotator, Validator, Arbitrator)
- tenacity for rate-limit retry/backoff on both providers
- Redis Vector Search for similar example retrieval in guidelines-mcp
- sentence-transformers/all-MiniLM-L6-v2 for local embeddings
- FastAPI for HTTP entrypoint
- Next.js 15 + shadcn/ui + Tailwind + Recharts for frontend
- SQLite via LangGraph SqliteSaver for state checkpointing
- Docker Compose for local dev
- pytest for tests, ruff for lint, mypy for types

## Dataset
- Banking77 (PolyAI/banking77 on Hugging Face)
- 13,083 customer service queries labeled with 77 banking intents
- 10,003 train / 3,080 test split
- Load via: from datasets import load_dataset; load_dataset("PolyAI/banking77")
- License: CC-BY 4.0

## Repo structure
agentic-annotator/
├── data/
│   ├── splits/                 # train/holdout parquet files
│   └── guidelines/             # annotation guidelines per intent
├── mcp_servers/
│   ├── guidelines_mcp/         # intent definitions + examples
│   ├── label_schema_mcp/       # valid labels + hierarchy
│   └── human_review_mcp/       # human-in-the-loop queue
├── agents/
│   ├── router.py               # complexity routing
│   ├── primary_annotator.py    # first-pass annotation
│   ├── validator.py            # independent second annotation
│   ├── arbitrator.py           # resolves disagreements
│   └── quality_controller.py  # consistency + drift detection
├── graph/                      # LangGraph state machine
├── eval/
│   ├── evaluators.py           # agreement rate, human review rate
│   ├── datasets.py             # holdout set construction
│   └── run_eval.py             # ablation runner (3 configs)
├── api/                        # FastAPI entrypoint
├── frontend/                   # Next.js 15 + shadcn/ui
├── tests/
│   ├── fixtures/               # sample Banking77 items
│   └── mcp/                    # MCP server smoke tests
├── notebooks/                  # data exploration only
├── docker-compose.yml
├── pyproject.toml
├── Makefile
└── .env.example

## The 5 agents

### 1. Router Agent (llama-3.1-8b via Groq — fast, cheap)
Input: raw query text
Job: assess annotation complexity
Output: route to SIMPLE or COMPLEX path
Simple = clear intent, route to fast annotation
Complex = ambiguous, needs deep reasoning

### 2. Primary Annotator Agent (llama-3.3-70b via Cerebras)
Input: query + intent guidelines from MCP + similar examples
Job: assign intent label + confidence score + reasoning
Output: {label, confidence, reasoning, evidence}
Uses guidelines-mcp and label-schema-mcp tools

### 3. Validator Agent (llama-3.3-70b via Cerebras — blind to Primary)
Input: same query + guidelines (does NOT see Primary's output)
Job: independently assign intent label
Output: {label, confidence, reasoning}
Agreement with Primary → high confidence annotation
Disagreement → routes to Arbitrator

### 4. Arbitrator Agent (llama-3.3-70b via Cerebras — only on disagreements)
Input: query + both annotations + reasoning
Job: make final call, explain decision
Output: {final_label, confidence, explanation}
If confidence still < threshold → human review queue

### 5. Quality Controller Agent (llama-3.1-8b via Groq — cheap)
Input: final label + original query + batch statistics
Job: consistency check + label drift detection
Output: {approved, flags, batch_quality_score}

## LangGraph flow
START
  → Router
    → [SIMPLE] → Primary Annotator → Quality Controller → END
    → [COMPLEX] → Primary Annotator
                → Validator (parallel where possible via Send API)
                  → [agreement] → Quality Controller → END
                  → [disagreement] → Arbitrator
                    → [confident] → Quality Controller → END
                    → [uncertain] → Human Review Queue → END

Key pattern: use LangGraph Send API for parallel Primary + Validator
execution — this is the sophisticated pattern to highlight in interviews.

## MCP servers

### guidelines-mcp
Wraps intent definitions + labeled examples indexed in Redis
Tools:
- get_guidelines_for_intent(intent_name) → definition + rules + edge cases
- search_similar_examples(query_text, k=5) → similar labeled examples
- get_edge_cases_for_intent(intent_name) → documented tricky cases

### label-schema-mcp
Wraps the 77-intent taxonomy
Tools:
- list_valid_intents() → all 77 intent names + descriptions
- get_similar_intents(intent_name) → easily confused intents
- validate_intent(intent_name) → is this a valid label?

### human-review-mcp
Wraps the human-in-the-loop queue (SQLite-backed)
Tools:
- add_to_review_queue(query, reason, agent_outputs) → queues item
- get_pending_count() → how many items awaiting review
- get_completed_reviews(since) → pull human decisions back in

## Eval design (3 ablation configs)
Config 1: Single LLM call (no multi-agent, no MCP) — baseline
Config 2: Multi-agent, no MCP (agents can't access guidelines)
Config 3: Full system (multi-agent + MCP) — target

Metrics per config:
- Agreement rate with Banking77 ground truth labels
- Human review rate (% routed to human queue)
- Cost per annotation (calculated from token usage)
- Throughput (items/hour)
- Macro-F1 on test split

Comparison baselines:
- JP Morgan MAFA: 86% agreement (AAAI 2026)
- Best supervised ML on Banking77: 87.35% Macro-F1
- Manual annotation: ~$0.15/item industry standard

## Working principles
1. Test before integrate. Every agent has unit tests against fixture items
   before wiring into LangGraph.
2. Trace everything. Every LLM call wrapped in Langfuse.
3. Parallel where possible. Use LangGraph Send API for Primary + Validator.
4. Cost discipline. Log token usage per agent per item.
5. Honest eval. Holdout split ONCE, never touched during development.
6. Ask before scope creep. If not in current phase checklist, stop and ask.

## Non-goals (do NOT build)
- Fine-tuning any model
- Real-time annotation UI (batch processing is enough)
- Multi-language support
- Active learning loops
- Integration with real annotation platforms (Label Studio etc.)

## API keys needed (.env, never commit)
GROQ_API_KEY=
CEREBRAS_API_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=http://localhost:3001

## How to verify each phase
After each phase run:
  make test      # all green
  make lint      # all green
  make eval-smoke  # 10 items through full pipeline, sanity check
Phase is not done until all three pass.