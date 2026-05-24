"""guidelines-mcp: intent guidelines + semantic example search via Redis.

Tools:
  get_guidelines_for_intent(intent_name)      → definition + examples + edge cases
  search_similar_examples(query_text, k=5)    → k nearest training examples
  get_edge_cases_for_intent(intent_name)      → edge cases + confused neighbours
"""

import json
import os
from pathlib import Path

import numpy as np
import redis
from fastmcp import FastMCP
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

_GUIDELINES_PATH = (
    Path(__file__).parent.parent.parent / "data" / "guidelines" / "intent_guidelines.json"
)
INDEX_NAME = "examples"

mcp = FastMCP("guidelines-mcp")

# Lazy singletons — initialised on first use to avoid slow startup
_guidelines_cache: list[dict] | None = None
_model: SentenceTransformer | None = None
_redis_client: redis.Redis | None = None


def _guidelines() -> list[dict]:
    global _guidelines_cache
    if _guidelines_cache is None:
        _guidelines_cache = json.loads(_GUIDELINES_PATH.read_text())
    return _guidelines_cache


def _model_instance() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def _redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        _redis_client = redis.Redis(host=host, port=port, decode_responses=False)
    return _redis_client


@mcp.tool()
def get_guidelines_for_intent(intent_name: str) -> dict:
    """Return definition, example queries, edge cases, and confused neighbours."""
    for g in _guidelines():
        if g["intent_name"] == intent_name:
            return {
                "intent_name": g["intent_name"],
                "definition": g["definition"],
                "example_queries": g["example_queries"],
                "edge_cases": g["edge_cases"],
                "commonly_confused_with": g["commonly_confused_with"],
            }
    return {"error": f"Intent '{intent_name}' not found"}


@mcp.tool()
def search_similar_examples(query_text: str, k: int = 5) -> list[dict]:
    """Return k training examples semantically closest to query_text."""
    vec = (
        _model_instance().encode(query_text, normalize_embeddings=True).astype(np.float32).tobytes()
    )
    q = (
        Query(f"*=>[KNN {k} @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("text", "intent_name", "score")
        .dialect(2)
    )
    results = _redis().ft(INDEX_NAME).search(q, query_params={"vec": vec})
    return [
        {
            "text": doc.text.decode() if isinstance(doc.text, bytes) else doc.text,
            "intent_name": (
                doc.intent_name.decode() if isinstance(doc.intent_name, bytes) else doc.intent_name
            ),
            "score": float(doc.score),
        }
        for doc in results.docs
    ]


@mcp.tool()
def get_edge_cases_for_intent(intent_name: str) -> dict:
    """Return edge cases and commonly confused intents for intent_name."""
    for g in _guidelines():
        if g["intent_name"] == intent_name:
            return {
                "intent_name": g["intent_name"],
                "edge_cases": g["edge_cases"],
                "commonly_confused_with": g["commonly_confused_with"],
            }
    return {"error": f"Intent '{intent_name}' not found"}


if __name__ == "__main__":
    mcp.run()
