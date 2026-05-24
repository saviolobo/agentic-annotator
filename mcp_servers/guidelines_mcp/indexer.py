"""Index Banking77 training examples into Redis Stack for vector search.

Run once (or after resetting Redis):
    uv run python mcp_servers/guidelines_mcp/indexer.py
"""

import io
import os
import urllib.request

import numpy as np
import pandas as pd
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from sentence_transformers import SentenceTransformer

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension
INDEX_NAME = "examples"
KEY_PREFIX = "example:"


def get_client() -> redis.Redis:
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    return redis.Redis(host=host, port=port, decode_responses=False)


def build_index(r: redis.Redis) -> None:
    try:
        r.ft(INDEX_NAME).dropindex(delete_documents=True)
        print("Dropped existing index.")
    except Exception:
        pass

    schema = [
        TextField("text"),
        TextField("intent_name"),
        VectorField(
            "embedding",
            "HNSW",
            {"TYPE": "FLOAT32", "DIM": EMBEDDING_DIM, "DISTANCE_METRIC": "COSINE"},
        ),
    ]
    r.ft(INDEX_NAME).create_index(
        schema,
        definition=IndexDefinition(prefix=[KEY_PREFIX], index_type=IndexType.HASH),
    )
    print("Index created.")


def index_documents(r: redis.Redis, df: pd.DataFrame, model: SentenceTransformer) -> None:
    texts = df["text"].tolist()
    print(f"Encoding {len(texts):,} texts...")
    embeddings = model.encode(
        texts, normalize_embeddings=True, batch_size=256, show_progress_bar=True
    )

    pipe = r.pipeline(transaction=False)
    for i, (_, row) in enumerate(df.iterrows()):
        pipe.hset(
            f"{KEY_PREFIX}{i}",
            mapping={
                "text": row["text"],
                "intent_name": row["category"],
                "embedding": embeddings[i].astype(np.float32).tobytes(),
            },
        )
        if (i + 1) % 1000 == 0:
            pipe.execute()
            pipe = r.pipeline(transaction=False)
            print(f"  Stored {i + 1:,}/{len(df):,}")
    pipe.execute()
    print(f"Indexed {len(df):,} documents.")


def main() -> None:
    print("Loading training data from GitHub...")
    with urllib.request.urlopen(f"{BASE}/train.csv") as resp:  # noqa: S310
        df = pd.read_csv(io.BytesIO(resp.read()))
    df.columns = [c.strip() for c in df.columns]
    print(f"Loaded {len(df):,} rows.")

    r = get_client()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    build_index(r)
    index_documents(r, df, model)

    info = r.ft(INDEX_NAME).info()
    print(f"\nDone. Index '{INDEX_NAME}': {info['num_docs']} documents.")


if __name__ == "__main__":
    main()
