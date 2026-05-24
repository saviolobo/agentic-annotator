"""human-review-mcp: SQLite-backed human-in-the-loop review queue.

Tools:
  add_to_review_queue(query_id, text, reason, agent_outputs) → queues item
  get_pending_count()                                         → items awaiting review
  get_completed_reviews(since)                                → pull human decisions
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from fastmcp import FastMCP

_DEFAULT_DB = Path(__file__).parent.parent.parent / "data" / "human_review.db"

mcp = FastMCP("human-review-mcp")


def _db_path() -> Path:
    """Reads env var each call so tests can override via monkeypatch."""
    return Path(os.getenv("HUMAN_REVIEW_DB_PATH", str(_DEFAULT_DB)))


def _conn() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS review_queue (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id      TEXT    NOT NULL,
            text          TEXT    NOT NULL,
            reason        TEXT    NOT NULL,
            agent_outputs TEXT    NOT NULL,
            status        TEXT    NOT NULL DEFAULT 'pending',
            human_decision TEXT,
            created_at    TEXT    NOT NULL,
            reviewed_at   TEXT
        )
    """)
    conn.commit()
    return conn


@mcp.tool()
def add_to_review_queue(
    query_id: str,
    text: str,
    reason: str,
    agent_outputs: dict,
) -> dict:
    """Enqueue an annotation item for human review."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO review_queue (query_id, text, reason, agent_outputs, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (query_id, text, reason, json.dumps(agent_outputs), now),
        )
    return {"id": cur.lastrowid, "query_id": query_id, "status": "queued"}


@mcp.tool()
def get_pending_count() -> dict:
    """Return the number of items currently awaiting human review."""
    with _conn() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM review_queue WHERE status = 'pending'"
        ).fetchone()[0]
    return {"pending_count": count}


@mcp.tool()
def get_completed_reviews(since: str) -> list[dict]:
    """Return completed human reviews with reviewed_at >= since (ISO 8601 string)."""
    with _conn() as conn:
        rows = conn.execute(
            """SELECT id, query_id, text, reason, human_decision, reviewed_at
               FROM review_queue
               WHERE status = 'completed' AND reviewed_at >= ?
               ORDER BY reviewed_at DESC""",
            (since,),
        ).fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    mcp.run()
