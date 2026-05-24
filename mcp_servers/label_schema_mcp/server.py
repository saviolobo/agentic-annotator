"""label-schema-mcp: serves the 77-intent Banking77 taxonomy.

Tools:
  list_valid_intents()          → all 77 intent names + definitions
  get_similar_intents(name)     → top confused neighbours
  validate_intent(name)         → bool validity check
"""

import json
from pathlib import Path

from fastmcp import FastMCP

_GUIDELINES_PATH = (
    Path(__file__).parent.parent.parent / "data" / "guidelines" / "intent_guidelines.json"
)


def _load() -> list[dict]:
    return json.loads(_GUIDELINES_PATH.read_text())


mcp = FastMCP("label-schema-mcp")


@mcp.tool()
def list_valid_intents() -> list[dict[str, str]]:
    """Return all 77 valid intent names with one-line definitions."""
    return [{"intent_name": g["intent_name"], "definition": g["definition"]} for g in _load()]


@mcp.tool()
def get_similar_intents(intent_name: str) -> list[str]:
    """Return the top-3 intents most commonly confused with intent_name."""
    for g in _load():
        if g["intent_name"] == intent_name:
            return g["commonly_confused_with"]
    return []


@mcp.tool()
def validate_intent(intent_name: str) -> dict[str, bool | str]:
    """Return whether intent_name is a valid Banking77 label."""
    valid = {g["intent_name"] for g in _load()}
    return {"intent_name": intent_name, "valid": intent_name in valid}


if __name__ == "__main__":
    mcp.run()
