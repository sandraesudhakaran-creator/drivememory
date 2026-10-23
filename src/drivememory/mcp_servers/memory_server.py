"""MCP server exposing the driver memory to any MCP-capable agent or LLM client.

Run:  python -m drivememory.mcp_servers.memory_server            (stdio)
      python -m drivememory.mcp_servers.memory_server --http     (streamable HTTP)
"""

from __future__ import annotations

import sys
from typing import Any

from ..agents.describe import describe
from ..runtime import get_system
from ._compat import Server

mcp = Server(
    name="drivememory-memory",
    instructions="Long-term memory about a vehicle's driver: preferences learned from vehicle signals, "
                 "trip episodes, and privacy controls. Always respect forget requests.",
)


@mcp.tool()
def get_preferences(driver_id: str, domain: str = "", include_candidates: bool = False) -> list[dict]:
    """List learned preferences for a driver. domain: climate | media | autonomy | routine | '' (all)."""
    memory, _ = get_system()
    prefs = memory.profile(driver_id, include_candidates=include_candidates)
    now = memory.now(driver_id)
    return [
        {"id": p.id, "value": p.value, "status": p.status(now),
         "confidence": round(p.effective_confidence(now), 2), "description": describe(p)}
        for p in prefs if p.key.startswith(domain)
    ]


@mcp.tool()
def search_memories(driver_id: str, query: str, k: int = 5) -> list[dict]:
    """Semantic search over the driver's past trips (episodic memory)."""
    memory, _ = get_system()
    return memory.search(driver_id, query, k)


@mcp.tool()
def explain_preference(driver_id: str, key: str, context: str = "*") -> dict[str, Any]:
    """Explain why a preference is believed: evidence, confidence and history."""
    memory, _ = get_system()
    return memory.explain(driver_id, key, context) or {"error": "no such preference"}


@mcp.tool()
def remember_preference(driver_id: str, key: str, value: Any, context: str = "*",
                        value_type: str = "categorical") -> dict:
    """Store a preference the driver stated explicitly (e.g. key='climate.cabin_temp_c', value=20)."""
    memory, _ = get_system()
    return memory.remember(driver_id, key, context, value, value_type).to_dict(memory.now(driver_id))


@mcp.tool()
def forget(driver_id: str, domain: str = "", key: str = "") -> dict:
    """Delete memories. Give a domain (climate/media/autonomy/routine), a key, or neither to delete everything."""
    memory, _ = get_system()
    n = memory.forget(driver_id, domain=domain or None, key=key or None)
    return {"deleted": n}


@mcp.resource("profile://{driver_id}")
def profile_resource(driver_id: str) -> str:
    """A plain-text summary of everything the assistant knows about a driver."""
    memory, _ = get_system()
    lines = [f"- {describe(p)}" for p in memory.profile(driver_id)]
    return "\n".join(lines) or "(no active memories yet)"


def main() -> None:
    mcp.run("streamable-http" if "--http" in sys.argv else "stdio")


if __name__ == "__main__":
    main()
