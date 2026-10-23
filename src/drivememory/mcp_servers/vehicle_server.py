"""MCP server exposing (simulated) vehicle signals and trip context.

In a real vehicle this would sit in front of the signal bus; here it is backed
by the persona simulator so agents can be developed and tested offline.

Run:  python -m drivememory.mcp_servers.vehicle_server
"""

from __future__ import annotations

import sys

from ..interpreter import plugins as _plugins  # noqa: F401  (registers built-in plugins)
from ..interpreter import registered_plugins
from ..runtime import get_system, simulate_driver
from ..signals.catalog import SIGNAL_CATALOG
from ..signals.simulator import PERSONAS, TripSimulator
from ._compat import Server

mcp = Server(
    name="drivememory-vehicle",
    instructions="Vehicle signal access: signal catalog, live state snapshots and trip simulation.",
)


@mcp.tool()
def list_signals() -> list[dict]:
    """List all vehicle signals the system can interpret, with units and the plugin that consumes them."""
    return [s.to_dict() for s in SIGNAL_CATALOG.values()]


@mcp.tool()
def list_interpreters() -> list[dict]:
    """List the registered signal-interpreter plugins."""
    return [{"name": n, "consumes": list(c.consumes), "doc": (c.__doc__ or "").strip()}
            for n, c in registered_plugins().items()]


@mcp.tool()
def get_vehicle_state(driver_id: str = "anna", minute: int = 10) -> dict:
    """Snapshot of all vehicle signals at a given minute of the driver's next simulated trip."""
    if driver_id not in PERSONAS:
        return {"error": f"unknown driver; try one of {sorted(PERSONAS)}"}
    sim = TripSimulator(PERSONAS[driver_id])
    trip = sim.simulate(sim.plan_week(0)[0])
    frame = trip.frames[min(minute, len(trip.frames) - 1)]
    return {"trip_id": trip.trip_id, "departure": trip.departure.isoformat(), **frame}


@mcp.tool()
def simulate_weeks(driver_id: str, weeks: int = 2, assisted: bool = True) -> dict:
    """Simulate weeks of driving for a persona and ingest it into memory."""
    memory, orchestrator = get_system()
    return simulate_driver(memory, orchestrator, driver_id, weeks, assisted)


def main() -> None:
    mcp.run("streamable-http" if "--http" in sys.argv else "stdio")


if __name__ == "__main__":
    main()
