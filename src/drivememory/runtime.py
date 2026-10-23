"""Builds the shared system (memory + orchestrator) used by the API, MCP servers and CLI."""

from __future__ import annotations

import os
from functools import cache

from .agents import Orchestrator
from .memory import MemoryManager, MemoryStore
from .signals.models import TripContext
from .signals.simulator import PERSONAS, TripSimulator


def default_db_path() -> str:
    # On AWS Lambda only /tmp is writable. Use a managed database in production.
    if os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
        return os.getenv("DRIVEMEMORY_DB", "/tmp/drivememory.db")
    return os.getenv("DRIVEMEMORY_DB", "drivememory.db")


@cache
def get_system(db_path: str | None = None) -> tuple[MemoryManager, Orchestrator]:
    memory = MemoryManager(MemoryStore(db_path or default_db_path()))
    return memory, Orchestrator(memory)


def simulate_driver(memory: MemoryManager, orchestrator: Orchestrator, driver_id: str, weeks: int = 4,
                    assisted: bool = True, seed: int = 7) -> dict:
    """Simulate ``weeks`` of driving for a persona and feed it into memory."""
    if driver_id not in PERSONAS:
        raise KeyError(f"unknown persona {driver_id!r}; choose from {sorted(PERSONAS)}")
    sim = TripSimulator(PERSONAS[driver_id], seed=seed)
    trips = adjustments = takeovers = 0
    for _, plans in sim.iter_weeks(weeks):
        for plan in plans:
            presets = orchestrator.on_trip_start(TripContext.from_plan(plan), compose=False).presets if assisted else {}
            trip = sim.simulate(plan, presets)
            memory.ingest_trip(trip)
            trips += 1
            adjustments += trip.manual_adjustments
            takeovers += trip.unplanned_takeovers
    return {"driver_id": driver_id, "weeks": weeks, "trips": trips, "manual_adjustments": adjustments,
            "unplanned_takeovers": takeovers, "active_preferences": len(memory.profile(driver_id))}
