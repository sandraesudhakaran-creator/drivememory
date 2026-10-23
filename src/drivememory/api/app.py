"""FastAPI application: REST + GraphQL + A2A endpoints.

Run locally:  uvicorn drivememory.api.app:app --reload
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from strawberry.fastapi import GraphQLRouter

from .. import __version__
from ..agents import Message
from ..agents.describe import describe
from ..runtime import get_system, simulate_driver
from ..signals.catalog import SIGNAL_CATALOG
from ..signals.models import Trip, TripContext
from .graphql_schema import schema


class TripStartIn(BaseModel):
    departure: datetime | None = None
    outside_temp_c: float = 10.0
    origin: str = "home"
    rain_expected: bool = False
    expected_duration_min: int | None = None


class ChatIn(BaseModel):
    text: str = Field(..., examples=["why is it so warm?"])


class RememberIn(BaseModel):
    key: str = Field(..., examples=["climate.cabin_temp_c"])
    value: Any = Field(..., examples=[20.5])
    context: str = "*"
    value_type: str = "numeric"


def create_app(db_path: str | None = None) -> FastAPI:
    memory, orchestrator = get_system(db_path)
    app = FastAPI(
        title="DriveMemory",
        version=__version__,
        description="Context-aware memory system for an in-car AI assistant.",
    )

    async def gql_context() -> dict:
        return {"memory": memory}

    app.include_router(GraphQLRouter(schema, context_getter=gql_context), prefix="/graphql")

    # ------------------------------------------------------------ general
    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/signals")
    def signals() -> list[dict]:
        return [s.to_dict() for s in SIGNAL_CATALOG.values()]

    # ------------------------------------------------------------ drivers
    @app.post("/drivers/{driver_id}/trips")
    def ingest_trip(driver_id: str, trip: dict) -> dict:
        trip.setdefault("driver_id", driver_id)
        try:
            t = Trip.from_dict(trip)
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(422, f"invalid trip: {e}") from e
        res = memory.ingest_trip(t)
        return {"trip_id": t.trip_id, "summary": res.episode.summary,
                "observations": [o.to_dict() for o in res.observations], "drifts": res.drifts}

    @app.post("/drivers/{driver_id}/simulate")
    def simulate(driver_id: str, weeks: int = 2, assisted: bool = True, seed: int = 7) -> dict:
        try:
            return simulate_driver(memory, orchestrator, driver_id, weeks, assisted, seed)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e

    @app.post("/drivers/{driver_id}/trip-start")
    def trip_start(driver_id: str, body: TripStartIn) -> dict:
        ctx = TripContext(driver_id, body.departure or memory.now(driver_id), body.outside_temp_c,
                          body.origin, body.rain_expected, body.expected_duration_min)
        return asdict(orchestrator.on_trip_start(ctx))

    @app.post("/drivers/{driver_id}/chat")
    def chat(driver_id: str, body: ChatIn) -> dict:
        return orchestrator.chat(driver_id, body.text)

    @app.get("/drivers/{driver_id}/preferences")
    def preferences(driver_id: str, include_candidates: bool = False) -> list[dict]:
        now = memory.now(driver_id)
        return [{**p.to_dict(now), "description": describe(p)}
                for p in memory.profile(driver_id, include_candidates)]

    @app.post("/drivers/{driver_id}/preferences")
    def remember(driver_id: str, body: RememberIn) -> dict:
        p = memory.remember(driver_id, body.key, body.context, body.value, body.value_type)
        return p.to_dict(memory.now(driver_id))

    @app.get("/drivers/{driver_id}/preferences/{key}")
    def explain(driver_id: str, key: str, context: str = "*") -> dict:
        ex = memory.explain(driver_id, key, context)
        if not ex:
            raise HTTPException(404, "no such preference")
        return ex

    @app.delete("/drivers/{driver_id}/memories")
    def forget(driver_id: str, domain: str | None = None, key: str | None = None) -> dict:
        return {"deleted": memory.forget(driver_id, domain=domain, key=key)}

    @app.get("/drivers/{driver_id}/episodes/search")
    def search(driver_id: str, q: str, k: int = 5) -> list[dict]:
        return memory.search(driver_id, q, k)

    @app.get("/drivers/{driver_id}/audit")
    def audit(driver_id: str) -> list[dict]:
        return memory.store.audit_log(driver_id)

    # ---------------------------------------------------------------- A2A
    @app.get("/a2a/agents")
    def agent_cards(request: Request) -> list[dict]:
        base = str(request.base_url).rstrip("/")
        return [{**c, "url": f"{base}/a2a/{c['name']}"} for c in orchestrator.bus.cards()]

    @app.get("/a2a/{agent}/.well-known/agent-card.json")
    def agent_card(agent: str, request: Request) -> dict:
        if agent not in orchestrator.bus.agents:
            raise HTTPException(404, "unknown agent")
        card = orchestrator.bus.agents[agent].card.to_dict()
        card["url"] = f"{str(request.base_url).rstrip('/')}/a2a/{agent}"
        return card

    @app.post("/a2a/{agent}")
    def a2a_rpc(agent: str, rpc: dict) -> dict:
        rid = rpc.get("id")
        if agent not in orchestrator.bus.agents:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32001, "message": "unknown agent"}}
        if rpc.get("method") != "message/send":
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "method not found"}}
        try:
            msg = Message.from_dict(rpc["params"]["message"])
        except (KeyError, TypeError) as e:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": f"invalid params: {e}"}}
        task = orchestrator.bus.send(agent, msg)
        return {"jsonrpc": "2.0", "id": rid, "result": task.to_dict()}

    return app


app = create_app()
