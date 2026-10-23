"""GraphQL schema over the memory system (Strawberry)."""

from __future__ import annotations

import strawberry
from strawberry.scalars import JSON
from strawberry.types import Info

from ..agents.describe import describe
from ..memory import MemoryManager, Preference


@strawberry.type
class PreferenceType:
    id: str
    key: str
    context: str
    value: JSON
    confidence: float
    evidence_count: float
    status: str
    user_confirmed: bool
    description: str
    history: JSON


@strawberry.type
class EpisodeType:
    trip_id: str
    timestamp: str
    summary: str
    manual_adjustments: int
    score: float | None = None


def _pref(memory: MemoryManager, p: Preference) -> PreferenceType:
    now = memory.now(p.driver_id)
    return PreferenceType(
        id=p.id, key=p.key, context=p.context, value=p.value,
        confidence=round(p.effective_confidence(now), 3), evidence_count=round(p.evidence_count, 2),
        status=p.status(now), user_confirmed=p.user_confirmed, description=describe(p), history=p.history,
    )


@strawberry.type
class Driver:
    id: str

    @strawberry.field
    def preferences(self, info: Info, domain: str = "", include_candidates: bool = False) -> list[PreferenceType]:
        memory: MemoryManager = info.context["memory"]
        return [_pref(memory, p) for p in memory.profile(self.id, include_candidates) if p.key.startswith(domain)]

    @strawberry.field
    def episodes(self, info: Info, limit: int = 10) -> list[EpisodeType]:
        memory: MemoryManager = info.context["memory"]
        return [EpisodeType(trip_id=e.trip_id, timestamp=e.timestamp.isoformat(), summary=e.summary,
                            manual_adjustments=e.manual_adjustments)
                for e in memory.store.episodes(self.id, limit)]

    @strawberry.field
    def search(self, info: Info, query: str, k: int = 5) -> list[EpisodeType]:
        memory: MemoryManager = info.context["memory"]
        return [EpisodeType(trip_id=s.trip_id, timestamp=s.timestamp.isoformat(), summary=s.summary,
                            manual_adjustments=s.manual_adjustments, score=round(score, 3))
                for score, s in memory.store.search_episodes(self.id, query, k)]


@strawberry.type
class Query:
    @strawberry.field
    def driver(self, id: str) -> Driver:
        return Driver(id=id)

    @strawberry.field
    def drivers(self, info: Info) -> list[Driver]:
        return [Driver(id=d) for d in info.context["memory"].store.drivers()]


@strawberry.type
class Mutation:
    @strawberry.mutation
    def remember(self, info: Info, driver_id: str, key: str, value: JSON, context: str = "*",
                 value_type: str = "categorical") -> PreferenceType:
        memory: MemoryManager = info.context["memory"]
        return _pref(memory, memory.remember(driver_id, key, context, value, value_type))

    @strawberry.mutation
    def forget(self, info: Info, driver_id: str, domain: str | None = None, key: str | None = None) -> int:
        return info.context["memory"].forget(driver_id, domain=domain, key=key)


schema = strawberry.Schema(query=Query, mutation=Mutation)
