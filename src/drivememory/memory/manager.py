"""MemoryManager: the public API of the memory system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..interpreter import Observation, SignalInterpreter, summarize_trip
from ..signals.models import Trip
from .consolidation import consolidate
from .models import Episode, Preference
from .store import MemoryStore

DOMAIN_PREFIXES = {
    "climate": "climate.", "temperature": "climate.", "seat": "climate.seat",
    "media": "media.", "music": "media.", "audio": "media.",
    "autonomy": "autonomy.", "driving": "autonomy.",
    "routine": "routine.", "location": "routine.", "destinations": "routine.",
}


@dataclass
class IngestResult:
    episode: Episode
    observations: list[Observation]
    updated: list[str] = field(default_factory=list)
    drifts: list[str] = field(default_factory=list)


class MemoryManager:
    """Ingests trips, consolidates preferences, and answers memory queries.

    Time is taken from the data (the latest trip seen per driver), so
    simulations with synthetic dates behave correctly.
    """

    def __init__(self, store: MemoryStore | None = None, interpreter: SignalInterpreter | None = None):
        self.store = store or MemoryStore()
        self.interpreter = interpreter or SignalInterpreter()
        self._clock: dict[str, datetime] = {}

    def now(self, driver_id: str) -> datetime:
        return self._clock.get(driver_id) or datetime.now()

    def _tick(self, driver_id: str, ts: datetime) -> None:
        if driver_id not in self._clock or ts > self._clock[driver_id]:
            self._clock[driver_id] = ts

    # -------------------------------------------------------------- writes
    def ingest_trip(self, trip: Trip) -> IngestResult:
        self._tick(trip.driver_id, trip.departure)
        observations = self.interpreter.interpret(trip)
        episode = Episode(
            driver_id=trip.driver_id,
            trip_id=trip.trip_id,
            timestamp=trip.departure,
            summary=summarize_trip(trip, observations),
            observations=[o.to_dict() for o in observations],
            manual_adjustments=trip.manual_adjustments,
        )
        self.store.add_episode(episode)
        result = IngestResult(episode, observations)
        for obs in observations:
            event = self.observe(trip.driver_id, obs)
            result.updated.append(f"{obs.key}@{obs.context}")
            if event == "drift":
                result.drifts.append(f"{obs.key}@{obs.context}")
        return result

    def observe(self, driver_id: str, obs: Observation) -> str | None:
        pref = self.store.get_preference(driver_id, obs.key, obs.context) or Preference(
            driver_id=driver_id, key=obs.key, context=obs.context, value_type=obs.value_type)
        event = consolidate(pref, obs)
        self.store.put_preference(pref)
        if event in ("drift", "stated"):
            self.store.audit(driver_id, event, {"preference": pref.id, "value": pref.value})
        return event

    def remember(self, driver_id: str, key: str, context: str, value: Any, value_type: str,
                 note: str = "") -> Preference:
        """Store something the driver explicitly told the assistant."""
        now = self.now(driver_id)
        obs = Observation(key, context, value, value_type, "chat", now, 1.0,
                          note or f"driver said they prefer {value}", source="stated")
        self.observe(driver_id, obs)
        return self.store.get_preference(driver_id, key, context)

    def forget(self, driver_id: str, domain: str | None = None, key: str | None = None,
               context: str | None = None) -> int:
        """Delete memories. ``domain=None and key=None`` forgets everything."""
        if key:
            n = self.store.delete_preferences(driver_id, key, context)
        elif domain:
            n = self.store.delete_preferences(driver_id, DOMAIN_PREFIXES.get(domain, domain))
        else:
            n = self.store.delete_preferences(driver_id) + self.store.delete_episodes(driver_id)
        self.store.audit(driver_id, "forget", {"domain": domain, "key": key, "context": context, "deleted": n})
        return n

    def prune(self, driver_id: str) -> int:
        """Remove memories that have faded below the forgetting threshold."""
        now = self.now(driver_id)
        n = 0
        for p in self.store.preferences(driver_id):
            if p.status(now) == "fading" and not p.user_confirmed:
                n += self.store.delete_preferences(driver_id, p.key, p.context)
        if n:
            self.store.audit(driver_id, "prune", {"deleted": n})
        return n

    # --------------------------------------------------------------- reads
    def recall(self, driver_id: str, key: str, contexts: list[str] | str = "*",
               min_status: str = "active") -> Preference | None:
        """Return the most specific *active* preference for ``key`` among ``contexts``."""
        if isinstance(contexts, str):
            contexts = [contexts]
        now = self.now(driver_id)
        for ctx in list(contexts) + ["*"]:
            p = self.store.get_preference(driver_id, key, ctx)
            if p and (p.status(now) == "active" or min_status == "any"):
                return p
        return None

    def profile(self, driver_id: str, include_candidates: bool = False) -> list[Preference]:
        now = self.now(driver_id)
        prefs = self.store.preferences(driver_id)
        if include_candidates:
            return prefs
        return [p for p in prefs if p.status(now) == "active"]

    def explain(self, driver_id: str, key: str, context: str = "*") -> dict | None:
        p = self.store.get_preference(driver_id, key, context)
        if not p:
            return None
        now = self.now(driver_id)
        return {
            "preference": p.id,
            "value": p.value,
            "status": p.status(now),
            "confidence": round(p.effective_confidence(now), 2),
            "evidence_count": round(p.evidence_count, 1),
            "user_confirmed": p.user_confirmed,
            "learned_since": p.first_seen.isoformat() if p.first_seen else None,
            "recent_evidence": p.sources[-3:],
            "history": p.history[-3:],
        }

    def search(self, driver_id: str, query: str, k: int = 5) -> list[dict]:
        return [
            {"score": round(s, 3), "trip_id": e.trip_id, "timestamp": e.timestamp.isoformat(), "summary": e.summary}
            for s, e in self.store.search_episodes(driver_id, query, k)
        ]
