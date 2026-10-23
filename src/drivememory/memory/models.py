"""Memory data models: episodes (what happened) and preferences (what we learned)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

ACTIVE_THRESHOLD = 0.55  # effective confidence needed before the assistant acts on a preference
MIN_EVIDENCE = 2.5  # weighted observations needed before a preference can become active
HALF_LIFE_DAYS = 45.0  # unconfirmed memories lose half their confidence after this long
FORGET_BELOW = 0.10  # memories this weak are pruned


@dataclass
class Episode:
    """Episodic memory: a summary of one trip plus the observations it produced."""

    driver_id: str
    trip_id: str
    timestamp: datetime
    summary: str
    observations: list[dict] = field(default_factory=list)
    manual_adjustments: int = 0


@dataclass
class Preference:
    """Semantic / profile memory: a consolidated belief about the driver."""

    driver_id: str
    key: str
    context: str
    value_type: str
    value: Any = None
    confidence: float = 0.0
    evidence_count: float = 0.0
    noise_count: float = 0.0
    first_seen: datetime | None = None
    last_evidence: datetime | None = None
    user_confirmed: bool = False
    state: dict = field(default_factory=dict)  # algorithm state (counts, pending conflicts)
    history: list[dict] = field(default_factory=list)  # previous values (drift / corrections)
    sources: list[dict] = field(default_factory=list)  # recent evidence for explainability

    @property
    def id(self) -> str:
        return f"{self.key}@{self.context}"

    def effective_confidence(self, now: datetime | None = None) -> float:
        """Confidence after time decay. Memories that are not re-confirmed fade."""
        if now is None or self.last_evidence is None:
            return self.confidence
        days = max(0.0, (now - self.last_evidence).total_seconds() / 86400)
        half_life = HALF_LIFE_DAYS * (4 if self.user_confirmed else 1)
        return self.confidence * 0.5 ** (days / half_life)

    def status(self, now: datetime | None = None) -> str:
        eff = self.effective_confidence(now)
        if self.user_confirmed and eff >= ACTIVE_THRESHOLD:
            return "active"
        if eff >= ACTIVE_THRESHOLD and self.evidence_count >= MIN_EVIDENCE:
            return "active"
        if eff < FORGET_BELOW:
            return "fading"
        return "candidate"

    def to_dict(self, now: datetime | None = None) -> dict:
        d = asdict(self)
        d["first_seen"] = self.first_seen.isoformat() if self.first_seen else None
        d["last_evidence"] = self.last_evidence.isoformat() if self.last_evidence else None
        d["id"] = self.id
        d["effective_confidence"] = round(self.effective_confidence(now), 3)
        d["status"] = self.status(now)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, s: str) -> Preference:
        d = json.loads(s)
        for k in ("id", "effective_confidence", "status"):
            d.pop(k, None)
        for k in ("first_seen", "last_evidence"):
            if d.get(k):
                d[k] = datetime.fromisoformat(d[k])
        return cls(**d)
