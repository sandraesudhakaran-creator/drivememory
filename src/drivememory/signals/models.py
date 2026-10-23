"""Core data models for trips and vehicle signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def time_slot(dt: datetime) -> str:
    """Bucket a departure time into a coarse slot used as memory context."""
    h = dt.hour + dt.minute / 60
    if 5 <= h < 11:
        return "morning"
    if 11 <= h < 16:
        return "midday"
    if 16 <= h < 19.5:
        return "evening"
    return "night"


def day_type(dt: datetime) -> str:
    return "weekend" if dt.weekday() >= 5 else "weekday"


@dataclass
class TripPlan:
    """What is known about a trip *before* it starts (used to build context)."""

    trip_id: str
    driver_id: str
    departure: datetime
    origin: str
    destination: str
    duration_min: int
    outside_temp_c: float
    highway: tuple[int, int] | None = None  # (start_min, end_min)
    rain_onset_min: int | None = None  # minute rain starts, None = dry trip
    rest_stop: tuple[int, int] | None = None  # (start_min, end_min) of a stop


@dataclass
class TripContext:
    """The context the assistant sees when a trip starts."""

    driver_id: str
    departure: datetime
    outside_temp_c: float
    origin: str = "unknown"
    rain_expected: bool = False
    expected_duration_min: int | None = None

    @property
    def slot(self) -> str:
        return time_slot(self.departure)

    @property
    def weekday(self) -> str:
        return WEEKDAYS[self.departure.weekday()]

    @property
    def weather(self) -> str:
        return "cold" if self.outside_temp_c < 8 else "mild"

    @classmethod
    def from_plan(cls, plan: TripPlan) -> TripContext:
        return cls(
            driver_id=plan.driver_id,
            departure=plan.departure,
            outside_temp_c=plan.outside_temp_c,
            origin=plan.origin,
            rain_expected=plan.rain_onset_min is not None,
            expected_duration_min=plan.duration_min,
        )


@dataclass
class DriverEvent:
    """A discrete action during a trip (a manual adjustment, a takeover, ...)."""

    minute: int
    kind: str  # "adjust" | "manual_takeover" | "planned_handover" | "engage_autonomy"
    signal: str | None = None
    old: Any = None
    new: Any = None
    actor: str = "driver"  # "driver" | "assistant"


@dataclass
class Trip:
    """A completed trip: per-minute signal frames plus discrete events."""

    trip_id: str
    driver_id: str
    departure: datetime
    origin: str
    destination: str
    frames: list[dict[str, Any]] = field(default_factory=list)
    events: list[DriverEvent] = field(default_factory=list)

    @property
    def duration_min(self) -> int:
        return len(self.frames)

    @property
    def manual_adjustments(self) -> int:
        return sum(1 for e in self.events if e.actor == "driver" and e.kind == "adjust")

    @property
    def unplanned_takeovers(self) -> int:
        return sum(1 for e in self.events if e.kind == "manual_takeover")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["departure"] = self.departure.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Trip:
        return cls(
            trip_id=d["trip_id"],
            driver_id=d["driver_id"],
            departure=datetime.fromisoformat(d["departure"]),
            origin=d["origin"],
            destination=d["destination"],
            frames=list(d.get("frames", [])),
            events=[DriverEvent(**e) for e in d.get("events", [])],
        )
