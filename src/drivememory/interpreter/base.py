"""Plugin framework that turns raw vehicle signals into semantic observations.

An *observation* is a single piece of evidence about the driver, e.g.
"on this trip, in cold weather, the driver chose seat heating level 2".
The memory system later consolidates many observations into preferences.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Literal

from ..signals.models import Trip

ValueType = Literal["numeric", "categorical", "bool"]


@dataclass
class Observation:
    key: str  # e.g. "climate.cabin_temp_c"
    context: str  # e.g. "cold", "morning", "*" (context-free)
    value: Any
    value_type: ValueType
    trip_id: str
    timestamp: datetime
    weight: float = 1.0  # 1.0 = explicit driver action, lower = implicit
    evidence: str = ""
    source: str = "observed"  # "observed" | "stated" (driver told the assistant)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


class InterpreterPlugin(ABC):
    """Base class for signal interpreters. Subclass and decorate with ``@register``."""

    name: str = "base"
    consumes: tuple[str, ...] = ()

    @abstractmethod
    def interpret(self, trip: Trip) -> list[Observation]:
        ...

    # Helpers shared by plugins ------------------------------------------------
    @staticmethod
    def driver_adjusted(trip: Trip, signal: str) -> bool:
        return any(e.actor == "driver" and e.kind == "adjust" and e.signal == signal for e in trip.events)

    def obs(self, trip: Trip, key: str, context: str, value: Any, vtype: ValueType,
            weight: float = 1.0, evidence: str = "") -> Observation:
        return Observation(key, context, value, vtype, trip.trip_id, trip.departure, weight, evidence)


_REGISTRY: dict[str, type[InterpreterPlugin]] = {}


def register(cls: type[InterpreterPlugin]) -> type[InterpreterPlugin]:
    _REGISTRY[cls.name] = cls
    return cls


def registered_plugins() -> dict[str, type[InterpreterPlugin]]:
    return dict(_REGISTRY)


class SignalInterpreter:
    """Runs every registered plugin over a trip."""

    def __init__(self, plugins: list[InterpreterPlugin] | None = None):
        if plugins is None:
            from . import plugins as _  # noqa: F401  (import registers built-ins)

            plugins = [cls() for cls in _REGISTRY.values()]
        self.plugins = plugins

    def interpret(self, trip: Trip) -> list[Observation]:
        out: list[Observation] = []
        for plugin in self.plugins:
            out.extend(plugin.interpret(trip))
        return out
