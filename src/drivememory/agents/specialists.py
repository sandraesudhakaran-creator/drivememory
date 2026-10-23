"""Specialist agents. Each one reads the memory system and returns recommendations.

A recommendation has an ``action``:

* ``apply``   – confident enough to preset it silently (it is explained on request)
* ``suggest`` – mention it to the driver and let them decide
* ``advise``  – a safety / comfort advisory (e.g. planned handover before rain)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from ..memory import MemoryManager, Preference
from ..signals.models import TripContext, day_type
from .describe import display_value
from .protocol import Agent, AgentCard, AgentSkill, Message

APPLY_THRESHOLD = 0.7


@dataclass
class Recommendation:
    key: str
    value: Any
    reason: str
    confidence: float
    action: str
    source: str | None = None


def context_from_data(data: dict) -> TripContext:
    c = dict(data["context"])
    if isinstance(c.get("departure"), str):
        c["departure"] = datetime.fromisoformat(c["departure"])
    return TripContext(**c)


class MemoryBackedAgent(Agent):
    def __init__(self, memory: MemoryManager):
        self.memory = memory

    def _rec(self, pref: Preference, key: str, reason: str, value: Any = None) -> Recommendation:
        conf = round(pref.effective_confidence(self.memory.now(pref.driver_id)), 2)
        action = "apply" if conf >= APPLY_THRESHOLD else "suggest"
        value = display_value(key, pref.value) if value is None else value
        return Recommendation(key, value, reason, conf, action, pref.id)

    def run(self, message: Message) -> dict:
        ctx = context_from_data(message.data())
        recs = self.recommend(ctx)
        return {"agent": self.card.name, "recommendations": [asdict(r) for r in recs]}

    def recommend(self, ctx: TripContext) -> list[Recommendation]:
        raise NotImplementedError


class ComfortAgent(MemoryBackedAgent):
    card = AgentCard(
        name="comfort-agent",
        description="Personalises cabin climate, seat heating and media from the driver's memory profile.",
        skills=[AgentSkill("recommend_cabin", "Recommend cabin settings",
                           "Cabin temperature, seat heating, media source and volume for a trip context.",
                           ["climate", "media", "personalisation"])],
    )

    def recommend(self, ctx: TripContext) -> list[Recommendation]:
        d, m = ctx.driver_id, self.memory
        out: list[Recommendation] = []
        if p := m.recall(d, "climate.cabin_temp_c"):
            temp = display_value(p.key, p.value)
            out.append(self._rec(p, "climate.cabin_temp_c", f"you usually set the cabin to {temp:g}°C"))
        if p := m.recall(d, "climate.seat_heating", ctx.weather):
            out.append(self._rec(p, "climate.seat_heating",
                                 f"in {ctx.weather} weather you use seat heating level {p.value}"))
        if p := m.recall(d, "media.source", ctx.slot):
            out.append(self._rec(p, "media.source", f"you usually listen to {p.value} in the {ctx.slot}"))
        if p := m.recall(d, "media.volume"):
            out.append(self._rec(p, "media.volume", f"your usual volume is {round(p.value)}"))
        return out


class NavigationAgent(MemoryBackedAgent):
    card = AgentCard(
        name="navigation-agent",
        description="Predicts destinations from driving routines and suggests breaks on long trips.",
        skills=[
            AgentSkill("predict_destination", "Predict destination",
                       "Most likely destination for this weekday and time.", ["routine", "navigation"]),
            AgentSkill("suggest_stops", "Suggest stops", "Break suggestions for long trips.", ["routine"]),
        ],
    )

    def recommend(self, ctx: TripContext) -> list[Recommendation]:
        d, m = ctx.driver_id, self.memory
        out: list[Recommendation] = []
        contexts = [f"{ctx.weekday}-{ctx.slot}", f"{day_type(ctx.departure)}-{ctx.slot}"]
        if (ctx.expected_duration_min or 0) >= 75:  # long trips have their own routine
            contexts.insert(0, f"{day_type(ctx.departure)}-long")
        p = m.recall(d, "routine.destination", contexts)
        if p and p.value != ctx.origin:
            out.append(self._rec(p, "routine.destination",
                                 f"on {ctx.weekday} {ctx.slot}s you usually drive to {p.value}"))
        if (ctx.expected_duration_min or 0) >= 75:
            s = m.recall(d, "routine.long_trip_stop", "long-trip")
            if s and s.value is True:
                r = self._rec(s, "routine.suggest_break", "you like to take a break on long drives",
                              {"after_min": ctx.expected_duration_min // 2, "type": "cafe"})
                r.action = "suggest"
                out.append(r)
        return out


class AutonomyAgent(MemoryBackedAgent):
    card = AgentCard(
        name="autonomy-agent",
        description="Adapts highway autonomous driving to the driver's style and trust, "
                    "and plans handovers before conditions they are uncomfortable with.",
        skills=[AgentSkill("recommend_driving_style", "Recommend driving style",
                           "Following gap, autonomy offers and rain handover advisories.",
                           ["autonomous-driving", "safety", "trust"])],
    )

    def recommend(self, ctx: TripContext) -> list[Recommendation]:
        d, m = ctx.driver_id, self.memory
        out: list[Recommendation] = []
        engage = m.recall(d, "autonomy.highway_engage", "dry")
        if not engage or engage.value is not True:
            return out
        if p := m.recall(d, "autonomy.following_gap_s", "highway"):
            out.append(self._rec(p, "autonomy.following_gap_s",
                                 f"you prefer a {display_value(p.key, p.value):g}s following gap in autonomous mode"))
        rain = m.recall(d, "autonomy.rain_comfort", "rain")
        if ctx.rain_expected and rain is not None:
            if rain.value is False:
                r = self._rec(rain, "autonomy.rain_handover_advisory",
                              "rain is expected and you usually take over from autonomy in rain, "
                              "so I'll give you an early, planned handover", True)
                r.action = "advise"
                out.append(r)
            else:
                out.append(self._rec(rain, "autonomy.rain_handover_advisory",
                                     "you're comfortable keeping autonomy engaged in rain", False))
        return out
