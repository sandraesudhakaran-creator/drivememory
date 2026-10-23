"""Persona-driven vehicle signal simulator.

Each ``Persona`` has *hidden* ground-truth preferences. The simulator generates
weeks of realistic trips (commutes, gym visits, weekend drives) as per-minute
signal frames. Drivers fix any setting that does not match their preference,
so if the assistant presets the cabin correctly, fewer manual adjustments
happen. This lets us measure the system end-to-end.

Noise is built in: guests change the temperature, drivers sometimes pick a
different playlist, and preferences can drift over time (``Drift``).
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any

from .models import WEEKDAYS, DriverEvent, Trip, TripPlan, time_slot

DEFAULTS = {
    "cabin_temp_setpoint_c": 22.0,
    "seat_heating_level": 0,
    "media_source": "radio",
    "media_volume": 10,
    "following_gap_s": 1.8,
}

MEDIA_OPTIONS = ["radio", "news_podcast", "jazz", "pop", "audiobook", "techno", "classical"]


@dataclass
class Drift:
    """A preference change that happens at a given week (e.g. new habit)."""

    week: int
    field: str
    value: Any


@dataclass
class Persona:
    driver_id: str
    name: str
    preferred_temp_c: float
    seat_heating_when_cold: int
    morning_media: str
    evening_media: str
    media_volume: int
    uses_highway_autonomy: bool
    preferred_gap_s: float
    comfortable_with_autonomy_in_rain: bool
    gym_days: tuple[str, ...] = ()
    coffee_stop_on_long_trips: bool = False
    noise: float = 0.1
    drifts: list[Drift] = field(default_factory=list)

    def ground_truth(self) -> list[tuple[str, str, Any]]:
        """Preferences the memory system *should* learn: (key, context, value)."""
        truth: list[tuple[str, str, Any]] = [
            ("climate.cabin_temp_c", "*", self.preferred_temp_c),
            ("climate.seat_heating", "cold", self.seat_heating_when_cold),
            ("climate.seat_heating", "mild", 0),
            ("media.source", "morning", self.morning_media),
            ("media.source", "evening", self.evening_media),
            ("media.volume", "*", self.media_volume),
            ("routine.destination", "weekday-morning", "work"),
            ("routine.long_trip_stop", "long-trip", self.coffee_stop_on_long_trips),
        ]
        for day in self.gym_days:
            slot = "evening" if day not in ("sat", "sun") else "morning"
            truth.append(("routine.destination", f"{day}-{slot}", "gym"))
        truth.append(("autonomy.highway_engage", "dry", self.uses_highway_autonomy))
        if self.uses_highway_autonomy:
            truth.append(("autonomy.following_gap_s", "highway", self.preferred_gap_s))
            truth.append(("autonomy.rain_comfort", "rain", self.comfortable_with_autonomy_in_rain))
        return truth

    def at_week(self, week: int) -> Persona:
        """Return the persona with all drifts up to ``week`` applied."""
        p = self
        for d in self.drifts:
            if week >= d.week:
                p = replace(p, **{d.field: d.value})
        return p


PERSONAS: dict[str, Persona] = {
    p.driver_id: p
    for p in [
        Persona(
            "anna", "Anna (commuter, cautious in rain)", 21.5, 2, "news_podcast", "jazz", 12,
            True, 2.4, False, gym_days=("tue", "thu"), coffee_stop_on_long_trips=True,
        ),
        Persona(
            "ben", "Ben (sporty, trusts autonomy)", 19.5, 0, "techno", "pop", 18,
            True, 1.2, True, gym_days=("mon", "wed", "fri"),
        ),
        Persona(
            "chen", "Chen (no autonomy, likes it warm)", 23.5, 3, "classical", "audiobook", 8,
            False, 1.8, False, gym_days=(), coffee_stop_on_long_trips=True,
        ),
        Persona(
            "dara", "Dara (noisy shared car)", 20.5, 1, "pop", "news_podcast", 14,
            True, 2.0, True, gym_days=("wed",), noise=0.25,
        ),
        Persona(
            "emil", "Emil (preference drift at week 4)", 22.5, 2, "jazz", "jazz", 11,
            True, 2.8, False, gym_days=("sat",), coffee_stop_on_long_trips=False,
            drifts=[Drift(4, "preferred_temp_c", 20.0), Drift(4, "evening_media", "audiobook")],
        ),
    ]
}


class TripSimulator:
    """Generates trip plans and simulates the driver's behaviour during them."""

    def __init__(self, persona: Persona, seed: int = 7, start: datetime | None = None):
        self.persona = persona
        self.rng = random.Random(seed)
        # Start on a Monday in late autumn, Munich-like weather.
        self.start = start or datetime(2026, 11, 2, 0, 0)
        self._trip_counter = 0

    # ------------------------------------------------------------------ plans
    def plan_week(self, week: int) -> list[TripPlan]:
        p = self.persona
        plans: list[TripPlan] = []
        for d in range(7):
            day = self.start + timedelta(days=7 * week + d)
            dname = WEEKDAYS[day.weekday()]
            temp = round(self.rng.gauss(6.5, 5.0), 1)
            if day.weekday() < 5:
                plans.append(self._plan(day, 7, 30, 30, "home", "work", 35, temp, highway=True))
                if dname in p.gym_days:
                    plans.append(self._plan(day, 17, 15, 30, "work", "gym", 20, temp + 1))
                    plans.append(self._plan(day, 20, 0, 30, "gym", "home", 20, temp - 1))
                else:
                    plans.append(self._plan(day, 17, 0, 60, "work", "home", 35, temp + 1, highway=True))
            else:
                if dname in p.gym_days:
                    plans.append(self._plan(day, 9, 30, 30, "home", "gym", 20, temp))
                if self.rng.random() < 0.6:
                    plans.append(
                        self._plan(day, 11, 0, 90, "home", "countryside", self.rng.randint(80, 140), temp,
                                   highway=True, long=True)
                    )
                plans.append(self._plan(day, 15, 0, 60, "home", "city_centre", 25, temp + 2))
        return plans

    def _plan(self, day, hour, minute, jitter, origin, dest, dur, temp, highway=False, long=False) -> TripPlan:
        self._trip_counter += 1
        dep = day.replace(hour=hour, minute=minute) + timedelta(minutes=self.rng.randint(0, jitter))
        dur = max(8, dur + self.rng.randint(-5, 5))
        hw = (int(dur * 0.2), int(dur * 0.8)) if highway else None
        rain = None
        if self.rng.random() < 0.35:
            # Rain starts sometime during the trip (or before it, which is minute 0).
            rain = max(0, self.rng.randint(-5, int(dur * 0.6)))
        stop = None
        if long and self.persona.at_week(self._week_of(dep)).coffee_stop_on_long_trips:
            mid = dur // 2
            stop = (mid, mid + 12)
            dur += 12
            if hw:
                hw = (hw[0], hw[1] + 12)
        return TripPlan(
            trip_id=f"{self.persona.driver_id}-{self._trip_counter:04d}",
            driver_id=self.persona.driver_id,
            departure=dep,
            origin=origin,
            destination=dest,
            duration_min=dur,
            outside_temp_c=temp,
            highway=hw,
            rain_onset_min=rain,
            rest_stop=stop,
        )

    def _week_of(self, dt: datetime) -> int:
        return max(0, (dt - self.start).days // 7)

    def iter_weeks(self, weeks: int) -> Iterator[tuple[int, list[TripPlan]]]:
        for w in range(weeks):
            yield w, self.plan_week(w)

    # ------------------------------------------------------------ simulation
    def simulate(self, plan: TripPlan, presets: dict[str, Any] | None = None) -> Trip:
        """Simulate one trip. ``presets`` are settings the assistant applied at start."""
        p = self.persona.at_week(self._week_of(plan.departure))
        rng = self.rng
        presets = presets or {}
        events: list[DriverEvent] = []

        # Initial vehicle state: factory defaults, overridden by assistant presets.
        state = dict(DEFAULTS)
        preset_map = {
            "climate.cabin_temp_c": "cabin_temp_setpoint_c",
            "climate.seat_heating": "seat_heating_level",
            "media.source": "media_source",
            "media.volume": "media_volume",
            "autonomy.following_gap_s": "following_gap_s",
        }
        for key, sig in preset_map.items():
            if key in presets and presets[key] is not None:
                state[sig] = presets[key]
                events.append(DriverEvent(0, "adjust", sig, DEFAULTS[sig], presets[key], actor="assistant"))

        # What the driver actually wants on this trip (with noise).
        slot = time_slot(plan.departure)
        want_temp = p.preferred_temp_c
        if rng.random() < p.noise:  # a passenger / guest changes it
            want_temp = rng.choice([18.0, 19.0, 24.0, 25.0, 26.0])
        want_seat = p.seat_heating_when_cold if plan.outside_temp_c < 8 else 0
        want_media = {"morning": p.morning_media, "evening": p.evening_media}.get(slot, "radio")
        if rng.random() < p.noise:
            want_media = rng.choice(MEDIA_OPTIONS)
        want_volume = p.media_volume + (rng.randint(-4, 4) if rng.random() < p.noise else 0)

        def adjust(minute: int, sig: str, target: Any, tol: float = 0.0):
            cur = state[sig]
            differs = abs(cur - target) > tol if isinstance(target, (int, float)) else cur != target
            if differs:
                events.append(DriverEvent(minute, "adjust", sig, cur, target))
                state[sig] = target

        rain_comfort = p.comfortable_with_autonomy_in_rain
        if rng.random() < p.noise / 2:
            rain_comfort = not rain_comfort
        advisory = bool(presets.get("autonomy.rain_handover_advisory"))

        frames: list[dict[str, Any]] = []
        engaged = False
        took_over_in_rain = False
        for m in range(plan.duration_min):
            # Drivers fix settings in the first minutes of a trip.
            if m == 1:
                adjust(m, "cabin_temp_setpoint_c", want_temp, tol=0.25)
                adjust(m, "media_source", want_media)
            if m == 2:
                adjust(m, "seat_heating_level", want_seat)
                adjust(m, "media_volume", want_volume, tol=1)

            raining = plan.rain_onset_min is not None and m >= plan.rain_onset_min
            rain = round(min(1.0, 0.5 + 0.1 * (m - plan.rain_onset_min)), 2) if raining else 0.0
            on_highway = plan.highway is not None and plan.highway[0] <= m < plan.highway[1]
            stopped = plan.rest_stop is not None and plan.rest_stop[0] <= m < plan.rest_stop[1]
            if stopped:
                on_highway = False

            # Highway autonomy behaviour.
            if p.uses_highway_autonomy and on_highway and not engaged and not took_over_in_rain:
                if not (raining and not rain_comfort):
                    engaged = True
                    events.append(DriverEvent(m, "engage_autonomy"))
                    adjust(m, "following_gap_s", p.preferred_gap_s, tol=0.15)
            if engaged and raining and not rain_comfort:
                kind = "planned_handover" if advisory else "manual_takeover"
                events.append(DriverEvent(m, kind, "autonomy_engaged", True, False))
                engaged = False
                took_over_in_rain = True
            if engaged and not on_highway:
                engaged = False

            if stopped:
                zone, speed = "rest_area", 0.0
            elif m == plan.duration_min - 1:
                zone, speed = plan.destination, 0.0
            elif m == 0:
                zone, speed = plan.origin, 5.0
            elif on_highway:
                zone, speed = "highway", round(rng.uniform(95, 130), 1)
            else:
                zone, speed = "city", round(rng.uniform(20, 55), 1)

            frames.append({
                "t": m,
                "speed_kmh": speed,
                "zone": zone,
                "outside_temp_c": plan.outside_temp_c,
                "cabin_temp_setpoint_c": state["cabin_temp_setpoint_c"],
                "seat_heating_level": state["seat_heating_level"],
                "media_source": state["media_source"],
                "media_volume": state["media_volume"],
                "rain_intensity": rain,
                "wiper_level": 0 if rain == 0 else (1 if rain < 0.7 else 2),
                "autonomy_engaged": engaged,
                "following_gap_s": state["following_gap_s"],
            })

        return Trip(
            trip_id=plan.trip_id,
            driver_id=plan.driver_id,
            departure=plan.departure,
            origin=plan.origin,
            destination=plan.destination,
            frames=frames,
            events=events,
        )
