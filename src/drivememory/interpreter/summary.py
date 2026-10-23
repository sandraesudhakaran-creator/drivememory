"""Turns a trip and its observations into a short natural-language episode."""

from __future__ import annotations

from ..signals.models import WEEKDAYS, Trip
from .base import Observation


def summarize_trip(trip: Trip, observations: list[Observation]) -> str:
    day = WEEKDAYS[trip.departure.weekday()].capitalize()
    first = trip.frames[0] if trip.frames else {}
    rain = any(f["rain_intensity"] > 0 for f in trip.frames)
    parts = [
        f"{day} {trip.departure:%H:%M}: {trip.origin} → {trip.destination}, {trip.duration_min} min, "
        f"{first.get('outside_temp_c', '?')}°C outside{', rain' if rain else ''}."
    ]
    parts += [o.evidence[0].upper() + o.evidence[1:] + "." for o in observations if o.evidence]
    if trip.manual_adjustments:
        parts.append(f"Driver made {trip.manual_adjustments} manual adjustment(s).")
    return " ".join(parts)
