from __future__ import annotations

from ...signals.models import WEEKDAYS, Trip, day_type, time_slot
from ..base import InterpreterPlugin, Observation, register

LONG_TRIP_MIN = 75
MIN_STOP_MIN = 8


@register
class RoutinePlugin(InterpreterPlugin):
    """Learns where the driver goes when, and whether they take breaks on long trips."""

    name = "routine"
    consumes = ("zone", "speed_kmh")

    def interpret(self, trip: Trip) -> list[Observation]:
        if not trip.frames:
            return []
        dest = trip.frames[-1]["zone"]
        slot = time_slot(trip.departure)
        day = WEEKDAYS[trip.departure.weekday()]
        out = [
            self.obs(trip, "routine.destination", f"{day_type(trip.departure)}-{slot}", dest, "categorical", 0.8,
                     f"drove {trip.origin} → {dest}"),
            self.obs(trip, "routine.destination", f"{day}-{slot}", dest, "categorical", 1.0),
        ]
        if trip.duration_min >= LONG_TRIP_MIN:
            out.append(self.obs(trip, "routine.destination", f"{day_type(trip.departure)}-long", dest,
                                "categorical", 1.0))
            run = best = 0
            for f in trip.frames:
                run = run + 1 if f["zone"] == "rest_area" and f["speed_kmh"] == 0 else 0
                best = max(best, run)
            stopped = best >= MIN_STOP_MIN
            out.append(self.obs(trip, "routine.long_trip_stop", "long-trip", stopped, "bool", 1.0,
                                f"{f'took a {best}-min break' if stopped else 'no break'} on a "
                                f"{trip.duration_min}-min trip"))
        return out
