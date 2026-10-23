from __future__ import annotations

from ...signals.models import Trip
from ..base import InterpreterPlugin, Observation, register

SETTLE_MINUTE = 5  # by then the driver has made their adjustments


@register
class ClimatePlugin(InterpreterPlugin):
    """Learns cabin temperature and weather-dependent seat heating."""

    name = "climate"
    consumes = ("cabin_temp_setpoint_c", "seat_heating_level", "outside_temp_c")

    def interpret(self, trip: Trip) -> list[Observation]:
        if not trip.frames:
            return []
        f = trip.frames[min(SETTLE_MINUTE, len(trip.frames) - 1)]
        weather = "cold" if f["outside_temp_c"] < 8 else "mild"
        temp_w = 1.0 if self.driver_adjusted(trip, "cabin_temp_setpoint_c") else 0.6
        seat_w = 1.0 if self.driver_adjusted(trip, "seat_heating_level") else 0.6
        return [
            self.obs(trip, "climate.cabin_temp_c", "*", f["cabin_temp_setpoint_c"], "numeric", temp_w,
                     f"cabin set to {f['cabin_temp_setpoint_c']}°C"),
            self.obs(trip, "climate.seat_heating", weather, int(f["seat_heating_level"]), "categorical", seat_w,
                     f"seat heating {f['seat_heating_level']} at {f['outside_temp_c']}°C outside"),
        ]
