from __future__ import annotations

from collections import Counter

from ...signals.models import Trip
from ..base import InterpreterPlugin, Observation, register


@register
class AutonomyPlugin(InterpreterPlugin):
    """Learns how the driver uses highway autonomy: engagement, gap, trust in rain."""

    name = "autonomy"
    consumes = ("autonomy_engaged", "following_gap_s", "rain_intensity", "zone")

    def interpret(self, trip: Trip) -> list[Observation]:
        out: list[Observation] = []
        highway = [f for f in trip.frames if f["zone"] == "highway"]
        if not highway:
            return out

        dry = [f for f in highway if f["rain_intensity"] == 0]
        if len(dry) >= 3:
            engaged = any(f["autonomy_engaged"] for f in dry)
            out.append(self.obs(trip, "autonomy.highway_engage", "dry", engaged, "bool", 1.0,
                                "engaged autonomy on dry highway" if engaged else "drove manually on dry highway"))

        engaged_frames = [f for f in trip.frames if f["autonomy_engaged"]]
        if engaged_frames:
            gap = Counter(f["following_gap_s"] for f in engaged_frames).most_common(1)[0][0]
            w = 1.0 if self.driver_adjusted(trip, "following_gap_s") else 0.6
            out.append(self.obs(trip, "autonomy.following_gap_s", "highway", gap, "numeric", w,
                                f"autonomous following gap {gap}s"))

        wet = [f for f in highway if f["rain_intensity"] > 0]
        handovers = [e for e in trip.events if e.kind in ("manual_takeover", "planned_handover")]
        if wet and handovers:
            out.append(self.obs(trip, "autonomy.rain_comfort", "rain", False, "bool", 1.0,
                                f"took over from autonomy when it started raining ({handovers[0].kind})"))
        elif wet and engaged_frames and all(f["autonomy_engaged"] for f in wet):
            out.append(self.obs(trip, "autonomy.rain_comfort", "rain", True, "bool", 1.0,
                                "kept autonomy engaged in rain"))
        return out
