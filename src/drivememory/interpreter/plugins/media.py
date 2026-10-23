from __future__ import annotations

from collections import Counter
from statistics import median

from ...signals.models import Trip, time_slot
from ..base import InterpreterPlugin, Observation, register


@register
class MediaPlugin(InterpreterPlugin):
    """Learns what the driver listens to at different times of day, and how loud."""

    name = "media"
    consumes = ("media_source", "media_volume")

    def interpret(self, trip: Trip) -> list[Observation]:
        frames = trip.frames[3:] or trip.frames
        if not frames:
            return []
        source = Counter(f["media_source"] for f in frames).most_common(1)[0][0]
        volume = float(median(f["media_volume"] for f in frames))
        slot = time_slot(trip.departure)
        src_w = 1.0 if self.driver_adjusted(trip, "media_source") else 0.6
        vol_w = 1.0 if self.driver_adjusted(trip, "media_volume") else 0.6
        return [
            self.obs(trip, "media.source", slot, source, "categorical", src_w, f"listened to {source} ({slot})"),
            self.obs(trip, "media.volume", "*", volume, "numeric", vol_w, f"volume {volume:g}"),
        ]
