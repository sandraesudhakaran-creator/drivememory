"""Catalog of the vehicle signals DriveMemory understands.

Each entry documents a signal that appears in a trip's per-minute frames.
To teach the system a new signal: add a ``SignalSpec`` here, emit it from the
simulator (or your real data source) and write an interpreter plugin for it
(see ``drivememory.interpreter.plugins``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SignalSpec:
    name: str
    unit: str
    dtype: str  # "float" | "int" | "bool" | "str"
    description: str
    interpreted_by: str  # name of the interpreter plugin that consumes it

    def to_dict(self) -> dict:
        return asdict(self)


SIGNAL_CATALOG: dict[str, SignalSpec] = {
    s.name: s
    for s in [
        SignalSpec("speed_kmh", "km/h", "float", "Vehicle speed", "routine"),
        SignalSpec("zone", "-", "str", "Semantic location zone (home, work, city, highway, rest_area, ...)", "routine"),
        SignalSpec("outside_temp_c", "°C", "float", "Ambient temperature outside the vehicle", "climate"),
        SignalSpec("cabin_temp_setpoint_c", "°C", "float", "Climate control target temperature", "climate"),
        SignalSpec("seat_heating_level", "0-3", "int", "Driver seat heating level", "climate"),
        SignalSpec("media_source", "-", "str", "Active media source / content category", "media"),
        SignalSpec("media_volume", "0-30", "int", "Media volume", "media"),
        SignalSpec("rain_intensity", "0-1", "float", "Rain sensor intensity", "autonomy"),
        SignalSpec("wiper_level", "0-3", "int", "Wiper stage", "autonomy"),
        SignalSpec("autonomy_engaged", "-", "bool", "Highway autonomous driving engaged", "autonomy"),
        SignalSpec("following_gap_s", "s", "float", "Autonomous following gap (time headway)", "autonomy"),
    ]
}
