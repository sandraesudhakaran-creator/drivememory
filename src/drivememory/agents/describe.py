"""Human-readable, second-person descriptions of preferences (used in chat and LLM grounding)."""

from __future__ import annotations

from ..memory import Preference

RESOLUTION = {"climate.cabin_temp_c": 0.5, "autonomy.following_gap_s": 0.1, "media.volume": 1.0}


def display_value(key: str, value):
    """Round a learned numeric value to what the vehicle can actually set."""
    step = RESOLUTION.get(key)
    if step is None or not isinstance(value, (int, float)) or isinstance(value, bool):
        return value
    v = round(round(value / step) * step, 2)
    return int(v) if step >= 1 else v


def describe(p: Preference) -> str:
    k, c, v = p.key, p.context, display_value(p.key, p.value)
    if k == "climate.cabin_temp_c":
        return f"you like the cabin at {v:g}°C"
    if k == "climate.seat_heating":
        return f"you use seat heating level {v} in {c} weather" if v else f"you don't use seat heating in {c} weather"
    if k == "media.source":
        when = {"morning": "in the morning", "evening": "in the evening", "midday": "at midday",
                "night": "at night"}.get(c, c)
        return f"you listen to {str(v).replace('_', ' ')} {when}"
    if k == "media.volume":
        return f"you keep the volume around {round(v)}"
    if k == "autonomy.highway_engage":
        return "you use highway autonomy" if v else "you prefer to drive the highway manually"
    if k == "autonomy.following_gap_s":
        return f"you prefer a {v:g}s following gap in autonomous mode"
    if k == "autonomy.rain_comfort":
        return "you're comfortable with autonomy in rain" if v else "you prefer to take over from autonomy in rain"
    if k == "routine.destination":
        dest = "home" if v == "home" else f"to {v.replace('_', ' ')}"
        dest = "to work" if v == "work" else dest
        return f"you usually drive {dest} ({c.replace('-', ' ')})"
    if k == "routine.long_trip_stop":
        return "you like a break on long trips" if v else "you drive long trips without breaks"
    return f"{k} ({c}) = {v}"


KEYWORDS = {
    "climate.cabin_temp_c": ["temperature", "temp", "warm", "cold", "degrees", "heat up", "cool"],
    "climate.seat_heating": ["seat"],
    "media.source": ["music", "podcast", "radio", "media", "playing", "jazz", "audiobook", "song"],
    "media.volume": ["volume", "loud", "quiet"],
    "autonomy.following_gap_s": ["gap", "distance", "following"],
    "autonomy.rain_handover_advisory": ["rain", "handover", "take over", "takeover"],
    "routine.destination": ["destination", "route", "where", "navigate", "going"],
    "routine.suggest_break": ["break", "stop", "coffee", "cafe"],
}


def key_from_text(text: str) -> str | None:
    t = text.lower()
    for key, words in KEYWORDS.items():
        if any(w in t for w in words):
            return key
    return None
