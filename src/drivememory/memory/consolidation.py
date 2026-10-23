"""Consolidation: folds new observations into preferences.

The rules are deliberately simple and explainable:

* **Numeric** values (temperature, volume, following gap) keep a running
  estimate. Observations far from it are treated as noise *unless* several
  recent ones agree with each other, which signals a real change (drift). The
  old value then moves to ``history``.
* **Categorical / bool** values keep exponentially decayed counts, so recent
  behaviour matters more and habits can change.
* Confidence grows with evidence and shrinks with inconsistency. Stated
  preferences ("I like 20 degrees") are trusted more than observed ones.
"""

from __future__ import annotations

import json
import math
from statistics import mean
from typing import Any

from ..interpreter.base import Observation
from .models import Preference

NUMERIC_TOLERANCE = {
    "climate.cabin_temp_c": 0.5,
    "media.volume": 2.0,
    "autonomy.following_gap_s": 0.25,
}
COUNT_DECAY = 0.92  # categorical counts shrink by this factor per new observation
DRIFT_WINDOW = 3  # agreeing outliers needed to accept a change
DRIFT_WINDOW_CONFIRMED = 5  # ... when the driver explicitly stated the old value
MAX_SOURCES = 8


def tolerance(key: str) -> float:
    return NUMERIC_TOLERANCE.get(key, 1.0)


def _saturation(n: float) -> float:
    return 1.0 - math.exp(-n / 3.0)


def _record_source(pref: Preference, obs: Observation) -> None:
    pref.sources.append({
        "trip_id": obs.trip_id,
        "timestamp": obs.timestamp.isoformat(),
        "value": obs.value,
        "evidence": obs.evidence,
        "source": obs.source,
    })
    del pref.sources[:-MAX_SOURCES]


def _change_value(pref: Preference, new_value: Any, obs: Observation, reason: str) -> None:
    pref.history.append({
        "value": pref.value,
        "until": obs.timestamp.isoformat(),
        "reason": reason,
    })
    pref.value = new_value


def update_numeric(pref: Preference, obs: Observation) -> str | None:
    x = float(obs.value)
    w = obs.weight
    tol = tolerance(pref.key)
    pending: list[float] = pref.state.setdefault("pending", [])
    event = None

    if pref.evidence_count == 0:
        pref.value = round(x, 2)
        pref.evidence_count = w
    elif abs(x - pref.value) <= tol:
        pref.evidence_count += w
        alpha = max(0.15, w / pref.evidence_count)
        pref.value = round(pref.value + alpha * (x - pref.value), 2)
        pending.clear()
        pref.noise_count *= 0.9
    else:
        pending.append(x)
        del pending[:-DRIFT_WINDOW_CONFIRMED]
        pref.noise_count += w
        window = DRIFT_WINDOW_CONFIRMED if pref.user_confirmed else DRIFT_WINDOW
        recent = pending[-window:]
        if len(recent) == window and max(recent) - min(recent) <= tol:
            _change_value(pref, round(mean(recent), 2), obs, "drift: behaviour changed consistently")
            pref.evidence_count = float(window)
            pref.noise_count = 0.0
            pref.user_confirmed = False
            pending.clear()
            event = "drift"

    n, noise = pref.evidence_count, pref.noise_count
    pref.confidence = round(_saturation(n) * n / (n + noise), 4)
    return event


def update_categorical(pref: Preference, obs: Observation) -> str | None:
    counts: dict[str, float] = pref.state.setdefault("counts", {})
    for k in counts:
        counts[k] *= COUNT_DECAY
    key = json.dumps(obs.value)
    counts[key] = counts.get(key, 0.0) + obs.weight
    pref.evidence_count += obs.weight

    best_key = max(counts, key=counts.get)
    best = json.loads(best_key)
    event = None
    if pref.value is None:
        pref.value = best
    elif best != pref.value:
        _change_value(pref, best, obs, "drift: a different choice became dominant")
        pref.user_confirmed = False
        event = "drift"
    share = counts[best_key] / sum(counts.values())
    pref.noise_count = round(sum(v for k, v in counts.items() if k != best_key), 3)
    pref.confidence = round(share * _saturation(pref.evidence_count), 4)
    return event


def consolidate(pref: Preference, obs: Observation) -> str | None:
    """Apply one observation to a preference. Returns "drift" if the value changed."""
    if pref.first_seen is None:
        pref.first_seen = obs.timestamp
    pref.last_evidence = max(pref.last_evidence or obs.timestamp, obs.timestamp)
    _record_source(pref, obs)

    if obs.source == "stated":
        if pref.value is not None and pref.value != obs.value:
            _change_value(pref, obs.value, obs, "driver correction")
        pref.value = obs.value
        pref.user_confirmed = True
        pref.evidence_count = max(pref.evidence_count, 3.0) + 1
        pref.confidence = 1.0
        pref.state = {}
        if pref.value_type != "numeric":
            pref.state["counts"] = {json.dumps(obs.value): 5.0}
        return "stated"

    if pref.value_type == "numeric":
        return update_numeric(pref, obs)
    return update_categorical(pref, obs)
