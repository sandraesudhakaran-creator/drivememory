from datetime import datetime, timedelta

from drivememory.interpreter import Observation
from drivememory.memory import MemoryManager
from drivememory.signals.simulator import PERSONAS, TripSimulator

T0 = datetime(2026, 11, 2, 8, 0)


def obs(key, value, vtype="numeric", context="*", day=0, weight=1.0):
    return Observation(key, context, value, vtype, f"t{day}", T0 + timedelta(days=day), weight, f"saw {value}")


def test_numeric_preference_becomes_active_after_repeated_evidence():
    m = MemoryManager()
    m.observe("d", obs("climate.cabin_temp_c", 21.5, day=0))
    assert m.recall("d", "climate.cabin_temp_c") is None  # one trip is not enough
    for day in range(1, 4):
        m.observe("d", obs("climate.cabin_temp_c", 21.5, day=day))
    p = m.recall("d", "climate.cabin_temp_c")
    assert p is not None and p.value == 21.5


def test_single_outlier_is_treated_as_noise():
    m = MemoryManager()
    for day in range(6):
        m.observe("d", obs("climate.cabin_temp_c", 21.5, day=day))
    m.observe("d", obs("climate.cabin_temp_c", 26.0, day=7))  # a passenger
    assert m.recall("d", "climate.cabin_temp_c").value == 21.5


def test_consistent_change_is_detected_as_drift():
    m = MemoryManager()
    for day in range(6):
        m.observe("d", obs("climate.cabin_temp_c", 22.5, day=day))
    events = [m.observe("d", obs("climate.cabin_temp_c", 20.0, day=10 + i)) for i in range(3)]
    p = m.store.get_preference("d", "climate.cabin_temp_c", "*")
    assert events[-1] == "drift"
    assert p.value == 20.0 and p.history[-1]["value"] == 22.5


def test_categorical_habits_can_change():
    m = MemoryManager()
    for day in range(8):
        m.observe("d", obs("media.source", "jazz", "categorical", "evening", day))
    for day in range(8, 20):
        m.observe("d", obs("media.source", "audiobook", "categorical", "evening", day))
    assert m.recall("d", "media.source", "evening").value == "audiobook"


def test_unconfirmed_memories_decay_over_time():
    m = MemoryManager()
    for day in range(5):
        m.observe("d", obs("media.volume", 12, day=day))
    p = m.store.get_preference("d", "media.volume", "*")
    fresh = p.effective_confidence(T0 + timedelta(days=5))
    stale = p.effective_confidence(T0 + timedelta(days=200))
    assert stale < fresh / 4


def test_stated_preference_overrides_and_is_confirmed():
    m = MemoryManager()
    for day in range(5):
        m.observe("d", obs("climate.cabin_temp_c", 22.0, day=day))
    p = m.remember("d", "climate.cabin_temp_c", "*", 20.0, "numeric")
    assert p.value == 20.0 and p.user_confirmed and p.confidence == 1.0
    assert p.history[-1]["reason"] == "driver correction"


def test_context_fallback_and_specificity():
    m = MemoryManager()
    for day in range(5):
        m.observe("d", obs("climate.seat_heating", 2, "categorical", "cold", day))
    assert m.recall("d", "climate.seat_heating", "cold").value == 2
    assert m.recall("d", "climate.seat_heating", "mild") is None


def test_forget_domain_and_everything():
    m = MemoryManager()
    sim = TripSimulator(PERSONAS["anna"])
    for plan in sim.plan_week(0):
        m.ingest_trip(sim.simulate(plan))
    assert m.forget("anna", domain="media") > 0
    assert not m.store.preferences("anna", "media.")
    assert m.store.preferences("anna", "climate.")
    m.forget("anna")
    assert not m.store.preferences("anna") and m.store.count_episodes("anna") == 0
    assert m.store.audit_log("anna")[0]["action"] == "forget"


def test_explain_returns_provenance():
    m = MemoryManager()
    for day in range(4):
        m.observe("d", obs("autonomy.following_gap_s", 2.4, context="highway", day=day))
    ex = m.explain("d", "autonomy.following_gap_s", "highway")
    assert ex["value"] == 2.4 and ex["recent_evidence"][-1]["trip_id"] == "t3"


def test_episode_search_finds_relevant_trip():
    m = MemoryManager()
    sim = TripSimulator(PERSONAS["anna"])
    for _, plans in sim.iter_weeks(2):
        for plan in plans:
            m.ingest_trip(sim.simulate(plan))
    hit = m.search("anna", "took a break on a long trip to the countryside", k=1)[0]
    assert "countryside" in hit["summary"]
