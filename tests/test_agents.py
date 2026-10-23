from drivememory.agents import Message, Orchestrator, Part
from drivememory.memory import MemoryManager
from drivememory.runtime import simulate_driver
from drivememory.signals.models import TripContext
from drivememory.signals.simulator import PERSONAS, TripSimulator


def _trained(driver="anna", weeks=4):
    memory = MemoryManager()
    orch = Orchestrator(memory, use_llm=False)
    simulate_driver(memory, orch, driver, weeks)
    return memory, orch


def _ctx(driver, rain=False, temp=3.0, duration=35, weekday_offset=0):
    sim = TripSimulator(PERSONAS[driver])
    plan = sim.plan_week(5)[weekday_offset]
    return TripContext(driver, plan.departure, temp, "home", rain, duration)


def test_trip_start_presets_learned_preferences():
    _, orch = _trained()
    res = orch.on_trip_start(_ctx("anna"))
    assert res.presets["climate.cabin_temp_c"] == 21.5
    assert res.presets["climate.seat_heating"] == 2
    assert res.presets["autonomy.following_gap_s"] == 2.4
    assert all(t["state"] == "completed" for t in res.tasks)


def test_rain_advisory_only_for_drivers_who_distrust_autonomy_in_rain():
    _, orch_anna = _trained("anna", 6)
    _, orch_ben = _trained("ben", 6)
    assert orch_anna.on_trip_start(_ctx("anna", rain=True)).presets.get("autonomy.rain_handover_advisory") is True
    assert orch_ben.on_trip_start(_ctx("ben", rain=True)).presets.get("autonomy.rain_handover_advisory") is not True


def test_no_autonomy_recommendations_for_manual_driver():
    _, orch = _trained("chen")
    res = orch.on_trip_start(_ctx("chen", rain=True))
    assert not any(k.startswith("autonomy.") for k in res.presets)


def test_cold_start_keeps_defaults():
    orch = Orchestrator(MemoryManager(), use_llm=False)
    res = orch.on_trip_start(_ctx("anna"))
    assert res.presets == {} and "defaults" in res.greeting


def test_assisted_driving_reduces_manual_adjustments():
    m1, o1 = MemoryManager(), None
    o1 = Orchestrator(m1, use_llm=False)
    base = simulate_driver(m1, o1, "ben", 4, assisted=False)
    m2 = MemoryManager()
    o2 = Orchestrator(m2, use_llm=False)
    assisted = simulate_driver(m2, o2, "ben", 4, assisted=True)
    assert assisted["manual_adjustments"] < base["manual_adjustments"] * 0.5


def test_chat_intents():
    memory, orch = _trained()
    orch.on_trip_start(_ctx("anna"))
    assert orch.chat("anna", "why is it so warm?")["intent"] == "explain"
    assert "21.5" in orch.chat("anna", "why is it so warm?")["reply"]
    assert orch.chat("anna", "what do you know about me?")["intent"] == "profile"
    r = orch.chat("anna", "I prefer 20 degrees")
    assert r["intent"] == "remember" and memory.recall("anna", "climate.cabin_temp_c").value == 20.0
    r = orch.chat("anna", "I don't like autonomy in rain, I'd rather take over")
    assert r["data"]["value"] is False
    r = orch.chat("anna", "forget my music")
    assert r["intent"] == "forget" and not memory.store.preferences("anna", "media.")
    assert orch.chat("anna", "tell me a joke")["intent"] == "help"


def test_a2a_message_roundtrip_and_failure():
    _, orch = _trained()
    ctx = {"driver_id": "anna", "departure": "2026-12-07T07:40:00", "outside_temp_c": 2.0, "rain_expected": True}
    msg = Message.from_dict(Message("user", [Part.of_data({"context": ctx})]).to_dict())
    task = orch.bus.send("comfort-agent", msg)
    assert task.state == "completed"
    assert any(r["key"] == "climate.cabin_temp_c" for r in task.result()["recommendations"])
    bad = orch.bus.send("comfort-agent", Message("user", [Part.of_text("hi")]))
    assert bad.state == "failed" and bad.error
