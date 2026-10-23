from drivememory.interpreter import InterpreterPlugin, SignalInterpreter, register, registered_plugins
from drivememory.signals.simulator import PERSONAS, TripSimulator


def _trips(driver="anna", n=None):
    sim = TripSimulator(PERSONAS[driver], seed=3)
    plans = sim.plan_week(0)
    return [sim.simulate(p) for p in plans[:n]]


def test_builtin_plugins_registered():
    SignalInterpreter()
    assert {"climate", "media", "autonomy", "routine"} <= set(registered_plugins())


def test_commute_produces_expected_observations():
    trip = _trips(n=1)[0]
    keys = {(o.key, o.context) for o in SignalInterpreter().interpret(trip)}
    assert ("climate.cabin_temp_c", "*") in keys
    assert ("routine.destination", "weekday-morning") in keys
    assert ("media.source", "morning") in keys


def test_driver_adjustment_gives_stronger_evidence():
    trip = _trips(n=1)[0]
    temp = next(o for o in SignalInterpreter().interpret(trip) if o.key == "climate.cabin_temp_c")
    adjusted = any(e.signal == "cabin_temp_setpoint_c" and e.actor == "driver" for e in trip.events)
    assert temp.weight == (1.0 if adjusted else 0.6)


def test_rain_takeover_is_interpreted_as_low_trust():
    sim = TripSimulator(PERSONAS["anna"], seed=1)
    for w in range(4):
        for plan in sim.plan_week(w):
            trip = sim.simulate(plan)
            if trip.unplanned_takeovers:
                o = [x for x in SignalInterpreter().interpret(trip) if x.key == "autonomy.rain_comfort"]
                assert o and o[0].value is False
                return
    raise AssertionError("expected at least one rain takeover in 4 weeks")


def test_new_signal_plugin_can_be_added():
    @register
    class WiperPlugin(InterpreterPlugin):
        name = "wipers-test"
        consumes = ("wiper_level",)

        def interpret(self, trip):
            level = max(f["wiper_level"] for f in trip.frames)
            return [self.obs(trip, "weather.max_wiper", "*", level, "categorical")]

    try:
        obs = SignalInterpreter().interpret(_trips(n=1)[0])
        assert any(o.key == "weather.max_wiper" for o in obs)
    finally:
        from drivememory.interpreter import base
        base._REGISTRY.pop("wipers-test")
