import asyncio

from fastapi.testclient import TestClient

from drivememory.api.app import create_app
from drivememory.signals.simulator import PERSONAS, TripSimulator


def _client(tmp_path):
    return TestClient(create_app(str(tmp_path / "api.db")))


def test_rest_flow(tmp_path):
    c = _client(tmp_path)
    assert c.get("/health").json()["status"] == "ok"
    assert len(c.get("/signals").json()) >= 10
    assert c.post("/drivers/ben/simulate?weeks=3").json()["trips"] > 30
    prefs = c.get("/drivers/ben/preferences").json()
    assert any(p["id"] == "climate.cabin_temp_c@*" for p in prefs)
    res = c.post("/drivers/ben/trip-start",
                 json={"outside_temp_c": 2, "departure": "2026-11-30T07:40:00"}).json()
    assert res["presets"]["climate.cabin_temp_c"] == 19.5
    assert c.post("/drivers/ben/chat", json={"text": "why this gap?"}).json()["intent"] == "explain"
    assert c.get("/drivers/ben/preferences/climate.cabin_temp_c").json()["value"] > 19
    assert c.delete("/drivers/ben/memories?domain=media").json()["deleted"] > 0
    assert c.get("/drivers/ben/audit").json()[0]["action"] == "forget"


def test_ingest_raw_trip(tmp_path):
    c = _client(tmp_path)
    sim = TripSimulator(PERSONAS["anna"])
    trip = sim.simulate(sim.plan_week(0)[0]).to_dict()
    r = c.post("/drivers/anna/trips", json=trip).json()
    assert r["observations"] and "home → work" in r["summary"]
    assert c.post("/drivers/anna/trips", json={"bad": 1}).status_code == 422


def test_graphql(tmp_path):
    c = _client(tmp_path)
    c.post("/drivers/chen/simulate?weeks=3")
    q = '{ driver(id: "chen") { preferences(domain: "climate") { id value status description } } }'
    data = c.post("/graphql", json={"query": q}).json()["data"]["driver"]["preferences"]
    assert any(p["id"] == "climate.cabin_temp_c@*" and abs(p["value"] - 23.5) <= 0.5 for p in data)
    m = 'mutation { forget(driverId: "chen", domain: "climate") }'
    assert c.post("/graphql", json={"query": m}).json()["data"]["forget"] > 0


def test_a2a_endpoints(tmp_path):
    c = _client(tmp_path)
    c.post("/drivers/anna/simulate?weeks=3")
    cards = c.get("/a2a/agents").json()
    assert {a["name"] for a in cards} == {"comfort-agent", "navigation-agent", "autonomy-agent"}
    assert c.get("/a2a/autonomy-agent/.well-known/agent-card.json").json()["skills"]
    rpc = {"jsonrpc": "2.0", "id": 7, "method": "message/send", "params": {"message": {
        "role": "user", "parts": [{"kind": "data", "data": {"context": {
            "driver_id": "anna", "departure": "2026-11-23T07:40:00", "outside_temp_c": 3}}}]}}}
    r = c.post("/a2a/comfort-agent", json=rpc).json()
    assert r["id"] == 7 and r["result"]["status"]["state"] == "completed"
    assert c.post("/a2a/comfort-agent", json={**rpc, "method": "nope"}).json()["error"]["code"] == -32601


def test_mcp_servers_expose_tools():
    from drivememory.mcp_servers.memory_server import mcp as memory_mcp
    from drivememory.mcp_servers.vehicle_server import mcp as vehicle_mcp

    async def names(server):
        return {t.name for t in await server.list_tools()}

    assert {"get_preferences", "search_memories", "forget"} <= asyncio.run(names(memory_mcp))
    assert {"list_signals", "simulate_weeks"} <= asyncio.run(names(vehicle_mcp))
