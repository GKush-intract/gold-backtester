import io

from fastapi.testclient import TestClient

from replayer import server as srv
from replayer.server import app


def test_health():
    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def _client(tmp_path, monkeypatch):
    import pandas as pd
    from replayer.market import MarketFeed
    base = pd.Timestamp("2024-01-01", tz="UTC")
    df = pd.DataFrame([{"timestamp": base + pd.Timedelta(minutes=k), "open": 2000, "high": 2001,
                        "low": 1999, "close": 2000.5, "volume": 1.0} for k in range(3)])
    monkeypatch.setattr(srv, "FEED", MarketFeed(df))
    monkeypatch.setattr(srv, "SESSIONS_ROOT", tmp_path)
    srv.REGISTRY.clear()
    return TestClient(srv.app)


def test_create_session_and_candles(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    r = client.post("/api/sessions", json={"trader": "alice", "config": {"balance": 10000}})
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert (tmp_path / sid / "meta.json").exists()
    c = client.get(f"/api/sessions/{sid}/candles")
    assert c.status_code == 200
    assert len(c.json()["candles"]) == 3


def test_candles_unknown_session_404(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    assert client.get("/api/sessions/nope/candles").status_code == 404


def test_audio_upload(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    sid = client.post("/api/sessions", json={"trader": "a", "config": {}}).json()["session_id"]
    files = {"file": ("note.webm", io.BytesIO(b"fake-audio"), "audio/webm")}
    r = client.post(f"/api/sessions/{sid}/audio", files=files, data={"market_ts": "1700000000000"})
    assert r.status_code == 200
    aid = r.json()["audio_id"]
    assert (tmp_path / sid / "audio" / f"{aid}.webm").exists()


def test_ws_step_order_and_logging(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    sid = client.post("/api/sessions", json={"trader": "a", "config": {"spread": 0}}).json()["session_id"]
    with client.websocket_connect(f"/ws/{sid}") as wsconn:
        wsconn.send_json({"kind": "control", "action": "step"})
        m = wsconn.receive_json()
        assert m["type"] == "bar"
        wsconn.receive_json()  # account
        wsconn.send_json({"kind": "order", "data": {"client_id": "o1", "side": "buy",
                                                    "order_type": "market", "qty_lots": 1.0}})
        got = [wsconn.receive_json() for _ in range(2)]
        assert any(x.get("type") == "fill" for x in got)
        wsconn.send_json({"kind": "note_text", "data": {"text": "why"}})
    import json as J
    lines = (tmp_path / sid / "events.jsonl").read_text().strip().splitlines()
    types = [J.loads(x)["type"] for x in lines]
    assert "order_submit" in types and "fill" in types and "note_text" in types
