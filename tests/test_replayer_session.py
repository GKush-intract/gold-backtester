import json

from replayer.session import Session


def test_create_and_log(tmp_path):
    s = Session.create(root=tmp_path, trader="alice", config={"balance": 10000})
    assert (tmp_path / s.session_id / "meta.json").exists()
    s.log("clock", {"action": "play", "speed": 60}, market_ts=1700000000000)
    s.log("order_submit", {"side": "buy"}, market_ts=1700000060000)
    lines = (tmp_path / s.session_id / "events.jsonl").read_text().strip().splitlines()
    # session_start is logged by create(), so these two are seq 1 and 2
    playline = json.loads(lines[-2])
    assert playline["type"] == "clock" and playline["market_ts"] == 1700000000000
    assert "wall_ts" in playline
    assert json.loads(lines[-1])["type"] == "order_submit"
    # seq is monotonic across all lines starting at 0
    seqs = [json.loads(l)["seq"] for l in lines]
    assert seqs == list(range(len(lines)))


def test_save_audio(tmp_path):
    s = Session.create(root=tmp_path, trader="bob", config={})
    aid = s.save_audio(b"OggS-fake-bytes", ext="webm")
    p = tmp_path / s.session_id / "audio" / f"{aid}.webm"
    assert p.exists() and p.read_bytes() == b"OggS-fake-bytes"


def test_load_appends(tmp_path):
    s = Session.create(root=tmp_path, trader="c", config={})
    n0 = sum(1 for _ in (tmp_path / s.session_id / "events.jsonl").open())
    s.log("a", {}, market_ts=1)
    s2 = Session.load(s.session_id, root=tmp_path)
    s2.log("b", {}, market_ts=2)
    lines = (tmp_path / s.session_id / "events.jsonl").read_text().strip().splitlines()
    seqs = [json.loads(l)["seq"] for l in lines]
    assert seqs == list(range(len(lines)))          # continued monotonic after reload
    assert json.loads(lines[-1])["type"] == "b"
