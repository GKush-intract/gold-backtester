from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = Path(__file__).parent / "sessions"


class Session:
    """One replay run. Owns meta.json, an append-only events.jsonl, and an audio/ folder."""

    def __init__(self, session_id: str, root: Path, seq: int = 0):
        self.session_id = session_id
        self.dir = Path(root) / session_id
        self._seq = seq

    @classmethod
    def create(cls, trader: str, config: dict, root: Path = DEFAULT_ROOT) -> "Session":
        session_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        d = Path(root) / session_id
        (d / "audio").mkdir(parents=True, exist_ok=True)
        meta = {"session_id": session_id, "trader": trader, "config": config,
                "created_at": datetime.now(timezone.utc).isoformat()}
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        (d / "events.jsonl").touch()
        s = cls(session_id, root)
        s.log("session_start", {"trader": trader, "config": config}, market_ts=None)
        return s

    @classmethod
    def load(cls, session_id: str, root: Path = DEFAULT_ROOT) -> "Session":
        path = Path(root) / session_id / "events.jsonl"
        seq = sum(1 for _ in path.open()) if path.exists() else 0
        return cls(session_id, root, seq=seq)

    def log(self, type: str, data: dict, market_ts: int | None) -> int:
        rec = {"seq": self._seq, "wall_ts": datetime.now(timezone.utc).isoformat(),
               "market_ts": market_ts, "type": type, "data": data}
        with (self.dir / "events.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        self._seq += 1
        return rec["seq"]

    def save_audio(self, blob: bytes, ext: str = "webm") -> str:
        aid = uuid.uuid4().hex[:12]
        (self.dir / "audio" / f"{aid}.{ext}").write_bytes(blob)
        return aid
