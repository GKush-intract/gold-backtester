from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.engine import BacktestConfig
from .engine_loop import ReplaySession
from .market import MarketFeed
from .session import Session

STATIC_DIR = Path(__file__).parent / "static"
SESSIONS_ROOT = Path(__file__).parent / "sessions"

app = FastAPI(title="Gold Replayer")

# Loaded once (86k m1 bars). Overridable in tests.
FEED: MarketFeed | None = None
REGISTRY: dict[str, Session] = {}


def get_feed() -> MarketFeed:
    global FEED
    if FEED is None:
        FEED = MarketFeed.from_parquet()
    return FEED


class NewSession(BaseModel):
    trader: str = "anon"
    config: dict = {}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/sessions")
def create_session(body: NewSession):
    s = Session.create(trader=body.trader, config=body.config, root=SESSIONS_ROOT)
    REGISTRY[s.session_id] = s
    return {"session_id": s.session_id}


@app.get("/api/sessions/{sid}/candles")
def candles(sid: str):
    if sid not in REGISTRY:
        raise HTTPException(404, "unknown session")
    return {"candles": get_feed().candles()}


@app.post("/api/sessions/{sid}/audio")
async def upload_audio(sid: str, file: UploadFile = File(...), market_ts: str = Form(None)):
    if sid not in REGISTRY:
        raise HTTPException(404, "unknown session")
    s = REGISTRY[sid]
    blob = await file.read()
    ext = (file.filename or "clip.webm").rsplit(".", 1)[-1]
    aid = s.save_audio(blob, ext=ext)
    s.log("note_audio", {"audio_id": aid, "bytes": len(blob)},
          market_ts=int(market_ts) if market_ts else None)
    return {"audio_id": aid}


def _feed_frame():
    """A fresh dataframe copy of the m1 feed data (each WS connection needs its own cursor)."""
    import numpy as np
    import pandas as pd
    cs = get_feed().candles()
    ts = pd.to_datetime(np.array([c["t"] for c in cs]), unit="ms", utc=True)
    return pd.DataFrame({"timestamp": ts,
                         "open": [c["o"] for c in cs], "high": [c["h"] for c in cs],
                         "low": [c["l"] for c in cs], "close": [c["c"] for c in cs],
                         "volume": [c["v"] for c in cs]})


def _cfg_from(meta_config: dict) -> BacktestConfig:
    c = meta_config or {}
    return BacktestConfig(
        opening_balance=float(c.get("balance", 10_000.0)),
        spread=float(c.get("spread", 0.30)),
        slippage=float(c.get("slippage", 0.0)),
        commission_per_trade=float(c.get("commission_per_trade", 0.0)),
        commission_per_lot=float(c.get("commission_per_lot", 0.0)),
        max_leverage=float(c.get("max_leverage", 20.0)),
        intrabar=c.get("intrabar", "stop_first"),
    )


@app.websocket("/ws/{sid}")
async def ws(websocket: WebSocket, sid: str):
    await websocket.accept()
    if sid not in REGISTRY:
        await websocket.send_json({"type": "error", "reason": "unknown session"})
        await websocket.close()
        return
    session = REGISTRY[sid]
    import json as _json
    meta = _json.loads((session.dir / "meta.json").read_text())
    rs = ReplaySession(session, MarketFeed(_feed_frame()), _cfg_from(meta.get("config", {})))

    async def play_loop():
        while True:
            if rs.playing and not rs.feed.at_end:
                for _ in range(max(1, rs.speed // 10)):
                    if not rs.playing or rs.feed.at_end:
                        break
                    for m in rs.tick():
                        await websocket.send_json(m)
                await asyncio.sleep(0.05)
            else:
                await asyncio.sleep(0.05)

    loop_task = asyncio.create_task(play_loop())
    try:
        while True:
            msg = await websocket.receive_json()
            for out in rs.handle(msg):
                await websocket.send_json(out)
    except WebSocketDisconnect:
        pass
    finally:
        loop_task.cancel()


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
