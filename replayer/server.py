from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
