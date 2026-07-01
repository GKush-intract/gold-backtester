from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Gold Replayer")


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Static frontend (mounted last so /api/* wins). Created in later tasks.
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
