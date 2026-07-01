"""End-to-end record-loop smoke test. Requires a running server on :8502 and Playwright.
Run manually:
    .venv/bin/pip install playwright && .venv/bin/playwright install chromium
    bash scripts/run_replayer.sh &            # in another shell
    .venv/bin/python -m pytest tests/test_replayer_smoke.py -m smoke -v
"""
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke


def _latest_session() -> Path:
    root = Path("replayer/sessions")
    return max(root.iterdir(), key=lambda p: p.stat().st_mtime)


def test_record_loop():
    pw = pytest.importorskip("playwright.sync_api")
    with pw.sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(permissions=["microphone"])
        page = ctx.new_page()
        page.on("dialog", lambda d: d.accept("smoke-trader"))
        page.goto("http://localhost:8502")
        page.wait_for_timeout(1500)
        page.get_by_text("Step", exact=False).click()
        page.wait_for_timeout(300)
        page.get_by_text("BUY", exact=True).click()
        page.wait_for_timeout(500)
        page.locator("#note-text").fill("smoke rationale")
        page.get_by_text("Save note", exact=False).click()
        page.wait_for_timeout(500)
        browser.close()

    sess = _latest_session()
    types = [json.loads(l)["type"] for l in (sess / "events.jsonl").read_text().splitlines()]
    assert "order_submit" in types
    assert "note_text" in types
