# Replayer — Design Spec

**Date:** 2026-07-01
**Branch:** `replayer`
**Status:** Approved (design), pending implementation plan

## Purpose

A market **replayer** that plays historical price data back as a simulated live feed so a
trader can trade it discretionarily — with fast-forward and speed control — and have **every
action recorded**. The first window is **XAUUSD, 1 Jan 2024 → 31 Mar 2024** on **m1** data
(fetched: 86,605 bars, 2024-01-01 → 2024-03-28; the tail is Good Friday + weekend).

The recording is the point: trades, orders, fills, on-chart drawings, timeframe/indicator
changes, text notes, and **voice notes** are all logged so a later phase can reconstruct and
**automate the trader's strategy**. This spec covers **step 1 only: record everything**. The
strategy-inference/automation phase is explicitly out of scope here.

Goal quality bar: "as good as a trading product" — professional charting, multi-timeframe,
drawing tools, a real order ticket with brackets, positions/orders/account panels, a blotter.

## Non-goals (step 1)

- No strategy inference, clustering, or code generation from the logs (future phase).
- No multi-user auth/accounts (a simple trader-name field only; local single-user).
- No live/real broker connectivity. Replay only, over cached historical data.
- No changes to the existing Streamlit backtester; the replayer is a separate service.

## Architecture

**FastAPI backend + static KLineChart frontend, server-authoritative.**

- The replay **clock, order-matching engine, and account state live on the backend** (Python).
  The backend is the single source of truth. This lets us **reuse the tested `src/engine.py`**
  matching/cost semantics and produces a faithful, re-simulatable event log — important for the
  future automation phase. (Alternative considered: static file server with matching duplicated
  in JS — rejected because it splits the source of truth and complicates later re-derivation.)
- **REST** for bootstrap: create session, fetch m1 history, upload audio clips.
- **WebSocket** for the live loop: control messages in (play/pause/speed/seek/step/skip-gap,
  order submit/cancel, drawing, note); bar + account/position/order updates out.
- Runs as its own service on **port 8502**, separate from the Streamlit backtester (8501).
- New `replayer/` Python package. Runtime recordings under `replayer/sessions/` (gitignored).

### Data & timeframes

- Base feed = the **m1 parquet** for the window
  (`data/cache/dukascopy/XAUUSD_m1_20240101_20240401.parquet`).
- Higher timeframes (m5 / m15 / m30 / h1 / h4 / d1) are **resampled from m1 in the frontend**,
  so TF switching is instant and the **in-progress candle forms live** as the clock advances,
  exactly like a real terminal.
- **Causal by construction:** the frontend only ever holds bars up to the current replay time;
  the backend streams one m1 bar at a time. No look-ahead is possible.

### Replay controls

- Play / Pause.
- **Speed** presets in bars/sec: `1 / 4 / 15 / 60 / 240 / MAX`. MAX runs as fast as the loop
  allows (clears 3 months in minutes).
- **Step** (+1 bar) when paused.
- **Jump to date** (seek to an arbitrary market timestamp in the window).
- **Skip-gap** (jump over weekend / market-closed gaps to the next bar).
- A **scrubber** showing current position within the 3-month window.
- Backend **batches WS updates (~100 ms)** so high speeds stay smooth (no per-bar round trip).

### Trading (full bracket orders)

- Order ticket: **Market / Limit / Stop**, quantity in **lots** (1 lot = 100 oz), optional
  **SL / TP**.
- Matching (mirrors `src/engine.py`, reusing its helpers where possible):
  - **Market**: fills immediately at the current m1 price ± half-spread + slippage.
  - **Limit**: rests; fills when an m1 bar trades through the limit (buy limit when `low <= px`),
    at the limit price ± slippage.
  - **Stop**: rests; fills when an m1 bar trades through the stop (buy stop when `high >= px`),
    at the stop price ± slippage.
  - **SL/TP** on an open position resolved **intrabar** with the engine's `stop_first` policy
    (configurable), including gap-through-at-open handling.
  - **Costs**: spread / slippage / commission and **leverage cap** via `src/engine.py` +
    `src/strategy.py` constants (`OZ_PER_LOT`, `clamp_lots`, `_buy_fill`/`_sell_fill`,
    `_resolve_bracket`).
- **Panels:** open positions, working orders, account (balance / equity / margin / unrealized
  P&L), trade blotter. **Equity ≤ 0 → margin stop.**

### Charting

- **KLineChart** (klinecharts.com) frontend library.
- Built-in **drawing/overlay tools** (trendlines, rays, rectangles, horizontal levels, etc.).
- A small set of **built-in indicators** (e.g. MA, RSI) exposed via KLineChart natives; toggling
  them is logged.
- Timeframe switcher (m1 … d1).

### Recording (the deliverable)

Every action is appended to **`replayer/sessions/<session_id>/events.jsonl`**, one JSON object
per line:

```json
{"seq": 0, "wall_ts": "<ISO real time>", "market_ts": "<ISO replay bar time>", "type": "...", "data": {...}}
```

Event types:

| type | data |
|---|---|
| `session_start` | trader name, symbol, window, opening balance, leverage, spread, slippage, commission, intrabar policy |
| `clock` | action (`play`/`pause`/`seek`/`speed`/`step`/`skip_gap`), speed, to_market_ts |
| `tf_change` | timeframe |
| `indicator` | action (`add`/`remove`), name, params |
| `order_submit` | client_id, side, order_type (`market`/`limit`/`stop`), qty_lots, price?, sl?, tp? |
| `order_fill` | client_id, fill_price, market_ts |
| `order_cancel` | client_id |
| `order_reject` | client_id, reason |
| `position_update` | pos_id, qty_lots, avg_price, sl, tp, unrealized |
| `position_close` | pos_id, exit_price, reason (`manual`/`sl`/`tp`/`margin`), pnl, r_multiple |
| `drawing` | action (`add`/`edit`/`remove`), overlay_id, tool, points `[{ts, price}]`, style |
| `note_text` | text, ref? (order/position id) |
| `note_audio` | audio_id, duration_ms, ref? |
| `account` | balance, equity, margin |
| `session_end` | summary |

- **Drawings** captured via KLineChart overlay create/modify/remove events.
- **Voice**: MediaRecorder clip → `POST /api/sessions/<id>/audio` (multipart) → saved as
  `audio/<audio_id>.webm` → frontend logs a `note_audio` event referencing `audio_id`.
- **Text notes** via a notes box, optionally attached to a specific trade.

Session directory layout:

```
replayer/sessions/<session_id>/
  meta.json         # trader, window, config, created_at
  events.jsonl      # append-only event log
  audio/<id>.webm   # voice clips
```

## Components

```
replayer/
  __init__.py
  server.py     # FastAPI app: REST endpoints + WebSocket + static mount
  session.py    # session lifecycle, JSONL event writer, audio store
  market.py     # load m1 window, bar iterator, seek/skip-gap helpers
  matching.py   # incremental bracket matcher (reuses src/engine helpers)
  models.py     # dataclasses / pydantic models for orders, positions, events, WS messages
  static/
    index.html
    css/style.css
    js/
      ws.js       # WebSocket client, message dispatch
      chart.js    # KLineChart setup, render forming candle
      resample.js # m1 -> selected TF aggregation
      clock.js    # replay controls UI
      trade.js    # order ticket, positions/orders/account panels, blotter
      drawing.js  # overlay capture -> drawing events
      notes.js    # text + voice (MediaRecorder) capture
      app.js      # wiring
  sessions/       # runtime recordings (gitignored)
```

### REST endpoints

- `POST /api/sessions` → `{session_id}` (body: trader name + config). Writes `meta.json`,
  `session_start` event.
- `GET  /api/sessions/<id>/candles` → m1 OHLCV for the window (JSON) for initial chart load.
- `POST /api/sessions/<id>/audio` (multipart) → saves clip, returns `{audio_id}`.
- `GET  /` and static assets → the frontend.

### WebSocket `/ws/<session_id>`

- **In** (client→server): `control` (clock ops), `order` (submit/cancel), `draw`, `note_text`,
  `note_audio_ref`, `tf_change`, `indicator`.
- **Out** (server→client): `bar` (new/forming m1 bar batch), `account`, `position`, `order`,
  `fill`, `reject`, `clock_state`, `session_stopped`.
- The server authoritatively advances the clock, runs matching per m1 bar, appends every
  resulting event to `events.jsonl`, and pushes state to the client.

## Dependencies

Add to `requirements.txt`: `fastapi`, `uvicorn[standard]`, `websockets`, `python-multipart`,
`httpx` (tests). KLineChart via vendored JS asset or CDN `<script>`.

## Testing

- `tests/test_replayer_market.py` — m1 window load; resample correctness (m1→m5/m15 aggregation
  matches a pandas reference); causal slicing / seek / skip-gap.
- `tests/test_replayer_matching.py` — market/limit/stop fills; SL/TP intrabar (`stop_first`);
  leverage cap; mark-to-market equity; margin stop. Asserts parity with `src/engine.py`
  semantics on shared scenarios.
- `tests/test_replayer_server.py` — Starlette `TestClient`: create session (writes meta +
  `session_start`), get candles, WS control appends events to JSONL, audio upload saves a file
  and a `note_audio` event references it.
- **Playwright smoke** (manual `▶` verify, not CI): load page → play → place a bracket order →
  draw a trendline → record a voice note → assert `events.jsonl` grew with the expected event
  types and an audio file exists.

## Run

```
uvicorn replayer.server:app --port 8502
```

(A `/run` project skill or launch script can be added once the app stabilizes.)

## Open items / defaults chosen

- **Server-authoritative WebSocket** confirmed over lighter static+JS.
- **Indicators** (MA/RSI natives) and a **trader-name field** included in step 1.
- Costs default to the backtester's: spread 0.30, slippage 0, leverage cap 20×, `stop_first`.
- Single local user; filesystem persistence (no DB).
- Chrome-targeted (MediaRecorder).
