# Gold Backtesting Platform

Local, single-user backtester for gold (XAUUSD). Fetches candles from Dukascopy, runs plug-in
strategies through a correctness-focused bar-by-bar engine, and shows results in a Streamlit UI.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

- **UI:** `.venv/bin/streamlit run app.py`
- **CLI (synthetic):** `.venv/bin/python run_cli.py --synthetic`
- **CLI (fetch real data):** `.venv/bin/python run_cli.py --fetch --start 2024-01-02 --end 2024-02-01`
- **Tests:** `.venv/bin/pytest`

## Data

- Source 1: **Dukascopy fetch** (default) — XAUUSD candles, cached to `data/cache/dukascopy/`.
- Source 2: **Local CSV** in `data/raw/` with header `timestamp,open,high,low,close,volume`,
  `timestamp` ISO-8601 UTC, prices float, `volume` is tick volume (may be 0).
- The loader sorts, dedupes, validates (rejects bad OHLC / NaNs / naive timestamps without `tz`),
  and never forward-fills across weekend gaps.

## Adding a strategy

Drop a file in `src/strategies/` with a `Strategy` subclass. It appears in the UI/CLI automatically
(zero edits elsewhere). Copy `smc_template.py` as a starting point.

```python
from ..strategy import Strategy

class MyStrategy(Strategy):
    name = "My Strategy"
    htf_timeframes = ["h1"]          # optional higher-timeframe views (ctx.htf["h1"])
    params = {                        # (type, default, min, max, help) — drives the UI
        "risk_pct": ("float", 0.01, 0.001, 0.05, "Risk per trade"),
    }
    def on_bar(self, ctx):
        # ctx.bar, ctx.history, ctx.htf, ctx.position, ctx.equity
        # ctx.enter(direction, size, stop_loss=, take_profit=, tag=)  -> fills next bar open
        # ctx.close(reason=)                                          -> fills next bar open
        # ctx.size_for_risk(risk_pct, entry, stop)                    -> size in LOTS
        ...
```

## Engine assumptions (read before trusting numbers)

- **No look-ahead:** `on_bar` sees only bars `0..index`; orders fill at the **next bar's open**.
- **Intrabar SL/TP:** default `stop_first` (pessimistic) when both lie in a bar's range;
  configurable `tp_first` / `optimistic`. Fill at the bracket price; **gap-through** fills at the
  (worse) open.
- **Trailing stop:** attach a price distance at entry via `ctx.enter(..., trail=<distance>)`. The
  engine trails it **intrabar** — the level starts at `entry ∓ trail` (acting as the initial stop)
  and ratchets toward price using each bar's high/low *after* surviving the bar. Resolution is
  **conservative**: a retrace can hit the level carried into a bar before that bar's extreme would
  advance it. (TradingView is optimistic and its `trail_points`/`trail_offset` are in **ticks**,
  not price — a tight tick-trail there books unrealistic intrabar wins; ours won't.) Exits show as
  `trailing_stop`; a fixed `stop_loss` and a `trail` can coexist (the tighter binds).
- **Costs:** half-spread + slippage adverse on every fill; commission (`per_trade` flat +
  `per_lot`) charged once per round-trip at close.
- **Sizing/P&L** in **lots** (1 lot = 100 oz): `pnl = (exit-entry)*lots*100*(±1) - commission`.
  `size_for_risk` returns lots, rounded down to 0.01 (micro-lot). Each trade also records its
  `notional` (lots×100×price) and `leverage` (notional ÷ equity at entry).
- **Leverage cap:** `max_leverage` caps a position's notional at `max_leverage × equity` (default
  20×, `0` = unlimited). Risk-based sizing on a tight stop can otherwise imply large notional
  because no broker margin is modeled — the cap is the guardrail. Size is rounded to the lot step
  after capping; if it rounds to 0 lots, the trade is skipped.
- **Equity vs balance:** `balance` = realized cash; `equity` = balance + unrealized at close;
  drawdown is computed off `equity`. Run stops if `equity <= 0`.
- **Entry-bar brackets:** a position fills at a bar's open and its SL/TP are first checked on the
  **next** bar — the entry bar's own high/low cannot stop you out. This avoids ambiguous
  intra-entry-bar ordering, but for very tight stops it slightly understates stop-outs (optimistic).
  Pinned by `test_entry_bar_bracket_not_checked`.
- **Sharpe / Sortino** are annualized from per-bar equity returns (risk-free = 0) using
  √(periods-per-year). On intraday (e.g. M5) data this multiplier is large and the figures are
  **inflated / not directly comparable to daily Sharpe** — treat them as relative, not absolute.
- **v1 limits (TODO):** one position at a time (no pyramiding), single TP (no partials),
  market entries only (no pending limit/stop entries), no margin/leverage modeling.

## Replayer

A standalone market-replayer (separate from the Streamlit backtester) that plays the
Jan–Mar 2024 XAUUSD m1 feed back as a simulated live market for discretionary trading, and
records every action for later strategy analysis.

Run:

    bash scripts/run_replayer.sh          # http://localhost:8502

Controls: Play/Pause, speed (1–MAX ×), Step, Skip-gap, and a timeframe switcher (m1–d1,
resampled live from m1). Trade with market/limit/stop bracket orders (SL/TP), reusing the
backtester's fill/cost/leverage model. Draw on the chart, write text notes, and record voice
notes explaining each trade.

Every action is recorded to `replayer/sessions/<id>/`:

- `meta.json` — trader name + config
- `events.jsonl` — every action (clock, orders, fills, drawings, notes) with wall + market timestamps
- `audio/*.webm` — voice notes

Browser: use Chrome (voice capture uses MediaRecorder).
