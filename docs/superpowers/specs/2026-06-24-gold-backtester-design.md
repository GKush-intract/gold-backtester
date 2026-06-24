# Gold Backtesting Platform — Design

**Date:** 2026-06-24
**Status:** Approved (design phase)
**Source:** Refines `FIRST_PLAN.md` with decisions from the brainstorming session.

---

## 1. Goal

A local, single-user backtesting platform for gold (XAUUSD). The user writes strategies one at a
time as plug-in classes. Each strategy fully owns its trading logic (entry, exit, stop-loss,
take-profit, position sizing). The engine is "dumb": it loads OHLC data, simulates trading bar-by-bar
from an opening balance, executes what the strategy requests, enforces SL/TP brackets, and produces
results (equity curve, profit factor, win rate, drawdown, R-distribution, trade log).

A Streamlit UI lets the user fetch data, pick a time period, set parameters (opening balance, costs,
strategy params), select a strategy, run the backtest, and view results.

---

## 2. Decisions (this session)

1. **UI:** Streamlit + Plotly. Fastest path for a single-user local research tool; pure Python.
2. **Data source:** Fetch XAUUSD directly from **Dukascopy** as part of the build, store/cache
   locally. CSV drop-in remains a supported alternate source. (Replaces the spec's "user always
   drops a CSV" as the *primary* path.)
3. **Data granularity:** Fetch **pre-aggregated candles** at a chosen base timeframe (M1/M5), not
   raw ticks. Fast, offline-cacheable, sufficient for bar-based backtesting.
4. **Build flow:** Full spec, built in the layered order below, verifying each layer (tests/CLI)
   before moving to the next.

---

## 3. Tech stack

- **Language:** Python 3.11+ (3.13 available locally).
- **Core:** `pandas`, `numpy`.
- **Data fetch:** a Dukascopy downloader library (`dukascopy-python` or `duka`); fallback to a
  direct `.bi5` (LZMA) downloader against Dukascopy's public datafeed if neither installs cleanly
  on 3.13.
- **UI:** `streamlit`. **Charts:** `plotly`.
- **Cache:** `pyarrow` (parquet).
- **Tests:** `pytest`.
- No database, no live trading, no broker connections. Everything runs locally.
- Versions pinned in `requirements.txt`.

---

## 4. Project structure

```
gold-backtester/
├── data/
│   ├── raw/                     # user-dropped OHLC CSVs (alternate source)
│   └── cache/
│       └── dukascopy/           # cached downloads: <symbol>_<tf>_<start>_<end>.parquet
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py          # NEW: download XAUUSD candles from Dukascopy + cache
│   ├── data_loader.py           # validate, cache, slice, resample OHLC
│   ├── engine.py                # event loop, Position, fills, cost model, BacktestConfig/Result
│   ├── strategy.py              # Strategy base class + Context + Order
│   ├── metrics.py               # all stats from trade log + equity curve
│   ├── runner.py                # run_backtest(config, strategy, data) -> BacktestResult
│   └── strategies/
│       ├── __init__.py          # auto-discovery registry
│       ├── ma_crossover.py      # reference example strategy (smoke test)
│       └── smc_template.py      # documented stub for real strategies
├── tests/
│   ├── test_engine.py           # correctness tests (§9)
│   ├── test_metrics.py
│   └── test_data.py             # loader validation + resample + fetch cache
├── app.py                       # Streamlit UI
├── run_cli.py                   # headless runner for fast iteration / CI
├── requirements.txt
└── README.md
```

---

## 5. Data layer

### 5.1 `data_fetcher.py` (NEW)

- **`fetch_ohlc(symbol="XAUUSD", timeframe="m5", start, end) -> pd.DataFrame`**
  - Downloads candle data from Dukascopy's free historical feed for the requested range.
  - Output conforms to the §5.2 contract: UTC tz-aware index, columns
    `timestamp,open,high,low,close,volume`, sorted ascending, deduped.
  - **Cache:** writes/reads `data/cache/dukascopy/<symbol>_<tf>_<start>_<end>.parquet`. On a cache
    hit, no network call — runs offline. Partial-range reuse is a later enhancement; v1 keys on the
    exact requested range.
  - **Implementation choice:** prefer a maintained Dukascopy downloader library. If unavailable on
    Python 3.13, fall back to fetching Dukascopy's hourly `.bi5` LZMA-compressed candle files
    directly and decoding them. The choice is isolated inside this module; the rest of the system
    only sees the returned DataFrame.
  - `volume` is **tick volume** (Dukascopy) — never assumed to be exchange volume.

### 5.2 `data_loader.py`

- **Input contract** (for both fetched data and user CSVs in `data/raw/`):
  `timestamp,open,high,low,close,volume`; `timestamp` ISO 8601 UTC, parsed tz-aware (reject naive
  unless a `tz` is supplied); OHLC float; volume float (may be 0).
- **`load_ohlc(source, start=None, end=None) -> pd.DataFrame`**: load (from CSV path or fetched
  frame), validate schema, sort, assert monotonic, drop exact-duplicate timestamps, slice to
  `[start, end]`. Validated frames cached to parquet keyed on content hash.
- **`resample(df, timeframe) -> pd.DataFrame`**: standard OHLC resample
  (`open=first, high=max, low=min, close=last, volume=sum`), right-labeled, dropping incomplete
  trailing bars. Enables HTF context (H1/H4 bias + M5 entry).
- **Gap handling:** do NOT forward-fill across weekend/session gaps; leave gaps as-is. Optional
  `max_gap_minutes` warning.
- **Validation catches:** missing columns, NaNs, `high < low`, `high < open/close`,
  `low > open/close`, negative prices, non-monotonic timestamps — fail loudly.

---

## 6. Strategy interface (`strategy.py`)

The strategy owns all trading decisions. The engine only executes requests and enforces brackets.

- **`Order`**: `direction` ("long"/"short"), `size` (units/oz, >0), `stop_loss` (abs price | None),
  `take_profit` (abs price | None), `tag`.
- **`Position`**: `direction, size, entry_price, entry_time, stop_loss, take_profit, tag`.
- **`Context`** (passed each bar):
  - Data (no look-ahead): `bar` (current OHLC + time), `index`, `history` (bars `0..index`
    inclusive), `htf` (dict of resampled higher-TF views, also sliced to "now").
  - Account/position: `equity` (mark-to-market), `balance` (realized cash), `position`.
  - Order API: `enter(direction, size, stop_loss=None, take_profit=None, tag="")` (fills next bar
    open), `close(reason="manual")` (fills next bar open).
  - Sizing helper: `size_for_risk(risk_pct, entry_price, stop_price) -> units`
    = `(equity * risk_pct) / abs(entry - stop)`.
- **`Strategy`** base: `name`, `params` schema (`{name: (type, default, min, max, help)}` — drives
  the UI), `__init__(**params)`, `on_start(ctx)`, `on_bar(ctx)`, `on_finish(ctx)`.

**Must be expressible:** long/short; SL+TP at entry (engine auto-manages exits); discretionary
`ctx.close()`; strategy-computed sizing; reading history + HTF views; session/time-of-day filtering.

**v1 simplifications (TODOs, designed to lift later):** one open position at a time (ignore + warn
on `enter()` while open — no pyramiding); single TP (no partial exits); market entries only filled
next-bar-open (no pending limit/stop entries).

**Auto-discovery (`strategies/__init__.py`):** discover all `Strategy` subclasses in
`src/strategies/`; expose `get_strategy_registry() -> dict[name, class]`. Adding a strategy file
requires **zero** changes to engine/UI/metrics.

---

## 7. Engine (`engine.py`)

Event-driven, single pass over bars. Loop ordering (enforces no look-ahead):

```
init: balance = opening_balance; equity = balance; position = None; trades = []; equity_curve = []
for index, bar in enumerate(bars):
    1) MANAGE existing position vs THIS bar (SL/TP may fire intrabar) -> realize pnl, close
    2) Execute order requested on PREVIOUS bar (fill at THIS bar.open ± costs) if flat
    3) Build ctx from data up to THIS bar; call strategy.on_bar(ctx) -> may set pending order / close
    4) Mark-to-market equity at bar.close; record equity_curve point + running max drawdown
```

### Correctness conventions (non-negotiable defaults)

1. **No look-ahead.** `on_bar` sees only bars `0..index`. Orders on bar `t` fill at `t+1` open.
   Test: a "tomorrow's close" cheat earns no risk-free profit.
2. **Intrabar SL/TP.** Only OHLC within a bar → if both stop and target lie in `[low, high]`,
   can't know order. Default **`stop_first`** (pessimistic), configurable
   `"stop_first" | "tp_first" | "optimistic"`.
   - Long: stop if `low <= stop_loss`; TP if `high >= take_profit`. Short: mirrored.
   - Fill at the **bracket price** + slippage. **Gap-through:** if the bar *opens* beyond the stop,
     fill at the open (worse).
3. **Fill model.** Market entries/exits at next bar open; bracket exits at bracket price.
   Costs on every fill: **spread** (buys at `price + spread/2`, sells at `price - spread/2`),
   **slippage** (adverse, price units), **commission** (`commission_per_trade` flat and/or
   `commission_per_unit`).
4. **Sizing & P&L** in units (oz): `pnl = (exit - entry) * size * (+1 long / -1 short) - costs`.
   README notes XAUUSD standard lot = 100 oz (`lots = units / 100`); engine stays in oz.
5. **Equity vs balance.** `balance` = realized cash; `equity` = balance + unrealized at current
   close. Equity curve and drawdown use `equity`.
6. **Margin/liquidation:** ignored v1 (cash-funded, no leverage). Flag and stop the run if
   equity ≤ 0.

### `BacktestConfig`
`opening_balance=10_000, spread=0.30, slippage=0.0, commission_per_trade=0.0,
commission_per_unit=0.0, intrabar="stop_first", entry_fill="next_open"`.

### `BacktestResult`
Equity-curve DataFrame (`time, equity, balance, drawdown`), closed-trade list/DataFrame, config
used, data range. Consumed by `metrics.py`.

### `runner.py`
`run_backtest(config, strategy, data) -> BacktestResult` — single entry point shared by CLI, UI,
and a future optimizer/sweep.

---

## 8. Metrics (`metrics.py`)

Pure functions over `BacktestResult`, returning a dict + tidy DataFrame:

- **Returns:** total return %, absolute P&L, CAGR (annualized over actual elapsed data range).
- **Trade stats:** # trades, win/loss rate, avg win/loss, largest win/loss, avg holding time
  (bars + wall-clock).
- **Profit factor** = gross profit / gross loss.
- **Expectancy** per trade ($ and R).
- **R-multiples:** `R = realized_pnl / initial_risk`, `initial_risk = abs(entry - stop) * size`.
  Full R distribution (histogram) + mean R, R std.
- **Max drawdown** (% and abs) + max drawdown duration.
- **Sharpe / Sortino** (optional; per-bar equity returns, annualized — assumption stated in UI).
- **Exposure %** (fraction of bars with an open position).
- Guard divide-by-zero: no losing trades → PF = inf; no trades → zeros + "no trades" flag.

---

## 9. UI (`app.py`, Streamlit)

**Sidebar:**
- **Data source:** (a) Fetch from Dukascopy — symbol fixed XAUUSD v1, base timeframe + date range
  → fetch & cache; or (b) pick a local CSV from `data/raw/`. Optional resample to M5/M15/H1/H4.
- **Date range** pickers (default to data's full range).
- **Strategy** dropdown (from auto-discovery registry).
- **Strategy params** rendered dynamically from the selected strategy's `params` schema.
- **Account & cost params:** opening balance, spread, slippage, commission, intrabar rule,
  entry fill mode.
- **Run backtest** button.

**Main panel:**
- **Metrics cards:** total return, profit factor, win rate, max drawdown, expectancy (R), # trades.
- **Equity curve** (Plotly) with balance overlaid + drawdown subplot/shading.
- **R-multiple histogram.**
- **Price candlestick** with entry/exit markers + SL/TP lines for a selected trade (behind a
  checkbox; can be slow on a year of M5).
- **Trade log table** (sortable): entry/exit time, direction, entry, exit, size, SL, TP, P&L, R,
  exit reason, tag. **Download CSV**. Download equity curve PNG/CSV.
- Show exact config used (reproducibility) + run time.

UI is thin: collect params → `run_backtest(...)` → render. No trading logic in the UI.

---

## 10. Reference strategies

- **`ma_crossover.py`** (smoke test): params `fast=20, slow=50, risk_pct=0.01`, fixed-distance or
  ATR-based SL/TP (pick one, document). Long on fast SMA crossing above slow, short on cross below;
  SL/TP at entry; size via `ctx.size_for_risk`. Exists to validate the engine.
- **`smc_template.py`** (non-functional, documented stub): how to read HTF bias via
  `ctx.htf["H1"]`/`["H4"]` while trading M5; session filtering (London ~07:00–10:00 UTC, NY
  ~12:00–15:00 UTC) off `ctx.bar["time"]`; stubbed swing/BOS/CHoCH helper signatures; entry with a
  structural stop + R-based TP, size via risk %. The file to copy when starting a new strategy.

---

## 11. Tests (`tests/`)

1. **No look-ahead:** signal on bar `t` fills at `t+1` open; verify fill prices; cheat earns nothing.
2. **Long winner/loser** hand-computed (~5 synthetic bars): exact P&L incl. spread + commission.
3. **Short winner/loser:** mirrored.
4. **Intrabar both-in-range:** `stop_first` → stop fill; flip config → TP fill.
5. **Gap-through stop:** bar opens beyond stop → fill at open (worse than stop).
6. **Equity vs balance:** open position → equity reflects unrealized, balance unchanged till close.
7. **Metrics:** profit factor, win rate, R, max drawdown on a tiny known trade list.
8. **Data:** loader rejects bad schemas/NaNs/bad OHLC; resample correctness; fetch cache hit avoids
   network (mock the downloader).

`run_cli.py` runs a full headless backtest on cached data and prints the metrics table — fast
iteration + sanity gate before the UI.

---

## 12. Non-goals (v1)

No live/paper trading or broker connectivity. No multi-asset portfolio (gold only; engine stays
asset-agnostic internally). No parameter optimization / walk-forward / Monte Carlo (but
`run_backtest()` is designed to be wrapped by a sweep later). No pyramiding, scaled/partial exits,
or pending limit/stop entries (TODOs in the relevant modules).

---

## 13. Acceptance criteria

- Launch the Streamlit app, fetch XAUUSD M5 from Dukascopy for a date range (cached locally), pick
  `ma_crossover`, set opening balance + costs, click Run, and see an equity curve, metrics, and a
  downloadable trade log. (Dropping a CSV into `data/raw/` also works.)
- All §11 tests pass.
- Adding a new strategy file to `src/strategies/` makes it appear in the dropdown with its params,
  with **zero** edits to engine/UI/metrics.
- README documents: data format + Dukascopy fetch, how to add a strategy (params schema), every
  engine assumption from §7, and how to run tests + CLI.

---

## 14. Build order (verify each layer before the next)

1. `requirements.txt` + scaffold + git.
2. `data_fetcher.py` — Dukascopy fetch + parquet cache; verify by pulling a small XAUUSD M5 range.
3. `data_loader.py` — validate/resample/cache + `test_data.py`.
4. `engine.py` + `strategy.py` + §11 correctness tests (the honesty gate).
5. `metrics.py` + `test_metrics.py`.
6. `ma_crossover.py` + `runner.py` + `run_cli.py` — full headless backtest prints metrics.
7. `app.py` — Streamlit UI.
8. `smc_template.py` + README.
