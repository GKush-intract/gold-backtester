# Gold Backtesting Platform — Build Spec

**For:** Claude Code
**Goal:** Build a local, single-user backtesting platform for gold (XAUUSD spot / GC futures OHLC data) with a UI. I write strategies one at a time as plug-in classes. Each strategy fully owns its trading logic (entry, exit, stop-loss, take-profit, position sizing). The backtest engine is "dumb": it loads OHLC data, simulates trading bar-by-bar starting from an opening balance, and produces results (equity curve, profit factor, win rate, drawdown, trade log, etc.).

Build this incrementally and verify each layer before moving on. Read the **Conventions & Correctness** section carefully — that's where backtesters quietly produce wrong numbers.

---

## 1. Tech stack

- **Language:** Python 3.11+
- **Core libs:** `pandas`, `numpy`
- **UI:** `streamlit` (single-language, fastest path for an internal research tool)
- **Charts:** `plotly` (equity curve, drawdown, R-distribution, price+trade markers)
- **Storage/cache:** `pyarrow` (parquet caching of loaded data)
- **Tests:** `pytest`
- No database. No live trading. No broker connections. Everything runs locally off CSV/parquet files.

Pin versions in `requirements.txt`.

---

## 2. Project structure

```
gold-backtester/
├── data/
│   ├── raw/                  # user-dropped OHLC CSVs (e.g. XAUUSD_m5.csv)
│   └── cache/                # parquet cache (auto-generated)
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # load, validate, cache, resample OHLC
│   ├── engine.py             # event loop, Position, fills, cost model
│   ├── strategy.py           # Strategy base class + Context + Order
│   ├── metrics.py            # all stats from trade log + equity curve
│   └── strategies/
│       ├── __init__.py       # auto-discovery registry
│       ├── ma_crossover.py   # reference example strategy (smoke test)
│       └── smc_template.py   # documented stub for my real strategies
├── tests/
│   ├── test_engine.py        # correctness tests (see §9)
│   └── test_metrics.py
├── app.py                    # Streamlit UI
├── run_cli.py                # headless runner for quick iteration / CI
├── requirements.txt
└── README.md
```

Build order: `data_loader` → `engine` + `strategy` → `metrics` → `ma_crossover` + `run_cli` (verify end-to-end in terminal) → `app.py` (UI) → `smc_template`.

---

## 3. Data layer (`data_loader.py`)

### Input contract
Raw OHLC CSV with these columns (header required):

```
timestamp,open,high,low,close,volume
```

- `timestamp`: ISO 8601, **UTC**. Parse to timezone-aware `datetime`. Reject naive timestamps unless a `--tz` is supplied.
- `open/high/low/close`: float.
- `volume`: float (tick volume is fine; may be 0 — never assume it's exchange volume).
- Sorted ascending by timestamp; loader must **sort and assert monotonic** and drop exact-duplicate timestamps.

### Responsibilities
- `load_ohlc(path, start=None, end=None) -> pd.DataFrame`: load, validate schema, sort, dedupe, slice to `[start, end]`. Cache the validated frame to `data/cache/<hash>.parquet` keyed on file content hash; reload from cache when unchanged.
- `resample(df, timeframe) -> pd.DataFrame`: standard OHLC resample (`open=first, high=max, low=min, close=last, volume=sum`), right-labeled, dropping incomplete trailing bars. Needed because strategies may want higher-timeframe context (HTF bias on H1/H4 + LTF entry on M5).
- **Gap handling:** do NOT forward-fill across weekend gaps. Gold has a daily/weekend close; leave gaps as-is. Optionally expose a `max_gap_minutes` warning.
- Validation must catch: missing columns, NaNs, `high < low`, `high < open/close`, `low > open/close`, negative prices, non-monotonic timestamps. Fail loudly with a clear message.

---

## 4. Strategy interface (`strategy.py`) — the heart of the system

Each strategy is a class subclassing `Strategy`. **The strategy owns everything about trading decisions.** The engine never decides when to enter/exit — it only executes what the strategy requests and enforces the SL/TP brackets the strategy attaches.

### Base class (target API — refine signatures as needed but keep the contract)

```python
from dataclasses import dataclass, field

@dataclass
class Order:
    direction: str          # "long" or "short"
    size: float             # units (ounces of gold). >0
    stop_loss: float | None # absolute price
    take_profit: float | None # absolute price (single TP for v1)
    tag: str = ""           # optional label for the trade log

@dataclass
class Position:
    direction: str
    size: float
    entry_price: float
    entry_time: object
    stop_loss: float | None
    take_profit: float | None
    tag: str = ""

class Context:
    """Passed to the strategy each bar. Read-only view of state + order API."""
    # --- data access (NO look-ahead: only data up to and including current bar) ---
    bar: dict                # current bar: time, open, high, low, close, volume
    index: int               # current bar index
    history: "pd.DataFrame"  # all bars[0 .. index] inclusive
    htf: dict[str, "pd.DataFrame"]  # optional resampled higher-TF views, also sliced to "now"

    # --- account/position state ---
    equity: float            # mark-to-market equity (cash + unrealized)
    balance: float           # realized cash only
    position: Position | None

    # --- order API (strategy calls these; engine executes next bar) ---
    def enter(self, direction, size, stop_loss=None, take_profit=None, tag=""): ...
    def close(self, reason="manual"): ...   # discretionary exit; fills next bar open

    # --- sizing helper (optional convenience; sizing still "lives in" the strategy) ---
    def size_for_risk(self, risk_pct, entry_price, stop_price) -> float:
        """units = (equity * risk_pct) / |entry - stop|.  risk_pct e.g. 0.01 = 1%."""

class Strategy:
    name: str = "Unnamed"
    # Parameter schema drives the UI (see §6). Each entry: (type, default, min, max, help)
    params: dict = {}

    def __init__(self, **params): ...   # params merged over defaults
    def on_start(self, ctx: Context): ...      # called once before the loop
    def on_bar(self, ctx: Context): ...        # called every bar — main logic
    def on_finish(self, ctx: Context): ...     # called once after the loop (cleanup)
```

### Rules the strategy must be able to express (so the engine must support them)
- Go **long or short**.
- Attach **stop-loss and take-profit** at entry → engine manages those exits automatically each subsequent bar.
- **Discretionary exit** mid-trade via `ctx.close()`.
- **Position sizing** computed by the strategy (risk-based via helper, or fixed units, or fixed notional).
- Read **price history** and optional **higher-timeframe** views for multi-timeframe logic (HTF bias + LTF trigger — needed for SMC/ICT).
- Filter by **session/time of day** (e.g. London/NY killzones) using `ctx.bar["time"]`.

### v1 simplifications (document as TODOs, design so they're easy to lift later)
- **One open position at a time** per backtest. If a strategy calls `enter()` while a position is open, ignore + log a warning (no pyramiding yet).
- **Single TP** (no partial/scaled exits yet).
- **No pending limit/stop entry orders** — entries are market, filled next bar open (see §5). Stop/limit entries are a later enhancement.

### Auto-discovery (`strategies/__init__.py`)
Maintain a registry so adding a strategy = dropping a file. Discover all `Strategy` subclasses in `src/strategies/`, expose `get_strategy_registry() -> dict[name, class]`. The UI and CLI both read from this. **Adding a strategy must require zero changes to the engine or UI.**

---

## 5. Engine (`engine.py`) — the simulator

Event-driven, single pass over bars. Pseudocode of the loop:

```
init: balance = opening_balance; equity = balance; position = None
trades = []            # closed trades
equity_curve = []      # (time, equity, balance, drawdown) per bar

for index, bar in enumerate(bars):
    # 1) MANAGE existing position against THIS bar (SL/TP can fire intrabar)
    if position is not None:
        exit_fill = check_bracket_exit(position, bar)   # see intrabar rules
        if exit_fill is not None:
            realize pnl, append trade, balance += pnl, position = None

    # 2) Execute any order requested on the PREVIOUS bar (next-bar-open fill)
    if pending_order is not None and position is None:
        fill_price = bar.open  (+/- half_spread +/- slippage)
        open position; pending_order = None

    # 3) Ask strategy for new decisions using data UP TO this bar
    ctx = build_context(index, bar, history[0..index], position, equity, balance)
    strategy.on_bar(ctx)
    # ctx.enter(...) sets pending_order (fills next bar);  ctx.close() → exit next bar open

    # 4) Mark-to-market equity at this bar's close
    equity = balance + unrealized_pnl(position, bar.close)
    record equity_curve point + running max drawdown
```

### Conventions & Correctness (READ THIS — non-negotiable defaults)

1. **No look-ahead bias.** In `on_bar`, the strategy may only see bars `0..index` (current bar's OHLC is allowed since it's "closed" by the time we evaluate). Orders requested on bar `t` fill at bar `t+1`'s open — never at `t`'s close on the same bar that produced the signal. Enforce this in the loop ordering above. Add a test that proves a "tomorrow's close > today's close" cheat strategy can't earn risk-free profit.

2. **Intrabar SL/TP resolution.** Within one bar you only have OHLC, so if both stop and target lie inside `[low, high]` you cannot know which hit first. Default to the **pessimistic stop-first rule**: if the bar's range contains the stop, assume stop filled (even if TP also in range). Make this configurable: `intrabar = "stop_first" | "tp_first" | "optimistic"`. Default `"stop_first"`.
   - Long: stop hit if `bar.low <= stop_loss`; TP hit if `bar.high >= take_profit`.
   - Short: stop hit if `bar.high >= stop_loss`; TP hit if `bar.low <= take_profit`.
   - Fill at the **bracket price** (stop_loss / take_profit), not the bar open/close, plus slippage. (Optionally model gap-through: if the bar *opens* beyond the stop, fill at the open, which is worse — model this.)

3. **Fill model.** Market entries/exits fill at the **next bar's open**. Bracket (SL/TP) exits fill at the bracket price during the bar that touched it. Apply costs to every fill:
   - **Spread:** `spread` in price units (gold ~0.2–0.4). Apply half-spread adverse to each side, or model bid/ask — keep simple: buys fill at `price + spread/2`, sells at `price - spread/2`.
   - **Slippage:** `slippage` in price units, adverse, applied to every fill.
   - **Commission:** `commission_per_unit` (per ounce) or flat `commission_per_trade`. Support at least one; default flat per trade = 0.
   - All cost params live on the engine config, surfaced in the UI.

4. **Position sizing & P&L.** Size is in **units (ounces)**. 
   - `pnl = (exit_price - entry_price) * size * (+1 if long else -1) - costs`
   - Provide `size_for_risk(risk_pct, entry, stop)` = `(equity * risk_pct) / abs(entry - stop)`.
   - Note in README: XAUUSD standard lot = 100 oz, so `lots = units / 100`. Keep the engine in oz; conversion is cosmetic.

5. **Equity vs balance.** `balance` = realized cash. `equity` = balance + unrealized P&L marked at current close. Equity curve uses `equity`. Drawdown computed off the equity curve.

6. **Margin / liquidation:** ignore for v1 (assume cash-funded, no leverage limit). Add a TODO. Do flag if equity goes ≤ 0 and stop the run.

### Engine config (dataclass)
```python
@dataclass
class BacktestConfig:
    opening_balance: float = 10_000
    spread: float = 0.30
    slippage: float = 0.0
    commission_per_trade: float = 0.0
    commission_per_unit: float = 0.0
    intrabar: str = "stop_first"
    entry_fill: str = "next_open"   # ("next_open" | "this_close") — default next_open
```

### Engine output
A `BacktestResult` object containing: the equity curve DataFrame (`time, equity, balance, drawdown`), the closed-trade list/DataFrame, the config used, and the data range. `metrics.py` consumes this.

---

## 6. Metrics (`metrics.py`)

Pure functions over `BacktestResult`. Compute and return a dict + a tidy DataFrame for display:

- **Returns:** total return %, absolute P&L, CAGR (annualize using actual elapsed time of the data range).
- **Trade stats:** number of trades, win rate, loss rate, average win, average loss, largest win/loss, average holding time (bars and wall-clock).
- **Profit factor** = gross profit / gross loss ("profit ratio").
- **Expectancy** per trade (in $ and in **R**).
- **R-multiples:** for every trade, `R = realized_pnl / initial_risk`, where initial risk = `abs(entry - stop) * size` at entry. Output the full R distribution (for a histogram) + average R, R std. *(This matters to me — I think in R for SMC setups.)*
- **Max drawdown** (% and absolute) and **max drawdown duration**.
- **Sharpe** and **Sortino** (optional; compute on per-bar equity returns, annualized — state the assumption in a tooltip).
- **Exposure %** (fraction of bars with an open position).

Guard against divide-by-zero (no losing trades → profit factor = inf; no trades → return zeros with a clear "no trades" flag).

---

## 7. UI (`app.py`, Streamlit)

### Sidebar (inputs)
- **Data file** picker (lists CSVs in `data/raw/`) + timeframe selector (use file as-is, or resample to M5/M15/H1/H4).
- **Date range** pickers (start/end), defaulting to the data's full range.
- **Strategy** dropdown (populated from the auto-discovery registry).
- **Strategy params** — render dynamically from the selected strategy's `params` schema (number inputs / sliders / selectboxes based on declared type + min/max). This is what makes "add a strategy → it just shows up configurable" work.
- **Account & cost params:** opening balance, spread, slippage, commission, intrabar rule, entry fill mode.
- **Run backtest** button.

### Main panel (outputs)
- **Metrics cards** row: total return, profit factor, win rate, max drawdown, expectancy (R), # trades.
- **Equity curve** (Plotly line) with the balance line overlaid; shaded drawdown underneath or a second drawdown subplot.
- **R-multiple histogram.**
- **Price chart** (candlestick) with entry/exit markers (green/red triangles, long/short) and SL/TP lines for the selected trade — optional but high-value for debugging a strategy; gate behind a checkbox if it's slow on a year of M5.
- **Trade log table:** entry/exit time, direction, entry, exit, size, SL, TP, P&L, R, exit reason, tag. Sortable. **Download as CSV** button. Also a **download equity curve PNG/CSV** button.
- Show the exact config used (for reproducibility) and run time.

Keep the UI thin — it just collects params, calls a single `run_backtest(config, strategy, data) -> BacktestResult`, and renders. No trading logic in the UI.

---

## 8. Reference strategies

### `ma_crossover.py` (example — must be the smoke test)
Simple, unambiguous, so the engine's correctness is easy to eyeball:
- Params: `fast` (default 20), `slow` (default 50), `risk_pct` (default 0.01), `sl_atr_mult`, `tp_atr_mult` (or fixed-distance SL/TP — pick one and document).
- Long when fast SMA crosses above slow; short on cross below. SL/TP set at entry (ATR- or fixed-distance based). Size via `ctx.size_for_risk`.
- This exists to validate the engine, not to make money.

### `smc_template.py` (stub for my real strategies)
A documented, **non-functional template** showing the patterns I'll use, with `# TODO` markers:
- How to read **HTF bias** via `ctx.htf["H1"]` / `ctx.htf["H4"]` while trading M5.
- How to **filter by session** (London killzone ~07:00–10:00 UTC, NY ~12:00–15:00 UTC) off `ctx.bar["time"]`.
- A placeholder for structure detection (swing highs/lows, BOS/CHoCH) with a stubbed helper signature.
- Entry with a **structural stop** (below/above the swing that defines the setup) and an R-based TP, size via risk %.
Make the template instructive — it's the file I'll copy to start each new strategy.

---

## 9. Tests (`tests/`) — prove the engine is honest

At minimum:
1. **No look-ahead:** a strategy that "peeks" by acting on the current bar's signal must still only fill at next open; verify fill prices.
2. **Long winner / loser** hand-computed: feed ~5 synthetic bars, known entry, known TP/SL hit, assert exact P&L including spread + commission.
3. **Short winner / loser:** same, mirrored.
4. **Intrabar both-in-range** bar → assert `stop_first` returns the stop fill; flip config → TP fill.
5. **Gap-through stop:** bar opens beyond the stop → fill at open (worse than stop), not at stop.
6. **Equity vs balance:** open position → equity reflects unrealized, balance unchanged until close.
7. **Metrics:** profit factor, win rate, R, max drawdown on a tiny known trade list.

`run_cli.py` should run a full backtest headless (no UI) on a sample file and print the metrics table — use it for fast iteration and as a sanity gate before opening the UI.

---

## 10. Non-goals (keep scope tight for v1)
- No live/paper trading, no broker/exchange connectivity, no order routing.
- No multi-asset portfolio (gold only; engine should stay asset-agnostic internally but UI targets one symbol at a time).
- No parameter optimization / walk-forward / Monte Carlo yet (design metrics + `run_backtest()` so a sweep can wrap it later).
- No pyramiding, no scaled/partial exits, no pending limit/stop entries (all noted as TODOs in the relevant modules).

---

## 11. Acceptance criteria
- I can drop `XAUUSD_m5.csv` into `data/raw/`, launch the Streamlit app, pick `ma_crossover`, set opening balance + date range + costs, click Run, and see an equity curve, metrics, and a downloadable trade log.
- All tests in §9 pass.
- Adding a new strategy file to `src/strategies/` makes it appear in the dropdown with its params, with **zero** edits to engine/UI/metrics.
- README documents: data format, how to add a strategy (with the `params` schema convention), every engine assumption from §5, and how to run tests + CLI.

---

### Notes for the data
I'll source XAUUSD M5 OHLC from Dukascopy (timestamps in UTC, tick volume). Build to the data contract in §3; don't assume `volume` is real exchange volume.