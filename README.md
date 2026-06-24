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
        # ctx.size_for_risk(risk_pct, entry, stop)                    -> units (oz)
        ...
```

## Engine assumptions (read before trusting numbers)

- **No look-ahead:** `on_bar` sees only bars `0..index`; orders fill at the **next bar's open**.
- **Intrabar SL/TP:** default `stop_first` (pessimistic) when both lie in a bar's range;
  configurable `tp_first` / `optimistic`. Fill at the bracket price; **gap-through** fills at the
  (worse) open.
- **Costs:** half-spread + slippage adverse on every fill; commission (`per_trade` + `per_unit`)
  charged once per round-trip at close.
- **Sizing/P&L** in units (oz): `pnl = (exit-entry)*size*(±1) - commission`. XAUUSD lot = 100 oz.
- **Equity vs balance:** `balance` = realized cash; `equity` = balance + unrealized at close;
  drawdown is computed off `equity`. Run stops if `equity <= 0`.
- **v1 limits (TODO):** one position at a time (no pyramiding), single TP (no partials),
  market entries only (no pending limit/stop entries), no margin/leverage modeling.
