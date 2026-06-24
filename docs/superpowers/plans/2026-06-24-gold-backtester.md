# Gold Backtester Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local single-user gold backtesting platform: fetch XAUUSD candles from Dukascopy, run plug-in strategies through a correctness-focused bar-by-bar engine, and view equity curve / metrics / trade log in a Streamlit UI.

**Architecture:** Layered. `data_fetcher` (Dukascopy → parquet cache) → `data_loader` (validate/resample) → `strategy` (Order/Position/Context/Strategy base) + `engine` (dumb simulator, owns fills/costs/brackets) → `metrics` (pure stats) → `runner` (single `run_backtest` seam) → strategies + CLI + Streamlit UI. The engine never makes trading decisions; strategies do.

**Tech Stack:** Python 3.13, pandas 3.0, numpy 2.5, dukascopy-python 4.0.1, streamlit 1.58, plotly 6.8, pyarrow 24.0, pytest 9.1.

**Design doc:** `docs/superpowers/specs/2026-06-24-gold-backtester-design.md`

**Conventions for every task:** TDD (write failing test → run red → implement → run green → commit). Exact file paths. Run pytest via `.venv/bin/pytest`. Commit at the end of each task.

---

### Task 0: Project scaffold + requirements

**Goal:** Create the directory tree, pinned `requirements.txt`, package `__init__.py` files, and confirm the venv installs everything.

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`, `src/strategies/__init__.py` (placeholder, replaced in Task 6)
- Create: `tests/__init__.py`
- Create: `data/raw/.gitkeep`
- Create: `conftest.py` (repo root — makes `src` importable in tests)

**Acceptance Criteria:**
- [ ] `.venv/bin/pip install -r requirements.txt` succeeds.
- [ ] `.venv/bin/python -c "import src"` works.
- [ ] `.venv/bin/pytest` runs (collects 0 tests, exit 0 or 5).

**Verify:** `.venv/bin/pip install -r requirements.txt && .venv/bin/python -c "import src; print('ok')"` → prints `ok`

**Steps:**

- [ ] **Step 1: Write `requirements.txt`** (versions verified installed on Python 3.13)

```
dukascopy-python==4.0.1
pandas==3.0.3
numpy==2.5.0
streamlit==1.58.0
plotly==6.8.0
pyarrow==24.0.0
pytest==9.1.1
```

- [ ] **Step 2: Create package files**

`src/__init__.py`: empty file.
`tests/__init__.py`: empty file.
`src/strategies/__init__.py`: empty for now (Task 6 adds auto-discovery).
`data/raw/.gitkeep`: empty file.

`conftest.py` (repo root):

```python
import sys
from pathlib import Path

# Make repo root importable so `import src...` works in tests.
sys.path.insert(0, str(Path(__file__).parent))
```

- [ ] **Step 3: Verify install + import**

Run: `.venv/bin/pip install -r requirements.txt && .venv/bin/python -c "import src; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt src tests conftest.py data/raw/.gitkeep
git commit -m "chore: scaffold project structure and pin requirements"
```

---

### Task 1: Dukascopy data fetcher

**Goal:** `fetch_ohlc()` downloads XAUUSD candles from Dukascopy and caches to parquet; a cache hit returns instantly without any network call.

**Files:**
- Create: `src/data_fetcher.py`
- Test: `tests/test_data_fetcher.py`

**Acceptance Criteria:**
- [ ] `fetch_ohlc` returns a DataFrame with columns `timestamp,open,high,low,close,volume`.
- [ ] First call writes a parquet under `data/cache/dukascopy/`; second call reads cache without invoking the downloader.
- [ ] Unsupported timeframe and missing dates raise `ValueError`.

**Verify:** `.venv/bin/pytest tests/test_data_fetcher.py -v` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

`tests/test_data_fetcher.py`:

```python
import datetime as dt
import pandas as pd
import pytest
from src import data_fetcher


def _fake_df():
    idx = pd.to_datetime(["2024-01-02 00:00", "2024-01-02 00:05"], utc=True)
    return pd.DataFrame(
        {"timestamp": idx, "open": [2000.0, 2001.0], "high": [2002.0, 2003.0],
         "low": [1999.0, 2000.5], "close": [2001.0, 2002.5], "volume": [1.0, 2.0]}
    )


def test_fetch_writes_then_reads_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(data_fetcher, "CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def fake_download(symbol, timeframe, start, end):
        calls["n"] += 1
        return _fake_df()

    monkeypatch.setattr(data_fetcher, "_download", fake_download)
    start, end = dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 3)

    df1 = data_fetcher.fetch_ohlc("XAUUSD", "m5", start, end)
    df2 = data_fetcher.fetch_ohlc("XAUUSD", "m5", start, end)

    assert calls["n"] == 1  # second call served from cache, no download
    assert list(df1.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(df2) == 2


def test_unsupported_timeframe_raises():
    with pytest.raises(ValueError):
        data_fetcher.fetch_ohlc("XAUUSD", "m7", dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 3))


def test_missing_dates_raise():
    with pytest.raises(ValueError):
        data_fetcher.fetch_ohlc("XAUUSD", "m5", None, None)
```

- [ ] **Step 2: Run red**

Run: `.venv/bin/pytest tests/test_data_fetcher.py -v`
Expected: FAIL (module/functions not defined).

- [ ] **Step 3: Implement `src/data_fetcher.py`**

```python
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

CACHE_DIR = Path("data/cache/dukascopy")

# UI/CLI timeframe string -> dukascopy_python INTERVAL_* attribute name.
_INTERVAL_MAP = {
    "m1": "INTERVAL_MIN_1", "m5": "INTERVAL_MIN_5", "m15": "INTERVAL_MIN_15",
    "m30": "INTERVAL_MIN_30", "h1": "INTERVAL_HOUR_1", "h4": "INTERVAL_HOUR_4",
    "d1": "INTERVAL_DAY_1",
}

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _cache_path(symbol: str, timeframe: str, start, end) -> Path:
    s, e = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    return CACHE_DIR / f"{symbol}_{timeframe}_{s}_{e}.parquet"


def fetch_ohlc(symbol="XAUUSD", timeframe="m5", start=None, end=None, use_cache=True) -> pd.DataFrame:
    """Fetch OHLC candles from Dukascopy, caching to parquet keyed on (symbol, tf, range)."""
    if start is None or end is None:
        raise ValueError("start and end datetimes are required")
    timeframe = timeframe.lower()
    if timeframe not in _INTERVAL_MAP:
        raise ValueError(f"unsupported timeframe {timeframe!r}; options: {list(_INTERVAL_MAP)}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(symbol, timeframe, start, end)
    if use_cache and path.exists():
        return pd.read_parquet(path)

    df = _download(symbol, timeframe, start, end)
    df.to_parquet(path)
    return df


def _download(symbol: str, timeframe: str, start, end) -> pd.DataFrame:
    """Hit the Dukascopy feed. Isolated so tests can monkeypatch it (no network)."""
    import dukascopy_python as d
    from dukascopy_python import instruments

    interval = getattr(d, _INTERVAL_MAP[timeframe])
    # "XAUUSD" -> INSTRUMENT_FX_METALS_XAU_USD
    instr_const = f"INSTRUMENT_FX_METALS_{symbol[:3]}_{symbol[3:]}"
    instrument = getattr(instruments, instr_const)

    raw = d.fetch(instrument, interval, d.OFFER_SIDE_BID, start, end)
    df = raw.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})
    return df[COLUMNS]
```

- [ ] **Step 4: Run green**

Run: `.venv/bin/pytest tests/test_data_fetcher.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/data_fetcher.py tests/test_data_fetcher.py
git commit -m "feat: Dukascopy data fetcher with parquet cache"
```

---

### Task 2: Data loader (validate / slice / resample)

**Goal:** `load_ohlc()` normalizes & validates OHLC into a UTC-indexed frame; `resample()` produces higher-timeframe OHLC.

**Files:**
- Create: `src/data_loader.py`
- Test: `tests/test_data_loader.py`

**Acceptance Criteria:**
- [ ] Accepts both a CSV path and an in-memory DataFrame; returns UTC `DatetimeIndex` sorted ascending, deduped.
- [ ] Rejects: missing columns, NaNs in OHLC, `high<low`, `high<open/close`, `low>open/close`, non-positive prices, non-monotonic-after-sort impossible (sorted), naive timestamps without `tz`.
- [ ] `resample` aggregates `open=first,high=max,low=min,close=last,volume=sum`, right-labeled, drops empty (gap) bins.

**Verify:** `.venv/bin/pytest tests/test_data_loader.py -v` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

`tests/test_data_loader.py`:

```python
import pandas as pd
import pytest
from src import data_loader


def _good_df():
    idx = pd.to_datetime(
        ["2024-01-02 00:00", "2024-01-02 00:01", "2024-01-02 00:02"], utc=True
    )
    return pd.DataFrame(
        {"timestamp": idx, "open": [2000, 2001, 2002], "high": [2003, 2004, 2005],
         "low": [1999, 2000, 2001], "close": [2001, 2002, 2003], "volume": [1, 1, 1]}
    )


def test_load_sorts_and_indexes_utc():
    df = _good_df().sample(frac=1, random_state=1)  # shuffle rows
    out = data_loader.load_ohlc(df)
    assert out.index.is_monotonic_increasing
    assert str(out.index.tz) == "UTC"
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_missing_column_raises():
    df = _good_df().drop(columns=["volume"])
    with pytest.raises(ValueError):
        data_loader.load_ohlc(df)


def test_high_below_low_raises():
    df = _good_df()
    df.loc[1, "high"] = df.loc[1, "low"] - 1
    with pytest.raises(ValueError):
        data_loader.load_ohlc(df)


def test_naive_timestamp_requires_tz():
    df = _good_df()
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    with pytest.raises(ValueError):
        data_loader.load_ohlc(df)
    out = data_loader.load_ohlc(df, tz="UTC")  # tz supplied -> ok
    assert str(out.index.tz) == "UTC"


def test_resample_m1_to_m5():
    idx = pd.date_range("2024-01-02 00:00", periods=10, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {"open": range(10), "high": [v + 2 for v in range(10)],
         "low": [v - 1 for v in range(10)], "close": [v + 1 for v in range(10)],
         "volume": [1] * 10}, index=idx)
    df.index.name = "timestamp"
    out = data_loader.resample(df, "m5")
    assert out["volume"].iloc[0] == 5
    assert out["high"].iloc[0] == max(v + 2 for v in range(5))
```

- [ ] **Step 2: Run red**

Run: `.venv/bin/pytest tests/test_data_loader.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/data_loader.py`**

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]
PRICE_COLS = ["open", "high", "low", "close"]

_RESAMPLE_RULE = {
    "m1": "1min", "m5": "5min", "m15": "15min", "m30": "30min",
    "h1": "1h", "h4": "4h", "d1": "1D",
}


def load_ohlc(source, start=None, end=None, tz=None) -> pd.DataFrame:
    """Load OHLC from a CSV path or DataFrame; validate, sort, dedupe, slice. UTC-indexed."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source)

    df = _normalize(df, tz)
    _validate(df)

    if start is not None:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    return df


def _normalize(df: pd.DataFrame, tz) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    df = df[REQUIRED_COLS].copy()
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        if tz is None:
            raise ValueError("naive timestamps; pass tz= to localize")
        ts = ts.dt.tz_localize(tz).dt.tz_convert("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    df["timestamp"] = ts

    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    df = df.set_index("timestamp")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df


def _validate(df: pd.DataFrame) -> None:
    if df[PRICE_COLS].isna().any().any():
        raise ValueError("NaNs present in OHLC columns")
    if not df.index.is_monotonic_increasing:
        raise ValueError("timestamps not monotonic after sort (duplicates?)")
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    if (h < l).any():
        raise ValueError("invalid bar: high < low")
    if (h < o).any() or (h < c).any():
        raise ValueError("invalid bar: high < open/close")
    if (l > o).any() or (l > c).any():
        raise ValueError("invalid bar: low > open/close")
    if (df[PRICE_COLS] <= 0).any().any():
        raise ValueError("non-positive prices present")


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Standard OHLC resample, right-labeled, dropping empty (gap) bins."""
    tf = timeframe.lower()
    if tf not in _RESAMPLE_RULE:
        raise ValueError(f"unsupported timeframe {tf!r}; options: {list(_RESAMPLE_RULE)}")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(_RESAMPLE_RULE[tf], label="right", closed="right").agg(agg)
    return out.dropna(subset=["open"])
```

- [ ] **Step 4: Run green**

Run: `.venv/bin/pytest tests/test_data_loader.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/data_loader.py tests/test_data_loader.py
git commit -m "feat: data loader with validation and resampling"
```

---

### Task 3: Strategy interface (Order, Position, Context, Strategy)

**Goal:** Define the plug-in contract: dataclasses + a `Context` that records order intents and exposes data/account state, plus the `Strategy` base with a params schema.

**Files:**
- Create: `src/strategy.py`
- Test: `tests/test_strategy.py`

**Acceptance Criteria:**
- [ ] `Context.enter()` records an `Order`; `Context.close()` sets a close flag; both are readable after `on_bar`.
- [ ] `Context.bar` returns the current OHLC dict; `Context.history` returns bars `0..index` inclusive.
- [ ] `size_for_risk` returns `(equity*risk_pct)/abs(entry-stop)`, and `0.0` when stop==entry.
- [ ] `Strategy(**params)` merges params over schema defaults into `self.p`.

**Verify:** `.venv/bin/pytest tests/test_strategy.py -v` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

`tests/test_strategy.py`:

```python
import pandas as pd
from src.strategy import Context, Strategy, Order


def _data():
    idx = pd.date_range("2024-01-02", periods=3, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open": [10, 11, 12], "high": [12, 13, 14], "low": [9, 10, 11],
         "close": [11, 12, 13], "volume": [1, 1, 1]}, index=idx)


def test_enter_records_order():
    ctx = Context(_data(), index=1, position=None, equity=1000, balance=1000, htf={})
    ctx.enter("long", size=2, stop_loss=10.0, take_profit=15.0, tag="t")
    assert isinstance(ctx._order, Order)
    assert ctx._order.direction == "long" and ctx._order.size == 2


def test_bar_and_history():
    ctx = Context(_data(), index=1, position=None, equity=1000, balance=1000, htf={})
    assert ctx.bar["close"] == 12
    assert len(ctx.history) == 2  # bars 0..1 inclusive


def test_size_for_risk():
    ctx = Context(_data(), index=0, position=None, equity=10000, balance=10000, htf={})
    assert ctx.size_for_risk(0.01, entry_price=2000, stop_price=1990) == 10.0
    assert ctx.size_for_risk(0.01, entry_price=2000, stop_price=2000) == 0.0


def test_strategy_param_merge():
    class S(Strategy):
        params = {"fast": ("int", 20, 1, 100, "fast ma")}
    s = S(fast=5)
    assert s.p["fast"] == 5
    assert S().p["fast"] == 20
```

- [ ] **Step 2: Run red**

Run: `.venv/bin/pytest tests/test_strategy.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/strategy.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


@dataclass
class Order:
    direction: str                     # "long" | "short"
    size: float                        # units (oz), > 0
    stop_loss: Optional[float] = None  # absolute price
    take_profit: Optional[float] = None
    tag: str = ""


@dataclass
class Position:
    direction: str
    size: float
    entry_price: float
    entry_time: Any
    entry_index: int
    stop_loss: Optional[float]
    take_profit: Optional[float]
    initial_risk: Optional[float]      # abs(entry_fill - stop) * size, set at fill
    tag: str = ""


class Context:
    """Per-bar read-only view of state + order API. Records intent; engine executes it."""

    def __init__(self, data, index, position, equity, balance, htf):
        self.data = data
        self.index = index
        self.position = position
        self.equity = equity
        self.balance = balance
        self.htf = htf
        self._order: Optional[Order] = None
        self._close_requested: bool = False
        self._close_reason: str = "manual"

    @property
    def bar(self) -> dict:
        row = self.data.iloc[self.index]
        return {
            "time": self.data.index[self.index],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }

    @property
    def history(self) -> pd.DataFrame:
        """All bars 0..index inclusive (no look-ahead)."""
        return self.data.iloc[: self.index + 1]

    def enter(self, direction, size, stop_loss=None, take_profit=None, tag=""):
        if direction not in ("long", "short"):
            raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")
        if size is None or size <= 0:
            return  # ignore non-positive sizing
        self._order = Order(direction, float(size), stop_loss, take_profit, tag)

    def close(self, reason="manual"):
        self._close_requested = True
        self._close_reason = reason

    def size_for_risk(self, risk_pct, entry_price, stop_price) -> float:
        risk_per_unit = abs(entry_price - stop_price)
        if risk_per_unit <= 0:
            return 0.0
        return (self.equity * risk_pct) / risk_per_unit


class Strategy:
    name: str = "Unnamed"
    # schema entry: name -> (type, default, min, max, help)
    params: dict = {}
    # higher-timeframe views the strategy wants (e.g. ["h1", "h4"]); built by runner.
    htf_timeframes: list = []

    def __init__(self, **params):
        merged = {k: spec[1] for k, spec in self.params.items()}
        merged.update(params)
        self.p = merged

    def on_start(self, ctx: Context): ...
    def on_bar(self, ctx: Context): ...
    def on_finish(self, ctx: Context): ...
```

- [ ] **Step 4: Run green**

Run: `.venv/bin/pytest tests/test_strategy.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/strategy.py tests/test_strategy.py
git commit -m "feat: strategy interface (Order/Position/Context/Strategy)"
```

---

### Task 4: Engine (the honest simulator) + correctness tests

**Goal:** Implement the bar-by-bar engine with next-bar-open fills, cost model, intrabar bracket resolution, gap-through, equity/balance tracking — and prove it with hand-computed correctness tests.

**Files:**
- Create: `src/engine.py`
- Test: `tests/test_engine.py`

**Acceptance Criteria:**
- [ ] No look-ahead: an order requested on bar `t` fills at bar `t+1` open (+ costs).
- [ ] Long & short winner/loser P&L match hand computation incl. spread + commission.
- [ ] Intrabar both-in-range → `stop_first` gives stop fill; `tp_first` gives TP fill.
- [ ] Gap-through: bar opens beyond stop → fill at open (worse than stop).
- [ ] Equity reflects unrealized while open; balance only changes on close.
- [ ] `equity <= 0` stops the run with `stopped_out=True`.

**Verify:** `.venv/bin/pytest tests/test_engine.py -v` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

`tests/test_engine.py`:

```python
import pandas as pd
import pytest

from src.engine import BacktestConfig, run_backtest
from src.strategy import Strategy


def make_data(bars):
    """bars: list of (open, high, low, close)."""
    idx = pd.date_range("2024-01-02", periods=len(bars), freq="5min", tz="UTC")
    df = pd.DataFrame(bars, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1.0
    df.index.name = "timestamp"
    return df


class EnterOnceLong(Strategy):
    """Enter long on the first bar; fixed SL/TP; never re-enter."""
    name = "enter_once_long"

    def __init__(self, size=1.0, sl=None, tp=None, **kw):
        super().__init__(**kw)
        self.size, self.sl, self.tp, self.done = size, sl, tp, False

    def on_bar(self, ctx):
        if not self.done and ctx.position is None:
            ctx.enter("long", self.size, stop_loss=self.sl, take_profit=self.tp)
            self.done = True


class EnterOnceShort(EnterOnceLong):
    name = "enter_once_short"

    def on_bar(self, ctx):
        if not self.done and ctx.position is None:
            ctx.enter("short", self.size, stop_loss=self.sl, take_profit=self.tp)
            self.done = True


def test_no_lookahead_fills_next_open():
    # Signal on bar 0; must fill at bar 1 open, not bar 0 close.
    data = make_data([(100, 101, 99, 100), (110, 111, 109, 110), (110, 111, 109, 110)])
    cfg = BacktestConfig(spread=0.0, slippage=0.0)
    strat = EnterOnceLong(size=1.0, sl=50, tp=1000)  # wide so it stays open
    res = run_backtest(cfg, strat, data)
    # position opened at bar 1 open = 110 (no costs)
    assert strat.done
    # equity at end marks unrealized vs entry 110, close 110 -> 0 pnl, equity == opening
    assert res.equity_curve["equity"].iloc[-1] == pytest.approx(cfg.opening_balance)


def test_long_winner_pnl_with_costs():
    # Enter bar0 -> fill bar1 open. TP hit on bar2. spread=0.2, commission_per_trade=1.
    data = make_data([(100, 100, 100, 100), (100, 100, 100, 100), (100, 110, 100, 105)])
    cfg = BacktestConfig(spread=0.2, slippage=0.0, commission_per_trade=1.0)
    strat = EnterOnceLong(size=2.0, sl=90, tp=108)
    res = run_backtest(cfg, strat, data)
    t = res.trades.iloc[0]
    # entry fill = 100 + 0.2/2 = 100.1 ; tp base 108 -> sell fill = 108 - 0.1 = 107.9
    # pnl = (107.9 - 100.1) * 2 - 1 = 15.6 - 1 = 14.6
    assert t["entry_price"] == pytest.approx(100.1)
    assert t["exit_price"] == pytest.approx(107.9)
    assert t["pnl"] == pytest.approx(14.6)
    assert t["exit_reason"] == "tp"


def test_short_loser_pnl():
    data = make_data([(100, 100, 100, 100), (100, 100, 100, 100), (100, 112, 100, 110)])
    cfg = BacktestConfig(spread=0.0, slippage=0.0, commission_per_trade=0.0)
    strat = EnterOnceShort(size=1.0, sl=110, tp=80)
    res = run_backtest(cfg, strat, data)
    t = res.trades.iloc[0]
    # entry (sell) fill = 100 ; stop base 110 -> exit (buy) fill = 110
    # short pnl = (110 - 100) * 1 * -1 = -10
    assert t["pnl"] == pytest.approx(-10.0)
    assert t["exit_reason"] == "stop"


def test_intrabar_stop_first_vs_tp_first():
    # bar2 range contains both stop(95) and tp(108): low 94, high 110
    data = make_data([(100, 100, 100, 100), (100, 100, 100, 100), (100, 110, 94, 100)])
    strat_kw = dict(size=1.0, sl=95, tp=108)
    res_stop = run_backtest(BacktestConfig(spread=0, slippage=0), EnterOnceLong(**strat_kw), data)
    res_tp = run_backtest(BacktestConfig(spread=0, slippage=0, intrabar="tp_first"),
                          EnterOnceLong(**strat_kw), data)
    assert res_stop.trades.iloc[0]["exit_reason"] == "stop"
    assert res_tp.trades.iloc[0]["exit_reason"] == "tp"


def test_gap_through_stop_fills_at_open():
    # Long entered at bar1 open=100, stop=95. bar2 opens at 90 (gaps below stop).
    data = make_data([(100, 100, 100, 100), (100, 100, 100, 100), (90, 92, 88, 91)])
    cfg = BacktestConfig(spread=0.0, slippage=0.0)
    strat = EnterOnceLong(size=1.0, sl=95, tp=200)
    res = run_backtest(cfg, strat, data)
    t = res.trades.iloc[0]
    assert t["exit_price"] == pytest.approx(90.0)  # filled at open, worse than 95
    assert t["exit_reason"] == "stop"


def test_equity_vs_balance_open_position():
    data = make_data([(100, 100, 100, 100), (100, 100, 100, 100), (100, 105, 100, 105)])
    cfg = BacktestConfig(spread=0.0, slippage=0.0)
    strat = EnterOnceLong(size=1.0, sl=50, tp=1000)  # stays open
    res = run_backtest(cfg, strat, data)
    ec = res.equity_curve
    # entered bar1 @100; bar2 close 105 -> unrealized +5; balance unchanged
    assert ec["balance"].iloc[-1] == pytest.approx(cfg.opening_balance)
    assert ec["equity"].iloc[-1] == pytest.approx(cfg.opening_balance + 5)
```

- [ ] **Step 2: Run red**

Run: `.venv/bin/pytest tests/test_engine.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/engine.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .strategy import Context, Position, Strategy


@dataclass
class BacktestConfig:
    opening_balance: float = 10_000.0
    spread: float = 0.30
    slippage: float = 0.0
    commission_per_trade: float = 0.0
    commission_per_unit: float = 0.0
    intrabar: str = "stop_first"   # "stop_first" | "tp_first" | "optimistic"
    entry_fill: str = "next_open"  # "next_open" only in v1 (kept for forward-compat)


@dataclass
class Trade:
    entry_time: object
    exit_time: object
    direction: str
    size: float
    entry_price: float
    exit_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    pnl: float
    r_multiple: Optional[float]
    exit_reason: str
    tag: str
    bars_held: int
    initial_risk: Optional[float]


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame   # index=time, cols: equity, balance, peak, drawdown, drawdown_pct
    trades: pd.DataFrame
    config: BacktestConfig
    data_start: object
    data_end: object
    timeframe_seconds: float
    stopped_out: bool = False


# --- cost helpers: every fill pays half-spread + slippage adverse to its side ---
def _buy_fill(base, cfg):
    return base + cfg.spread / 2 + cfg.slippage


def _sell_fill(base, cfg):
    return base - cfg.spread / 2 - cfg.slippage


def _commission(size, cfg):
    return cfg.commission_per_trade + cfg.commission_per_unit * size


def _trade_pnl(pos: Position, exit_fill: float, cfg: BacktestConfig) -> float:
    sign = 1 if pos.direction == "long" else -1
    gross = (exit_fill - pos.entry_price) * pos.size * sign
    return gross - _commission(pos.size, cfg)


def _unrealized(pos: Optional[Position], price: float) -> float:
    if pos is None:
        return 0.0
    sign = 1 if pos.direction == "long" else -1
    return (price - pos.entry_price) * pos.size * sign


def _resolve_bracket(pos: Position, o, h, l, cfg: BacktestConfig):
    """Return (reason, base_price) if SL/TP fires on this bar, else None.
    Handles gap-through at the open and the stop/tp intrabar policy."""
    sl, tp = pos.stop_loss, pos.take_profit
    stop_hit = tp_hit = False
    stop_base = tp_base = None

    if pos.direction == "long":
        if sl is not None:
            if o <= sl:            # gapped through the stop at the open
                stop_hit, stop_base = True, o
            elif l <= sl:
                stop_hit, stop_base = True, sl
        if tp is not None:
            if o >= tp:            # gapped through the target at the open
                tp_hit, tp_base = True, o
            elif h >= tp:
                tp_hit, tp_base = True, tp
    else:  # short
        if sl is not None:
            if o >= sl:
                stop_hit, stop_base = True, o
            elif h >= sl:
                stop_hit, stop_base = True, sl
        if tp is not None:
            if o <= tp:
                tp_hit, tp_base = True, o
            elif l <= tp:
                tp_hit, tp_base = True, tp

    if stop_hit and tp_hit:
        if cfg.intrabar == "tp_first" or cfg.intrabar == "optimistic":
            return ("tp", tp_base)
        return ("stop", stop_base)          # "stop_first" default (pessimistic)
    if stop_hit:
        return ("stop", stop_base)
    if tp_hit:
        return ("tp", tp_base)
    return None


def _make_trade(pos, exit_time, exit_fill, pnl, reason, exit_index) -> Trade:
    r = pnl / pos.initial_risk if pos.initial_risk not in (None, 0) else None
    return Trade(
        entry_time=pos.entry_time, exit_time=exit_time, direction=pos.direction,
        size=pos.size, entry_price=pos.entry_price, exit_price=exit_fill,
        stop_loss=pos.stop_loss, take_profit=pos.take_profit, pnl=pnl, r_multiple=r,
        exit_reason=reason, tag=pos.tag, bars_held=exit_index - pos.entry_index,
        initial_risk=pos.initial_risk,
    )


def _slice_htf(htf, current_time):
    # Only HTF bars that have fully closed (right-labeled close <= now) -> no look-ahead.
    return {k: df[df.index <= current_time] for k, df in htf.items()}


def run_backtest(config: BacktestConfig, strategy: Strategy, data: pd.DataFrame,
                 htf: Optional[dict] = None) -> BacktestResult:
    cfg = config
    htf = htf or {}
    balance = cfg.opening_balance
    equity = balance
    position: Optional[Position] = None
    pending_order = None
    pending_close = False
    pending_close_reason = "manual"
    trades: list[Trade] = []
    rows = []
    stopped = False

    times = data.index
    opens = data["open"].to_numpy(dtype=float)
    highs = data["high"].to_numpy(dtype=float)
    lows = data["low"].to_numpy(dtype=float)
    closes = data["close"].to_numpy(dtype=float)
    n = len(data)

    strategy.on_start(Context(data, 0, None, equity, balance, _slice_htf(htf, times[0])))

    i = 0
    for i in range(n):
        o, h, l, c, t = opens[i], highs[i], lows[i], closes[i], times[i]

        # 1) discretionary close requested last bar fills at THIS open (first event of bar)
        if position is not None and pending_close:
            base = o
            exit_fill = _sell_fill(base, cfg) if position.direction == "long" else _buy_fill(base, cfg)
            pnl = _trade_pnl(position, exit_fill, cfg)
            balance += pnl
            trades.append(_make_trade(position, t, exit_fill, pnl, pending_close_reason, i))
            position = None
        pending_close = False

        # 2) manage existing position: SL/TP may fire intrabar
        if position is not None:
            hit = _resolve_bracket(position, o, h, l, cfg)
            if hit is not None:
                reason, base = hit
                exit_fill = _sell_fill(base, cfg) if position.direction == "long" else _buy_fill(base, cfg)
                pnl = _trade_pnl(position, exit_fill, cfg)
                balance += pnl
                trades.append(_make_trade(position, t, exit_fill, pnl, reason, i))
                position = None

        # 3) execute pending entry at THIS open, if flat
        if pending_order is not None and position is None:
            order = pending_order
            entry_fill = _buy_fill(o, cfg) if order.direction == "long" else _sell_fill(o, cfg)
            init_risk = (abs(entry_fill - order.stop_loss) * order.size
                         if order.stop_loss is not None else None)
            position = Position(order.direction, order.size, entry_fill, t, i,
                                order.stop_loss, order.take_profit, init_risk, order.tag)
        pending_order = None

        # 4) mark-to-market at this close (so ctx.equity is current for sizing)
        equity = balance + _unrealized(position, c)

        # 5) ask strategy for decisions using data up to this bar
        ctx = Context(data, i, position, equity, balance, _slice_htf(htf, t))
        strategy.on_bar(ctx)
        if ctx._order is not None and position is None:
            pending_order = ctx._order         # pyramiding not supported: ignore if in position
        if ctx._close_requested and position is not None:
            pending_close = True
            pending_close_reason = ctx._close_reason

        rows.append((t, equity, balance))

        if equity <= 0:
            stopped = True
            break

    strategy.on_finish(Context(data, i, position, equity, balance, _slice_htf(htf, times[i])))

    ec = pd.DataFrame(rows, columns=["time", "equity", "balance"]).set_index("time")
    peak = ec["equity"].cummax()
    ec["peak"] = peak
    ec["drawdown"] = ec["equity"] - peak
    ec["drawdown_pct"] = ec["drawdown"] / peak

    trades_df = pd.DataFrame([t.__dict__ for t in trades])

    diffs = np.diff(times.view("int64")) / 1e9 if n > 1 else np.array([0.0])
    tf_seconds = float(np.median(diffs)) if n > 1 else 0.0

    return BacktestResult(
        equity_curve=ec, trades=trades_df, config=cfg,
        data_start=times[0], data_end=times[-1],
        timeframe_seconds=tf_seconds, stopped_out=stopped,
    )
```

- [ ] **Step 4: Run green**

Run: `.venv/bin/pytest tests/test_engine.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/engine.py tests/test_engine.py
git commit -m "feat: backtest engine with bracket/cost/look-ahead correctness tests"
```

---

### Task 5: Metrics

**Goal:** Pure `compute_metrics(result)` returning a stats dict + tidy DataFrame, guarding divide-by-zero.

**Files:**
- Create: `src/metrics.py`
- Test: `tests/test_metrics.py`

**Acceptance Criteria:**
- [ ] Profit factor, win rate, expectancy ($ and R), avg/largest win/loss correct on a known trade set.
- [ ] Max drawdown (abs + pct) and duration computed from the equity curve.
- [ ] No trades → zeros + `no_trades=True`; no losses → profit factor `inf`.
- [ ] Returns `(metrics_dict, summary_df)`.

**Verify:** `.venv/bin/pytest tests/test_metrics.py -v` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:

```python
import numpy as np
import pandas as pd

from src.engine import BacktestConfig, BacktestResult
from src import metrics


def _result(pnls, rs, opening=10_000):
    trades = pd.DataFrame({
        "pnl": pnls, "r_multiple": rs,
        "entry_time": pd.date_range("2024-01-02", periods=len(pnls), freq="1h", tz="UTC"),
        "exit_time": pd.date_range("2024-01-02 00:30", periods=len(pnls), freq="1h", tz="UTC"),
        "bars_held": [6] * len(pnls),
    })
    eq = opening + np.cumsum([0.0] + list(pnls))
    idx = pd.date_range("2024-01-02", periods=len(eq), freq="1h", tz="UTC")
    ec = pd.DataFrame({"equity": eq, "balance": eq}, index=idx)
    ec["peak"] = ec["equity"].cummax()
    ec["drawdown"] = ec["equity"] - ec["peak"]
    ec["drawdown_pct"] = ec["drawdown"] / ec["peak"]
    return BacktestResult(ec, trades, BacktestConfig(opening_balance=opening),
                          idx[0], idx[-1], 3600.0)


def test_basic_stats():
    m, df = metrics.compute_metrics(_result([100, -50, 200, -50], [2, -1, 4, -1]))
    assert m["num_trades"] == 4
    assert m["win_rate"] == 0.5
    assert m["profit_factor"] == (300) / (100)  # gross win 300 / gross loss 100
    assert m["expectancy_r"] == np.mean([2, -1, 4, -1])
    assert isinstance(df, pd.DataFrame)


def test_no_trades_flag():
    m, _ = metrics.compute_metrics(_result([], []))
    assert m["no_trades"] is True
    assert m["num_trades"] == 0


def test_profit_factor_inf_when_no_losses():
    m, _ = metrics.compute_metrics(_result([100, 50], [1, 1]))
    assert m["profit_factor"] == float("inf")


def test_max_drawdown():
    # equity goes 10000 -> 10100 -> 10050 -> 10250 -> 10200 ; max dd from 10100->10050 etc.
    m, _ = metrics.compute_metrics(_result([100, -50, 200, -50], [1, -1, 1, -1]))
    assert m["max_drawdown_abs"] <= 0  # drawdown stored as <= 0
    assert m["max_drawdown_pct"] <= 0
```

- [ ] **Step 2: Run red**

Run: `.venv/bin/pytest tests/test_metrics.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/metrics.py`**

```python
from __future__ import annotations

import numpy as np
import pandas as pd

SECONDS_PER_YEAR = 365.25 * 24 * 3600


def compute_metrics(result) -> tuple[dict, pd.DataFrame]:
    trades = result.trades
    ec = result.equity_curve
    opening = result.config.opening_balance
    final_equity = float(ec["equity"].iloc[-1]) if len(ec) else opening

    m: dict = {}
    m["opening_balance"] = opening
    m["final_equity"] = final_equity
    m["absolute_pnl"] = final_equity - opening
    m["total_return_pct"] = (final_equity / opening - 1.0) if opening else 0.0

    # elapsed time / CAGR
    elapsed_s = (result.data_end - result.data_start).total_seconds()
    years = elapsed_s / SECONDS_PER_YEAR if elapsed_s > 0 else 0.0
    if years > 0 and opening > 0 and final_equity > 0:
        m["cagr"] = (final_equity / opening) ** (1 / years) - 1.0
    else:
        m["cagr"] = 0.0

    n = 0 if trades is None or trades.empty else len(trades)
    m["num_trades"] = n
    m["no_trades"] = n == 0

    if n == 0:
        m.update({
            "win_rate": 0.0, "loss_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "largest_win": 0.0, "largest_loss": 0.0, "profit_factor": 0.0,
            "expectancy_dollars": 0.0, "expectancy_r": 0.0, "avg_r": 0.0, "r_std": 0.0,
            "avg_bars_held": 0.0, "exposure_pct": 0.0,
            "max_drawdown_abs": float(ec["drawdown"].min()) if len(ec) else 0.0,
            "max_drawdown_pct": float(ec["drawdown_pct"].min()) if len(ec) else 0.0,
            "max_drawdown_duration_bars": 0, "sharpe": 0.0, "sortino": 0.0,
        })
        return m, _summary_df(m)

    pnl = trades["pnl"].to_numpy(dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())

    m["win_rate"] = len(wins) / n
    m["loss_rate"] = len(losses) / n
    m["avg_win"] = float(wins.mean()) if len(wins) else 0.0
    m["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
    m["largest_win"] = float(wins.max()) if len(wins) else 0.0
    m["largest_loss"] = float(losses.min()) if len(losses) else 0.0
    m["profit_factor"] = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    m["expectancy_dollars"] = float(pnl.mean())

    r = trades["r_multiple"].dropna().to_numpy(dtype=float)
    m["expectancy_r"] = float(r.mean()) if len(r) else 0.0
    m["avg_r"] = m["expectancy_r"]
    m["r_std"] = float(r.std(ddof=0)) if len(r) else 0.0
    m["avg_bars_held"] = float(trades["bars_held"].mean())

    # drawdown
    m["max_drawdown_abs"] = float(ec["drawdown"].min())
    m["max_drawdown_pct"] = float(ec["drawdown_pct"].min())
    m["max_drawdown_duration_bars"] = _max_dd_duration(ec["drawdown"].to_numpy())

    # exposure: fraction of bars holding a position (bars_held sum / total bars)
    total_bars = len(ec)
    m["exposure_pct"] = float(trades["bars_held"].sum() / total_bars) if total_bars else 0.0

    # Sharpe / Sortino on per-bar equity returns, annualized (risk-free = 0)
    rets = ec["equity"].pct_change().dropna().to_numpy()
    if len(rets) > 1 and rets.std(ddof=0) > 0:
        ppy = SECONDS_PER_YEAR / result.timeframe_seconds if result.timeframe_seconds else 0.0
        ann = np.sqrt(ppy) if ppy > 0 else 0.0
        m["sharpe"] = float(rets.mean() / rets.std(ddof=0) * ann)
        downside = rets[rets < 0]
        m["sortino"] = (float(rets.mean() / downside.std(ddof=0) * ann)
                        if len(downside) and downside.std(ddof=0) > 0 else 0.0)
    else:
        m["sharpe"] = 0.0
        m["sortino"] = 0.0

    return m, _summary_df(m)


def _max_dd_duration(dd: np.ndarray) -> int:
    """Longest run of consecutive bars with drawdown < 0."""
    best = cur = 0
    for v in dd:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return int(best)


def _summary_df(m: dict) -> pd.DataFrame:
    order = [
        ("Total return %", m["total_return_pct"] * 100),
        ("Absolute P&L", m["absolute_pnl"]),
        ("CAGR %", m["cagr"] * 100),
        ("# Trades", m["num_trades"]),
        ("Win rate %", m["win_rate"] * 100),
        ("Profit factor", m["profit_factor"]),
        ("Expectancy ($)", m["expectancy_dollars"]),
        ("Expectancy (R)", m["expectancy_r"]),
        ("Avg R", m["avg_r"]),
        ("Max drawdown %", m["max_drawdown_pct"] * 100),
        ("Max drawdown $", m["max_drawdown_abs"]),
        ("Sharpe", m["sharpe"]),
        ("Sortino", m["sortino"]),
        ("Exposure %", m["exposure_pct"] * 100),
    ]
    return pd.DataFrame(order, columns=["metric", "value"])
```

- [ ] **Step 4: Run green**

Run: `.venv/bin/pytest tests/test_metrics.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: metrics (profit factor, R, drawdown, sharpe/sortino)"
```

---

### Task 6: Runner + MA crossover strategy + auto-discovery + CLI

**Goal:** Wire `runner.run_backtest` (builds HTF + delegates to engine), implement the `ma_crossover` smoke-test strategy, the strategy auto-discovery registry, and a headless CLI that prints metrics end-to-end.

**Files:**
- Create: `src/runner.py`
- Create: `src/strategies/ma_crossover.py`
- Modify: `src/strategies/__init__.py` (auto-discovery registry)
- Create: `run_cli.py`
- Test: `tests/test_runner.py`

**Acceptance Criteria:**
- [ ] `get_strategy_registry()` discovers `MACrossover` by `name` with zero engine/UI edits.
- [ ] `runner.run_backtest(config, strategy, data)` builds declared HTF views and returns a `BacktestResult`.
- [ ] `MACrossover` produces ≥1 trade on a synthetic trending dataset.
- [ ] `run_cli.py --synthetic` runs a full backtest and prints a metrics table.

**Verify:** `.venv/bin/pytest tests/test_runner.py -v && .venv/bin/python run_cli.py --synthetic` → tests pass; CLI prints metrics

**Steps:**

- [ ] **Step 1: Write the failing tests**

`tests/test_runner.py`:

```python
import numpy as np
import pandas as pd

from src import runner
from src.strategies import get_strategy_registry
from src.engine import BacktestConfig


def _trending_data(n=300):
    idx = pd.date_range("2024-01-02", periods=n, freq="5min", tz="UTC")
    # up then down so a crossover strategy trades both sides
    base = np.concatenate([np.linspace(2000, 2100, n // 2), np.linspace(2100, 1980, n - n // 2)])
    df = pd.DataFrame({
        "open": base, "high": base + 1.0, "low": base - 1.0, "close": base, "volume": 1.0,
    }, index=idx)
    df.index.name = "timestamp"
    return df


def test_registry_discovers_ma_crossover():
    reg = get_strategy_registry()
    assert any("crossover" in name.lower() for name in reg)


def test_runner_runs_ma_crossover():
    reg = get_strategy_registry()
    cls = next(c for name, c in reg.items() if "crossover" in name.lower())
    strat = cls(fast=5, slow=20, risk_pct=0.01)
    res = runner.run_backtest(BacktestConfig(), strat, _trending_data())
    assert res.equity_curve.shape[0] == 300
    assert res.trades.shape[0] >= 1
```

- [ ] **Step 2: Run red**

Run: `.venv/bin/pytest tests/test_runner.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/runner.py`**

```python
from __future__ import annotations

from typing import Optional

import pandas as pd

from .data_loader import resample
from .engine import BacktestConfig, BacktestResult, run_backtest as _engine_run
from .strategy import Strategy


def run_backtest(config: BacktestConfig, strategy: Strategy,
                 data: pd.DataFrame, htf: Optional[dict] = None) -> BacktestResult:
    """Public entry point shared by CLI/UI. Builds the HTF views the strategy declares,
    then delegates to the engine."""
    if htf is None:
        htf = {tf: resample(data, tf) for tf in getattr(strategy, "htf_timeframes", [])}
    return _engine_run(config, strategy, data, htf=htf)
```

- [ ] **Step 4: Implement `src/strategies/ma_crossover.py`**

```python
from __future__ import annotations

import numpy as np

from ..strategy import Strategy


class MACrossover(Strategy):
    """Reference smoke-test strategy. SMA crossover with ATR-based SL/TP, risk-% sizing.
    Exists to validate the engine, not to make money."""

    name = "MA Crossover"
    params = {
        "fast": ("int", 20, 2, 200, "Fast SMA period"),
        "slow": ("int", 50, 3, 400, "Slow SMA period"),
        "risk_pct": ("float", 0.01, 0.001, 0.1, "Risk per trade (fraction of equity)"),
        "atr_period": ("int", 14, 2, 100, "ATR period"),
        "sl_atr_mult": ("float", 2.0, 0.1, 10.0, "Stop = entry ∓ mult*ATR"),
        "tp_atr_mult": ("float", 4.0, 0.1, 20.0, "Target = entry ± mult*ATR"),
    }

    def on_bar(self, ctx):
        p = self.p
        hist = ctx.history
        if len(hist) < max(p["slow"], p["atr_period"]) + 1:
            return

        close = hist["close"]
        fast_now = close.iloc[-p["fast"]:].mean()
        slow_now = close.iloc[-p["slow"]:].mean()
        fast_prev = close.iloc[-p["fast"] - 1:-1].mean()
        slow_prev = close.iloc[-p["slow"] - 1:-1].mean()

        atr = self._atr(hist, p["atr_period"])
        if atr <= 0:
            return
        price = ctx.bar["close"]

        crossed_up = fast_prev <= slow_prev and fast_now > slow_now
        crossed_down = fast_prev >= slow_prev and fast_now < slow_now

        if ctx.position is not None:
            # exit on opposite cross (discretionary); brackets handle SL/TP otherwise
            if (ctx.position.direction == "long" and crossed_down) or \
               (ctx.position.direction == "short" and crossed_up):
                ctx.close(reason="opposite_cross")
            return

        if crossed_up:
            stop = price - p["sl_atr_mult"] * atr
            tp = price + p["tp_atr_mult"] * atr
            size = ctx.size_for_risk(p["risk_pct"], price, stop)
            ctx.enter("long", size, stop_loss=stop, take_profit=tp, tag="ma_long")
        elif crossed_down:
            stop = price + p["sl_atr_mult"] * atr
            tp = price - p["tp_atr_mult"] * atr
            size = ctx.size_for_risk(p["risk_pct"], price, stop)
            ctx.enter("short", size, stop_loss=stop, take_profit=tp, tag="ma_short")

    @staticmethod
    def _atr(hist, period):
        h = hist["high"].to_numpy()[-period - 1:]
        l = hist["low"].to_numpy()[-period - 1:]
        c = hist["close"].to_numpy()[-period - 1:]
        if len(c) < 2:
            return 0.0
        prev_c = c[:-1]
        tr = np.maximum(h[1:] - l[1:],
                        np.maximum(np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)))
        return float(tr.mean()) if len(tr) else 0.0
```

- [ ] **Step 5: Implement auto-discovery `src/strategies/__init__.py`**

```python
from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path

from ..strategy import Strategy

_PACKAGE = __name__


def get_strategy_registry() -> dict[str, type]:
    """Discover all Strategy subclasses in this package. name -> class.
    Adding a strategy file here requires zero changes elsewhere."""
    registry: dict[str, type] = {}
    pkg_dir = Path(__file__).parent
    for mod in pkgutil.iter_modules([str(pkg_dir)]):
        if mod.name.startswith("_"):
            continue
        module = importlib.import_module(f"{_PACKAGE}.{mod.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Strategy) and obj is not Strategy and obj.__module__ == module.__name__:
                registry[obj.name] = obj
    return registry
```

- [ ] **Step 6: Implement `run_cli.py`**

```python
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd

from src import runner
from src.data_fetcher import fetch_ohlc
from src.data_loader import load_ohlc
from src.engine import BacktestConfig
from src.metrics import compute_metrics
from src.strategies import get_strategy_registry


def _synthetic(n=500):
    idx = pd.date_range("2024-01-02", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(42)
    steps = rng.normal(0, 1.5, n).cumsum()
    base = 2000 + steps
    df = pd.DataFrame({
        "open": base, "high": base + np.abs(rng.normal(0, 1, n)),
        "low": base - np.abs(rng.normal(0, 1, n)), "close": base + rng.normal(0, 0.5, n),
        "volume": 1.0,
    }, index=idx)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    df.index.name = "timestamp"
    return df


def main():
    ap = argparse.ArgumentParser(description="Headless gold backtest")
    ap.add_argument("--synthetic", action="store_true", help="use generated random-walk data")
    ap.add_argument("--csv", help="path to OHLC CSV in data/raw/")
    ap.add_argument("--fetch", action="store_true", help="fetch XAUUSD from Dukascopy")
    ap.add_argument("--timeframe", default="m5")
    ap.add_argument("--start", default="2024-01-02")
    ap.add_argument("--end", default="2024-02-01")
    ap.add_argument("--strategy", default="MA Crossover")
    ap.add_argument("--balance", type=float, default=10_000)
    args = ap.parse_args()

    if args.synthetic:
        data = load_ohlc(_synthetic())
    elif args.fetch:
        start = dt.datetime.fromisoformat(args.start)
        end = dt.datetime.fromisoformat(args.end)
        data = load_ohlc(fetch_ohlc("XAUUSD", args.timeframe, start, end))
    elif args.csv:
        data = load_ohlc(args.csv)
    else:
        raise SystemExit("pick one of --synthetic / --fetch / --csv")

    reg = get_strategy_registry()
    if args.strategy not in reg:
        raise SystemExit(f"unknown strategy {args.strategy!r}; available: {list(reg)}")
    strat = reg[args.strategy]()

    res = runner.run_backtest(BacktestConfig(opening_balance=args.balance), strat, data)
    m, summary = compute_metrics(res)
    print(f"\nStrategy: {args.strategy} | bars: {len(data)} | trades: {m['num_trades']}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run green + CLI smoke**

Run: `.venv/bin/pytest tests/test_runner.py -v`
Expected: PASS (2 tests).
Run: `.venv/bin/python run_cli.py --synthetic`
Expected: prints a metrics table (num_trades may vary).

- [ ] **Step 8: Commit**

```bash
git add src/runner.py src/strategies/ run_cli.py tests/test_runner.py
git commit -m "feat: runner, MA crossover strategy, auto-discovery, CLI"
```

---

### Task 7: Streamlit UI

**Goal:** Thin UI that collects data source + date range + strategy params + costs, calls `runner.run_backtest`, and renders metrics cards, equity/drawdown chart, R histogram, and a downloadable trade log.

**Files:**
- Create: `app.py`

**Acceptance Criteria:**
- [ ] App imports without error and the script compiles (`python -c "import ast; ast.parse(open('app.py').read())"`).
- [ ] Sidebar: data source (Fetch Dukascopy / local CSV / synthetic), date range, strategy dropdown from registry, dynamic param widgets from the schema, account+cost inputs, Run button.
- [ ] Main: metrics cards, Plotly equity+drawdown, R histogram, trade-log table with CSV download.

**Verify:** `.venv/bin/python -c "import ast; ast.parse(open('app.py').read()); print('ok')"` → `ok` (full UI verified manually via `.venv/bin/streamlit run app.py`)

**Steps:**

- [ ] **Step 1: Implement `app.py`**

```python
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import runner
from src.data_fetcher import fetch_ohlc
from src.data_loader import load_ohlc, resample
from src.engine import BacktestConfig
from src.metrics import compute_metrics
from src.strategies import get_strategy_registry

st.set_page_config(page_title="Gold Backtester", layout="wide")
st.title("🥇 Gold Backtesting Platform")

registry = get_strategy_registry()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Fetch from Dukascopy", "Local CSV", "Synthetic"])
    timeframe = st.selectbox("Base timeframe", ["m1", "m5", "m15", "m30", "h1", "h4"], index=1)
    start = st.date_input("Start", dt.date(2024, 1, 2))
    end = st.date_input("End", dt.date(2024, 2, 1))
    csv_path = None
    if source == "Local CSV":
        files = [p.name for p in Path("data/raw").glob("*.csv")]
        csv_path = st.selectbox("CSV file", files) if files else None
        if not files:
            st.warning("No CSVs in data/raw/")
    resample_to = st.selectbox("Resample to (optional)",
                               ["(none)", "m5", "m15", "m30", "h1", "h4"], index=0)

    st.header("Strategy")
    strat_name = st.selectbox("Strategy", list(registry.keys()))
    strat_cls = registry[strat_name]
    param_values = {}
    for pname, spec in strat_cls.params.items():
        ptype, default, pmin, pmax, help_ = spec
        if ptype == "int":
            param_values[pname] = st.number_input(pname, int(pmin), int(pmax), int(default),
                                                   step=1, help=help_)
        elif ptype == "float":
            param_values[pname] = st.number_input(pname, float(pmin), float(pmax),
                                                   float(default), help=help_)
        else:
            param_values[pname] = st.text_input(pname, str(default), help=help_)

    st.header("Account & costs")
    balance = st.number_input("Opening balance", 100.0, 1e9, 10_000.0, step=1000.0)
    spread = st.number_input("Spread (price units)", 0.0, 10.0, 0.30)
    slippage = st.number_input("Slippage (price units)", 0.0, 10.0, 0.0)
    commission_trade = st.number_input("Commission per trade ($)", 0.0, 1000.0, 0.0)
    commission_unit = st.number_input("Commission per unit ($/oz)", 0.0, 100.0, 0.0)
    intrabar = st.selectbox("Intrabar SL/TP rule", ["stop_first", "tp_first", "optimistic"])
    run = st.button("▶ Run backtest", type="primary")

# ---------------- Run ----------------
if run:
    with st.spinner("Loading data..."):
        if source == "Fetch from Dukascopy":
            raw = fetch_ohlc("XAUUSD", timeframe,
                             dt.datetime.combine(start, dt.time()),
                             dt.datetime.combine(end, dt.time()))
            data = load_ohlc(raw)
        elif source == "Local CSV" and csv_path:
            data = load_ohlc(str(Path("data/raw") / csv_path),
                             start=str(start), end=str(end))
        elif source == "Synthetic":
            import numpy as np
            idx = pd.date_range(start, end, freq="5min", tz="UTC")
            rng = np.random.default_rng(7)
            base = 2000 + rng.normal(0, 1.5, len(idx)).cumsum()
            data = pd.DataFrame({"open": base, "high": base + 1, "low": base - 1,
                                 "close": base + rng.normal(0, .5, len(idx)), "volume": 1.0},
                                index=idx)
            data["high"] = data[["open", "high", "close"]].max(axis=1)
            data["low"] = data[["open", "low", "close"]].min(axis=1)
            data.index.name = "timestamp"
            data = load_ohlc(data)
        else:
            st.error("No data source selected / CSV missing.")
            st.stop()

        if resample_to != "(none)":
            data = resample(data, resample_to)

    cfg = BacktestConfig(opening_balance=balance, spread=spread, slippage=slippage,
                         commission_per_trade=commission_trade,
                         commission_per_unit=commission_unit, intrabar=intrabar)
    strat = strat_cls(**param_values)

    t0 = dt.datetime.now()
    res = runner.run_backtest(cfg, strat, data)
    elapsed = (dt.datetime.now() - t0).total_seconds()
    m, summary = compute_metrics(res)

    st.success(f"Done in {elapsed:.2f}s — {len(data)} bars, {m['num_trades']} trades"
               + ("  ⚠️ run stopped: equity hit 0" if res.stopped_out else ""))

    c = st.columns(6)
    c[0].metric("Total return", f"{m['total_return_pct']*100:.1f}%")
    c[1].metric("Profit factor", f"{m['profit_factor']:.2f}")
    c[2].metric("Win rate", f"{m['win_rate']*100:.1f}%")
    c[3].metric("Max drawdown", f"{m['max_drawdown_pct']*100:.1f}%")
    c[4].metric("Expectancy (R)", f"{m['expectancy_r']:.2f}")
    c[5].metric("# Trades", m["num_trades"])

    # equity + drawdown
    ec = res.equity_curve
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.05, subplot_titles=("Equity / Balance", "Drawdown"))
    fig.add_trace(go.Scatter(x=ec.index, y=ec["equity"], name="Equity"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ec.index, y=ec["balance"], name="Balance",
                             line=dict(dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ec.index, y=ec["drawdown_pct"] * 100, name="Drawdown %",
                             fill="tozeroy"), row=2, col=1)
    fig.update_layout(height=520, margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)

    # R histogram + trade log
    left, right = st.columns([1, 2])
    if not res.trades.empty and res.trades["r_multiple"].notna().any():
        rhist = go.Figure(go.Histogram(x=res.trades["r_multiple"].dropna(), nbinsx=30))
        rhist.update_layout(title="R-multiple distribution", height=360)
        left.plotly_chart(rhist, use_container_width=True)
    else:
        left.info("No R data (no trades with stops).")

    right.subheader("Trade log")
    right.dataframe(res.trades, use_container_width=True, height=360)
    if not res.trades.empty:
        right.download_button("⬇ Download trades CSV",
                              res.trades.to_csv(index=False).encode(),
                              file_name="trades.csv", mime="text/csv")

    with st.expander("Full metrics + config"):
        st.dataframe(summary, use_container_width=True)
        st.json(cfg.__dict__)
        st.download_button("⬇ Download equity curve CSV",
                           ec.to_csv().encode(), file_name="equity_curve.csv", mime="text/csv")
else:
    st.info("Configure the run in the sidebar and click **Run backtest**.")
```

- [ ] **Step 2: Verify it parses**

Run: `.venv/bin/python -c "import ast; ast.parse(open('app.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Manual smoke (optional but recommended)**

Run: `.venv/bin/streamlit run app.py` → in the browser pick **Synthetic**, **MA Crossover**, click Run, confirm cards + charts render. Ctrl-C to stop.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: Streamlit UI for backtests"
```

---

### Task 8: SMC template strategy + README

**Goal:** Add the documented, non-functional `smc_template.py` (the file to copy for real strategies) and a README covering data format, Dukascopy fetch, adding strategies, engine assumptions, and how to run tests/CLI/UI.

**Files:**
- Create: `src/strategies/smc_template.py`
- Create: `README.md`

**Acceptance Criteria:**
- [ ] `smc_template.py` shows HTF bias via `ctx.htf`, session filtering off `ctx.bar["time"]`, stubbed swing/BOS helpers, structural-stop entry with R-based TP and risk sizing — all with `# TODO` markers, and does NOT trade as-is.
- [ ] It appears in `get_strategy_registry()` without breaking other strategies.
- [ ] Full suite passes: `.venv/bin/pytest`.
- [ ] README documents the data contract, fetch, add-a-strategy convention (params schema), every engine assumption from the spec §7, and run commands.

**Verify:** `.venv/bin/pytest -q` → all pass; `.venv/bin/python -c "from src.strategies import get_strategy_registry as g; print(list(g()))"` lists both strategies

**Steps:**

- [ ] **Step 1: Implement `src/strategies/smc_template.py`**

```python
from __future__ import annotations

import datetime as dt

from ..strategy import Strategy

# London/NY killzones (UTC). Adjust to taste.
LONDON = (dt.time(7, 0), dt.time(10, 0))
NEW_YORK = (dt.time(12, 0), dt.time(15, 0))


def _in_session(t, window) -> bool:
    return window[0] <= t.timetz().replace(tzinfo=None) <= window[1]


class SMCTemplate(Strategy):
    """COPY THIS FILE to start a new SMC/ICT strategy. Non-functional as-is.

    Demonstrates the patterns you'll use:
      - HTF bias via ctx.htf["h1"] / ctx.htf["h4"] while trading the base timeframe
      - session filtering off ctx.bar["time"]
      - structural stop (below/above the swing) + R-based TP, risk-% sizing
    Fill in the TODOs and remove the `return` guard in on_bar to activate.
    """

    name = "SMC Template (inactive)"
    htf_timeframes = ["h1", "h4"]   # runner builds these resampled views
    params = {
        "risk_pct": ("float", 0.01, 0.001, 0.05, "Risk per trade"),
        "rr": ("float", 2.0, 0.5, 10.0, "Reward:risk for TP"),
        "swing_lookback": ("int", 20, 3, 200, "Bars to scan for swing high/low"),
    }

    def on_bar(self, ctx):
        return  # TODO: remove once implemented — template must not trade by default

        # --- 1) Higher-timeframe bias -------------------------------------------------
        h1 = ctx.htf.get("h1")
        # TODO: derive bias, e.g. bullish if last closed H1 close > H1 SMA.
        bias = self._htf_bias(h1)  # "long" | "short" | None

        # --- 2) Session filter --------------------------------------------------------
        now = ctx.bar["time"]
        if not (_in_session(now, LONDON) or _in_session(now, NEW_YORK)):
            return

        # --- 3) Structure on the base timeframe --------------------------------------
        swing_hi, swing_lo = self._recent_swings(ctx.history, self.p["swing_lookback"])
        # TODO: detect BOS/CHoCH and your entry trigger here.
        setup_long = bias == "long"   # placeholder condition
        setup_short = bias == "short"

        price = ctx.bar["close"]
        if ctx.position is None and setup_long and swing_lo is not None:
            stop = swing_lo                       # structural stop below the swing
            tp = price + self.p["rr"] * (price - stop)
            size = ctx.size_for_risk(self.p["risk_pct"], price, stop)
            ctx.enter("long", size, stop_loss=stop, take_profit=tp, tag="smc_long")
        elif ctx.position is None and setup_short and swing_hi is not None:
            stop = swing_hi
            tp = price - self.p["rr"] * (stop - price)
            size = ctx.size_for_risk(self.p["risk_pct"], price, stop)
            ctx.enter("short", size, stop_loss=stop, take_profit=tp, tag="smc_short")

    # ----- stubbed helpers: implement for your edge -----
    @staticmethod
    def _htf_bias(htf_df):
        # TODO: return "long"/"short"/None from higher-timeframe structure.
        return None

    @staticmethod
    def _recent_swings(history, lookback):
        # TODO: real swing detection. Placeholder: window extremes.
        window = history.iloc[-lookback:]
        if len(window) < lookback:
            return None, None
        return float(window["high"].max()), float(window["low"].min())
```

- [ ] **Step 2: Verify registry + full suite**

Run: `.venv/bin/python -c "from src.strategies import get_strategy_registry as g; print(list(g()))"`
Expected: lists both `MA Crossover` and `SMC Template (inactive)`.
Run: `.venv/bin/pytest -q`
Expected: all tests pass.

- [ ] **Step 3: Write `README.md`**

````markdown
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
````

- [ ] **Step 4: Commit**

```bash
git add src/strategies/smc_template.py README.md
git commit -m "feat: SMC strategy template and README"
```

---

## Self-Review

**Spec coverage:** data_fetcher (Task 1) ✓, data_loader validate/resample (Task 2) ✓, strategy interface incl. htf/size_for_risk (Task 3) ✓, engine fills/costs/intrabar/gap/equity + all §11 correctness tests (Task 4) ✓, metrics incl. R/PF/drawdown/sharpe/exposure (Task 5) ✓, runner single seam + ma_crossover + auto-discovery + CLI (Task 6) ✓, Streamlit UI cards/equity/drawdown/R-hist/trade-log/downloads (Task 7) ✓, smc_template + README documenting all §7 assumptions (Task 8) ✓. Non-goals respected (no optimization/portfolio/pyramiding).

**Type consistency:** `BacktestConfig`, `BacktestResult`, `Trade`, `Position`, `Order`, `Context` field/method names are consistent across tasks; `run_backtest` signature `(config, strategy, data, htf=None)` matches between engine and runner; registry keyed on `Strategy.name`; equity_curve columns `equity,balance,peak,drawdown,drawdown_pct` consistent between engine and metrics/UI.

**Placeholders:** none in plan steps (the `# TODO`s in `smc_template.py` are intentional product content — that file is a template). No "TBD"/"implement later" in any step.
