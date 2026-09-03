from __future__ import annotations

import datetime
import numpy as np
import pandas as pd

from ...strategy import Strategy


class SachinVWAPConfluence(Strategy):
    """1-minute XAUUSD strategy based on VWAP crossovers confirmed by CCI, MACD, and EMA filters.

    Entry: price crosses VWAP with CCI, MACD histogram, and EMA trend all aligned.
    Exit 1 (TP1): partial close of 50% at 2R.
    Exit 2 (TP2): close remaining 50% at tp2_r × R (default 4R).
    Exit 3 (max-hold): unconditional full close after max_hold_bars bars.
    Stop-loss: anchored to recent swing low/high, capped at max_stop_pts.
    """

    name = "Sachin VWAP Confluence"

    params = {
        "min_clear": ("float", 0.5, 0.0, 10.0,
                      "Minimum distance (pts) the close must be beyond VWAP for a valid cross signal"),
        "cool_bars": ("int", 15, 0, 200,
                      "Cooldown in bars after any signal (long or short) before a new signal is allowed"),
        "ema_period": ("int", 33, 5, 200,
                       "Period for the EMA trend filter"),
        "cci_period": ("int", 20, 5, 100,
                       "Period for the CCI momentum filter (applied to hlc3)"),
        "macd_fast": ("int", 12, 2, 50,
                      "MACD fast EMA period"),
        "macd_slow": ("int", 26, 5, 100,
                      "MACD slow EMA period"),
        "macd_signal": ("int", 9, 2, 50,
                        "MACD signal EMA period"),
        "sl_lookback": ("int", 3, 1, 20,
                        "Number of bars to look back for the swing low/high to anchor the stop-loss"),
        "max_stop_pts": ("float", 5.0, 1.0, 50.0,
                         "Maximum allowed stop-loss distance in points"),
        "risk_pct": ("float", 1.0, 0.1, 5.0,
                     "Percentage of equity risked per trade for position sizing"),
        "max_leverage": ("float", 20.0, 1.0, 100.0,
                         "Maximum leverage cap applied to computed lot size"),
        "tp2_r": ("float", 4.0, 2.0, 20.0,
                  "R-multiple for the second (full) take-profit on the remaining 50% of the position after TP1"),
        "max_hold_bars": ("int", 480, 1, 2000,
                          "Maximum number of bars a position may be held; any remaining position is closed "
                          "unconditionally at this limit (~8 hours on M1)"),
    }

    htf_timeframes = []

    def __init__(self, **params):
        super().__init__(**params)
        # trade-state
        self._last_signal_bar: int = -9999
        self._tp1_hit: bool = False
        self._sl_distance: float = 0.0
        self._entry_price: float = 0.0
        self._entry_bar: int = 0

        # incremental EMA caches
        self._ema_cache: float | None = None
        self._macd_fast_cache: float | None = None
        self._macd_slow_cache: float | None = None
        self._macd_signal_cache: float | None = None

        # incremental VWAP caches
        self._vwap_sum_pv: float = 0.0
        self._vwap_sum_v: float = 0.0
        self._vwap_session_date: datetime.date | None = None
        self._vwap_prev_val: float | None = None

        # previous close for crossover detection
        self._prev_close: float | None = None

        # bar counter for warm-up tracking
        self._bar_count: int = 0

    def on_start(self, ctx):
        self._last_signal_bar = -9999
        self._tp1_hit = False
        self._sl_distance = 0.0
        self._entry_price = 0.0
        self._entry_bar = 0

        self._ema_cache = None
        self._macd_fast_cache = None
        self._macd_slow_cache = None
        self._macd_signal_cache = None

        self._vwap_sum_pv = 0.0
        self._vwap_sum_v = 0.0
        self._vwap_session_date = None
        self._vwap_prev_val = None

        self._prev_close = None
        self._bar_count = 0

    def on_bar(self, ctx):
        p = self.p
        bar = ctx.bar
        idx = ctx.index

        close_now = float(bar["close"])
        high_now  = float(bar["high"])
        low_now   = float(bar["low"])
        vol_now   = float(bar["volume"])
        hlc3_now  = (high_now + low_now + close_now) / 3.0

        bar_time  = bar["time"]
        bar_date  = pd.Timestamp(bar_time).date()

        self._bar_count += 1

        # ── Update VWAP state ────────────────────────────────────────────────
        # Save previous VWAP before updating
        if self._vwap_sum_v > 0:
            self._vwap_prev_val = self._vwap_sum_pv / self._vwap_sum_v
        else:
            self._vwap_prev_val = None

        if self._vwap_session_date is None or bar_date != self._vwap_session_date:
            # New session — reset accumulators
            self._vwap_sum_pv = 0.0
            self._vwap_sum_v = 0.0
            self._vwap_session_date = bar_date
            self._vwap_prev_val = None  # no previous VWAP in new session

        self._vwap_sum_pv += hlc3_now * vol_now
        self._vwap_sum_v  += vol_now

        if self._vwap_sum_v <= 0:
            self._prev_close = close_now
            return
        vwap_val = self._vwap_sum_pv / self._vwap_sum_v

        # ── Update incremental EMA caches ────────────────────────────────────
        ema_k       = 2.0 / (p["ema_period"] + 1)
        macd_fast_k = 2.0 / (p["macd_fast"] + 1)
        macd_slow_k = 2.0 / (p["macd_slow"] + 1)

        if self._ema_cache is None:
            self._ema_cache = close_now
        else:
            self._ema_cache = close_now * ema_k + self._ema_cache * (1.0 - ema_k)

        if self._macd_fast_cache is None:
            self._macd_fast_cache = close_now
        else:
            self._macd_fast_cache = close_now * macd_fast_k + self._macd_fast_cache * (1.0 - macd_fast_k)

        if self._macd_slow_cache is None:
            self._macd_slow_cache = close_now
        else:
            self._macd_slow_cache = close_now * macd_slow_k + self._macd_slow_cache * (1.0 - macd_slow_k)

        macd_line_now = self._macd_fast_cache - self._macd_slow_cache

        macd_sig_k = 2.0 / (p["macd_signal"] + 1)
        if self._macd_signal_cache is None:
            self._macd_signal_cache = macd_line_now
        else:
            self._macd_signal_cache = macd_line_now * macd_sig_k + self._macd_signal_cache * (1.0 - macd_sig_k)

        macd_hist = macd_line_now - self._macd_signal_cache
        ema_val   = self._ema_cache

        # ── CCI (needs last cci_period bars of hlc3) — use bounded slice ──────
        cci_period = p["cci_period"]
        sl_lookback = p["sl_lookback"]

        # Bounded tail slice — only pull what we need, never full history
        tail_needed = max(cci_period, sl_lookback + 2)
        full_hist = ctx.history
        if len(full_hist) < cci_period:
            self._prev_close = close_now
            return

        # Slice a bounded tail to keep this O(lookback) not O(n)
        hist = full_hist.iloc[-tail_needed:] if len(full_hist) > tail_needed else full_hist

        cci_slice = hist["close"].to_numpy(dtype=float)[-cci_period:]
        cci_h     = hist["high"].to_numpy(dtype=float)[-cci_period:]
        cci_l     = hist["low"].to_numpy(dtype=float)[-cci_period:]
        hlc3_arr  = (cci_h + cci_l + cci_slice) / 3.0
        cci_val   = self._cci_last(hlc3_arr, cci_period)
        if cci_val is None:
            self._prev_close = close_now
            return

        # ── Need at least max_period bars before trusting indicators ──────────
        max_period = max(p["ema_period"], p["cci_period"],
                         p["macd_slow"] + p["macd_signal"],
                         p["sl_lookback"] + 1)
        if self._bar_count < max_period + 2:
            self._prev_close = close_now
            return

        # ── Crossover detection ───────────────────────────────────────────────
        price_now  = close_now
        price_prev = self._prev_close
        vwap_prev  = self._vwap_prev_val

        # Update prev close for next bar
        self._prev_close = close_now

        if price_prev is None or vwap_prev is None or vwap_prev <= 0:
            return

        crossed_up   = (price_prev <= vwap_prev) and (price_now > vwap_prev)
        crossed_down = (price_prev >= vwap_prev) and (price_now < vwap_prev)

        # ── Manage open position ──────────────────────────────────────────────
        if ctx.position is not None:
            pos = ctx.position

            # Max-hold-bars check takes priority over TP1/TP2 logic
            bars_held = idx - self._entry_bar
            if bars_held >= p["max_hold_bars"]:
                ctx.close(reason="max_hold", fraction=1.0)
                return

            if not self._tp1_hit:
                # Check TP1 at 2R
                if pos.direction == "long":
                    tp1_price = self._entry_price + 2.0 * self._sl_distance
                    if bar["high"] >= tp1_price:
                        ctx.close(reason="tp1_partial", fraction=0.5)
                        self._tp1_hit = True
                else:
                    tp1_price = self._entry_price - 2.0 * self._sl_distance
                    if bar["low"] <= tp1_price:
                        ctx.close(reason="tp1_partial", fraction=0.5)
                        self._tp1_hit = True
            else:
                # TP1 already hit — check TP2 at tp2_r × R on remaining 50%
                tp2_r = p["tp2_r"]
                if pos.direction == "long":
                    tp2_price = self._entry_price + tp2_r * self._sl_distance
                    if bar["high"] >= tp2_price:
                        ctx.close(reason="tp2_full", fraction=1.0)
                else:
                    tp2_price = self._entry_price - tp2_r * self._sl_distance
                    if bar["low"] <= tp2_price:
                        ctx.close(reason="tp2_full", fraction=1.0)

            return  # one position at a time; no new entries while open

        # ── Cooldown check ────────────────────────────────────────────────────
        bars_since = idx - self._last_signal_bar
        if bars_since <= p["cool_bars"]:
            return

        min_clear = p["min_clear"]

        # ── LONG entry ────────────────────────────────────────────────────────
        if (crossed_up
                and close_now > vwap_val + min_clear
                and cci_val > 0
                and macd_hist > 0
                and close_now > ema_val):

            swing_low     = float(hist["low"].to_numpy(dtype=float)[-(sl_lookback + 1):-1].min())
            sl_dist_swing = close_now - swing_low
            sl_dist = min(p["max_stop_pts"], max(sl_dist_swing, 0.01))
            sl_dist = max(sl_dist, 0.01)

            entry_approx = close_now
            stop_price   = entry_approx - sl_dist

            size = ctx.size_for_risk(
                p["risk_pct"] / 100.0, entry_approx, stop_price
            )
            if size <= 0:
                return

            ctx.enter(
                "long",
                size,
                stop_loss=stop_price,
                take_profit=None,
                tag="vwap_long",
            )
            self._last_signal_bar = idx
            self._tp1_hit = False
            self._sl_distance = sl_dist
            self._entry_price = entry_approx
            self._entry_bar = idx

        # ── SHORT entry ───────────────────────────────────────────────────────
        elif (crossed_down
              and close_now < vwap_val - min_clear
              and cci_val < 0
              and macd_hist < 0
              and close_now < ema_val):

            swing_high    = float(hist["high"].to_numpy(dtype=float)[-(sl_lookback + 1):-1].max())
            sl_dist_swing = swing_high - close_now
            sl_dist = min(p["max_stop_pts"], max(sl_dist_swing, 0.01))
            sl_dist = max(sl_dist, 0.01)

            entry_approx = close_now
            stop_price   = entry_approx + sl_dist

            size = ctx.size_for_risk(
                p["risk_pct"] / 100.0, entry_approx, stop_price
            )
            if size <= 0:
                return

            ctx.enter(
                "short",
                size,
                stop_loss=stop_price,
                take_profit=None,
                tag="vwap_short",
            )
            self._last_signal_bar = idx
            self._tp1_hit = False
            self._sl_distance = sl_dist
            self._entry_price = entry_approx
            self._entry_bar = idx

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cci_last(hlc3: np.ndarray, period: int) -> float | None:
        if len(hlc3) < period:
            return None
        window = hlc3[-period:]
        tp   = window[-1]
        mean = float(np.mean(window))
        md   = float(np.mean(np.abs(window - mean)))
        if md == 0:
            return 0.0
        return (tp - mean) / (0.015 * md)
