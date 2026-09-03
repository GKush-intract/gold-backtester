from __future__ import annotations

import numpy as np
import pandas as pd

from ...strategy import Strategy


class EMA33BreakoutTrailingStop(Strategy):
    """
    EMA33 Breakout Trailing Stop - XAUUSD M5

    Long-only M5 strategy. Enter when a bar closes above EMA(ema_period).
    Higher-timeframe trend filter: only take longs when the HTF is bullish
    (HTF close > HTF EMA(htf_ema_period)).

    For M5 base TF  → HTF is H4  (default, htf_timeframe = "h4")
    For M1 base TF  → HTF is H1  (set htf_timeframe = "h1")

    Initial SL anchor per `sl_mode` (EMA value, signal-candle low/open, or the
    lowest of them), never closer than min_stop_atr_mult * ATR to the entry.
    1R = entry_price - initial SL.

    At 1R profit (bar high >= entry + R): partial close `partial_fraction` of
    the position via ctx.close(fraction=...) — a TRUE partial close, so the
    remainder keeps running in the same position (no re-entry, no second
    spread). The remainder ("runner") is then managed as:
      - original engine SL stays live at the initial level (exact, intrabar)
      - breakeven exit if a bar CLOSES at/below the entry price
      - target exit when a bar's high reaches entry + runner_target_r * R

    Engine note: SL/TP cannot be modified after entry, so the runner's
    breakeven and target are strategy-managed checks evaluated per bar and
    filled at the NEXT bar open (approximate), while the initial SL remains
    the exact intrabar backstop.
    """

    name = "EMA33 Breakout Trailing Stop - XAUUSD M5"

    params = {
        "ema_period": ("int", 33, 5, 200,
                       "Period for the Exponential Moving Average used as entry filter and initial stop-loss anchor."),
        "risk_pct": ("float", 1.0, 0.1, 5.0,
                     "Percentage of account equity to risk per trade (e.g. 1.0 = 1%)."),
        "atr_period": ("int", 14, 2, 100,
                       "ATR period used for the minimum stop distance."),
        "min_stop_atr_mult": ("float", 1.0, 0.1, 5.0,
                              "Minimum initial stop distance in ATR multiples. Prevents microscopic "
                              "stops (and absurd R multiples) when price barely crosses the EMA."),
        "sl_mode": ("str", "lowest", "", "",
                    "Initial SL anchor: 'ema' (EMA value), 'candle_low' (low of the signal "
                    "candle), 'candle_open' (open of the signal candle), or 'lowest' "
                    "(lowest of the three). The ATR minimum-distance floor always applies."),
        "partial_fraction": ("float", 0.5, 0.1, 0.9,
                             "Fraction of the position closed when price reaches 1R."),
        "runner_target_r": ("float", 2.0, 0.5, 10.0,
                            "Runner take-profit at entry + this multiple of R (checked per bar, "
                            "filled next open)."),
        # ── Higher-timeframe trend filter ──────────────────────────────────────
        "htf_ema_period": ("int", 21, 3, 200,
                           "EMA period applied to the higher-timeframe bars for trend detection. "
                           "HTF bar close must be above this EMA for a long entry to be allowed."),
        "htf_timeframe": ("str", "h4", "", "",
                          "Which higher timeframe to use for the trend filter. "
                          "Typical choices: 'h1' (for M1 base TF) or 'h4' (for M5 base TF). "
                          "Must match one of the names declared in htf_timeframes."),
    }

    # Declare both common HTF resolutions; the runner will only use whichever
    # htf_timeframe the user picks.  Having both pre-built costs nothing extra
    # in memory and avoids having to dynamically change the class attribute.
    htf_timeframes = ["h1", "h4"]

    # ------------------------------------------------------------------
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

    @staticmethod
    def _ema_last(series: pd.Series, period: int) -> float:
        """Return the last value of an EMA computed on `series` (bounded tail)."""
        tail = period * 6 + 2
        s = series.iloc[-tail:] if len(series) > tail else series
        if len(s) < period:
            return float("nan")
        return float(s.ewm(span=period, adjust=False).mean().iloc[-1])

    # ------------------------------------------------------------------
    def on_start(self, ctx):
        self._reset()

    def _reset(self):
        # phase 0 = flat, 1 = initial leg (waiting for 1R), 2 = runner
        self._phase: int = 0
        self._entry_price: float = 0.0
        self._initial_sl: float = 0.0
        self._r: float = 0.0
        self._one_r_target: float = 0.0

    # ------------------------------------------------------------------
    def _htf_is_bullish(self, ctx) -> bool:
        """Return True when the chosen HTF EMA filter says the trend is up."""
        htf_name = str(self.p["htf_timeframe"]).strip().lower()
        htf_ema_period = self.p["htf_ema_period"]

        htf_data: pd.DataFrame | None = ctx.htf.get(htf_name)
        if htf_data is None or htf_data.empty:
            # If the HTF data is unavailable, be conservative and block entry.
            return False

        # Only use HTF rows whose timestamp is <= current bar time (no look-ahead).
        current_time = ctx.bar["time"]
        htf_data = htf_data[htf_data.index <= current_time]
        if len(htf_data) < htf_ema_period:
            return False

        htf_close = htf_data["close"]
        htf_ema_val = self._ema_last(htf_close, htf_ema_period)
        if np.isnan(htf_ema_val):
            return False

        last_close = float(htf_close.iloc[-1])
        return last_close > htf_ema_val

    # ------------------------------------------------------------------
    def on_bar(self, ctx):
        p = self.p
        ema_period = p["ema_period"]
        risk_pct = p["risk_pct"] / 100.0

        # Bounded tail for O(lookback) behaviour
        tail = max(ema_period, p["atr_period"]) * 6 + 2
        hist = ctx.history
        if len(hist) > tail:
            hist = hist.iloc[-tail:]

        if len(hist) < ema_period:
            return

        ema_now = float(hist["close"].ewm(span=ema_period, adjust=False).mean().iloc[-1])

        bar = ctx.bar
        close_now = bar["close"]
        high_now = bar["high"]

        # ── Manage open position ──────────────────────────────────────────────
        if ctx.position is not None:
            pos = ctx.position

            # ----- Phase 1: initial leg (waiting to reach 1R) ----------------
            if self._phase == 1:
                # Initialise R on the first bar we see the position live
                if self._r == 0.0 and pos.entry_price > 0:
                    self._entry_price = pos.entry_price
                    if self._initial_sl <= 0.0 and pos.stop_loss is not None:
                        self._initial_sl = pos.stop_loss
                    r = self._entry_price - self._initial_sl
                    if r <= 0:
                        ctx.close(reason="invalid_r")
                        self._reset()
                        return
                    self._r = r
                    self._one_r_target = self._entry_price + r

                if self._r <= 0:
                    return

                # 1R reached: scale out, keep the rest running in the SAME position
                if high_now >= self._one_r_target:
                    ctx.close(reason="1r_partial", fraction=p["partial_fraction"])
                    self._phase = 2
                return

            # ----- Phase 2: runner (engine SL live; BE + target on bar close) -
            if self._phase == 2:
                if high_now >= self._entry_price + p["runner_target_r"] * self._r:
                    ctx.close(reason="runner_target")
                    self._reset()
                    return
                if close_now <= self._entry_price:
                    ctx.close(reason="breakeven_exit")
                    self._reset()
                    return
                return

        # ── Flat ──────────────────────────────────────────────────────────────
        if self._phase != 0:
            self._reset()

        # ── Entry logic ───────────────────────────────────────────────────────
        # Signal: current bar closes above EMA → enter long, fills next bar open.
        if close_now > ema_now:
            # ── Higher-timeframe trend filter ─────────────────────────────────
            # Only enter if the HTF is also bullish (close > HTF EMA).
            if not self._htf_is_bullish(ctx):
                return

            entry_approx = close_now
            # SL anchor per sl_mode, but never closer than min_stop_atr_mult * ATR.
            atr = self._atr(hist, p["atr_period"])
            if atr <= 0:
                return
            min_stop = p["min_stop_atr_mult"] * atr
            anchors = {
                "ema": ema_now,
                "candle_low": bar["low"],
                "candle_open": bar["open"],
                "lowest": min(ema_now, bar["low"], bar["open"]),
            }
            anchor = anchors.get(str(p["sl_mode"]).strip().lower(), anchors["lowest"])
            initial_sl = min(anchor, entry_approx - min_stop)

            if entry_approx <= initial_sl:
                return

            size = ctx.size_for_risk(risk_pct, entry_approx, initial_sl)
            if size <= 0:
                return

            self._reset()
            self._initial_sl = initial_sl
            self._phase = 1

            ctx.enter(
                "long",
                size,
                stop_loss=initial_sl,
                take_profit=None,
                tag="ema_breakout_long",
            )

    def on_finish(self, ctx):
        pass
