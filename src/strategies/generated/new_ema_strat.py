from __future__ import annotations

import numpy as np
import pandas as pd

from ...strategy import Strategy, OZ_PER_LOT, clamp_lots, round_lots


class EMA33CrossoverLadderGoldM1(Strategy):
    """EMA33 Crossover Ladder Gold m1 strategy.

    Long-only momentum entry on XAUUSD m1 bars with a doubling TP ladder.
    """

    name = "EMA33 Crossover Ladder Gold m1"

    params = {
        "ema_period": ("int", 33, 5, 200, "Period for the EMA used in the crossover condition."),
        "atr_period": ("int", 14, 5, 100, "Period for the ATR used in the candle-size filter."),
        "atr_filter_mult": ("float", 1.5, 0.5, 5.0, "Signal candle range must be less than this multiple of ATR to allow entry."),
        "lookback_candles": ("int", 3, 1, 10, "Number of candles before the signal candle that must have their entire range below the EMA."),
        "risk_pct": ("float", 1.0, 0.1, 10.0, "Percentage of account equity risked per trade."),
        "max_leverage": ("float", 20.0, 1.0, 500.0, "Maximum leverage cap on notional position size."),
        "ladder_start_r": ("float", 2.0, 0.5, 10.0, "First take-profit level expressed as a multiple of R (subsequent levels double each time)."),
        "ladder_close_fraction": ("float", 0.5, 0.1, 0.9, "Fraction of the remaining position closed at each ladder TP level."),
    }

    htf_timeframes = []

    def __init__(self, **params):
        super().__init__(**params)
        # Ladder state tracking (managed externally since engine doesn't natively support ladder)
        self._in_trade = False
        self._entry_price: float = 0.0
        self._stop_price: float = 0.0
        self._R: float = 0.0
        self._current_rung: int = 0          # 0 means we are between entry and first TP
        self._next_tp: float = 0.0
        self._current_sl: float = 0.0

    def on_start(self, ctx):
        self._in_trade = False
        self._entry_price = 0.0
        self._stop_price = 0.0
        self._R = 0.0
        self._current_rung = 0
        self._next_tp = 0.0
        self._current_sl = 0.0

    def _compute_ema(self, closes: np.ndarray, period: int) -> np.ndarray:
        """Compute EMA on the given closes array."""
        if len(closes) < period:
            return np.full(len(closes), np.nan)
        k = 2.0 / (period + 1)
        ema = np.empty(len(closes))
        ema[:] = np.nan
        # Seed with SMA of first `period` values
        ema[period - 1] = closes[:period].mean()
        for i in range(period, len(closes)):
            ema[i] = closes[i] * k + ema[i - 1] * (1 - k)
        return ema

    def _compute_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float:
        """Compute the ATR (Wilder's) for the last `period` bars using numpy."""
        n = len(closes)
        if n < period + 1:
            return 0.0
        h = highs[-(period + 1):]
        l = lows[-(period + 1):]
        c = closes[-(period + 1):]
        prev_c = c[:-1]
        tr = np.maximum(h[1:] - l[1:],
                        np.maximum(np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)))
        return float(tr.mean())

    def on_bar(self, ctx):
        p = self.p
        ema_period = p["ema_period"]
        atr_period = p["atr_period"]
        lookback = p["lookback_candles"]

        # Bound the tail for O(lookback) per bar
        tail = max(ema_period, atr_period) * 6 + lookback + 10
        hist = ctx.history
        if len(hist) > tail:
            hist = hist.iloc[-tail:]

        bar = ctx.bar
        current_price = bar["close"]
        current_high = bar["high"]
        current_low = bar["low"]

        closes = hist["close"].to_numpy(dtype=float)
        highs = hist["high"].to_numpy(dtype=float)
        lows = hist["low"].to_numpy(dtype=float)
        opens = hist["open"].to_numpy(dtype=float)

        n = len(closes)
        min_needed = max(ema_period, atr_period) + lookback + 1
        if n < min_needed:
            return

        # Compute EMA on the slice
        ema_arr = self._compute_ema(closes, ema_period)
        ema_now = ema_arr[-1]
        if np.isnan(ema_now):
            return

        # --- Manage existing position via ladder ---
        if ctx.position is not None and self._in_trade:
            # Check if position was closed externally (stop hit by engine)
            # We manage SL/TP manually through close() calls + updating position SL.
            # The engine handles the actual stop being hit via position.stop_loss.
            # We only need to handle TP hits (price reaches next_tp) here.

            # Update our current SL to match what we've set
            # Check for ladder TP hit
            if self._next_tp > 0 and current_high >= self._next_tp:
                # TP rung hit — close fraction and advance ladder
                frac = p["ladder_close_fraction"]
                ctx.close(reason=f"ladder_rung_{self._current_rung}", fraction=frac)

                # Advance ladder state
                self._current_rung += 1
                self._current_sl = self._next_tp
                # Next TP doubles in R distance from entry
                ladder_mult = p["ladder_start_r"] * (2 ** self._current_rung)
                self._next_tp = self._entry_price + ladder_mult * self._R

                # Note: We cannot directly update position.stop_loss from here.
                # We'll need to re-enter with a new SL — but the engine doesn't support
                # modifying SL in-place. We'll track the ladder SL ourselves and use
                # ctx.close() when price falls below it.
                return

            # Check if price has fallen below our current ladder SL
            if current_low <= self._current_sl and self._current_rung > 0:
                ctx.close(reason="ladder_sl")
                self._in_trade = False
                return

            return  # still in trade, nothing to do

        elif ctx.position is None:
            # Position was closed (by engine or our close() call)
            self._in_trade = False

        # --- Entry logic ---
        if ctx.position is not None:
            # Still in a position (in_trade flag was out of sync)
            return

        # Compute ATR
        atr = self._compute_atr(highs, lows, closes, atr_period)
        if atr <= 0:
            return

        # Signal candle = current bar (index -1 in slice)
        # Previous `lookback` candles are at indices -2, -3, ..., -(lookback+1)
        if n < lookback + 2:
            return

        signal_open = opens[-1]
        signal_close = closes[-1]
        signal_high = highs[-1]
        signal_low = lows[-1]
        signal_range = signal_high - signal_low
        ema_signal = ema_arr[-1]

        # Condition 2: signal candle open < EMA and close > EMA
        if not (signal_open < ema_signal and signal_close > ema_signal):
            return

        # Condition 3: prior `lookback` candles have entire range below EMA
        prior_emas = ema_arr[-(lookback + 1):-1]
        prior_highs = highs[-(lookback + 1):-1]
        prior_lows = lows[-(lookback + 1):-1]

        if len(prior_emas) < lookback:
            return

        all_below = True
        for i in range(lookback):
            if np.isnan(prior_emas[i]):
                all_below = False
                break
            if prior_highs[i] >= prior_emas[i] or prior_lows[i] >= prior_emas[i]:
                all_below = False
                break

        if not all_below:
            return

        # Condition 4: signal candle range < atr_filter_mult * ATR
        if signal_range >= p["atr_filter_mult"] * atr:
            return

        # Entry conditions met — place order
        # SL = signal candle open; entry fills at next bar open
        # R will be computed using actual fill (next bar open) — but we approximate
        # using current close for sizing; actual R is set when position opens.
        # Per spec: SL = signal candle open price
        sl_price = signal_open

        # Size for risk: use current close as proxy for entry price
        # The actual entry will be next bar open, but we size conservatively on close
        entry_proxy = signal_close
        risk_fraction = p["risk_pct"] / 100.0

        # Ensure stop is meaningful distance away
        if abs(entry_proxy - sl_price) <= 0:
            return

        size = ctx.size_for_risk(risk_fraction, entry_proxy, sl_price)
        if size <= 0:
            return

        # Compute initial R and first TP based on proxy
        # Actual R will be recalculated after fill using position.entry_price
        initial_r = abs(entry_proxy - sl_price)
        first_tp = entry_proxy + p["ladder_start_r"] * initial_r

        ctx.enter(
            "long",
            size,
            stop_loss=sl_price,
            take_profit=first_tp,
            tag="ema33_cross",
        )

        # Store state — we'll finalise entry/R on next bar when position exists
        self._pending_sl = sl_price
        self._pending_entry_proxy = entry_proxy
        self._in_trade = False  # will be set to True on next bar when position confirmed

    def on_bar(self, ctx):
        """Override to also handle the 'just entered' bar."""
        p = self.p
        ema_period = p["ema_period"]
        atr_period = p["atr_period"]
        lookback = p["lookback_candles"]

        # Bound the tail for O(lookback) per bar
        tail = max(ema_period, atr_period) * 6 + lookback + 10
        hist = ctx.history
        if len(hist) > tail:
            hist = hist.iloc[-tail:]

        closes = hist["close"].to_numpy(dtype=float)
        highs = hist["high"].to_numpy(dtype=float)
        lows = hist["low"].to_numpy(dtype=float)
        opens = hist["open"].to_numpy(dtype=float)

        n = len(closes)
        bar = ctx.bar

        # Sync in_trade with actual position state
        if ctx.position is None:
            self._in_trade = False

        # --- Manage existing position via ladder ---
        if ctx.position is not None:
            if not self._in_trade:
                # Position just opened — capture actual fill price and set R
                pos = ctx.position
                self._in_trade = True
                self._entry_price = pos.entry_price
                self._stop_price = pos.stop_loss if pos.stop_loss is not None else getattr(self, '_pending_sl', pos.entry_price)
                self._R = abs(self._entry_price - self._stop_price)
                self._current_rung = 0
                self._current_sl = self._stop_price
                if self._R > 0:
                    self._next_tp = self._entry_price + p["ladder_start_r"] * self._R
                else:
                    self._next_tp = 0.0
            else:
                # Check for ladder TP hit
                current_high = bar["high"]
                current_low = bar["low"]

                if self._next_tp > 0 and current_high >= self._next_tp:
                    frac = p["ladder_close_fraction"]
                    ctx.close(reason=f"ladder_rung_{self._current_rung}", fraction=frac)

                    # Advance ladder state
                    prev_rung = self._current_rung
                    self._current_rung += 1
                    self._current_sl = self._next_tp
                    # Next TP doubles the R distance from entry
                    ladder_mult = p["ladder_start_r"] * (2 ** self._current_rung)
                    self._next_tp = self._entry_price + ladder_mult * self._R
                    return

                # Check if price falls below our tracked ladder SL (for rungs > 0)
                if self._current_rung > 0 and current_low <= self._current_sl:
                    ctx.close(reason="ladder_sl")
                    self._in_trade = False
                    return

            return  # In a position; don't look for entries

        # --- Entry logic ---
        min_needed = max(ema_period, atr_period) + lookback + 2
        if n < min_needed:
            return

        # Compute EMA
        ema_arr = self._compute_ema(closes, ema_period)
        ema_now = ema_arr[-1]
        if np.isnan(ema_now):
            return

        # Compute ATR
        atr = self._compute_atr(highs, lows, closes, atr_period)
        if atr <= 0:
            return

        # Signal candle = current bar
        signal_open = opens[-1]
        signal_close = closes[-1]
        signal_high = highs[-1]
        signal_low = lows[-1]
        signal_range = signal_high - signal_low
        ema_signal = ema_arr[-1]

        # Condition 2: signal candle open < EMA and close > EMA
        if not (signal_open < ema_signal and signal_close > ema_signal):
            return

        # Condition 3: prior `lookback` candles have entire range below EMA
        if n < lookback + 2:
            return

        prior_emas = ema_arr[-(lookback + 1):-1]
        prior_highs = highs[-(lookback + 1):-1]
        prior_lows = lows[-(lookback + 1):-1]

        if len(prior_emas) < lookback:
            return

        all_below = True
        for i in range(lookback):
            if np.isnan(prior_emas[i]):
                all_below = False
                break
            if prior_highs[i] >= prior_emas[i] or prior_lows[i] >= prior_emas[i]:
                all_below = False
                break

        if not all_below:
            return

        # Condition 4: signal candle range < atr_filter_mult * ATR
        if signal_range >= p["atr_filter_mult"] * atr:
            return

        # SL = signal candle open price; entry fills next bar open
        sl_price = signal_open
        entry_proxy = signal_close  # conservative proxy for sizing

        stop_dist = abs(entry_proxy - sl_price)
        if stop_dist <= 0:
            return

        risk_fraction = p["risk_pct"] / 100.0
        size = ctx.size_for_risk(risk_fraction, entry_proxy, sl_price)
        if size <= 0:
            return

        # First TP
        initial_r = stop_dist
        first_tp = entry_proxy + p["ladder_start_r"] * initial_r

        ctx.enter(
            "long",
            size,
            stop_loss=sl_price,
            take_profit=first_tp,
            tag="ema33_cross",
        )

    def _compute_ema(self, closes: np.ndarray, period: int) -> np.ndarray:
        """Compute EMA on the given closes array."""
        if len(closes) < period:
            return np.full(len(closes), np.nan)
        k = 2.0 / (period + 1)
        ema = np.empty(len(closes))
        ema[:] = np.nan
        ema[period - 1] = closes[:period].mean()
        for i in range(period, len(closes)):
            ema[i] = closes[i] * k + ema[i - 1] * (1 - k)
        return ema

    def _compute_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float:
        """Compute ATR (simple mean of TR) for the last `period` bars."""
        if len(closes) < period + 1:
            return 0.0
        h = highs[-(period + 1):]
        l = lows[-(period + 1):]
        c = closes[-(period + 1):]
        prev_c = c[:-1]
        tr = np.maximum(h[1:] - l[1:],
                        np.maximum(np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)))
        return float(tr.mean()) if len(tr) else 0.0
