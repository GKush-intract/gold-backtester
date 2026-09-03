from __future__ import annotations

import numpy as np
import pandas as pd

from ...strategy import Strategy


class XAUUSDEMARSIEFITrendM15(Strategy):
    """XAUUSD EMA-RSI-EFI Trend M15

    Long-only trend strategy. Enters when:
      - EMA(fast) crosses above EMA(slow)
      - RSI(rsi_period) > rsi_threshold
      - EFI(efi_period) > 0  [EMA of (close - prev_close) * volume]

    Exits on condition reversal (any signal flips) or TP/SL hit.
    SL distance = equity * risk_pct/100 / (lot_size * OZ_PER_LOT)
    TP = entry + SL distance (1R).
    Fixed lot sizing.
    """

    name = "XAUUSD EMA-RSI-EFI Trend M15"

    params = {
        "fast_ema_period": ("int", 5, 2, 50, "Period for the fast EMA used in the crossover condition."),
        "slow_ema_period": ("int", 13, 3, 200, "Period for the slow EMA used in the crossover condition."),
        "rsi_period": ("int", 14, 2, 50, "Lookback period for the RSI calculation (TradingView standard)."),
        "rsi_threshold": ("float", 50.0, 30.0, 70.0, "RSI must be above this value to allow entry."),
        "efi_period": ("int", 13, 2, 50, "EMA smoothing period for the Elder Force Index."),
        "lot_size": ("float", 0.1, 0.01, 100.0, "Fixed lot size per trade (1 lot = 100 oz of gold)."),
        "risk_pct": ("float", 5.0, 0.5, 20.0, "Maximum loss per trade as % of equity, sets SL distance."),
    }

    htf_timeframes = []

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _ema(series: np.ndarray, period: int) -> np.ndarray:
        """Compute EMA over a numpy array; return same-length array."""
        if len(series) == 0:
            return series.copy()
        alpha = 2.0 / (period + 1)
        out = np.empty(len(series))
        out[0] = series[0]
        for i in range(1, len(series)):
            out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
        return out

    @staticmethod
    def _rsi(close: np.ndarray, period: int) -> float:
        """Wilder-smoothed RSI (TradingView standard). Returns latest value."""
        if len(close) < period + 1:
            return float("nan")
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        # Initial averages (simple mean for first period)
        avg_gain = gains[:period].mean()
        avg_loss = losses[:period].mean()
        # Wilder smoothing for remainder
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    def _compute_indicators(self, hist: pd.DataFrame, fast: int, slow: int,
                             rsi_period: int, efi_period: int):
        """
        Returns (fast_ema_cur, fast_ema_prev, slow_ema_cur, slow_ema_prev,
                 rsi_cur, efi_cur)
        where _prev means the value one bar before the last bar.
        """
        close = hist["close"].to_numpy(dtype=float)
        volume = hist["volume"].to_numpy(dtype=float)

        # Need at least slow+1 bars for previous EMA
        need = max(slow, rsi_period, efi_period) + 2
        if len(close) < need:
            return None

        # EMA arrays
        fast_ema = self._ema(close, fast)
        slow_ema = self._ema(close, slow)

        fast_ema_cur = fast_ema[-1]
        fast_ema_prev = fast_ema[-2]
        slow_ema_cur = slow_ema[-1]
        slow_ema_prev = slow_ema[-2]

        # RSI
        # Use last (rsi_period * 3 + 1) bars to ensure warm-up is adequate
        rsi_lookback = min(len(close), max(rsi_period * 3 + 1, rsi_period + 10))
        rsi_cur = self._rsi(close[-rsi_lookback:], rsi_period)

        # EFI = EMA(efi_period) of (close - prev_close) * volume
        raw_efi = (close[1:] - close[:-1]) * volume[1:]
        efi_ema = self._ema(raw_efi, efi_period)
        efi_cur = efi_ema[-1]

        return fast_ema_cur, fast_ema_prev, slow_ema_cur, slow_ema_prev, rsi_cur, efi_cur

    # ---------------------------------------------------------------------------
    # Strategy logic
    # ---------------------------------------------------------------------------

    def on_bar(self, ctx):
        p = self.p
        hist = ctx.history

        fast = p["fast_ema_period"]
        slow = p["slow_ema_period"]
        rsi_period = p["rsi_period"]
        rsi_threshold = p["rsi_threshold"]
        efi_period = p["efi_period"]
        lot_size = p["lot_size"]
        risk_pct = p["risk_pct"]

        # Bound per-bar cost: the EMAs/RSI/EFI only need a recent tail of history.
        # 6x the longest period leaves EMA weights < e^-12 for anything older, so
        # values match full-history computation while keeping on_bar O(lookback).
        tail = max(slow, rsi_period, efi_period) * 6 + 2
        if len(hist) > tail:
            hist = hist.iloc[-tail:]

        result = self._compute_indicators(hist, fast, slow, rsi_period, efi_period)
        if result is None:
            return

        fast_ema_cur, fast_ema_prev, slow_ema_cur, slow_ema_prev, rsi_cur, efi_cur = result

        if np.isnan(rsi_cur):
            return

        # Crossover condition
        crossed_up = (fast_ema_cur > slow_ema_cur) and (fast_ema_prev <= slow_ema_prev)

        # Current signal state (used for exit check)
        ema_bull = fast_ema_cur > slow_ema_cur
        rsi_bull = rsi_cur > rsi_threshold
        efi_bull = efi_cur > 0.0

        # --- Manage existing position ---
        if ctx.position is not None:
            # Exit if any condition reverses
            if not (ema_bull and rsi_bull and efi_bull):
                ctx.close(reason="condition_reversal")
            return

        # --- Entry logic ---
        if crossed_up and rsi_bull and efi_bull:
            # Entry filled at next bar open (engine convention)
            entry_price = ctx.bar["close"]  # approximate; actual fill = next open

            # Dynamic SL distance: equity * risk_pct% / (lot_size * OZ_PER_LOT)
            oz_per_lot = 100.0
            sl_distance = (ctx.equity * risk_pct / 100.0) / (lot_size * oz_per_lot)

            if sl_distance <= 0:
                return

            stop_loss = entry_price - sl_distance
            take_profit = entry_price + sl_distance  # 1R

            from ...strategy import round_lots
            size = round_lots(lot_size)
            if size <= 0:
                return

            ctx.enter(
                "long",
                size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                tag="ema_rsi_efi_long",
            )
