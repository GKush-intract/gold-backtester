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
        "sl_atr_mult": ("float", 2.0, 0.1, 10.0, "Stop = entry -/+ mult*ATR"),
        "tp_atr_mult": ("float", 4.0, 0.1, 20.0, "Target = entry +/- mult*ATR"),
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
