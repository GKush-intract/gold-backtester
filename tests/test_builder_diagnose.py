import numpy as np
import pandas as pd

from src.builder.diagnose import DIAG_REQUEST, build_digest
from src.engine import BacktestConfig, run_backtest
from src.metrics import compute_metrics
from src.strategy import Strategy


def _data(n=600):
    idx = pd.date_range("2026-01-05", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(7)
    base = 2000 + rng.normal(0, 1.0, n).cumsum()
    df = pd.DataFrame({"open": base, "high": base + 1.5, "low": base - 1.5,
                       "close": base + rng.normal(0, 0.5, n), "volume": 1.0}, index=idx)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    df.index.name = "timestamp"
    return df


class Churner(Strategy):
    """Enters periodically with a fixed bracket — guarantees trades for the digest."""
    name = "churner"

    def on_bar(self, ctx):
        if ctx.position is None and ctx.index % 25 == 10:
            price = ctx.bar["close"]
            ctx.enter("long", 0.1, stop_loss=price - 3, take_profit=price + 3)


class NeverTrades(Strategy):
    name = "never_trades"

    def on_bar(self, ctx):
        pass


def test_digest_has_all_sections_and_is_compact():
    data = _data()
    res = run_backtest(BacktestConfig(), Churner(), data)
    m, _ = compute_metrics(res)
    d = build_digest(res, m, data)
    for section in ["PERIOD", "RESULT", "R DIST", "EXIT REASONS", "BY UTC SESSION",
                    "BY WEEKDAY", "WORST TRADES", "BEST TRADES", "MONTHLY", "EQUITY"]:
        assert section in d, f"missing section {section}"
    # token efficiency: hard ceiling well under typical chat budgets (~2k tokens)
    assert len(d) < 8000, f"digest too large: {len(d)} chars"


def test_digest_handles_zero_trades():
    data = _data(200)
    res = run_backtest(BacktestConfig(), NeverTrades(), data)
    m, _ = compute_metrics(res)
    d = build_digest(res, m, data)
    assert "NO TRADES" in d


def test_diag_request_mentions_opt_in_context():
    assert "Diagnose with AI" in DIAG_REQUEST
    assert "finalize_revision" in DIAG_REQUEST
