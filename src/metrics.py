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

    m["max_drawdown_abs"] = float(ec["drawdown"].min())
    m["max_drawdown_pct"] = float(ec["drawdown_pct"].min())
    m["max_drawdown_duration_bars"] = _max_dd_duration(ec["drawdown"].to_numpy())

    total_bars = len(ec)
    m["exposure_pct"] = float(trades["bars_held"].sum() / total_bars) if total_bars else 0.0

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
