"""Downside-risk measures: historical VaR/ES and drawdown (paper Sec. III.C, IV.B).

Custom implementation: Context7 was consulted for `quantstats`, whose VaR/ES
helpers use a fixed horizon of 1 and a different quantile convention; the
paper's Basel-style positive-loss convention and explicit h-period horizon
(Eq. in Sec. III.C) are implemented directly.
"""
from __future__ import annotations

import numpy as np


def cumulative_h_period_returns(daily_returns: np.ndarray, horizon: int) -> np.ndarray:
    """Non-overlapping... actually rolling h-period cumulative log returns:
    sum of `horizon` consecutive daily returns, one value per valid start."""
    daily_returns = np.asarray(daily_returns, dtype=float)
    if len(daily_returns) < horizon:
        raise ValueError(f"need at least {horizon} daily returns, got {len(daily_returns)}")
    kernel = np.ones(horizon)
    return np.convolve(daily_returns, kernel, mode="valid")


def historical_var(h_period_returns: np.ndarray, confidence: float) -> float:
    """VaR_alpha(L_p) = Q_alpha(L_p) = -Q_{1-alpha}(R_p), reported as a positive loss."""
    losses = -np.asarray(h_period_returns, dtype=float)
    return float(np.percentile(losses, 100.0 * confidence))


def historical_es(h_period_returns: np.ndarray, confidence: float) -> float:
    """ES_alpha(L_p) = E[L_p | L_p >= VaR_alpha(L_p)], reported as a positive loss."""
    losses = -np.asarray(h_period_returns, dtype=float)
    var = historical_var(h_period_returns, confidence)
    tail = losses[losses >= var]
    if tail.size == 0:
        return var
    return float(tail.mean())


def max_drawdown(daily_returns: np.ndarray) -> float:
    """Largest peak-to-trough decline of the cumulative log-return path,
    reported as a positive percentage loss."""
    daily_returns = np.asarray(daily_returns, dtype=float)
    cumulative = np.cumsum(daily_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    return float(drawdown.max() * 100.0)


def total_return(daily_returns: np.ndarray) -> float:
    """Cumulative sum of daily log returns over the window (Sec. IV.B)."""
    return float(np.sum(daily_returns))
