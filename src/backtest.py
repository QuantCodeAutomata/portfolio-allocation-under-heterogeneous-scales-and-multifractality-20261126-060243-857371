"""Rolling-window in-sample / out-of-sample portfolio backtest (paper Sec. IV.B,
Table III/Figures 5-6; robustness design of Sec. IV.C, Table II).

Custom implementation (paper Sec. IV.B): `vectorbt` was consulted for rolling
backtests, but its portfolio-from-signals/from-orders engines assume
fixed trading rules on a price series, not a re-optimized quadratic program
at every roll with a multiscale/multifractal risk matrix; the paper's
window/rebalancing design is implemented directly on top of
`src.portfolio.solve_min_risk_portfolio` and `src.portfolio.aggregate_weights`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.mfcca import fluctuation_matrix
from src.portfolio import (
    aggregate_weights,
    covariance_risk_matrix,
    expected_returns_sample_mean,
    nearest_psd,
    solve_min_risk_portfolio,
)
from src.risk import cumulative_h_period_returns, historical_es, historical_var, max_drawdown, total_return


PORTFOLIO_NAMES = ("MV", "MC", "MD", "MMFC", "MMFD")


@dataclass
class RollingWindow:
    """One rolling-window split: training slice and the following test month."""

    train_start: int
    train_end: int  # inclusive index of the last training observation
    test_start: int
    test_end: int  # inclusive index of the last test observation
    window_end_date: pd.Timestamp
    test_month: pd.Period


def make_rolling_windows(
    dates: pd.DatetimeIndex,
    window: int,
    first_window_end: str,
    n_windows: int | None = None,
) -> list[RollingWindow]:
    """Build the 131-window rolling design of Sec. IV.B: the first window of
    `window` observations ends at the last trading date on/before
    `first_window_end`, in-sample test month is the following calendar month,
    and the window is shifted forward by one month until the data ends."""
    dates = pd.DatetimeIndex(dates)
    months = pd.PeriodIndex(dates, freq="M")
    first_end_period = pd.Period(first_window_end, freq="M")
    candidate_idx = np.where(months <= first_end_period)[0]
    if candidate_idx.size == 0:
        raise ValueError(f"no observations on or before {first_window_end}")
    end_idx = int(candidate_idx[-1])

    windows: list[RollingWindow] = []
    while True:
        if end_idx + 1 - window < 0:
            raise ValueError("insufficient history for the first rolling window")
        train_start = end_idx + 1 - window
        test_month = months[end_idx] + 1
        test_mask = months == test_month
        if not test_mask.any():
            break
        test_idx = np.where(test_mask)[0]
        windows.append(RollingWindow(
            train_start=train_start,
            train_end=end_idx,
            test_start=int(test_idx[0]),
            test_end=int(test_idx[-1]),
            window_end_date=dates[end_idx],
            test_month=test_month,
        ))
        end_idx = int(test_idx[-1])
        if n_windows is not None and len(windows) >= n_windows:
            break
    return windows


def annual_return_to_daily(r_e_annual_pct: float, trading_days_per_year: int) -> float:
    """Convert an annual percentage required return to a daily log-return
    target; not specified in the paper beyond the 0%-3% annual grid of
    Sec. IV.B, so the standard trading-day annualization is used."""
    return (r_e_annual_pct / 100.0) / trading_days_per_year


def window_fluctuation_cache(
    window_returns: pd.DataFrame,
    scales: list[int],
    qorders: list[float],
    algorithms: tuple[str, ...] = ("MFCCA", "MFDCCA"),
) -> dict[tuple[str, float, int], np.ndarray]:
    """Precompute the MFCCA and/or MFDCCA fluctuation matrices for every
    (s, q) pair once per window, shared across the MC/MD (q=2 subset) and
    MMFC/MMFD (full S x Q aggregation) portfolios (paper Eq. 9, Eq. 11)."""
    series = {col: window_returns[col].to_numpy(dtype=float) for col in window_returns.columns}
    cache: dict[tuple[str, float, int], np.ndarray] = {}
    for algorithm in algorithms:
        for s in scales:
            for q in qorders:
                matrix, _ = fluctuation_matrix(series, s, q, algorithm)
                cache[(algorithm, q, s)] = matrix
    return cache


def solve_window_portfolios(
    window_returns: pd.DataFrame,
    scales: list[int],
    qorders: list[float],
    required_return_daily: float | None,
    alpha: dict[int, float] | None = None,
    beta: dict[float, float] | None = None,
    fluctuation_cache: dict[tuple[str, float, int], np.ndarray] | None = None,
    portfolios: tuple[str, ...] = PORTFOLIO_NAMES,
) -> dict[str, np.ndarray]:
    """Solve the requested subset of the MV, MC, MD, MMFC, MMFD portfolios of
    Sec. IV.B for one rolling window, reusing a shared fluctuation-matrix
    cache so that each (algorithm, q, s) quadratic program is solved exactly
    once per window. Only the algorithms needed by `portfolios` are solved
    (MFCCA for MC/MMFC, MFDCCA for MD/MMFD), so a robustness run that only
    needs MMFC (Sec. IV.C) does not pay for the unused MFDCCA grid."""
    expected = expected_returns_sample_mean(window_returns.to_numpy(dtype=float))
    needs_mfcca = "MC" in portfolios or "MMFC" in portfolios
    needs_mfdcca = "MD" in portfolios or "MMFD" in portfolios
    if fluctuation_cache is None:
        algorithms = [a for a, needed in (("MFCCA", needs_mfcca), ("MFDCCA", needs_mfdcca)) if needed]
        fluctuation_cache = window_fluctuation_cache(window_returns, scales, qorders, algorithms)

    weights: dict[str, np.ndarray] = {}
    if "MV" in portfolios:
        weights["MV"] = solve_min_risk_portfolio(
            covariance_risk_matrix(window_returns.to_numpy(dtype=float)), expected, required_return_daily
        )

    lookups: dict[str, dict[tuple[float, int], np.ndarray]] = {}
    for algorithm, needed in (("MFCCA", needs_mfcca), ("MFDCCA", needs_mfdcca)):
        if not needed:
            continue
        lookup: dict[tuple[float, int], np.ndarray] = {}
        for s in scales:
            for q in qorders:
                lookup[(q, s)] = solve_min_risk_portfolio(
                    fluctuation_cache[(algorithm, q, s)], expected, required_return_daily
                )
        lookups[algorithm] = lookup

    if "MC" in portfolios:
        weights["MC"] = aggregate_weights(lookups["MFCCA"], scales, [2.0])
    if "MD" in portfolios:
        weights["MD"] = aggregate_weights(lookups["MFDCCA"], scales, [2.0])
    if "MMFC" in portfolios:
        weights["MMFC"] = aggregate_weights(lookups["MFCCA"], scales, qorders, alpha, beta)
    if "MMFD" in portfolios:
        weights["MMFD"] = aggregate_weights(lookups["MFDCCA"], scales, qorders, alpha, beta)
    return weights


def portfolio_performance_measures(
    daily_returns: np.ndarray,
    horizon_days: int,
    var_conf: float,
    es_conf: float,
) -> dict[str, float]:
    """Monthly drawdown, h-day historical VaR/ES, and total return for one
    return path (paper Sec. IV.B: in-sample within the estimation window,
    out-of-sample within the held-out month)."""
    daily_returns = np.asarray(daily_returns, dtype=float)
    h_period = cumulative_h_period_returns(daily_returns, horizon_days)
    return {
        "drawdown": max_drawdown(daily_returns),
        "var": historical_var(h_period, var_conf),
        "es": historical_es(h_period, es_conf),
        "total_return": total_return(daily_returns),
    }


def monthly_drawdown_average(daily_returns: np.ndarray, month_labels: pd.PeriodIndex) -> float:
    """Average of the largest peak-to-trough decline computed separately
    within each calendar month spanned by `daily_returns` (Sec. IV.B:
    "the largest peak-to-trough decline ... within each month")."""
    frame = pd.DataFrame({"r": np.asarray(daily_returns, dtype=float)}, index=month_labels)
    drawdowns = frame.groupby(level=0)["r"].apply(lambda s: max_drawdown(s.to_numpy()))
    return float(drawdowns.mean())


def run_rolling_backtest(
    returns: pd.DataFrame,
    window: int,
    first_window_end: str,
    scales: list[int],
    qorders: list[float],
    required_returns_annual_pct: list[float],
    trading_days_per_year: int,
    horizon_days: int,
    var_conf: float,
    es_conf: float,
    alpha: dict[int, float] | None = None,
    beta: dict[float, float] | None = None,
    n_windows: int | None = None,
    portfolios: tuple[str, ...] = PORTFOLIO_NAMES,
) -> dict[float, dict[str, list[dict[str, float]]]]:
    """Roll the estimation window forward one month at a time (Sec. IV.B).

    Returns, for every annual required return in `required_returns_annual_pct`,
    a dict keyed "in_sample"/"out_of_sample" -> portfolio name -> list of
    per-window performance-measure dicts (drawdown, var, es, total_return),
    one entry per rolling window.
    """
    dates = returns.index
    month_labels = pd.PeriodIndex(dates, freq="M")
    windows = make_rolling_windows(dates, window, first_window_end, n_windows)

    results: dict[float, dict[str, dict[str, list[dict[str, float]]]]] = {
        r_e: {"in_sample": {p: [] for p in portfolios}, "out_of_sample": {p: [] for p in portfolios}}
        for r_e in required_returns_annual_pct
    }

    values = returns.to_numpy(dtype=float)
    columns = list(returns.columns)
    for w in windows:
        train_frame = pd.DataFrame(values[w.train_start:w.train_end + 1], columns=columns)
        test_frame = pd.DataFrame(values[w.test_start:w.test_end + 1], columns=columns)
        test_month_labels = month_labels[w.test_start:w.test_end + 1]
        train_month_labels = month_labels[w.train_start:w.train_end + 1]
        needs_mfcca = "MC" in portfolios or "MMFC" in portfolios
        needs_mfdcca = "MD" in portfolios or "MMFD" in portfolios
        cache_algorithms = tuple(a for a, needed in (("MFCCA", needs_mfcca), ("MFDCCA", needs_mfdcca)) if needed)
        cache = window_fluctuation_cache(train_frame, scales, qorders, cache_algorithms)

        for r_e in required_returns_annual_pct:
            r_e_daily = annual_return_to_daily(r_e, trading_days_per_year) if r_e > 0 else None
            weights = solve_window_portfolios(
                train_frame, scales, qorders, r_e_daily, alpha, beta, fluctuation_cache=cache, portfolios=portfolios
            )
            for name in portfolios:
                w_vec = weights[name]
                in_sample_returns = train_frame.to_numpy() @ w_vec
                out_sample_returns = test_frame.to_numpy() @ w_vec

                in_measures = portfolio_performance_measures(in_sample_returns, horizon_days, var_conf, es_conf)
                in_measures["drawdown"] = monthly_drawdown_average(in_sample_returns, train_month_labels)
                results[r_e]["in_sample"][name].append(in_measures)

                if len(out_sample_returns) >= horizon_days:
                    out_measures = portfolio_performance_measures(out_sample_returns, horizon_days, var_conf, es_conf)
                else:
                    out_measures = {"drawdown": np.nan, "var": np.nan, "es": np.nan, "total_return": float(np.sum(out_sample_returns))}
                out_measures["drawdown"] = max_drawdown(out_sample_returns)
                out_measures["year"] = int(w.test_month.year)
                results[r_e]["out_of_sample"][name].append(out_measures)
    return results


def summarize_backtest(
    results: dict[float, dict[str, dict[str, list[dict[str, float]]]]],
) -> pd.DataFrame:
    """Average in-sample and out-of-sample measures over all rolling windows
    for every (required return, portfolio) pair, as in Figs. 5-6."""
    rows = []
    for r_e, by_period in results.items():
        for period, by_portfolio in by_period.items():
            for name, records in by_portfolio.items():
                frame = pd.DataFrame(records)
                row = {"required_return_annual_pct": r_e, "period": period, "portfolio": name}
                for col in ("drawdown", "var", "es"):
                    row[f"{col}_mean"] = float(frame[col].mean())
                if period == "out_of_sample":
                    yearly_total = frame.groupby("year")["total_return"].sum()
                    row["total_return_mean"] = float(yearly_total.mean())
                else:
                    row["total_return_mean"] = float(frame["total_return"].mean())
                rows.append(row)
    return pd.DataFrame(rows)


def tilted_weights(values: list[float], tilt: str) -> dict[float, float]:
    """Linear preference-weight tilt normalized to sum to one (Sec. IV.C:
    "linearly proportional to 1/s" or "proportional to s", same for q)."""
    values = list(values)
    if tilt == "short_scale" or tilt == "small_q":
        raw = {v: 1.0 / v for v in values}
    elif tilt == "long_scale" or tilt == "large_q":
        raw = {v: float(v) for v in values}
    else:
        raise ValueError(f"unknown tilt '{tilt}'")
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}
