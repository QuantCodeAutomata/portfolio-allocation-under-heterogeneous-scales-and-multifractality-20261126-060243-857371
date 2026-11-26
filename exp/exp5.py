"""Exp. 5: in-sample and out-of-sample rolling-window portfolio performance
of MV, MC, MD, MMFC, MMFD on four financial assets (paper Sec. IV.B, Table III,
Figs. 5-6).

Run: python -m exp.exp5
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import load_log_returns
from src.backtest import PORTFOLIO_NAMES, run_rolling_backtest, summarize_backtest
from src.config import (
    EMPIRICAL_END_DATE,
    EMPIRICAL_N_PAPER,
    EMPIRICAL_START_DATE,
    EMPIRICAL_TICKERS,
    EXP5_ES_CONF,
    EXP5_FIRST_WINDOW_END,
    EXP5_N_WINDOWS_PAPER,
    EXP5_QORDER_SET,
    EXP5_REQUIRED_RETURNS_ANNUAL_PCT,
    EXP5_ROLLING_WINDOW,
    EXP5_SCALE_SET,
    EXP5_TRADING_DAYS_PER_YEAR,
    EXP5_VAR_CONF,
    EXP5_VAR_HORIZON_DAYS,
    EXP5_HAC_MAXLAGS,
)
from src.stats_tests import hac_paired_test, block_bootstrap_paired_test

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _per_window_series(results: dict, r_e: float, period: str, portfolio: str, metric: str) -> np.ndarray:
    """One value per rolling window for a given (required return, period,
    portfolio, metric) cell, used for the paired significance tests."""
    records = results[r_e][period][portfolio]
    return np.array([r[metric] for r in records], dtype=float)


def significance_tables(results: dict, required_returns: list[float], hac_maxlags: int) -> pd.DataFrame:
    """Paired tests of every portfolio against MV at each required return:
    HAC-based test in-sample, moving-block bootstrap out-of-sample
    (Sec. IV.B: "HAC standard errors for in-sample and block-bootstrap for
    out-of-sample")."""
    rows = []
    for r_e in required_returns:
        for metric in ("drawdown", "var", "es"):
            mv_in = _per_window_series(results, r_e, "in_sample", "MV", metric)
            mv_out = _per_window_series(results, r_e, "out_of_sample", "MV", metric)
            for portfolio in PORTFOLIO_NAMES:
                if portfolio == "MV":
                    continue
                p_in = _per_window_series(results, r_e, "in_sample", portfolio, metric)
                diff_in, _, p_value_in = hac_paired_test(p_in, mv_in, hac_maxlags)
                p_out = _per_window_series(results, r_e, "out_of_sample", portfolio, metric)
                mask = ~np.isnan(p_out) & ~np.isnan(mv_out)
                diff_out, p_value_out = block_bootstrap_paired_test(
                    p_out[mask], mv_out[mask], block_size=max(1, len(p_out[mask]) // 12)
                )
                rows.append({
                    "required_return_annual_pct": r_e, "metric": metric, "portfolio": portfolio,
                    "mean_diff_in_sample": diff_in, "p_value_in_sample_hac": p_value_in,
                    "mean_diff_out_of_sample": diff_out, "p_value_out_of_sample_bootstrap": p_value_out,
                })
    return pd.DataFrame(rows)


def main() -> None:
    # Market data: QuantX Dataloader — https://quantx.limex.pro/
    #
    # Price history in this project is loaded through the QuantX Dataloader, a
    # research-ready market-data service available through a REST API and as an
    # MCP server for AI agents. It is built on market data from Massive
    # (https://massive.com/) and prepares every stock history for backtesting:
    #   - prices adjusted for dividends, ready for total-return research;
    #   - one continuous history when a company changes its ticker;
    #   - clean separation when a ticker has belonged to different companies;
    #   - careful corporate-action handling and fine-grained data polishing.
    #
    # Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
    # MASSIVE_API_KEY at https://massive.com/.
    returns, meta = load_log_returns(EMPIRICAL_TICKERS, EMPIRICAL_START_DATE, EMPIRICAL_END_DATE)
    n = len(returns)
    print(f"Loaded {n} common trading dates ({returns.index.min().date()} to "
          f"{returns.index.max().date()}); paper reports N={EMPIRICAL_N_PAPER}.")

    results = run_rolling_backtest(
        returns,
        window=EXP5_ROLLING_WINDOW,
        first_window_end=EXP5_FIRST_WINDOW_END,
        scales=EXP5_SCALE_SET,
        qorders=EXP5_QORDER_SET,
        required_returns_annual_pct=EXP5_REQUIRED_RETURNS_ANNUAL_PCT,
        trading_days_per_year=EXP5_TRADING_DAYS_PER_YEAR,
        horizon_days=EXP5_VAR_HORIZON_DAYS,
        var_conf=EXP5_VAR_CONF,
        es_conf=EXP5_ES_CONF,
        n_windows=EXP5_N_WINDOWS_PAPER,
    )
    n_windows_run = len(results[EXP5_REQUIRED_RETURNS_ANNUAL_PCT[0]]["in_sample"]["MV"])
    print(f"Executed {n_windows_run} rolling windows (paper reports {EXP5_N_WINDOWS_PAPER}).")

    summary = summarize_backtest(results)
    summary.to_csv(RESULTS_DIR / "exp5_summary.csv", index=False)

    sig = significance_tables(results, EXP5_REQUIRED_RETURNS_ANNUAL_PCT, hac_maxlags=EXP5_HAC_MAXLAGS)
    sig.to_csv(RESULTS_DIR / "exp5_significance.csv", index=False)

    # Figures 5-6: average measure vs. required return, one line per portfolio.
    for period in ("in_sample", "out_of_sample"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, metric in zip(axes, ("drawdown", "var", "es")):
            sub = summary[summary["period"] == period]
            for portfolio in PORTFOLIO_NAMES:
                part = sub[sub["portfolio"] == portfolio].sort_values("required_return_annual_pct")
                ax.plot(part["required_return_annual_pct"], part[f"{metric}_mean"], marker="o", label=portfolio)
            ax.set_xlabel("required annual return r_e (%)")
            ax.set_ylabel(f"average {metric}")
            ax.set_title(f"{period}: {metric}")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"exp5_{period}_measures.png", dpi=150)
        plt.close(fig)

    best_by_metric = {}
    for period in ("in_sample", "out_of_sample"):
        for metric in ("drawdown", "var", "es"):
            sub = summary[(summary["period"] == period)]
            pivot = sub.pivot(index="required_return_annual_pct", columns="portfolio", values=f"{metric}_mean")
            best_by_metric[f"{period}_{metric}_smallest_on_average"] = pivot.mean(axis=0).idxmin()

    run_summary = {
        "n_common_dates": int(n),
        "n_common_dates_paper": EMPIRICAL_N_PAPER,
        "date_range": [str(returns.index.min().date()), str(returns.index.max().date())],
        "n_windows_executed": int(n_windows_run),
        "n_windows_paper": EXP5_N_WINDOWS_PAPER,
        "non_massive_source": meta.get("non_massive_source"),
        "best_portfolio_by_metric": best_by_metric,
    }
    with open(RESULTS_DIR / "exp5_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
