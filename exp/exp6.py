"""Exp. 6: robustness of the MMFC portfolio to the scale set S, the
fluctuation-order set Q, and the preference profiles alpha(s)/beta(q)
(paper Sec. IV.C, Table II), at the fixed required return r_e = 1%.

Run: python -m exp.exp6
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.data_loader import load_log_returns
from src.backtest import PORTFOLIO_NAMES, run_rolling_backtest, summarize_backtest, tilted_weights
from src.config import (
    EMPIRICAL_END_DATE,
    EMPIRICAL_START_DATE,
    EMPIRICAL_TICKERS,
    EXP5_ES_CONF,
    EXP5_FIRST_WINDOW_END,
    EXP5_N_WINDOWS_PAPER,
    EXP5_ROLLING_WINDOW,
    EXP5_TRADING_DAYS_PER_YEAR,
    EXP5_VAR_CONF,
    EXP5_VAR_HORIZON_DAYS,
    EXP6_ALPHA_TILTS,
    EXP6_BASELINE_Q,
    EXP6_BASELINE_S,
    EXP6_BETA_TILTS,
    EXP6_Q_ALTERNATIVES,
    EXP6_RE_ANNUAL_PCT,
    EXP6_S_ALTERNATIVES,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _summary_row(results: dict, r_e: float, portfolio: str) -> dict[str, float]:
    """In-sample/out-of-sample average drawdown, VaR, ES, total return for one
    portfolio at one required return, from a `run_rolling_backtest` result."""
    row: dict[str, float] = {}
    for period in ("in_sample", "out_of_sample"):
        frame = pd.DataFrame(results[r_e][period][portfolio])
        for col in ("drawdown", "var", "es"):
            row[f"{period}_{col}"] = float(frame[col].mean())
        if period == "out_of_sample":
            row[f"{period}_total_return"] = float(frame.groupby("year")["total_return"].sum().mean())
        else:
            row[f"{period}_total_return"] = float(frame["total_return"].mean())
    return row


def run_spec(
    returns: pd.DataFrame,
    name: str,
    scales: list[int],
    qorders: list[float],
    alpha: dict[int, float] | None,
    beta: dict[float, float] | None,
    portfolios: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Run the full 131-window rolling backtest for one specification at
    r_e = 1% and return the summary row per requested portfolio."""
    results = run_rolling_backtest(
        returns,
        window=EXP5_ROLLING_WINDOW,
        first_window_end=EXP5_FIRST_WINDOW_END,
        scales=scales,
        qorders=qorders,
        required_returns_annual_pct=[EXP6_RE_ANNUAL_PCT],
        trading_days_per_year=EXP5_TRADING_DAYS_PER_YEAR,
        horizon_days=EXP5_VAR_HORIZON_DAYS,
        var_conf=EXP5_VAR_CONF,
        es_conf=EXP5_ES_CONF,
        alpha=alpha,
        beta=beta,
        n_windows=EXP5_N_WINDOWS_PAPER,
        portfolios=portfolios,
    )
    print(f"  spec '{name}': {len(results[EXP6_RE_ANNUAL_PCT]['in_sample'][portfolios[0]])} windows done")
    return {p: _summary_row(results, EXP6_RE_ANNUAL_PCT, p) for p in portfolios}


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
    print(f"Loaded {len(returns)} common trading dates for the robustness check at "
          f"r_e = {EXP6_RE_ANNUAL_PCT}%.")

    rows: list[dict] = []

    print("Baseline: MV, MC, MD, MMFC, MMFD at S0, Q0 ...")
    baseline = run_spec(
        returns, "baseline", EXP6_BASELINE_S, EXP6_BASELINE_Q, None, None, PORTFOLIO_NAMES
    )
    for portfolio, row in baseline.items():
        rows.append({"specification": "baseline", "varied_input": "none", "portfolio": portfolio, **row})

    specs: list[tuple[str, str, list[int], list[float], dict | None, dict | None]] = []
    for q_name, q_set in EXP6_Q_ALTERNATIVES.items():
        specs.append((q_name, "Q", EXP6_BASELINE_S, q_set, None, None))
    for s_name, s_set in EXP6_S_ALTERNATIVES.items():
        specs.append((s_name, "S", s_set, EXP6_BASELINE_Q, None, None))
    for tilt in EXP6_ALPHA_TILTS:
        alpha = tilted_weights(EXP6_BASELINE_S, tilt)
        specs.append((f"alpha_{tilt}", "alpha", EXP6_BASELINE_S, EXP6_BASELINE_Q, alpha, None))
    for tilt in EXP6_BETA_TILTS:
        beta = tilted_weights(EXP6_BASELINE_Q, tilt)
        specs.append((f"beta_{tilt}", "beta", EXP6_BASELINE_S, EXP6_BASELINE_Q, None, beta))

    for name, varied_input, scales, qorders, alpha, beta in specs:
        print(f"Specification '{name}' (varying {varied_input}) ...")
        out = run_spec(returns, name, scales, qorders, alpha, beta, ("MMFC",))
        rows.append({"specification": name, "varied_input": varied_input, "portfolio": "MMFC", **out["MMFC"]})

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "exp6_robustness.csv", index=False)

    # Flag the paper's two stated exceptions (Sec. IV.C): MMFC in-sample
    # drawdown exceeding MV's when beta(q) is tilted to small q, and when
    # Q is restricted to the small-fluctuation subset {1, 1.5, 2}.
    mv_in_drawdown = float(table.loc[(table.specification == "baseline") & (table.portfolio == "MV"), "in_sample_drawdown"].iloc[0])
    exceptions = {}
    for name in ("beta_small_q", "Q_small_1_2"):
        mmfc_row = table[(table.specification == name) & (table.portfolio == "MMFC")]
        if not mmfc_row.empty:
            exceptions[name] = {
                "mmfc_in_sample_drawdown": float(mmfc_row["in_sample_drawdown"].iloc[0]),
                "mv_in_sample_drawdown": mv_in_drawdown,
                "mmfc_exceeds_mv": bool(mmfc_row["in_sample_drawdown"].iloc[0] > mv_in_drawdown),
            }

    run_summary = {
        "n_specifications": len(specs) + 1,
        "required_return_annual_pct": EXP6_RE_ANNUAL_PCT,
        "n_windows_paper": EXP5_N_WINDOWS_PAPER,
        "non_massive_source": meta.get("non_massive_source"),
        "paper_stated_exceptions": exceptions,
    }
    with open(RESULTS_DIR / "exp6_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
