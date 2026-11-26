"""exp_3: Combined multiscale and multifractal portfolio performance with
Monte Carlo simulation (paper Sec. III.B-C, Table I).

For each of EXP3_N_REALIZATIONS_RUN Monte Carlo realizations, a 4-asset
synthetic market is built from one two-component ARFIMA pair (Eq. 12) and one
MSM pair (Eq. 15), and five minimum-risk portfolios (Sec. II.B, Eq. 9) are
solved from five different risk matrices:

  MV    - sample covariance (Eq. 8).
  MD    - "single-order Q={2} counterpart" of MMFD (Sec. III.C, Table I
          caption): weights aggregated via Eq. 11 over the same scale set S
          as MMFD/MMFC, with the fluctuation-order set restricted to {2}.
  MC    - the analogous Q={2} counterpart of MMFC (MFCCA fluctuation
          matrix).
  MMFD  - mean-MFDCCA portfolio: weights aggregated (Eq. 11) over the full
          scale x fluctuation-order grid, at the portfolio-weight level.
  MMFC  - mean-MFCCA portfolio: identical aggregation with the MFCCA
          fluctuation matrix.

10-day historical VaR/ES and total return are computed per portfolio per
realization; only the cross-realization mean/std and Wilcoxon-Holm pairwise
test tables are persisted (never the per-realization arrays themselves).
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.arfima import generate_two_component_arfima
from src.config import (
    ARFIMA_D1,
    ARFIMA_D2,
    ARFIMA_J_MAX,
    ARFIMA_N,
    ARFIMA_W,
    EXP3_ES_CONF,
    EXP3_HORIZON_DAYS,
    EXP3_N_REALIZATIONS_RUN,
    EXP3_QORDER_SET,
    EXP3_SCALE_SET,
    EXP3_SEED_BASE,
    EXP3_VAR_CONF,
    MSM_B,
    MSM_GAMMA_K,
    MSM_K,
    MSM_MX,
    MSM_MY,
    MSM_N,
    MSM_SIGMA,
)
from src.mfcca import fluctuation_matrix
from src.msm import generate_msm_pair
from src.portfolio import (
    aggregate_weights,
    covariance_risk_matrix,
    expected_returns_sample_mean,
    solve_min_risk_portfolio,
)
from src.risk import cumulative_h_period_returns, historical_es, historical_var, total_return
from src.stats_tests import pairwise_wilcoxon_holm

RESULTS_DIR = Path("results")

PORTFOLIO_NAMES = ["MV", "MD", "MMFD", "MC", "MMFC"]
ASSET_NAMES = ["arfima_y", "arfima_yp", "msm_x", "msm_y"]

# Table I caption: "In addition to MMFD and MMFC, their single-order Q={2}
# counterparts MD and MC are reported separately." MD/MC therefore use the
# same Eq. 11 aggregation as MMFD/MMFC over the full scale set S, but with
# the fluctuation-order set collapsed to the singleton {2}.
MD_MC_QORDER_SET = [2.0]


def _solve_min_risk_portfolio_scaled(
    risk_matrix: np.ndarray, expected_returns: np.ndarray
) -> np.ndarray:
    """Solve the minimum-risk portfolio after rescaling `risk_matrix` by its
    largest absolute entry.

    The QP of Eq. 9 minimizes 0.5*w'Mw subject to sum(w)=1, w>=0 (no return
    constraint here), a feasible set that does not depend on M; multiplying M
    by any positive constant therefore leaves the argmin w* unchanged. This
    rescaling is needed because the MFDCCA/MFCCA fluctuation matrices grow by
    several orders of magnitude at large scale and high fluctuation order
    (observed up to ~1e6 at s=1000, q=4.5-5.0), which otherwise makes the
    SLSQP solver inside `solve_min_risk_portfolio` fail to converge from a
    poorly-scaled objective.
    """
    scale = np.max(np.abs(risk_matrix))
    scale = scale if scale > 0 else 1.0
    return solve_min_risk_portfolio(risk_matrix / scale, expected_returns, None)


def _solve_weight_grid(
    series: dict[str, np.ndarray],
    expected_returns: np.ndarray,
    algorithm: str,
    qorders: list[float],
) -> np.ndarray:
    """Solve w(q, s) for every (s, q) in `EXP3_SCALE_SET x qorders` and
    aggregate via Eq. 11 (uniform alpha, beta)."""
    weight_lookup: dict[tuple[float, int], np.ndarray] = {}
    for s in EXP3_SCALE_SET:
        for q in qorders:
            matrix, _ = fluctuation_matrix(series, s, q, algorithm)
            weight_lookup[(q, s)] = _solve_min_risk_portfolio_scaled(matrix, expected_returns)
    return aggregate_weights(weight_lookup, EXP3_SCALE_SET, qorders)


def _run_realization(i: int) -> dict[str, dict[str, float]]:
    """Generate one Monte Carlo realization and return its per-portfolio
    VaR/ES/total-return metrics."""
    seed_arfima = EXP3_SEED_BASE + 2 * i
    seed_msm = EXP3_SEED_BASE + 2 * i + 1

    y, y_prime = generate_two_component_arfima(
        ARFIMA_D1, ARFIMA_D2, ARFIMA_W, ARFIMA_N, ARFIMA_J_MAX, seed=seed_arfima
    )
    x, y_msm = generate_msm_pair(
        MSM_MX, MSM_MY, MSM_K, MSM_B, MSM_GAMMA_K, MSM_SIGMA, MSM_N, seed=seed_msm
    )

    returns = np.column_stack([y, y_prime, x, y_msm])
    series = {"arfima_y": y, "arfima_yp": y_prime, "msm_x": x, "msm_y": y_msm}
    expected_returns = expected_returns_sample_mean(returns)

    weights: dict[str, np.ndarray] = {}

    mv_matrix = covariance_risk_matrix(returns)
    weights["MV"] = _solve_min_risk_portfolio_scaled(mv_matrix, expected_returns)

    weights["MD"] = _solve_weight_grid(series, expected_returns, "MFDCCA", MD_MC_QORDER_SET)
    weights["MC"] = _solve_weight_grid(series, expected_returns, "MFCCA", MD_MC_QORDER_SET)
    weights["MMFD"] = _solve_weight_grid(series, expected_returns, "MFDCCA", EXP3_QORDER_SET)
    weights["MMFC"] = _solve_weight_grid(series, expected_returns, "MFCCA", EXP3_QORDER_SET)

    metrics: dict[str, dict[str, float]] = {}
    for name in PORTFOLIO_NAMES:
        portfolio_returns = returns @ weights[name]
        h_returns = cumulative_h_period_returns(portfolio_returns, EXP3_HORIZON_DAYS)
        metrics[name] = {
            "var": historical_var(h_returns, EXP3_VAR_CONF),
            "es": historical_es(h_returns, EXP3_ES_CONF),
            "total_return": total_return(portfolio_returns),
        }
    return metrics


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    var_samples = {name: np.empty(EXP3_N_REALIZATIONS_RUN) for name in PORTFOLIO_NAMES}
    es_samples = {name: np.empty(EXP3_N_REALIZATIONS_RUN) for name in PORTFOLIO_NAMES}
    tr_samples = {name: np.empty(EXP3_N_REALIZATIONS_RUN) for name in PORTFOLIO_NAMES}

    start = time.time()
    for i in range(EXP3_N_REALIZATIONS_RUN):
        metrics = _run_realization(i)
        for name in PORTFOLIO_NAMES:
            var_samples[name][i] = metrics[name]["var"]
            es_samples[name][i] = metrics[name]["es"]
            tr_samples[name][i] = metrics[name]["total_return"]
    elapsed = time.time() - start
    print(
        f"Completed {EXP3_N_REALIZATIONS_RUN} Monte Carlo realizations in "
        f"{elapsed:.1f}s ({elapsed / EXP3_N_REALIZATIONS_RUN:.2f}s/realization)."
    )

    summary_rows = []
    for name in PORTFOLIO_NAMES:
        summary_rows.append(
            {
                "portfolio": name,
                "var_mean": var_samples[name].mean(),
                "var_std": var_samples[name].std(ddof=1),
                "es_mean": es_samples[name].mean(),
                "es_std": es_samples[name].std(ddof=1),
                "total_return_mean": tr_samples[name].mean(),
                "total_return_std": tr_samples[name].std(ddof=1),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "exp3_summary.csv", index=False)

    wilcoxon_var = pairwise_wilcoxon_holm(var_samples)
    wilcoxon_es = pairwise_wilcoxon_holm(es_samples)
    wilcoxon_tr = pairwise_wilcoxon_holm(tr_samples)
    wilcoxon_var.to_csv(RESULTS_DIR / "exp3_wilcoxon_var.csv", index=False)
    wilcoxon_es.to_csv(RESULTS_DIR / "exp3_wilcoxon_es.csv", index=False)
    wilcoxon_tr.to_csv(RESULTS_DIR / "exp3_wilcoxon_total_return.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    x_pos = np.arange(len(PORTFOLIO_NAMES))
    width = 0.35
    ax.bar(
        x_pos - width / 2,
        summary["var_mean"],
        width,
        yerr=summary["var_std"],
        capsize=4,
        label=f"VaR ({EXP3_VAR_CONF:.1%})",
    )
    ax.bar(
        x_pos + width / 2,
        summary["es_mean"],
        width,
        yerr=summary["es_std"],
        capsize=4,
        label=f"ES ({EXP3_ES_CONF:.1%})",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(PORTFOLIO_NAMES)
    ax.set_ylabel(f"{EXP3_HORIZON_DAYS}-day loss")
    ax.set_title("Mean VaR / ES across Monte Carlo realizations (error bars: std)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_var_es_barplot.png", dpi=150)
    plt.close(fig)

    # Informational ranking printout only (stochastic Monte Carlo comparison,
    # not asserted). Paper's expected ranking (Table I), worst -> best:
    expected_worst_to_best = ["MV", "MD", "MMFD", "MC", "MMFC"]
    expected_best_to_worst = expected_worst_to_best[::-1]
    var_rank = summary.sort_values("var_mean")["portfolio"].tolist()
    es_rank = summary.sort_values("es_mean")["portfolio"].tolist()
    print(f"Ranking by mean VaR (ascending, best first): {var_rank}")
    print(f"Ranking by mean ES  (ascending, best first): {es_rank}")
    print(f"Paper's expected ranking (worst -> best): {expected_worst_to_best}")
    print(f"VaR ranking matches paper's expected ranking: {var_rank == expected_best_to_worst}")
    print(f"ES  ranking matches paper's expected ranking: {es_rank == expected_best_to_worst}")


if __name__ == "__main__":
    main()
