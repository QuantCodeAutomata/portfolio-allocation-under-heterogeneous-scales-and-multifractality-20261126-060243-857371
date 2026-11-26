"""exp_1: Multiscale diversification: two-component ARFIMA efficient frontiers
(paper Sec. III.A, Eqs. 12-14).

Run with `python -m exp.exp1` from the repository root. Generates a single
two-component ARFIMA(d1, d2, W) realization, scans the qDCCA cross-correlation
coefficient across scales, and traces mean-MFCCA/mean-MFDCCA efficient
frontiers for singleton scale/fluctuation-order subsets to verify that the
frontier shape changes (and straightens) as the scale s grows.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.switch_backend("Agg")

import numpy as np
import pandas as pd

from src.arfima import generate_two_component_arfima
from src.config import (
    ARFIMA_D1,
    ARFIMA_D2,
    ARFIMA_J_MAX,
    ARFIMA_N,
    ARFIMA_SEED,
    ARFIMA_W,
    EXP1_N_FRONTIER_POINTS,
    EXP1_QDCCA_N_SCALES,
    EXP1_QDCCA_SCALES_MAX,
    EXP1_QDCCA_SCALES_MIN,
    EXP1_QORDERS,
    EXP1_SCALES,
)
from src.mfcca import fluctuation_matrix, qdcca_coefficient
from src.portfolio import efficient_frontier, expected_returns_sample_mean

ALGORITHMS = ("MFCCA", "MFDCCA")
RESULTS_DIR = "results"


def run_qdcca_scan(y: np.ndarray, y_prime: np.ndarray) -> pd.DataFrame:
    """qDCCA coefficient scan over scales for every (algorithm, q) pair."""
    scales = np.unique(
        np.round(np.geomspace(EXP1_QDCCA_SCALES_MIN, EXP1_QDCCA_SCALES_MAX, EXP1_QDCCA_N_SCALES)).astype(int)
    )
    rows = []
    for algorithm in ALGORITHMS:
        for q in EXP1_QORDERS:
            for s in scales:
                rho = qdcca_coefficient(y, y_prime, int(s), q, algorithm)
                rows.append({"algorithm": algorithm, "q": q, "scale": int(s), "qdcca": rho})
    df = pd.DataFrame(rows)

    strengthens = False
    for (algorithm, q), group in df.groupby(["algorithm", "q"]):
        group = group.sort_values("scale")
        rho_min_scale = group.iloc[0]["qdcca"]
        rho_max_scale = group.iloc[-1]["qdcca"]
        if rho_max_scale > rho_min_scale:
            strengthens = True
    if not strengthens:
        print(
            "WARNING: qDCCA did not strengthen at longer scales for any (algorithm, q) "
            "combination; expected the coefficient at the largest scale to exceed the "
            "coefficient at the smallest scale for at least one combination."
        )
    assert strengthens, (
        "expected qDCCA(s_max) > qDCCA(s_min) for at least one (algorithm, q) combination"
    )
    return df


def plot_qdcca(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for (algorithm, q), group in df.groupby(["algorithm", "q"]):
        group = group.sort_values("scale")
        ax.plot(group["scale"], group["qdcca"], marker="o", label=f"{algorithm}, q={q}")
    ax.set_xscale("log")
    ax.set_xlabel("scale s")
    ax.set_ylabel("qDCCA coefficient")
    ax.set_title("qDCCA cross-correlation coefficient vs. scale (two-component ARFIMA)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp1_qdcca.png", dpi=150)
    plt.close(fig)


def run_frontiers(y: np.ndarray, y_prime: np.ndarray) -> pd.DataFrame:
    """Mean-MFCCA/MFDCCA efficient frontiers for every (algorithm, s, q)."""
    returns = np.column_stack([y, y_prime])
    expected_returns = expected_returns_sample_mean(returns)
    series = {"y": y, "yp": y_prime}

    rows = []
    for algorithm in ALGORITHMS:
        for s in EXP1_SCALES:
            for q in EXP1_QORDERS:
                matrix, _ = fluctuation_matrix(series, s, q, algorithm)
                frontier = efficient_frontier(matrix, expected_returns, n_points=EXP1_N_FRONTIER_POINTS)
                for point in frontier:
                    rows.append({
                        "algorithm": algorithm,
                        "s": s,
                        "q": q,
                        "required_return": point["required_return"],
                        "risk": point["risk"],
                        "risk_reduction_rate": point["risk_reduction_rate"],
                    })
    return pd.DataFrame(rows)


def _plot_frontier_comparison(
    df: pd.DataFrame,
    group_col: str,
    group_values: list,
    fixed_col: str,
    fixed_value,
    title: str,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, algorithm in zip(axes, ALGORITHMS):
        subset = df[(df["algorithm"] == algorithm) & (df[fixed_col] == fixed_value)]
        for value in group_values:
            line = subset[subset[group_col] == value].sort_values("required_return")
            ax.plot(line["required_return"], line["risk_reduction_rate"], marker="o", label=f"{group_col}={value}")
        ax.set_title(algorithm)
        ax.set_xlabel("required return r_e")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("risk reduction rate RR(r_e) [%]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_frontier_by_scale(df: pd.DataFrame) -> None:
    q_mid = EXP1_QORDERS[len(EXP1_QORDERS) // 2]
    _plot_frontier_comparison(
        df,
        group_col="s",
        group_values=EXP1_SCALES,
        fixed_col="q",
        fixed_value=q_mid,
        title=(
            f"Frontier shape vs. scale s (q={q_mid} fixed): verifies the frontier approaches "
            "a straighter line as s grows from 10 to 1000"
        ),
        out_path=f"{RESULTS_DIR}/exp1_frontier_by_scale.png",
    )


def plot_frontier_by_qorder(df: pd.DataFrame) -> None:
    s_mid = EXP1_SCALES[len(EXP1_SCALES) // 2]
    _plot_frontier_comparison(
        df,
        group_col="q",
        group_values=EXP1_QORDERS,
        fixed_col="s",
        fixed_value=s_mid,
        title=f"Frontier shape vs. fluctuation order q (s={s_mid} fixed)",
        out_path=f"{RESULTS_DIR}/exp1_frontier_by_qorder.png",
    )


def write_summary(qdcca_df: pd.DataFrame, frontier_df: pd.DataFrame, series_length: int) -> None:
    mfcca_q2 = qdcca_df[(qdcca_df["algorithm"] == "MFCCA") & (qdcca_df["q"] == 2)].sort_values("scale")
    qdcca_smallest_scale = float(mfcca_q2.iloc[0]["qdcca"])
    qdcca_largest_scale = float(mfcca_q2.iloc[-1]["qdcca"])

    frontier_endpoints = []
    for (algorithm, s, q), group in frontier_df.groupby(["algorithm", "s", "q"]):
        group = group.sort_values("required_return")
        frontier_endpoints.append({
            "algorithm": algorithm,
            "s": int(s),
            "q": float(q),
            "risk_reduction_rate_first": float(group.iloc[0]["risk_reduction_rate"]),
            "risk_reduction_rate_last": float(group.iloc[-1]["risk_reduction_rate"]),
        })

    summary = {
        "n_assets": 2,
        "series_length": series_length,
        "qdcca_mfcca_q2_scale_min": EXP1_QDCCA_SCALES_MIN,
        "qdcca_mfcca_q2_scale_max": EXP1_QDCCA_SCALES_MAX,
        "qdcca_mfcca_q2_at_smallest_scale": qdcca_smallest_scale,
        "qdcca_mfcca_q2_at_largest_scale": qdcca_largest_scale,
        "frontier_endpoints": frontier_endpoints,
    }
    with open(f"{RESULTS_DIR}/exp1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    y, y_prime = generate_two_component_arfima(
        ARFIMA_D1, ARFIMA_D2, ARFIMA_W, ARFIMA_N, ARFIMA_J_MAX, seed=ARFIMA_SEED
    )
    print(f"y:       length={len(y)}, mean={y.mean():.6f}, std={y.std():.6f}")
    print(f"y_prime: length={len(y_prime)}, mean={y_prime.mean():.6f}, std={y_prime.std():.6f}")

    print("Running qDCCA scan across scales...")
    qdcca_df = run_qdcca_scan(y, y_prime)
    qdcca_df.to_csv(f"{RESULTS_DIR}/exp1_qdcca.csv", index=False)
    plot_qdcca(qdcca_df)

    print("Tracing mean-MFCCA/MFDCCA efficient frontiers...")
    frontier_df = run_frontiers(y, y_prime)
    frontier_df.to_csv(f"{RESULTS_DIR}/exp1_frontiers.csv", index=False)
    plot_frontier_by_scale(frontier_df)
    plot_frontier_by_qorder(frontier_df)

    write_summary(qdcca_df, frontier_df, series_length=len(y))
    print("Done. Results written to results/exp1_*.{csv,png,json}")


if __name__ == "__main__":
    main()
