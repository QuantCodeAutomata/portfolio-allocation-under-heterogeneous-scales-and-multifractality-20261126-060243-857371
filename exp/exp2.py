"""exp_2: Multifractal diversification: Markov-switching multifractal (MSM)
efficient frontiers (paper Sec. III.B).

Generates a single realization of the MSM pair (x, y), scans the qDCCA
coefficient (Eq. 13) over fluctuation orders q and scales s for both the
MFCCA and MFDCCA algorithms, and compares the mean-variance (MV) efficient
frontier (Eq. 8) against the MFCCA/MFDCCA efficient frontiers (Eq. 9, 14)
built from the fluctuation matrix at a few (s, q) combinations. Run with:

    python -m exp.exp2
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    MSM_MX, MSM_MY, MSM_K, MSM_B, MSM_GAMMA_K, MSM_SIGMA, MSM_N, MSM_SEED,
    EXP2_SCALES, EXP2_QORDERS_FRONTIER, EXP2_QORDERS_QDCCA, EXP2_N_FRONTIER_POINTS,
)
from src.mfcca import qdcca_coefficient, fluctuation_matrix
from src.msm import generate_msm_pair
from src.portfolio import covariance_risk_matrix, expected_returns_sample_mean, efficient_frontier

RESULTS_DIR = Path("results")
ALGORITHMS = ("MFCCA", "MFDCCA")


def run_qdcca_scan(x: np.ndarray, y: np.ndarray) -> list[dict]:
    """qDCCA coefficient (Eq. 13) over every (algorithm, scale, q) combination."""
    rows: list[dict] = []
    for algorithm in ALGORITHMS:
        for s in EXP2_SCALES:
            for q in EXP2_QORDERS_QDCCA:
                coeff = qdcca_coefficient(x, y, s, q, algorithm)
                rows.append({"algorithm": algorithm, "scale": s, "q": q, "qdcca": coeff})
    return rows


def save_qdcca_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algorithm", "scale", "q", "qdcca"])
        writer.writeheader()
        writer.writerows(rows)


def plot_qdcca(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, len(ALGORITHMS), figsize=(11, 4.5), sharey=True)
    for ax, algorithm in zip(axes, ALGORITHMS):
        for s in EXP2_SCALES:
            qs = [r["q"] for r in rows if r["algorithm"] == algorithm and r["scale"] == s]
            coeffs = [r["qdcca"] for r in rows if r["algorithm"] == algorithm and r["scale"] == s]
            ax.plot(qs, coeffs, marker="o", label=f"s={s}")
        ax.set_xlabel("fluctuation order q")
        ax.set_title(f"{algorithm} qDCCA(q, s)")
        ax.set_xlim(0.5, 5)
        ax.axhline(0.0, color="gray", linewidth=0.5)
        ax.legend()
    axes[0].set_ylabel("qDCCA coefficient")
    fig.suptitle("exp_2: MSM qDCCA coefficient vs. fluctuation order q")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_frontiers(x: np.ndarray, y: np.ndarray, expected_returns: np.ndarray) -> tuple[list[dict], dict]:
    """Build the MV frontier once and the MFCCA/MFDCCA frontiers over the
    (algorithm, s, q) grid. The MV frontier does not depend on (s, q), so it
    is stored once with model="MV", s=None, q=None, rather than repeated for
    every (s, q) combination.
    """
    rows: list[dict] = []
    frontier_lookup: dict = {}

    mv_matrix = covariance_risk_matrix(np.column_stack([x, y]))
    mv_frontier = efficient_frontier(mv_matrix, expected_returns, n_points=EXP2_N_FRONTIER_POINTS)
    frontier_lookup[("MV", None, None)] = mv_frontier
    for point in mv_frontier:
        rows.append({
            "model": "MV", "s": None, "q": None,
            "required_return": point["required_return"],
            "risk": point["risk"],
            "risk_reduction_rate": point["risk_reduction_rate"],
        })

    for algorithm in ALGORITHMS:
        for s in EXP2_SCALES:
            for q in EXP2_QORDERS_FRONTIER:
                matrix, _ = fluctuation_matrix({"x": x, "y": y}, s, q, algorithm)
                frontier = efficient_frontier(matrix, expected_returns, n_points=EXP2_N_FRONTIER_POINTS)
                frontier_lookup[(algorithm, s, q)] = frontier
                for point in frontier:
                    rows.append({
                        "model": algorithm, "s": s, "q": q,
                        "required_return": point["required_return"],
                        "risk": point["risk"],
                        "risk_reduction_rate": point["risk_reduction_rate"],
                    })
    return rows, frontier_lookup


def save_frontiers_csv(rows: list[dict], path: Path) -> None:
    fieldnames = ["model", "s", "q", "required_return", "risk", "risk_reduction_rate"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_frontier_by_qorder(frontier_lookup: dict, path: Path) -> None:
    """One subplot per algorithm (MFCCA, MFDCCA), fixed s=100 (the middle
    scale), one line per q in EXP2_QORDERS_FRONTIER, plus the MV frontier as
    a dashed reference line in both subplots."""
    s_fixed = EXP2_SCALES[len(EXP2_SCALES) // 2]
    mv_frontier = frontier_lookup[("MV", None, None)]
    mv_x = [p["required_return"] for p in mv_frontier]
    mv_y = [p["risk_reduction_rate"] for p in mv_frontier]

    fig, axes = plt.subplots(1, len(ALGORITHMS), figsize=(11, 4.5), sharey=True)
    for ax, algorithm in zip(axes, ALGORITHMS):
        ax.plot(mv_x, mv_y, linestyle="--", color="black", label="MV")
        for q in EXP2_QORDERS_FRONTIER:
            frontier = frontier_lookup[(algorithm, s_fixed, q)]
            fx = [p["required_return"] for p in frontier]
            fy = [p["risk_reduction_rate"] for p in frontier]
            ax.plot(fx, fy, marker=".", label=f"q={q}")
        ax.set_xlabel("required return $r_e$")
        ax.set_title(f"{algorithm}, s={s_fixed}")
        ax.legend()
    axes[0].set_ylabel("risk reduction rate (%)")
    fig.suptitle(
        "exp_2: frontier straightens as q grows (weaker cross-correlation for large fluctuations)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_summary(x: np.ndarray, y: np.ndarray, qdcca_rows: list[dict], frontier_rows: list[dict]) -> dict:
    def qdcca_at(q: float, s: int, algorithm: str) -> float:
        for r in qdcca_rows:
            if r["algorithm"] == algorithm and r["scale"] == s and r["q"] == q:
                return r["qdcca"]
        raise KeyError((algorithm, s, q))

    rrr_ranges = []
    for model in ("MV",) + ALGORITHMS:
        combos = {(r["s"], r["q"]) for r in frontier_rows if r["model"] == model}
        for s, q in sorted(combos, key=lambda t: (t[0] is None, t[0], t[1] is None, t[1])):
            values = [
                r["risk_reduction_rate"] for r in frontier_rows
                if r["model"] == model and r["s"] == s and r["q"] == q
            ]
            rrr_ranges.append({
                "model": model, "s": s, "q": q,
                "risk_reduction_rate_range": max(values) - min(values),
            })

    return {
        "n_assets": 2,
        "series_length": len(x),
        "qdcca_q0.5_s100_MFCCA": qdcca_at(0.5, 100, "MFCCA"),
        "qdcca_q5_s100_MFCCA": qdcca_at(5, 100, "MFCCA"),
        "risk_reduction_rate_ranges": rrr_ranges,
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    x, y = generate_msm_pair(
        MSM_MX, MSM_MY, MSM_K, MSM_B, MSM_GAMMA_K, MSM_SIGMA, MSM_N, seed=MSM_SEED
    )
    print(f"x: length={len(x)}, mean={x.mean():.6f}, std={x.std(ddof=1):.6f}")
    print(f"y: length={len(y)}, mean={y.mean():.6f}, std={y.std(ddof=1):.6f}")

    qdcca_rows = run_qdcca_scan(x, y)
    save_qdcca_csv(qdcca_rows, RESULTS_DIR / "exp2_qdcca.csv")
    plot_qdcca(qdcca_rows, RESULTS_DIR / "exp2_qdcca.png")

    def qdcca_at(q: float, s: int, algorithm: str) -> float:
        return next(r["qdcca"] for r in qdcca_rows if r["algorithm"] == algorithm and r["scale"] == s and r["q"] == q)

    mfcca_q05_s100 = qdcca_at(0.5, 100, "MFCCA")
    mfcca_q5_s100 = qdcca_at(5, 100, "MFCCA")
    print(f"MFCCA qDCCA(q=0.5, s=100) = {mfcca_q05_s100:.6f}")
    print(f"MFCCA qDCCA(q=5, s=100) = {mfcca_q5_s100:.6f}")

    decreasing_somewhere = any(
        qdcca_at(0.5, s, "MFCCA") > qdcca_at(5, s, "MFCCA") for s in EXP2_SCALES
    )
    if not decreasing_somewhere:
        print(
            "WARNING: qDCCA(q=0.5) is not larger than qDCCA(q=5) at any scale; "
            "expected weaker (but still positive) cross-correlation for large fluctuations."
        )

    returns = np.column_stack([x, y])
    expected_returns = expected_returns_sample_mean(returns)
    frontier_rows, frontier_lookup = run_frontiers(x, y, expected_returns)
    save_frontiers_csv(frontier_rows, RESULTS_DIR / "exp2_frontiers.csv")
    plot_frontier_by_qorder(frontier_lookup, RESULTS_DIR / "exp2_frontier_by_qorder.png")

    summary = build_summary(x, y, qdcca_rows, frontier_rows)
    with open(RESULTS_DIR / "exp2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote qDCCA table ({len(qdcca_rows)} rows) and frontier table ({len(frontier_rows)} rows) to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
