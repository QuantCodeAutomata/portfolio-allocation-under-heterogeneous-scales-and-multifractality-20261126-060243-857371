"""Exp. 4: empirical multifractal scaling and multiscale/amplitude-dependent
cross-correlation structure of four financial return series (paper Sec. IV.A).

Run: python -m exp.exp4
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import load_log_returns
from src.config import (
    EMPIRICAL_END_DATE,
    EMPIRICAL_N_PAPER,
    EMPIRICAL_START_DATE,
    EMPIRICAL_TICKERS,
    EXP4_CROSSOVER_SCALE,
    EXP4_DCCA_SCALES,
    EXP4_HURST_Q_MAX,
    EXP4_HURST_Q_MIN,
    EXP4_HURST_Q_STEP,
    EXP4_QDCCA_QORDERS,
    EXP4_QDCCA_SCALE_FIXED,
    HURST_SCALING_RANGE,
)
from src.mfcca import fluctuation_function, generalized_hurst_exponent, qdcca_coefficient

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def scaling_curve(x: np.ndarray, q: float, scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[F^q_XX,MFCCA(s)]^{1/q} at each scale in `scales` (Sec. IV.A scaling plot)."""
    log_s, log_f = [], []
    for s in scales:
        f = fluctuation_function(x, x, int(s), q, "MFCCA")
        if f <= 0:
            continue
        f_q = f ** (1.0 / q) if q != 0 else np.exp(f / 2.0)
        if f_q > 0:
            log_s.append(np.log(s))
            log_f.append(np.log(f_q))
    return np.array(log_s), np.array(log_f)


def hurst_over_ranges(x: np.ndarray, q_values: np.ndarray, n: int) -> pd.DataFrame:
    """h(q) over the full, short (20<=s<s_x), and long (s_x<=s<=N/5) ranges
    (Sec. IV.A: crossover scale s_x=200 trading days)."""
    lo, denom = HURST_SCALING_RANGE
    hi = n // denom
    s_x = EXP4_CROSSOVER_SCALE
    full_scales = np.unique(np.round(np.geomspace(lo, hi, 25)).astype(int))
    short_scales = full_scales[full_scales < s_x]
    long_scales = full_scales[full_scales >= s_x]

    rows = []
    for q in q_values:
        for label, scales in (("full", full_scales), ("short_s_lt_200", short_scales), ("long_s_ge_200", long_scales)):
            if len(scales) < 4:
                rows.append({"q": q, "range": label, "h_q": np.nan, "n_scales": len(scales)})
                continue
            h, _, _ = generalized_hurst_exponent(x, q, scales=scales)
            rows.append({"q": q, "range": label, "h_q": h, "n_scales": len(scales)})
    return pd.DataFrame(rows)


def dcca_scan(x: np.ndarray, y: np.ndarray, scales: list[int]) -> pd.DataFrame:
    """DCCA cross-correlation coefficient (qDCCA at q=2) across scales
    (Sec. IV.A: scale dependence of cross-correlations)."""
    rows = [{"scale": s, "dcca": qdcca_coefficient(x, y, s, 2.0, "MFCCA")} for s in scales]
    return pd.DataFrame(rows)


def qdcca_scan(x: np.ndarray, y: np.ndarray, s_fixed: int, q_values: list[float]) -> pd.DataFrame:
    """qDCCA coefficient across fluctuation orders at a fixed scale
    (Sec. IV.A: amplitude dependence of cross-correlations)."""
    rows = [{"q": q, "qdcca": qdcca_coefficient(x, y, s_fixed, q, "MFCCA")} for q in q_values]
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
    assets = list(returns.columns)
    n = len(returns)
    print(f"Loaded {n} common trading dates for {assets} "
          f"({returns.index.min().date()} to {returns.index.max().date()}); "
          f"paper reports N={EMPIRICAL_N_PAPER}.")
    print("non-massive sources:", meta.get("non_massive_source"))

    q_values = np.arange(EXP4_HURST_Q_MIN, EXP4_HURST_Q_MAX + 1e-9, EXP4_HURST_Q_STEP)
    hurst_rows = []
    scaling_rows = []
    for asset in assets:
        x = returns[asset].to_numpy()
        hurst = hurst_over_ranges(x, q_values, n)
        hurst["asset"] = asset
        hurst_rows.append(hurst)
        log_s, log_f = scaling_curve(x, 2.0, np.unique(np.round(np.geomspace(20, n // 5, 30)).astype(int)))
        for s, f in zip(np.exp(log_s), np.exp(log_f)):
            scaling_rows.append({"asset": asset, "scale": s, "F_q2": f})
    hurst_table = pd.concat(hurst_rows, ignore_index=True)
    scaling_table = pd.DataFrame(scaling_rows)
    hurst_table.to_csv(RESULTS_DIR / "exp4_generalized_hurst.csv", index=False)
    scaling_table.to_csv(RESULTS_DIR / "exp4_scaling_curve_q2.csv", index=False)

    multifractal_check = (
        hurst_table[hurst_table["range"] == "full"]
        .pivot(index="asset", columns="q", values="h_q")
    )
    is_decreasing = multifractal_check.apply(lambda row: row.is_monotonic_decreasing, axis=1)
    print("h(q) monotonically decreasing in q (full range) per asset:\n", is_decreasing)

    pairs = list(itertools.combinations(assets, 2))
    dcca_rows, qdcca_rows = [], []
    for a, b in pairs:
        x, y = returns[a].to_numpy(), returns[b].to_numpy()
        dcca = dcca_scan(x, y, EXP4_DCCA_SCALES)
        dcca["pair"] = f"{a}-{b}"
        dcca_rows.append(dcca)
        qd = qdcca_scan(x, y, EXP4_QDCCA_SCALE_FIXED, EXP4_QDCCA_QORDERS)
        qd["pair"] = f"{a}-{b}"
        qdcca_rows.append(qd)
    dcca_table = pd.concat(dcca_rows, ignore_index=True)
    qdcca_table = pd.concat(qdcca_rows, ignore_index=True)
    dcca_table.to_csv(RESULTS_DIR / "exp4_dcca_scale_scan.csv", index=False)
    qdcca_table.to_csv(RESULTS_DIR / "exp4_qdcca_amplitude_scan.csv", index=False)

    # Figure 1: ln F(q=2,s) vs ln s per asset, marking the crossover scale s_x.
    fig, ax = plt.subplots(figsize=(7, 5))
    for asset in assets:
        sub = scaling_table[scaling_table["asset"] == asset]
        ax.plot(np.log(sub["scale"]), np.log(sub["F_q2"]), marker="o", ms=3, label=asset)
    ax.axvline(np.log(EXP4_CROSSOVER_SCALE), color="gray", ls="--", label=f"s_x={EXP4_CROSSOVER_SCALE}")
    ax.set_xlabel("ln s")
    ax.set_ylabel("ln F(q=2, s)")
    ax.set_title("MFCCA scaling curves (q=2), Sec. IV.A")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4_scaling_curves.png", dpi=150)
    plt.close(fig)

    # Figure 2: h(q) vs q per asset and scaling range.
    fig, axes = plt.subplots(1, len(assets), figsize=(4 * len(assets), 4), sharey=True)
    for ax, asset in zip(axes, assets):
        sub = hurst_table[hurst_table["asset"] == asset]
        for label in ("full", "short_s_lt_200", "long_s_ge_200"):
            part = sub[sub["range"] == label]
            ax.plot(part["q"], part["h_q"], marker="o", ms=3, label=label)
        ax.set_title(asset)
        ax.set_xlabel("q")
    axes[0].set_ylabel("h(q)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4_hurst_by_q.png", dpi=150)
    plt.close(fig)

    # Figure 3: DCCA coefficient vs scale, all pairs.
    fig, ax = plt.subplots(figsize=(7, 5))
    for a, b in pairs:
        sub = dcca_table[dcca_table["pair"] == f"{a}-{b}"]
        ax.plot(sub["scale"], sub["dcca"], marker=".", ms=3, label=f"{a}-{b}")
    ax.set_xlabel("scale s")
    ax.set_ylabel("DCCA coefficient (q=2)")
    ax.set_title("Scale dependence of cross-correlations, Sec. IV.A")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4_dcca_scan.png", dpi=150)
    plt.close(fig)

    # Figure 4: qDCCA coefficient vs q at s=100, all pairs.
    fig, ax = plt.subplots(figsize=(7, 5))
    for a, b in pairs:
        sub = qdcca_table[qdcca_table["pair"] == f"{a}-{b}"]
        ax.plot(sub["q"], sub["qdcca"], marker="o", ms=3, label=f"{a}-{b}")
    ax.set_xlabel("fluctuation order q")
    ax.set_ylabel(f"qDCCA coefficient (s={EXP4_QDCCA_SCALE_FIXED})")
    ax.set_title("Amplitude dependence of cross-correlations, Sec. IV.A")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4_qdcca_scan.png", dpi=150)
    plt.close(fig)

    summary = {
        "n_common_dates": int(n),
        "n_common_dates_paper": EMPIRICAL_N_PAPER,
        "date_range": [str(returns.index.min().date()), str(returns.index.max().date())],
        "assets": assets,
        "non_massive_source": meta.get("non_massive_source"),
        "failed_tickers": meta.get("failed"),
        "multifractal_h_decreasing_full_range": {k: bool(v) for k, v in is_decreasing.to_dict().items()},
    }
    with open(RESULTS_DIR / "exp4_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
