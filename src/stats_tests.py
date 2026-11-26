"""Paired significance tests used for portfolio comparisons (paper Sec. III.C, IV.B).

`scipy.stats.wilcoxon` and `statsmodels` provide the Wilcoxon signed-rank test
and HAC covariance estimators used directly below (no custom reimplementation
needed); the Holm step-down correction is implemented from its standard
definition since neither library target exposes it as a plain function.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.stattools import durbin_watson  # noqa: F401  (imported for completeness checks)
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def holm_correction(p_values: list[float]) -> list[float]:
    """Holm (1979) step-down adjustment of a list of raw p-values."""
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p_values[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def pairwise_wilcoxon_holm(samples: dict[str, np.ndarray]) -> pd.DataFrame:
    """Wilcoxon signed-rank test with Holm correction for every pair of
    named paired samples (paper Sec. III.C: "Wilcoxon signed-rank tests with
    Holm correction for all pairwise comparisons")."""
    names = list(samples.keys())
    rows = []
    for a, b in combinations(names, 2):
        stat, p_raw = wilcoxon(samples[a], samples[b])
        rows.append({"pair": f"{a} vs {b}", "statistic": stat, "p_raw": p_raw})
    p_adj = holm_correction([row["p_raw"] for row in rows])
    for row, p in zip(rows, p_adj):
        row["p_holm"] = p
    return pd.DataFrame(rows)


def hac_mean_se(x: np.ndarray, maxlags: int) -> tuple[float, float]:
    """Mean of `x` and its Newey-West (Bartlett-kernel) HAC standard error,
    obtained by regressing x on a constant with `statsmodels` HAC covariance."""
    x = np.asarray(x, dtype=float)
    model = OLS(x, add_constant(np.ones_like(x))).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(model.params[0]), float(model.bse[0])


def hac_paired_test(x: np.ndarray, y: np.ndarray, maxlags: int) -> tuple[float, float, float]:
    """Paired difference test (x - y) with a HAC standard error; returns
    (mean difference, t-statistic, two-sided p-value)."""
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    model = OLS(diff, add_constant(np.ones_like(diff))).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )
    return float(model.params[0]), float(model.tvalues[0]), float(model.pvalues[0])


def block_bootstrap_paired_test(
    x: np.ndarray, y: np.ndarray, block_size: int, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Moving-block bootstrap p-value for the paired mean difference (x - y),
    used for out-of-sample significance testing (Sec. IV.B)."""
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    n = len(diff)
    rng = np.random.default_rng(seed)
    observed = diff.mean()
    n_blocks = int(np.ceil(n / block_size))
    boot_means = np.empty(n_boot)
    starts = np.arange(n - block_size + 1)
    for b in range(n_boot):
        chosen = rng.choice(starts, size=n_blocks)
        sample = np.concatenate([diff[s : s + block_size] for s in chosen])[:n]
        boot_means[b] = sample.mean()
    centered = boot_means - boot_means.mean()
    p_value = float(np.mean(np.abs(centered) >= np.abs(observed)))
    return observed, p_value
