"""Unit tests for paired significance tests (src/stats_tests.py)."""
from __future__ import annotations

import numpy as np
import pytest

from src.stats_tests import (
    holm_correction,
    pairwise_wilcoxon_holm,
    hac_mean_se,
    hac_paired_test,
    block_bootstrap_paired_test,
)


def test_holm_correction_monotone_and_bounded():
    raw = [0.001, 0.02, 0.5, 0.04]
    adjusted = holm_correction(raw)
    assert all(0.0 <= p <= 1.0 for p in adjusted)
    assert all(a >= r for a, r in zip(adjusted, raw))  # Holm only increases p-values


def test_holm_correction_single_pvalue_unchanged():
    assert holm_correction([0.03]) == pytest.approx([0.03])


def test_pairwise_wilcoxon_holm_detects_systematic_difference(rng):
    a = rng.standard_normal(100)
    b = a + 5.0  # b is always much larger: must reject equality
    df = pairwise_wilcoxon_holm({"a": a, "b": b})
    assert df.loc[0, "p_holm"] < 0.01


def test_pairwise_wilcoxon_holm_three_groups_produces_three_pairs(rng):
    samples = {name: rng.standard_normal(50) for name in ["x", "y", "z"]}
    df = pairwise_wilcoxon_holm(samples)
    assert len(df) == 3  # C(3,2) = 3 pairwise comparisons
    assert set(df["pair"]) == {"x vs y", "x vs z", "y vs z"}


def test_hac_mean_se_recovers_known_mean():
    rng = np.random.default_rng(1)
    x = 2.0 + 0.1 * rng.standard_normal(2000)
    mean, se = hac_mean_se(x, maxlags=5)
    assert mean == pytest.approx(2.0, abs=0.05)
    assert se > 0


def test_hac_paired_test_detects_shift():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(500) + 1.0
    y = rng.standard_normal(500)
    diff, tstat, pval = hac_paired_test(x, y, maxlags=5)
    assert diff == pytest.approx(1.0, abs=0.2)
    assert pval < 0.01


def test_block_bootstrap_detects_no_difference_for_identical_samples():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(300)
    _, p_value = block_bootstrap_paired_test(x, x, block_size=10, n_boot=200, seed=0)
    assert p_value == pytest.approx(1.0)  # zero observed difference is never exceeded in magnitude
