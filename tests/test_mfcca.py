"""Unit tests for MFCCA/MFDCCA fluctuation functions and qDCCA (src/mfcca.py)."""
from __future__ import annotations

import numpy as np
import pytest

from src.mfcca import (
    profile,
    fluctuation_function,
    fluctuation_matrix,
    qdcca_coefficient,
    generalized_hurst_exponent,
)


def test_profile_is_cumulative_sum_of_demeaned_series():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    prof = profile(x)
    assert prof[-1] == pytest.approx(0.0, abs=1e-10)  # profile returns to zero at the end
    assert prof == pytest.approx(np.cumsum(x - x.mean()))


def test_fluctuation_function_positive_for_q2_self_correlation(rng):
    x = rng.standard_normal(2000)
    f_xx = fluctuation_function(x, x, s=50, q=2.0, algorithm="MFCCA")
    assert f_xx > 0  # F^2_XX is a variance-like quantity, must be positive


def test_fluctuation_matrix_symmetric(rng):
    series = {"a": rng.standard_normal(2000), "b": rng.standard_normal(2000)}
    matrix, names = fluctuation_matrix(series, s=50, q=2.0, algorithm="MFDCCA")
    assert names == ["a", "b"]
    assert matrix[0, 1] == pytest.approx(matrix[1, 0])
    assert matrix[0, 0] > 0 and matrix[1, 1] > 0


def test_qdcca_coefficient_near_one_for_identical_series(rng):
    x = rng.standard_normal(2000)
    rho = qdcca_coefficient(x, x, s=50, q=2.0, algorithm="MFCCA")
    assert rho == pytest.approx(1.0, abs=1e-8)  # self-correlation is exactly one


def test_qdcca_coefficient_bounded_and_low_for_independent_noise(rng):
    x = rng.standard_normal(4000)
    y = rng.standard_normal(4000)
    rho = qdcca_coefficient(x, y, s=50, q=2.0, algorithm="MFCCA")
    assert -1.0 <= rho <= 1.0
    assert abs(rho) < 0.3  # independent Gaussian noise: weak cross-correlation


def test_qdcca_coefficient_strong_for_identical_shared_signal(rng):
    common = rng.standard_normal(4000)
    x = common + 0.01 * rng.standard_normal(4000)
    y = common + 0.01 * rng.standard_normal(4000)
    rho = qdcca_coefficient(x, y, s=50, q=2.0, algorithm="MFCCA")
    assert rho > 0.9  # near-identical series must show strong positive cross-correlation


def test_generalized_hurst_exponent_recovers_known_value_for_white_noise(rng):
    x = rng.standard_normal(8192)
    h, scales, f_q = generalized_hurst_exponent(x, q=2.0, algorithm="MFDCCA")
    assert 0.3 < h < 0.7  # white noise has H close to 0.5
    assert np.all(scales > 0) and np.all(f_q > 0)


def test_fluctuation_function_rejects_unknown_algorithm(rng):
    x = rng.standard_normal(500)
    with pytest.raises(ValueError):
        fluctuation_function(x, x, s=50, q=2.0, algorithm="NOT_A_METHOD")
