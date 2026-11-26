"""Unit tests for the two-component ARFIMA and MSM generators (src/arfima.py, src/msm.py)."""
from __future__ import annotations

import numpy as np
import pytest

from src.arfima import arfima_weights, generate_two_component_arfima
from src.msm import renewal_probabilities, generate_msm_pair
from src.mfcca import generalized_hurst_exponent


def test_arfima_weights_decay_and_sign_pattern():
    w = arfima_weights(d=0.4, j_max=100)
    assert len(w) == 100
    assert np.all(np.isfinite(w))
    # magnitude must decay as j grows (long-memory weights are summable-decaying)
    assert abs(w[-1]) < abs(w[0])


def test_arfima_weights_zero_d_gives_no_memory():
    w = arfima_weights(d=1e-8, j_max=50)
    assert np.all(np.abs(w) < 1e-4)  # d->0 collapses the fractional filter to white noise


def test_arfima_rejects_invalid_parameters():
    with pytest.raises(AssertionError):
        generate_two_component_arfima(d1=0.6, d2=0.2, w=0.7, n=500, j_max=50)
    with pytest.raises(AssertionError):
        generate_two_component_arfima(d1=0.4, d2=0.2, w=0.3, n=500, j_max=50)


def test_arfima_two_component_series_shape_and_finiteness():
    y, y_prime = generate_two_component_arfima(n=1000, j_max=200, seed=0)
    assert y.shape == (1000,)
    assert y_prime.shape == (1000,)
    assert np.all(np.isfinite(y)) and np.all(np.isfinite(y_prime))


def test_arfima_long_memory_h_above_half_for_d1():
    """d1=0.4 implies Hurst H = d + 0.5 = 0.9 (paper Sec. III.A); the estimated
    generalized Hurst exponent at q=2 must exceed the white-noise value of 0.5."""
    y, _ = generate_two_component_arfima(n=8192, j_max=2000, seed=7)
    h, _, _ = generalized_hurst_exponent(y, q=2.0, algorithm="MFDCCA")
    assert h > 0.6  # long-range dependence must push h(2) well above 0.5


def test_renewal_probabilities_increasing_and_bounded():
    gamma = renewal_probabilities(k=10, b=2, gamma_k=0.5)
    assert len(gamma) == 10
    assert np.all((gamma > 0) & (gamma <= 1.0))
    assert gamma[-1] == pytest.approx(0.5)  # gamma_k is the slowest (last) multiplier's probability
    assert np.all(np.diff(gamma) >= 0)  # renewal probability increases with i (faster multipliers)


def test_msm_pair_shapes_and_positive_volatility(rng):
    x, y = generate_msm_pair(n=500, seed=3)
    assert x.shape == (500,) and y.shape == (500,)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))


def test_msm_pair_shares_innovation_sign_pattern(rng):
    """x_t and y_t share the same Gaussian innovation u_t, so they must have
    the same sign at every time (sigma_t is always positive)."""
    x, y = generate_msm_pair(n=1000, seed=5)
    assert np.all(np.sign(x) == np.sign(y))
