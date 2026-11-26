"""Two-component ARFIMA process generator (paper Sec. III.A, Eq. 12).

Custom implementation: Context7 was consulted for `statsmodels` and `arch`;
neither implements the cross-coupled two-component ARFIMA construction of
Podobnik et al. (2008) used by the paper, so the recursion is built directly
from the fractional-differencing weights a_j(d).
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln, gammasgn

from src.config import ARFIMA_D1, ARFIMA_D2, ARFIMA_W, ARFIMA_N, ARFIMA_J_MAX


def arfima_weights(d: float, j_max: int) -> np.ndarray:
    """a_j(d) = Gamma(j-d) / (Gamma(-d) * Gamma(1+j)) for j=1,...,j_max (Eq. 12).

    Computed in log-space (gammaln/gammasgn) because Gamma(1+j) overflows
    float64 for j of a few hundred; magnitudes of a_j(d) stay bounded because
    the three log-gamma terms largely cancel (a_j(d) ~ j^(d-1)/Gamma(-d)).
    """
    j = np.arange(1, j_max + 1, dtype=float)
    log_mag = gammaln(j - d) - gammaln(-d) - gammaln(1.0 + j)
    sign = gammasgn(j - d) * gammasgn(-d) * gammasgn(1.0 + j)
    return sign * np.exp(log_mag)


def generate_two_component_arfima(
    d1: float = ARFIMA_D1,
    d2: float = ARFIMA_D2,
    w: float = ARFIMA_W,
    n: int = ARFIMA_N,
    j_max: int = ARFIMA_J_MAX,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the coupled pair (y_t, y'_t) of Eq. 12 with independent
    Gaussian innovations eps_t != eps'_t (zero mean, unit variance).

    y_t  = W*sum_j a_j(d1) y_{t-j}     + (1-W)*sum_j a_j(d2) y'_{t-j}     + eps_t
    y'_t = (1-W)*sum_j a_j(d1) y_{t-j} + W*sum_j a_j(d2) y'_{t-j}        + eps'_t

    a_j(d) are exactly the expansion coefficients of the fractional-difference
    filter (1-L)^d = sum_{j=0}^infty a_j(d) L^j (a_0=1; Hosking 1981, Granger
    & Joyeux 1980). Since (1-L)^d y_t = eps_t, solving for y_t gives
    y_t = eps_t - sum_{j>=1} a_j(d) y_{t-j}, i.e. the feedback sum enters with
    a MINUS sign (a_j(d) itself is negative for d>0, e.g. a_1(d)=-d), which is
    what reproduces the stated H=d+0.5 relationship; verified numerically
    against the generalized Hurst exponent estimator of mfcca.py.
    """
    assert -0.5 < d1 < 0.5 and -0.5 < d2 < 0.5, "d must lie in the stationary ARFIMA range (-0.5, 0.5)"
    assert 0.5 <= w <= 1.0, "W must lie in [0.5, 1] (Eq. 12)"
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    eps_prime = rng.standard_normal(n)  # independent innovations, eps_t != eps'_t

    j_max = min(j_max, n - 1)
    a_d1 = arfima_weights(d1, j_max)  # a_d1[k-1] = a_k(d1)
    a_d2 = arfima_weights(d2, j_max)

    y = np.zeros(n)
    y_prime = np.zeros(n)
    for t in range(n):
        window = min(t, j_max)
        if window > 0:
            # sum_{j=1}^{window} a_j(d) y[t-j] = dot(a_d[:window][::-1], y[t-window:t])
            sum1 = float(np.dot(a_d1[:window][::-1], y[t - window : t]))
            sum2 = float(np.dot(a_d2[:window][::-1], y_prime[t - window : t]))
        else:
            sum1 = sum2 = 0.0
        y[t] = -(w * sum1 + (1 - w) * sum2) + eps[t]
        y_prime[t] = -((1 - w) * sum1 + w * sum2) + eps_prime[t]
    return y, y_prime
