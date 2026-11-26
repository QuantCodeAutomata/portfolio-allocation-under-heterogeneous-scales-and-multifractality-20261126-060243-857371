"""Markov-switching multifractal (MSM) process generator (paper Sec. III.B, Eq. 15).

Custom implementation (paper Eq. 15): Context7 was consulted for `arch` and
`hmmlearn`; neither implements the MSM multiplicative-cascade volatility
model of Calvet-Fisher used here, so the renewal cascade is built directly.
"""
from __future__ import annotations

import numpy as np

from src.config import (
    MSM_MX, MSM_MY, MSM_K, MSM_B, MSM_GAMMA_K, MSM_SIGMA, MSM_N,
    MSM_MULTIPLIER_DRAW_PROB,
)


def renewal_probabilities(k: int = MSM_K, b: float = MSM_B, gamma_k: float = MSM_GAMMA_K) -> np.ndarray:
    """gamma_i = 1 - (1 - gamma_k)^{b^{i-k}}, i=1,...,k (Eq. 15)."""
    i = np.arange(1, k + 1, dtype=float)
    return 1.0 - (1.0 - gamma_k) ** (b ** (i - k))


def generate_msm_pair(
    m_x: float = MSM_MX,
    m_y: float = MSM_MY,
    k: int = MSM_K,
    b: float = MSM_B,
    gamma_k: float = MSM_GAMMA_K,
    sigma: float = MSM_SIGMA,
    n: int = MSM_N,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the pair (x_t, y_t) of Eq. 15 sharing the same renewal
    structure (which multiplier is redrawn, and the same low/high draw when
    it is) and the same Gaussian innovation u_t, with different multiplier
    strengths m_x, m_y. The shared binary draw is an assumption documented in
    results/RESULTS.md since the paper only states that the renewal structure
    and the innovation u_t are shared, not the multiplier-value draw itself.
    """
    gamma = renewal_probabilities(k, b, gamma_k)  # shape (k,)
    rng = np.random.default_rng(seed)

    u = rng.standard_normal(n)  # shared Gaussian innovation u_t
    renew_draw = rng.random((n, k))  # shared renewal-timing uniform draws
    value_draw = rng.random((n, k))  # shared low/high draw when a multiplier is renewed

    mult_x = np.ones(k)
    mult_y = np.ones(k)
    sigma_x = np.empty(n)
    sigma_y = np.empty(n)
    for t in range(n):
        renew = renew_draw[t] < gamma
        low_draw = value_draw[t] < MSM_MULTIPLIER_DRAW_PROB
        new_mx = np.where(low_draw, m_x, 2.0 - m_x)
        new_my = np.where(low_draw, m_y, 2.0 - m_y)
        mult_x = np.where(renew, new_mx, mult_x)
        mult_y = np.where(renew, new_my, mult_y)
        sigma_x[t] = sigma * np.sqrt(np.prod(mult_x))
        sigma_y[t] = sigma * np.sqrt(np.prod(mult_y))

    x = sigma_x * u
    y = sigma_y * u
    return x, y
