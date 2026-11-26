"""Multifractal (de)trended cross-correlation analysis: MFCCA, MFDCCA, qDCCA.

Custom implementation (paper Sec. II.A, Eqs. 1-7, 13). Context7 was consulted
for `arch`, `statsmodels` and `ta`; none of them implement MFCCA/MFDCCA/qDCCA,
so the fluctuation functions below follow the paper's equations directly.
"""
from __future__ import annotations

import numpy as np

from src.config import POLY_ORDER, HURST_SCALING_RANGE


def profile(x: np.ndarray) -> np.ndarray:
    """Cumulative profile X(t) = sum_{i=1}^t (x_i - mean(x)) (Sec. II.A)."""
    x = np.asarray(x, dtype=float)
    return np.cumsum(x - x.mean())


def _segment_matrix(values: np.ndarray, s: int) -> np.ndarray:
    """2*floor(N/s) non-overlapping length-s segments taken from both ends.

    The division is repeated from the other end of the series so that no
    observation is dropped when N/s is not an integer (Sec. II.A).
    """
    n = len(values)
    n_s = n // s
    if n_s < 1:
        raise ValueError(f"scale s={s} exceeds series length N={n}")
    head = values[: n_s * s].reshape(n_s, s)
    tail = values[n - n_s * s :].reshape(n_s, s)
    return np.vstack([head, tail])  # shape (2*n_s, s)


def _detrending_projector(s: int, order: int = POLY_ORDER) -> np.ndarray:
    """Residual projector P = I - A(A'A)^{-1}A' for a degree-`order` polynomial
    fit on a length-s segment (Eq. 1). Applying P to a batch of segments
    (each a row) via `segments @ P.T` detrends every segment in one matrix
    product, since P is identical for all segments of a fixed scale s."""
    t = np.arange(1, s + 1, dtype=float)
    design = np.vander(t, N=order + 1, increasing=True)  # (s, order+1)
    projector = np.eye(s) - design @ np.linalg.pinv(design.T @ design) @ design.T
    return projector


def local_detrended_covariance(x: np.ndarray, y: np.ndarray, s: int) -> np.ndarray:
    """f^2_{XY}(s, v) for every segment v, signed (Eq. 1)."""
    profile_x, profile_y = profile(x), profile(y)
    segments_x = _segment_matrix(profile_x, s)
    segments_y = _segment_matrix(profile_y, s)
    projector = _detrending_projector(s)
    resid_x = segments_x @ projector.T
    resid_y = segments_y @ projector.T
    return (resid_x * resid_y).mean(axis=1)  # (1/s) sum_t ..., shape (2*N_s,)


def mfcca_fluctuation(x: np.ndarray, y: np.ndarray, s: int, q: float) -> float:
    """F^q_{XY,MFCCA}(s): sign-preserving fluctuation function (Eq. 3-4)."""
    f2 = local_detrended_covariance(x, y, s)
    if q == 0:
        with np.errstate(divide="ignore"):
            terms = np.sign(f2) * np.log(np.abs(f2))
        terms = terms[np.isfinite(terms)]
        return float(terms.mean())
    terms = np.sign(f2) * np.abs(f2) ** (q / 2.0)
    return float(terms.mean())


def mfdcca_fluctuation(x: np.ndarray, y: np.ndarray, s: int, q: float) -> float:
    """F^q_{XY,MFDCCA}(s): absolute-value fluctuation function (Eq. 5-7)."""
    profile_x, profile_y = profile(x), profile(y)
    segments_x = _segment_matrix(profile_x, s)
    segments_y = _segment_matrix(profile_y, s)
    projector = _detrending_projector(s)
    resid_x = segments_x @ projector.T
    resid_y = segments_y @ projector.T
    f2_abs = (np.abs(resid_x) * np.abs(resid_y)).mean(axis=1)  # Eq. 5
    if q == 0:
        return float(np.exp(np.log(f2_abs).mean() / 2.0))  # Eq. 7 (note the 1/(4N_s) normalization reduces to mean/2 of ln)
    return float((f2_abs ** (q / 2.0)).mean())  # Eq. 6


def fluctuation_function(x: np.ndarray, y: np.ndarray, s: int, q: float, algorithm: str) -> float:
    """Dispatch to the MFCCA or MFDCCA fluctuation function for a pair of series."""
    if algorithm == "MFCCA":
        return mfcca_fluctuation(x, y, s, q)
    if algorithm == "MFDCCA":
        return mfdcca_fluctuation(x, y, s, q)
    raise ValueError(f"unknown algorithm {algorithm!r}, expected 'MFCCA' or 'MFDCCA'")


def fluctuation_matrix(series: dict[str, np.ndarray], s: int, q: float, algorithm: str) -> tuple[np.ndarray, list[str]]:
    """n x n fluctuation matrix F^q_{ij,A}(s) for a dict of named asset series."""
    names = list(series.keys())
    n = len(names)
    matrix = np.empty((n, n))
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if j < i:
                matrix[i, j] = matrix[j, i]
            else:
                matrix[i, j] = fluctuation_function(series[name_i], series[name_j], s, q, algorithm)
    return matrix, names


def qdcca_coefficient(x: np.ndarray, y: np.ndarray, s: int, q: float, algorithm: str = "MFCCA") -> float:
    """rho_qDCCA(q, s) = F_XY,A^q(s) / sqrt(F_XX,A^q(s) * F_YY,A^q(s)) (Eq. 13)."""
    f_xy = fluctuation_function(x, y, s, q, algorithm)
    f_xx = fluctuation_function(x, x, s, q, algorithm)
    f_yy = fluctuation_function(y, y, s, q, algorithm)
    denom = np.sqrt(f_xx * f_yy)
    return float(f_xy / denom) if denom != 0 else np.nan


def generalized_hurst_exponent(
    x: np.ndarray, q: float, scales: np.ndarray | None = None, algorithm: str = "MFCCA"
) -> tuple[float, np.ndarray, np.ndarray]:
    """h(q) via log-log regression of [F^q_XX,A(s)]^{1/q} against s over a
    scaling range 20 <= s <= N/5 (Sec. II.A). Returns (h(q), scales, log F)."""
    n = len(x)
    lo, denom = HURST_SCALING_RANGE
    hi = n // denom
    if scales is None:
        scales = np.unique(np.round(np.geomspace(lo, hi, 20)).astype(int))
    scales = scales[(scales >= lo) & (scales <= hi)]
    log_s, log_f = [], []
    for s in scales:
        f = fluctuation_function(x, x, int(s), q, algorithm)
        if f <= 0:
            continue
        f_q = f ** (1.0 / q) if q != 0 else np.exp(f / 2.0)
        if f_q > 0:
            log_s.append(np.log(s))
            log_f.append(np.log(f_q))
    log_s, log_f = np.array(log_s), np.array(log_f)
    slope, _ = np.polyfit(log_s, log_f, 1)
    return float(slope), np.exp(log_s), np.exp(log_f)
