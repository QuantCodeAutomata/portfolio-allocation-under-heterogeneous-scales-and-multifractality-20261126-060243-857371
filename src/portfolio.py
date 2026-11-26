"""Mean-MFCCA / mean-MFDCCA portfolio optimization (paper Sec. II.B, Eqs. 8-11, 14).

Custom implementation (paper Eqs. 8-11): Context7 was consulted for
`pyportfolioopt` and `skfolio`. Both assume a classical return-covariance risk
model built from a mean/covariance estimator; here the risk matrix is the
MFCCA/MFDCCA fluctuation matrix of Eq. 9, and the paper's optimization keeps
the required-return constraint as an inequality (not the equality assumed by
`EfficientFrontier.efficient_return`). A single custom quadratic-program
solver is therefore used for every risk matrix (MV included), so that the
inequality constraint, the no-short-sale bound, and the PSD regularization of
footnote [33] are handled identically across all portfolio types.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from src.mfcca import fluctuation_matrix


def covariance_risk_matrix(returns: np.ndarray) -> np.ndarray:
    """Sample covariance matrix, the MV risk measure of Eq. 8.

    `returns` holds one column per asset and one row per observation.
    """
    returns = np.asarray(returns, dtype=float)
    return np.cov(returns, rowvar=False, ddof=1)


def expected_returns_sample_mean(returns: np.ndarray) -> np.ndarray:
    """E(r_i) estimated as the sample mean of each asset's return series."""
    return np.asarray(returns, dtype=float).mean(axis=0)


def solve_aggregated_portfolio(
    series: dict[str, np.ndarray],
    scales: list[int],
    qorders: list[float],
    algorithm: str,
    expected_returns: np.ndarray,
    required_return: float | None = None,
    alpha: dict[int, float] | None = None,
    beta: dict[float, float] | None = None,
) -> tuple[np.ndarray, dict[tuple[float, int], np.ndarray]]:
    """Mean-MFCCA / mean-MFDCCA portfolio weight of Eq. 11.

    Solves Eq. 9 for every (q, s) in the Cartesian product of `qorders` and
    `scales` using the MFCCA/MFDCCA fluctuation matrix (`algorithm` in
    {"MFCCA", "MFDCCA"}), then aggregates the weights with Eq. 11. Singleton
    `scales`/`qorders` (as in the ARFIMA/MSM efficient-frontier experiments)
    reduce this to the single-matrix portfolio of Eq. 9. The paper's MC/MD
    benchmarks (Sec. II.B) are the `qorders=[2.0]` special case of this same
    aggregation, restricted to the scale set S.
    """
    weight_lookup: dict[tuple[float, int], np.ndarray] = {}
    for s in scales:
        for q in qorders:
            matrix, _ = fluctuation_matrix(series, s, q, algorithm)
            weight_lookup[(q, s)] = solve_min_risk_portfolio(matrix, expected_returns, required_return)
    weights = aggregate_weights(weight_lookup, scales, qorders, alpha, beta)
    return weights, weight_lookup


def nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Symmetrize and clip negative eigenvalues to zero (paper footnote [33])."""
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals_clipped = np.clip(eigvals, 0.0, None)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T


def solve_min_risk_portfolio(
    risk_matrix: np.ndarray,
    expected_returns: np.ndarray,
    required_return: float | None = None,
) -> np.ndarray:
    """Solve Eq. 9 (or Eq. 8 for MV): minimize (1/2) w'Mw s.t. sum w=1, w>=0,
    and w'E(r) >= required_return if `required_return` is not None (minimum-
    risk portfolio of Sec. III.C when the return constraint is inactive).
    """
    n = len(expected_returns)
    risk_matrix = nearest_psd(np.asarray(risk_matrix, dtype=float))
    expected_returns = np.asarray(expected_returns, dtype=float)

    def objective(w: np.ndarray) -> float:
        return 0.5 * w @ risk_matrix @ w

    def objective_grad(w: np.ndarray) -> np.ndarray:
        return risk_matrix @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones(n)}]
    if required_return is not None:
        constraints.append({
            "type": "ineq",
            "fun": lambda w: w @ expected_returns - required_return,
            "jac": lambda w: expected_returns,
        })
    bounds = [(0.0, 1.0)] * n

    best_w, best_obj = None, np.inf
    starts = [np.full(n, 1.0 / n)] + list(np.eye(n))
    for w0 in starts:
        result = minimize(
            objective, w0, jac=objective_grad, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if result.success and result.fun < best_obj:
            # clip tiny negative weights from numerical noise, then renormalize
            w = np.clip(result.x, 0.0, None)
            w = w / w.sum()
            feasible = required_return is None or w @ expected_returns >= required_return - 1e-6
            if feasible:
                best_w, best_obj = w, result.fun
    if best_w is None:
        raise RuntimeError("quadratic program failed to converge for any starting point")
    return best_w


def analytical_equality_solution(
    risk_matrix: np.ndarray, expected_returns: np.ndarray, required_return: float
) -> np.ndarray:
    """Equality-constrained Lagrange solution of Eq. 10, used only as a cross
    check of the numerical solver when no-short-sale bounds are inactive."""
    theta = np.linalg.inv(nearest_psd(risk_matrix))
    r = np.asarray(expected_returns, dtype=float)
    ones = np.ones_like(r)
    a = theta @ ones
    b = theta @ r
    A = ones @ a
    B = ones @ b
    C = r @ b
    D = A * C - B**2
    return ((C - B * required_return) * a + (A * required_return - B) * b) / D


def portfolio_risk(weights: np.ndarray, risk_matrix: np.ndarray) -> float:
    """R(w) = sqrt(w'Mw) (Eq. 14)."""
    value = weights @ nearest_psd(risk_matrix) @ weights
    return float(np.sqrt(max(value, 0.0)))


def efficient_frontier(
    risk_matrix: np.ndarray,
    expected_returns: np.ndarray,
    n_points: int = 25,
) -> list[dict]:
    """Trace the frontier w*(r_e) over required returns spanning the feasible
    range between the single-asset returns, and report its risk reduction
    rate (Eq. 14)."""
    expected_returns = np.asarray(expected_returns, dtype=float)
    r_lo, r_hi = expected_returns.min(), expected_returns.max()
    grid = np.linspace(r_lo, r_hi, n_points)
    points = []
    for r_e in grid:
        w = solve_min_risk_portfolio(risk_matrix, expected_returns, required_return=r_e)
        risk = portfolio_risk(w, risk_matrix)
        points.append({"required_return": float(r_e), "weights": w, "risk": risk})
    risk_max = max(p["risk"] for p in points)
    for p in points:
        p["risk_reduction_rate"] = 100.0 * (p["risk"] - risk_max) / risk_max if risk_max > 0 else 0.0
    return points


def aggregate_weights(
    weight_lookup: dict[tuple[float, int], np.ndarray],
    scales: list[int],
    qorders: list[float],
    alpha: dict[int, float] | None = None,
    beta: dict[float, float] | None = None,
) -> np.ndarray:
    """w_i^opt = sum_{s in S} sum_{q in Q} alpha(s) beta(q) w_i(q,s) (Eq. 11).

    `weight_lookup[(q, s)]` must already hold the optimal weight vector
    w(q, s). Uniform preference profiles alpha(s)=1/#S, beta(q)=1/#Q are used
    unless `alpha`/`beta` are given explicitly (Sec. II.B).
    """
    if alpha is None:
        alpha = {s: 1.0 / len(scales) for s in scales}
    if beta is None:
        beta = {q: 1.0 / len(qorders) for q in qorders}
    assert abs(sum(alpha.values()) - 1.0) < 1e-8, "alpha(s) must sum to one (Eq. 11)"
    assert abs(sum(beta.values()) - 1.0) < 1e-8, "beta(q) must sum to one (Eq. 11)"
    total = None
    for s in scales:
        for q in qorders:
            w = weight_lookup[(q, s)]
            term = alpha[s] * beta[q] * w
            total = term if total is None else total + term
    return total
