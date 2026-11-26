"""Unit tests for portfolio optimization (src/portfolio.py) and risk measures (src/risk.py)."""
from __future__ import annotations

import numpy as np
import pytest

from src.portfolio import (
    nearest_psd,
    solve_min_risk_portfolio,
    analytical_equality_solution,
    portfolio_risk,
    efficient_frontier,
    aggregate_weights,
)
from src.risk import (
    cumulative_h_period_returns,
    historical_var,
    historical_es,
    max_drawdown,
    total_return,
)


def test_nearest_psd_fixes_negative_eigenvalues():
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues -1 and 3: not PSD
    fixed = nearest_psd(bad)
    eigvals = np.linalg.eigvalsh(fixed)
    assert np.all(eigvals >= -1e-10)


def test_min_risk_portfolio_weights_sum_to_one_and_nonnegative():
    risk_matrix = np.array([[1.0, 0.2], [0.2, 2.0]])
    returns = np.array([0.01, 0.02])
    w = solve_min_risk_portfolio(risk_matrix, returns)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(w >= -1e-8)  # no-short-selling constraint (Eq. 9)


def test_min_risk_portfolio_respects_required_return_constraint():
    risk_matrix = np.eye(2)
    returns = np.array([0.0, 0.05])
    w = solve_min_risk_portfolio(risk_matrix, returns, required_return=0.03)
    assert w @ returns >= 0.03 - 1e-6


def test_equal_risk_assets_get_equal_weight_absent_return_tilt():
    risk_matrix = np.eye(3)
    returns = np.array([0.01, 0.01, 0.01])
    w = solve_min_risk_portfolio(risk_matrix, returns)
    np.testing.assert_allclose(w, np.full(3, 1.0 / 3.0), atol=1e-4)


def test_analytical_vs_numerical_solution_agree_when_bounds_inactive():
    risk_matrix = np.array([[1.0, 0.1], [0.1, 1.5]])
    returns = np.array([0.01, 0.02])
    r_e = 0.015
    w_num = solve_min_risk_portfolio(risk_matrix, returns, required_return=r_e)
    w_ana = analytical_equality_solution(risk_matrix, returns, r_e)
    if np.all(w_ana >= -1e-6):  # only compare when the analytical (unbounded) solution is feasible
        np.testing.assert_allclose(w_num, w_ana, atol=1e-3)


def test_efficient_frontier_risk_reduction_rate_at_max_is_zero():
    risk_matrix = np.array([[1.0, 0.0], [0.0, 2.0]])
    returns = np.array([0.0, 0.05])
    points = efficient_frontier(risk_matrix, returns, n_points=10)
    rr_values = [p["risk_reduction_rate"] for p in points]
    assert max(rr_values) == pytest.approx(0.0, abs=1e-8)  # Eq. 14: RR=0 at R_max
    assert all(rr <= 1e-8 for rr in rr_values)  # every other point reduces risk (RR <= 0)


def test_efficient_frontier_diversifies_below_single_asset_risk():
    """At the midpoint return, the frontier's minimum-risk portfolio must not
    exceed the risk of either single asset held alone (diversification)."""
    risk_matrix = np.array([[1.0, -0.5], [-0.5, 1.0]])
    returns = np.array([0.0, 0.02])
    points = efficient_frontier(risk_matrix, returns, n_points=5)
    mid = points[2]
    single_asset_risk = min(np.sqrt(risk_matrix[0, 0]), np.sqrt(risk_matrix[1, 1]))
    assert mid["risk"] <= single_asset_risk + 1e-6


def test_aggregate_weights_uniform_matches_manual_average():
    lookup = {
        (1.0, 10): np.array([0.6, 0.4]),
        (1.0, 20): np.array([0.4, 0.6]),
    }
    w = aggregate_weights(lookup, scales=[10, 20], qorders=[1.0])
    np.testing.assert_allclose(w, np.array([0.5, 0.5]))


def test_aggregate_weights_rejects_non_normalized_preferences():
    lookup = {(1.0, 10): np.array([1.0, 0.0])}
    with pytest.raises(AssertionError):
        aggregate_weights(lookup, scales=[10], qorders=[1.0], alpha={10: 0.5}, beta={1.0: 1.0})


def test_cumulative_h_period_returns_length_and_sum():
    daily = np.array([0.01, -0.02, 0.03, 0.01, -0.01])
    cum = cumulative_h_period_returns(daily, horizon=2)
    assert len(cum) == 4
    assert cum[0] == pytest.approx(daily[0] + daily[1])


def test_historical_var_and_es_are_positive_losses_and_es_ge_var():
    rng = np.random.default_rng(0)
    returns = rng.standard_normal(5000) * 0.01
    var99 = historical_var(returns, 0.99)
    es975 = historical_es(returns, 0.975)
    assert var99 > 0  # reported as a positive loss
    assert es975 >= 0
    # with alpha_ES < alpha_VaR (0.975 < 0.99) the ES tail is wider, so ES<=VaR is not guaranteed;
    # instead check ES(99%) >= VaR(99%) which must always hold by definition
    es99 = historical_es(returns, 0.99)
    assert es99 >= var99 - 1e-8


def test_max_drawdown_known_path():
    # cumulative path: 0, 0.01, -0.01, 0.02 -> cumsum 0.01,0.00,0.02; max drawdown is 0.01 (from 0.01 to 0.00)
    daily = np.array([0.01, -0.01, 0.02])
    dd = max_drawdown(daily)
    assert dd == pytest.approx(1.0, abs=1e-6)  # reported in percent


def test_total_return_is_sum_of_daily_log_returns():
    daily = np.array([0.01, 0.02, -0.005])
    assert total_return(daily) == pytest.approx(daily.sum())
