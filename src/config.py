"""Paper-derived parameters for every experiment, as module-level constants.

All parameter values are cited to the section/equation of "Portfolio Allocation
under Heterogeneous Scales and Multifractality" that specifies them, or marked
as not specified when the paper leaves them open.
"""

# ---------------------------------------------------------------------------
# Shared MFCCA / MFDCCA settings (Sec. II.A)
# ---------------------------------------------------------------------------
POLY_ORDER = 2  # degree of the local detrending polynomial (Sec. II.A, Eq. 1)
HURST_SCALING_RANGE = (20, 5)  # 20 <= s <= N/5, denominator of the upper bound (Sec. II.A)

# ---------------------------------------------------------------------------
# exp_1: two-component ARFIMA multiscale diversification (Sec. III.A, Eq. 12-14)
# ---------------------------------------------------------------------------
ARFIMA_D1 = 0.4  # long-memory parameter of series y_t (Eq. 12)
ARFIMA_D2 = 0.2  # long-memory parameter of series y'_t (Eq. 12)
ARFIMA_W = 0.7  # coupling weight, W in [0.5, 1] (Eq. 12); W=1 weakest, W=0.5 strongest
ARFIMA_N = 2 ** 13  # series length N=8192 (Sec. III.A)
ARFIMA_J_MAX = 5000  # truncation lag of the infinite sum a_j(d); not specified in the paper, "e.g. j_max=5000" per implementation notes
ARFIMA_SEED = 1  # fixed seed for the single-realization frontier experiment; not specified in the paper

EXP1_SCALES = [10, 100, 1000]  # scale subsets S={10},{100},{1000} used one at a time (Sec. III.A)
EXP1_QORDERS = [0.5, 2, 5]  # fluctuation order subsets Q={0.5},{2},{5} used one at a time (Sec. III.A)
EXP1_QDCCA_SCALES_MIN = 10  # qDCCA scan: scales s=10..1000 (Sec. III.A)
EXP1_QDCCA_SCALES_MAX = 1000
EXP1_QDCCA_N_SCALES = 25  # number of log-spaced scales in the qDCCA scan; not specified in the paper
EXP1_N_FRONTIER_POINTS = 25  # number of r_e grid points per frontier; not specified in the paper

# ---------------------------------------------------------------------------
# exp_2: Markov-switching multifractal (MSM) multifractal diversification (Sec. III.B, Eq. 15)
# ---------------------------------------------------------------------------
MSM_MX = 1.2  # multiplier parameter of series x_t (Sec. III.B)
MSM_MY = 1.5  # multiplier parameter of series y_t (Sec. III.B)
MSM_K = 10  # number of volatility multipliers per series (Sec. III.B)
MSM_B = 2  # renewal-probability geometric decay base b (Sec. III.B)
MSM_GAMMA_K = 0.5  # renewal probability of the slowest (k-th) multiplier (Sec. III.B)
MSM_SIGMA = 1.0  # baseline volatility sigma (Sec. III.B, Eq. 15)
MSM_N = 2 ** 13  # series length N=8192 (Sec. III.B)
MSM_SEED = 2  # fixed seed for the single-realization frontier experiment; not specified in the paper
MSM_MULTIPLIER_DRAW_PROB = 0.5  # probability of drawing m_z vs. 2-m_z when a multiplier is renewed; not specified in the paper, equal probability is assumed

EXP2_SCALES = [10, 100, 1000]  # scales s=10,100,1000 used to study fluctuation-order dependence (Sec. III.B)
EXP2_QORDERS_FRONTIER = [0.5, 2, 5]  # fluctuation order subsets Q={0.5},{2},{5} used one at a time (Sec. III.B)
EXP2_QORDERS_QDCCA = [0.5, 1, 2, 3, 4, 5]  # q grid for qDCCA coefficients across fluctuation orders (Sec. III.B)
EXP2_N_FRONTIER_POINTS = 25  # number of r_e grid points per frontier; not specified in the paper

# ---------------------------------------------------------------------------
# exp_3: combined multiscale + multifractal Monte Carlo performance (Sec. III.B-C, Table I)
# ---------------------------------------------------------------------------
EXP3_N_REALIZATIONS_PAPER = 50  # 50 Monte Carlo realizations (Table I caption)
EXP3_N_REALIZATIONS_RUN = 50  # executed count; kept equal to the paper's count (inexpensive relative to exp_5/6)
EXP3_SCALE_SET = [10, 100, 1000]  # S={10,100,1000} for MMFC/MMFD aggregation (Table I caption)
EXP3_QORDER_SET = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]  # Q={0.5,...,5} step 0.5 (Table I caption)
EXP3_HORIZON_DAYS = 10  # 10-period cumulative returns used for VaR/ES (Sec. III.B, Table I)
EXP3_VAR_CONF = 0.99  # 99% VaR (Table I)
EXP3_ES_CONF = 0.975  # 97.5% ES (Table I)
EXP3_SEED_BASE = 1000  # base seed; realization i uses seed EXP3_SEED_BASE + i; fixed-seed convention not specified in the paper

# ---------------------------------------------------------------------------
# exp_4 / exp_5 / exp_6: empirical assets (Sec. IV, data availability statement)
# ---------------------------------------------------------------------------
EMPIRICAL_START_DATE = "2011-01-03"  # January 2011 start of the sample (Sec. IV)
EMPIRICAL_END_DATE = "2023-11-30"  # November 2023 end of the sample (Sec. IV)
EMPIRICAL_N_PAPER = 3330  # N=3330 common trading-date observations reported in the paper (Sec. IV)

# Loaded through the QuantX Dataloader / Massive REST APIs in place of the
# histdata.com one-minute dealer quotes used by the paper (see
# results/RESULTS.md, "Deviations from the paper"). The Nikkei index
# (I:NKY) is not entitled on this Massive plan (NOT_AUTHORIZED) and WTI
# futures continuous contracts are not served by fetch-ohlcvs, so liquid
# tracking ETFs are used as tradable proxies for those two. The Massive
# forex spot symbol C:XAUUSD has only sparse daily coverage before mid-2013
# (confirmed by a direct query of its aggregates endpoint), so GLD (SPDR
# Gold Shares), the primary gold ETF tracking spot bullion, is used instead.
EMPIRICAL_TICKERS = {
    "NIKKEI": "JPXN",      # iShares JPX-Nikkei 400 ETF, proxy for the Nikkei stock index
    "SP500": "SPY",        # SPDR S&P 500 ETF Trust, proxy for the S&P 500 stock index
    "WTI": "USO",          # United States Oil Fund, proxy for WTI crude oil futures
    "XAU": "GLD",          # SPDR Gold Shares ETF, proxy for XAU/USD gold spot
}

EXP4_HURST_Q_MIN = -5  # generalized Hurst exponent estimated over q in [-5, 5] (Sec. IV.A)
EXP4_HURST_Q_MAX = 5
EXP4_HURST_Q_STEP = 1.0  # q grid step; not specified in the paper beyond the [-5,5] range
EXP4_CROSSOVER_SCALE = 200  # fixed crossover scale s_x=200 trading days for all assets (Sec. IV.A)
EXP4_DCCA_SCALES = list(range(10, 1001, 10))  # DCCA coefficient scan s=10,20,...,1000 (Sec. IV.A)
EXP4_QDCCA_SCALE_FIXED = 100  # qDCCA amplitude-dependence scan uses fixed scale s=100 (Sec. IV.A)
EXP4_QDCCA_QORDERS = [0.5, 1, 2, 3, 4, 5]  # q grid for the qDCCA amplitude-dependence scan (Sec. IV.A)

EXP5_ROLLING_WINDOW = 520  # N_w=520 data points, approx. 2 years (Sec. IV.B)
EXP5_FIRST_WINDOW_END = "2013-01-31"  # first window ends in January 2013 (Sec. IV.B)
EXP5_N_WINDOWS_PAPER = 131  # 131 rolling windows through November 2023 (Sec. IV.B)
EXP5_STEP_MONTHS = 1  # window shifted forward by one month (Sec. IV.B)
EXP5_SCALE_SET = [5, 30, 55, 80, 105, 130]  # S={5,30,...,130}, max = N_w/4 = 130 (Sec. IV.B)
EXP5_QORDER_SET = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]  # Q={1,...,4} step 0.5 (Sec. IV.B)
EXP5_REQUIRED_RETURNS_ANNUAL_PCT = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # annual r_e grid, 0%-3% (Sec. IV.B)
EXP5_TRADING_DAYS_PER_YEAR = 252  # used to convert annual r_e to a daily target return; not specified in the paper
EXP5_VAR_HORIZON_DAYS = 10  # 10-day VaR/ES (Sec. IV.B)
EXP5_VAR_CONF = 0.99
EXP5_ES_CONF = 0.975
EXP5_HAC_MAXLAGS = 12  # HAC (Bartlett-kernel) lag length for paired in-sample tests; not specified in the paper, set to approx. one window's overlap scale

# ---------------------------------------------------------------------------
# exp_6: robustness of the MMFC aggregation to S, Q, alpha, beta (Sec. IV.C, Table II)
# ---------------------------------------------------------------------------
EXP6_RE_ANNUAL_PCT = 1.0  # all robustness checks use r_e = 1% (Table II caption)
EXP6_BASELINE_S = EXP5_SCALE_SET  # S0={5,30,55,80,105,130} (Table II caption)
EXP6_BASELINE_Q = EXP5_QORDER_SET  # Q0={1,1.5,...,4} (Table II caption)

EXP6_Q_ALTERNATIVES = {
    "Q_wide_0.5_5": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    "Q_large_2_4": [2.0, 2.5, 3.0, 3.5, 4.0],
    "Q_small_1_2": [1.0, 1.5, 2.0],
}
EXP6_S_ALTERNATIVES = {
    "S_drop_5": [30, 55, 80, 105, 130],
    "S_dense": [5, 10, 20, 40, 80, 130],
    "S_sparse": [30, 80, 130],
}
# Tilted preference profiles: linear in s (or q), normalized to sum to one (Sec. IV.C text)
EXP6_ALPHA_TILTS = ["short_scale", "long_scale"]
EXP6_BETA_TILTS = ["small_q", "large_q"]
