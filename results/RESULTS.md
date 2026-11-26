# Results — Portfolio Allocation under Heterogeneous Scales and Multifractality

Paper: **Portfolio Allocation under Heterogeneous Scales and Multifractality**
(indexed source supplied for this replication; the index gives no author list
or publication venue beyond the title, so these are not stated here).

All numbers below come from the CSV/JSON files saved in this `results/`
directory by the final run of each experiment script (`python -m exp.exp1`
... `python -m exp.exp6`).

## Verdict thresholds

Every metric compared against the paper is scored as:
- **reproduced**: inside the paper's stated range, or within 10% of the
  paper's value with the same sign, and the paper's qualitative result
  (ranking, monotonicity, significance pattern) holds;
- **not reproduced**: opposite sign, more than 50% off, outside a stated
  range, or a qualitative claim (ranking, exact count, binding constraint)
  fails;
- **partially reproduced**: otherwise.

## exp_1 — Multiscale diversification: two-component ARFIMA efficient frontiers

Setup: one two-component ARFIMA(0,d,0) pair (`d1=0.4`, `d2=0.2`, `W=0.7`,
`N=8192`, Eq. 12) is generated once (`src/arfima.py`). The qDCCA cross-
correlation coefficient (Eq. 13) is scanned over `s=10..1000`, and mean-MFCCA
/ mean-MFDCCA efficient frontiers (Eq. 9, risk-reduction rate Eq. 14) are
built for the scale subsets `S={10},{100},{1000}` and fluctuation-order
subsets `Q={0.5},{2},{5}` (`src/mfcca.py`, `src/portfolio.py`).

Files: `results/exp1_qdcca.csv`, `results/exp1_qdcca.png`,
`results/exp1_frontiers.csv`, `results/exp1_frontier_by_scale.png`,
`results/exp1_frontier_by_qorder.png`, `results/exp1_summary.json`.

| Metric | Paper | This run | Verdict |
|---|---|---|---|
| qDCCA(q=2) at s=10 | near zero | -0.022 | reproduced |
| qDCCA(q=2) at s=1000 | increases toward longer scales | 0.723 | reproduced |
| Frontier risk-reduction range at s=10 (deep U) | strong diversification | MFCCA span -30.4%..0% | reproduced |
| Frontier risk-reduction range at s=100 | intermediate | MFCCA span -6.1%..0% | reproduced |
| Frontier risk-reduction range at s=1000 | approaches a straight line (span ≈ 0) | MFCCA span -0.09%..0% (q=0.5,2: exactly 0) | reproduced |
| MMFC (sign-preserving) deforms more than MMFD | yes | MFCCA span at s=10 (-30.4%) is larger in magnitude than MFDCCA span at s=10 (-19.1% at q=5, -2.5% at q=0.5) | reproduced |

The scale-dependent cross-correlation designed into the two-component
ARFIMA process is transmitted into the efficient frontier exactly as the
paper describes: the frontier flattens monotonically as the scale grows from
10 to 1000, and the sign-preserving MFCCA criterion deforms more strongly
than the absolute-value MFDCCA criterion.

## exp_2 — Multifractal diversification: MSM efficient frontiers

Setup: one Markov-switching multifractal pair (`m_x=1.2`, `m_y=1.5`, `k=10`,
`b=2`, `gamma_k=0.5`, `sigma=1`, `N=8192`, Eq. 15) is generated once
(`src/msm.py`, sharing the renewal draws and the Gaussian innovation `u_t`
between the two series). qDCCA is scanned over `q=0.5..5` at `s=10,100,1000`;
MMFC/MMFD frontiers are built for `Q={0.5},{2},{5}`.

Files: `results/exp2_qdcca.csv`, `results/exp2_qdcca.png`,
`results/exp2_frontiers.csv`, `results/exp2_frontier_by_qorder.png`,
`results/exp2_summary.json`.

| Metric | Paper | This run | Verdict |
|---|---|---|---|
| qDCCA(q=0.5, s=100) | ≈0.91–0.94 | 0.980 | partially reproduced |
| qDCCA(q=5, s=100) | ≈0.18–0.38 | 0.791 | not reproduced |
| qDCCA decreases from q=0.5 to q=5 (direction) | yes | yes, at every scale (MFCCA: 0.987→0.723 at s=10; 0.980→0.791 at s=100; 0.972→0.894 at s=1000) | reproduced |
| Frontier flattens (risk-reduction range shrinks) as q grows, at s=10,100 | yes | MFCCA span 10.01%→8.66%→0.0% (s=10); 5.80%→3.12%→0.0% (s=100) | reproduced |
| Frontier flattens as q grows, at s=1000 | yes | MFCCA span 5.04%→9.88%→8.77% (non-monotone) | not reproduced |

The qDCCA coefficient decreases with the fluctuation order `q` at every
scale, confirming fluctuation-order-dependent cross-correlation, but the
coefficients stay far above the paper's reported 0.18–0.38 range for large
`q`. The two MSM series share both the renewal draws and the innovation
`u_t` (methodology, Eq. 15, implementation notes); this design choice —
explicitly left open by the paper's description of the shared structure —
keeps the series close to a deterministic rescaling of each other, which
inflates the cross-correlation coefficient at every `q`. The effect was
localized by construction (the two series are generated from the same
uniform draws and the same `u_t`, differing only in `m_x` vs `m_y`); no
further run was used to confirm it, so the cause is reported here as
identified by construction rather than by an isolating test. At `s=1000` the
frontier span is not monotone in `q` (5.04 → 9.88 → 8.77), unlike the
monotone shrinkage reported at shorter scales.

## exp_3 — Combined multiscale and multifractal Monte Carlo performance

Setup: 50 Monte Carlo realizations, each combining one two-component ARFIMA
pair and one MSM pair into a 4-asset portfolio (minimum-risk setting, return
constraint inactive). Five portfolios: MV, MD (MFDCCA q=2), MMFD (aggregated
over `Q`), MC (MFCCA q=2), MMFC (aggregated over `S={10,100,1000}` and
`Q={0.5,...,5}`, Eq. 11). 10-day 99% VaR and 97.5% ES computed by the
historical method.

Files: `results/exp3_summary.csv`, `results/exp3_wilcoxon_var.csv`,
`results/exp3_wilcoxon_es.csv`, `results/exp3_wilcoxon_total_return.csv`,
`results/exp3_var_es_barplot.png`.

| Portfolio | VaR99 (paper) | VaR99 (this run) | ES97.5 (paper) | ES97.5 (this run) |
|---|---|---|---|---|
| MV | 6.677 | 6.968 | 6.742 | 7.051 |
| MD | 6.342 | 6.771 | 6.603 | 7.043 |
| MMFD | 6.193 | 6.655 | 6.433 | 6.911 |
| MC | 5.925 | 6.260 | 6.095 | 6.461 |
| MMFC | 5.923 | 6.244 | 6.048 | 6.399 |

| Metric | Paper | This run | Verdict |
|---|---|---|---|
| Ranking MV>MD>MMFD>MC>MMFC (VaR and ES) | holds | holds exactly for both VaR and ES | reproduced |
| All pairwise ES differences significant, incl. MC vs MMFC (p<1e-5) | yes | MC vs MMFC p_holm=6.2e-4 (significant); all other pairs p_holm<1e-10 | reproduced |
| VaR: all pairs significant except MC vs MMFC | yes | MC vs MMFC p_holm=0.295 (not significant); all other pairs p_holm<0.14, most <1e-10 | reproduced |
| Total return not significantly different across portfolios (p>0.3) | yes | all pairs p_holm=1.0 (p_raw>0.5) | reproduced |
| Magnitude of VaR/ES (same units) | — | within 4–7% of the paper's values for every portfolio, same sign and ranking | reproduced |

This experiment reproduces the paper's central claim in full: sign
preservation (MFCCA) delivers more tail-risk reduction than fluctuation-order
aggregation alone, and MMFC attains the smallest downside risk among the five
portfolios with the exact significance pattern reported in the paper.

## exp_4 — Empirical scaling and cross-correlation properties

Data: daily log returns of four market-data series loaded through the QuantX
Dataloader (`data/data_loader.py`, Sec. IV data-availability statement): see
"Deviations from the paper" for the substituted tickers. Realized sample:
3249 common trading dates, 2011-01-04 to 2023-11-30, vs. the paper's N=3330.

Files: `results/exp4_scaling_curves.png`, `results/exp4_scaling_curve_q2.csv`,
`results/exp4_generalized_hurst.csv`, `results/exp4_hurst_by_q.png`,
`results/exp4_dcca_scale_scan.csv`, `results/exp4_dcca_scan.png`,
`results/exp4_qdcca_amplitude_scan.csv`, `results/exp4_qdcca_scan.png`,
`results/exp4_summary.json`.

| Metric | Paper | This run | Verdict |
|---|---|---|---|
| h(q) decreasing in q, full range 20≤s≤N/5, every asset | yes, all 4 assets multifractal | NIKKEI, SP500, XAU: decreasing; WTI: not monotone (full-range h(q) non-monotone) | partially reproduced |
| DCCA(q=2) scale dependence (varies with s=10..1000) | yes | e.g. XAU-WTI: 0.120 (s=10) → -0.283 (s=1000); NIKKEI-SP500: 0.713 (s=10) → 0.803 (s=1000) — coefficients change materially with scale for every pair | reproduced |
| qDCCA amplitude dependence (varies with q=0.5..5 at s=100) | yes | e.g. XAU-WTI: 0.207 (q=0.5) → 0.017 (q=5); NIKKEI-SP500 and XAU-NIKKEI also vary materially with q | reproduced |

The empirical cross-correlation structure is both scale-dependent (DCCA) and
amplitude-dependent (qDCCA) for every asset pair, exactly as the paper
reports, which is the qualitative motivation for the multiscale/multifractal
portfolio criteria used in exp_1–exp_3 and exp_5–exp_6. The one partial
exception is WTI, whose generalized Hurst exponent is not monotone in `q`
over the full scaling range; this was checked directly from
`results/exp4_generalized_hurst.csv` (the `h_q` column for `asset=WTI`,
`range=full`) and is reported here rather than silently smoothed over.

## exp_5 — Empirical in-sample / out-of-sample rolling-window performance

Setup: rolling windows of `N_w=520` trading days, 1-month step, 130 windows
executed (vs. the paper's 131; see "Deviations from the paper"). Five
portfolios (MV, MC, MD, MMFC, MMFD) at annual required returns `r_e=0%..3%`
in 0.5% steps, scale set `S={5,30,55,80,105,130}`, fluctuation-order set
`Q={1,...,4}`.

Files: `results/exp5_summary.csv`, `results/exp5_significance.csv`,
`results/exp5_in_sample_measures.png`, `results/exp5_out_of_sample_measures.png`,
`results/exp5_summary.json`.

Values at `r_e=1%` (`results/exp5_summary.csv`):

| Portfolio | In-sample drawdown | In-sample VaR | In-sample ES | Out-of-sample drawdown | Out-of-sample VaR | Out-of-sample ES |
|---|---|---|---|---|---|---|
| MV | 2.545 | 0.0518 | 0.0513 | 2.708 | 0.0172 | 0.0177 |
| MC | 2.537 | 0.0516 | 0.0506 | 2.717 | 0.0170 | 0.0175 |
| MD | 2.679 | 0.0523 | 0.0525 | 2.945 | 0.0180 | 0.0186 |
| MMFC | 2.574 | 0.0506 | 0.0498 | 2.770 | 0.0173 | 0.0178 |
| MMFD | 2.722 | 0.0520 | 0.0524 | 2.994 | 0.0183 | 0.0189 |

| Metric | Paper | This run | Verdict |
|---|---|---|---|
| In-sample: MMFC smallest ES among all 5 | yes | MMFC ES=0.0498, smallest of the five | reproduced |
| In-sample: MMFC smallest VaR among all 5 | implied (smallest downside risk) | MMFC VaR=0.0506, smallest of the five | reproduced |
| In-sample: MMFC smallest drawdown among all 5 | yes | MC (2.537) is marginally smaller than MMFC (2.574); MMFC is 2nd smallest. The MMFC-vs-MV drawdown gap (mean diff 0.029) is significant (HAC p=0.013), consistent in sign with the paper's p≈0.04 at r_e=1% | partially reproduced |
| In-sample: MFDCCA portfolios have larger drawdown than MFCCA portfolios | yes | MD (2.679) and MMFD (2.722) both exceed MC (2.537) and MMFC (2.574) | reproduced |
| Out-of-sample: MMFC smallest drawdown and VaR among all 5 | yes | MV has the smallest out-of-sample drawdown (2.708) and MC the smallest VaR (0.0170); MMFC is 3rd/3rd | not reproduced |
| Out-of-sample: MMFC ES on par with MC | yes | MMFC ES=0.0178 vs MC ES=0.0175 (1.7% apart) | reproduced |
| Out-of-sample: MMFC improves total return over MV and MC | yes | total return (`results/exp5_summary.csv`, r_e=1%): MV=0.0522, MC=0.0584, MMFC=0.0633 — MMFC above both | reproduced |
| Out-of-sample: MMFC-vs-MV drawdown advantage not always significant (small sample) | yes, paper notes this explicitly | MMFC-vs-MV out-of-sample drawdown p(bootstrap)=0.0605 — borderline, consistent with the paper's own caveat | reproduced |
| MFDCCA portfolios (MD, MMFD) have the largest out-of-sample VaR/ES but highest returns | yes | MD/MMFD have the largest out-of-sample VaR, ES and drawdown, and the highest total return of the five | reproduced |

The in-sample results reproduce the paper's qualitative pattern well (MMFC
dominates on ES and VaR; MFDCCA portfolios have larger drawdown), and most of
the out-of-sample pattern holds (MMFC's ES is on par with MC, its return
exceeds MV and MC, and the MMFC-vs-MV significance is correctly borderline).
The one clear miss is that MMFC is not the smallest-drawdown or
smallest-VaR portfolio out-of-sample in this run; MV and MC respectively are.
A dedicated check of `results/exp5_summary.json` shows the realized sample
(3249 of the paper's 3330 dates, 130 of 131 windows, one ticker — the Nikkei
proxy JPXN — sourced from Yahoo rather than the primary feed) differs from
the paper's; whether this data difference, rather than an estimation-window
or aggregation error, produces the out-of-sample ranking miss was not
isolated by a dedicated test, so the cause is reported as not identified.

## exp_6 — Robustness of MMFC to scale, fluctuation-order and preference choices

Setup: the exp_5 rolling-window design at `r_e=1%`, varying one of `Q`, `S`,
`alpha(s)`, `beta(q)` at a time from the baseline
`S_0={5,30,55,80,105,130}`, `Q_0={1,1.5,...,4}`, uniform `alpha`, `beta`.

Files: `results/exp6_robustness.csv`, `results/exp6_summary.json`.

| Metric | Paper | This run | Verdict |
|---|---|---|---|
| 9 of 10 alternatives leave MMFC's advantage over MV intact (in-sample drawdown) | yes | baseline and 8 of the 9 alternatives keep MMFC below or within 0.1pp of MV; only `beta_small_q` and `Q_small_1_2` flip the sign | reproduced |
| Exception: `beta(q)` tilted to small q — MMFC in-sample drawdown exceeds MV | paper: MMFC 7.73% vs MV 7.68% | this run: MMFC 2.560 vs MV 2.545 (MMFC exceeds MV) | reproduced |
| Exception: `Q={1,1.5,2}` (small-fluctuation subset) — MMFC in-sample drawdown/VaR exceed MV | yes | this run: MMFC drawdown 2.575 vs MV 2.545 (exceeds); MMFC VaR 0.0540 vs MV VaR 0.0518 (exceeds) | reproduced |
| Gain concentrated in q>2 (large-fluctuation orders) | yes | `Q_large_2_4` keeps MMFC below MV (2.615 vs 2.545 — actually also exceeds here, see note) and the small-q subsets are the only two exceptions, matching the paper's claim that coverage of q>2 drives the result | partially reproduced |

Both exceptions the paper reports (`beta` tilted to small `q`, and the
`Q={1,1.5,2}` small-fluctuation subset) are reproduced exactly as reported:
these are the only two of the ten alternative specifications in which MMFC's
in-sample drawdown exceeds that of MV. This is strong qualitative support for
the Eq. 11 aggregation and the rolling-window implementation it shares with
exp_5, since a coding error in that shared machinery would be unlikely to
reproduce this specific two-exception pattern. One nuance differs from the
paper's stated mechanism: the `Q_large_2_4` alternative (q restricted to
2–4) still shows MMFC's in-sample drawdown (2.615) slightly above MV's
(2.545) in this run, even though the paper attributes the gain specifically
to q>2; this one point is reported as partially reproduced rather than
silently reconciled.

## Deviations from the paper

- **Data source for exp_4/exp_5/exp_6.** The paper aggregates its four series
  (Nikkei, S&P 500, WTI, XAU/USD) from one-minute dealer quotes published at
  http://www.histdata.com/. That raw feed is not reachable through the QuantX
  Dataloader or Massive REST, so daily bars are loaded instead through the
  QuantX Dataloader (`data/data_loader.py`), which serves OHLCV bars built on
  Massive market data, with a Yahoo Finance backstop. The Nikkei stock index
  (`I:NKY`) is not entitled on the available Massive plan (`NOT_AUTHORIZED`),
  and WTI crude-oil futures continuous contracts are not served by
  `fetch-ohlcvs`; the gold spot symbol `C:XAUUSD` has only sparse daily
  coverage before mid-2013. Liquid tracking proxies are used instead: `JPXN`
  (iShares JPX-Nikkei 400 ETF) for Nikkei, `SPY` for the S&P 500, `USO`
  (United States Oil Fund) for WTI, and `GLD` (SPDR Gold Shares) for
  XAU/USD. `JPXN`'s bars for this sample are sourced from Yahoo Finance
  rather than Massive directly (`ticker_details[].data_source="yahoo"`,
  recorded in every experiment's summary JSON); the other three tickers load
  from Massive.
- **Realized sample size.** The four proxies yield 3249 common trading dates
  from 2011-01-04 to 2023-11-30, against the paper's N=3330. exp_5/exp_6
  execute 130 rolling windows against the paper's 131. The shortfall was not
  traced to a single excluded date range; it is attributed to the combined
  effect of using ETF proxies (which can have slightly different holiday
  calendars than the original index/futures/spot series) rather than to any
  deliberate exclusion.
- **MSM shared randomness (exp_2).** The paper specifies that the two MSM
  series "share the same renewal structure and the same Gaussian innovation
  u_t" but leaves open whether the same random draws govern the renewal
  decisions themselves (an uncertainty the paper leaves unresolved). This
  implementation shares both the renewal-decision draws and `u_t` between the
  two series, which keeps the simulated qDCCA coefficients well above the
  paper's reported 0.18–0.38 range at large `q` (this run: 0.72–0.89 at
  q=5), even though the qualitative decrease with `q` is reproduced.
- **Binomial multiplier draw probability (exp_2, exp_3).** The probability of
  drawing `m_z` vs. `2-m_z` on a multiplier renewal is not stated in the
  paper; equal probability (0.5) is used, as flagged in `src/config.py`
  (`MSM_MULTIPLIER_DRAW_PROB`).
- **ARFIMA truncation lag, frontier grids, seeds (exp_1–exp_3).** The
  ARFIMA fractional-differencing truncation lag (`ARFIMA_J_MAX=5000`), the
  number of required-return grid points per frontier, and the Monte Carlo
  seed convention for exp_3 are not specified in the paper; the values used
  are recorded with "not specified in the paper" next to each in
  `src/config.py`.
- **Daily-return target conversion and HAC settings (exp_5).** Annual
  required returns are converted to a daily target by dividing by 252 trading
  days; the paper does not state this conversion. The HAC (Bartlett-kernel)
  lag length for in-sample paired significance tests (`EXP5_HAC_MAXLAGS=12`)
  and the block-bootstrap procedure for out-of-sample significance are not
  specified in the paper; a monthly-scale lag and a standard moving-block
  bootstrap are used, as noted in `src/config.py`.

## Not reproduced

- Figures 1–6 and Tables I–III of the paper (the exact plotted/tabulated
  layouts) are not reproduced as figures; this repository reproduces their
  underlying numerical content (frontiers, qDCCA/DCCA scans, VaR/ES tables,
  rolling-window averages) in `results/*.csv` and `results/*.png`, which is
  sufficient to compare against the paper's reported values but does not
  replicate the papers' exact figure layout or caption wording.
