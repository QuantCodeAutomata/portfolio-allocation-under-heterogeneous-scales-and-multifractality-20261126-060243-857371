# Portfolio Allocation under Heterogeneous Scales and Multifractality

Reproduction of six experiments from the paper **"Portfolio Allocation under
Heterogeneous Scales and Multifractality"**, covering multifractal
cross-correlation analysis (MFCCA/MFDCCA, qDCCA), two synthetic
dependence-generating processes (two-component ARFIMA and the
Markov-switching multifractal model), and mean-MFCCA/mean-MFDCCA portfolio
optimization on both synthetic and empirical data.

See `results/RESULTS.md` for the full comparison against the paper's reported
values, including every deviation and every result not reproduced.

## Layout

```
src/            core library: MFCCA/MFDCCA, ARFIMA, MSM, portfolio QP, risk measures, statistical tests, config
exp/            one script per experiment (exp1 .. exp6)
data/           data_loader.py: OHLCV bars through the QuantX Dataloader REST API
tests/          pytest unit tests for src/ and experiment-facing checks
results/        tables (.csv/.json) and static figures (.png) saved by the experiment scripts, plus RESULTS.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY (needed only for exp_4, exp_5, exp_6)
```

## Tests

```bash
python -m pytest -m "not market_data"      # unit tests, no network access required
python -m pytest                            # full suite, including market-data tests (requires the API keys above)
```

## Running the experiments

Each experiment is a standalone module run from the repository root; every
run overwrites its files under `results/`.

```bash
python -m exp.exp1   # two-component ARFIMA: qDCCA scan and MMFC/MMFD efficient frontiers across scales
python -m exp.exp2   # Markov-switching multifractal: qDCCA scan and MMFC/MMFD efficient frontiers across fluctuation orders
python -m exp.exp3   # 50-realization Monte Carlo: MV/MD/MMFD/MC/MMFC downside-risk comparison (VaR, ES)
python -m exp.exp4   # empirical scaling and cross-correlation properties of four financial return series
python -m exp.exp5   # empirical rolling-window in-sample / out-of-sample portfolio performance
python -m exp.exp6   # robustness of MMFC to the scale set, fluctuation-order set, and preference weights
```

`exp4`, `exp5` and `exp6` load daily market data through the QuantX
Dataloader REST API (`data/data_loader.py`) and require `DATALOADER_API_KEY`
and `MASSIVE_API_KEY` in the environment (or in `.env`). `exp1`, `exp2` and
`exp3` use only synthetic data generated in-process (`src/arfima.py`,
`src/msm.py`) and need no network access.

`reproduce.sh` runs the unit tests and all six experiments in sequence.

## Data

Daily log returns for four financial assets — proxies for Nikkei, S&P 500,
WTI crude oil, and XAU/USD gold, as detailed in `results/RESULTS.md` — are
loaded from January 2011 to November 2023 through the QuantX Dataloader
REST API, which serves OHLCV bars built on Massive market data with a Yahoo
Finance backstop. No market data or synthetic series are stored in this
repository; `data/data_loader.py` downloads and caches bars outside the
repository on each run.

## Paper-derived parameters

Every numeric parameter used by the experiments (ARFIMA and MSM process
parameters, scale and fluctuation-order sets, rolling-window sizes, VaR/ES
confidence levels, etc.) is a module-level constant in `src/config.py`, each
commented with the paper section or equation that specifies it, or marked
"not specified in the paper" when the paper leaves it open.
