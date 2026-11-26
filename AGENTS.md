# AGENTS.md

Guide for coding agents and contributors working on this repository.

## What this repository implements

A reproduction of six experiments from the paper "Portfolio Allocation under
Heterogeneous Scales and Multifractality": multifractal cross-correlation
analysis (MFCCA, MFDCCA, qDCCA), two synthetic dependence-generating
processes (two-component ARFIMA, Markov-switching multifractal), mean-MFCCA
and mean-MFDCCA portfolio optimization, and downside-risk evaluation (VaR,
ES, drawdown) on both synthetic and empirical data. Full results and their
comparison against the paper are in `results/RESULTS.md`.

## Layout

```
src/            flat library modules: mfcca.py, arfima.py, msm.py, portfolio.py, risk.py, stats_tests.py, config.py
exp/            exp1.py .. exp6.py, one script per experiment, calling only src/ and data/
data/           data_loader.py (QuantX Dataloader REST client); no data files are committed
tests/          pytest tests for src/ and for the data-dependent experiments
results/        results saved by the experiment scripts (.csv, .json, .png) plus RESULTS.md
```

No other top-level files or directories are used. `src/`, `exp/` and `data/`
each contain an empty `__init__.py`.

## Commands

```bash
pip install -r requirements.txt
python -m pytest -m "not market_data"     # unit tests, no network required
python -m pytest                           # full suite, requires DATALOADER_API_KEY / MASSIVE_API_KEY
python -m exp.exp1                         # ... through exp.exp6
python -m exp.exp4 --full                  # exp4/exp5/exp6 accept --full to use the paper's unreduced configuration where applicable
./reproduce.sh                             # tests + all six experiments
```

## Conventions

- All experiment parameters live in `src/config.py` as module-level
  UPPER_CASE variables with a comment stating their meaning, unit, and the
  paper section/equation that specifies them (or "not specified in the
  paper"). No classes or parameter dictionaries.
- Market data is loaded only through `data/data_loader.py`
  (`src/` and `exp/` never call a network API directly). No synthetic-data
  generator is needed beyond `src/arfima.py` and `src/msm.py`, which are the
  paper's own stochastic processes (not generic fixtures) and therefore live
  in `src/`, used by `exp1`-`exp3`.
- Tests of `src/` components use small fixtures or synthetic inputs and run
  without network access. Tests that need market data are marked
  `@pytest.mark.market_data`, use a reduced slice cached by `tests/conftest.py`,
  and are skipped automatically when `DATALOADER_API_KEY` is absent.
- API keys are read from the environment via `python-dotenv`
  (`load_dotenv()`), kept only in an uncommitted `.env` file; `.env.example`
  holds empty placeholders.
- Rerunning an experiment script overwrites its files under `results/`. The
  numbers and verdicts written by hand in `results/RESULTS.md` are not
  regenerated automatically — after any rerun, check them against the new
  files in `results/` and update `results/RESULTS.md` accordingly.
- No raw or market-derived time series, and no `.html` plots, are ever
  committed; `results/` holds only aggregate tables and static `.png`/`.pdf`
  figures.
