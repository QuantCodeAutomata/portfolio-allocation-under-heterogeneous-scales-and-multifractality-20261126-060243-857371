"""Shared pytest fixtures and marker registration."""
from __future__ import annotations

import os

import numpy as np
import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: tests that load real data via the QuantX Dataloader"
    )


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def empirical_slice(pytestconfig: pytest.Config):
    """A reduced real-data slice: two tickers, one year of daily log returns.

    Cached in pytest's cache directory so repeated test runs do not
    re-download the data. Skipped when DATALOADER_API_KEY is absent.
    """
    if not os.environ.get("DATALOADER_API_KEY"):
        pytest.skip("DATALOADER_API_KEY not set: market-data tests are skipped")
    cache_key = "qca/empirical_slice_v1"
    cached = pytestconfig.cache.get(cache_key, None)
    if cached is not None:
        import pandas as pd

        df = pd.DataFrame(cached["data"])
        df.index = pd.to_datetime(df.pop("__date__"))
        return df

    from data.data_loader import load_log_returns

    tickers = {"SP500": "SPY", "XAU": "C:XAUUSD"}
    log_returns, meta = load_log_returns(
        tickers, "2019-01-02", "2019-12-31", use_cache=True
    )
    to_store = log_returns.reset_index().rename(columns={"date": "__date__"})
    to_store["__date__"] = to_store["__date__"].astype(str)
    pytestconfig.cache.set(cache_key, {"data": to_store.to_dict(orient="list")})
    return log_returns
