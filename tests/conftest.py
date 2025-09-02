# tests/conftest.py
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIX = ROOT / "tests" / "fixtures"

def _ensure_fixtures():
    # Auto-generate if missing
    if not (FIX / "prices_sim_3y.csv").exists():
        from scripts.make_fixtures import main as make_fx
        make_fx()

_ensure_fixtures()

@pytest.fixture(scope="session")
def prices_df():
    return pd.read_csv(FIX / "prices_sim_3y.csv", index_col=0, parse_dates=True)

@pytest.fixture(scope="session")
def rets_df(prices_df):
    return prices_df.pct_change().dropna()

@pytest.fixture(scope="session")
def benchmark_series():
    s = pd.read_csv(FIX / "benchmark_sim_3y.csv", index_col=0, parse_dates=True).iloc[:,0]
    return s.pct_change().dropna()

@pytest.fixture(scope="session")
def sectors_series(prices_df):
    # deterministic fake sector map: first two "Technology", third "Fixed Income"
    return pd.Series(
        {"AAA": "Technology", "BBB": "Technology", "BND": "Fixed Income"}
    ).reindex(prices_df.columns).fillna("Unknown")
