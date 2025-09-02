# tests/test_betas.py
import numpy as np
import pandas as pd
from utils.helper import compute_asset_betas

def test_compute_asset_betas_basic(rets_df, benchmark_series):
    # Align indices
    idx = rets_df.index.intersection(benchmark_series.index)
    betas = compute_asset_betas(rets_df.loc[idx], benchmark_series.loc[idx])
    assert isinstance(betas, pd.Series)
    assert betas.index.tolist() == rets_df.columns.tolist()
    # Simulated data: expect finite betas
    assert np.isfinite(betas.values).all()
