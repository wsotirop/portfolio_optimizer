# tests/test_backtest_smoke.py
import numpy as np
import pandas as pd
from backtest.rolling import run_rolling_backtest

def test_rolling_backtest_smoke(prices_df):
    port, W, to_ser, cost_ser = run_rolling_backtest(
        prices=prices_df,
        rebalance="Monthly",
        lookback=126,
        model="Minimum-Variance",
        allow_short=False,
        max_weight=1.0,
        cov_model="Sample",
        rf_annual=0.0,
        benchmark_rets=None,
        tc_bps=0,
    )

    # Basic shapes
    assert isinstance(port, pd.Series)
    assert isinstance(W, pd.DataFrame)
    assert not W.empty

    # Weights sum to 1 (within tolerance) at each rebalance
    row_sums = W.sum(axis=1)
    assert np.allclose(row_sums.values, 1.0, atol=1e-6)

    # Portfolio returns span at least as long as last rebalance→end
    assert len(port) > 0
    assert port.index.is_monotonic_increasing
    # Costs/turnover aligned with W’s index
    assert set(to_ser.index) == set(W.index)
    assert set(cost_ser.index) == set(W.index)
