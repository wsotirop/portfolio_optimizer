# tests/test_backtest_smoke.py

import numpy as np
from backtest.rolling import run_rolling_backtest

def test_rolling_backtest_smoke(prices_df):
    # keep it tiny: ~6 months lookback, monthly rebal
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
    # Should return series/dataframes with sensible shapes or empties
    assert port is not None and W is not None
