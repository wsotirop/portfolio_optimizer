# tests/test_sector_cap.py

import numpy as np
from optimizers.mean_variance import mean_variance_optimization
from estimators.returns import annualized_return
from estimators.risk import annualized_covariance

def test_sector_cap_feasible(prices_df, sectors_series):
    mu  = annualized_return(prices_df, freq="daily")
    cov = annualized_covariance(prices_df.pct_change().dropna(), freq="daily")
    w = mean_variance_optimization(
        expected_returns=mu,
        cov_matrix=cov,
        allow_short=False,
        sector_map=sectors_series,
        max_sector_exposure=0.7,  # generous cap
        max_weight=1.0
    )
    assert np.isfinite(w.values).all()
    # check sector sums respect cap
    exp_by_sec = w.groupby(sectors_series).sum()
    assert (exp_by_sec["Technology"] <= 0.7 + 1e-6) if "Technology" in exp_by_sec.index else True
    assert abs(float(w.sum()) - 1.0) < 1e-6
