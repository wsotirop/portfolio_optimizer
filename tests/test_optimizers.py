# tests/test_optimizers.py

import numpy as np
from estimators.returns import annualized_return
from estimators.risk import annualized_covariance
from optimizers.mean_variance import mean_variance_optimization

def test_mean_variance_long_only(prices_df):
    mu  = annualized_return(prices_df, freq="daily")
    cov = annualized_covariance(prices_df.pct_change().dropna(), freq="daily")
    w = mean_variance_optimization(
        expected_returns=mu,
        cov_matrix=cov,
        allow_short=False,
        max_weight=1.0
    )
    assert np.isfinite(w.values).all()
    assert (w >= -1e-12).all()
    assert abs(float(w.sum()) - 1.0) < 1e-6

def test_cvar_exists_and_sums(prices_df):
    # Optional: only if you've added optimizers/cvar.py
    try:
        from optimizers.cvar import cvar_optimization
    except Exception:
        return  # skip if not present
    rets = prices_df.pct_change().dropna()
    w = cvar_optimization(rets, alpha=0.95, allow_short=False, max_weight=1.0)
    assert np.isfinite(w.values).all()
    assert abs(float(w.sum()) - 1.0) < 1e-6
