# optimizers/minimum_variance.py

import pandas as pd
from optimizers.mean_variance import mean_variance_optimization


def minimum_variance_optimization(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    allow_short: bool = False
) -> pd.Series:
    """
    Compute the global minimum-variance portfolio by calling the mean-variance optimizer
    with no target return.

    Args:
        expected_returns: Series of asset expected returns (not used for weights calculation)
        cov_matrix: DataFrame of asset covariances
        allow_short: if False, enforces w >= 0

    Returns:
        Series of weights summing to 1.
    """
    # Delegate to mean_variance_optimization with target_return=None
    return mean_variance_optimization(
        expected_returns,
        cov_matrix,
        target_return=None,
        allow_short=allow_short
    )
