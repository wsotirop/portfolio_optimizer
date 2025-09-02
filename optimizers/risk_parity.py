# optimizers/risk_parity.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def risk_parity_optimization(
    cov_matrix: pd.DataFrame,
    allow_short: bool = False,
    eps: float = 1e-8
) -> pd.Series:
    """
    Solve a risk parity portfolio using a nonlinear optimizer (SciPy).
    Each asset's contribution to portfolio risk is equalized.

    Args:
        cov_matrix: DataFrame of asset covariances
        allow_short: if False, enforces w >= 0
        eps: small value to keep operations stable

    Returns:
        Series of weights summing to 1.
    """
    Sigma = cov_matrix.values
    n = len(Sigma)

    # Objective: sum squared deviations of risk contributions
    def objective(w):
        # normalize weights
        w = np.array(w)
        port_var = w.dot(Sigma).dot(w)
        # individual risk contributions: w_i * (Sigma @ w)_i
        RC = w * (Sigma.dot(w))
        target = port_var / n
        return np.sum((RC - target) ** 2)

    # Constraints: sum(w) == 1
    cons = ({
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1
    },)

    # Bounds: either (0,1) or unrestricted
    if allow_short:
        bounds = None
    else:
        bounds = tuple((0, 1) for _ in range(n))

    # Initial guess: equal weights
    x0 = np.ones(n) / n

    # Run solver
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'ftol': eps, 'disp': False}
    )

    if not result.success:
        raise ValueError("Risk parity optimization failed: " + result.message)

    return pd.Series(result.x, index=cov_matrix.index)
