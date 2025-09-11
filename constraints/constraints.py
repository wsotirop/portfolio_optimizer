# constraints/constraints.py

import cvxpy as cp
import pandas as pd

def weight_bounds(w, lower: float = 0.0, upper: float = 1.0):
    """
    Elementwise lower/upper bound constraints on weights.

    Returns
    -------
    list[cp.Constraint]
        [w >= lower, w <= upper]
    """
    return [w >= lower, w <= upper]

def sector_neutral_constraints(w, sector_map: pd.Series, max_exposure: float):
    """
    Enforce that the sum of weights for each sector <= max_exposure.
    sector_map: pandas Series mapping tickers (in order) -> sector labels
                Its index order must match the ordering used when defining `w`.
    """
    constraints = []
    # Build a list of positions for each sector
    for sector in sector_map.unique():
        mask = (sector_map == sector).values
        # get integer positions where mask is True
        positions = [i for i, flag in enumerate(mask) if flag]
        if positions:
            # sum the corresponding entries of w
            constraints.append(cp.sum(cp.vstack([w[i] for i in positions])) <= max_exposure)
    return constraints

def beta_neutral_constraints(w, betas: pd.Series, tol: float = 1e-3):
    """
    Enforce portfolio beta ≈ 0 (or within ±tol).
    betas: Series mapping tickers -> asset betas vs. a benchmark
    """
    beta_expr = betas.values @ w
    if tol > 0:
        return [beta_expr<= tol, beta_expr >= -tol]
    else:
        return [beta_expr == 0]
    