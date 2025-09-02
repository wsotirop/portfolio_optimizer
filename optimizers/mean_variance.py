# optimizers/mean_variance.py

import numpy as np
import cvxpy as cp
import pandas as pd

def mean_variance_optimization(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    target_return: float = None,
    allow_short: bool = False,
    sector_map: pd.Series = None,
    max_sector_exposure: float = None,
    betas: pd.Series = None,
    beta_neutral: bool = False,
    beta_tolerance: float = None,   
    max_weight: float = None,
    max_leverage: float = None,
    min_weights: pd.Series | None = None, 
) -> pd.Series:
    """
    Mean-variance optimization with optional sector cap, beta neutrality (exact or tolerant),
    per-asset bounds and L1 leverage cap.
    """
    
    # --- Align everything to the same ticker order
    tickers = list(expected_returns.index)
    mu      = expected_returns.loc[tickers].values
    Sigma   = cov_matrix.loc[tickers, tickers].values

    # Small ridge for numerical stability
    Sigma = Sigma + 1e-8 * np.eye(len(tickers))

    # Decision variable
    w = cp.Variable(len(tickers))

    # Objective: minimize variance
    objective = cp.Minimize(cp.quad_form(w, Sigma))

    # Base constraints
    constraints = [cp.sum(w) == 1]
    if not allow_short:
        constraints.append(w >= 0)
    if target_return is not None:
        constraints.append(mu @ w >= target_return)
    
    # Floors (only meaningful when long-only)
    if (min_weights is not None) and (not allow_short):
        floors = min_weights.reindex(tickers).fillna(0.0).values  
        constraints.append(w >= floors)


    # Sector-cap constraint (mask-based: works with CVXPY)
    if sector_map is not None and max_sector_exposure is not None:
        sectors = sector_map.reindex(tickers).fillna("Unknown")
        for sec in sectors.unique():
            if sec == "Unknown":      # <-- don't cap the Unknown bucket (ETFs, etc.)
                continue
            mask = (sectors == sec).astype(float).values
            constraints.append(mask @ w <= max_sector_exposure)

    # Beta neutrality (choose ONE path: exact or tolerant)
    if beta_neutral and (betas is not None):
        b = betas.reindex(tickers).fillna(0.0).astype(float).values
        if beta_tolerance is None:
            constraints.append(b @ w == 0)
        else:
            constraints += [b @ w <= beta_tolerance, -b @ w <= beta_tolerance]

    # Per-asset bounds
    if max_weight is not None:
        constraints.append(w <= max_weight)
        constraints.append(w >= (-max_weight if allow_short else 0))

    # Leverage cap
    if max_leverage is not None:
        constraints.append(cp.norm1(w) <= max_leverage)

    # Solve with robust fallback (try only solvers that are installed)
    prob = cp.Problem(objective, constraints)

    installed = set(cp.installed_solvers())
    candidates = [s for s in ("OSQP", "CLARABEL", "SCIPY", "SCS", "ECOS") if s in installed]

    last_err = None
    w_val = None
    tried = []

    for s in candidates:
        try:
            tried.append(s)
            prob.solve(solver=s, warm_start=True, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None and np.isfinite(w.value).all():
                w_val = w.value
                break
        except Exception as e:
            last_err = repr(e)
    
    # final fallback: let cvxpy pick default
    if w_val is None:
        try:
            prob.solve(warm_start=True, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                w_val = w.value
        except Exception as e:
            last_err = repr(e)

    if w_val is None:
        print(f"[optimizer] status={getattr(prob, 'status', None)}, tried={tried}, installed={list(installed)}, last_err={last_err}")
        return pd.Series([np.nan] * len(tickers), index=tickers)

    return pd.Series(w_val, index=tickers)
