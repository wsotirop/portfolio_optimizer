# optimizers/cvar.py

import cvxpy as cp
import pandas as pd
import numpy as np

def cvar_optimization(
    returns: pd.DataFrame,
    alpha: float = 0.95,
    target_return: float | None = None,
    allow_short: bool = False,
    max_weight: float | None = None,
    max_leverage: float | None = None,
) -> pd.Series:
    R = returns.values        # T x N (daily)
    T, N = R.shape
    w = cp.Variable(N)
    t = cp.Variable()
    z = cp.Variable(T)

    cons = [cp.sum(w) == 1, z >= 0, z >= -(R @ w) - t]
    if not allow_short:
        cons += [w >= 0]
    if max_weight is not None:
        cons += [w <= max_weight, w >= (-max_weight if allow_short else 0)]
    if max_leverage is not None:
        cons += [cp.norm1(w) <= max_leverage]
    if target_return is not None:
        mu_ann = returns.mean().values * 252
        cons += [mu_ann @ w >= target_return]

    obj = cp.Minimize(t + (1.0/((1 - alpha) * T)) * cp.sum(z))
    prob = cp.Problem(obj, cons)
    prob.solve(solver="SCS", verbose=False)
    if w.value is None:
        return pd.Series([np.nan]*N, index=returns.columns)
    return pd.Series(w.value, index=returns.columns)
