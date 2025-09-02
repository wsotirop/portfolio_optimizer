# estimators/risk.py

import pandas as pd
import numpy as np

def cov_ewma(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    r = returns.to_numpy()
    mu = r.mean(axis=0, keepdims=True)
    x = r - mu
    S = np.zeros((x.shape[1], x.shape[1]))
    w = 1.0
    for t in range(x.shape[0]-1, -1, -1):
        S = lam*S + (1-lam)*np.outer(x[t], x[t])
    S = S / (1 - lam**x.shape[0]) 
    return pd.DataFrame(S, index=returns.columns, columns=returns.columns)

def cov_ledoit_wolf(returns: pd.DataFrame) -> pd.DataFrame:
    try:
        from sklearn.covariance import LedoitWolf
    except ImportError:
        raise ImportError("pip install scikit-learn for Ledoit–Wolf.")
    lw = LedoitWolf().fit(returns.values)
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)

def annualized_covariance(returns: pd.DataFrame, freq: str = "daily") -> pd.DataFrame:
    """
    Annualized covariance matrix of asset returns.
    `returns` should be simple returns, not prices.
    """
    # If prices were passed by mistake, compute returns
    if returns.columns.difference(returns.columns).any():  # cheap check
        returns = returns.pct_change().dropna()  # fallback
    periods = 252 if freq == "daily" else 12
    return returns.cov() * periods

def historical_cvar(returns: pd.DataFrame, alpha: float = 0.05, freq: str = "daily") -> pd.Series:
    """
    Compute portfolio CVaR at level alpha for each individual asset.
    - returns: simple returns DataFrame (aligned)
    - alpha: tail probability (e.g., 0.05 for 95% CVaR)
    Returns: Series of negative expected losses beyond VaR.
    """
    # Ensure simple returns
    rets = returns.copy()
    # If these look like prices, convert:
    if (rets > 1).all().all():
        rets = rets.pct_change().dropna()
    
    # VaR per asset
    var = rets.quantile(alpha)
    # CVaR = average return below the VaR quantile
    cvar = rets[rets.le(var, axis=1)].mean()
    # CVaR expressed as a positive risk
    return -cvar * (252 if freq=="daily" else 12)

if __name__ == "__main__":
    from data.fetch_data import get_price_data
    from estimators.returns import compute_returns

    prices = get_price_data(["AAPL","MSFT","GOOGL"], "2020-01-01", "2024-12-31")
    rets = compute_returns(prices)

    cov = annualized_covariance(rets)
    cvar = historical_cvar(rets, alpha=0.05)

    print("Annualized Covariance:\n", cov.round(4))
    print("95% CVaR:\n", cvar.round(4))
