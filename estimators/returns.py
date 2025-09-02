# estimators/returns.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Tuple

def compute_returns(prices: pd.DataFrame, freq: str = "daily") -> pd.DataFrame:
    """
    Compute simple returns from price data.
    freq: "daily" or "monthly"
    """
    if freq == "daily":
        rets = prices.pct_change().dropna()
    elif freq == "monthly":
        rets = prices.resample("M").last().pct_change().dropna()
    else:
        raise ValueError("freq must be 'daily' or 'monthly'")
    return rets

def annualized_return(rets: pd.DataFrame, freq: str = "daily") -> pd.Series:
    """
    Annualize mean returns.
    """
    rets = compute_returns(rets, freq)
    periods = 252 if freq == "daily" else 12
    mean = rets.mean() * periods
    return mean

def annualized_covariance(rets: pd.DataFrame, freq: str = "daily") -> pd.DataFrame:
    """
    Annualized covariance matrix.
    """
    rets = compute_returns(rets, freq)
    periods = 252 if freq == "daily" else 12
    cov = rets.cov() * periods
    return cov

def factor_expected_returns(
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    risk_free: pd.Series,
    freq: str = "daily"
) -> pd.Series:
    """
    Estimate expected returns via factor model.
    - prices: asset prices
    - factors: DataFrame with columns like ['Mkt-RF','SMB','HML']
    - risk_free: Series of risk-free rates (aligned index)
    Returns: Series of expected annual returns.
    """
    # Compute excess returns
    rets = compute_returns(prices, freq)
    # Align indexes
    ff = factors.loc[rets.index]
    rf = risk_free.loc[rets.index] / (252 if freq=="daily" else 12)
    excess = rets.sub(rf, axis=0)

    # Prepare regression data
    X = sm.add_constant(ff)  # adds intercept
    betas = {}
    for ticker in excess.columns:
        y = excess[ticker]
        model = sm.OLS(y, X).fit()
        betas[ticker] = model.params

    betas_df = pd.DataFrame(betas).T  # rows=tickers, cols=['const','Mkt-RF',...]
    
    # Average factor premia
    avg_f = ff.mean()  # per period
    periods = 252 if freq=="daily" else 12
    annual_premia = avg_f * periods
    
    # Expected excess return = β × factor premia
    exp_ex = betas_df.drop(columns="const").dot(annual_premia)
    # Add back risk-free
    rf_ann = risk_free.mean() * periods
    return exp_ex + rf_ann


if __name__ == "__main__":
    from data.fetch_data import get_price_data, get_fama_french_factors
    # sample tickers & dates
    prices = get_price_data(["AAPL","MSFT","GOOGL"], "2020-01-01", "2024-12-31")
    ff = get_fama_french_factors("2020-01-01", "2024-12-31")
    # Assuming ff has a column 'RF' for risk-free
    rf = ff["RF"]
    fac = ff.drop(columns="RF")
    
    print("Annualized Returns:\n", annualized_return(prices).round(4))
    print("Factor-Based Exp Returns:\n", factor_expected_returns(prices, fac, rf).round(4))
