# utils/helper.py

import yfinance as yf
import pandas as pd
import statsmodels.api as sm

def get_sector_map(tickers):
    """Fetches GICS sector for each ticker via yfinance."""
    sectors = {}
    for t in tickers:
        info = yf.Ticker(t).info
        sectors[t] = info.get("sector", "Unknown")
    return pd.Series(sectors)

def get_market_caps(tickers: list) -> pd.Series:
    """
    Fetch market capitalizations from yfinance and normalize to sum to 1.
    Falls back to equal weights if data is missing.
    """
    caps = {}
    for t in tickers:
        info = yf.Ticker(t).info
        caps[t] = info.get("marketCap", 0)
    s = pd.Series(caps)
    if s.sum() > 0:
        return s / s.sum()
    else:
        # fallback to equal-weight
        n = len(tickers)
        return pd.Series(1/n, index=tickers)

def compute_asset_betas(returns: pd.DataFrame, benchmark_ret: pd.Series) -> pd.Series:
    X = sm.add_constant(benchmark_ret.loc[returns.index])
    betas = {}
    for t in returns.columns:
        res = sm.OLS(returns[t], X).fit()
        betas[t] = res.params.iloc[1]
    return pd.Series(betas)

