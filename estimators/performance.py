# estimators/performance.py

import pandas as pd
import numpy as np

def compute_portfolio_performance(weights: pd.Series, returns: pd.DataFrame, freq: str = "daily", risk_free_rate: float = 0.0):
    """
    Compute expected annualized return, volatility, and Sharpe ratio of a portfolio.

    Args:
        weights (pd.Series): Portfolio weights.
        returns (pd.DataFrame): Asset return time series.
        freq (str): "daily" or "monthly" to scale results.
        risk_free_rate (float): Annualized risk-free rate (e.g., 0.02 for 2%).

    Returns:
        dict: Dictionary with keys 'return', 'volatility', 'sharpe'
    """
    # Convert weights to numpy
    w = weights.values

    # Mean and covariance of returns
    mean_ret = returns.mean().values
    cov = returns.cov().values

    # Annualization factor
    ann_factor = 252 if freq == "daily" else 12

    port_ret = np.dot(w, mean_ret) * ann_factor
    port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w))) * np.sqrt(ann_factor)
    sharpe = (port_ret - risk_free_rate) / port_vol if port_vol != 0 else np.nan

    return {
        "return": port_ret,
        "volatility": port_vol,
        "sharpe": sharpe
    }
