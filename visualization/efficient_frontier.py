# visualization/efficient_frontier.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from optimizers.mean_variance import mean_variance_optimization
from estimators.returns     import annualized_return
from estimators.risk        import annualized_covariance

def compute_efficient_frontier(
    prices: pd.DataFrame,
    n_points: int = 50,
    allow_short: bool = False,
    **mv_kwargs
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      - 'target_return'
      - 'volatility'
      - plus one column per asset weight
    """
    # 1) Compute inputs
    # pop freq out of mv_kwargs so we don't forward it to mean_variance
    freq = mv_kwargs.pop("freq", "daily")
    rets = prices.pct_change().dropna()
    mu   = annualized_return(prices, freq=freq)
    cov  = annualized_covariance(rets,   freq=freq)   

    # 2) Determine return grid
    min_ret = mu.min()
    max_ret = mu.max()
    targets = np.linspace(min_ret, max_ret, n_points)

    records = []
    for tr in targets:
        w = mean_variance_optimization(
            expected_returns    = mu,
            cov_matrix          = cov,
            target_return       = float(tr),
            allow_short         = allow_short,
            **mv_kwargs        
        )
        vol = np.sqrt(w.values @ cov.values @ w.values)
        rec = {"target_return": tr, "volatility": vol}
        rec.update(w.to_dict())
        records.append(rec)

    return pd.DataFrame(records)

def plot_efficient_frontier(df: pd.DataFrame, ax=None):
    """
    Given the DataFrame from compute_efficient_frontier,
    plots return vs. volatility.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(df["volatility"], df["target_return"], lw=2)
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title("Efficient Frontier")
    return ax

# Example usage (to be run in a script, not on import):
if __name__ == "__main__":
    from data.fetch_data import get_price_data
    tickers = ["AAPL","MSFT","GOOGL","AMZN"]
    prices  = get_price_data(tickers, "2020-01-01", "2024-12-31")
    ef = compute_efficient_frontier(prices, n_points=30, allow_short=False, freq="daily")
    plot_efficient_frontier(ef)
    plt.show()
