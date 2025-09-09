# estimators/risk.py

import pandas as pd
import numpy as np

TRADING_DAYS = 252


def _coerce_to_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If df looks like price levels (mostly positive, 'large-ish' values),
    convert to simple returns via pct_change(); otherwise return as-is.

    This is a heuristic meant to catch common mistakes where price data
    is passed to functions expecting returns.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return df

    # Heuristic:
    # - price series are almost always positive (pos_ratio ~ 1)
    # - magnitudes are typically much larger than returns (p90 >> 1)
    pos_ratio = (numeric > 0).to_numpy().mean()
    p90 = numeric.quantile(0.90, numeric_only=True).mean()

    looks_like_price = (pos_ratio > 0.95) and (p90 is not None) and (p90 > 2.0)

    if looks_like_price:
        return numeric.pct_change().dropna()

    return df


def cov_ewma(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """
    EWMA covariance (NOT annualized). Callers may multiply by 252 if needed.

    Accepts returns or (accidentally) prices; prices are converted to returns.
    """
    rets = _coerce_to_returns(returns)
    r = rets.to_numpy()
    # center
    mu = r.mean(axis=0, keepdims=True)
    x = r - mu

    S = np.zeros((x.shape[1], x.shape[1]))
    # recursive EWMA on outer products
    for t in range(x.shape[0] - 1, -1, -1):
        S = lam * S + (1 - lam) * np.outer(x[t], x[t])

    # small-sample normalization so weights sum to ~1
    T = x.shape[0]
    if T > 0:
        S = S / (1 - lam ** T)

    return pd.DataFrame(S, index=rets.columns, columns=rets.columns)


def cov_ledoit_wolf(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Ledoit–Wolf covariance (NOT annualized). Callers may multiply by 252 if needed.

    Accepts returns or (accidentally) prices; prices are converted to returns.
    """
    try:
        from sklearn.covariance import LedoitWolf
    except ImportError as e:
        raise ImportError("pip install scikit-learn for Ledoit–Wolf.") from e

    rets = _coerce_to_returns(returns)
    lw = LedoitWolf().fit(rets.values)
    return pd.DataFrame(lw.covariance_, index=rets.columns, columns=rets.columns)


def annualized_covariance(returns: pd.DataFrame, freq: str = "daily") -> pd.DataFrame:
    """
    Sample covariance, ANNUALIZED.

    Accepts returns or (accidentally) prices; prices are converted to returns.
    """
    rets = _coerce_to_returns(returns)
    periods = TRADING_DAYS if freq == "daily" else 12
    return rets.cov() * periods


def historical_cvar(returns: pd.DataFrame, alpha: float = 0.05, freq: str = "daily") -> pd.Series:
    """
    Historical CVaR per asset at level alpha.

    Accepts returns or (accidentally) prices; prices are converted to returns.

    Returns CVaR as a positive risk number (i.e., magnitude of tail loss),
    annualized by multiplying the mean tail loss by periods.
    """
    rets = _coerce_to_returns(returns)
    periods = TRADING_DAYS if freq == "daily" else 12

    # VaR per asset at alpha
    var = rets.quantile(alpha)

    # Mean of returns <= VaR (i.e., left tail)
    tail_means = rets[rets.le(var, axis=1)].mean()

    # Return positive risk
    return -tail_means * periods


if __name__ == "__main__":
    # Quick local check (optional)
    from data.fetch_data import get_price_data

    prices = get_price_data(["AAPL", "MSFT", "GOOGL"], "2020-01-01", "2024-12-31")
    # Intentionally pass PRICES to functions to confirm auto-coercion
    cov_ann = annualized_covariance(prices, freq="daily")
    cvar_95 = historical_cvar(prices, alpha=0.05, freq="daily")

    print("Annualized Covariance:\n", cov_ann.round(4))
    print("95% CVaR:\n", cvar_95.round(4))
