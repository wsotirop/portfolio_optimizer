# optimizers/black_litterman.py

import numpy as np
import pandas as pd
from optimizers.mean_variance import mean_variance_optimization


def black_litterman_expected_returns(
    cov_matrix: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    P: pd.DataFrame = None,
    Q: pd.Series = None,
    Omega: pd.DataFrame = None
) -> pd.Series:
    """
    Compute the Black-Litterman posterior expected returns.

    Args:
        cov_matrix: n x n covariance matrix DataFrame
        market_weights: prior market-cap weights Series (n)
        risk_aversion: delta, risk aversion coefficient
        tau: scalar for uncertainty in prior
        P: k x n pick matrix DataFrame for k views
        Q: k-length Series of view returns
        Omega: k x k DataFrame, covariance of view errors
    Returns:
        Series of posterior expected returns (n)
    """
    Σ = cov_matrix.values
    w = market_weights.values.reshape(-1, 1)
    # Implied equilibrium returns (excess)
    π = risk_aversion * Σ.dot(w)

    # If no user views, return implied returns
    if P is None or Q is None:
        return pd.Series(π.flatten(), index=cov_matrix.index)

    P_mat = P.values
    q = Q.values.reshape(-1, 1)
    # Default Omega: diagonal of P tau Σ P'
    if Omega is None:
        temp = P_mat.dot(tau * Σ).dot(P_mat.T)
        Omega = pd.DataFrame(np.diag(np.diag(temp)), index=P.index, columns=P.index)
    Ω = Omega.values

    # Posterior expected returns formula
    tauΣ = tau * Σ
    inv_tauΣ = np.linalg.inv(tauΣ)
    invΩ = np.linalg.inv(Ω)
    M = np.linalg.inv(inv_tauΣ + P_mat.T.dot(invΩ).dot(P_mat))
    mu_bl = M.dot(inv_tauΣ.dot(π) + P_mat.T.dot(invΩ).dot(q))

    return pd.Series(mu_bl.flatten(), index=cov_matrix.index)


def black_litterman_optimization(
    cov_matrix: pd.DataFrame,
    market_weights: pd.Series,
    P: pd.DataFrame,
    Q: pd.Series,
    Omega: pd.DataFrame = None,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    allow_short: bool = False
) -> pd.Series:
    """
    Perform Black-Litterman optimization: compute posterior returns, then apply mean-variance.

    Args:
        cov_matrix: covariance matrix DataFrame
        market_weights: prior market-cap weights Series
        P: pick matrix DataFrame for views
        Q: Series of view returns
        Omega: covariance of view errors
        tau: scalar for uncertainty in prior
        risk_aversion: risk aversion coefficient
        allow_short: allow negative weights
    Returns:
        Series of optimized weights summing to 1
    """
    # Compute posterior expected returns
    mu_bl = black_litterman_expected_returns(
        cov_matrix, market_weights, risk_aversion, tau, P, Q, Omega
    )
    # Use inverse-variance scaling: w* ∝ Σ^{-1} μ_bl
    invΣ = np.linalg.inv(cov_matrix.values)
    raw = invΣ.dot(mu_bl.values)
    if not allow_short:
        raw = np.clip(raw, 0, None)
    w_star = raw / np.sum(raw)
    return pd.Series(w_star, index=cov_matrix.index)
