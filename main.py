# main.py

import argparse
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm

from data.fetch_data import get_price_data, get_fama_french_factors
from estimators.returns import annualized_return
from estimators.risk import annualized_covariance
from optimizers.mean_variance import mean_variance_optimization
from optimizers.risk_parity import risk_parity_optimization
from estimators.risk import annualized_covariance
from optimizers.minimum_variance import minimum_variance_optimization
from optimizers.black_litterman import black_litterman_optimization
from constraints.constraints import (weight_bounds,sector_neutral_constraints,beta_neutral_constraints)
from visualization.efficient_frontier import compute_efficient_frontier, plot_efficient_frontier
from estimators.performance import compute_portfolio_performance
from optimizers.black_litterman import black_litterman_expected_returns
from utils.helper import get_sector_map, get_market_caps, compute_asset_betas

def run_mean_variance(args):
    # --- Inputs
    prices = get_price_data(args.tickers, args.start, args.end)
    rets   = prices.pct_change().dropna()
    mu     = annualized_return(prices, freq=args.freq)
    cov    = annualized_covariance(rets,   freq=args.freq)

    # --- Sector map (only if a cap was provided)
    sector_map = get_sector_map(args.tickers) if args.sector_limit is not None else None

    # --- Real CAPM betas vs Fama–French Mkt-RF (daily)
    if args.beta_neutral:
        ff        = get_fama_french_factors(args.start, args.end, freq="D")
        rf_daily  = ff["RF"].loc[rets.index]          # daily decimal (already /100 in fetcher)
        mktrf     = ff["Mkt-RF"].loc[rets.index]      # daily market excess
        excess    = rets.sub(rf_daily, axis=0)        # <-- NO /252
        X         = sm.add_constant(mktrf)
        betas_map = {}
        for t in excess.columns:
            res = sm.OLS(excess[t], X).fit()
            betas_map[t] = res.params.iloc[1]         # slope coefficient
        betas = pd.Series(betas_map)
    else:
        betas = None

    # --- Optimize
    weights = mean_variance_optimization(
        expected_returns    = mu,
        cov_matrix          = cov,
        target_return       = args.target,
        allow_short         = args.short,
        sector_map          = sector_map,
        max_sector_exposure = args.sector_limit,
        betas               = betas,
        beta_neutral        = args.beta_neutral,
        beta_tolerance      = args.beta_tol,
        max_weight          = args.max_weight,
        max_leverage        = args.max_leverage,
    )

    # --- Report
    metrics = compute_portfolio_performance(weights, rets, freq=args.freq, risk_free_rate=0.02)
    print("\nWeights:\n", weights.round(4))
    print("\nPerformance:\n", {k: float(round(v, 4)) for k, v in metrics.items()})

    # Optional diagnostics (helpful sanity checks)
    if sector_map is not None:
        exp_by_sector = weights.groupby(sector_map).sum().sort_values(ascending=False)
        print("\nSector exposures:\n", exp_by_sector.round(4))
    if args.beta_neutral and betas is not None:
        beta_dot_w = float((betas * weights).sum())
        print(f"\nBeta neutrality check (β·w): {beta_dot_w:.4f}")

def run_risk_parity(args):
    prices = get_price_data(args.tickers, args.start, args.end)
    rets = prices.pct_change().dropna()
    cov = annualized_covariance(rets, freq=args.freq)
    w = risk_parity_optimization(cov, allow_short=args.short)
    rets = prices.pct_change().dropna()
    metrics = compute_portfolio_performance(w, rets, freq=args.freq, risk_free_rate=0.02)
    print("\nWeights:\n", w.round(4))
    print("\nPerformance:\n", {k: round(v, 4) for k, v in metrics.items()})

def run_min_variance(args):
    prices = get_price_data(args.tickers, args.start, args.end)
    rets = prices.pct_change().dropna()
    mu  = annualized_return(prices, freq=args.freq)
    cov = annualized_covariance(rets, freq=args.freq)
    w   = minimum_variance_optimization(mu, cov, allow_short=args.short)
    rets = prices.pct_change().dropna()
    metrics = compute_portfolio_performance(w, rets, freq=args.freq, risk_free_rate=0.02)
    print("\nWeights:\n", w.round(4))
    print("\nPerformance:\n", {k: round(v, 4) for k, v in metrics.items()})

def run_black_litterman(args):
    # 1) Fetch price data & compute inputs
    prices = get_price_data(args.tickers, args.start, args.end)
    rets   = prices.pct_change().dropna()
    cov    = annualized_covariance(rets, freq=args.freq)

    # 2) Fetch true market-cap priors
    market_w = get_market_caps(args.tickers)

    # 3) Compute implied returns via Black–Litterman
    mu_bl = black_litterman_expected_returns(
        cov_matrix     = cov,
        market_weights = market_w,
        P              = None,
        Q              = None,
        Omega          = None,
        tau            = args.tau,
        risk_aversion  = args.risk_aversion,
    )

    # 4) Optimize under the same constraints as Mean-Variance
    w = mean_variance_optimization(
        expected_returns    = mu_bl,
        cov_matrix          = cov,
        allow_short         = args.short,
        max_weight          = args.max_weight,
        max_leverage        = args.max_leverage,
        sector_map          = args.sector_map,
        max_sector_exposure = args.max_sector_exposure,
        betas               = args.betas,
        beta_neutral        = args.beta_neutral,
    )

def run_frontier(args):
    prices = get_price_data(args.tickers, args.start, args.end)
    df = compute_efficient_frontier(
        prices,
        n_points    = args.points,
        allow_short = args.short,
        freq        = args.freq,
        # you can also pass max_weight, max_leverage, sector_map, etc.
    )
    ax = plot_efficient_frontier(df)
    plt.show()    
    

def main():
    parser = argparse.ArgumentParser(
        description="Portfolio Optimizer: fetch data, compute returns/risk, run optimizers"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Mean-Variance subcommand
    mv = sub.add_parser("meanvar", help="Mean-Variance Optimization")
    mv.add_argument("--tickers", nargs="+", default=["AAPL","MSFT","GOOGL","AMZN"])
    mv.add_argument("--start",   type=str, default="2020-01-01")
    mv.add_argument("--end",     type=str, default="2024-12-31")
    mv.add_argument("--freq",    choices=["daily","monthly"], default="daily")
    mv.add_argument("--target",  type=float, help="Target return (annualized)", default=None)
    mv.add_argument("--short",   action="store_true", help="Allow short positions")
    mv.set_defaults(func=run_mean_variance)
    mv.add_argument("--sector-limit", type=float, default=None,
                    help="Max portfolio weight in any single sector (e.g., 0.4)")
    mv.add_argument("--beta-neutral", action="store_true",
                    help="Enforce beta neutrality across the portfolio")
    mv.add_argument("--max-weight",   type=float, default=None,
                help="Cap each weight in [-max_weight, +max_weight]")
    mv.add_argument("--max-leverage", type=float, default=None,
                help="Cap total leverage: sum(abs(w)) ≤ max_leverage")
    mv.add_argument("--beta-tol",type=float,default=None,
                help="Use |beta·w| ≤ beta_tol instead of strict 0")


    # Risk-Parity subcommand
    rp = sub.add_parser("riskparity", help="Risk Parity Optimization")
    rp.add_argument("--tickers", nargs="+", default=["AAPL","MSFT","GOOGL","AMZN"])
    rp.add_argument("--start", type=str, default="2020-01-01")
    rp.add_argument("--end",   type=str, default="2024-12-31")
    rp.add_argument("--freq",  choices=["daily","monthly"], default="daily")
    rp.add_argument("--short", action="store_true", help="Allow short positions")
    rp.set_defaults(func=run_risk_parity)

    # Minimum-Variance subcommand
    mv2 = sub.add_parser("minvar", help="Minimum-Variance Optimization")
    mv2.add_argument("--tickers", nargs="+", default=["AAPL","MSFT","GOOGL","AMZN"])
    mv2.add_argument("--start",   type=str, default="2020-01-01")
    mv2.add_argument("--end",     type=str, default="2024-12-31")
    mv2.add_argument("--freq",    choices=["daily","monthly"], default="daily")
    mv2.add_argument("--short",   action="store_true", help="Allow short positions")
    mv2.set_defaults(func=run_min_variance)    
    
    # Black-Litterman sub-command
    bl = sub.add_parser(
        "blacklitterman", help="Black-Litterman Optimization (implied returns if no views)"
    )
    bl.add_argument("--tickers",        nargs="+", default=["AAPL","MSFT","GOOGL","AMZN"])
    bl.add_argument("--start",          type=str,   default="2020-01-01")
    bl.add_argument("--end",            type=str,   default="2024-12-31")
    bl.add_argument("--freq",           choices=["daily","monthly"], default="daily")
    bl.add_argument("--tau",            type=float, default=0.05, help="τ: uncertainty in prior")
    bl.add_argument("--risk-aversion",  type=float, default=2.5,  help="δ: risk aversion coeff")
    bl.add_argument("--short",          action="store_true",         help="Allow short positions")
    bl.set_defaults(func=run_black_litterman)   
    
    #Efficient-Frontier sub-command
    ef = sub.add_parser("frontier", help="Compute & plot efficient frontier")
    ef.add_argument("--tickers",    nargs="+", default=["AAPL","MSFT","GOOGL","AMZN"])
    ef.add_argument("--start",      type=str, default="2020-01-01")
    ef.add_argument("--end",        type=str, default="2024-12-31")
    ef.add_argument("--freq",       choices=["daily","monthly"], default="daily")
    ef.add_argument("--points",     type=int,   default=50,
                    help="Number of target-return points")
    ef.add_argument("--short",      action="store_true", help="Allow short positions")
    # You can also pass sector/beta constraints here as mv_kwargs
    ef.set_defaults(func=run_frontier)    
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
