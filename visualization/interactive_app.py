# visualization/interactive_app.py
import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np

from data.fetch_data import get_price_data, get_fama_french_factors
from estimators.returns import annualized_return
from estimators.risk import annualized_covariance  # sample cov (already annualizes)
from optimizers.mean_variance import mean_variance_optimization
from optimizers.minimum_variance import minimum_variance_optimization
from optimizers.risk_parity import risk_parity_optimization
from optimizers.black_litterman import black_litterman_optimization
from visualization.efficient_frontier import compute_efficient_frontier, plot_efficient_frontier
from estimators.performance import compute_portfolio_performance
from optimizers.black_litterman import black_litterman_expected_returns
from utils.helper import get_sector_map, get_market_caps, compute_asset_betas
from optimizers.cvar import cvar_optimization
from backtest.rolling import run_rolling_backtest



benchmark_ticker = "SPY"

# ---------- Caching helpers ----------
@st.cache_data(show_spinner=False)
def fetch_prices_cached(tickers, start, end):
    return get_price_data(tickers, start, end)

@st.cache_data(show_spinner=False)
def fetch_ff_cached(start, end, freq="daily"):
    return get_fama_french_factors(start, end, freq=freq)

@st.cache_data(show_spinner=False)
def fetch_sector_map_cached(tickers):
    return get_sector_map(tickers)
# ------------------------------------


st.set_page_config(layout="wide")
st.title("📊 Portfolio Optimizer Dashboard")

# --- Sidebar inputs ---
st.sidebar.header("Data & Assets")

DEFAULT_OPTIONS = [
    # Large-cap US
    "AAPL","MSFT","GOOGL","AMZN","TSLA","JPM","XOM",
    # Benchmarks & styles
    "SPY","QQQ","IWM","EFA","EEM",
    # Bonds / cash
    "TLT","IEF","LQD","HYG","BIL",
    # Commodities & REITs
    "GLD","SLV","DBC","USO","UNG","VNQ",
    # Sectors
    "XLF","XLK","XLY","XLP","XLE","XLI","XLV","XLU","XLRE","XLB",
    # Low-vol / min-var
    "SPLV","USMV",
    # Hedges
    "SH","SDS"
]

tickers = st.sidebar.multiselect(
    "Select tickers",
    options=DEFAULT_OPTIONS,
    default=["AAPL","MSFT","GOOGL","AMZN"]
)

raw_custom = st.sidebar.text_input("Custom tickers (comma-separated)", value="")
custom_tickers = []
if raw_custom.strip():
    custom_tickers = [t.strip().upper() for t in raw_custom.split(",") if t.strip()]
    tickers = sorted(set(tickers) | set(custom_tickers))

# Dates and optimizer
start = st.sidebar.date_input("Start date", value=pd.to_datetime("2020-01-01"))
end   = st.sidebar.date_input("End date",   value=pd.to_datetime("2024-12-31"))

st.sidebar.header("Optimizer Settings")
opt = st.sidebar.selectbox(
    "Choose optimizer",
    ["Mean-Variance", "Minimum-Variance", "Risk Parity", "Black-Litterman", "CVaR (min)", "Efficient Frontier"]
)

# NEW: covariance model selector
cov_model = st.sidebar.selectbox(
    "Covariance model",
    ["Sample", "EWMA (0.94)", "Ledoit–Wolf"]
)

# Common flags
allow_short = st.sidebar.checkbox("Allow short positions", value=False)

# If shorts are allowed, ask for a per-asset bound (± max_weight)
if allow_short:
    max_weight_slider = st.sidebar.slider(
        "Max |weight| (shorts allowed)",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1
    )
else:
    max_weight_slider = None        # means “no explicit bound”

# Method-specific params (common to MV/MinVar/Frontier)
mv_kwargs = {}
if opt in ["Mean-Variance","Minimum-Variance","Efficient Frontier"]:
    mv_kwargs["freq"] = "daily"
    mv_kwargs["allow_short"] = allow_short
    mv_kwargs["max_weight"] = max_weight_slider

    # Beta-neutral control + tolerance slider
    beta_neutral = st.sidebar.checkbox("Beta-neutral constraint", value=False)
    mv_kwargs["beta_neutral"] = beta_neutral
    if beta_neutral:
        mv_kwargs["beta_tolerance"] = st.sidebar.slider("|β·w| tolerance", 0.0, 0.50, 0.10, 0.01)
    else:
        mv_kwargs["beta_tolerance"] = None

if opt == "Mean-Variance":
    mv_kwargs["target_return"]    = st.sidebar.number_input("Target return (ann.)", 
                                                            min_value=0.0, max_value=1.0, value=None, step=0.01)
    mv_kwargs["max_leverage"]     = st.sidebar.number_input("Max total leverage", 
                                                            min_value=1.0, max_value=5.0, value=None, step=0.1)
    mv_kwargs["max_sector_exposure"] = st.sidebar.number_input("Max sector exposure", 
                                                               min_value=0.0, max_value=1.0, value=None, step=0.05)

elif opt == "Black-Litterman":
    mv_kwargs["tau"]             = st.sidebar.slider("Tau (uncertainty)", 0.001, 1.0, 0.05)
    mv_kwargs["risk_aversion"]   = st.sidebar.slider("Risk aversion δ", 0.5, 5.0, 2.5)
    mv_kwargs["allow_short"]     = allow_short

elif opt == "Efficient Frontier":
    points = st.sidebar.slider("Number of points", 10, 100, 50)
    
# --- CVaR sidebar controls ---
if opt == "CVaR (min)":
    cvar_alpha = st.sidebar.slider("CVaR level α", 0.80, 0.99, 0.95, 0.01)
    cvar_target = st.sidebar.number_input(
        "Target return (annualized, optional)",
        min_value=0.0, max_value=1.0, value=None, step=0.01
    )
    cvar_max_leverage = st.sidebar.number_input(
        "Max total leverage",
        min_value=1.0, max_value=5.0, value=2.0, step=0.1
    )


# ---- Diversification / floors (applies only when shorting is OFF) ----
floor = None
apply_floor_to_custom = False
if not allow_short and opt in ["Mean-Variance", "Minimum-Variance", "Efficient Frontier"]:
    st.sidebar.header("Diversification")
    enable_floor = st.sidebar.checkbox("Set minimum weight floor", value=False,
                                       help="Helps ensure new/custom names get some allocation.")
    if enable_floor:
        floor_pct = st.sidebar.slider("Minimum weight per asset (%)", 0.0, 5.0, 1.0, 0.25)
        floor = floor_pct / 100.0
        apply_floor_to_custom = st.sidebar.checkbox("Apply floor only to custom tickers",
                                                    value=True,
                                                    help="If off, floor applies to ALL assets.")

# Optionally auto-add hedges when beta-neutral is on
if mv_kwargs.get("beta_neutral"):
    auto_hedge = st.sidebar.checkbox("Auto-add hedges (TLT, SH)", value=True)
    if auto_hedge:
        for h in ("TLT", "SH"):
            if h not in tickers:
                tickers.append(h)

# --- Fetch data (cached) ---
if len(tickers) < 2:
    st.error("Select at least 2 tickers.")
    st.stop()

prices = fetch_prices_cached(tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
prices = prices.dropna(axis=1, how="all").dropna()
if prices.shape[1] < 2:
    st.error("Not enough assets with usable data for the selected period. Try different tickers/dates.")
    st.stop()

rets = prices.pct_change().dropna()
mu   = annualized_return(prices, freq=mv_kwargs.get("freq","daily"))

# NEW: choose covariance model (annualize where needed)
periods = 252 if mv_kwargs.get("freq","daily") == "daily" else 12
if cov_model == "Sample":
    cov = annualized_covariance(rets, freq=mv_kwargs.get("freq","daily"))
elif cov_model == "EWMA (0.94)":
    try:
        from estimators.risk import cov_ewma
        cov = cov_ewma(rets) * periods
    except Exception as e:
        st.warning(f"EWMA unavailable ({e}); using Sample covariance instead.")
        cov = annualized_covariance(rets, freq=mv_kwargs.get("freq","daily"))
else:  # Ledoit–Wolf
    try:
        from estimators.risk import cov_ledoit_wolf
        cov = cov_ledoit_wolf(rets) * periods
    except Exception as e:
        st.warning(f"Ledoit–Wolf unavailable ({e}); using Sample covariance instead.")
        cov = annualized_covariance(rets, freq=mv_kwargs.get("freq","daily"))

missing = sorted(set(tickers) - set(prices.columns))
if missing:
    st.warning("Dropped due to missing data in the selected date range: " + ", ".join(missing))

st.caption("Optimized over: " + ", ".join(prices.columns))

# --- Optional minimum weight floors ---
min_weights = None
if floor is not None:
    if apply_floor_to_custom and custom_tickers:
        idx = [t for t in prices.columns if t in custom_tickers]
        if idx:
            min_weights = pd.Series(0.0, index=prices.columns)
            min_weights.loc[idx] = floor
    else:
        min_weights = pd.Series(floor, index=prices.columns)

    # Quick feasibility check (sum of floors must be <= 1)
    if min_weights is not None:
        total_floor = float(min_weights.sum())
        if total_floor > 1.0:
            st.error(f"Weight floors sum to {total_floor:.2f} > 1. Reduce floor or number of floored assets.")
            st.stop()

# --- Risk-free from Fama–French (annualized for the chosen freq) ---
ff = fetch_ff_cached(
    start.strftime("%Y-%m-%d"),
    end.strftime("%Y-%m-%d"),
    freq=mv_kwargs.get("freq", "daily")
)

periods = 252 if mv_kwargs.get("freq","daily") == "daily" else 12
if "RF" in ff.columns:
    # We only need the annualized average; no need to align RF to asset dates.
    rf_annual = float(ff["RF"].mean() * periods)
else:
    st.warning("Fama–French 'RF' column not found; assuming 0% risk-free for metrics.")
    rf_annual = 0.0


# --- Benchmark (SPY) — robust to yfinance's shape quirks ---
bench_df = fetch_prices_cached([benchmark_ticker], start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
bench_series = bench_df.iloc[:, 0] if isinstance(bench_df, pd.DataFrame) else bench_df
bench_ann = annualized_return(bench_series.to_frame(), freq=mv_kwargs.get("freq", "daily"))
benchmark_return = float(bench_ann.iloc[0])
benchmark_rets = bench_series.pct_change().dropna()

# --- Build real sector map (only if you’ve set a sector cap) ---
if mv_kwargs.get("max_sector_exposure") is not None:
    sector_map = fetch_sector_map_cached(tickers)
else:
    sector_map = None

# --- Compute real CAPM betas vs. SPY (only if requested) ---
if mv_kwargs.get("beta_neutral"):
    common_idx = rets.index.intersection(benchmark_rets.index)
    if len(common_idx) < 10:
        st.warning("Not enough overlapping data with SPY to estimate betas reliably.")
        betas = None
    else:
        betas = compute_asset_betas(rets.loc[common_idx], benchmark_rets.loc[common_idx])
else:
    betas = None

# --- Run selected optimizer and display ---
if opt == "Mean-Variance":
    w = mean_variance_optimization(
        expected_returns    = mu,
        cov_matrix          = cov,
        target_return       = mv_kwargs.get("target_return"),
        allow_short         = mv_kwargs["allow_short"],
        max_weight          = mv_kwargs.get("max_weight"),
        max_leverage        = mv_kwargs.get("max_leverage"),
        sector_map          = sector_map,
        max_sector_exposure = mv_kwargs.get("max_sector_exposure"),
        beta_neutral        = mv_kwargs.get("beta_neutral"),
        betas               = betas,
        beta_tolerance      = mv_kwargs.get("beta_tolerance"),
        min_weights         = min_weights,
    )
    
    if (w.isna().any()) or (not np.isfinite(w.values).all()):
        st.error(
            "Optimization infeasible with current constraints. "
            "Try loosening |β·w| tolerance, increasing max leverage, relaxing sector caps, "
            "or adding hedging assets like TLT/SH."
        )
        st.stop()

    st.subheader("Mean-Variance Weights")
    st.write(w.round(4))
    
    metrics = compute_portfolio_performance(
        w, rets, freq=mv_kwargs.get("freq", "daily"), risk_free_rate=rf_annual
    )

    st.subheader("📈 Portfolio Performance Metrics")
    cols = st.columns(4)
    cols[0].metric("Return",     f"{metrics['return']:.2%}")
    cols[1].metric("Volatility", f"{metrics['volatility']:.2%}")
    cols[2].metric("Sharpe",     f"{metrics['sharpe']:.2f}")
    cols[3].metric("SPY Return", f"{benchmark_return:.2%}")

    # --- Cumulative Return Plot ---
    daily_port_returns   = (rets @ w).dropna()
    cum_port_returns     = (1 + daily_port_returns).cumprod()
    cum_benchmark_returns = (1 + benchmark_rets.loc[cum_port_returns.index]).cumprod()

    st.subheader("📊 Cumulative Return: Portfolio vs. SPY")
    fig3, ax3 = plt.subplots()
    cum_port_returns.plot(ax=ax3, label="Portfolio")
    cum_benchmark_returns.plot(ax=ax3, label="SPY", linestyle="--")
    ax3.set_ylabel("Cumulative Return")
    ax3.legend()
    st.pyplot(fig3)

elif opt == "Minimum-Variance":
    w = mean_variance_optimization(
        expected_returns   = mu,
        cov_matrix         = cov,
        target_return      = None,
        allow_short        = mv_kwargs['allow_short'],
        max_weight         = mv_kwargs.get("max_weight"),
        max_leverage       = mv_kwargs.get("max_leverage"),
        sector_map         = sector_map,
        max_sector_exposure= mv_kwargs.get("max_sector_exposure"),
        beta_neutral       = mv_kwargs.get("beta_neutral"),
        betas              = betas,
        beta_tolerance     = mv_kwargs.get("beta_tolerance"),
        min_weights        = min_weights,
    )
    if (w.isna().any()) or (not np.isfinite(w.values).all()):
        st.error(
            "Optimization infeasible with current constraints. "
            "Try loosening |β·w| tolerance, increasing max leverage, relaxing sector caps, "
            "or adding hedging assets like TLT/SH."
        )
        st.stop()

    st.subheader("Global Minimum-Variance Weights")
    st.write(w.round(4))
    
    metrics = compute_portfolio_performance(
        w, rets, freq=mv_kwargs.get("freq", "daily"), risk_free_rate=rf_annual
    )
    st.subheader("📈 Portfolio Performance Metrics")
    cols = st.columns(4)
    cols[0].metric("Return",     f"{metrics['return']:.2%}")
    cols[1].metric("Volatility", f"{metrics['volatility']:.2%}")
    cols[2].metric("Sharpe",     f"{metrics['sharpe']:.2f}")
    cols[3].metric("SPY Return", f"{benchmark_return:.2%}")

    # --- Cumulative Return: Portfolio vs SPY ---
    daily_port_returns   = (rets @ w).dropna()
    cum_port_returns     = (1 + daily_port_returns).cumprod()
    cum_benchmark_returns = (1 + benchmark_rets.loc[cum_port_returns.index]).cumprod()

    fig3, ax3 = plt.subplots()
    cum_port_returns.plot(ax=ax3, label="Portfolio")
    cum_benchmark_returns.plot(ax=ax3, label="SPY", linestyle="--")
    ax3.set_ylabel("Cumulative Return")
    ax3.legend()
    st.pyplot(fig3)

elif opt == "Risk Parity":
    w = risk_parity_optimization(cov, allow_short=allow_short)
    if (w.isna().any()) or (not np.isfinite(w.values).all()):
        st.error(
            "Optimization infeasible with current constraints. "
            "Try loosening |β·w| tolerance, increasing max leverage, relaxing sector caps, "
            "or adding hedging assets like TLT/SH."
        )
        st.stop()
    
    st.subheader("Risk Parity Weights")
    st.write(w.round(4))
    
    metrics = compute_portfolio_performance(
        w, rets, freq=mv_kwargs.get("freq", "daily"), risk_free_rate=rf_annual
    )
    st.subheader("📈 Portfolio Performance Metrics")
    cols = st.columns(4)
    cols[0].metric("Return",     f"{metrics['return']:.2%}")
    cols[1].metric("Volatility", f"{metrics['volatility']:.2%}")
    cols[2].metric("Sharpe",     f"{metrics['sharpe']:.2f}")
    cols[3].metric("SPY Return", f"{benchmark_return:.2%}")
    
    # --- Cumulative Return: Portfolio vs SPY ---
    daily_port_returns   = (rets @ w).dropna()
    cum_port_returns     = (1 + daily_port_returns).cumprod()
    cum_benchmark_returns = (1 + benchmark_rets.loc[cum_port_returns.index]).cumprod()

    fig3, ax3 = plt.subplots()
    cum_port_returns.plot(ax=ax3, label="Portfolio")
    cum_benchmark_returns.plot(ax=ax3, label="SPY", linestyle="--")
    ax3.set_ylabel("Cumulative Return")
    ax3.legend()
    st.pyplot(fig3)

elif opt == "Black-Litterman":
    # 1) Market weights from helper (handles normalization/fallbacks)
    market_w = get_market_caps(tickers)

    # 2) Compute posterior expected returns μ_BL
    mu_bl = black_litterman_expected_returns(
        cov_matrix     = cov,
        market_weights = market_w,
        P              = None,
        Q              = None,
        Omega          = None,
        tau            = mv_kwargs["tau"],
        risk_aversion  = mv_kwargs["risk_aversion"],
    )

    # 3) Run mean-variance with the SAME constraints you use elsewhere
    w = mean_variance_optimization(
        expected_returns    = mu_bl,
        cov_matrix          = cov,
        allow_short         = allow_short,
        max_weight          = mv_kwargs.get("max_weight"),
        max_leverage        = mv_kwargs.get("max_leverage"),
        sector_map          = sector_map,
        max_sector_exposure = mv_kwargs.get("max_sector_exposure"),
        beta_neutral        = mv_kwargs.get("beta_neutral"),
        betas               = betas,
        beta_tolerance      = mv_kwargs.get("beta_tolerance"),
        min_weights         = min_weights,
    )
    if (w.isna().any()) or (not np.isfinite(w.values).all()):
        st.error(
            "Optimization infeasible with current constraints. "
            "Try loosening |β·w| tolerance, increasing max leverage, relaxing sector caps, "
            "or adding hedging assets like TLT/SH."
        )
        st.stop()

    # --- Display results --------------------------------------------------
    st.subheader("Black – Litterman Weights (with constraints)")
    st.write(w.round(4))

    metrics = compute_portfolio_performance(
        w, rets, freq=mv_kwargs.get("freq", "daily"), risk_free_rate=rf_annual
    )
    st.subheader("📈 Portfolio Performance Metrics")
    cols = st.columns(4)
    cols[0].metric("Return",     f"{metrics['return']:.2%}")
    cols[1].metric("Volatility", f"{metrics['volatility']:.2%}")
    cols[2].metric("Sharpe",     f"{metrics['sharpe']:.2f}")
    cols[3].metric("SPY Return", f"{benchmark_return:.2%}")

    # --- Cumulative Return: Portfolio vs SPY ---
    daily_port_returns   = (rets @ w).dropna()
    cum_port_returns     = (1 + daily_port_returns).cumprod()
    cum_benchmark_returns = (1 + benchmark_rets.loc[cum_port_returns.index]).cumprod()

    fig3, ax3 = plt.subplots()
    cum_port_returns.plot(ax=ax3, label="Portfolio")
    cum_benchmark_returns.plot(ax=ax3, label="SPY", linestyle="--")
    ax3.set_ylabel("Cumulative Return")
    ax3.legend()
    st.pyplot(fig3)

elif opt == "CVaR (min)":
    w = cvar_optimization(
        returns       = rets,                # daily returns matrix (T x N)
        alpha         = cvar_alpha,
        target_return = cvar_target,         # annualized target (optional)
        allow_short   = allow_short,
        max_weight    = max_weight_slider,
        max_leverage  = cvar_max_leverage,
    )

    # Infeasible/failed check
    if (w.isna().any()) or (not np.isfinite(w.values).all()):
        st.error(
            "CVaR optimization infeasible with current constraints. "
            "Try lowering α, removing the target return, increasing max leverage, or allowing shorts."
        )
        st.stop()

    st.subheader("CVaR-Min Weights")
    st.write(w.round(4))

    # Metrics (same as other blocks)
    metrics = compute_portfolio_performance(
        w, rets, freq=mv_kwargs.get("freq", "daily"), risk_free_rate=rf_annual
    )
    st.subheader("📈 Portfolio Performance Metrics")
    cols = st.columns(4)
    cols[0].metric("Return",     f"{metrics['return']:.2%}")
    cols[1].metric("Volatility", f"{metrics['volatility']:.2%}")
    cols[2].metric("Sharpe",     f"{metrics['sharpe']:.2f}")
    cols[3].metric("SPY Return", f"{benchmark_return:.2%}")

    # Cumulative chart
    daily_port_returns   = (rets @ w).dropna()
    cum_port_returns     = (1 + daily_port_returns).cumprod()
    cum_benchmark_returns = (1 + benchmark_rets.loc[cum_port_returns.index]).cumprod()

    st.subheader("📊 Cumulative Return: Portfolio vs. SPY")
    fig3, ax3 = plt.subplots()
    cum_port_returns.plot(ax=ax3, label="Portfolio")
    cum_benchmark_returns.plot(ax=ax3, label="SPY", linestyle="--")
    ax3.set_ylabel("Cumulative Return")
    ax3.legend()
    st.pyplot(fig3)


elif opt == "Efficient Frontier":
    mv_kwargs.pop("allow_short", None)  # remove duplicate before unpacking
    ef = compute_efficient_frontier(
        prices,
        n_points    = points,
        allow_short = allow_short,
        **mv_kwargs
    )

    st.subheader("Efficient Frontier")
    fig, ax = plt.subplots()
    plot_efficient_frontier(ef, ax=ax)
    st.pyplot(fig)

# --- Allocation visualization (handles shorts) ---
if opt in ["Mean-Variance", "Minimum-Variance", "Risk Parity", "Black-Litterman", "CVaR (min)"]:
    alloc = w[w.abs() > 1e-4]  # keep only meaningful weights
    if alloc.empty:
        st.info("All selected assets received ~0 weight under the current constraints.")
    else:
        longs  = alloc[alloc > 0].sort_values(ascending=False)
        shorts = (-alloc[alloc < 0]).sort_values(ascending=False)  # use magnitudes for the pie

        if shorts.empty:
            st.subheader("Portfolio Allocation")
            fig2, ax2 = plt.subplots()
            ax2.pie(longs.values, labels=longs.index, autopct="%0.1f%%")
            st.pyplot(fig2)
        else:
            c1, c2 = st.columns(2)

            if not longs.empty:
                c1.subheader("Long allocation")
                figL, axL = plt.subplots()
                axL.pie(longs.values, labels=longs.index, autopct="%0.1f%%")
                c1.pyplot(figL)

            if not shorts.empty:
                c2.subheader("Short allocation (absolute weights)")
                figS, axS = plt.subplots()
                axS.pie(shorts.values, labels=shorts.index, autopct="%0.1f%%")
                c2.pyplot(figS)

            # Net weights bar (shows signs explicitly)
            st.subheader("Net weights (longs + / shorts −)")
            figB, axB = plt.subplots()
            alloc.sort_values().plot(kind="barh", ax=axB)
            axB.axvline(0, linewidth=1)
            axB.set_xlabel("Weight")
            st.pyplot(figB)
            
# =========================
# 🕒 Rolling Backtest (new)
# =========================
bt_tab, bl_views_tab = st.tabs(["🕒 Rolling Backtest", "🧠 BL Views"])

with bt_tab:
    st.caption("Walk-forward, out-of-sample performance with scheduled rebalances, turnover & transaction costs.")

    # Optimizer for backtest
    bt_opt = st.selectbox(
        "Backtest optimizer",
        ["Mean-Variance", "Minimum-Variance"],
        index=0
    )

    # Controls
    rebal_freq_label = st.selectbox(
        "Rebalance frequency",
        ["Monthly", "Quarterly", "Weekly", "Daily"],
        index=0
    )
    lookback_months = st.slider("Lookback window (months)", 6, 60, 24, 1)
    lookback_days = int(lookback_months*21)

    tc_bps = st.slider(
        "Transaction cost (bps per $ traded)",
        min_value=0, max_value=50, value=5, step=1,
        help="Deducted at each rebalance on the first holdout day: cost = (bps/10000) × ∑|Δw|."
    )

    run_bt = st.button("Run backtest", type="primary")

    if run_bt:
        # Run the engine (returns net daily returns, weights at rebalances, and per-rebalance turnover/cost)
        port_rets_bt, W, to_ser, cost_ser = run_rolling_backtest(
            prices=prices,
            rebalance=rebal_freq_label,            # "Daily"|"Weekly"|"Monthly"|"Quarterly"
            lookback=lookback_days,
            model=bt_opt,                          # "Mean-Variance" or "Minimum-Variance"
            allow_short=mv_kwargs.get("allow_short", False),
            max_weight=mv_kwargs.get("max_weight"),
            max_leverage=mv_kwargs.get("max_leverage"),
            sector_map=sector_map,
            beta_neutral=mv_kwargs.get("beta_neutral"),
            betas=betas,
            beta_tolerance=mv_kwargs.get("beta_tolerance"),
            cov_model=cov_model,                   # "Sample"|"EWMA (0.94)"|"Ledoit–Wolf"
            target_return=mv_kwargs.get("target_return"),
            rf_annual=rf_annual,
            benchmark_rets=benchmark_rets,
            tc_bps=tc_bps,
        )

        if port_rets_bt.empty:
            st.error("Backtest produced no segments (constraints likely too tight or not enough data).")
        else:
            # Build a cost-by-day series (cost applied on the first holdout day after each rebalance)
            idx_all = port_rets_bt.index
            cost_by_day = pd.Series(0.0, index=idx_all)

            # map each rebalance date to the next trading day in the returns index
            for rd, cval in cost_ser.items():
                pos = idx_all.searchsorted(rd, side="right")  # first index > rd
                if pos < len(idx_all):
                    cost_by_day.iloc[pos] += float(cval)

            # Net (already costed) and Gross (add costs back)
            net_ret   = port_rets_bt
            gross_ret = (port_rets_bt + cost_by_day).reindex(idx_all).fillna(0.0)

            nav_net   = (1 + net_ret).cumprod()
            nav_gross = (1 + gross_ret).cumprod()

            # Assemble a handy DataFrame for display
            bt = pd.DataFrame({
                "NAV_gross": nav_gross,
                "NAV_net":   nav_net,
                "net_ret":   net_ret,
                "gross_ret": gross_ret,
            })

            # Turnover per rebalance (index = rebalance dates)
            # For display convenience, join turnover to bt on the rebalance dates
            turnover_df = to_ser.to_frame("turnover")
            cost_df     = cost_ser.to_frame("cost")

            # Plot equity curves
            st.subheader("Equity Curve (Gross vs Net of Costs)")
            fig_bt, ax_bt = plt.subplots()
            bt[["NAV_gross","NAV_net"]].plot(ax=ax_bt)
            ax_bt.set_ylabel("NAV")
            ax_bt.legend()
            st.pyplot(fig_bt)

            # Metrics
            years = max((bt.index[-1] - bt.index[0]).days / 365.25, 1e-9)
            cagr_gross = bt["NAV_gross"].iloc[-1] ** (1/years) - 1
            cagr_net   = bt["NAV_net"].iloc[-1]   ** (1/years) - 1
            avg_turn   = float(to_ser.mean()) if not to_ser.empty else 0.0
            tot_cost   = float(cost_ser.sum()) if not cost_ser.empty else 0.0

            c = st.columns(4)
            c[0].metric("Avg turnover (rebals)", f"{avg_turn:.2%}")
            c[1].metric("Total costs (sum of drags)", f"{tot_cost:.2%}")
            c[2].metric("CAGR Gross",            f"{cagr_gross:.2%}")
            c[3].metric("CAGR Net",              f"{cagr_net:.2%}")

            with st.expander("Rebalance dates — turnover & costs"):
                # Align costs to the *applied* (next-trading-day) date for clarity
                # And show turnover on the rebalance date
                # Provide both tables side-by-side
                colL, colR = st.columns(2)
                colL.write("**Turnover (on rebalance dates)**")
                colL.dataframe(turnover_df.style.format({"turnover": "{:.2%}"}))

                colR.write("**Costs (applied next trading day)**")
                colR.dataframe(cost_df.style.format({"cost": "{:.4%}"}))

            with st.expander("Weights at each rebalance"):
                st.dataframe(W.round(4))


with bl_views_tab:
    st.caption("Express absolute return views on specific tickers (annualized). We’ll blend with market priors via Black–Litterman, then optimize with your current constraints.")

    view_tickers = st.multiselect("Tickers to add views on", options=list(prices.columns), default=[])
    views = {}
    confs = {}

    if view_tickers:
        st.write("Enter **expected annual return** and **view uncertainty (stdev)** for each.")
        grid = st.container()
        for t in view_tickers:
            col1, col2 = grid.columns(2)
            views[t] = col1.number_input(f"{t} expected return (%)", -50.0, 50.0, value=8.0, step=0.5, key=f"v_{t}") / 100.0
            confs[t] = col2.number_input(f"{t} view stdev (%)",       0.1, 50.0, value=5.0, step=0.5, key=f"s_{t}") / 100.0

    tau = st.slider("Tau (overall prior uncertainty)", 0.001, 1.0, 0.05)
    delta = st.slider("Risk aversion δ", 0.5, 5.0, 2.5)

    apply_bl = st.button("Apply views and optimize", type="primary")

    if apply_bl:
        # Market caps -> priors
        market_w = get_market_caps(list(prices.columns))

        # Build P, Q, Omega in the current column order
        if view_tickers:
            P = pd.DataFrame(0.0, index=view_tickers, columns=prices.columns)
            for t in view_tickers:
                if t in P.columns:
                    P.loc[t, t] = 1.0
            Q = pd.Series({t: views[t] for t in view_tickers}).reindex(P.index).values
            Omega = np.diag([(confs[t] ** 2) for t in view_tickers])
        else:
            P = None; Q = None; Omega = None

        # Posterior expected returns
        mu_bl = black_litterman_expected_returns(
            cov_matrix     = cov,
            market_weights = market_w,
            P              = P,
            Q              = Q,
            Omega          = Omega,
            tau            = tau,
            risk_aversion  = delta,
        )

        # Optimize with same constraints you’re using elsewhere
        w_bl = mean_variance_optimization(
            expected_returns    = mu_bl,
            cov_matrix          = cov,
            allow_short         = allow_short,
            max_weight          = mv_kwargs.get("max_weight"),
            max_leverage        = mv_kwargs.get("max_leverage"),
            sector_map          = sector_map,
            max_sector_exposure = mv_kwargs.get("max_sector_exposure"),
            beta_neutral        = mv_kwargs.get("beta_neutral"),
            betas               = betas,
            beta_tolerance      = mv_kwargs.get("beta_tolerance"),
            min_weights         = min_weights,
        )

        if (w_bl.isna().any()) or (not np.isfinite(w_bl.values).all()):
            st.error("Optimization infeasible with these views/constraints. Try increasing uncertainty (stdev), loosening β tolerance, or relaxing sector/leverage caps.")
        else:
            st.subheader("BL (with views) — Weights")
            st.write(w_bl.round(4))

            m_bl = compute_portfolio_performance(w_bl, rets, freq=mv_kwargs.get("freq", "daily"), risk_free_rate=rf_annual)
            c = st.columns(3)
            c[0].metric("Return",     f"{m_bl['return']:.2%}")
            c[1].metric("Volatility", f"{m_bl['volatility']:.2%}")
            c[2].metric("Sharpe",     f"{m_bl['sharpe']:.2f}")

            # Cum return plot vs SPY
            dpr = (rets @ w_bl).dropna()
            cum_p = (1 + dpr).cumprod()
            cum_b = (1 + benchmark_rets.loc[cum_p.index]).cumprod()
            figv, axv = plt.subplots()
            cum_p.plot(ax=axv, label="BL (views)")
            cum_b.plot(ax=axv, label="SPY", linestyle="--")
            axv.set_ylabel("Cumulative")
            axv.legend()
            st.pyplot(figv)
