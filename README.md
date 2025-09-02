# Portfolio Optimizer

A practical, batteries-included portfolio research toolkit.  
It fetches asset data, estimates returns & risk, and runs multiple optimizers with real-world constraints — plus an interactive Streamlit dashboard and a rolling backtest with turnover & transaction costs.

## ✨ Features

- **Optimizers**
  - Mean–Variance (with target return optional)
  - Global Minimum Variance
  - Risk Parity
  - **CVaR (Conditional VaR)** — downside risk optimization
  - Black–Litterman (market priors + optional user views)

- **Constraints**
  - Long-only or long/short
  - Per-asset bounds (± `max_weight`)
  - Total leverage cap (`L1` norm)
  - **Sector exposure cap** (by sector map)
  - **Beta-neutrality** (exact or with tolerance |β·w| ≤ ε)
  - **Per-asset weight floors** (for diversification; optional)

- **Estimators**
  - Annualized returns (daily/monthly)
  - Covariance models: **Sample**, **EWMA(0.94)**, **Ledoit–Wolf**

- **Data**
  - Prices via `yfinance`
  - Risk-free rate via Fama–French (pandas-datareader)

- **Visualization**
  - **Streamlit dashboard**: Weights, metrics, cumulative returns, Efficient Frontier
  - Allocation plots (pie for long-only; bar/stacked for long/short)

- **Backtesting**
  - **Rolling walk-forward** backtest with scheduled rebalances (Daily/Weekly/Monthly/Quarterly)
  - Tracks **turnover** and applies **transaction costs (bps per $ traded)**
  - Reports gross vs. net equity curves & summary stats

- **Tests**
  - `pytest` smoke & unit tests for core optimizers, sector caps, and backtest

---

## 🗂️ Project Structure
portfolio_optimizer/
├─ backtest/
│ └─ rolling.py # rolling walk-forward backtester (turnover & costs)
├─ data/
│ └─ fetch_data.py # yfinance + Fama–French fetchers
├─ estimators/
│ ├─ returns.py # annualized return
│ ├─ risk.py # sample/EWMA/Ledoit–Wolf covariance
│ └─ performance.py # return/vol/Sharpe metrics
├─ optimizers/
│ ├─ mean_variance.py # MV + constraints (sector/beta/leverage/floors)
│ ├─ minimum_variance.py
│ ├─ risk_parity.py
│ ├─ cvar.py # NEW: CVaR optimizer
│ └─ black_litterman.py # BL posterior + convenience wrapper
├─ utils/
│ └─ helper.py # sector map, market caps, compute betas, etc.
├─ visualization/
│ ├─ interactive_app.py # Streamlit dashboard
│ └─ efficient_frontier.py
├─ tests/
│ ├─ conftest.py
│ ├─ test_optimizers.py
│ ├─ test_sector_cap.py
│ └─ test_backtest_smoke.py
├─ main.py # CLI for running optimizers
├─ requirements.txt
└─ README.md



---

## ⚙️ Installation

```bash
# clone your repo
git clone https://github.com/<you>/portfolio_optimizer.git
cd portfolio_optimizer

# (recommended) create a virtual env
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# install deps
pip install -r requirements.txt
```

## 🚀 Quick Start

**A) Streamlit App**
streamlit run visualization/interactive_app.py

**What you can do in the UI**
-Select tickers (and add Custom tickers).
-Choose optimizer & constraints (long/short, max_weight, leverage, sector cap).
-Toggle Beta-neutral (+ tolerance).
-Switch covariance model (Sample / EWMA / Ledoit–Wolf).
-Visualize weights, performance, Efficient Frontier.
-Rolling Backtest tab: choose rebal. freq., lookback, transaction costs, see net vs. gross equity & turnover.
-BL Views tab: add absolute return views with uncertainty; optimizer blends with market priors.

**B) CLI**

```bash
# Mean–Variance with sector cap, beta-neutral tolerance, leverage cap
python main.py meanvar \
  --tickers AAPL MSFT GOOGL AMZN TLT \
  --start 2020-01-01 --end 2024-12-31 \
  --short --sector-limit 0.5 \
  --beta-neutral --beta-tol 0.10 \
  --max-weight 1.0 --max-leverage 2.0
```

## 🧠 Key Concepts & Defaults
-Annualization: daily → ×252, monthly → ×12.
-Sector caps: sector map gathered via yfinance; unknowns fall into "Unknown".
-Beta neutrality: in the app, betas estimated vs SPY by default; in CLI you can switch to Fama–French Mkt-RF.
-Charts & shorts: pie charts only for non-negative weights; the app auto-switches to a bar/stacked chart if shorts exist.
-Transaction costs: charged per $ traded at each rebalance:
cost = (bps / 10,000) × ∑|Δw|, applied on the first holdout day after rebalance.

## 🧪 Run Tests

```bash
pytest -q
```

## 📦 Requirements

```nginx
yfinance
pandas
numpy
matplotlib
cvxpy
streamlit
scikit-learn
scipy
pandas_datareader
statsmodels
osqp
scs

```