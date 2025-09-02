# backtest/rolling.py

import numpy as np
import pandas as pd

from estimators.returns import annualized_return
from estimators.risk import annualized_covariance

def _cov_from_model(rets_window: pd.DataFrame, cov_model: str, freq: str) -> pd.DataFrame:
    periods = 252 if freq == "daily" else 12
    if cov_model == "Sample":
        return annualized_covariance(rets_window, freq=freq)
    elif cov_model.startswith("EWMA"):
        from estimators.risk import cov_ewma
        return cov_ewma(rets_window) * periods
    else:  # Ledoit–Wolf
        from estimators.risk import cov_ledoit_wolf
        return cov_ledoit_wolf(rets_window) * periods

def _rebalance_mask(dates: pd.DatetimeIndex, freq: str) -> np.ndarray:
    idx = pd.DatetimeIndex(dates)
    if freq == "Daily":
        return np.ones(len(idx), dtype=bool)

    s = pd.Series(np.ones(len(idx), dtype=int), index=idx)
    if freq == "Weekly":
        last = s.resample("W-FRI").last().index
    elif freq == "Monthly":
        last = s.resample("ME").last().index   # no deprecation
    elif freq == "Quarterly":
        last = s.resample("QE").last().index   # no deprecation
    else:
        raise ValueError(f"Unknown rebalance freq: {freq}")

    mask = idx.isin(last)                      # may be ndarray already
    return np.asarray(mask, dtype=bool)


def run_rolling_backtest(
    prices: pd.DataFrame,
    rebalance: str = "Monthly",          # "Daily" | "Weekly" | "Monthly" | "Quarterly"
    lookback: int = 252,                 # <-- trading days (not months)
    model: str = "Mean-Variance",        # or "Minimum-Variance"
    allow_short: bool = False,
    max_weight: float | None = None,
    max_leverage: float | None = None,
    sector_map: pd.Series | None = None,
    beta_neutral: bool = False,
    betas: pd.Series | None = None,      # optional precomputed betas (usually None; we re-estimate per window)
    beta_tolerance: float | None = None,
    cov_model: str = "Sample",           # "Sample" | "EWMA (0.94)" | "Ledoit–Wolf"
    target_return: float | None = None,
    rf_annual: float = 0.0,
    benchmark_rets: pd.Series | None = None,
    tc_bps: float = 0.0,                 # transaction cost in bps per $ traded
):
    import numpy as np
    import pandas as pd
    from estimators.risk import annualized_covariance
    from estimators.returns import annualized_return
    from optimizers.mean_variance import mean_variance_optimization

    rets_all = prices.pct_change().dropna()
    if rets_all.empty or prices.shape[1] < 2:
        return (
            pd.Series(dtype=float),
            pd.DataFrame(columns=prices.columns),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
        )

    periods = 252  # daily
    dates   = rets_all.index

    # --- Rebalance dates ---
    is_rebal = _rebalance_mask(dates, rebalance)  # must return np.ndarray[bool] aligned to 'dates'
    rebal_points = dates[is_rebal]

    # Keep only rebalances where we have at least 'lookback' prior return rows
    # (i.e., the rebal index in rets_all is >= lookback-1)
    rp = []
    for d in rebal_points:
        pos = dates.get_loc(d)  # integer location in rets_all
        if isinstance(pos, slice):  # safety for duplicate indices (shouldn't happen)
            pos = pos.stop - 1
        if pos >= lookback - 1:
            rp.append(d)
    rebal_points = pd.Index(rp)

    port_rets_segments = []
    weights_records    = {}   # {rebalance_date: weight_series}
    turnover_points    = []
    cost_points        = []
    last_w             = None

    for i, d in enumerate(rebal_points):
        # Window: last 'lookback' return rows up to and including 'd'
        pos_d     = dates.get_loc(d)
        if isinstance(pos_d, slice):
            pos_d = pos_d.stop - 1
        start_pos = pos_d - (lookback - 1)
        train_idx = dates[start_pos: pos_d + 1]

        train_rets   = rets_all.loc[train_idx]
        train_prices = prices.loc[train_idx]  # same index as returns

        if train_prices.shape[0] < max(30, lookback // 10) or train_prices.shape[1] < 2:
            continue

        # ---- covariance model ----
        if cov_model == "Sample":
            cov_w = annualized_covariance(train_rets, freq="daily")
        elif cov_model.startswith("EWMA"):
            from estimators.risk import cov_ewma
            cov_w = cov_ewma(train_rets) * periods
        else:  # Ledoit–Wolf
            from estimators.risk import cov_ledoit_wolf
            cov_w = cov_ledoit_wolf(train_rets) * periods

        mu_w = annualized_return(train_prices, freq="daily")

        # Optional betas per window (if beta neutrality is requested)
        if beta_neutral and (benchmark_rets is not None):
            spy_w = benchmark_rets.loc[train_idx].dropna()
            if len(spy_w) >= 10:
                from utils.helper import compute_asset_betas
                betas_w = compute_asset_betas(train_rets.loc[spy_w.index], spy_w)
            else:
                betas_w = None
        else:
            betas_w = None

        # Floors per window (if you add floors, reindex here); currently none:
        floors_w = None

        # ---- Optimize at the rebalance date ----
        if model == "Mean-Variance":
            w_t = mean_variance_optimization(
                expected_returns    = mu_w,
                cov_matrix          = cov_w,
                target_return       = target_return,
                allow_short         = allow_short,
                max_weight          = max_weight,
                max_leverage        = max_leverage,
                sector_map          = sector_map.reindex(train_prices.columns) if (sector_map is not None) else None,
                max_sector_exposure = None,  # or pass through if you want it active here
                beta_neutral        = beta_neutral,
                betas               = betas_w,
                beta_tolerance      = beta_tolerance,
                min_weights         = floors_w,
            )
        else:  # "Minimum-Variance"
            w_t = mean_variance_optimization(
                expected_returns    = mu_w,
                cov_matrix          = cov_w,
                target_return       = None,
                allow_short         = allow_short,
                max_weight          = max_weight,
                max_leverage        = max_leverage,
                sector_map          = sector_map.reindex(train_prices.columns) if (sector_map is not None) else None,
                max_sector_exposure = None,
                beta_neutral        = beta_neutral,
                betas               = betas_w,
                beta_tolerance      = beta_tolerance,
                min_weights         = floors_w,
            )

        if (w_t.isna().any()) or (not np.isfinite(w_t.values).all()):
            continue

        # Holdout: (d, next_d] with fixed weights
        next_d = rebal_points[i+1] if (i+1 < len(rebal_points)) else rets_all.index[-1]
        mask   = (rets_all.index > d) & (rets_all.index <= next_d)
        seg    = (rets_all.loc[mask, w_t.index] @ w_t).dropna()
        if seg.empty:
            continue

        # --- turnover & transaction cost at rebalance ---
        if last_w is None:
            prev_w = pd.Series(0.0, index=w_t.index)
        else:
            prev_w = last_w.reindex(w_t.index).fillna(0.0)

        turnover = float((w_t - prev_w).abs().sum())
        cost     = turnover * (tc_bps / 10_000.0)

        # Deduct cost on the first holdout day
        seg.iloc[0] = seg.iloc[0] - cost

        turnover_points.append(turnover)
        cost_points.append(cost)
        weights_records[d] = w_t.copy()
        port_rets_segments.append(seg)
        last_w = w_t.copy()

    if not port_rets_segments:
        return (
            pd.Series(dtype=float),
            pd.DataFrame(columns=prices.columns),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
        )

    port_rets_bt = pd.concat(port_rets_segments).sort_index()
    W            = pd.DataFrame(weights_records).T.reindex(columns=prices.columns).fillna(0.0)

    idx = W.index
    to_ser   = pd.Series(turnover_points, index=idx, name="turnover")
    cost_ser = pd.Series(cost_points,     index=idx, name="cost")

    return port_rets_bt, W, to_ser, cost_ser
