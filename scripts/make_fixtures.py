# scripts/make_fixtures.py

import numpy as np
import pandas as pd
from pathlib import Path

def simulate_prices(tickers, start="2020-01-01", periods=756, mu=None, vol=None, seed=42):
    """
    Deterministic GBM-ish simulator: returns daily prices DataFrame.
    periods ~ 3y of business days (252 * 3).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=periods)
    n = len(tickers)
    mu  = np.array(mu if mu is not None else [0.12, 0.10, 0.04])[:n]        # annual drift
    vol = np.array(vol if vol is not None else [0.25, 0.20, 0.10])[:n]      # annual vol

    dt = 1/252
    # daily drift/vol
    mu_d  = mu * dt
    vol_d = vol * np.sqrt(dt)

    # log-returns
    Z = rng.standard_normal((periods, n))
    r = mu_d + vol_d * Z
    # price paths
    P0 = np.array([100.0] * n)
    prices = pd.DataFrame(P0 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=tickers)
    return prices

def main():
    outdir = Path("tests/fixtures")
    outdir.mkdir(parents=True, exist_ok=True)
    tickers = ["AAA", "BBB", "BND"]  # two 'equities', one 'bond'
    prices  = simulate_prices(tickers)
    prices.to_csv(outdir / "prices_sim_3y.csv", float_format="%.6f")
    # also a fake market (benchmark) series to estimate betas in tests
    mkt = simulate_prices(["MKT"], mu=[0.08], vol=[0.15]).iloc[:, 0]
    mkt.to_csv(outdir / "benchmark_sim_3y.csv", header=True, float_format="%.6f")
    print("Wrote tests/fixtures/prices_sim_3y.csv and benchmark_sim_3y.csv")

if __name__ == "__main__":
    main()
