"""
Generate realistic synthetic ETF price and macro data using Geometric Brownian Motion.
Injects three historical regimes:
  - COVID crash  : 2020-02-20 to 2020-04-01  (bear)
  - 2022 bear    : 2022-01-01 to 2022-10-15  (bear)
  - 2024 AI bull : 2024-01-01 to 2024-12-31  (bull)
Saves to data/sample/prices.csv and data/sample/macro.csv.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

# ETF parameters: (annualised_mu, annualised_sigma, S0)
ETF_PARAMS = {
    "SPY": (0.118, 0.155, 113.0),
    "QQQ": (0.155, 0.195,  44.0),
    "TLT": (0.025, 0.140,  92.0),
    "IEF": (0.020, 0.060,  88.0),
    "GLD": (0.060, 0.130, 109.0),
    "XLF": (0.110, 0.185,  13.0),
    "XLV": (0.120, 0.145,  32.0),
    "XLK": (0.175, 0.210,  21.0),
}

# Regime shocks: (start_date, end_date, daily_drift_override, vol_multiplier)
REGIME_SHOCKS = [
    ("2020-02-20", "2020-04-01", -0.003, 3.0),   # COVID crash
    ("2022-01-01", "2022-10-15", -0.001, 1.8),   # 2022 bear
    ("2024-01-01", "2024-12-31",  0.002, 0.8),   # 2024 AI bull
]


def _gbm_path(mu: float, sigma: float, S0: float,
               n: int, dt: float, rng, shock_drift, shock_vol) -> np.ndarray:
    daily_drift = mu * dt - 0.5 * (sigma ** 2) * dt
    daily_vol   = sigma * np.sqrt(dt)
    z = rng.standard_normal(n)
    drift = np.where(shock_drift != 0, shock_drift, daily_drift)
    vol   = daily_vol * np.where(shock_vol != 1.0, shock_vol, 1.0)
    log_ret = drift + vol * z
    prices = S0 * np.exp(np.cumsum(log_ret))
    return prices


def generate_sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range(start=config.START_DATE, end=config.END_DATE)
    n = len(dates)
    dt = 1.0 / config.TRADING_DAYS
    rng = np.random.default_rng(42)

    shock_drift = np.zeros(n)
    shock_vol   = np.ones(n)
    for start, end, drift_ov, vol_mult in REGIME_SHOCKS:
        mask = (dates >= start) & (dates <= end)
        shock_drift[mask] = drift_ov
        shock_vol[mask]   = vol_mult

    price_dict = {}
    for ticker, (mu, sigma, S0) in ETF_PARAMS.items():
        price_dict[ticker] = _gbm_path(
            mu, sigma, S0, n, dt, rng, shock_drift, shock_vol)

    prices = pd.DataFrame(price_dict, index=dates)

    # Macro: realistic synthetic
    xp = np.arange(0, n, 90)
    gdp_q = np.cumsum(rng.normal(0.005, 0.012, len(xp)))
    gdp = np.interp(np.arange(n), xp, gdp_q)

    xp2 = np.arange(0, n, 30)
    cpi_m = np.cumsum(rng.normal(0.002, 0.004, len(xp2)))
    cpi = np.interp(np.arange(n), xp2, cpi_m)

    t = np.linspace(0, 4 * np.pi, n)
    fed = 2.5 + 2.5 * np.sin(t) + rng.normal(0, 0.1, n)
    fed = np.clip(fed, 0.0, 6.0)

    xp3 = np.arange(0, n, 30)
    ur_m = 5.0 + np.cumsum(rng.normal(0, 0.05, len(xp3)))
    ur_m = np.clip(ur_m, 3.0, 12.0)
    unrate = np.interp(np.arange(n), xp3, ur_m)

    macro = pd.DataFrame(
        {"GDP": gdp, "CPI": cpi, "FEDFUNDS": fed, "UNRATE": unrate},
        index=dates,
    )
    return prices, macro


if __name__ == "__main__":
    os.makedirs(config.SAMPLE_DIR, exist_ok=True)
    prices, macro = generate_sample_data()
    prices_path = os.path.join(config.SAMPLE_DIR, "prices.csv")
    macro_path  = os.path.join(config.SAMPLE_DIR, "macro.csv")
    prices.to_csv(prices_path)
    macro.to_csv(macro_path)
    print(f"[generate_sample] Saved {prices.shape} prices → {prices_path}")
    print(f"[generate_sample] Saved {macro.shape}  macro  → {macro_path}")
