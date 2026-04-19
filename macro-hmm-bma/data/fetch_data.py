"""
Download price data from yfinance and macro data from FRED.
Falls back to synthetic macro if FRED_API_KEY is not set.
Saves to data/raw/prices.csv and data/raw/macro.csv.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def fetch_prices(retries: int = 3) -> pd.DataFrame:
    import yfinance as yf
    import time
    print(f"[fetch_data] Downloading prices for {config.ASSETS} ...")
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                config.ASSETS,
                start=config.START_DATE,
                end=config.END_DATE,
                auto_adjust=True,
                progress=False,
            )
            break
        except Exception as exc:
            print(f"[fetch_data] Attempt {attempt}/{retries} failed: {exc}")
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw
    prices = prices[config.ASSETS].dropna(how="all")
    prices.index = pd.to_datetime(prices.index)
    print(f"[fetch_data] Prices shape: {prices.shape}")
    return prices


def _synthetic_macro(n: int, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Realistic synthetic macro when FRED key is unavailable."""
    rng = np.random.default_rng(42)

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

    df = pd.DataFrame(
        {"GDP": gdp, "CPI": cpi, "FEDFUNDS": fed, "UNRATE": unrate},
        index=index,
    )
    return df


def fetch_macro(index: pd.DatetimeIndex) -> pd.DataFrame:
    if not config.FRED_API_KEY:
        print("[fetch_data] No FRED_API_KEY — using synthetic macro.")
        return _synthetic_macro(len(index), index)

    try:
        from fredapi import Fred
        fred = Fred(api_key=config.FRED_API_KEY)
        frames = {}
        for name, series_id in config.FRED_SERIES.items():
            print(f"[fetch_data] Downloading FRED series {series_id} ...")
            s = fred.get_series(series_id, observation_start=config.START_DATE,
                                observation_end=config.END_DATE)
            frames[name] = s
        macro = pd.DataFrame(frames)
        macro.index = pd.to_datetime(macro.index)
        macro = macro.reindex(index).ffill().bfill()
        print(f"[fetch_data] Macro shape: {macro.shape}")
        return macro
    except Exception as exc:
        print(f"[fetch_data] FRED download failed ({exc}). Using synthetic macro.")
        return _synthetic_macro(len(index), index)


def run_fetch():
    os.makedirs(config.RAW_DIR, exist_ok=True)
    prices = fetch_prices()
    prices.to_csv(os.path.join(config.RAW_DIR, "prices.csv"))
    print(f"[fetch_data] Saved prices → {config.RAW_DIR}/prices.csv")

    macro = fetch_macro(prices.index)
    macro.to_csv(os.path.join(config.RAW_DIR, "macro.csv"))
    print(f"[fetch_data] Saved macro  → {config.RAW_DIR}/macro.csv")
    return prices, macro


if __name__ == "__main__":
    run_fetch()
