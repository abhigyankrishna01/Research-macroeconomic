"""
Preprocessing pipeline:
  1. Compute log-returns from price CSV
  2. Interpolate quarterly/monthly macro to daily
  3. Rolling z-score normalisation (252-day window, no look-ahead)
  4. Train / val / test splits
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.ffill().dropna(how="all")
    log_ret = np.log(prices / prices.shift(1)).dropna()
    return log_ret


def interpolate_macro(macro: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    macro = macro.copy()
    macro.index = pd.to_datetime(macro.index)
    macro = macro.reindex(macro.index.union(target_index))
    macro = macro.interpolate(method="time")
    macro = macro.reindex(target_index)
    macro = macro.ffill().bfill()
    return macro


def _rolling_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    roll_mean = df.rolling(window, min_periods=1).mean()
    roll_std  = df.rolling(window, min_periods=1).std().replace(0, 1e-8)
    return (df - roll_mean) / roll_std


def split_data(log_ret: pd.DataFrame, macro_daily: pd.DataFrame):
    train_mask = log_ret.index <= config.TRAIN_END
    val_mask   = (log_ret.index > config.TRAIN_END) & (log_ret.index <= config.VAL_END)
    test_mask  = log_ret.index > config.VAL_END

    X_train = log_ret[train_mask].values
    X_val   = log_ret[val_mask].values
    X_test  = log_ret[test_mask].values

    macro_train = macro_daily[train_mask].values
    macro_val   = macro_daily[val_mask].values
    macro_test  = macro_daily[test_mask].values

    dates_train = log_ret.index[train_mask]
    dates_val   = log_ret.index[val_mask]
    dates_test  = log_ret.index[test_mask]

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "macro_train": macro_train, "macro_val": macro_val, "macro_test": macro_test,
        "dates_train": dates_train, "dates_val": dates_val, "dates_test": dates_test,
        "log_returns": log_ret,
        "macro_daily": macro_daily,
    }


def run_preprocessing(prices_path: str = None, macro_path: str = None) -> dict:
    if prices_path is None:
        prices_path = os.path.join(config.RAW_DIR, "prices.csv")
    if macro_path is None:
        macro_path = os.path.join(config.RAW_DIR, "macro.csv")

    print(f"[preprocess] Loading prices from {prices_path}")
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    # keep only known asset columns
    cols = [c for c in config.ASSETS if c in prices.columns]
    prices = prices[cols]

    print(f"[preprocess] Loading macro from {macro_path}")
    macro_raw = pd.read_csv(macro_path, index_col=0, parse_dates=True)

    log_ret = compute_log_returns(prices)

    macro_daily = interpolate_macro(macro_raw, log_ret.index)

    log_ret_norm   = _rolling_zscore(log_ret,    config.ROLLING_NORM_WINDOW)
    macro_norm     = _rolling_zscore(macro_daily, config.ROLLING_NORM_WINDOW)

    features_norm = pd.concat([log_ret_norm, macro_norm], axis=1).dropna()
    common_idx    = log_ret_norm.index.intersection(features_norm.index)
    log_ret_norm  = log_ret_norm.loc[common_idx]
    macro_norm    = macro_norm.loc[common_idx]

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    log_ret.to_csv(os.path.join(config.PROCESSED_DIR, "log_returns.csv"))
    features_norm.to_csv(os.path.join(config.PROCESSED_DIR, "features_norm.csv"))
    macro_daily.to_csv(os.path.join(config.PROCESSED_DIR, "macro_daily.csv"))

    data = split_data(log_ret_norm, macro_norm)

    # Add raw (un-normalised) return splits for backtesting
    for split, mask_key in [("train", data["dates_train"]),
                            ("val",   data["dates_val"]),
                            ("test",  data["dates_test"])]:
        log_ret_norm.loc[mask_key].to_csv(
            os.path.join(config.PROCESSED_DIR, f"log_ret_{split}.csv"))
        data[f"ret_{split}"] = log_ret.reindex(mask_key).values

    data["log_returns_raw"]  = log_ret
    data["macro_daily_raw"]  = macro_daily
    data["features_norm"]    = features_norm
    data["macro_daily_norm"] = macro_norm

    print(f"[preprocess] Train: {len(data['X_train'])} | Val: {len(data['X_val'])} | Test: {len(data['X_test'])}")
    return data


if __name__ == "__main__":
    run_preprocessing()
    print("[preprocess] Done.")
