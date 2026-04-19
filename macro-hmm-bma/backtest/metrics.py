"""
Performance metrics — all functions hardened against NaN, Inf, and empty arrays.
"""
import numpy as np
import config


def _clean(returns) -> np.ndarray:
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    return r


def sharpe_ratio(returns, risk_free: float = None) -> float:
    if risk_free is None:
        risk_free = config.RISK_FREE_RATE
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    daily_rf = risk_free / config.TRADING_DAYS
    excess   = r - daily_rf
    std      = excess.std()
    if std < 1e-10:
        return 0.0
    return float(excess.mean() / std * np.sqrt(config.TRADING_DAYS))


def sortino_ratio(returns, risk_free: float = None) -> float:
    if risk_free is None:
        risk_free = config.RISK_FREE_RATE
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    daily_rf   = risk_free / config.TRADING_DAYS
    excess     = r - daily_rf
    downside   = excess[excess < 0]
    if len(downside) == 0:
        return float(excess.mean() * config.TRADING_DAYS)
    downside_std = np.sqrt((downside ** 2).mean())
    if downside_std < 1e-10:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(config.TRADING_DAYS))


def max_drawdown(returns) -> float:
    r = _clean(returns)
    if len(r) == 0:
        return 0.0
    cum = np.exp(np.cumsum(r))
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / np.where(peak == 0, 1.0, peak)
    return float(dd.min())


def calmar_ratio(returns) -> float:
    r = _clean(returns)
    if len(r) == 0:
        return 0.0
    ann = cagr(r)
    mdd = abs(max_drawdown(r))
    if mdd < 1e-10:
        return 0.0
    return float(ann / mdd)


def cagr(returns) -> float:
    r = _clean(returns)
    if len(r) == 0:
        return 0.0
    total = float(np.exp(r.sum()))
    years = len(r) / config.TRADING_DAYS
    if years <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)


def total_return(returns) -> float:
    r = _clean(returns)
    if len(r) == 0:
        return 0.0
    return float(np.exp(r.sum()) - 1.0)


def compute_metrics(returns, name: str = "") -> dict:
    r = _clean(returns)
    return {
        "Strategy":     name,
        "Total Return": round(total_return(r) * 100, 2),
        "CAGR (%)":     round(cagr(r) * 100, 2),
        "Sharpe":       round(sharpe_ratio(r), 3),
        "Sortino":      round(sortino_ratio(r), 3),
        "Max Drawdown": round(max_drawdown(r) * 100, 2),
        "Calmar":       round(calmar_ratio(r), 3),
    }
