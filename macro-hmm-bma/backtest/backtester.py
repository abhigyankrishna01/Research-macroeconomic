"""
Walk-forward backtester.
Compares 7 strategies over the test set:
  1. Equal Weight
  2. 60/40 (equity/bond)
  3. Buy & Hold SPY
  4. Markowitz MVO
  5. Standard PPO (no regime conditioning)
  6. HMM-Only (regime-based, no BMA/PPO)
  7. Macro-HMM-BMA (full system)
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from backtest.metrics import compute_metrics

warnings.filterwarnings("ignore")

EQUITY_IDX = [0, 1, 5, 6, 7]   # SPY, QQQ, XLF, XLV, XLK
BOND_IDX   = [2, 3]             # TLT, IEF
SPY_IDX    = 0


def _portfolio_returns(log_returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """weights shape: (T, N) or (N,) for static."""
    log_returns = np.asarray(log_returns, dtype=float)
    weights     = np.asarray(weights,     dtype=float)
    if weights.ndim == 1:
        weights = np.tile(weights, (len(log_returns), 1))
    return (log_returns * weights).sum(axis=1)


def _avg_turnover(weights: np.ndarray) -> float:
    """Mean daily L1 weight change — proxy for transaction cost pressure."""
    w = np.asarray(weights, dtype=float)
    if w.ndim == 1:
        return 0.0
    diffs = np.abs(np.diff(w, axis=0))
    return float(diffs.sum(axis=1).mean())


def equal_weight(log_returns: np.ndarray) -> np.ndarray:
    T, N = log_returns.shape
    w = np.ones(N) / N
    return _portfolio_returns(log_returns, w)


def sixty_forty(log_returns: np.ndarray) -> np.ndarray:
    T, N = log_returns.shape
    w = np.zeros(N)
    if len(EQUITY_IDX) > 0:
        w[EQUITY_IDX] = 0.60 / len(EQUITY_IDX)
    if len(BOND_IDX) > 0:
        w[BOND_IDX]   = 0.40 / len(BOND_IDX)
    w = w / (w.sum() + 1e-12)
    return _portfolio_returns(log_returns, w)


def buy_hold_spy(log_returns: np.ndarray) -> np.ndarray:
    return log_returns[:, SPY_IDX]


def markowitz_mvo(log_returns: np.ndarray, window: int = 252) -> np.ndarray:
    T, N = log_returns.shape
    weights = np.zeros((T, N))

    for t in range(T):
        start = max(0, t - window)
        hist  = log_returns[start:t + 1]
        if len(hist) < N + 1:
            weights[t] = np.ones(N) / N
            continue
        mu  = hist.mean(axis=0)
        cov = np.cov(hist.T) + np.eye(N) * 1e-6

        def neg_sharpe(w):
            port_ret = w @ mu
            port_var = w @ cov @ w
            return -port_ret / (np.sqrt(port_var) + 1e-8)

        cons   = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bounds = [(0.0, 1.0)] * N
        w0     = np.ones(N) / N
        try:
            res = minimize(neg_sharpe, w0, method="SLSQP",
                           bounds=bounds, constraints=cons,
                           options={"maxiter": 200, "ftol": 1e-9})
            w = res.x if res.success else w0
        except Exception:
            w = w0
        w = np.clip(w, 0, 1)
        w /= w.sum() + 1e-12
        weights[t] = w

    return _portfolio_returns(log_returns, weights)


def hmm_only(log_returns: np.ndarray, regime_posteriors: np.ndarray) -> np.ndarray:
    """Allocate based on regime: Bull→equity-heavy, Bear→bond-heavy, Volatile→balanced."""
    T, N = log_returns.shape
    weights = np.zeros((T, N))

    for t in range(T):
        p_bull, p_bear, p_vol = regime_posteriors[t]
        w = np.zeros(N)
        eq_w   = p_bull * 0.90 + p_vol * 0.50 + p_bear * 0.20
        bond_w = p_bear * 0.70 + p_vol * 0.30 + p_bull * 0.10
        other  = 1.0 - eq_w - bond_w
        if len(EQUITY_IDX) > 0:
            w[EQUITY_IDX] = eq_w / len(EQUITY_IDX)
        if len(BOND_IDX) > 0:
            w[BOND_IDX]   = bond_w / len(BOND_IDX)
        other_idx = [i for i in range(N) if i not in EQUITY_IDX + BOND_IDX]
        if other_idx:
            w[other_idx] = max(other, 0.0) / len(other_idx)
        w = np.clip(w, 0, 1)
        w /= w.sum() + 1e-12
        weights[t] = w

    return _portfolio_returns(log_returns, weights)


def run_backtest(data: dict, hmm_model, bma_engine,
                 ppo_model=None, bma_result: dict = None) -> dict:
    X     = data["X_test"]        # z-scored features for HMM/BMA inference
    macro = data["macro_test"]
    # Use raw log returns for performance metrics; fall back to X if unavailable
    R = data.get("ret_test", X)

    regime_post = hmm_model.predict_proba(X, macro)
    if bma_result is None:
        bma_result = bma_engine.predict(X)
    bma_w = bma_result["bma_weights"]

    results = {}

    results["Equal Weight"] = compute_metrics(equal_weight(R),        "Equal Weight")
    results["60/40"]        = compute_metrics(sixty_forty(R),         "60/40")
    results["Buy&Hold SPY"] = compute_metrics(buy_hold_spy(R),        "Buy&Hold SPY")
    results["Markowitz MVO"]= compute_metrics(markowitz_mvo(R),       "Markowitz MVO")
    results["HMM-Only"]     = compute_metrics(hmm_only(R, regime_post),"HMM-Only")
    results["BMA (no PPO)"] = compute_metrics(_portfolio_returns(R, bma_w), "BMA (no PPO)")

    if ppo_model is not None:
        from models.ppo_agent import run_inference
        ppo_out = run_inference(ppo_model, data, hmm_model, bma_engine, split="test")
        results["Macro-HMM-BMA"] = compute_metrics(ppo_out["returns"], "Macro-HMM-BMA")
        results["_ppo_weights"]  = ppo_out["weights"]
    else:
        results["Macro-HMM-BMA"] = compute_metrics(_portfolio_returns(R, bma_w), "Macro-HMM-BMA")

    results["_turnover"] = {
        "BMA (no PPO)":  _avg_turnover(bma_w),
        "Macro-HMM-BMA": _avg_turnover(results.get("_ppo_weights", bma_w)),
    }
    results["_bma_weights"]      = bma_w
    results["_regime_posteriors"]= regime_post
    results["_bma_posteriors"]   = bma_result["model_posteriors"]
    results["_X_test"]           = X
    results["_dates_test"]       = data["dates_test"]

    return results


def generate_ablation_study(results: dict) -> pd.DataFrame:
    strategies = [
        "Equal Weight", "60/40", "Buy&Hold SPY",
        "Markowitz MVO", "HMM-Only", "BMA (no PPO)", "Macro-HMM-BMA",
    ]
    rows = []
    for s in strategies:
        if s in results and isinstance(results[s], dict):
            rows.append(results[s])
    return pd.DataFrame(rows).set_index("Strategy") if rows else pd.DataFrame()
