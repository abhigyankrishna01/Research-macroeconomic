"""
Three portfolio strategies that compete inside the BMA engine.
Each strategy receives log-returns and outputs a weight vector over N_ASSETS.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


class MomentumStrategy:
    """Overweight assets with the strongest recent returns."""

    def predict(self, log_returns: np.ndarray) -> np.ndarray:
        """
        log_returns: (T, N_ASSETS)
        Returns: (T, N_ASSETS) weight matrix
        """
        T, N = log_returns.shape
        weights = np.zeros((T, N))
        for t in range(T):
            start = max(0, t - config.MOMENTUM_WINDOW + 1)
            window = log_returns[start:t + 1]
            cum_ret = window.sum(axis=0)
            weights[t] = _softmax(cum_ret)
        return weights


class MeanReversionStrategy:
    """Overweight assets that have underperformed recently (contrarian)."""

    def predict(self, log_returns: np.ndarray) -> np.ndarray:
        T, N = log_returns.shape
        weights = np.zeros((T, N))
        for t in range(T):
            start = max(0, t - config.MEAN_REV_WINDOW + 1)
            window = log_returns[start:t + 1]
            cum_ret = window.sum(axis=0)
            weights[t] = _softmax(-cum_ret)
        return weights


class LowVolatilityStrategy:
    """Overweight low-volatility assets (inverse volatility weighting)."""

    def predict(self, log_returns: np.ndarray) -> np.ndarray:
        T, N = log_returns.shape
        weights = np.zeros((T, N))
        for t in range(T):
            start = max(0, t - config.LOW_VOL_WINDOW + 1)
            window = log_returns[start:t + 1]
            vols = window.std(axis=0) + 1e-8
            inv_vol = 1.0 / vols
            weights[t] = inv_vol / inv_vol.sum()
        return weights
