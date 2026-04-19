"""
Bayesian Model Averaging engine.
Evaluates three strategies (Momentum, MeanReversion, LowVolatility) over a rolling
window of H days, computes MSE-based likelihoods, and produces softmax posteriors.
"""
import os
import sys
import numpy as np
from scipy.special import softmax

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from models.strategies import MomentumStrategy, MeanReversionStrategy, LowVolatilityStrategy


class BMAEngine:
    def __init__(self):
        self._strategies = [
            MomentumStrategy(),
            MeanReversionStrategy(),
            LowVolatilityStrategy(),
        ]
        self.strategy_names  = ["Momentum", "MeanReversion", "LowVolatility"]
        self._strategy_weights = None   # (T, N_ASSETS) per strategy
        self.bma_weights       = None   # (T, N_ASSETS) final BMA portfolio
        self.model_posteriors  = None   # (T, 3) posterior per strategy
        self.uncertainty       = None   # (T,) entropy

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        all_weights = []
        for strat in self._strategies:
            w = strat.predict(X)
            all_weights.append(w)
        self._strategy_weights = all_weights
        self._X_fit = X
        return self

    def predict(self, X: np.ndarray = None) -> dict:
        if X is None:
            X = self._X_fit
        X = np.asarray(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        T, N = X.shape
        H = config.BMA_WINDOW
        lam = config.BMA_LAMBDA

        all_weights = []
        for strat in self._strategies:
            w = strat.predict(X)
            all_weights.append(w)

        posteriors = np.zeros((T, len(self._strategies)))
        bma_w      = np.zeros((T, N))

        for t in range(T):
            start = max(0, t - H)
            mses = []
            for w in all_weights:
                # realized portfolio return in window
                port_ret = (X[start:t + 1] * w[start:t + 1]).sum(axis=1)
                # benchmark: equal weight
                eq_ret   = X[start:t + 1].mean(axis=1)
                mse = np.mean((port_ret - eq_ret) ** 2) if len(port_ret) > 0 else 1.0
                mses.append(mse)
            mses = np.array(mses)
            post = softmax(-lam * mses)
            posteriors[t] = post
            bma_w[t] = sum(post[i] * all_weights[i][t] for i in range(len(self._strategies)))

        # Entropy as uncertainty measure
        eps = 1e-12
        entropy = -(posteriors * np.log(posteriors + eps)).sum(axis=1)

        self.bma_weights      = bma_w
        self.model_posteriors = posteriors
        self.uncertainty      = entropy

        return {
            "bma_weights":      bma_w,
            "model_posteriors": posteriors,
            "uncertainty":      entropy,
            "strategy_weights": all_weights,
        }
