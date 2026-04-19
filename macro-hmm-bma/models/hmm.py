"""
HMM regime detection.
BaseHMM        — wraps GaussianHMM and resolves Bull/Bear/Volatile semantics post-fit.
MacroConditionedHMM — adds a logistic regression layer that blends HMM posteriors
                      with macroeconomic features (alpha=0.7 HMM, 0.3 macro).
"""
import os
import sys
import warnings
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

warnings.filterwarnings("ignore", category=RuntimeWarning)


class BaseHMM:
    def __init__(self):
        self._model = GaussianHMM(
            n_components=config.HMM_N_STATES,
            covariance_type=config.HMM_COV_TYPE,
            n_iter=config.HMM_N_ITER,
            min_covar=config.HMM_MIN_COVAR,
            random_state=config.HMM_RANDOM_STATE,
        )
        self._state_map = None   # maps raw HMM state → Bull/Bear/Volatile index
        self.is_fitted  = False

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self._model.fit(X)

        # Resolve state semantics by mean return of each state
        means = self._model.means_.mean(axis=1)   # avg return across assets
        order = np.argsort(means)[::-1]           # high → low

        self._state_map = {}
        labels = ["Bull", "Bear", "Volatile"]
        # highest mean → Bull, lowest mean → Bear, middle → Volatile
        sorted_by_mean = np.argsort(means)[::-1]
        vols = [np.mean(np.diag(self._model.covars_[s]) if config.HMM_COV_TYPE == "full"
                        else self._model.covars_[s]) for s in range(config.HMM_N_STATES)]
        # Bull: highest mean, Bear: lowest mean, Volatile: highest variance of remainder
        bull_state = sorted_by_mean[0]
        bear_state = sorted_by_mean[-1]
        vol_states = [s for s in range(config.HMM_N_STATES) if s not in (bull_state, bear_state)]
        volatile_state = vol_states[0] if vol_states else sorted_by_mean[1]

        self._state_map = {bull_state: 0, bear_state: 1, volatile_state: 2}
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        log_post = self._model.predict_proba(X)          # shape (T, n_states)
        # rearrange columns: col 0 = Bull, 1 = Bear, 2 = Volatile
        out = np.zeros((len(X), config.HMM_N_STATES))
        for raw_state, mapped in self._state_map.items():
            out[:, mapped] = log_post[:, raw_state]
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


class MacroConditionedHMM:
    """
    Blends HMM posteriors with a logistic regression model trained on macro features.
    Final posterior = alpha * hmm_posterior + (1-alpha) * macro_posterior
    """
    ALPHA = 0.7   # weight given to HMM signal

    def __init__(self):
        self._hmm    = BaseHMM()
        self._lr     = LogisticRegression(max_iter=1000, random_state=42)
        self._scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, macro: np.ndarray):
        X     = np.asarray(X, dtype=float)
        macro = np.asarray(macro, dtype=float)
        X     = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        macro = np.nan_to_num(macro, nan=0.0, posinf=0.0, neginf=0.0)

        self._hmm.fit(X)
        hmm_states = self._hmm.predict(X)

        macro_scaled = self._scaler.fit_transform(macro)
        self._lr.fit(macro_scaled, hmm_states)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray, macro: np.ndarray) -> np.ndarray:
        X     = np.asarray(X, dtype=float)
        macro = np.asarray(macro, dtype=float)
        X     = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        macro = np.nan_to_num(macro, nan=0.0, posinf=0.0, neginf=0.0)

        hmm_post  = self._hmm.predict_proba(X)                  # (T, 3)
        macro_scaled = self._scaler.transform(macro)
        macro_post   = self._lr.predict_proba(macro_scaled)      # (T, k) k<=3
        lr_classes   = self._lr.classes_

        # Always build (T, 3) regardless of how many classes LR saw during training
        macro_reord = np.zeros((len(macro_post), config.HMM_N_STATES))
        for i, cls in enumerate(lr_classes):
            col = int(cls)
            if 0 <= col < config.HMM_N_STATES and i < macro_post.shape[1]:
                macro_reord[:, col] = macro_post[:, i]

        blended = self.ALPHA * hmm_post + (1 - self.ALPHA) * macro_reord
        # Normalise rows to sum to 1
        row_sums = blended.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return blended / row_sums

    def predict(self, X: np.ndarray, macro: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X, macro), axis=1)
