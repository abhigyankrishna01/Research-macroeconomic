"""
PortfolioEnv — Gymnasium environment for portfolio allocation.

State (18-dim):
  [0:8]  log-returns of 8 assets (current step)
  [8:12] macro features (4 indicators)
  [12:15] regime posteriors from HMM [P_Bull, P_Bear, P_Volatile]
  [15:18] BMA model posteriors [P_Momentum, P_MeanReversion, P_LowVol]

Action: 8-dim continuous → softmax → portfolio weights

Reward: rolling Sharpe ratio (PPO_REWARD_WINDOW days) minus turnover penalty
"""
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

STATE_DIM  = config.N_ASSETS + 4 + config.HMM_N_STATES + 3   # 8+4+3+3 = 18
ACTION_DIM = config.N_ASSETS


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, log_returns, macro, regime_posteriors, bma_posteriors):
        super().__init__()
        self.log_returns       = np.asarray(log_returns,       dtype=float)
        self.macro             = np.asarray(macro,             dtype=float)
        self.regime_posteriors = np.asarray(regime_posteriors, dtype=float)
        self.bma_posteriors    = np.asarray(bma_posteriors,    dtype=float)

        T = len(self.log_returns)
        assert len(self.macro) == T
        assert len(self.regime_posteriors) == T
        assert len(self.bma_posteriors) == T

        self.T = T
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

        self._t          = 0
        self._weights    = np.ones(ACTION_DIM) / ACTION_DIM
        self._ret_buffer = []

    def _get_obs(self) -> np.ndarray:
        t = min(self._t, self.T - 1)
        obs = np.concatenate([
            self.log_returns[t],
            self.macro[t],
            self.regime_posteriors[t],
            self.bma_posteriors[t],
        ]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t          = 0
        self._weights    = np.ones(ACTION_DIM) / ACTION_DIM
        self._ret_buffer = []
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=float)
        # softmax to get valid portfolio weights
        action = action - action.max()
        exp_a  = np.exp(action)
        new_weights = exp_a / (exp_a.sum() + 1e-12)

        turnover = np.abs(new_weights - self._weights).sum()
        self._weights = new_weights

        t = min(self._t, self.T - 1)
        port_ret = float(np.dot(self.log_returns[t], self._weights))
        port_ret -= config.PPO_TRANSACTION_COST * turnover

        self._ret_buffer.append(port_ret)
        W = config.PPO_REWARD_WINDOW
        if len(self._ret_buffer) >= W:
            window  = np.array(self._ret_buffer[-W:])
            mean_r  = window.mean()
            std_r   = window.std() + 1e-8
            reward  = float(mean_r / std_r * np.sqrt(config.TRADING_DAYS))
        else:
            reward = port_ret

        self._t += 1
        done = self._t >= self.T
        obs  = self._get_obs()
        return obs, reward, done, False, {"port_return": port_ret}

    def render(self):
        pass
