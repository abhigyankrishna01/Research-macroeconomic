"""
PPO agent training and inference.

USE_PRETRAINED = True  (default):
  - Tries to load best_ppo_model.zip from PPO_MODEL_DIR
  - Falls back to final_ppo_model.zip
  - Falls back to BMA weights if no saved model exists
  - Dashboard always works without training

USE_PRETRAINED = False:
  - Trains PPO for PPO_TOTAL_TIMESTEPS steps
  - Saves best checkpoint and final model to PPO_MODEL_DIR
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

from models.ppo_env import PortfolioEnv


class SharpeCheckpointCallback:
    def __init__(self, save_path: str, check_freq: int = 2048):
        self.save_path   = save_path
        self.check_freq  = check_freq
        self.best_sharpe = -np.inf
        self.n_calls     = 0

    def __call__(self, locals_: dict, globals_: dict) -> bool:
        self.n_calls += 1
        if self.n_calls % self.check_freq == 0:
            ep_rets = locals_.get("ep_info_buffer", [])
            if ep_rets:
                rets = [ep["r"] for ep in ep_rets]
                mean_r = np.mean(rets)
                std_r  = np.std(rets) + 1e-8
                sharpe = mean_r / std_r * np.sqrt(config.TRADING_DAYS)
                if sharpe > self.best_sharpe:
                    self.best_sharpe = sharpe
                    best_path = os.path.join(self.save_path, "best_ppo_model")
                    locals_["self"].save(best_path)
        return True


def train_ppo(data: dict, hmm_model, bma_engine, total_timesteps: int = None):
    from stable_baselines3 import PPO

    if total_timesteps is None:
        total_timesteps = config.PPO_TOTAL_TIMESTEPS

    os.makedirs(config.PPO_MODEL_DIR, exist_ok=True)

    bma_result = bma_engine.predict(data["X_train"])
    regime_post = hmm_model.predict_proba(data["X_train"], data["macro_train"])
    bma_post    = bma_result["model_posteriors"]

    env = PortfolioEnv(
        log_returns       = data["X_train"],
        macro             = data["macro_train"],
        regime_posteriors = regime_post,
        bma_posteriors    = bma_post,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate   = config.PPO_LEARNING_RATE,
        n_steps         = config.PPO_N_STEPS,
        batch_size      = config.PPO_BATCH_SIZE,
        n_epochs        = config.PPO_N_EPOCHS,
        gamma           = config.PPO_GAMMA,
        gae_lambda      = config.PPO_GAE_LAMBDA,
        clip_range      = config.PPO_CLIP_RANGE,
        policy_kwargs   = config.PPO_POLICY_KWARGS,
        verbose         = 1,
    )

    callback = SharpeCheckpointCallback(save_path=config.PPO_MODEL_DIR)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    final_path = os.path.join(config.PPO_MODEL_DIR, "final_ppo_model")
    model.save(final_path)
    print(f"[ppo_agent] Saved final model → {final_path}.zip")
    return model


def load_ppo():
    from stable_baselines3 import PPO
    for fname in ("best_ppo_model.zip", "final_ppo_model.zip"):
        path = os.path.join(config.PPO_MODEL_DIR, fname)
        if os.path.exists(path):
            print(f"[ppo_agent] Loading saved model: {path}")
            return PPO.load(path)
    print("[ppo_agent] No saved model found.")
    return None


def run_inference(model, data: dict, hmm_model, bma_engine, split: str = "test") -> dict:
    X     = data[f"X_{split}"]
    macro = data[f"macro_{split}"]

    regime_post = hmm_model.predict_proba(X, macro)
    bma_result  = bma_engine.predict(X)
    bma_post    = bma_result["model_posteriors"]

    env = PortfolioEnv(
        log_returns       = X,
        macro             = macro,
        regime_posteriors = regime_post,
        bma_posteriors    = bma_post,
    )

    obs, _ = env.reset()
    weights_list = []
    returns_list = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        weights_list.append(env._weights.copy())
        returns_list.append(info["port_return"])
        if truncated:
            break

    weights = np.array(weights_list)
    returns = np.array(returns_list)
    return {"weights": weights, "returns": returns}
