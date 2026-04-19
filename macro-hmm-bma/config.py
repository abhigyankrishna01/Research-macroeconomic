"""
Central configuration — all hyperparameters and runtime flags in one place.
Change settings here; nothing else needs to be edited for a standard run.
"""
import os

# ── Asset universe ──────────────────────────────────────────────────────────
ASSETS   = ["SPY", "QQQ", "TLT", "IEF", "GLD", "XLF", "XLV", "XLK"]
N_ASSETS = len(ASSETS)

# ── Date ranges ─────────────────────────────────────────────────────────────
START_DATE = "2010-01-01"
END_DATE   = "2025-12-31"
TRAIN_END  = "2019-12-31"
VAL_END    = "2022-12-31"
TEST_START = "2023-01-01"

# ── FRED macro series ────────────────────────────────────────────────────────
FRED_SERIES = {
    "GDP":      "GDPC1",
    "CPI":      "CPIAUCSL",
    "FEDFUNDS": "FEDFUNDS",
    "UNRATE":   "UNRATE",
}
MACRO_COLS   = list(FRED_SERIES.keys())   # ["GDP", "CPI", "FEDFUNDS", "UNRATE"]
N_MACRO      = len(MACRO_COLS)
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# ── Data paths ───────────────────────────────────────────────────────────────
DATA_DIR      = os.path.join(os.path.dirname(__file__), "data")
RAW_DIR       = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SAMPLE_DIR    = os.path.join(DATA_DIR, "sample")       # pre-generated fallback CSVs

# ── Runtime flags ────────────────────────────────────────────────────────────
# Set USE_PRETRAINED=True to skip PPO training entirely.
# • If a saved model exists in PPO_MODEL_DIR it will be loaded.
# • If no saved model exists the BMA weights are used as final output
#   so the dashboard still works without any RL training.
USE_PRETRAINED = True

# Set USE_SAMPLE_DATA=True to force the sample CSVs even when live data exists.
USE_SAMPLE_DATA = False

# ── HMM ──────────────────────────────────────────────────────────────────────
HMM_N_STATES     = 3          # Bull · Bear · Volatile
HMM_N_ITER       = 200        # EM iterations
HMM_COV_TYPE     = "diag"     # 'diag' is numerically stable; 'full' can go non-PD
HMM_MIN_COVAR    = 1e-3       # floor on diagonal covariance to prevent degenerate states
HMM_REGIME_NAMES = ["Bull", "Bear", "Volatile"]
HMM_RANDOM_STATE = 42

# ── BMA ──────────────────────────────────────────────────────────────────────
BMA_WINDOW = 20     # rolling window (days) for MSE evaluation
BMA_LAMBDA = 10.0   # sharpness: higher → winner-takes-all; lower → uniform mix

# ── Strategy windows ─────────────────────────────────────────────────────────
MOMENTUM_WINDOW  = 20
MEAN_REV_WINDOW  = 20
LOW_VOL_WINDOW   = 20

# ── PPO environment ──────────────────────────────────────────────────────────
PPO_TRANSACTION_COST = 0.001   # one-way transaction cost fraction
PPO_REWARD_WINDOW    = 20      # rolling window for Sharpe reward

# ── PPO agent ────────────────────────────────────────────────────────────────
PPO_LEARNING_RATE   = 3e-4
PPO_N_STEPS         = 2048
PPO_BATCH_SIZE      = 64
PPO_N_EPOCHS        = 10
PPO_GAMMA           = 0.99
PPO_GAE_LAMBDA      = 0.95
PPO_CLIP_RANGE      = 0.2
PPO_TOTAL_TIMESTEPS = 500_000
PPO_POLICY_KWARGS   = dict(net_arch=[256, 256, 256])
PPO_MODEL_DIR       = os.path.join(os.path.dirname(__file__), "models", "saved")

# ── Rolling normalisation window ────────────────────────────────────────────
ROLLING_NORM_WINDOW = 252

# ── Backtest ─────────────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.02    # annualised risk-free rate used in Sharpe / Sortino
TRADING_DAYS   = 252
