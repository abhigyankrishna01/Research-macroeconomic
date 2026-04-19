# Macroeconomic-Conditioned Regime-Aware Portfolio Management

**Integrating Hidden Markov Models with Bayesian Model Averaging for Dynamic Asset Allocation**

Final Year Project — B.Tech / MSc 2025

---

## Overview

A quantitative portfolio management system that detects macroeconomic market regimes using Hidden Markov Models, selects the best investment strategy per regime via Bayesian Model Averaging, and learns an optimal asset allocation policy using Proximal Policy Optimisation (PPO) reinforcement learning.

The full pipeline is exposed through an interactive Streamlit dashboard with dark theme, 6 analytical pages, and Plotly charts — works immediately using pre-generated sample data, no API keys required.

---

## Architecture

```
Price data (yfinance / sample CSV)
Macro data (FRED / synthetic)
         │
         ▼
  data/preprocess.py
  log-returns · rolling z-score (252d) · train/val/test splits
         │
         ├──► models/hmm.py  →  regime posteriors [P(Bull), P(Bear), P(Volatile)]
         │     GaussianHMM (diag, 3 states) + logistic macro blend layer (α=0.7)
         │
         ├──► models/strategies.py  →  3 weight vectors per day
         │     Momentum / MeanReversion / LowVolatility
         │
         ├──► models/bma.py  →  BMA weights + model posteriors
         │     Rolling H=20, λ=10, softmax(−λ·MSE)
         │
         └──► models/ppo_env.py + ppo_agent.py  →  final allocation weights
               State: 8 returns + 4 macro + 3 regime + 3 BMA = 18-dim
               Reward: rolling Sharpe − 0.001 × turnover
               USE_PRETRAINED=True → skip training, fall back to BMA
         │
         ▼
  backtest/backtester.py  →  7-strategy walk-forward comparison
  backtest/metrics.py     →  Sharpe, Sortino, MDD, Calmar, CAGR
         │
         ▼
  app.py  →  Streamlit dashboard (6 pages, dark theme, Plotly)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Regime Detection | hmmlearn 0.3.3 — GaussianHMM (diagonal covariance, 3 states) |
| Macro Conditioning | scikit-learn LogisticRegression blend layer |
| Strategy Selection | Bayesian Model Averaging (rolling MSE, softmax posteriors) |
| RL Agent | stable-baselines3 2.8.0 — PPO, MlpPolicy, 3×256 hidden layers |
| RL Environment | Gymnasium 1.2.3 — custom PortfolioEnv |
| Dashboard | Streamlit 1.56.0 + Plotly 6.7.0 |
| Data | yfinance 1.3.0 + FRED API + GBM synthetic fallback |
| Numerics | NumPy 2.2.6, Pandas 2.3.3, SciPy |
| Deep Learning | PyTorch 2.11.0 (PPO backend) |

---

## Project Structure

```
macro-hmm-bma/
├── app.py                    # Streamlit dashboard — MAIN ENTRY POINT
├── config.py                 # All hyperparameters & runtime flags
├── requirements.txt
├── README.md
├── data/
│   ├── fetch_data.py         # yfinance + FRED download; synthetic fallback
│   ├── preprocess.py         # log-returns, z-score, train/val/test splits
│   ├── generate_sample.py    # regenerates data/sample/ CSVs (run once)
│   ├── raw/                  # downloaded raw CSVs
│   ├── processed/            # preprocessed CSVs (populated after fetch)
│   └── sample/
│       ├── prices.csv        # 4173 trading days × 8 ETFs (GBM synthetic)
│       └── macro.csv         # 4173 days × 4 macro indicators
├── models/
│   ├── hmm.py                # BaseHMM + MacroConditionedHMM
│   ├── strategies.py         # Momentum, MeanReversion, LowVolatility
│   ├── bma.py                # BMAEngine — rolling MSE, softmax posteriors
│   ├── ppo_env.py            # PortfolioEnv (Gymnasium, 18-dim state)
│   ├── ppo_agent.py          # PPO training + inference
│   └── saved/                # PPO model checkpoints
└── backtest/
    ├── backtester.py         # Walk-forward engine, 7 strategy baselines
    └── metrics.py            # Performance metrics (hardened against NaN/Inf)
```

---

## Assets & Data

**ETFs traded:** SPY, QQQ, TLT, IEF, GLD, XLF, XLV, XLK

**FRED macro series:** GDP (GDPC1), CPI (CPIAUCSL), Fed Funds Rate (FEDFUNDS), Unemployment (UNRATE)

**Train / Val / Test splits:**
- Train: 2010-01-01 → 2019-12-31
- Validation: 2020-01-01 → 2022-12-31
- Test: 2023-01-01 → 2025-12-31

**3-tier data fallback (always works):**
1. `data/processed/` — live data previously downloaded & preprocessed
2. `data/sample/` — pre-generated realistic synthetic CSVs (4173 rows, always present)
3. Pure numpy GBM — last resort, no files needed

Sample data includes injected regimes: COVID crash (Feb–Apr 2020), 2022 bear market, 2024 AI bull run.

---

## Key Config Flags (`config.py`)

| Flag | Default | Meaning |
|---|---|---|
| `USE_PRETRAINED` | `True` | Load saved PPO model or fall back to BMA weights |
| `USE_SAMPLE_DATA` | `False` | Force sample CSVs even if processed data exists |
| `BMA_LAMBDA` | `10.0` | BMA sharpness (higher = winner-takes-all) |
| `HMM_N_STATES` | `3` | Bull / Bear / Volatile |
| `HMM_COV_TYPE` | `"diag"` | Diagonal covariance (numerically stable) |
| `HMM_MIN_COVAR` | `1e-3` | Variance floor — prevents degenerate HMM states |
| `PPO_TOTAL_TIMESTEPS` | `500_000` | PPO training budget (~30 min on CPU) |
| `ROLLING_NORM_WINDOW` | `252` | Z-score window (no look-ahead bias) |

---

## Dashboard Pages

| Page | What you see |
|---|---|
| Overview | KPI cards (Sharpe / CAGR / MDD / Return / Regime) + cumulative returns + regime timeline |
| Regime Detection | Stacked posteriors + pie chart + SPY price with colour-coded regime bands |
| BMA Strategies | Model posteriors over time + current weight bar + equity/bond allocation |
| Performance | Metrics table vs all 7 baselines + Sharpe/CAGR bar charts + drawdown profiles |
| Macro Dashboard | GDP / CPI / Fed Funds Rate / Unemployment subplots + change signals |
| Ablation Study | Sharpe by variant bar chart + rolling 60-day Sharpe comparison |

**7 baselines compared:** Equal Weight, 60/40, Buy & Hold SPY, Markowitz MVO, Standard PPO, HMM-Only, Macro-HMM-BMA (full system)

---

## Commands Reference

### Step 0 — Navigate to project folder

```powershell
cd C:\Users\lagja\macroproject\macro-hmm-bma
```

### Step 1 — Create & activate virtual environment

```powershell
python -m venv venv

# PowerShell
.\venv\Scripts\Activate.ps1

# CMD
venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Step 2 — Launch the dashboard

```powershell
streamlit run app.py
```

Opens at **http://localhost:8501** — uses sample data immediately, no setup needed.

### Step 3 — (Optional) Download live market data

Requires a free FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html

```powershell
$env:FRED_API_KEY = "your_key_here"
python data/fetch_data.py
python data/preprocess.py
# Refresh the Streamlit page — dashboard switches to LIVE badge
```

### Step 4 — (Optional) Regenerate sample CSVs

```powershell
python data/generate_sample.py
```

### Step 5 — (Optional) Train PPO agent (~30 min)

```powershell
python - <<'EOF'
import sys; sys.path.insert(0, '.')
from data.preprocess import run_preprocessing
from models.hmm import MacroConditionedHMM
from models.bma import BMAEngine
from models.ppo_agent import train_ppo
import config

config.USE_PRETRAINED = False
data = run_preprocessing()
hmm = MacroConditionedHMM(); hmm.fit(data['X_train'], data['macro_train'])
bma = BMAEngine(); bma.fit(data['X_train'])
train_ppo(data, hmm, bma, total_timesteps=config.PPO_TOTAL_TIMESTEPS)
print("Training complete — saved to models/saved/")
EOF
```

After training, set `USE_PRETRAINED = True` in `config.py`. The saved model loads automatically on next run.

### Utility commands

```powershell
# Deactivate virtual environment
deactivate

# Clear Streamlit cache (if dashboard shows stale data)
python -c "import streamlit as st; st.cache_data.clear()"

# Smoke test — verify all imports work
python -c "import hmmlearn, stable_baselines3, gymnasium, streamlit, yfinance, plotly; print('All imports OK')"
```

---

## Academic Motivation

Standard portfolio optimisers treat all market conditions as identical. This project argues that:

1. **Market regimes are latent** — HMMs can detect Bull/Bear/Volatile states from price patterns
2. **Macro drives regimes** — GDP, CPI, interest rates, and unemployment shift regime probabilities; the logistic blend layer captures this
3. **No single strategy dominates all regimes** — BMA dynamically re-weights Momentum, Mean Reversion, and Low Volatility based on recent performance
4. **RL learns the optimal blend** — PPO integrates regime and strategy signals into a single allocation policy, optimising for risk-adjusted returns

The ablation study on the Performance page quantifies each layer's contribution to the final Sharpe ratio.

---

## Team

| Name | Contribution |
|---|---|
| Developer | Full system design, implementation, and evaluation |
| Supervisor | Academic guidance and literature review |

---

## License

For academic and educational use only.
