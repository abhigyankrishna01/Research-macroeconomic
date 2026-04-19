"""
Streamlit dashboard for Macro-HMM-BMA portfolio management.
6 pages: Overview · Regime Detection · BMA Strategies · Performance · Macro · Ablation
3-tier data fallback: processed/ → sample/ → numpy synthetic
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
import config

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Macro-HMM-BMA Portfolio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
body, .stApp { background-color: #0e1117; color: #fafafa; }
.kpi-card {
    background: linear-gradient(135deg, #1e2130, #252a3a);
    border: 1px solid #2e3450;
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
    margin-bottom: 8px;
}
.kpi-label { font-size: 12px; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; }
.kpi-value { font-size: 28px; font-weight: 700; color: #64ffda; }
.kpi-sub   { font-size: 11px; color: #6272a4; margin-top: 4px; }
.regime-bull     { background:#0d3321; border:1px solid #2ecc71; border-radius:6px; padding:6px 14px; color:#2ecc71; font-weight:600; }
.regime-bear     { background:#3d0d0d; border:1px solid #e74c3c; border-radius:6px; padding:6px 14px; color:#e74c3c; font-weight:600; }
.regime-volatile { background:#2d2700; border:1px solid #f39c12; border-radius:6px; padding:6px 14px; color:#f39c12; font-weight:600; }
.section-header  { font-size:20px; font-weight:700; color:#cdd6f4; border-bottom:1px solid #313244; padding-bottom:8px; margin-bottom:16px; }
.data-badge-live    { background:#0d3321; color:#2ecc71; border:1px solid #2ecc71; border-radius:4px; padding:2px 10px; font-size:11px; }
.data-badge-sample  { background:#2d2700; color:#f39c12; border:1px solid #f39c12; border-radius:4px; padding:2px 10px; font-size:11px; }
.data-badge-synth   { background:#1a1a2e; color:#8892b0; border:1px solid #6272a4; border-radius:4px; padding:2px 10px; font-size:11px; }
</style>
""", unsafe_allow_html=True)

REGIME_COLORS = ["#2ecc71", "#e74c3c", "#f39c12"]
REGIME_NAMES  = ["Bull", "Bear", "Volatile"]


def _rgba(hex_color: str, alpha: float = 0.6) -> str:
    """Convert #rrggbb to rgba(r,g,b,a) for Plotly fillcolor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Data loading ─────────────────────────────────────────────────────────────

def _numpy_fallback() -> dict:
    """Tier 3: pure numpy GBM data, no files needed."""
    from data.preprocess import _rolling_zscore
    np.random.seed(42)
    n = 3000
    dates = pd.bdate_range("2010-01-04", periods=n)
    prices = np.ones((n, config.N_ASSETS)) * 100.0
    for i in range(1, n):
        shock = np.random.normal(0.0004, 0.012, config.N_ASSETS)
        prices[i] = prices[i - 1] * np.exp(shock)
    log_ret_raw = np.diff(np.log(prices), axis=0)
    dates_r = dates[1:]

    macro_raw = np.column_stack([
        np.cumsum(np.random.normal(0.005, 0.012, n - 1)),
        np.cumsum(np.random.normal(0.002, 0.004, n - 1)),
        np.clip(2.5 + 2.5 * np.sin(np.linspace(0, 4 * np.pi, n - 1)), 0, 6),
        np.clip(5 + np.cumsum(np.random.normal(0, 0.05, n - 1)), 3, 12),
    ])

    lr_df = pd.DataFrame(log_ret_raw, index=dates_r, columns=config.ASSETS)
    mc_df = pd.DataFrame(macro_raw,   index=dates_r, columns=["GDP","CPI","FEDFUNDS","UNRATE"])
    lr_norm = _rolling_zscore(lr_df, config.ROLLING_NORM_WINDOW).values
    mc_norm = _rolling_zscore(mc_df, config.ROLLING_NORM_WINDOW).values

    train_mask = dates_r <= config.TRAIN_END
    val_mask   = (dates_r > config.TRAIN_END) & (dates_r <= config.VAL_END)
    test_mask  = dates_r > config.VAL_END

    return {
        "X_train": lr_norm[train_mask], "X_val": lr_norm[val_mask], "X_test": lr_norm[test_mask],
        "ret_train": log_ret_raw[train_mask], "ret_val": log_ret_raw[val_mask], "ret_test": log_ret_raw[test_mask],
        "macro_train": mc_norm[train_mask], "macro_val": mc_norm[val_mask], "macro_test": mc_norm[test_mask],
        "dates_train": dates_r[train_mask], "dates_val": dates_r[val_mask], "dates_test": dates_r[test_mask],
        "log_returns": lr_df,
        "macro_daily_raw": mc_df,
        "_source": "SYNTHETIC",
    }


def _process_from_sample() -> dict:
    """Tier 2: load sample CSVs and preprocess in-memory."""
    from data.preprocess import compute_log_returns, interpolate_macro, _rolling_zscore, split_data
    prices_path = os.path.join(config.SAMPLE_DIR, "prices.csv")
    macro_path  = os.path.join(config.SAMPLE_DIR, "macro.csv")
    if not os.path.exists(prices_path) or not os.path.exists(macro_path):
        return None
    prices    = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    macro_raw = pd.read_csv(macro_path,  index_col=0, parse_dates=True)
    cols      = [c for c in config.ASSETS if c in prices.columns]
    prices    = prices[cols]
    log_ret   = compute_log_returns(prices)
    macro_d   = interpolate_macro(macro_raw, log_ret.index)
    log_norm  = _rolling_zscore(log_ret,  config.ROLLING_NORM_WINDOW)
    macro_n   = _rolling_zscore(macro_d,  config.ROLLING_NORM_WINDOW)
    feat      = pd.concat([log_norm, macro_n], axis=1).dropna()
    idx       = log_norm.index.intersection(feat.index)
    data      = split_data(log_norm.loc[idx], macro_n.loc[idx])
    # raw return splits for backtesting
    for split in ("train", "val", "test"):
        split_idx = data[f"dates_{split}"]
        data[f"ret_{split}"] = log_ret.reindex(split_idx).values
    data["log_returns"]     = log_ret
    data["macro_daily_raw"] = macro_d
    data["_source"]         = "SAMPLE"
    return data


@st.cache_data(show_spinner=False)
def load_processed_data() -> dict:
    """Try processed/ → sample/ → numpy fallback."""
    train_csv = os.path.join(config.PROCESSED_DIR, "log_ret_train.csv")
    if not config.USE_SAMPLE_DATA and os.path.exists(train_csv):
        try:
            from data.preprocess import split_data, _rolling_zscore, interpolate_macro
            log_norm  = pd.read_csv(os.path.join(config.PROCESSED_DIR, "log_ret_train.csv"), index_col=0, parse_dates=True)
            # load all splits
            splits = {}
            for sp in ("train", "val", "test"):
                p = os.path.join(config.PROCESSED_DIR, f"log_ret_{sp}.csv")
                if os.path.exists(p):
                    splits[sp] = pd.read_csv(p, index_col=0, parse_dates=True)
            macro_p = os.path.join(config.PROCESSED_DIR, "macro_daily.csv")
            macro_d = pd.read_csv(macro_p, index_col=0, parse_dates=True) if os.path.exists(macro_p) else None
            log_ret_raw_p = os.path.join(config.PROCESSED_DIR, "log_returns.csv")
            log_ret_raw = pd.read_csv(log_ret_raw_p, index_col=0, parse_dates=True) if os.path.exists(log_ret_raw_p) else None

            if splits.get("train") is not None and splits.get("test") is not None:
                macro_norm = _rolling_zscore(macro_d, config.ROLLING_NORM_WINDOW) if macro_d is not None else None

                def _to_arr(df, cols):
                    if df is None: return np.zeros((len(df or splits["train"]), len(cols)))
                    c = [x for x in cols if x in df.columns]
                    return df[c].values if c else np.zeros((len(df), len(cols)))

                macro_cols = ["GDP","CPI","FEDFUNDS","UNRATE"]
                data = {
                    "X_train": splits["train"].values,
                    "X_val":   splits.get("val", splits["train"]).values,
                    "X_test":  splits["test"].values,
                    "macro_train": _to_arr(macro_norm.loc[splits["train"].index] if macro_norm is not None else None, macro_cols),
                    "macro_val":   _to_arr(macro_norm.loc[splits.get("val", splits["train"]).index] if macro_norm is not None else None, macro_cols),
                    "macro_test":  _to_arr(macro_norm.loc[splits["test"].index] if macro_norm is not None else None, macro_cols),
                    "dates_train": splits["train"].index,
                    "dates_val":   splits.get("val", splits["train"]).index,
                    "dates_test":  splits["test"].index,
                    "log_returns": log_ret_raw,
                    "macro_daily_raw": macro_d,
                    "_source": "LIVE",
                }
                return data
        except Exception as e:
            st.warning(f"Could not load processed data ({e}). Falling back to sample.")

    sample = _process_from_sample()
    if sample is not None:
        return sample

    return _numpy_fallback()


# ── Pipeline ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_full_pipeline(_data_key: str) -> dict:
    data = load_processed_data()

    from models.hmm import MacroConditionedHMM
    from models.bma import BMAEngine

    with st.spinner("Fitting HMM regime model..."):
        hmm = MacroConditionedHMM()
        hmm.fit(data["X_train"], data["macro_train"])

    with st.spinner("Computing BMA strategy posteriors..."):
        bma = BMAEngine()
        bma.fit(data["X_train"])
        bma_result_train = bma.predict(data["X_train"])
        bma_result_test  = bma.predict(data["X_test"])

    ppo_model = None
    if config.USE_PRETRAINED:
        from models.ppo_agent import load_ppo
        ppo_model = load_ppo()

    with st.spinner("Running backtest..."):
        from backtest.backtester import run_backtest, generate_ablation_study
        results = run_backtest(data, hmm, bma, ppo_model=ppo_model,
                               bma_result=bma_result_test)
        ablation = generate_ablation_study(results)

    regime_all = hmm.predict_proba(
        np.vstack([data["X_train"], data["X_val"], data["X_test"]]),
        np.vstack([data["macro_train"], data["macro_val"], data["macro_test"]]),
    )
    bma_all = bma.predict(
        np.vstack([data["X_train"], data["X_val"], data["X_test"]]))

    return {
        "data": data, "hmm": hmm, "bma": bma,
        "bma_result_train": bma_result_train,
        "bma_result_test":  bma_result_test,
        "bma_all": bma_all,
        "results": results,
        "ablation": ablation,
        "regime_all": regime_all,
        "source": data.get("_source", "UNKNOWN"),
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.markdown("## 📈 Macro-HMM-BMA")
st.sidebar.markdown("*Regime-Aware Portfolio Management*")
st.sidebar.divider()

page = st.sidebar.radio("Navigate", [
    "Overview",
    "Regime Detection",
    "BMA Strategies",
    "Performance",
    "Macro Dashboard",
    "Ablation Study",
])

st.sidebar.divider()
if st.sidebar.button("🔄 Download live data"):
    st.cache_data.clear()
    st.rerun()

if st.sidebar.button("🗑 Clear cache & reload"):
    st.cache_data.clear()
    st.rerun()

# ── Load data ─────────────────────────────────────────────────────────────────

data_key = "v1"
with st.spinner("Loading data and running pipeline..."):
    try:
        pipeline = run_full_pipeline(data_key)
    except Exception as e:
        st.cache_data.clear()
        st.error(f"Pipeline error: {e}. Cleared cache — refreshing...")
        st.rerun()

data      = pipeline["data"]
results   = pipeline["results"]
ablation  = pipeline["ablation"]
regime_all= pipeline["regime_all"]
bma_all   = pipeline["bma_all"]
source    = pipeline["source"]

# Build full date index
all_dates = np.concatenate([
    data["dates_train"].values,
    data["dates_val"].values,
    data["dates_test"].values,
])
all_X = np.vstack([data["X_train"], data["X_val"], data["X_test"]])

# Source badge
badge_cls = {"LIVE": "data-badge-live", "SAMPLE": "data-badge-sample"}.get(source, "data-badge-synth")
st.sidebar.markdown(f'<span class="{badge_cls}">DATA: {source}</span>', unsafe_allow_html=True)


# ── Helper: cumulative returns chart ─────────────────────────────────────────

def _cum_ret_fig(strategies: dict, title: str = "") -> go.Figure:
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    for i, (name, rets) in enumerate(strategies.items()):
        r = np.asarray(rets, dtype=float)
        r = r[np.isfinite(r)]
        cum = np.exp(np.cumsum(r)) - 1
        fig.add_trace(go.Scatter(
            y=cum * 100, mode="lines", name=name,
            line=dict(color=palette[i % len(palette)], width=2)))
    fig.update_layout(
        title=title, template="plotly_dark",
        xaxis_title="Time Steps", yaxis_title="Cumulative Return (%)",
        legend=dict(orientation="h", y=-0.2), height=420,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown('<div class="section-header">📊 Portfolio Overview</div>', unsafe_allow_html=True)

    full_metrics = results.get("Macro-HMM-BMA", {})
    eq_metrics   = results.get("Equal Weight",   {})

    last_regime_idx = int(np.argmax(regime_all[-1]))
    regime_name     = REGIME_NAMES[last_regime_idx]
    regime_class    = ["regime-bull", "regime-bear", "regime-volatile"][last_regime_idx]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Sharpe Ratio</div>
            <div class="kpi-value">{full_metrics.get('Sharpe', 0):.2f}</div>
            <div class="kpi-sub">vs {eq_metrics.get('Sharpe', 0):.2f} Equal Weight</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">CAGR</div>
            <div class="kpi-value">{full_metrics.get('CAGR (%)', 0):.1f}%</div>
            <div class="kpi-sub">annualised</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Max Drawdown</div>
            <div class="kpi-value">{full_metrics.get('Max Drawdown', 0):.1f}%</div>
            <div class="kpi-sub">worst peak-to-trough</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Total Return</div>
            <div class="kpi-value">{full_metrics.get('Total Return', 0):.1f}%</div>
            <div class="kpi-sub">test period</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Current Regime</div>
            <div class="kpi-value"><span class="{regime_class}">{regime_name}</span></div>
            <div class="kpi-sub">HMM + Macro</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Cumulative Returns — Test Period</div>', unsafe_allow_html=True)

    test_X = data["X_test"]
    strat_rets = {}
    from backtest.backtester import equal_weight, sixty_forty, buy_hold_spy
    strat_rets["Equal Weight"]  = equal_weight(test_X)
    strat_rets["60/40"]         = sixty_forty(test_X)
    strat_rets["Buy&Hold SPY"]  = buy_hold_spy(test_X)
    bma_w = results.get("_bma_weights")
    if bma_w is not None:
        strat_rets["Macro-HMM-BMA"] = (test_X * bma_w).sum(axis=1)

    st.plotly_chart(_cum_ret_fig(strat_rets, "Cumulative Returns (%)"), use_container_width=True)

    st.markdown('<div class="section-header">Regime Timeline</div>', unsafe_allow_html=True)
    fig_reg = go.Figure()
    T_all = min(len(all_dates), len(regime_all))
    for i, name in enumerate(REGIME_NAMES):
        fig_reg.add_trace(go.Scatter(
            x=all_dates[:T_all], y=regime_all[:T_all, i],
            mode="lines", stackgroup="one", name=name,
            line=dict(width=0), fillcolor=_rgba(REGIME_COLORS[i], 0.65),
        ))
    fig_reg.update_layout(template="plotly_dark", height=200,
                          xaxis_title="Date", yaxis_title="Probability",
                          legend=dict(orientation="h"))
    st.plotly_chart(fig_reg, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Regime Detection
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Regime Detection":
    st.markdown('<div class="section-header">🔍 Regime Detection</div>', unsafe_allow_html=True)

    T_all = min(len(all_dates), len(regime_all))

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Stacked Regime Posteriors", "Regime Distribution",
                                        "SPY Price with Regime Bands", "Current Regime Posteriors"),
                        specs=[[{"type": "scatter"}, {"type": "pie"}],
                               [{"type": "scatter"}, {"type": "bar"}]])

    for i, name in enumerate(REGIME_NAMES):
        fig.add_trace(go.Scatter(
            x=all_dates[:T_all], y=regime_all[:T_all, i],
            mode="lines", stackgroup="one", name=name,
            line=dict(width=0), fillcolor=_rgba(REGIME_COLORS[i], 0.6),
        ), row=1, col=1)

    dominant = np.argmax(regime_all[:T_all], axis=1)
    counts   = [np.sum(dominant == i) for i in range(3)]
    fig.add_trace(go.Pie(
        labels=REGIME_NAMES, values=counts,
        marker_colors=REGIME_COLORS, hole=0.4,
    ), row=1, col=2)

    spy_rets = all_X[:T_all, 0]
    spy_cum  = np.exp(np.cumsum(spy_rets)) * 100
    fig.add_trace(go.Scatter(
        x=all_dates[:T_all], y=spy_cum,
        mode="lines", name="SPY", line=dict(color="#64ffda", width=1.5),
    ), row=2, col=1)

    last_post = regime_all[-1]
    fig.add_trace(go.Bar(
        x=REGIME_NAMES, y=last_post,
        marker_color=REGIME_COLORS, name="Current",
    ), row=2, col=2)

    fig.update_layout(template="plotly_dark", height=700,
                      showlegend=False, title_text="Regime Analysis")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Regime Statistics")
    stats = pd.DataFrame({
        "Regime": REGIME_NAMES,
        "Days": counts,
        "% Time": [f"{c/T_all*100:.1f}%" for c in counts],
        "Current P": [f"{p:.1%}" for p in last_post],
    })
    st.dataframe(stats, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BMA Strategies
# ══════════════════════════════════════════════════════════════════════════════
elif page == "BMA Strategies":
    st.markdown('<div class="section-header">⚖️ Bayesian Model Averaging</div>', unsafe_allow_html=True)

    bma_post  = bma_all.get("model_posteriors", np.zeros((1, 3)))
    bma_w_all = bma_all.get("bma_weights", np.zeros((1, config.N_ASSETS)))
    T_bma     = min(len(all_dates), len(bma_post))

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("BMA Model Posteriors Over Time", "Current Strategy Weights",
                                        "Portfolio Allocation (BMA)", "Strategy Uncertainty (Entropy)"))

    strategy_names = ["Momentum", "MeanReversion", "LowVolatility"]
    strat_colors   = ["#64ffda", "#ff6e6e", "#ffb86c"]

    for i, (name, col) in enumerate(zip(strategy_names, strat_colors)):
        if bma_post.shape[1] > i:
            fig.add_trace(go.Scatter(
                x=all_dates[:T_bma], y=bma_post[:T_bma, i],
                mode="lines", name=name, stackgroup="one",
                line=dict(width=0), fillcolor=_rgba(col, 0.6),
            ), row=1, col=1)

    last_bma_post = bma_post[-1] if len(bma_post) > 0 else np.ones(3) / 3
    fig.add_trace(go.Bar(
        x=strategy_names[:len(last_bma_post)],
        y=last_bma_post,
        marker_color=strat_colors[:len(last_bma_post)],
    ), row=1, col=2)

    last_w = bma_w_all[-1] if len(bma_w_all) > 0 else np.ones(config.N_ASSETS) / config.N_ASSETS
    fig.add_trace(go.Bar(
        x=config.ASSETS, y=last_w,
        marker_color="#64ffda",
    ), row=2, col=1)

    uncertainty = bma_all.get("uncertainty", np.zeros(T_bma))
    if len(uncertainty) >= T_bma:
        fig.add_trace(go.Scatter(
            x=all_dates[:T_bma], y=uncertainty[:T_bma],
            mode="lines", line=dict(color="#ff6e6e", width=1.5),
        ), row=2, col=2)

    fig.update_layout(template="plotly_dark", height=700,
                      showlegend=False, title_text="BMA Analysis")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Performance":
    st.markdown('<div class="section-header">📉 Performance Comparison</div>', unsafe_allow_html=True)

    strategy_keys = ["Equal Weight", "60/40", "Buy&Hold SPY", "Markowitz MVO",
                     "HMM-Only", "BMA (no PPO)", "Macro-HMM-BMA"]
    rows = [results[k] for k in strategy_keys if k in results and isinstance(results[k], dict)]

    if rows:
        df = pd.DataFrame(rows).set_index("Strategy")
        st.dataframe(df.style.highlight_max(
            subset=["Sharpe", "CAGR (%)", "Calmar", "Sortino", "Total Return"],
            color="#0d3321",
        ).highlight_min(
            subset=["Max Drawdown"],
            color="#0d3321",
        ), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        sharpes = {k: results[k]["Sharpe"] for k in strategy_keys if k in results and isinstance(results[k], dict)}
        fig_s = go.Figure(go.Bar(
            x=list(sharpes.keys()), y=list(sharpes.values()),
            marker_color=["#64ffda" if k == "Macro-HMM-BMA" else "#6272a4" for k in sharpes],
        ))
        fig_s.update_layout(template="plotly_dark", title="Sharpe Ratio Comparison", height=350)
        st.plotly_chart(fig_s, use_container_width=True)

    with c2:
        cagrs = {k: results[k]["CAGR (%)"] for k in strategy_keys if k in results and isinstance(results[k], dict)}
        fig_c = go.Figure(go.Bar(
            x=list(cagrs.keys()), y=list(cagrs.values()),
            marker_color=["#64ffda" if k == "Macro-HMM-BMA" else "#6272a4" for k in cagrs],
        ))
        fig_c.update_layout(template="plotly_dark", title="CAGR (%) Comparison", height=350)
        st.plotly_chart(fig_c, use_container_width=True)

    st.markdown('<div class="section-header">Drawdown Profiles</div>', unsafe_allow_html=True)
    test_X = data["X_test"]
    from backtest.backtester import equal_weight, sixty_forty, buy_hold_spy
    dd_strats = {
        "Equal Weight": equal_weight(test_X),
        "60/40": sixty_forty(test_X),
    }
    bma_w = results.get("_bma_weights")
    if bma_w is not None:
        dd_strats["Macro-HMM-BMA"] = (test_X * bma_w).sum(axis=1)

    fig_dd = go.Figure()
    palette = ["#6272a4", "#ffb86c", "#64ffda"]
    for i, (name, rets) in enumerate(dd_strats.items()):
        r   = np.asarray(rets, dtype=float)
        r   = r[np.isfinite(r)]
        cum = np.exp(np.cumsum(r))
        peak = np.maximum.accumulate(cum)
        dd   = (cum - peak) / np.where(peak == 0, 1, peak) * 100
        fig_dd.add_trace(go.Scatter(
            y=dd, mode="lines", name=name, fill="tozeroy",
            line=dict(color=palette[i % len(palette)], width=1.5),
        ))
    fig_dd.update_layout(template="plotly_dark", title="Drawdown (%)", height=350,
                         yaxis_title="Drawdown %")
    st.plotly_chart(fig_dd, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Macro Dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Macro Dashboard":
    st.markdown('<div class="section-header">🌐 Macro Dashboard</div>', unsafe_allow_html=True)

    macro_raw = data.get("macro_daily_raw")
    if macro_raw is None or len(macro_raw) == 0:
        st.info("Macro data not available.")
    else:
        macro_cols = [c for c in ["GDP", "CPI", "FEDFUNDS", "UNRATE"] if c in macro_raw.columns]
        titles     = {"GDP": "GDP Growth", "CPI": "CPI Inflation", "FEDFUNDS": "Fed Funds Rate", "UNRATE": "Unemployment Rate"}
        colors_m   = {"GDP": "#64ffda", "CPI": "#ff6e6e", "FEDFUNDS": "#ffb86c", "UNRATE": "#8be9fd"}

        fig_m = make_subplots(rows=2, cols=2,
                              subplot_titles=[titles.get(c, c) for c in macro_cols])
        positions = [(1,1),(1,2),(2,1),(2,2)]
        for idx, col in enumerate(macro_cols):
            r, c = positions[idx]
            series = macro_raw[col].dropna()
            fig_m.add_trace(go.Scatter(
                x=series.index, y=series.values,
                mode="lines", name=col,
                line=dict(color=colors_m.get(col, "#ffffff"), width=1.5),
            ), row=r, col=c)

        fig_m.update_layout(template="plotly_dark", height=600,
                            showlegend=False, title_text="Macroeconomic Indicators")
        st.plotly_chart(fig_m, use_container_width=True)

        st.markdown("### Latest Macro Readings")
        latest = macro_raw.iloc[-1][macro_cols]
        prev   = macro_raw.iloc[-22][macro_cols] if len(macro_raw) > 22 else macro_raw.iloc[0][macro_cols]
        c1, c2, c3, c4 = st.columns(4)
        for col_widget, series_name in zip([c1, c2, c3, c4], macro_cols):
            val  = float(latest[series_name])
            chg  = float(latest[series_name] - prev[series_name])
            sign = "▲" if chg >= 0 else "▼"
            clr  = "#2ecc71" if chg >= 0 else "#e74c3c"
            with col_widget:
                st.markdown(f"""<div class="kpi-card">
                    <div class="kpi-label">{series_name}</div>
                    <div class="kpi-value">{val:.3f}</div>
                    <div class="kpi-sub" style="color:{clr}">{sign} {abs(chg):.3f} (1mo)</div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Ablation Study
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Ablation Study":
    st.markdown('<div class="section-header">🔬 Ablation Study</div>', unsafe_allow_html=True)

    if not ablation.empty:
        st.markdown("### Sharpe Ratio by System Variant")
        sharpe_vals = ablation["Sharpe"].values
        names       = ablation.index.tolist()
        colors_a    = ["#64ffda" if "BMA" in n else "#6272a4" for n in names]

        fig_ab = go.Figure(go.Bar(
            x=names, y=sharpe_vals, marker_color=colors_a,
            text=[f"{v:.3f}" for v in sharpe_vals], textposition="outside",
        ))
        fig_ab.update_layout(template="plotly_dark", height=400,
                             title="Sharpe Ratio: Ablation Comparison",
                             yaxis_title="Sharpe Ratio")
        st.plotly_chart(fig_ab, use_container_width=True)

        st.markdown("### Full Metrics Table")
        st.dataframe(ablation, use_container_width=True)

        # Rolling 60-day Sharpe
        st.markdown("### Rolling 60-Day Sharpe")
        test_X = data["X_test"]
        bma_w  = results.get("_bma_weights")
        if bma_w is not None:
            from backtest.backtester import equal_weight
            ew_rets  = equal_weight(test_X)
            bma_rets = (test_X * bma_w).sum(axis=1)
            W = 60
            fig_roll = go.Figure()
            for name, rets in [("Equal Weight", ew_rets), ("Macro-HMM-BMA", bma_rets)]:
                r = np.asarray(rets, dtype=float)
                r = r[np.isfinite(r)]
                rolling_sharpe = [
                    r[max(0,t-W):t+1].mean() / (r[max(0,t-W):t+1].std() + 1e-8) * np.sqrt(252)
                    for t in range(len(r))
                ]
                fig_roll.add_trace(go.Scatter(
                    y=rolling_sharpe, mode="lines", name=name,
                ))
            fig_roll.update_layout(template="plotly_dark", height=350,
                                   title="Rolling 60-Day Sharpe Ratio",
                                   yaxis_title="Sharpe")
            st.plotly_chart(fig_roll, use_container_width=True)
    else:
        st.info("Ablation data not available. Run the full pipeline first.")
