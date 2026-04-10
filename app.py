"""
P2-ETF-SIGNATURE-ENGINE  ·  app.py
Streamlit dashboard — two tabs (FI and Equity).
Each tab shows TWO hero boxes:
  (a) Full dataset prediction   (2008-present, single model)
  (b) Consensus prediction      (expanding windows, weighted vote)
"""

import json, os
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
from config import HF_DATASET_OUT

st.set_page_config(
    page_title="P2 ETF Signature Engine",
    page_icon="〜",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Full-dataset hero — blue accent */
.hero-full {
    background: #ffffff;
    border: 1px solid #dde3ed;
    border-left: 6px solid #1a6ef5;
    border-radius: 10px;
    padding: 1.3rem 1.6rem;
    margin-bottom: 0.6rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
/* Consensus hero — green accent */
.hero-cons {
    background: #ffffff;
    border: 1px solid #dde3ed;
    border-left: 6px solid #16a34a;
    border-radius: 10px;
    padding: 1.3rem 1.6rem;
    margin-bottom: 0.6rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.hero-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.3rem;
}
.hero-label-full { color: #1a6ef5; }
.hero-label-cons { color: #16a34a; }
.hero-ticker-full { font-size: 2.8rem; font-weight: 800; color: #1a6ef5; letter-spacing: 3px; line-height: 1.1; }
.hero-ticker-cons { font-size: 2.8rem; font-weight: 800; color: #16a34a; letter-spacing: 3px; line-height: 1.1; }
.hero-conviction  { font-size: 1.3rem; font-weight: 700; color: #1a202c; margin-top: 0.1rem; }
.hero-row  { font-size: 0.9rem; color: #4a5568; margin-top: 0.4rem; }
.hero-row b { color: #1a202c; }
.hero-bm   { font-size: 0.78rem; color: #9aa5b4; margin-top: 0.45rem; }

/* 2nd / 3rd picks */
.pick-card {
    background: #f7f9fc;
    border: 1px solid #dde3ed;
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.4rem;
}
.pick-rank   { font-size: 0.68rem; color: #9aa5b4; text-transform: uppercase; letter-spacing: 1px; }
.pick-ticker { font-size: 1.25rem; font-weight: 700; color: #2d3748; }
.pick-pct    { font-size: 0.85rem; color: #4a5568; }

/* Macro pills */
.pill {
    display: inline-block;
    background: #eef2fa;
    border: 1px solid #c9d6f0;
    border-radius: 20px;
    padding: 4px 12px;
    margin: 3px 3px 3px 0;
    font-size: 0.8rem;
}
.pill-key { color: #6b7a99; }
.pill-val { color: #1a6ef5; font-weight: 700; margin-left: 3px; }

/* Metric tiles */
.metric-tile {
    background: #f7f9fc;
    border: 1px solid #e0e6ef;
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
    text-align: center;
    margin-bottom: 0.4rem;
}
.metric-label { font-size: 0.72rem; color: #9aa5b4; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-value { font-size: 1.25rem; font-weight: 700; color: #1a202c; margin-top: 0.15rem; }
.metric-pos { color: #16a34a; }
.metric-neg { color: #dc2626; }

/* Windows table */
.section-label {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #6b7a99;
    margin: 1rem 0 0.3rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def _token():
    return os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN", None)


@st.cache_data(ttl=300, show_spinner="Loading signals from Hugging Face...")
def load_signal() -> dict:
    token  = _token()
    errors = []
    for fname in ["results/signature_signal.json", "signature_signal.json"]:
        try:
            path = hf_hub_download(
                repo_id=HF_DATASET_OUT, filename=fname,
                repo_type="dataset", token=token, force_download=True,
            )
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            errors.append(f"{fname}: {e}")
    st.error("Could not load signal JSON:\n" + "\n".join(errors))
    return {}


@st.cache_data(ttl=300, show_spinner=False)
def load_history(module: str) -> pd.DataFrame:
    fname = f"signal_history_{module.lower()}.csv"
    token = _token()
    for sub in ["results/", ""]:
        try:
            path = hf_hub_download(
                repo_id=HF_DATASET_OUT, filename=f"{sub}{fname}",
                repo_type="dataset", token=token, force_download=True,
            )
            return pd.read_csv(path, parse_dates=["date"])
        except Exception:
            continue
    return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_metrics(key: str) -> dict:
    """key e.g. 'metrics_full_fi', 'metrics_windows_eq'"""
    fname = f"{key}.json"
    token = _token()
    for sub in ["results/", ""]:
        try:
            path = hf_hub_download(
                repo_id=HF_DATASET_OUT, filename=f"{sub}{fname}",
                repo_type="dataset", token=token, force_download=True,
            )
            with open(path) as f:
                return json.load(f)
        except Exception:
            continue
    return {}


# ── Render helpers ────────────────────────────────────────────────────────────

def render_hero(sig: dict, hero_class: str, ticker_class: str,
                label: str, label_class: str, benchmark: str):
    """Render one hero card."""
    pred_pct = sig.get("pred_return", 0) * 100
    depth    = sig.get("sig_depth", "—")
    lb       = sig.get("lookback_days", "—")
    model    = str(sig.get("model_type", "—")).capitalize()
    n_win    = sig.get("n_windows_used")
    win_txt  = f" · Windows used: <b>{n_win}</b>" if n_win is not None else ""

    st.markdown(f"""
    <div class="{hero_class}">
      <div class="hero-label {label_class}">{label}</div>
      <div class="{ticker_class}">{sig.get('pick', '—')}</div>
      <div class="hero-conviction">{sig.get('conviction_pct', 0):.1f}% conviction</div>
      <div class="hero-row">
        <b>Next trading day:</b> {sig.get('next_trading_day', '—')}
      </div>
      <div class="hero-row">
        <b>Lookback:</b> {lb}d &nbsp;·&nbsp;
        <b>Sig depth:</b> {depth} &nbsp;·&nbsp;
        <b>Model:</b> {model}{win_txt}
      </div>
      <div class="hero-row">
        <b>Predicted return:</b> {pred_pct:.4f}% &nbsp;·&nbsp;
        <b>Regime:</b> {sig.get('regime_name', '—')}
      </div>
      <div class="hero-bm">Benchmark: {benchmark} (not traded · no CASH output)</div>
    </div>
    """, unsafe_allow_html=True)


def render_picks_and_pills(sig: dict):
    """2nd/3rd picks + macro pills."""
    for rank, pk, cv in [("2nd pick","second_pick","second_conviction"),
                          ("3rd pick","third_pick", "third_conviction")]:
        t = sig.get(pk)
        c = sig.get(cv) or 0
        if t:
            st.markdown(f"""
            <div class="pick-card">
              <div class="pick-rank">{rank}</div>
              <div class="pick-ticker">{t}</div>
              <div class="pick-pct">{c:.1f}% conviction</div>
            </div>""", unsafe_allow_html=True)

    pills = sig.get("macro_pills", {})
    if pills:
        html = "".join(
            f'<span class="pill"><span class="pill-key">{k}</span>'
            f'<span class="pill-val">{v}</span></span>'
            for k, v in pills.items()
        )
        st.markdown(html, unsafe_allow_html=True)


def render_scores(sig: dict):
    sc = sig.get("all_scores", [])
    if not sc:
        return
    df = pd.DataFrame(sc)
    if "pred_return" in df.columns:
        df["pred_return"] = (df["pred_return"] * 100).round(5)
    df.columns = [c.replace("_", " ").title() for c in df.columns]
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_metrics(metrics: dict, label: str = ""):
    if not metrics:
        st.info("Backtest metrics not yet available.")
        return
    if label:
        st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)

    fields = [
        ("Ann. Return",  f"{metrics.get('ann_return_pct',0):.2f}%",  metrics.get('ann_return_pct',0) >= 0),
        ("Ann. Vol",     f"{metrics.get('ann_vol_pct',0):.2f}%",     True),
        ("Sharpe",       f"{metrics.get('sharpe',0):.3f}",           metrics.get('sharpe',0) >= 0),
        ("Max Drawdown", f"{metrics.get('max_drawdown_pct',0):.2f}%",False),
        ("Hit Rate",     f"{metrics.get('hit_rate_pct',0):.1f}%",    metrics.get('hit_rate_pct',0) >= 50),
        ("Alpha vs BM",  f"{metrics.get('ann_alpha_pct',0):.2f}%",   metrics.get('ann_alpha_pct',0) >= 0),
    ]
    cols = st.columns(6)
    for col, (lbl, val, pos) in zip(cols, fields):
        colour = "metric-pos" if pos else "metric-neg"
        with col:
            st.markdown(
                f'<div class="metric-tile">'
                f'<div class="metric-label">{lbl}</div>'
                f'<div class="metric-value {colour}">{val}</div>'
                f'</div>', unsafe_allow_html=True)


def render_windows_table(window_metrics):
    if not window_metrics:
        st.info("Window metrics not yet available.")
        return
    df = pd.DataFrame(window_metrics)
    # Highlight excluded windows
    def style_row(row):
        if row.get("ann_return_pct", 0) < 0:
            return ["background-color: #fff5f5"] * len(row)
        return [""] * len(row)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_equity_curve(history: pd.DataFrame, pick_col: str = "pick_full"):
    if history.empty:
        return
    # We only have picks, not realised returns in history — show pick frequency
    if "pick_full" in history.columns and "pick_consensus" in history.columns:
        st.markdown('<div class="section-label">Pick history (most recent 60 days)</div>',
                    unsafe_allow_html=True)
        h = history.sort_values("date", ascending=False).head(60)
        st.dataframe(h, use_container_width=True, hide_index=True)


def render_debug(signal_data: dict):
    with st.sidebar:
        st.markdown("### Debug")
        st.write("**Signal keys:**", list(signal_data.keys()))
        for k in ["FI_full", "FI_consensus", "EQ_full", "EQ_consensus"]:
            present = k in signal_data
            pick    = signal_data[k].get("pick") if present else "—"
            st.write(f"**{k}:** {'✓' if present else '✗'}  {pick}")
        st.write("**Generated at:**", signal_data.get("generated_at", "—"))
        st.write("**HF_TOKEN set:**", _token() is not None)
        if st.button("List HF repo files"):
            try:
                files = list(list_repo_files(HF_DATASET_OUT, repo_type="dataset", token=_token()))
                st.write(files)
            except Exception as e:
                st.error(str(e))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.title("〜 P2-ETF-SIGNATURE-ENGINE")
    st.markdown("*Path signatures · Iterated integrals · Next-day ETF selection*")

    signal_data = load_signal()
    render_debug(signal_data)
    gen_at = signal_data.get("generated_at", "—")
    st.caption(f"Last generated: {gen_at} UTC  ·  Source: {HF_DATASET_OUT}")

    tab_fi, tab_eq = st.tabs(["🏦 Fixed Income / Commodities", "📈 Equity Sectors"])

    for tab, module, benchmark, full_key, cons_key, hist_key, met_full_key, met_win_key in [
        (tab_fi, "FI", "AGG", "FI_full", "FI_consensus",
         "FI", "metrics_full_fi", "metrics_windows_fi"),
        (tab_eq, "EQ", "SPY", "EQ_full", "EQ_consensus",
         "EQ", "metrics_full_eq", "metrics_windows_eq"),
    ]:
        with tab:
            sig_full = signal_data.get(full_key, {})
            sig_cons = signal_data.get(cons_key, {})
            history  = load_history(module)
            met_full = load_metrics(met_full_key)
            met_wins = load_metrics(met_win_key)

            if not sig_full and not sig_cons:
                st.warning(
                    f"No {module} signals found. Check the sidebar debug panel. "
                    "Click **List HF repo files** to verify what has been uploaded."
                )
                continue

            # ── Two hero boxes side by side ───────────────────────
            col_a, col_b = st.columns(2, gap="large")

            with col_a:
                if sig_full:
                    render_hero(sig_full, "hero-full", "hero-ticker-full",
                                "Option A — Full Dataset (2008-present)",
                                "hero-label-full", benchmark)
                    render_picks_and_pills(sig_full)
                else:
                    st.info("Full dataset signal not yet available.")

            with col_b:
                if sig_cons:
                    render_hero(sig_cons, "hero-cons", "hero-ticker-cons",
                                "Option B — Expanding Windows Consensus",
                                "hero-label-cons", benchmark)
                    render_picks_and_pills(sig_cons)
                else:
                    st.info("Consensus signal not yet available.")

            st.markdown("---")

            # ── Scores ────────────────────────────────────────────
            sc_col_a, sc_col_b = st.columns(2, gap="large")
            with sc_col_a:
                if sig_full:
                    st.subheader("ETF Scores — Full Dataset")
                    render_scores(sig_full)
            with sc_col_b:
                if sig_cons:
                    st.subheader("ETF Scores — Consensus")
                    render_scores(sig_cons)

            st.markdown("---")

            # ── Backtest metrics ──────────────────────────────────
            st.subheader("OOS Backtest — Full Dataset (test set)")
            render_metrics(met_full)

            st.markdown("---")

            # ── Expanding windows table ───────────────────────────
            st.subheader("Expanding Windows — Per-Window Metrics")
            render_windows_table(met_wins)

            st.markdown("---")

            # ── Signal history ────────────────────────────────────
            st.subheader("Pick History")
            render_equity_curve(history)

    st.markdown("---")
    st.caption("P2 Engine Suite · Signature Engine · Research only · Not financial advice")


if __name__ == "__main__":
    main()
