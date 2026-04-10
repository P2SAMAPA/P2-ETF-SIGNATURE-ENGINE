"""
P2-ETF-SIGNATURE-ENGINE  ·  config.py
All constants, ETF universes, and hyperparameters.
"""

# ── Hugging Face ──────────────────────────────────────────────────────────────
HF_DATASET_IN  = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATASET_OUT = "P2SAMAPA/p2-etf-signature-engine-results"

# ── ETF universes ─────────────────────────────────────────────────────────────
FI_ETFS     = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV", "VCIT"]
EQUITY_ETFS = ["QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
               "XLP", "XLU", "GDX", "XME", "IWM"]

BENCHMARK_FI     = "AGG"
BENCHMARK_EQUITY = "SPY"

# ── Macro features (appended to path) ────────────────────────────────────────
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

# ── Regime detection ──────────────────────────────────────────────────────────
N_REGIMES       = 3
REGIME_FEATURES = ["VIX", "T10Y2Y", "HY_SPREAD"]

# ── Train / val / test split (applied to every window) ───────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# test = remaining 0.10

# ── Hyperparameter grid (optimised on val set of full-dataset run) ────────────
LOOKBACK_CANDIDATES = [30, 45, 60]      # rolling window in trading days
DEPTH_CANDIDATES    = [2, 3, 4]         # signature truncation depth
MODEL_CANDIDATES    = ["ridge", "lasso"] # linear model type

# ── Expanding windows (option b) — all end at latest available date ───────────
EXPANDING_START_YEARS = [2008, 2010, 2012, 2014, 2016, 2018, 2020]

# ── Consensus scoring weights (expanding windows) ────────────────────────────
# Each window's vote is weighted by its val-set Sharpe ratio (floored at 0)
# Negative-return windows are excluded from the consensus entirely
CONSENSUS_MIN_RETURN = 0.0   # exclude windows with OOS cumulative return <= this

# ── Ridge / Lasso settings ────────────────────────────────────────────────────
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]   # cross-validated on val set
LASSO_ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0]

# ── Transaction cost ──────────────────────────────────────────────────────────
TRANSACTION_COST_BPS = 12

# ── Random seed ───────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Output filenames ──────────────────────────────────────────────────────────
OUTPUT_JSON          = "signature_signal.json"
SIGNAL_HISTORY_FI    = "signal_history_fi.csv"
SIGNAL_HISTORY_EQ    = "signal_history_eq.csv"
METRICS_FULL_FI      = "metrics_full_fi.json"
METRICS_FULL_EQ      = "metrics_full_eq.json"
METRICS_WINDOWS_FI   = "metrics_windows_fi.json"
METRICS_WINDOWS_EQ   = "metrics_windows_eq.json"
