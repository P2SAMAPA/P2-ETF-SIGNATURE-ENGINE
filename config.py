"""
P2-ETF-SIGNATURE-ENGINE · config.py
Reduced constants for memory-constrained GitHub Actions runners.
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

# ── REDUCED Hyperparameter grid for memory-constrained environments ───────────
# Reduced from [30, 45, 60] to [30, 45] to save memory
LOOKBACK_CANDIDATES = [30, 45]      # Removed 60 (saves 1/3 of combinations)
# Reduced from [2, 3, 4] to [2, 3] - depth 4 creates very large feature vectors
DEPTH_CANDIDATES    = [2, 3]         # Removed 4 (depth 4 signatures are memory-intensive)
# Using only ridge for now - lasso can be added back once memory is stable
MODEL_CANDIDATES    = ["ridge"]      # Removed lasso temporarily

# ── REDUCED Expanding windows ─────────────────────────────────────────────────
# Reduced from 7 windows to 4 windows to minimize memory usage
EXPANDING_START_YEARS = [2012, 2016, 2019, 2021]  # 4 windows instead of 7

# ── Consensus scoring weights ───────────────────────────────────────────────
CONSENSUS_MIN_RETURN = 0.0

# ── Ridge settings ────────────────────────────────────────────────────────────
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

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
