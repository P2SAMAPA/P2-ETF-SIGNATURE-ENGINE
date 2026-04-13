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
LOOKBACK_CANDIDATES = [30, 45]      # Removed 60
DEPTH_CANDIDATES    = [2, 3]        # Removed 4 (memory-intensive)
MODEL_CANDIDATES    = ["ridge"]     # Removed lasso temporarily

# ── REDUCED Expanding windows ─────────────────────────────────────────────────
EXPANDING_START_YEARS = [2012, 2016, 2019, 2021]  # 4 windows instead of 7

# ── Consensus scoring weights ───────────────────────────────────────────────
CONSENSUS_MIN_RETURN = 0.0

# ── Ridge / Lasso settings ───────────────────────────────────────────────────
# REMOVED very small alphas (0.01, 0.1) that cause numerical instability
# Using larger alphas + SVD solver for matrix conditioning
RIDGE_ALPHAS = [1.0, 10.0, 100.0, 500.0]  # Increased regularization
LASSO_ALPHAS = [0.001, 0.01, 0.1, 1.0]    # Kept for model.py compatibility

# ── Ridge solver settings (for numerical stability) ───────────────────────────
# 'svd' is the most stable solver for ill-conditioned matrices
# 'cholesky' is faster but less stable
RIDGE_SOLVER = 'svd'  # Options: 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'

# ── Feature scaling ──────────────────────────────────────────────────────────
# Enable StandardScaler for features (critical for numerical stability)
USE_FEATURE_SCALING = True

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
