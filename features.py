"""
P2-ETF-SIGNATURE-ENGINE · features.py
Build the rolling-window signature feature matrix with caching.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import lru_cache
import hashlib

from path_builder import build_path
from signature import compute_signature

# Global cache for signature computations
_signature_cache = {}

def _get_cache_key(ret_win, mac_win, depth):
    """Generate cache key from window data and depth."""
    # Use hash of concatenated data + depth
    ret_hash = hashlib.md5(ret_win.values.tobytes()).hexdigest()[:16]
    mac_hash = hashlib.md5(mac_win.values.tobytes()).hexdigest()[:16]
    return f"{ret_hash}_{mac_hash}_{depth}"

def build_feature_matrix(returns_df: pd.DataFrame,
                         macro_df: pd.DataFrame,
                         lookback: int,
                         depth: int,
                         verbose: bool = False,
                         use_cache: bool = True) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build signature feature matrix X and target matrix y with caching.
    """
    T = len(returns_df)
    n_etfs = returns_df.shape[1]
    rows_X = []
    rows_y = []
    dates = []

    iterator = range(lookback, T - 1)
    if verbose:
        iterator = tqdm(iterator, desc=f" Building signatures (lb={lookback}, d={depth})")

    for t in iterator:
        ret_win = returns_df.iloc[t - lookback: t]
        mac_win = macro_df.iloc[t - lookback: t]
        mac_win = mac_win.reindex(ret_win.index, method="ffill").fillna(0.0)

        # Check cache
        if use_cache:
            cache_key = _get_cache_key(ret_win, mac_win, depth)
            if cache_key in _signature_cache:
                sig = _signature_cache[cache_key]
            else:
                path = build_path(ret_win, mac_win)
                sig = compute_signature(path, depth)
                _signature_cache[cache_key] = sig
        else:
            path = build_path(ret_win, mac_win)
            sig = compute_signature(path, depth)

        next_ret = returns_df.iloc[t + 1].values
        next_date = returns_df.index[t + 1]

        rows_X.append(sig)
        rows_y.append(next_ret)
        dates.append(next_date)

    if not rows_X:
        raise ValueError(f"No samples generated — lookback={lookback} may be >= len(data).")

    X = np.stack(rows_X, axis=0).astype(np.float32)
    y = np.stack(rows_y, axis=0).astype(np.float32)

    return X, y, pd.DatetimeIndex(dates)

def build_live_feature(returns_df: pd.DataFrame,
                       macro_df: pd.DataFrame,
                       lookback: int,
                       depth: int) -> np.ndarray:
    """
    Build a single signature feature vector from the most recent lookback days.
    """
    ret_win = returns_df.iloc[-lookback:]
    mac_win = macro_df.reindex(ret_win.index, method="ffill").fillna(0.0)
    path = build_path(ret_win, mac_win)
    sig = compute_signature(path, depth)
    return sig.reshape(1, -1)

def clear_signature_cache():
    """Clear the signature cache to free memory."""
    global _signature_cache
    _signature_cache = {}
