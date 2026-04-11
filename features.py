"""
P2-ETF-SIGNATURE-ENGINE · features.py
Memory-efficient signature feature building with batching.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm

from path_builder import build_path
from signature import compute_signature, clear_signature_cache

# Global cache for signature computations
_signature_cache = {}
_MAX_CACHE_SIZE = 10000  # Limit cache size


def _get_cache_key(ret_win, mac_win, depth):
    """Generate cache key from window data and depth."""
    ret_hash = hashlib.md5(ret_win.values.tobytes()).hexdigest()[:16]
    mac_hash = hashlib.md5(mac_win.values.tobytes()).hexdigest()[:16]
    return f"{ret_hash}_{mac_hash}_{depth}"


def build_feature_matrix(returns_df: pd.DataFrame,
                         macro_df: pd.DataFrame,
                         lookback: int,
                         depth: int,
                         verbose: bool = False,
                         batch_size: int = 500) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build signature feature matrix X and target matrix y with memory-efficient batching.
    """
    import hashlib
    
    T = len(returns_df)
    n_etfs = returns_df.shape[1]
    dates = []
    
    # Pre-allocate arrays in batches to avoid memory fragmentation
    all_X = []
    all_y = []
    
    iterator = range(lookback, T - 1)
    if verbose:
        iterator = tqdm(iterator, desc=f" Building signatures (lb={lookback}, d={depth})")

    for t in iterator:
        ret_win = returns_df.iloc[t - lookback: t]
        mac_win = macro_df.iloc[t - lookback: t]
        mac_win = mac_win.reindex(ret_win.index, method="ffill").fillna(0.0)

        # Check cache with size limit
        cache_key = _get_cache_key(ret_win, mac_win, depth)
        if cache_key in _signature_cache:
            sig = _signature_cache[cache_key]
        else:
            path = build_path(ret_win, mac_win)
            sig = compute_signature(path, depth)
            # Limit cache size
            if len(_signature_cache) < _MAX_CACHE_SIZE:
                _signature_cache[cache_key] = sig

        next_ret = returns_df.iloc[t + 1].values
        next_date = returns_df.index[t + 1]

        all_X.append(sig)
        all_y.append(next_ret)
        dates.append(next_date)
        
        # Periodic garbage collection every batch
        if len(all_X) % batch_size == 0:
            import gc
            gc.collect()

    if not all_X:
        raise ValueError(f"No samples generated — lookback={lookback} may be >= len(data).")

    X = np.stack(all_X, axis=0).astype(np.float32)
    y = np.stack(all_y, axis=0).astype(np.float32)

    return X, y, pd.DatetimeIndex(dates)


def build_live_feature(returns_df: pd.DataFrame,
                       macro_df: pd.DataFrame,
                       lookback: int,
                       depth: int) -> np.ndarray:
    """Build a single signature feature vector from the most recent lookback days."""
    ret_win = returns_df.iloc[-lookback:]
    mac_win = macro_df.reindex(ret_win.index, method="ffill").fillna(0.0)
    path = build_path(ret_win, mac_win)
    sig = compute_signature(path, depth)
    return sig.reshape(1, -1)


def clear_signature_cache():
    """Clear the signature cache to free memory."""
    global _signature_cache
    _signature_cache = {}
    import gc
    gc.collect()
