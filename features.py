"""
P2-ETF-SIGNATURE-ENGINE · features.py
Build the rolling-window signature feature matrix (no cache, memory-efficient).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm

from path_builder import build_path
from signature import compute_signature


def build_feature_matrix(returns_df: pd.DataFrame,
                         macro_df: pd.DataFrame,
                         lookback: int,
                         depth: int,
                         verbose: bool = False) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build signature feature matrix X and target matrix y.
    No caching - processes sequentially to minimize memory usage.
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
    """
    Dummy function for API compatibility.
    No cache to clear in this version.
    """
    pass
