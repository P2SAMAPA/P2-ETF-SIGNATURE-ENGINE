"""
P2-ETF-SIGNATURE-ENGINE  ·  features.py
Build the rolling-window signature feature matrix.

For each trading day t, use the window [t-lookback : t] of returns + macro
to construct an augmented path, compute its truncated signature, and store
the result as one row of the feature matrix X.

The target y for day t is the log return of each ETF on day t+1
(next-day return — what we are trying to predict).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm

from path_builder import build_path
from signature    import compute_signature


def build_feature_matrix(returns_df: pd.DataFrame,
                          macro_df: pd.DataFrame,
                          lookback: int,
                          depth: int,
                          verbose: bool = False) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build signature feature matrix X and target matrix y.

    Parameters
    ----------
    returns_df : pd.DataFrame  shape (T, n_etfs)  daily log returns
    macro_df   : pd.DataFrame  shape (T, n_macro)  macro features
    lookback   : int           rolling window length in trading days
    depth      : int           signature truncation depth

    Returns
    -------
    X     : np.ndarray  shape (N, sig_dim)   signature features
    y     : np.ndarray  shape (N, n_etfs)    next-day log returns (targets)
    dates : pd.DatetimeIndex                 prediction dates (day t+1)
    """
    T       = len(returns_df)
    n_etfs  = returns_df.shape[1]
    rows_X  = []
    rows_y  = []
    dates   = []

    iterator = range(lookback, T - 1)
    if verbose:
        iterator = tqdm(iterator, desc=f"  Building signatures (lb={lookback}, d={depth})")

    for t in iterator:
        ret_win   = returns_df.iloc[t - lookback: t]
        mac_win   = macro_df.iloc[t - lookback: t]

        # Align macro to return index (macro may have fewer rows after dropna)
        mac_win   = mac_win.reindex(ret_win.index, method="ffill").fillna(0.0)

        path      = build_path(ret_win, mac_win)
        sig       = compute_signature(path, depth)

        next_ret  = returns_df.iloc[t + 1].values   # shape (n_etfs,)
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
    Build a single signature feature vector from the most recent
    `lookback` days — used for live next-day prediction.

    Returns
    -------
    np.ndarray  shape (1, sig_dim)
    """
    ret_win = returns_df.iloc[-lookback:]
    mac_win = macro_df.reindex(ret_win.index, method="ffill").fillna(0.0)
    path    = build_path(ret_win, mac_win)
    sig     = compute_signature(path, depth)
    return sig.reshape(1, -1)
