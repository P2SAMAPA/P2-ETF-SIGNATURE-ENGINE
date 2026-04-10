"""
P2-ETF-SIGNATURE-ENGINE  ·  path_builder.py
Construct augmented paths from return + macro data for signature computation.

Three augmentation steps applied in order:
  1. Time channel  — appends normalised [0,1] time coordinate so the signature
                     captures *when* events happen, not just *what* happened.
  2. Basepoint     — prepend a row of zeros so the signature is translation-
                     invariant (standard practice for financial paths).
  3. Lead-lag      — double the path dimension by interleaving each channel
                     with a lagged copy; this makes the depth-2 signature term
                     capture the quadratic covariation (realised covariance),
                     which is crucial for cross-asset return data.

The macro features are interpolated linearly between observations so the path
is well-defined at every trading day even when FRED series update weekly.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def build_path(returns_window: pd.DataFrame,
               macro_window: pd.DataFrame,
               add_time: bool = True,
               lead_lag: bool = True) -> np.ndarray:
    """
    Construct an augmented path array from a rolling window.

    Parameters
    ----------
    returns_window : pd.DataFrame  shape (T, n_etfs)  log returns
    macro_window   : pd.DataFrame  shape (T, n_macro)  macro features
    add_time       : bool  append normalised time channel
    lead_lag       : bool  apply lead-lag transformation

    Returns
    -------
    np.ndarray  shape (T', d)  augmented path ready for signature computation
                T' = T+1 (basepoint) or 2T-1 (lead-lag)
    """
    T = len(returns_window)

    # Normalise returns and macro to similar scale
    ret_vals   = returns_window.values.copy()
    macro_vals = macro_window.values.copy()

    # Z-score macro (FRED series have very different scales)
    macro_std  = macro_vals.std(axis=0)
    macro_std[macro_std < 1e-8] = 1.0
    macro_vals = (macro_vals - macro_vals.mean(axis=0)) / macro_std

    # Concatenate returns and macro into base path
    path = np.concatenate([ret_vals, macro_vals], axis=1)   # (T, n_etfs + n_macro)

    # 1. Time channel
    if add_time:
        t_channel = np.linspace(0, 1, T).reshape(-1, 1)
        path      = np.concatenate([path, t_channel], axis=1)

    # 2. Basepoint: prepend a row of zeros
    basepoint = np.zeros((1, path.shape[1]))
    path      = np.concatenate([basepoint, path], axis=0)   # (T+1, d)

    # 3. Lead-lag transformation
    if lead_lag:
        path = _lead_lag(path)

    return path.astype(np.float32)


def _lead_lag(path: np.ndarray) -> np.ndarray:
    """
    Apply the lead-lag transformation.
    For a d-dimensional path X of length T, returns a 2d-dimensional path
    of length 2T-1 by interleaving X(t) and X(t-1).

    This makes the depth-2 signature term equal to the realised covariance
    matrix of X — a key feature for financial return paths.
    """
    T, d  = path.shape
    ll    = np.zeros((2 * T - 1, 2 * d), dtype=path.dtype)

    for i in range(T - 1):
        ll[2 * i,     :d]  = path[i]       # lead: X(t)
        ll[2 * i,     d:]  = path[i]       # lag:  X(t)   (same at even steps)
        ll[2 * i + 1, :d]  = path[i + 1]  # lead: X(t+1)
        ll[2 * i + 1, d:]  = path[i]       # lag:  X(t)   (still old value)

    ll[-1, :d] = path[-1]
    ll[-1, d:] = path[-1]

    return ll


def path_dimension(n_etfs: int, n_macro: int,
                   add_time: bool = True,
                   lead_lag: bool = True) -> int:
    """Return the dimension d of the augmented path."""
    d = n_etfs + n_macro
    if add_time:
        d += 1
    if lead_lag:
        d *= 2
    return d


def sig_feature_dim(path_dim: int, depth: int) -> int:
    """
    Number of signature terms for a path of dimension d truncated at depth k.
    Formula: sum_{i=1}^{k} d^i = d * (d^k - 1) / (d - 1)
    """
    if depth == 0:
        return 0
    d = path_dim
    return int(d * (d ** depth - 1) / (d - 1))
