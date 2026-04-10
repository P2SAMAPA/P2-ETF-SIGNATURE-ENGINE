"""
P2-ETF-SIGNATURE-ENGINE  ·  signature.py
Compute truncated path signatures via iisignature.
Falls back to a manual numpy implementation if iisignature is unavailable.
"""

from __future__ import annotations
import numpy as np

try:
    import iisignature
    _HAS_IISIG = True
except ImportError:
    _HAS_IISIG = False
    print("[signature] WARNING: iisignature not found — using numpy fallback (slower, depth ≤ 2 only).")


def compute_signature(path: np.ndarray, depth: int) -> np.ndarray:
    """
    Compute the truncated signature of a path up to given depth.

    Parameters
    ----------
    path  : np.ndarray  shape (T, d)  augmented path
    depth : int         truncation depth (2, 3, or 4)

    Returns
    -------
    np.ndarray  1-D feature vector of iterated integrals
    """
    if path.shape[0] < 2:
        d   = path.shape[1]
        dim = _sig_dim(d, depth)
        return np.zeros(dim, dtype=np.float32)

    path = path.astype(np.float64)

    if _HAS_IISIG:
        return _iisig_compute(path, depth)
    else:
        return _numpy_sig_depth2(path)


def _iisig_compute(path: np.ndarray, depth: int) -> np.ndarray:
    """Use iisignature for fast signature computation."""
    try:
        s = iisignature.prepare(path.shape[1], depth)
        sig = iisignature.sig(path, s)
        return sig.astype(np.float32)
    except Exception as e:
        print(f"  [signature] iisignature error: {e} — falling back to numpy.")
        return _numpy_sig_depth2(path)


def _numpy_sig_depth2(path: np.ndarray) -> np.ndarray:
    """
    Manual depth-2 signature using numpy.
    Level-1: increments (∫dX^i)
    Level-2: iterated integrals (∫∫dX^i dX^j)
    Returns concatenated [level1, level2].
    """
    T, d   = path.shape
    dX     = np.diff(path, axis=0)       # (T-1, d)

    # Level 1: sum of increments
    level1 = dX.sum(axis=0)              # (d,)

    # Level 2: iterated integrals (Chen's identity approximation)
    level2 = np.zeros((d, d), dtype=np.float64)
    cum    = np.zeros(d, dtype=np.float64)
    for t in range(T - 1):
        dx        = dX[t]
        level2   += np.outer(cum, dx)
        cum      += dx
    level2 = level2.flatten()            # (d*d,)

    return np.concatenate([level1, level2]).astype(np.float32)


def _sig_dim(d: int, depth: int) -> int:
    """Total number of signature terms for dimension d, depth k."""
    return int(d * (d ** depth - 1) / (d - 1)) if d > 1 else depth


def batch_signatures(paths: list[np.ndarray], depth: int) -> np.ndarray:
    """
    Compute signatures for a list of paths.

    Returns
    -------
    np.ndarray  shape (len(paths), sig_dim)
    """
    sigs = [compute_signature(p, depth) for p in paths]
    return np.stack(sigs, axis=0)
