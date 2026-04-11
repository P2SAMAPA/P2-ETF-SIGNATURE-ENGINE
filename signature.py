"""
P2-ETF-SIGNATURE-ENGINE · signature.py
Compute truncated path signatures via esig (primary) or iisignature (fallback).
Falls back to numpy implementation if neither is available.
"""

from __future__ import annotations
import numpy as np

# Try esig first (actively maintained, numpy 2.x compatible)
try:
    import esig
    # Set to use iisignature backend if available in esig, otherwise libalgebra
    try:
        esig.set_backend("iisignature")
    except:
        pass
    _HAS_ESIG = True
    print("[signature] Using esig for signature computation")
except ImportError:
    _HAS_ESIG = False

# Fallback to iisignature (requires numpy 1.x)
if not _HAS_ESIG:
    try:
        import iisignature
        _HAS_IISIG = True
        print("[signature] Using iisignature for signature computation")
    except ImportError:
        _HAS_IISIG = False
        print("[signature] WARNING: Neither esig nor iisignature found — using numpy fallback (slower, depth ≤ 2 only).")
else:
    _HAS_IISIG = False


def compute_signature(path: np.ndarray, depth: int) -> np.ndarray:
    """
    Compute the truncated signature of a path up to given depth.

    Parameters
    ----------
    path : np.ndarray shape (T, d) augmented path
    depth : int truncation depth (2, 3, or 4)

    Returns
    -------
    np.ndarray 1-D feature vector of iterated integrals
    """
    if path.shape[0] < 2:
        d = path.shape[1]
        dim = _sig_dim(d, depth)
        return np.zeros(dim, dtype=np.float32)

    path = path.astype(np.float64)

    if _HAS_ESIG:
        return _esig_compute(path, depth)
    elif _HAS_IISIG:
        return _iisig_compute(path, depth)
    else:
        if depth > 2:
            print(f"[signature] WARNING: depth {depth} > 2 not supported in numpy fallback, using depth 2")
        return _numpy_sig_depth2(path)


def _esig_compute(path: np.ndarray, depth: int) -> np.ndarray:
    """Use esig for fast signature computation."""
    try:
        sig = esig.stream2sig(path, depth)
        return sig.astype(np.float32)
    except Exception as e:
        print(f"[signature] esig error: {e} — falling back to iisignature/numpy.")
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
        print(f"[signature] iisignature error: {e} — falling back to numpy.")
        return _numpy_sig_depth2(path)


def _numpy_sig_depth2(path: np.ndarray) -> np.ndarray:
    """
    Manual depth-2 signature using numpy.
    Level-1: increments (∫dX^i)
    Level-2: iterated integrals (∫∫dX^i dX^j)
    Returns concatenated [level1, level2].
    """
    T, d = path.shape
    dX = np.diff(path, axis=0)  # (T-1, d)

    # Level 1: sum of increments
    level1 = dX.sum(axis=0)  # (d,)

    # Level 2: iterated integrals (Chen's identity approximation)
    level2 = np.zeros((d, d), dtype=np.float64)
    cum = np.zeros(d, dtype=np.float64)
    for t in range(T - 1):
        dx = dX[t]
        level2 += np.outer(cum, dx)
        cum += dx
    level2 = level2.flatten()  # (d*d,)

    return np.concatenate([level1, level2]).astype(np.float32)


def _sig_dim(d: int, depth: int) -> int:
    """Total number of signature terms for dimension d, depth k."""
    if d == 1:
        return depth
    return int(d * (d ** depth - 1) / (d - 1))


def batch_signatures(paths: list[np.ndarray], depth: int) -> np.ndarray:
    """
    Compute signatures for a list of paths.

    Returns
    -------
    np.ndarray shape (len(paths), sig_dim)
    """
    sigs = [compute_signature(p, depth) for p in paths]
    return np.stack(sigs, axis=0)
