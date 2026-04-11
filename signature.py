"""
P2-ETF-SIGNATURE-ENGINE · signature.py
Compute truncated path signatures via esig (primary) or iisignature (fallback).
"""

from __future__ import annotations
import numpy as np

# Try esig first (actively maintained, numpy 2.x compatible)
_HAS_ESIG = False
_ESIG_SIG_FUNC = None

try:
    import esig
    
    if hasattr(esig, 'sig'):
        _ESIG_SIG_FUNC = lambda path, depth: esig.sig(path, depth)
        _HAS_ESIG = True
        print("[signature] Using esig.sig for signature computation")
    elif hasattr(esig, 'stream2sig'):
        _ESIG_SIG_FUNC = lambda path, depth: esig.stream2sig(path, depth)
        _HAS_ESIG = True
        print("[signature] Using esig.stream2sig for signature computation")
    else:
        print("[signature] WARNING: esig found but no compatible API detected")
        
except ImportError as e:
    print(f"[signature] esig not available: {e}")

# Fallback to iisignature
_HAS_IISIG = False
if not _HAS_ESIG:
    try:
        import iisignature
        _HAS_IISIG = True
        print("[signature] Using iisignature for signature computation")
    except ImportError as e:
        print(f"[signature] iisignature not available: {e}")
        print("[signature] WARNING: Using numpy fallback (slower, depth ≤ 2 only).")


def compute_signature(path: np.ndarray, depth: int) -> np.ndarray:
    """Compute the truncated signature of a path up to given depth."""
    if path.shape[0] < 2:
        d = path.shape[1]
        dim = _sig_dim(d, depth)
        return np.zeros(dim, dtype=np.float32)

    path = path.astype(np.float64)

    if _HAS_ESIG and _ESIG_SIG_FUNC is not None:
        try:
            return _esig_compute(path, depth)
        except Exception as e:
            print(f"[signature] esig error: {e}")
            if _HAS_IISIG:
                return _iisig_compute(path, depth)
            else:
                return _numpy_sig_depth2(path)
    elif _HAS_IISIG:
        return _iisig_compute(path, depth)
    else:
        if depth > 2:
            print(f"[signature] WARNING: depth {depth} > 2 not supported in numpy fallback, using depth 2")
        return _numpy_sig_depth2(path)


def _esig_compute(path: np.ndarray, depth: int) -> np.ndarray:
    """Use esig for fast signature computation."""
    sig = _ESIG_SIG_FUNC(path, depth)
    return np.array(sig).astype(np.float32)


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
    """Manual depth-2 signature using numpy."""
    T, d = path.shape
    dX = np.diff(path, axis=0)

    level1 = dX.sum(axis=0)
    level2 = np.zeros((d, d), dtype=np.float64)
    cum = np.zeros(d, dtype=np.float64)
    for t in range(T - 1):
        dx = dX[t]
        level2 += np.outer(cum, dx)
        cum += dx
    level2 = level2.flatten()

    return np.concatenate([level1, level2]).astype(np.float32)


def _sig_dim(d: int, depth: int) -> int:
    """Total number of signature terms for dimension d, depth k."""
    if d == 1:
        return depth
    return int(d * (d ** depth - 1) / (d - 1))


def batch_signatures(paths: list[np.ndarray], depth: int) -> np.ndarray:
    """Compute signatures for a list of paths."""
    sigs = [compute_signature(p, depth) for p in paths]
    return np.stack(sigs, axis=0)


# Add this function to fix the import error
def clear_signature_cache():
    """
    Dummy cache clear function for API compatibility.
    esig doesn't use a cache, so this is a no-op.
    """
    pass
