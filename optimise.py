"""
P2-ETF-SIGNATURE-ENGINE · optimise.py
Optimized grid search with feature pre-computation and model-only iteration.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from config import (
    LOOKBACK_CANDIDATES, DEPTH_CANDIDATES,
    MODEL_CANDIDATES, TRANSACTION_COST_BPS,
)
from features import build_feature_matrix, clear_signature_cache
from model import train_model, predict

def _val_cumulative_return(X_train, y_train, X_val, y_val, model_type: str) -> float:
    """Train on train split, evaluate cumulative return on val split."""
    tc = TRANSACTION_COST_BPS / 10_000.0
    models = train_model(X_train, y_train, model_type)
    preds = predict(models, X_val)
    prev = None
    cum_ret = 0.0

    for t in range(len(preds)):
        best_idx = int(preds[t].argmax())
        cost = tc if (prev is not None and best_idx != prev) else 0.0
        cum_ret += float(y_val[t, best_idx]) - cost
        prev = best_idx

    return cum_ret

def optimise_hyperparams(returns_df: pd.DataFrame,
                         macro_df: pd.DataFrame,
                         train_returns: pd.DataFrame,
                         train_macro: pd.DataFrame,
                         val_returns: pd.DataFrame,
                         val_macro: pd.DataFrame,
                         verbose: bool = True) -> dict:
    """
    Optimized grid search: pre-compute features once per (lb, depth) combo,
    then test only models.
    """
    all_scores = {}
    best_score = -np.inf
    best_params = {"lookback": 30, "depth": 2, "model": "ridge"}

    combined_ret = pd.concat([train_returns, val_returns])
    combined_mac = pd.concat([train_macro, val_macro])

    # Pre-compute features for each (lb, depth) combo once
    feature_cache = {}  # (lb, depth) -> (X_all, y_all, dates)
    
    n_combos = len(LOOKBACK_CANDIDATES) * len(DEPTH_CANDIDATES) * len(MODEL_CANDIDATES)
    done = 0

    if verbose:
        print(f"  Pre-computing features for {len(LOOKBACK_CANDIDATES) * len(DEPTH_CANDIDATES)} (lb, depth) combos...")

    for lb in LOOKBACK_CANDIDATES:
        for depth in DEPTH_CANDIDATES:
            try:
                # Build features once per (lb, depth)
                X_all, y_all, dates = build_feature_matrix(
                    combined_ret, combined_mac, lb, depth, verbose=False
                )
                feature_cache[(lb, depth)] = (X_all, y_all, dates)
            except Exception as e:
                print(f"  [optimise] skip lb={lb} d={depth}: {e}")
                continue

    # Now iterate through all combos using cached features
    if verbose:
        print(f"  Testing {n_combos} combinations...")

    for lb in LOOKBACK_CANDIDATES:
        for depth in DEPTH_CANDIDATES:
            if (lb, depth) not in feature_cache:
                continue
                
            X_all, y_all, dates = feature_cache[(lb, depth)]
            
            # Split at val boundary
            val_start = val_returns.index[0]
            mask_val = dates >= val_start
            mask_train = ~mask_val

            if mask_train.sum() < 10 or mask_val.sum() < 5:
                continue

            X_tr, y_tr = X_all[mask_train], y_all[mask_train]
            X_v, y_v = X_all[mask_val], y_all[mask_val]

            for mt in MODEL_CANDIDATES:
                try:
                    score = _val_cumulative_return(X_tr, y_tr, X_v, y_v, mt)
                    all_scores[(lb, depth, mt)] = score
                    done += 1

                    if verbose:
                        print(f"  [{done:2d}/{n_combos}] lb={lb:3d} depth={depth} "
                              f"model={mt:5s} val_cum_ret={score:.5f}")

                    if score > best_score:
                        best_score = score
                        best_params = {"lookback": lb, "depth": depth, "model": mt}

                except Exception as e:
                    print(f"  [optimise] error lb={lb} d={depth} mt={mt}: {e}")
                    continue

    # Clear cache to free memory before expanding windows
    feature_cache.clear()
    clear_signature_cache()

    if verbose:
        print(f"\n  Best: lookback={best_params['lookback']} "
              f"depth={best_params['depth']} model={best_params['model']} "
              f"val_cum_ret={best_score:.5f}")

    return {
        "best_lookback": best_params["lookback"],
        "best_depth": best_params["depth"],
        "best_model": best_params["model"],
        "best_score": best_score,
        "all_scores": {str(k): v for k, v in all_scores.items()},
    }
