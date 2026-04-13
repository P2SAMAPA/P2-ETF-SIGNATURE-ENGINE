"""
P2-ETF-SIGNATURE-ENGINE · optimise.py
Optimized grid search with minimal memory usage and numerical stability.
Uses explicit alpha tuning with SVD solver for Ridge.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from config import (
    LOOKBACK_CANDIDATES, DEPTH_CANDIDATES,
    MODEL_CANDIDATES, TRANSACTION_COST_BPS, RIDGE_ALPHAS
)
from features import build_feature_matrix
from model import train_model_with_alpha, predict


def _val_cumulative_return(X_train, y_train, X_val, y_val, model_type: str, alpha: float = None) -> float:
    """
    Train on train split, evaluate cumulative return on val split.
    For Ridge, we use explicit alpha with SVD solver for stability.
    """
    tc = TRANSACTION_COST_BPS / 10_000.0
    
    if model_type == "ridge" and alpha is not None:
        models = train_model_with_alpha(X_train, y_train, alpha, model_type)
    else:
        # Fallback for lasso or ridge without alpha (uses default)
        from model import train_model
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
    Grid search over (lookback, depth, alpha) on the validation set.
    Memory-efficient: processes one (lb, depth) combo at a time.
    Uses SVD solver for Ridge to ensure numerical stability.
    """
    all_scores = {}
    best_score = -np.inf
    best_params = {"lookback": 30, "depth": 2, "model": "ridge", "alpha": 1.0}

    combined_ret = pd.concat([train_returns, val_returns])
    combined_mac = pd.concat([train_macro, val_macro])

    n_combos = len(LOOKBACK_CANDIDATES) * len(DEPTH_CANDIDATES) * len(MODEL_CANDIDATES) * len(RIDGE_ALPHAS)
    done = 0

    if verbose:
        print(f"  Testing {n_combos} combinations...")

    for lb in LOOKBACK_CANDIDATES:
        for depth in DEPTH_CANDIDATES:
            try:
                # Build features for this (lb, depth) combo
                X_all, y_all, dates = build_feature_matrix(
                    combined_ret, combined_mac, lb, depth, verbose=False
                )
            except Exception as e:
                print(f"  [optimise] skip lb={lb} d={depth}: {e}")
                continue
            
            # Split at val boundary
            val_start = val_returns.index[0]
            mask_val = dates >= val_start
            mask_train = ~mask_val

            if mask_train.sum() < 10 or mask_val.sum() < 5:
                continue

            X_tr, y_tr = X_all[mask_train], y_all[mask_train]
            X_v, y_v = X_all[mask_val], y_all[mask_val]

            for mt in MODEL_CANDIDATES:
                if mt == "ridge":
                    # For Ridge, tune alpha as well
                    for alpha in RIDGE_ALPHAS:
                        try:
                            score = _val_cumulative_return(X_tr, y_tr, X_v, y_v, mt, alpha)
                            all_scores[(lb, depth, mt, alpha)] = score
                            done += 1

                            if verbose:
                                print(f"  [{done:4d}/{n_combos}] lb={lb:3d} depth={depth} "
                                      f"model={mt:5s} alpha={alpha:6.2f} val_cum_ret={score:.5f}")

                            if score > best_score:
                                best_score = score
                                best_params = {"lookback": lb, "depth": depth, "model": mt, "alpha": alpha}
                        except Exception as e:
                            print(f"  [optimise] error lb={lb} d={depth} mt={mt} alpha={alpha}: {e}")
                            continue
                else:
                    # Lasso (kept for compatibility)
                    try:
                        score = _val_cumulative_return(X_tr, y_tr, X_v, y_v, mt, None)
                        all_scores[(lb, depth, mt)] = score
                        done += 1

                        if verbose:
                            print(f"  [{done:4d}/{n_combos}] lb={lb:3d} depth={depth} "
                                  f"model={mt:5s} val_cum_ret={score:.5f}")

                        if score > best_score:
                            best_score = score
                            best_params = {"lookback": lb, "depth": depth, "model": mt, "alpha": None}
                    except Exception as e:
                        print(f"  [optimise] error lb={lb} d={depth} mt={mt}: {e}")
                        continue

            # Delete features to free memory before next combo
            del X_all, y_all, X_tr, y_tr, X_v, y_v
            import gc
            gc.collect()

    if verbose:
        alpha_str = f" alpha={best_params.get('alpha', 'N/A')}" if best_params.get('alpha') else ""
        print(f"\n  Best: lookback={best_params['lookback']} "
              f"depth={best_params['depth']} model={best_params['model']}{alpha_str} "
              f"val_cum_ret={best_score:.5f}")

    # Return best alpha if found
    result = {
        "best_lookback": best_params["lookback"],
        "best_depth": best_params["depth"],
        "best_model": best_params["model"],
        "best_score": best_score,
        "all_scores": {str(k): v for k, v in all_scores.items()},
    }
    
    if "alpha" in best_params and best_params["alpha"] is not None:
        result["best_alpha"] = best_params["alpha"]
    
    return result
