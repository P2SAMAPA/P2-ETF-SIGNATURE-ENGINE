"""
P2-ETF-SIGNATURE-ENGINE · train_equity.py
Full training pipeline for Equity Sectors module.
MODIFIED: Uses trained alpha from hyperparameter optimization with stable Ridge.
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from config import (
    HF_DATASET_OUT, MODEL_CANDIDATES,
    SIGNAL_HISTORY_EQ,
    METRICS_FULL_EQ, METRICS_WINDOWS_EQ,
    EXPANDING_START_YEARS,
    RIDGE_ALPHAS,
)
from loader import get_module_data
from features import build_feature_matrix, build_live_feature, clear_signature_cache
from model import train_model, train_model_with_alpha, predict, select_best_model
from optimise import optimise_hyperparams
from backtest import run_backtest
from regime import fit_regime_model, predict_regime
from scorer import score_from_predictions, consensus_score, build_signal
from calendar_utils import next_trading_day
from upload import upload_results

OUTPUT_JSON = os.environ.get('OUTPUT_JSON', 'equity_signal.json')


def run_equity():
    print("=" * 60)
    print("P2-ETF-SIGNATURE-ENGINE | Module: Equity Sectors")
    print("=" * 60)

    # ── 1. Load full dataset ───────────────────────────────────────
    print("\n[1/8] Loading full dataset (2008-present)...")
    data = get_module_data("EQ")
    rets = data["returns"]
    macro = data["macro"]
    bm_r = data["benchmark"]
    etfs = data["etfs"]

    train_r, train_m = data["splits"]["train"]
    val_r, val_m = data["splits"]["val"]
    test_r, test_m = data["splits"]["test"]

    print(f"  ETFs       : {etfs}")
    print(f"  Total days : {len(rets)} ({rets.index[0].date()} → {rets.index[-1].date()})")
    print(f"  Train      : {len(train_r)} ({train_r.index[0].date()} → {train_r.index[-1].date()})")
    print(f"  Val        : {len(val_r)} ({val_r.index[0].date()} → {val_r.index[-1].date()})")
    print(f"  Test       : {len(test_r)} ({test_r.index[0].date()} → {test_r.index[-1].date()})")

    # ── 2. Hyperparameter optimisation ─────────────────────────────
    print("\n[2/8] Optimising hyperparameters on val set...")
    hp = optimise_hyperparams(rets, macro, train_r, train_m, val_r, val_m, verbose=True)
    lb, depth, mt = hp["best_lookback"], hp["best_depth"], hp["best_model"]
    best_alpha = hp.get("best_alpha", 1.0)  # Get best alpha if available
    print(f"  Locked: lookback={lb} depth={depth} model={mt} alpha={best_alpha}")

    # ── 3. Full dataset training ───────────────────────────────────
    print(f"\n[3/8] Full dataset training...")
    X_train_full, y_train_full, _ = build_feature_matrix(train_r, train_m, lb, depth, verbose=True)
    X_val_full, y_val_full, _ = build_feature_matrix(
        pd.concat([train_r, val_r]),
        pd.concat([train_m, val_m]),
        lb, depth
    )
    val_mask = slice(len(X_train_full), None)
    Xv, yv = X_val_full[val_mask], y_val_full[val_mask]

    # Use train_model_with_alpha with the best alpha found
    if mt == "ridge":
        models_ridge = train_model_with_alpha(X_train_full, y_train_full, best_alpha, "ridge")
    else:
        models_ridge = train_model(X_train_full, y_train_full, "ridge")
    
    if "lasso" in MODEL_CANDIDATES:
        models_lasso = train_model(X_train_full, y_train_full, "lasso")
        full_models, _ = select_best_model(Xv, yv, models_ridge, models_lasso)
    else:
        full_models, _ = select_best_model(Xv, yv, models_ridge, None)

    # ── 4. Backtest on test set ────────────────────────────────────
    print(f"\n[4/8] Backtest on test set...")
    bt_full = run_backtest(
        test_r, test_m, rets, macro,
        full_models, lb, depth, etfs,
        bm_r.reindex(test_r.index), verbose=True
    )

    # ── 5. Live prediction ────────────────────────────
    print(f"\n[5/8] Live prediction...")
    X_live_full = build_live_feature(rets, macro, lb, depth)
    preds_full = predict(full_models, X_live_full)[0]

    regime_model = fit_regime_model(macro)
    latest_macro = macro.iloc[-1]
    regime_id = predict_regime(regime_model, latest_macro)
    regime_name = regime_model["regime_names"].get(regime_id, str(regime_id))
    ntd = next_trading_day(rets.index[-1].date())

    prev_pick_full = _load_prev_pick_from_hf(SIGNAL_HISTORY_EQ, col="pick_full")
    scores_full = score_from_predictions(preds_full, etfs, prev_pick_full)
    signal_full = build_signal(scores_full, "EQ", "full_dataset",
                               hp, regime_id, regime_name, ntd, latest_macro)

    print(f"  Full dataset pick : {signal_full['pick']} ({signal_full['conviction_pct']:.1f}%)")

    # ── 6. Expanding windows ─────────────────────────
    print(f"\n[6/8] Expanding windows ({len(EXPANDING_START_YEARS)} windows)...")
    window_results = []
    window_metrics = []

    for start_yr in EXPANDING_START_YEARS:
        start_str = f"{start_yr}-01-01"
        print(f"\n  Window start: {start_str}")
        
        try:
            wd = get_module_data("EQ", start_date=start_str)
        except Exception as e:
            print(f"    Skipped: {e}")
            continue

        wt_r, wt_m = wd["splits"]["train"]
        wv_r, wv_m = wd["splits"]["val"]
        we_r, we_m = wd["splits"]["test"]
        w_rets = wd["returns"]
        w_mac = wd["macro"]
        w_bm = wd["benchmark"]

        if len(wt_r) < lb + 10:
            print(f"    Skipped: train too short ({len(wt_r)} rows).")
            continue

        try:
            Xwt, ywt, _ = build_feature_matrix(wt_r, wt_m, lb, depth, verbose=False)
            Xwv, ywv, _ = build_feature_matrix(
                pd.concat([wt_r, wv_r]),
                pd.concat([wt_m, wv_m]),
                lb, depth,
                verbose=False
            )
            wv_mask = slice(len(Xwt), None)
            Xwv_only, ywv_only = Xwv[wv_mask], ywv[wv_mask]
        except Exception as e:
            print(f"    Feature build failed: {e}")
            continue

        # Use best alpha from hyperparameter optimization
        if mt == "ridge":
            w_models_r = train_model_with_alpha(Xwt, ywt, best_alpha, "ridge")
        else:
            w_models_r = train_model(Xwt, ywt, "ridge")
        
        if "lasso" in MODEL_CANDIDATES:
            w_models_l = train_model(Xwt, ywt, "lasso")
            w_models, _ = select_best_model(Xwv_only, ywv_only, w_models_r, w_models_l)
        else:
            w_models, _ = select_best_model(Xwv_only, ywv_only, w_models_r, None)

        val_preds = predict(w_models, Xwv_only)
        val_picks = val_preds.argmax(axis=1)
        val_rets = ywv_only[np.arange(len(val_picks)), val_picks]
        val_sharpe = float(val_rets.mean() / (val_rets.std() + 1e-9) * np.sqrt(252))

        bt_w = run_backtest(
            we_r, we_m, w_rets, w_mac,
            w_models, lb, depth, etfs,
            w_bm.reindex(we_r.index), verbose=False
        )
        oos_cum = float(bt_w["signal_log"]["net_return"].sum()) if not bt_w["signal_log"].empty else 0.0

        X_live_w = build_live_feature(w_rets, w_mac, lb, depth)
        preds_w = predict(w_models, X_live_w)[0]

        window_results.append({
            "start_year": start_yr,
            "preds": preds_w,
            "val_sharpe": val_sharpe,
            "oos_cum_ret": oos_cum,
        })
        window_metrics.append({
            "start_year": start_yr,
            **bt_w["metrics"],
            "val_sharpe": round(val_sharpe, 3),
        })
        print(f"    OOS cum_ret={oos_cum:.4f} val_sharpe={val_sharpe:.3f} "
              f"→ {'included' if oos_cum > 0 else 'EXCLUDED (negative OOS)'}")

    clear_signature_cache()

    # ── 7. Consensus signal ────────────────────────────────────────
    print(f"\n[7/8] Building consensus signal...")
    prev_pick_cons = _load_prev_pick_from_hf(SIGNAL_HISTORY_EQ, col="pick_consensus")
    scores_cons = consensus_score(window_results, etfs, prev_pick_cons)
    n_used = sum(1 for w in window_results if w["oos_cum_ret"] > 0)
    signal_cons = build_signal(scores_cons, "EQ", "consensus",
                               hp, regime_id, regime_name, ntd, latest_macro,
                               n_windows_used=n_used)

    print(f"  Consensus pick : {signal_cons['pick']} ({signal_cons['conviction_pct']:.1f}%)")
    print(f"  Windows used   : {n_used} / {len(window_results)}")

    # ── 8. Save and upload ─────────────────────────────────────────
    print(f"\n[8/8] Saving and uploading...")
    
    output_data = {
        "EQ_full": signal_full,
        "EQ_consensus": signal_cons,
        "generated_at": datetime.datetime.utcnow().isoformat()
    }
    
    _save_json(output_data, OUTPUT_JSON)
    _save_json(bt_full["metrics"], METRICS_FULL_EQ)
    _save_json(window_metrics, METRICS_WINDOWS_EQ)

    _append_history(signal_full["pick"], signal_cons["pick"], ntd, SIGNAL_HISTORY_EQ)

    upload_results([OUTPUT_JSON, METRICS_FULL_EQ, METRICS_WINDOWS_EQ, SIGNAL_HISTORY_EQ])
    print("\nDone — Equity module complete.")


def _load_prev_pick_from_hf(csv_filename: str, col: str = "pick_full") -> str | None:
    token = os.environ.get("HF_TOKEN")
    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_OUT,
            filename=f"results/{csv_filename}",
            repo_type="dataset",
            token=token, force_download=True,
        )
        df = pd.read_csv(path)
        if col in df.columns and len(df) > 0:
            return str(df[col].iloc[-1])
    except Exception:
        pass
    return None


def _append_history(pick_full: str, pick_cons: str, date_str: str, path: str):
    row = pd.DataFrame([{
        "date": date_str,
        "pick_full": pick_full,
        "pick_consensus": pick_cons,
    }])
    header = not os.path.exists(path)
    row.to_csv(path, mode="a", header=header, index=False)


def _save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


if __name__ == "__main__":
    run_equity()
