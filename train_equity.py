"""
P2-ETF-SIGNATURE-ENGINE  ·  train_equity.py
Full training pipeline for Equity Sectors module.
Mirrors train_fi.py exactly — two predictions per run:
  (a) Full dataset model (2008-present, 80/10/10)
  (b) Expanding windows consensus (7 start years, each 80/10/10)
Downloads existing signal JSON from HF before writing to preserve FI keys.
"""

import os, json, datetime
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from config import (
    HF_DATASET_OUT, OUTPUT_JSON,
    SIGNAL_HISTORY_EQ,
    METRICS_FULL_EQ, METRICS_WINDOWS_EQ,
    EXPANDING_START_YEARS,
)
from loader        import get_module_data
from features      import build_feature_matrix, build_live_feature
from model         import train_model, predict, select_best_model
from optimise      import optimise_hyperparams
from backtest      import run_backtest
from regime        import fit_regime_model, predict_regime
from scorer        import score_from_predictions, consensus_score, build_signal
from calendar_utils import next_trading_day
from upload        import upload_results


def run_equity():
    print("=" * 60)
    print("P2-ETF-SIGNATURE-ENGINE  |  Module: Equity Sectors")
    print("=" * 60)

    # ── 1. Load full dataset ───────────────────────────────────────
    print("\n[1/8] Loading full dataset (2008-present)...")
    data    = get_module_data("EQ")
    rets    = data["returns"]
    macro   = data["macro"]
    bm_r    = data["benchmark"]
    etfs    = data["etfs"]

    train_r, train_m = data["splits"]["train"]
    val_r,   val_m   = data["splits"]["val"]
    test_r,  test_m  = data["splits"]["test"]

    print(f"      ETFs       : {etfs}")
    print(f"      Total days : {len(rets)}  ({rets.index[0].date()} → {rets.index[-1].date()})")
    print(f"      Train      : {len(train_r)}  ({train_r.index[0].date()} → {train_r.index[-1].date()})")
    print(f"      Val        : {len(val_r)}   ({val_r.index[0].date()} → {val_r.index[-1].date()})")
    print(f"      Test       : {len(test_r)}   ({test_r.index[0].date()} → {test_r.index[-1].date()})")

    # ── 2. Hyperparameter optimisation ────────────────────────────
    print("\n[2/8] Optimising hyperparameters on val set (18 combos)...")
    hp = optimise_hyperparams(rets, macro, train_r, train_m, val_r, val_m, verbose=True)
    lb, depth, mt = hp["best_lookback"], hp["best_depth"], hp["best_model"]
    print(f"      Locked: lookback={lb}  depth={depth}  model={mt}")

    # ── 3. Option (a): Full dataset training ───────────────────────
    print(f"\n[3/8] Option (a) — Full dataset training...")
    X_train_full, y_train_full, _ = build_feature_matrix(train_r, train_m, lb, depth, verbose=True)
    combined_rv = pd.concat([train_r, val_r])
    combined_mv = pd.concat([train_m, val_m])
    X_all_v, y_all_v, _ = build_feature_matrix(combined_rv, combined_mv, lb, depth)
    val_mask = slice(len(X_train_full), None)
    Xv, yv   = X_all_v[val_mask], y_all_v[val_mask]

    models_ridge = train_model(X_train_full, y_train_full, "ridge")
    models_lasso = train_model(X_train_full, y_train_full, "lasso")
    full_models, _ = select_best_model(Xv, yv, models_ridge, models_lasso)

    # ── 4. Option (a): Backtest ────────────────────────────────────
    print(f"\n[4/8] Option (a) — Backtest on test set...")
    bt_full = run_backtest(
        test_r, test_m, rets, macro,
        full_models, lb, depth, etfs,
        bm_r.reindex(test_r.index), verbose=True
    )

    # ── 5. Option (a): Live prediction ────────────────────────────
    print(f"\n[5/8] Option (a) — Live prediction...")
    X_live_full    = build_live_feature(rets, macro, lb, depth)
    preds_full     = predict(full_models, X_live_full)[0]

    regime_model   = fit_regime_model(macro)
    latest_macro   = macro.iloc[-1]
    regime_id      = predict_regime(regime_model, latest_macro)
    regime_name    = regime_model["regime_names"].get(regime_id, str(regime_id))
    ntd            = next_trading_day(rets.index[-1].date())

    prev_pick_full = _load_prev_pick_from_hf(SIGNAL_HISTORY_EQ, col="pick_full")
    scores_full    = score_from_predictions(preds_full, etfs, prev_pick_full)
    signal_full    = build_signal(scores_full, "EQ", "full_dataset",
                                   hp, regime_id, regime_name, ntd, latest_macro)

    print(f"      Full dataset pick : {signal_full['pick']}  ({signal_full['conviction_pct']:.1f}%)")

    # ── 6. Option (b): Expanding windows ─────────────────────────
    print(f"\n[6/8] Option (b) — Expanding windows ({len(EXPANDING_START_YEARS)} windows)...")
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
        w_mac  = wd["macro"]
        w_bm   = wd["benchmark"]

        if len(wt_r) < lb + 10:
            print(f"    Skipped: train too short ({len(wt_r)} rows).")
            continue

        try:
            Xwt, ywt, _ = build_feature_matrix(wt_r, wt_m, lb, depth)
            Xwv_all, ywv_all, _ = build_feature_matrix(
                pd.concat([wt_r, wv_r]),
                pd.concat([wt_m, wv_m]),
                lb, depth
            )
            wv_mask = slice(len(Xwt), None)
            Xwv, ywv = Xwv_all[wv_mask], ywv_all[wv_mask]
        except Exception as e:
            print(f"    Feature build failed: {e}")
            continue

        w_models_r = train_model(Xwt, ywt, "ridge")
        w_models_l = train_model(Xwt, ywt, "lasso")
        w_models, _ = select_best_model(Xwv, ywv, w_models_r, w_models_l)

        val_preds  = predict(w_models, Xwv)
        val_picks  = val_preds.argmax(axis=1)
        val_rets_v = ywv[np.arange(len(val_picks)), val_picks]
        val_sharpe = float(val_rets_v.mean() / (val_rets_v.std() + 1e-9) * np.sqrt(252))

        bt_w = run_backtest(
            we_r, we_m, w_rets, w_mac,
            w_models, lb, depth, etfs,
            w_bm.reindex(we_r.index), verbose=False
        )
        oos_cum = float(bt_w["signal_log"]["net_return"].sum()) if not bt_w["signal_log"].empty else 0.0

        X_live_w = build_live_feature(w_rets, w_mac, lb, depth)
        preds_w  = predict(w_models, X_live_w)[0]

        window_results.append({
            "start_year":  start_yr,
            "preds":       preds_w,
            "val_sharpe":  val_sharpe,
            "oos_cum_ret": oos_cum,
        })
        window_metrics.append({
            "start_year": start_yr,
            **bt_w["metrics"],
            "val_sharpe": round(val_sharpe, 3),
        })
        print(f"    OOS cum_ret={oos_cum:.4f}  val_sharpe={val_sharpe:.3f}  "
              f"→ {'included' if oos_cum > 0 else 'EXCLUDED (negative OOS)'}")

    # ── 7. Consensus signal ────────────────────────────────────────
    print(f"\n[7/8] Option (b) — Building consensus signal...")
    prev_pick_cons = _load_prev_pick_from_hf(SIGNAL_HISTORY_EQ, col="pick_consensus")
    scores_cons    = consensus_score(window_results, etfs, prev_pick_cons)
    n_used         = sum(1 for w in window_results if w["oos_cum_ret"] > 0)
    signal_cons    = build_signal(scores_cons, "EQ", "consensus",
                                   hp, regime_id, regime_name, ntd, latest_macro,
                                   n_windows_used=n_used)

    print(f"      Consensus pick    : {signal_cons['pick']}  ({signal_cons['conviction_pct']:.1f}%)")
    print(f"      Windows used      : {n_used} / {len(window_results)}")

    # ── 8. Save and upload ─────────────────────────────────────────
    print(f"\n[8/8] Saving and uploading to Hugging Face...")
    existing = _fetch_signal_json_from_hf()
    existing["EQ_full"]      = signal_full
    existing["EQ_consensus"] = signal_cons
    existing["generated_at"] = datetime.datetime.utcnow().isoformat()
    _save_json(existing, OUTPUT_JSON)

    _save_json(bt_full["metrics"], METRICS_FULL_EQ)
    _save_json(window_metrics,     METRICS_WINDOWS_EQ)
    _append_history(signal_full["pick"], signal_cons["pick"], ntd, SIGNAL_HISTORY_EQ)

    upload_results([OUTPUT_JSON, METRICS_FULL_EQ, METRICS_WINDOWS_EQ, SIGNAL_HISTORY_EQ])
    print("\nDone — Equity module complete.")
    print(f"Signal JSON keys: {list(existing.keys())}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fetch_signal_json_from_hf() -> dict:
    token = os.environ.get("HF_TOKEN")
    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_OUT,
            filename="results/signature_signal.json",
            repo_type="dataset",
            token=token, force_download=True,
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [info] No existing signal JSON ({e}). Starting fresh.")
        return {}


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
        "date":           date_str,
        "pick_full":      pick_full,
        "pick_consensus": pick_cons,
    }])
    header = not os.path.exists(path)
    row.to_csv(path, mode="a", header=header, index=False)


def _save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


if __name__ == "__main__":
    run_equity()
