"""
P2-ETF-SIGNATURE-ENGINE  ·  backtest.py
Walk-forward OOS backtest on the held-out test set.
At each step t, predict next-day returns, pick top ETF, record realised return.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from config import TRANSACTION_COST_BPS
from features import build_feature_matrix
from model    import predict


def run_backtest(test_returns: pd.DataFrame,
                 test_macro: pd.DataFrame,
                 full_returns: pd.DataFrame,
                 full_macro: pd.DataFrame,
                 fitted_models: list,
                 lookback: int,
                 depth: int,
                 etfs: list[str],
                 benchmark_r: pd.Series,
                 verbose: bool = True) -> dict:
    """
    Walk-forward backtest on the test set.

    At each test day t, the feature vector is built from the `lookback`
    days ending at t (which may include rows from the training set —
    this is correct, not leakage, because we only look at the features
    not the targets).

    Parameters
    ----------
    test_returns    : test-set returns (just for y targets + dates)
    full_returns    : full returns DataFrame (train+val+test) for window building
    fitted_models   : list of sklearn pipelines from train_model()
    benchmark_r     : benchmark log returns aligned to test dates

    Returns
    -------
    dict with equity_curve, bm_curve, signal_log, metrics
    """
    tc   = TRANSACTION_COST_BPS / 10_000.0
    rows = []
    prev = None

    test_dates = test_returns.index

    for i, date in enumerate(test_dates[:-1]):
        # Index of `date` in full_returns
        t = full_returns.index.get_loc(date)
        if t < lookback:
            continue

        ret_win = full_returns.iloc[t - lookback: t]
        mac_win = full_macro.reindex(ret_win.index, method="ffill").fillna(0.0)

        from path_builder import build_path
        from signature    import compute_signature
        path  = build_path(ret_win, mac_win)
        sig   = compute_signature(path, depth).reshape(1, -1)

        preds    = predict(fitted_models, sig)[0]   # (n_etfs,)
        best_idx = int(preds.argmax())
        pick     = etfs[best_idx]

        next_date   = test_dates[i + 1]
        actual_r    = float(test_returns.loc[next_date, pick])
        switch      = (prev is not None and best_idx != prev)
        net_r       = actual_r - (tc if switch else 0.0)
        hit         = actual_r > 0

        rows.append({
            "date":            next_date,
            "pick":            pick,
            "pred_return":     float(preds[best_idx]),
            "actual_return":   actual_r,
            "net_return":      net_r,
            "switched":        switch,
            "hit":             hit,
        })
        prev = best_idx

    if not rows:
        return {"equity_curve": pd.Series(dtype=float),
                "bm_curve":     pd.Series(dtype=float),
                "signal_log":   pd.DataFrame(),
                "metrics":      {}}

    log      = pd.DataFrame(rows).set_index("date")
    eq_curve = log["net_return"].cumsum()
    bm_common = benchmark_r.reindex(log.index).fillna(0.0)
    bm_curve  = bm_common.cumsum()

    metrics = _compute_metrics(log, bm_common)
    if verbose:
        print(f"  OOS backtest ({len(log)} days):")
        print(f"    Ann. return : {metrics['ann_return_pct']:.2f}%")
        print(f"    Sharpe      : {metrics['sharpe']:.3f}")
        print(f"    Max DD      : {metrics['max_drawdown_pct']:.2f}%")
        print(f"    Hit rate    : {metrics['hit_rate_pct']:.1f}%")
        print(f"    Alpha vs BM : {metrics['ann_alpha_pct']:.2f}%")

    return {
        "equity_curve": eq_curve,
        "bm_curve":     bm_curve,
        "signal_log":   log,
        "metrics":      metrics,
    }


def _compute_metrics(log: pd.DataFrame, bm_r: pd.Series) -> dict:
    n        = len(log)
    ann_r    = float(log["net_return"].mean() * 252)
    ann_vol  = float(log["net_return"].std() * np.sqrt(252)) + 1e-9
    sharpe   = ann_r / ann_vol

    cum      = np.exp(log["net_return"].cumsum().values)
    peak     = np.maximum.accumulate(cum)
    max_dd   = float(((cum - peak) / peak).min())
    hit_rate = float(log["hit"].mean())

    ann_bm   = float(bm_r.mean() * 252)
    alpha    = ann_r - ann_bm

    log2     = log.copy()
    log2["yr"] = log2.index.year
    yr_ret   = log2.groupby("yr")["net_return"].sum()
    pos_yrs  = int((yr_ret > 0).sum())

    return {
        "n_days":           n,
        "ann_return_pct":   round(ann_r * 100, 2),
        "ann_vol_pct":      round(ann_vol * 100, 2),
        "sharpe":           round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "hit_rate_pct":     round(hit_rate * 100, 1),
        "ann_alpha_pct":    round(alpha * 100, 2),
        "positive_years":   pos_yrs,
    }
