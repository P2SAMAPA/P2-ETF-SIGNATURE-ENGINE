"""
P2-ETF-SIGNATURE-ENGINE  ·  scorer.py
Score ETFs from model predicted returns and build the signal dict.

Two scoring paths:
  (a) Full-dataset: single model → ranked ETF list
  (b) Expanding windows: weighted consensus across multiple models
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from config import TRANSACTION_COST_BPS, CONSENSUS_MIN_RETURN


# ── (a) Single model scoring ──────────────────────────────────────────────────

def score_from_predictions(preds: np.ndarray,
                           etfs: list[str],
                           prev_pick: str | None = None) -> pd.DataFrame:
    """
    Rank ETFs by predicted return with 12bps transaction cost on switches.

    Parameters
    ----------
    preds    : np.ndarray  shape (n_etfs,)  predicted next-day log returns
    etfs     : list of ETF tickers
    prev_pick: previous day's pick for transaction cost

    Returns
    -------
    pd.DataFrame sorted by net_score descending
    """
    tc   = TRANSACTION_COST_BPS / 10_000.0
    rows = []

    for i, ticker in enumerate(etfs):
        gross = float(preds[i])
        cost  = tc if (prev_pick is not None and ticker != prev_pick) else 0.0
        net   = gross - cost
        rows.append({"ticker": ticker, "pred_return": gross, "net_score": net})

    df  = pd.DataFrame(rows).sort_values("net_score", ascending=False).reset_index(drop=True)
    df["rank"]           = df.index + 1
    df["conviction_pct"] = _conviction(df["net_score"].values)
    return df


# ── (b) Expanding-window consensus ───────────────────────────────────────────

def consensus_score(window_results: list[dict],
                    etfs: list[str],
                    prev_pick: str | None = None) -> pd.DataFrame:
    """
    Weighted consensus across expanding windows.

    Each window contributes a vote vector (predicted returns per ETF).
    Weight = max(0, val_sharpe) of that window.
    Windows with OOS cumulative return <= CONSENSUS_MIN_RETURN are excluded.

    Parameters
    ----------
    window_results : list of dicts, each with:
        preds       np.ndarray (n_etfs,) predicted returns from this window's model
        val_sharpe  float       Sharpe ratio on this window's val set
        oos_cum_ret float       cumulative log return on this window's test set
        start_year  int         start year of window (for labelling)

    Returns
    -------
    pd.DataFrame  consensus scores with conviction percentages
    """
    tc         = TRANSACTION_COST_BPS / 10_000.0
    votes      = np.zeros(len(etfs))
    total_w    = 0.0
    used        = 0

    for wr in window_results:
        if wr["oos_cum_ret"] <= CONSENSUS_MIN_RETURN:
            continue                             # skip negative-return windows
        w = max(0.0, wr["val_sharpe"])
        if w < 1e-6:
            w = 0.1                              # small floor so all positive windows count
        votes   += w * np.array(wr["preds"])
        total_w += w
        used    += 1

    if total_w < 1e-9 or used == 0:
        # Fallback: simple average across all windows
        votes   = np.mean([wr["preds"] for wr in window_results], axis=0)
        total_w = 1.0

    consensus = votes / total_w

    rows = []
    for i, ticker in enumerate(etfs):
        gross = float(consensus[i])
        cost  = tc if (prev_pick is not None and ticker != prev_pick) else 0.0
        net   = gross - cost
        rows.append({
            "ticker":       ticker,
            "pred_return":  gross,
            "net_score":    net,
            "windows_used": used,
        })

    df  = pd.DataFrame(rows).sort_values("net_score", ascending=False).reset_index(drop=True)
    df["rank"]           = df.index + 1
    df["conviction_pct"] = _conviction(df["net_score"].values)
    return df


def _conviction(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.min()
    total   = shifted.sum()
    if total < 1e-12:
        return np.full(len(scores), round(100 / len(scores), 1))
    return np.round(100.0 * shifted / total, 1)


# ── Signal dict builder ───────────────────────────────────────────────────────

def build_signal(scores_df: pd.DataFrame,
                 module: str,
                 source: str,               # "full_dataset" | "consensus"
                 hyperparams: dict,
                 regime_id: int,
                 regime_name: str,
                 next_trading_day: str,
                 macro_latest: pd.Series,
                 n_windows_used: int | None = None) -> dict:
    top    = scores_df.iloc[0]
    second = scores_df.iloc[1] if len(scores_df) > 1 else None
    third  = scores_df.iloc[2] if len(scores_df) > 2 else None

    sig = {
        "module":            module,
        "source":            source,
        "next_trading_day":  next_trading_day,
        "pick":              top["ticker"],
        "conviction_pct":    float(top["conviction_pct"]),
        "pred_return":       round(float(top["pred_return"]), 6),
        "second_pick":       second["ticker"] if second is not None else None,
        "second_conviction": float(second["conviction_pct"]) if second is not None else None,
        "third_pick":        third["ticker"] if third is not None else None,
        "third_conviction":  float(third["conviction_pct"]) if third is not None else None,
        "lookback_days":     hyperparams.get("best_lookback"),
        "sig_depth":         hyperparams.get("best_depth"),
        "model_type":        hyperparams.get("best_model"),
        "regime_id":         int(regime_id),
        "regime_name":       regime_name,
        "n_windows_used":    n_windows_used,
        "macro_pills": {
            "VIX":       round(float(macro_latest.get("VIX", 0)), 2),
            "T10Y2Y":    round(float(macro_latest.get("T10Y2Y", 0)), 3),
            "HY_SPREAD": round(float(macro_latest.get("HY_SPREAD", 0)), 2),
            "IG_SPREAD": round(float(macro_latest.get("IG_SPREAD", 0)), 2),
            "DXY":       round(float(macro_latest.get("DXY", 0)), 2),
        },
        "all_scores": scores_df[["ticker", "pred_return", "net_score",
                                  "conviction_pct"]].to_dict(orient="records"),
    }
    return sig
