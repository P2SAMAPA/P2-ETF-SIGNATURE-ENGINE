# P2-ETF-SIGNATURE-ENGINE

**Path Signature Methods for ETF Return Forecasting**
Rough path theory · Iterated integrals · Two prediction modes · CPU-only · 12bps transaction cost

---

## Overview

Path signatures are a mathematically rigorous way to characterise the **shape of a time-series path** as a fixed-length feature vector with provable convergence guarantees. Unlike models that operate on point values or discrete lags, signatures capture the full geometric information of the path — including the order in which events happen and the interaction between channels.

For ETF selection, this means the model learns not just "VIX went up" but "VIX went up *while* TLT was falling *and* HYG spreads were widening" — the joint trajectory of the return + macro path, expressed as algebraic invariants (iterated integrals).

**Two prediction modes per module:**

| Mode | Description |
|------|-------------|
| **(a) Full dataset** | Single model trained on 80% of 2008–present data, validated on next 10%, tested on final 10% |
| **(b) Expanding windows** | 7 models trained on windows starting in 2008, 2010, 2012, 2014, 2016, 2018, 2020 — each with its own 80/10/10 split — consensus-weighted by val Sharpe |

---

## Two Modules

| Module | ETF Universe | Benchmark |
|--------|-------------|-----------|
| **FI / Commodities** | TLT · LQD · HYG · VNQ · GLD · SLV · VCIT | AGG |
| **Equity Sectors**   | QQQ · XLK · XLF · XLE · XLV · XLI · XLY · XLP · XLU · GDX · XME · IWM | SPY |

No CASH output — the engine always picks the best ETF from the universe.

---

## Data

**Source:** [`P2SAMAPA/fi-etf-macro-signal-master-data`](https://huggingface.co/datasets/P2SAMAPA/fi-etf-macro-signal-master-data)

**Period:** 2008-01-01 → present (daily NYSE trading days)

**Features used:**
- ETF prices → daily log-returns (path channels)
- Macro: VIX · DXY · T10Y2Y · TBILL_3M · IG_SPREAD · HY_SPREAD (additional path channels + regime detection)

---

## Path Construction Pipeline

```
Raw returns (T × n_etfs) + Macro (T × 6)
    ↓
Concatenate → base path (T × (n_etfs + 6))
    ↓
Time channel appended → (T × (n_etfs + 7))     captures WHEN events happen
    ↓
Basepoint prepended (row of zeros)              translation invariance
    ↓
Lead-lag transform → (2T-1 × 2(n_etfs+7))      depth-2 terms = realised covariance
    ↓
Truncated signature at depth d → fixed-length feature vector
```

The **lead-lag transformation** doubles the path dimension. This is crucial for financial data: it makes the depth-2 signature term equal to the realised covariance matrix of the return path — capturing the same information as a GARCH-DCC model, but without the parametric assumptions.

---

## Hyperparameter Optimisation

The grid search runs **once on the full-dataset validation set** and locks in the best combination for all subsequent runs (including all expanding windows):

| Parameter | Candidates |
|-----------|-----------|
| Lookback window | 30 · 45 · 60 trading days |
| Signature depth | 2 · 3 · 4 |
| Linear model | Ridge · Lasso |

**18 combinations total.** Selection criterion: cumulative log-return on val set with 12bps transaction cost applied on switches.

---

## Expanding Windows Consensus (Option b)

```
Window 1: 2008 → present  →  80/10/10 split  →  Ridge/Lasso  →  preds, val_sharpe, OOS_cum_ret
Window 2: 2010 → present  →  80/10/10 split  →  Ridge/Lasso  →  preds, val_sharpe, OOS_cum_ret
...
Window 7: 2020 → present  →  80/10/10 split  →  Ridge/Lasso  →  preds, val_sharpe, OOS_cum_ret

Consensus:
  weight(i) = max(0, val_sharpe(i))     [windows with negative OOS return excluded]
  vote(i)   = weight(i) × pred_returns(i)
  final      = Σ vote(i) / Σ weight(i)
  pick       = argmax(final) after 12bps switch penalty
```

---

## Architecture

```
config.py            ETF universes, HF repos, hyperparameter grid, window start years
loader.py            Load HF dataset, log-returns, 80/10/10 chronological split
path_builder.py      Time channel, basepoint, lead-lag transformation
signature.py         Truncated iterated integrals via iisignature (numpy fallback)
features.py          Rolling-window signature matrix builder (X, y, dates)
model.py             Ridge/Lasso per ETF, model selection on val set
optimise.py          Grid search: lookback × depth × model_type on val set
backtest.py          Walk-forward OOS backtest, metrics vs benchmark
regime.py            KMeans K=3 on VIX + T10Y2Y + HY_SPREAD
scorer.py            Single-model scoring + weighted consensus across windows
calendar_utils.py    NYSE next trading day
train_fi.py          Full pipeline: FI module (both options a and b)
train_equity.py      Full pipeline: Equity module (both options a and b)
upload.py            Push results to HF dataset
app.py               Streamlit UI (two tabs × two hero boxes each)
```

---

## Output

Results uploaded to [`P2SAMAPA/p2-etf-signature-engine-results`](https://huggingface.co/datasets/P2SAMAPA/p2-etf-signature-engine-results):

| File | Contents |
|------|---------|
| `results/signature_signal.json` | Daily picks: FI_full, FI_consensus, EQ_full, EQ_consensus |
| `results/signal_history_fi.csv` | Running log of FI picks (full + consensus) |
| `results/signal_history_eq.csv` | Running log of Equity picks |
| `results/metrics_full_fi.json` | OOS backtest metrics — FI full dataset |
| `results/metrics_full_eq.json` | OOS backtest metrics — Equity full dataset |
| `results/metrics_windows_fi.json` | Per-window metrics — FI expanding windows |
| `results/metrics_windows_eq.json` | Per-window metrics — Equity expanding windows |

### Signal JSON schema

```json
{
  "FI_full": {
    "module": "FI",
    "source": "full_dataset",
    "next_trading_day": "2026-04-11",
    "pick": "GLD",
    "conviction_pct": 38.4,
    "pred_return": 0.000312,
    "second_pick": "TLT",
    "lookback_days": 45,
    "sig_depth": 3,
    "model_type": "ridge",
    "regime_name": "Transitional",
    "macro_pills": { "VIX": 18.4, "T10Y2Y": -0.12, ... }
  },
  "FI_consensus": {
    "source": "consensus",
    "n_windows_used": 5,
    ...
  },
  "EQ_full": { ... },
  "EQ_consensus": { ... },
  "generated_at": "2026-04-10T23:05:00"
}
```

---

## Daily Cron Schedule

Runs at **23:00 UTC Mon–Fri** (after data update at 22:00 UTC):
- `train_fi.py` → FI module (runs first)
- `train_equity.py` → Equity module (runs after FI, sequential to preserve JSON keys)

---

## Streamlit UI

The app shows two tabs (FI and Equity). Each tab has:
- **Two hero boxes side by side** — Option A (blue, full dataset) and Option B (green, consensus)
- Each box: ticker, conviction %, next NYSE trading day, lookback, depth, model type, predicted return, regime
- 2nd and 3rd best picks with conviction for each option
- Macro environment pills (VIX, T10Y2Y, HY Spread, IG Spread, DXY)
- Full ETF score table for both options
- OOS backtest metrics (full dataset test set)
- Expanding windows per-window metrics table (with negative-OOS windows highlighted)
- Pick history log

---

## GitHub Actions Setup

1. Fork this repository
2. Add `HF_TOKEN` as a repository secret (Settings → Secrets → Actions)
3. Workflow runs automatically at 23:00 UTC Mon–Fri

Manual run: Actions → Daily Signature Training → Run workflow → choose `fi`, `eq`, or `both`

---

## CPU Requirements (GitHub Actions free tier)

| Step | Est. time per module |
|------|---------------------|
| Load data + hyperparameter grid (18 combos) | ~4 min |
| Full dataset training + backtest | ~2 min |
| 7 expanding windows (train + backtest each) | ~8 min |
| Live prediction + upload | ~1 min |
| **Total per module** | **~15 min** |

Both modules together ~30 minutes — well within the 6-hour free-tier limit.

`iisignature` is a compiled C++ library — signature computation for a 60-day path at depth 3 takes milliseconds per window on CPU.

---

## Key References

- Chen, K.T. (1954). *Iterated path integrals.* Bulletin of the AMS.
- Lyons, T. (1998). *Differential equations driven by rough signals.* Revista Matemática Iberoamericana.
- Chevyrev, I. & Kormilitzin, A. (2016). *A primer on the signature method in machine learning.* arXiv:1603.03788.
- Morrill, J. et al. (2021). *A generalised signature method for time series.* arXiv:2006.00873.
- `iisignature` library: https://github.com/bottler/iisignature

---

## Disclaimer

Research and educational purposes only. Not financial advice. Past performance does not guarantee future results.
