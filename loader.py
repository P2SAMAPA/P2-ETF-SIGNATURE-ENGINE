"""
P2-ETF-SIGNATURE-ENGINE  ·  loader.py
Load fi-etf-macro-signal-master-data from Hugging Face, compute log-returns,
apply 80/10/10 chronological split.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datasets import load_dataset

from config import (
    HF_DATASET_IN,
    FI_ETFS, EQUITY_ETFS,
    BENCHMARK_FI, BENCHMARK_EQUITY,
    MACRO_COLS,
    TRAIN_RATIO, VAL_RATIO,
)


def load_raw() -> pd.DataFrame:
    """Download master dataset from HF and return clean DataFrame indexed by date."""
    ds = load_dataset(HF_DATASET_IN, split="train")
    df = ds.to_pandas()
    if "__index_level_0__" in df.columns:
        df = df.rename(columns={"__index_level_0__": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns. First row dropped (NaN)."""
    return np.log(prices / prices.shift(1)).iloc[1:]


def get_module_data(module: str, start_date: str | None = None) -> dict:
    """
    Prepare returns and macro for one module ('FI' or 'EQ').

    Parameters
    ----------
    module     : 'FI' or 'EQ'
    start_date : optional ISO date string to trim the start of history
                 (used for expanding windows, e.g. '2012-01-01')

    Returns
    -------
    dict with keys:
        returns   – pd.DataFrame  log returns for module ETFs
        benchmark – pd.Series     log returns for benchmark
        macro     – pd.DataFrame  macro features (forward-filled)
        splits    – dict {train, val, test} each a tuple (returns, macro)
        etfs      – list[str]
        bm_name   – str
        start_date – str  actual start of data used
    """
    raw = load_raw()

    if start_date:
        raw = raw[raw.index >= start_date]

    if module == "FI":
        etfs    = FI_ETFS
        bm_name = BENCHMARK_FI
    else:
        etfs    = EQUITY_ETFS
        bm_name = BENCHMARK_EQUITY

    all_tickers = etfs + [bm_name]

    # Safety: drop any tickers not present in dataset
    available = [t for t in all_tickers if t in raw.columns]
    missing   = [t for t in all_tickers if t not in raw.columns]
    if missing:
        print(f"  [loader] WARNING: dropping tickers not in dataset: {missing}")

    etfs    = [t for t in etfs if t in available]
    bm_name = bm_name if bm_name in available else None
    if bm_name is None:
        raise ValueError(f"Benchmark not found in dataset columns.")

    prices = raw[etfs + [bm_name]].ffill().dropna()
    rets   = log_returns(prices[etfs])
    bm_r   = log_returns(prices[[bm_name]])[bm_name]

    # Macro: forward-fill (FRED updates weekly/monthly)
    macro  = raw[MACRO_COLS].ffill()
    macro  = macro.reindex(rets.index, method="ffill").dropna()

    common = rets.index.intersection(macro.index)
    rets   = rets.loc[common]
    bm_r   = bm_r.loc[common]
    macro  = macro.loc[common]

    n       = len(rets)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": (rets.iloc[:n_train],                   macro.iloc[:n_train]),
        "val":   (rets.iloc[n_train:n_train + n_val],    macro.iloc[n_train:n_train + n_val]),
        "test":  (rets.iloc[n_train + n_val:],           macro.iloc[n_train + n_val:]),
    }

    return {
        "returns":    rets,
        "benchmark":  bm_r,
        "macro":      macro,
        "splits":     splits,
        "etfs":       etfs,
        "bm_name":    bm_name,
        "start_date": str(rets.index[0].date()),
    }
