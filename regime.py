"""
P2-ETF-SIGNATURE-ENGINE  ·  regime.py
KMeans regime detection on macro features.
Regime label conditions the display (macro context pill) in the Streamlit UI.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import N_REGIMES, REGIME_FEATURES, RANDOM_SEED


def fit_regime_model(macro_df: pd.DataFrame) -> dict:
    feat   = macro_df[REGIME_FEATURES].ffill().dropna()
    scaler = StandardScaler()
    X      = scaler.fit_transform(feat.values)

    km     = KMeans(n_clusters=N_REGIMES, random_state=RANDOM_SEED, n_init=10)
    km.fit(X)

    labels = pd.Series(km.labels_, index=feat.index, name="regime")

    # Sort regimes by VIX level: 0 = low vol (risk-on), 2 = high vol (risk-off)
    centers_raw = scaler.inverse_transform(km.cluster_centers_)
    centers_df  = pd.DataFrame(centers_raw, columns=REGIME_FEATURES)
    vix_order   = centers_df["VIX"].argsort().values
    remap        = {old: new for new, old in enumerate(vix_order)}
    labels       = labels.map(remap)

    regime_names = {
        0: "Risk-On (Low Vol)",
        1: "Transitional",
        2: "Risk-Off (High Vol / Stress)",
    }

    return {
        "kmeans":       km,
        "scaler":       scaler,
        "labels":       labels,
        "centers":      centers_df,
        "regime_names": regime_names,
        "remap":        remap,
    }


def predict_regime(regime_model: dict, macro_row: pd.Series) -> int:
    feat      = macro_row[REGIME_FEATURES].values.reshape(1, -1)
    X         = regime_model["scaler"].transform(feat)
    raw_label = int(regime_model["kmeans"].predict(X)[0])
    return regime_model["remap"].get(raw_label, raw_label)
