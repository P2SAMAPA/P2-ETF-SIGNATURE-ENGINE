"""
P2-ETF-SIGNATURE-ENGINE  ·  model.py
Train Ridge or Lasso regression on signature features to predict
next-day log returns for each ETF independently.

One model is fitted per ETF (multi-output via separate univariate fits)
so Lasso can zero out irrelevant signature terms per ETF rather than
applying a single global sparsity.
"""

from __future__ import annotations
import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import RIDGE_ALPHAS, LASSO_ALPHAS, RANDOM_SEED


def train_model(X_train: np.ndarray,
                y_train: np.ndarray,
                model_type: str = "ridge") -> list:
    """
    Train one Ridge/Lasso model per ETF column.

    Parameters
    ----------
    X_train    : np.ndarray  shape (N_train, sig_dim)
    y_train    : np.ndarray  shape (N_train, n_etfs)
    model_type : "ridge" | "lasso"

    Returns
    -------
    list of fitted sklearn Pipeline objects, one per ETF
    """
    n_etfs   = y_train.shape[1]
    models   = []

    for i in range(n_etfs):
        y_col = y_train[:, i]

        if model_type == "ridge":
            reg = RidgeCV(alphas=RIDGE_ALPHAS, cv=5)
        else:
            reg = LassoCV(alphas=LASSO_ALPHAS, cv=5,
                          max_iter=5000, random_state=RANDOM_SEED)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("reg",    reg),
        ])
        pipe.fit(X_train, y_col)
        models.append(pipe)

    return models


def predict(models: list, X: np.ndarray) -> np.ndarray:
    """
    Generate return forecasts for all ETFs.

    Parameters
    ----------
    models : list of fitted pipelines (one per ETF)
    X      : np.ndarray  shape (N, sig_dim)

    Returns
    -------
    np.ndarray  shape (N, n_etfs)  predicted log returns
    """
    preds = np.stack([m.predict(X) for m in models], axis=1)
    return preds.astype(np.float32)


def select_best_model(X_val: np.ndarray,
                      y_val: np.ndarray,
                      models_ridge: list,
                      models_lasso: list) -> tuple[list, str]:
    """
    Select Ridge vs Lasso based on cumulative return on the val set.
    For each candidate, pick the ETF with the highest predicted return
    each day and sum up the realised returns.

    Returns
    -------
    (best_models, best_type)  where best_type is "ridge" or "lasso"
    """
    results = {}
    for name, models in [("ridge", models_ridge), ("lasso", models_lasso)]:
        preds   = predict(models, X_val)           # (N_val, n_etfs)
        picks   = preds.argmax(axis=1)             # index of best ETF each day
        cum_ret = float(y_val[np.arange(len(picks)), picks].sum())
        results[name] = cum_ret

    best = max(results, key=results.get)
    print(f"  [model] Val cumulative return — Ridge: {results['ridge']:.4f}  Lasso: {results['lasso']:.4f}")
    print(f"  [model] Selected: {best}")
    return (models_ridge if best == "ridge" else models_lasso), best
