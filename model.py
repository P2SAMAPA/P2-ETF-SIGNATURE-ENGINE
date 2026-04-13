"""
P2-ETF-SIGNATURE-ENGINE · model.py
Train one model per ETF (multi-output Ridge/Lasso) with numerical stability.
Uses SVD solver for Ridge to handle ill-conditioned matrices.
"""

from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from config import RIDGE_ALPHAS, LASSO_ALPHAS, RANDOM_SEED, MODEL_CANDIDATES, RIDGE_SOLVER, USE_FEATURE_SCALING


def train_model(X_train: np.ndarray,
                y_train: np.ndarray,
                model_type: str = "ridge") -> list:
    """
    Train one multi-output model per ETF.
    Returns a list of fitted models (one per ETF).
    
    For Ridge: Uses SVD solver which is numerically stable even with 
    ill-conditioned or singular matrices. The SVD solver works by:
    - Decomposing X = U @ diag(s) @ V^T
    - Computing coefficients as V @ diag(s / (s^2 + alpha)) @ U^T @ y
    This avoids inverting the Gram matrix entirely.
    """
    n_etfs = y_train.shape[1]
    models = []

    for i in range(n_etfs):
        yi = y_train[:, i]

        if model_type == "ridge":
            # Use Ridge with explicit solver='svd' instead of RidgeCV
            # RidgeCV uses dual formulation which is unstable for ill-conditioned matrices
            # We'll do manual cross-validation in optimise.py
            model = Ridge(
                alpha=1.0,  # Default, will be tuned via hyperparameter search
                solver=RIDGE_SOLVER,
                random_state=RANDOM_SEED,
                tol=1e-4,
                max_iter=10000
            )
            
            # Optionally wrap with StandardScaler for feature normalization
            if USE_FEATURE_SCALING:
                model = make_pipeline(StandardScaler(), model)
                
        elif model_type == "lasso":
            model = LassoCV(
                alphas=LASSO_ALPHAS,
                cv=3,
                random_state=RANDOM_SEED,
                max_iter=5000,
                tol=1e-3,
                selection='random'
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.fit(X_train, yi)
        models.append(model)

    return models


def train_model_with_alpha(X_train: np.ndarray,
                           y_train: np.ndarray,
                           alpha: float,
                           model_type: str = "ridge") -> list:
    """
    Train Ridge model with a specific alpha value.
    Used during hyperparameter optimization.
    """
    n_etfs = y_train.shape[1]
    models = []

    for i in range(n_etfs):
        yi = y_train[:, i]
        
        if model_type == "ridge":
            model = Ridge(
                alpha=alpha,
                solver=RIDGE_SOLVER,
                random_state=RANDOM_SEED,
                tol=1e-4,
                max_iter=10000
            )
            if USE_FEATURE_SCALING:
                model = make_pipeline(StandardScaler(), model)
        else:
            model = LassoCV(
                alphas=[alpha],
                cv=3,
                random_state=RANDOM_SEED,
                max_iter=5000,
                tol=1e-3,
                selection='random'
            )
        
        model.fit(X_train, yi)
        models.append(model)
    
    return models


def predict(models: list, X: np.ndarray) -> np.ndarray:
    """
    Predict returns for all ETFs.
    Returns array shape (n_samples, n_etfs).
    """
    preds = np.column_stack([m.predict(X) for m in models])
    return preds


def select_best_model(X_val: np.ndarray,
                      y_val: np.ndarray,
                      models_ridge: list,
                      models_lasso: list | None = None) -> tuple[list, str]:
    """
    Select best model type based on MSE on validation set.
    """
    preds_ridge = predict(models_ridge, X_val)
    mse_ridge = np.mean((preds_ridge - y_val) ** 2)

    if models_lasso is not None:
        preds_lasso = predict(models_lasso, X_val)
        mse_lasso = np.mean((preds_lasso - y_val) ** 2)
        
        if mse_lasso < mse_ridge:
            return models_lasso, "lasso"
    
    return models_ridge, "ridge"
