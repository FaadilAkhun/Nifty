"""Log-volatility regressors (HAR-type OLS, Random Forest, XGBoost) with a common wrapper."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterSampler
from xgboost import XGBRegressor

from . import config

HAR_COLS = ["log_rv_5", "log_rv_21", "log_rv_63"]


class LogVolRegressor:
    """Wraps an sklearn-style estimator trained on log(RV). `smear` is Duan's smearing
    factor (mean of exp(out-of-sample residual)) so that exp(prediction) estimates the
    conditional MEAN of volatility rather than its median."""

    def __init__(self, estimator, cols: list[str], name: str):
        self.estimator, self.cols, self.name, self.smear = estimator, list(cols), name, 1.0

    def fit(self, X: pd.DataFrame, y_log: pd.Series):
        self.estimator.fit(X[self.cols], y_log)
        return self

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(X[self.cols]))

    def predict_vol(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(X)) * self.smear


def make_har() -> LinearRegression:
    return LinearRegression()


def make_rf(**p) -> RandomForestRegressor:
    return RandomForestRegressor(random_state=config.SEED, n_jobs=-1, **p)


def make_xgb(**p) -> XGBRegressor:
    return XGBRegressor(random_state=config.SEED, n_jobs=-1, objective="reg:squarederror", tree_method="hist", **p)


RF_SPACE = {
    "n_estimators": [200, 300, 500],
    "max_depth": [4, 6, 8, 12, None],
    "min_samples_leaf": [5, 10, 20, 40],
    "max_features": [0.3, 0.5, 0.8, "sqrt"],
}
XGB_SPACE = {
    "n_estimators": [150, 300, 500],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.7, 1.0],
    "min_child_weight": [1, 5, 10, 20],
    "reg_lambda": [1.0, 5.0, 10.0],
}


def tune(factory, space: dict, X_tr, y_tr, X_va, y_va, cols, n_iter: int, seed: int = config.SEED):
    """Random search; candidates are fit on TRAIN only and scored on the chronologically
    later VALIDATION block (RMSE in log-vol space). Returns (best_params, trials table)."""
    results = []
    for params in ParameterSampler(space, n_iter=n_iter, random_state=seed):
        params = {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}
        m = LogVolRegressor(factory(**params), cols, "tmp").fit(X_tr, y_tr)
        rmse = float(np.sqrt(np.mean((y_va - m.predict_log(X_va)) ** 2)))
        results.append((rmse, params))
    results.sort(key=lambda t: t[0])
    trials = pd.DataFrame([{**p, "val_rmse_log": r} for r, p in results]).astype(object)
    return results[0][1], trials
