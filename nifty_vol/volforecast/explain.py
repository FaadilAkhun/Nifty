"""SHAP explanations for the tree models (explains predicted LOG volatility)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def compute_shap(estimator, X: pd.DataFrame, max_rows: int = 600) -> dict:
    Xs = X.sample(min(max_rows, len(X)), random_state=config.SEED).sort_index()
    try:
        import shap

        expl = shap.TreeExplainer(estimator)
        values = np.asarray(expl.shap_values(Xs))
        base = float(np.ravel(expl.expected_value)[0])
    except Exception as exc:  # fall back to XGBoost's native SHAP (pred_contribs)
        if not hasattr(estimator, "get_booster"):
            raise
        import xgboost as xgb

        print(f"[warn] shap.TreeExplainer failed ({exc}); using XGBoost native contributions.")
        contrib = estimator.get_booster().predict(xgb.DMatrix(Xs), pred_contribs=True)
        values, base = contrib[:, :-1], float(contrib[0, -1])
    imp = pd.Series(np.abs(values).mean(axis=0), index=Xs.columns).sort_values(ascending=False)
    return {"values": values, "X": Xs, "base": base, "importance": imp}
