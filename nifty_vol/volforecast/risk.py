"""Translate volatility forecasts into risk analytics.

Important: a volatility forecast is NOT a drawdown forecast. `drawdown_series` is a purely
historical descriptive indicator; `expected_move` converts a vol forecast into a +/-1 sigma
band for the horizon (a distributional statement, not a loss bound)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def fit_thresholds(train_vol: pd.Series, pcts=config.RISK_PERCENTILES) -> dict:
    lo, hi = np.percentile(train_vol.dropna(), pcts)
    return {"low": float(lo), "high": float(hi), "percentiles": list(pcts)}


def classify(vol: float, thr: dict) -> str:
    return "LOW" if vol < thr["low"] else ("HIGH" if vol > thr["high"] else "MODERATE")


def trend(pred_vol: float, current_vol: float, tol: float = config.TREND_TOL) -> str:
    ratio = pred_vol / current_vol
    return "INCREASING" if ratio > 1 + tol else ("DECREASING" if ratio < 1 - tol else "STABLE")


def expected_move(vol: float, h: int = config.HORIZON, ann: int = config.ANN_FACTOR) -> float:
    """+/-1 standard deviation of the h-day return implied by an annualised vol."""
    return float(vol * np.sqrt(h / ann))


def drawdown_series(close: pd.Series) -> pd.Series:
    return close / close.cummax() - 1
