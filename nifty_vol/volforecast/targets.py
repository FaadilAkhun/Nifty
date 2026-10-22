"""Volatility target.

At row t (information set = everything up to the close of day t) the target is the
annualised realised volatility over the NEXT `h` trading days:

    RV(t, t+h) = sqrt( ANN / h * sum_{i=1..h} r_{t+i}^2 ),   r = log return

The model predicts log(RV). Working in logs stabilises the variance of the target
and guarantees positive forecasts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def forward_realized_vol(r: pd.Series, h: int = config.HORIZON, ann: int = config.ANN_FACTOR) -> pd.Series:
    sq = r**2
    # rolling(h).sum() at row s covers r_{s-h+1..s}; shifting by -h gives r_{t+1..t+h} at row t.
    fwd = sq.rolling(h).sum().shift(-h)
    return np.sqrt(ann / h * fwd).clip(lower=config.VOL_FLOOR)
