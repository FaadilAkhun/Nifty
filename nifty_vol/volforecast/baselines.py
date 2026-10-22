"""Statistical baselines. All return annualised volatility forecasts for the next h days
(decimal units, e.g. 0.15 = 15%)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def naive_rv(ds: pd.DataFrame, window: int) -> pd.Series:
    """'Tomorrow looks like the recent past': trailing realised vol."""
    return np.exp(ds[f"log_rv_{window}"])


def ewma_vol(r: pd.Series, lam: float = 0.94, ann: int = config.ANN_FACTOR) -> pd.Series:
    """RiskMetrics EWMA. Flat term structure, so the h-day forecast equals the 1-day one."""
    var = (r**2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(ann * var)


def garch_forecast(r: pd.Series, fit_end: pd.Timestamp, horizon: int = config.HORIZON,
                   ann: int = config.ANN_FACTOR, dist: str = "t") -> pd.Series:
    """GARCH(1,1) fitted ONCE on data up to `fit_end` (parameters frozen afterwards).
    For each later origin t the conditional variance is filtered through data up to t and
    the h-step-ahead variance path is averaged into an annualised h-day vol forecast.
    Parameters are not re-estimated through the test period (documented limitation)."""
    from arch import arch_model

    x = r.dropna() * 100
    res = arch_model(x, mean="Constant", vol="GARCH", p=1, q=1, dist=dist).fit(last_obs=fit_end, disp="off")
    fc = res.forecast(horizon=horizon, start=fit_end, reindex=False, method="analytic")
    avg_var = fc.variance.mean(axis=1)            # % ^2 per day
    out = np.sqrt(avg_var * ann) / 100
    out.name = "garch"
    return out
