"""Metrics, regime breakdown and Diebold-Mariano tests. All metrics are computed in
volatility space (annualised decimals) on the same test rows for every model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def qlike(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """QLIKE loss on variances (Patton 2011) - the standard robust loss for volatility."""
    ratio = (y / p) ** 2
    return ratio - np.log(ratio) - 1


def forecast_metrics(y, p, ref) -> dict:
    """y: realised future vol, p: forecast, ref: current trailing vol (known at t)."""
    y, p, ref = map(lambda a: np.asarray(a, float), (y, p, ref))
    err = y - p
    d_true, d_pred = np.sign(y - ref), np.sign(p - ref)
    # directional accuracy is undefined for a forecast identical to the reference
    dir_acc = np.nan if np.allclose(p, ref) else float(np.mean(d_true == d_pred))
    sst = np.sum((np.log(y) - np.log(y).mean()) ** 2)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE_%": float(np.mean(np.abs(err) / y) * 100),
        "QLIKE": float(np.mean(qlike(y, p))),
        "R2_log": float(1 - np.sum((np.log(y) - np.log(p)) ** 2) / sst),
        "DirAcc": dir_acc,
    }


def metrics_table(pred: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    rows = {names[m]: forecast_metrics(pred["y_vol"], pred[m], pred["rv_5"]) for m in names if m in pred}
    return pd.DataFrame(rows).T.sort_values("QLIKE")


def assign_regime(rv21: pd.Series, cuts: tuple[float, float]) -> pd.Series:
    """Ex-ante regime from trailing 21d vol, thresholds fixed from the training period."""
    return pd.cut(rv21, [-np.inf, cuts[0], cuts[1], np.inf], labels=["calm", "normal", "stressed"])


def regime_table(pred: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    rows = []
    for reg, g in pred.groupby("regime", observed=True):
        for m, label in names.items():
            if m in g and len(g) > 5:
                fm = forecast_metrics(g["y_vol"], g[m], g["rv_5"])
                rows.append({"regime": str(reg), "model": label, "n": len(g),
                             "MAE": fm["MAE"], "RMSE": fm["RMSE"], "QLIKE": fm["QLIKE"]})
    return pd.DataFrame(rows)


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int) -> tuple[float, float]:
    """H0: equal expected loss. Negative statistic => model A has lower loss.
    Bartlett-kernel HAC variance with h-1 lags (h-step labels overlap) and the
    Harvey-Leybourne-Newbold small-sample correction."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    dc = d - d.mean()
    lrv = np.sum(dc * dc) / n
    for k in range(1, h):
        lrv += 2 * (1 - k / h) * np.sum(dc[k:] * dc[:-k]) / n
    lrv = max(lrv, 1e-18)
    dm = d.mean() / np.sqrt(lrv / n)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(dm), float(2 * stats.t.sf(abs(dm), df=n - 1))


def dm_table(pred: pd.DataFrame, names: dict[str, str], benchmark: str, h: int) -> pd.DataFrame:
    rows = []
    y = pred["y_vol"].to_numpy()
    for m, label in names.items():
        if m == benchmark or m not in pred:
            continue
        for loss_name, fn in (("QLIKE", qlike), ("SqErr", lambda a, b: (a - b) ** 2)):
            stat, p = diebold_mariano(fn(y, pred[m].to_numpy()), fn(y, pred[benchmark].to_numpy()), h)
            rows.append({"model": label, "benchmark": names[benchmark], "loss": loss_name,
                         "DM_stat": stat, "p_value": p,
                         "verdict": ("better" if stat < 0 else "worse") if p < 0.05 else "no significant difference"})
    return pd.DataFrame(rows)
