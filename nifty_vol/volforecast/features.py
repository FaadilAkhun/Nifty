"""Feature engineering. EVERY feature at row t uses only data up to the close of
day t (rolling / ewm windows look backwards only). tests/test_leakage.py checks this
by truncating the data and confirming earlier features do not change."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .targets import forward_realized_vol, log_returns

ANN = config.ANN_FACTOR
FLOOR = config.VOL_FLOOR


def _log_vol(x: pd.Series) -> pd.Series:
    return np.log(x.clip(lower=FLOOR))


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    o, hi, lo, c = df["Open"], df["High"], df["Low"], df["Close"]
    r = log_returns(c)
    pc = c.shift(1)
    f = pd.DataFrame(index=df.index)
    groups: dict[str, list[str]] = {}

    # returns -----------------------------------------------------------------
    f["ret_0"] = r
    for k in range(1, 5):
        f[f"ret_lag{k}"] = r.shift(k)
    f["abs_gap"] = np.abs(np.log(o / pc))
    groups["returns"] = ["ret_0", "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4", "abs_gap"]

    # close-to-close realised vol over several look-backs ------------------------
    rv_cols = []
    for w in (5, 10, 21, 63):
        f[f"log_rv_{w}"] = _log_vol(np.sqrt(ANN * (r**2).rolling(w).mean()))
        rv_cols.append(f"log_rv_{w}")
    groups["realized_vol"] = rv_cols

    # range-based (Parkinson) vol: uses intraday High/Low, more efficient than close-to-close
    f["log_park_21"] = _log_vol(np.sqrt(ANN / (4 * np.log(2)) * (np.log(hi / lo) ** 2).rolling(21).mean()))
    tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["atr_pct_14"] = atr / c
    groups["range_vol"] = ["log_park_21", "atr_pct_14"]

    # technical indicators ------------------------------------------------------
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ema12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    f["macd_pct"] = macd / c
    f["macd_hist_pct"] = (macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()) / c
    sma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_width"] = 4 * sd20 / sma20
    f["bb_pctb"] = (c - (sma20 - 2 * sd20)) / (4 * sd20)
    groups["technical"] = ["rsi_14", "macd_pct", "macd_hist_pct", "bb_width", "bb_pctb"]

    # trend / drawdown state (backward-looking) -----------------------------------
    f["sma_ratio_50"] = c / c.rolling(50).mean() - 1
    f["sma_ratio_200"] = c / c.rolling(200).mean() - 1
    f["dd_from_252d_high"] = c / c.rolling(252, min_periods=60).max() - 1
    groups["trend"] = ["sma_ratio_50", "sma_ratio_200", "dd_from_252d_high"]

    # India VIX (forward-looking implied vol, known at the close of day t) ----------
    if "VIX" in df and df["VIX"].notna().mean() > 0.9:
        vix = df["VIX"]
        f["log_vix"] = np.log(vix / 100)
        f["vix_chg_5"] = np.log(vix).diff(5)
        f["vix_rv_gap"] = f["log_vix"] - f["log_rv_21"]
        groups["vix"] = ["log_vix", "vix_chg_5", "vix_rv_gap"]

    # volume (Yahoo's ^NSEI volume is zero for long stretches -> only use if usable) --
    if "Volume" in df and (df["Volume"] > 0).mean() > 0.9:
        v = df["Volume"].where(df["Volume"] > 0)
        f["vol_ratio_20"] = np.log(v / v.rolling(20).mean())
        groups["volume"] = ["vol_ratio_20"]

    f = f.replace([np.inf, -np.inf], np.nan)
    return f, groups


def assemble_dataset(df: pd.DataFrame, h: int = config.HORIZON):
    """Features + target + helper columns used by baselines/evaluation."""
    feats, groups = build_features(df)
    r = log_returns(df["Close"])
    ds = feats.copy()
    ds["y_vol"] = forward_realized_vol(r, h)          # NaN for the last h rows
    ds["y_log"] = np.log(ds["y_vol"])
    ds["rv_5"] = np.exp(feats["log_rv_5"])            # current (trailing) vol, known at t
    ds["rv_21"] = np.exp(feats["log_rv_21"])
    ds["ret"] = r
    ds["close"] = df["Close"]
    if "VIX" in df:
        ds["vix"] = df["VIX"]
    feature_cols = list(feats.columns)
    return ds, groups, feature_cols
