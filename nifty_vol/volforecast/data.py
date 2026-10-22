"""Data loading, cleaning and validation.

Sources
-------
yahoo      : ^NSEI and ^INDIAVIX through the open-source `yfinance` package
csv        : files you download yourself (e.g. from NSE / niftyindices.com)
synthetic  : simulated GARCH data. ONLY for smoke-testing the pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ----------------------------------------------------------------- loaders --
def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download_yahoo(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(
            f"No data returned for {ticker}. Check your internet connection, or use "
            "--nifty-csv/--vix-csv with files downloaded from NSE."
        )
    df = _flatten(df)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep].copy()


def read_ohlc_csv(path: str) -> pd.DataFrame:
    """Read an OHLC(V) csv. Column names are matched case-insensitively; NSE's
    'Shares Traded' is treated as Volume."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip().title() for c in df.columns]
    df = df.rename(columns={"Shares Traded": "Volume"})
    if "Date" not in df.columns:
        raise ValueError(f"{path}: needs a 'Date' column, found {list(df.columns)}")
    df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(), format="mixed", dayfirst=True)
    df = df.set_index("Date").sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def merge_vix(nifty: pd.DataFrame, vix: pd.DataFrame | None) -> pd.DataFrame:
    df = nifty.copy()
    if vix is not None and "Close" in vix.columns:
        # NSE index and VIX calendars occasionally differ by a day; fill short gaps only.
        df["VIX"] = vix["Close"].reindex(df.index).ffill(limit=3)
    return df


# --------------------------------------------------------------- validation --
def clean_and_validate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rep: dict = {"rows_raw": int(len(df))}
    df = df[~df.index.duplicated(keep="last")].sort_index()
    px = ["Open", "High", "Low", "Close"]
    bad = df[px].isna().any(axis=1) | (df[px] <= 0).any(axis=1)
    rep["dropped_nan_or_nonpositive"] = int(bad.sum())
    df = df.loc[~bad].copy()

    hi = df[px].max(axis=1)
    lo = df[px].min(axis=1)
    rep["high_low_repaired"] = int(((df["High"] < hi) | (df["Low"] > lo)).sum())
    df["High"], df["Low"] = hi, lo

    r = np.log(df["Close"]).diff()
    rep["abs_return_over_15pct"] = int((r.abs() > 0.15).sum())
    rep["gaps_over_7_days"] = int((df.index.to_series().diff().dt.days > 7).sum())
    rep["start"], rep["end"] = str(df.index[0].date()), str(df.index[-1].date())
    rep["rows_clean"] = int(len(df))
    if "VIX" in df:
        rep["vix_missing_pct"] = round(float(df["VIX"].isna().mean() * 100), 2)
    if "Volume" in df:
        rep["volume_zero_or_missing_pct"] = round(float(((df["Volume"] <= 0) | df["Volume"].isna()).mean() * 100), 2)
    return df, rep


# ---------------------------------------------------------------- synthetic --
def make_synthetic(n: int = 4500, seed: int = 0) -> pd.DataFrame:
    """GARCH(1,1)-t simulation with stress episodes. NOT market data."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta, nu = 1.5e-6, 0.08, 0.90, 6
    mult = np.ones(n)
    for s in rng.choice(np.arange(300, n - 60), size=8, replace=False):
        mult[s : s + 25] = 2.5
    sig2 = np.empty(n)
    eps = np.empty(n)
    sig2[0] = omega / (1 - alpha - beta)
    for t in range(n):
        if t > 0:
            sig2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sig2[t - 1]
        z = rng.standard_t(nu) / np.sqrt(nu / (nu - 2))
        eps[t] = np.sqrt(sig2[t]) * z       # GARCH innovation (feeds the recursion)
    r = eps * mult                          # stress episodes scale observed returns only
    close = 4000 * np.exp(np.cumsum(r))
    prev = np.r_[close[0], close[:-1]]
    open_ = prev * np.exp(0.3 * np.sqrt(sig2) * rng.standard_normal(n))
    spread = np.sqrt(sig2) * mult
    high = np.maximum(open_, close) * np.exp(np.abs(rng.standard_normal(n)) * 0.5 * spread)
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.standard_normal(n)) * 0.5 * spread)
    sig2_next = np.r_[sig2[1:], sig2[-1]]     # known at t from the GARCH recursion
    vix = 100 * np.sqrt(252 * sig2_next) * mult * 1.1 * np.exp(0.1 * rng.standard_normal(n))
    volume = np.exp(rng.normal(19, 0.3, n))
    idx = pd.bdate_range(config.START_DATE, periods=n, name="Date")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume, "VIX": vix}, index=idx
    )


# ------------------------------------------------------------------- entry --
def load_market_data(source: str = "yahoo", nifty_csv: str | None = None, vix_csv: str | None = None,
                     refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if source == "synthetic":
        df = make_synthetic()
    elif source == "csv":
        if not nifty_csv:
            raise ValueError("--nifty-csv is required when --source csv")
        df = merge_vix(read_ohlc_csv(nifty_csv), read_ohlc_csv(vix_csv) if vix_csv else None)
    elif source == "yahoo":
        cache = config.DATA_DIR / "nifty_yahoo_cache.csv"
        if cache.exists() and not refresh:
            df = pd.read_csv(cache, index_col="Date", parse_dates=True)
        else:
            nifty = download_yahoo(config.NIFTY_TICKER, config.START_DATE)
            try:
                vix = download_yahoo(config.VIX_TICKER, config.START_DATE)
            except Exception as exc:  # VIX is optional; the project still runs without it
                print(f"[warn] India VIX download failed ({exc}); continuing without VIX features.")
                vix = None
            df = merge_vix(nifty, vix)
            df.to_csv(cache)
    else:
        raise ValueError(f"unknown source {source!r}")
    df, report = clean_and_validate(df)
    report["source"] = source
    return df, report
