"""Leakage and correctness tests. Run:  pytest -q"""
import numpy as np
import pandas as pd
import pytest

from volforecast import config
from volforecast.data import make_synthetic
from volforecast.evaluate import diebold_mariano, forecast_metrics
from volforecast.features import assemble_dataset, build_features
from volforecast.splits import chronological_split
from volforecast.targets import forward_realized_vol, log_returns

H = config.HORIZON


@pytest.fixture(scope="module")
def df():
    return make_synthetic(n=1200, seed=3)


def test_features_do_not_change_when_future_is_removed(df):
    """Features at rows <= T must be identical whether or not data after T exists."""
    full, _ = build_features(df)
    for T in (400, 700, 1000):
        part, _ = build_features(df.iloc[: T + 1])
        pd.testing.assert_frame_equal(full.iloc[: T + 1], part, check_exact=False, rtol=1e-10, atol=1e-12)


def test_features_do_not_change_when_future_is_corrupted(df):
    T = 700
    bad = df.copy()
    bad.iloc[T + 1 :, :] = bad.iloc[T + 1 :, :] * 7.0 + 123.0
    a, _ = build_features(df)
    b, _ = build_features(bad)
    pd.testing.assert_frame_equal(a.iloc[: T + 1], b.iloc[: T + 1], check_exact=False, rtol=1e-10, atol=1e-12)


def test_target_uses_exactly_the_next_h_returns(df):
    r = log_returns(df["Close"])
    y = forward_realized_vol(r, H)
    for t in (50, 300, 900):
        manual = np.sqrt(config.ANN_FACTOR / H * np.sum(r.iloc[t + 1 : t + 1 + H] ** 2))
        assert y.iloc[t] == pytest.approx(manual)
    assert y.iloc[-H:].isna().all() and y.iloc[:-H].notna().all()


def test_target_is_not_a_function_of_the_past(df):
    """Changing returns up to t must not change the target at t."""
    r = log_returns(df["Close"])
    y0 = forward_realized_vol(r, H)
    r2 = r.copy()
    r2.iloc[:500] = 0.5
    assert forward_realized_vol(r2, H).iloc[499] == pytest.approx(y0.iloc[499])


def test_split_is_chronological_and_purged():
    tr, va, te = chronological_split(1000, H, 0.7, 0.15)
    assert tr[-1] < va[0] < va[-1] < te[0] < te[-1]
    assert va[0] - tr[-1] > H - 1 and te[0] - va[-1] > H - 1   # labels of segment k never reach segment k+1


def test_dataset_columns_and_no_nan_after_warmup(df):
    ds, groups, cols = assemble_dataset(df)
    ok = ds.dropna(subset=cols + ["y_log"])
    assert len(ok) > 0.7 * len(ds)
    assert not np.isinf(ok[cols].to_numpy()).any()


def test_metrics_perfect_forecast():
    y = np.array([0.1, 0.2, 0.3])
    m = forecast_metrics(y, y, np.array([0.15, 0.15, 0.15]))
    assert m["MAE"] == 0 and m["QLIKE"] == pytest.approx(0) and m["DirAcc"] == 1.0


def test_diebold_mariano_detects_better_model():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1.0, 0.3, 500), rng.normal(1.3, 0.3, 500)
    stat, p = diebold_mariano(a, b, H)
    assert stat < 0 and p < 0.01


def test_garch_forecast_is_causal(df):
    pytest.importorskip("arch")
    from volforecast.baselines import garch_forecast

    r = log_returns(df["Close"])
    fit_end = df.index[800]
    base = garch_forecast(r, fit_end)
    r2 = r.copy()
    r2.iloc[850:] = r2.iloc[850:] * 5            # corrupt returns after origin 849
    alt = garch_forecast(r2, fit_end)
    origin = df.index[849]
    assert base.loc[origin] == pytest.approx(alt.loc[origin])
