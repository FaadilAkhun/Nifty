"""

    python run_pipeline.py                         # NIFTY 50 + India VIX via yfinance
    python run_pipeline.py --source csv --nifty-csv nifty.csv --vix-csv vix.csv
    python run_pipeline.py --source synthetic      # smoke test only, NOT market data
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from volforecast import baselines, config, evaluate, explain, risk
from volforecast.data import load_market_data
from volforecast.features import assemble_dataset
from volforecast.models import (HAR_COLS, RF_SPACE, XGB_SPACE, LogVolRegressor, make_har, make_rf, make_xgb, tune)
from volforecast.splits import chronological_split

warnings.filterwarnings("ignore", category=FutureWarning)

NAMES = {
    "naive_rv5": "Naive RV (5d)",
    "naive_rv21": "Naive RV (21d)",
    "ewma": "EWMA (lambda=0.94)",
    "garch": "GARCH(1,1)-t",
    "har": "HAR-type (OLS)",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "xgb_novix": "XGBoost (no VIX)",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["yahoo", "csv", "synthetic"], default="yahoo")
    ap.add_argument("--nifty-csv")
    ap.add_argument("--vix-csv")
    ap.add_argument("--refresh", action="store_true", help="re-download Yahoo data instead of using the cache")
    ap.add_argument("--fast", action="store_true", help="fewer hyper-parameter trials (quick run)")
    return ap.parse_args()


def main():
    args = parse_args()
    h = config.HORIZON
    config.OUT_DIR.mkdir(exist_ok=True)
    config.MODEL_DIR.mkdir(exist_ok=True, parents=True)

    # 1. data ---------------------------------------------------------------
    df, report = load_market_data(args.source, args.nifty_csv, args.vix_csv, args.refresh)
    print("Data report:", json.dumps(report, indent=2))
    ds, groups, feat_cols = assemble_dataset(df, h)
    valid = ds.dropna(subset=feat_cols + ["y_log"])
    print(f"Usable rows: {len(valid)} ({valid.index[0].date()} -> {valid.index[-1].date()}), {len(feat_cols)} features")

    # 2. leakage-safe chronological split ------------------------------------
    tr, va, te = chronological_split(len(valid), h, config.TRAIN_FRAC, config.VAL_FRAC)
    D_tr, D_va, D_te = valid.iloc[tr], valid.iloc[va], valid.iloc[te]
    D_tv = pd.concat([D_tr, D_va])
    spans = {k: [str(d.index[0].date()), str(d.index[-1].date()), len(d)] for k, d in
             {"train": D_tr, "val": D_va, "test": D_te}.items()}
    print("Splits (start, end, n):", spans)

    # 3. thresholds fixed on TRAIN only ---------------------------------------
    thr = risk.fit_thresholds(D_tr["y_vol"])
    regime_cuts = tuple(np.percentile(D_tr["rv_21"], config.REGIME_PERCENTILES))

    # 4. baselines ------------------------------------------------------------
    pred = pd.DataFrame(index=D_te.index)
    pred["y_vol"], pred["rv_5"], pred["rv_21"], pred["close"] = D_te["y_vol"], D_te["rv_5"], D_te["rv_21"], D_te["close"]
    pred["regime"] = evaluate.assign_regime(D_te["rv_21"], regime_cuts).astype(str)
    pred["naive_rv5"] = baselines.naive_rv(D_te, 5)
    pred["naive_rv21"] = baselines.naive_rv(D_te, 21)
    pred["ewma"] = baselines.ewma_vol(ds["ret"]).reindex(D_te.index)
    try:
        pred["garch"] = baselines.garch_forecast(ds["ret"], fit_end=D_va.index[-1], horizon=h).reindex(D_te.index)
        if pred["garch"].isna().any():
            print("[warn] GARCH forecasts missing for some test dates; dropping column.")
            pred = pred.drop(columns="garch")
    except Exception as exc:
        print(f"[warn] GARCH baseline skipped: {exc}")

    # 5. fit models: train -> tune/smear on val -> refit on train+val -> test ------
    meta_models: dict = {}

    def fit_final(factory, params, cols, name):
        m = LogVolRegressor(factory(**params), cols, name).fit(D_tr, D_tr["y_log"])
        resid = D_va["y_log"].to_numpy() - m.predict_log(D_va)
        final = LogVolRegressor(factory(**params), cols, name).fit(D_tv, D_tv["y_log"])
        final.smear = float(np.mean(np.exp(resid)))
        return final, float(np.sqrt(np.mean(resid**2)))

    models: dict[str, LogVolRegressor] = {}
    models["har"], v = fit_final(make_har, {}, HAR_COLS, "har")
    meta_models["har"] = {"params": {}, "val_rmse_log": v, "smear": models["har"].smear}

    for key, factory, space, n_iter in (("rf", make_rf, RF_SPACE, 4 if args.fast else 12),
                                        ("xgb", make_xgb, XGB_SPACE, 6 if args.fast else 24)):
        print(f"Tuning {NAMES[key]} ({n_iter} random trials, validated on the later val block)...")
        best, trials = tune(factory, space, D_tr, D_tr["y_log"], D_va, D_va["y_log"], feat_cols, n_iter)
        trials.to_csv(config.OUT_DIR / f"tuning_trials_{key}.csv", index=False)
        models[key], v = fit_final(factory, best, feat_cols, key)
        meta_models[key] = {"params": best, "val_rmse_log": v, "smear": models[key].smear}

    if "vix" in groups:  # ablation: does India VIX add information beyond price-based features?
        cols_nv = [c for c in feat_cols if c not in groups["vix"]]
        models["xgb_novix"], v = fit_final(make_xgb, meta_models["xgb"]["params"], cols_nv, "xgb_novix")
        meta_models["xgb_novix"] = {"params": meta_models["xgb"]["params"], "val_rmse_log": v,
                                    "smear": models["xgb_novix"].smear}

    for k, m in models.items():
        pred[k] = m.predict_vol(D_te)

    # 6. evaluation on the untouched test block -----------------------------------
    names = {k: v for k, v in NAMES.items() if k in pred}
    mt = evaluate.metrics_table(pred, names)
    rt = evaluate.regime_table(pred, names)
    benchmark = "har" if "har" in pred else "naive_rv21"
    dmt = evaluate.dm_table(pred, names, benchmark, h)
    print("\nTEST metrics (sorted by QLIKE, lower is better):\n", mt.round(4).to_string())
    print(f"\nDiebold-Mariano vs {names[benchmark]}:\n", dmt.round(4).to_string(index=False))

    # 7. explainability -----------------------------------------------------------
    shap_out = {}
    for k in ("xgb", "rf"):
        try:
            shap_out[k] = explain.compute_shap(models[k].estimator, D_te[models[k].cols])
        except Exception as exc:
            print(f"[warn] SHAP for {k} failed: {exc}")
    # 8. latest forecast (rows with features but no realised future yet) -----------
    last = ds.dropna(subset=feat_cols).iloc[[-1]]
    primary = min(("rf", "xgb"), key=lambda k: meta_models[k]["val_rmse_log"])  # chosen on VAL, never on test
    latest = {"asof_date": str(last.index[0].date()), "horizon_days": h, "primary_model": primary,
              "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "current_rv_5": float(last["rv_5"].iloc[0]), "models": {}}
    for k in ("har", "rf", "xgb"):
        v = float(models[k].predict_vol(last)[0])
        latest["models"][k] = {"name": NAMES[k], "predicted_vol": v, "risk": risk.classify(v, thr),
                               "trend": risk.trend(v, latest["current_rv_5"]),
                               "expected_move_1sigma": risk.expected_move(v, h)}

    # 9. persist ------------------------------------------------------------------
    pred.to_csv(config.OUT_DIR / "predictions.csv")
    mt.to_csv(config.OUT_DIR / "metrics_test.csv")
    rt.to_csv(config.OUT_DIR / "metrics_regime.csv", index=False)
    dmt.to_csv(config.OUT_DIR / "dm_tests.csv", index=False)
    ds.to_csv(config.OUT_DIR / "features_all.csv")
    joblib.dump({k: m for k, m in models.items()}, config.MODEL_DIR / "models.joblib")
    joblib.dump(shap_out, config.MODEL_DIR / "shap.joblib")
    (config.OUT_DIR / "latest_forecast.json").write_text(json.dumps(latest, indent=2))
    (config.OUT_DIR / "run_meta.json").write_text(json.dumps({
        "source": args.source, "data_report": report, "splits": spans, "thresholds": thr,
        "regime_cuts": list(regime_cuts), "horizon": h, "feature_cols": feat_cols, "feature_groups": groups,
        "models": meta_models, "primary_model": primary,
        "generated_at_utc": latest["generated_at_utc"]}, indent=2, default=str))
    print(f"\nSaved artefacts to {config.OUT_DIR}. Next: streamlit run app.py")
    if args.source == "synthetic":
        print("NOTE: synthetic data. These numbers say nothing about NIFTY 50.")


if __name__ == "__main__":
    sys.exit(main())
