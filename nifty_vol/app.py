"""Streamlit dashboard. Run the pipeline first:  python run_pipeline.py
then:  streamlit run app.py"""
import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from volforecast import config, risk

st.set_page_config(page_title="NIFTY 50 Volatility & Risk Analytics", layout="wide")
OUT = config.OUT_DIR
NAMES = {"naive_rv5": "Naive RV (5d)", "naive_rv21": "Naive RV (21d)", "ewma": "EWMA (lambda=0.94)",
         "garch": "GARCH(1,1)-t", "har": "HAR-type (OLS)", "rf": "Random Forest", "xgb": "XGBoost",
         "xgb_novix": "XGBoost (no VIX)"}
RISK_COLOR = {"LOW": "#2e7d32", "MODERATE": "#ef6c00", "HIGH": "#c62828"}


@st.cache_data
def load():
    if not (OUT / "run_meta.json").exists():
        return None
    meta = json.loads((OUT / "run_meta.json").read_text())
    return dict(
        meta=meta,
        latest=json.loads((OUT / "latest_forecast.json").read_text()),
        ds=pd.read_csv(OUT / "features_all.csv", index_col=0, parse_dates=True),
        pred=pd.read_csv(OUT / "predictions.csv", index_col=0, parse_dates=True),
        metrics=pd.read_csv(OUT / "metrics_test.csv", index_col=0),
        regime=pd.read_csv(OUT / "metrics_regime.csv"),
        dm=pd.read_csv(OUT / "dm_tests.csv"),
    )


@st.cache_resource
def load_shap():
    p = config.MODEL_DIR / "shap.joblib"
    return joblib.load(p) if p.exists() else {}


D = load()
if D is None:
    st.title("NIFTY 50 Volatility Forecasting & Risk Analytics")
    st.error("No results found. Run `python run_pipeline.py` first, then reload this page.")
    st.stop()

meta, latest, ds, pred = D["meta"], D["latest"], D["ds"], D["pred"]
thr = meta["thresholds"]
h = meta["horizon"]

st.title("NIFTY 50 Volatility Forecasting & Predictive Risk Analytics")
if meta["source"] == "synthetic":
    st.warning("SYNTHETIC DATA: this run used simulated prices to test the pipeline. Nothing here describes the real NIFTY 50.")
st.caption("Research and decision-support tool. Volatility forecasts are not trading signals and do not guarantee profit.")

tabs = st.tabs(["Latest forecast", "Market & volatility", "Forecast vs actual", "Model comparison",
                "Explainability", "Method & data"])

# ---------------------------------------------------------------- latest ----
with tabs[0]:
    keys = list(latest["models"])
    default = keys.index(latest["primary_model"])
    choice = st.selectbox("Model", keys, index=default, format_func=lambda k: latest["models"][k]["name"])
    m = latest["models"][choice]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted volatility (annualised)", f"{m['predicted_vol']*100:.1f}%")
    c2.markdown(f"**Risk classification**<br><span style='font-size:2rem;color:{RISK_COLOR[m['risk']]}'>{m['risk']}</span>",
                unsafe_allow_html=True)
    c3.metric("Recent volatility trend", m["trend"], f"vs trailing 5d vol {latest['current_rv_5']*100:.1f}%", delta_color="off")
    c4.metric(f"Expected {h}-day move (±1σ)", f"±{m['expected_move_1sigma']*100:.2f}%")
    st.table(pd.DataFrame({
        "Market": ["NIFTY 50"], "Forecast horizon": [f"Next {h} trading days"], "Model used": [m["name"]],
        "Data as of": [latest["asof_date"]], "Prediction timestamp (UTC)": [latest["generated_at_utc"]]}).T.rename(columns={0: ""}))
    st.caption(f"Risk thresholds are percentiles ({thr['percentiles'][0]:.0f}th / {thr['percentiles'][1]:.0f}th) of realised "
               f"volatility in the training period: LOW < {thr['low']*100:.1f}% ≤ MODERATE ≤ {thr['high']*100:.1f}% < HIGH. "
               "The ±1σ move is a distributional statement implied by the forecast, not a bound on losses.")
    if latest["asof_date"] < str(pd.Timestamp.today().date() - pd.Timedelta(days=7)) and meta["source"] != "synthetic":
        st.info("Data may be stale. Re-run `python run_pipeline.py --refresh` to pull the latest prices.")

# ---------------------------------------------------------------- market ----
with tabs[1]:
    start = st.date_input("From", ds.index[-1260].date() if len(ds) > 1260 else ds.index[0].date())
    d = ds[ds.index >= pd.Timestamp(start)]
    fig = go.Figure(go.Scatter(x=d.index, y=d["close"], name="Close"))
    fig.update_layout(title="NIFTY 50 close", height=320, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")
    fig = go.Figure()
    for col, lab in (("rv_5", "Realised vol 5d"), ("rv_21", "Realised vol 21d")):
        fig.add_scatter(x=d.index, y=d[col] * 100, name=lab)
    if "vix" in d:
        fig.add_scatter(x=d.index, y=d["vix"], name="India VIX", line=dict(dash="dot"))
    fig.add_hline(y=thr["low"] * 100, line_dash="dash", annotation_text="LOW/MOD")
    fig.add_hline(y=thr["high"] * 100, line_dash="dash", annotation_text="MOD/HIGH")
    fig.update_layout(title="Historical volatility (% annualised)", height=340, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")
    dd = risk.drawdown_series(d["close"])
    fig = go.Figure(go.Scatter(x=dd.index, y=dd * 100, fill="tozeroy", line=dict(color="#c62828")))
    fig.update_layout(title="Historical drawdown from running peak (%) - descriptive only, NOT a forecast",
                      height=260, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("A volatility forecast does not determine the size of a future drawdown; this chart only shows what happened.")

# --------------------------------------------------------------- forecast ----
with tabs[2]:
    avail = [k for k in NAMES if k in pred]
    sel = st.multiselect("Models", avail, default=[k for k in ("har", "garch", "xgb") if k in avail],
                         format_func=lambda k: NAMES[k])
    st.caption("Test period only: the models never saw these rows during training or tuning. Each point is "
               f"the forecast made at the close of that day for the following {h} trading days, plotted at the forecast date.")
    fig = go.Figure()
    fig.add_scatter(x=pred.index, y=pred["y_vol"] * 100, name="Realised (next 5d)", line=dict(color="black", width=2))
    for k in sel:
        fig.add_scatter(x=pred.index, y=pred[k] * 100, name=NAMES[k], opacity=0.85)
    fig.update_layout(height=420, yaxis_title="% annualised", margin=dict(t=20, b=10))
    st.plotly_chart(fig, width="stretch")
    if sel:
        k = st.selectbox("Scatter for", sel, format_func=lambda k: NAMES[k])
        mx = float(max(pred["y_vol"].max(), pred[k].max()) * 100)
        fig = go.Figure(go.Scatter(x=pred[k] * 100, y=pred["y_vol"] * 100, mode="markers", marker=dict(size=4, opacity=0.5)))
        fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dash"), name="perfect")
        fig.update_layout(xaxis_title="Predicted %", yaxis_title="Realised %", height=380, margin=dict(t=20, b=10))
        st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------- comparison ----
with tabs[3]:
    st.subheader("Test-set accuracy (lower is better except R2_log / DirAcc)")
    st.dataframe(D["metrics"].style.format("{:.4f}").highlight_min(subset=["MAE", "RMSE", "MAPE_%", "QLIKE"], color="#c8e6c9"),
                 width="stretch")
    st.caption("QLIKE is the preferred loss for volatility forecasts. DirAcc = share of times the model correctly called "
               "whether volatility would rise or fall relative to the trailing 5-day level (undefined for the 5d naive model).")
    st.subheader("Does it hold up across market conditions?")
    metric = st.radio("Metric", ["QLIKE", "MAE", "RMSE"], horizontal=True)
    piv = D["regime"].pivot(index="model", columns="regime", values=metric)[["calm", "normal", "stressed"]]
    st.dataframe(piv.style.format("{:.4f}"), width="stretch")
    cnt = D["regime"].groupby("regime")["n"].first()
    st.caption("Regimes use trailing 21-day volatility terciles from the training period. Test rows: "
               + ", ".join(f"{r} = {int(cnt[r])}" for r in cnt.index))
    st.subheader("Diebold–Mariano tests vs benchmark")
    st.dataframe(D["dm"].style.format({"DM_stat": "{:.2f}", "p_value": "{:.3f}"}), width="stretch", hide_index=True)
    st.caption("Negative DM statistic means the model has lower loss than the benchmark. Overlapping 5-day targets are handled "
               "with a HAC variance. With ~600 test days, many differences will not be significant; that is a finding, not a bug.")

# ------------------------------------------------------------- explain ----
with tabs[4]:
    S = load_shap()
    if not S:
        st.info("No SHAP output found.")
    else:
        k = st.selectbox("Model", list(S), format_func=lambda k: NAMES[k])
        imp = S[k]["importance"].head(15)[::-1]
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h"))
        fig.update_layout(title="Mean |SHAP| (impact on predicted log-volatility)", height=460, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")
        try:
            import matplotlib.pyplot as plt
            import shap

            plt.figure()
            shap.summary_plot(S[k]["values"], S[k]["X"], show=False, max_display=15)
            st.pyplot(plt.gcf(), clear_figure=True)
        except Exception as exc:
            st.caption(f"Beeswarm plot unavailable ({exc}).")
        st.caption("Explanations describe how the fitted model uses features, not causal effects on the market.")

# ---------------------------------------------------------------- method ----
with tabs[5]:
    rep = meta["data_report"]
    st.markdown(f"""
**Target.** Annualised realised volatility over the next **{h} trading days**, from daily log returns; models predict log(RV).
**Split.** Chronological with a {h}-row purge gap between segments: train {meta['splits']['train'][0]} → {meta['splits']['train'][1]},
validation {meta['splits']['val'][0]} → {meta['splits']['val'][1]}, test {meta['splits']['test'][0]} → {meta['splits']['test'][1]}.
Hyper-parameters are chosen on validation; final models are refit on train+validation; the test block is touched once.
**Data source.** `{meta['source']}`. **Rows after cleaning.** {rep['rows_clean']}. **Features.** {len(meta['feature_cols'])}.
""")
    st.json(rep, expanded=False)
    st.write("Feature groups:", meta["feature_groups"])
    st.write("Selected hyper-parameters:", {k: v["params"] for k, v in meta["models"].items()})
