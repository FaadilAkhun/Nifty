# Machine Learning-Based Dynamic Volatility Modeling and Predictive Risk Analytics (NIFTY 50)

Forecasts the **annualised realised volatility of the NIFTY 50 over the next 5 trading days**, compares
statistical baselines with Random Forest and XGBoost, converts forecasts into LOW / MODERATE / HIGH risk
classes, explains the models with SHAP and serves everything in a Streamlit dashboard.

## Quick start
```bash
pip install -r requirements.txt
python run_pipeline.py                 # downloads NIFTY 50 + India VIX (yfinance), trains, evaluates, saves outputs/
streamlit run app.py                   # dashboard
pytest -q                              # leakage + correctness tests
```
Other data options:
```bash
python run_pipeline.py --source csv --nifty-csv nifty50.csv --vix-csv indiavix.csv   # files you downloaded from NSE
python run_pipeline.py --source synthetic --fast                                     # pipeline smoke test ONLY
python run_pipeline.py --refresh                                                     # re-download Yahoo data
```
CSV files need a `Date` column and `Open, High, Low, Close` (and optionally `Volume`); the VIX file needs `Date, Close`.

## Data source and licensing (read this)
* **Default:** `yfinance` (open-source Python package, Apache-2.0) pulling `^NSEI` and `^INDIAVIX` from Yahoo Finance.
  The *code* is open source; the *data* is Yahoo's and is intended for personal / research use. It is fine for a
  course project. Do **not** commit the downloaded CSVs to a public repo or redistribute them, and cite the source.
* **Alternative:** download the official NSE / niftyindices.com historical files by hand and use `--source csv`.
  Also free, also under NSE's terms of use (no redistribution).
* There is no truly open-licensed (CC0 / ODbL) daily NIFTY 50 + India VIX history that I know of, so "open source"
  here means free and openly accessible, not openly licensed.
* Known quirks: Yahoo's `^NSEI` volume is zero for long stretches (the code detects this and drops the volume
  feature automatically); data is unofficial and occasionally has gaps (reported by the data-validation step); the
  endpoint is unofficial and can break without notice.

## Methodology
| Item | Decision | Why |
|---|---|---|
| Target | `RV(t,t+5) = sqrt(252/5 * sum r²)`, log returns; models predict `log(RV)` | 1-day RV from daily data is just `abs(return)`; 5 days is the shortest sensible horizon. Logs stabilise variance. Change `HORIZON` in `config.py` |
| Split | Chronological 70 / 15 / 15 with a **5-row purge gap** between segments | Labels overlap the next 5 days; the gap prevents any training label from seeing validation/test returns |
| Tuning | Random search, fit on train, scored on the later validation block | Time-ordered; test block untouched until the end |
| Final fit | Refit on train+validation; smearing factor from validation residuals so `exp(pred)` estimates mean vol | Uses all pre-test data |
| Baselines | Naive trailing RV (5d, 21d), EWMA (λ=0.94), GARCH(1,1)-t (fit once, params frozen), HAR-type OLS on log RV 5/21/63 | HAR and GARCH are the standard strong baselines; ML must beat these, not only a naive one |
| ML | Random Forest, XGBoost on 24 features (returns, multi-window RV, Parkinson, ATR, RSI, MACD, Bollinger, trend/drawdown state, India VIX, volume if usable) | As in the proposal |
| Ablation | XGBoost without VIX features | Tests whether implied vol adds information |
| Metrics | MAE, RMSE, MAPE, **QLIKE**, R² (log), directional accuracy of vol change vs trailing 5d vol, calm/normal/stressed breakdown, Diebold–Mariano tests (HAC, overlap-aware) | Same test rows for all models |
| Risk classes | Percentiles (25th / 75th) of **training-period** realised vol | Defensible, not tuned on test data |
| Drawdown | Shown as a historical descriptive chart only; ±1σ expected move shown separately | A vol forecast does not determine drawdown size |
| XAI | SHAP (TreeExplainer) on the test block, in log-vol units | Explains the model, not causal market effects |

## Leakage controls (tested in `tests/test_leakage.py`)
* Features at row *t* are unchanged if all data after *t* is removed or corrupted.
* The target at *t* uses exactly returns *t+1…t+5*, and does not depend on earlier returns.
* Split segments are chronological and separated by the purge gap.
* The GARCH forecast at origin *t* is unchanged if later returns are corrupted.
* Risk thresholds and regime cut-offs come from the training period only; the primary dashboard model is selected on validation, never on test.

## Layout
```
run_pipeline.py        end-to-end training/evaluation, writes outputs/
app.py                 Streamlit dashboard
volforecast/           config, data, targets, features, splits, baselines, models, evaluate, risk, explain
tests/                 leakage / correctness tests
outputs/               created by the pipeline (predictions, metrics, models, SHAP, latest forecast)
```

## Limitations
* Fixed-origin evaluation (one train/val/test split). A rolling re-fit walk-forward would be a natural extension.
* GARCH parameters are frozen after the first fit; HAR/ML models are not re-estimated through the test period.
* Daily data only: realised volatility is estimated from squared daily returns, which is noisy. Intraday data would give a better target.
* Single index, single test regime sequence; results may not transfer to other periods or assets.
* LSTM/GRU (optional in the proposal) is not included; add it only after the tree models and baselines are understood.
* Research and decision support only; nothing here is trading advice.
