"""Central configuration. Every methodological choice lives here so it can be
reported (and changed) in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
MODEL_DIR = OUT_DIR / "models"

# --- Data -------------------------------------------------------------------
NIFTY_TICKER = "^NSEI"
VIX_TICKER = "^INDIAVIX"
START_DATE = "2008-03-03"     # India VIX history on Yahoo starts around here

# --- Target -----------------------------------------------------------------
# Forecast horizon in trading days. 1-day realised volatility from daily data is
# just |return|, which is extremely noisy; 5 days (one trading week) is the
# shortest horizon at which a daily-data target is a reasonable volatility proxy.
HORIZON = 5
ANN_FACTOR = 252
VOL_FLOOR = 1e-4              # avoids log(0)

# --- Validation -------------------------------------------------------------
# Chronological split with a purge gap of HORIZON rows between segments, because
# the target at row t uses returns t+1..t+HORIZON.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15               # remainder is the untouched test set
SEED = 42

# --- Risk analytics ---------------------------------------------------------
# Risk classes are percentiles of the *training-period* distribution of realised
# volatility (never the test period), so the thresholds are not tuned on future data.
RISK_PERCENTILES = (25.0, 75.0)   # LOW < p25 <= MODERATE <= p75 < HIGH
TREND_TOL = 0.10                  # +/-10% vs current 5-day realised vol
REGIME_PERCENTILES = (33.3, 66.7) # calm / normal / stressed, from trailing 21d vol
