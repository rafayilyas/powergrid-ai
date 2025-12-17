import io
import pandas as pd
import joblib
from pathlib import Path

# -------------------------------------------------
# LOAD MODELS (NO TRAINING, LOAD ONLY WHEN CALLED)
# -------------------------------------------------
def load_models(models_dir: str = "models"):
    models = {}
    base = Path(models_dir)

    try:
        reg_path = base / "regression.pkl"
        if reg_path.exists():
            models["regression"] = joblib.load(reg_path)

        clf_path = base / "classifier.pkl"
        if clf_path.exists():
            models["classifier"] = joblib.load(clf_path)

        ts_path = base / "timeseries.pkl"
        if ts_path.exists():
            models["timeseries"] = joblib.load(ts_path)

    except Exception as e:
        # Fail gracefully instead of crashing deployment
        print(f"[Model Load Error] {e}")

    return models


# -------------------------------------------------
# DETECT CSV SEPARATOR
# -------------------------------------------------
def _detect_sep(contents: bytes) -> str:
    try:
        first_line = contents.decode("utf-8", errors="ignore").splitlines()[0]
        return ";" if ";" in first_line else ","
    except Exception:
        return ","


# -------------------------------------------------
# BATCH DEMAND PREDICTION
# -------------------------------------------------
def predict_demand_batch(contents: bytes, models: dict):
    sep = _detect_sep(contents)

    df = pd.read_csv(
        io.BytesIO(contents),
        sep=sep,
        na_values=["?", "", "NA"]
    )

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    required = ["hour", "temperature", "voltage"]

    # Attempt to derive hour if missing
    if "hour" not in df.columns:
        if "date" in df.columns and "time" in df.columns:
            df["datetime"] = pd.to_datetime(
                df["date"].astype(str) + " " + df["time"].astype(str),
                errors="coerce"
            )
            df["hour"] = df["datetime"].dt.hour

        elif "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df["hour"] = df["datetime"].dt.hour

    # Final validation
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    model = models.get("regression")
    if model is None:
        raise ValueError("Regression model not loaded")

    X = df[required]
    predictions = model.predict(X)

    return predictions.tolist()
