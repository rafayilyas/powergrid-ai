import io
import pandas as pd
import joblib
from pathlib import Path


def load_models(models_dir: str = "models"):
    models = {}
    base = Path(models_dir)
    if (base / "regression.pkl").exists():
        models['regression'] = joblib.load(base / "regression.pkl")
    if (base / "classifier.pkl").exists():
        models['classifier'] = joblib.load(base / "classifier.pkl")
    if (base / "timeseries.pkl").exists():
        models['timeseries'] = joblib.load(base / "timeseries.pkl")
    return models


def _detect_sep(contents: bytes):
    text = contents.decode('utf-8', errors='ignore')
    # check for semicolon usage
    if ';' in text.splitlines()[0]:
        return ';'
    return ','


def predict_demand_batch(contents: bytes, models: dict):
    sep = _detect_sep(contents)
    df = pd.read_csv(io.BytesIO(contents), sep=sep, na_values=['?', ''])
    required = ['hour', 'temperature', 'voltage']
    # If CSV uses Date/Time columns, try to compute hour column like in preprocessing
    if not all(c in df.columns for c in required):
        if 'Date' in df.columns and 'Time' in df.columns:
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
            df['hour'] = df['datetime'].dt.hour
        if 'datetime' in df.columns and 'hour' not in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df['hour'] = df['datetime'].dt.hour
    if not all(c in df.columns for c in required):
        raise ValueError('CSV missing required columns: hour, temperature, voltage')
    clf = models.get('regression')
    if clf is None:
        raise ValueError('Regression model not available')
    X = df[required]
    preds = clf.predict(X)
    return preds.tolist()
