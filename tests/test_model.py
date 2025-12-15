import os
from data.generate_sample_data import generate
from training.preprocess import preprocess
from training.train_regression import train_regression
from training.train_classification import train_classification
from training.train_timeseries import train_timeseries
import joblib
import pandas as pd


def test_training_pipeline(tmp_path):
    raw = os.path.join(str(tmp_path), 'household_power_consumption.txt')
    processed = os.path.join(str(tmp_path), 'processed.csv')
    generate(raw)
    preprocess(raw_path=raw, out_path=processed)
    r = train_regression(processed_path=processed, model_path=os.path.join(str(tmp_path), 'reg.pkl'))
    c = train_classification(processed_path=processed, model_path=os.path.join(str(tmp_path), 'clf.pkl'))
    try:
        ts = train_timeseries(processed_path=processed, model_path=os.path.join(str(tmp_path), 'ts.pkl'))
    except ImportError:
        import pytest
        pytest.skip("statsmodels not installed - skipping timeseries training test")
    # Load saved models
    rmod = joblib.load(os.path.join(str(tmp_path), 'reg.pkl'))
    clf = joblib.load(os.path.join(str(tmp_path), 'clf.pkl'))
    assert rmod is not None
    assert clf is not None
    # Quick prediction
    df = pd.read_csv(processed)
    row = df.iloc[0]
    X = [[int(row['hour']), float(row['temperature']), float(row['voltage'])]]
    assert rmod.predict(X) is not None
    assert clf.predict(X) is not None
