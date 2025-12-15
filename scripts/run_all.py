import os
from data.generate_sample_data import generate
from training.preprocess import preprocess
from training.train_regression import train_regression
from training.train_classification import train_classification
from training.train_timeseries import train_timeseries
import joblib
from pathlib import Path

def main():
    # Generate sample data
    raw_path = 'data/raw/household_power_consumption.txt'
    if not os.path.exists(raw_path):
        generate(raw_path)
    # Preprocess
    processed_path = 'data/processed/processed.csv'
    preprocess(raw_path=raw_path, out_path=processed_path)
    # Train models
    r = train_regression(processed_path=processed_path)
    print('Regression:', r)
    c = train_classification(processed_path=processed_path)
    print('Classification:', c)
    try:
        ts = train_timeseries(processed_path=processed_path)
        print('Timeseries:', ts)
    except ImportError as e:
        print('Timeseries skipped:', e)
    # Load models directly
    model_dir = Path('models')
    reg = None
    clf = None
    if (model_dir / 'regression.pkl').exists():
        reg = joblib.load(model_dir / 'regression.pkl')
    if (model_dir / 'classifier.pkl').exists():
        clf = joblib.load(model_dir / 'classifier.pkl')
    print('Loaded models:', ('regression' if reg else None, 'classifier' if clf else None))
    if reg is not None:
        sample = [[18, 32.0, 230.0, 1]]
        print('Regression predict:', reg.predict(sample)[0])
    if clf is not None:
        sample = [[18, 32.0, 230.0, 1]]
        print('Classifier predict:', clf.predict(sample)[0])

if __name__ == '__main__':
    main()
