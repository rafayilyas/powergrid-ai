import pandas as pd
from pathlib import Path
import joblib
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception as e:
    SARIMAX = None
from training.preprocess import preprocess


def train_timeseries(processed_path: str = 'data/processed/processed.csv', model_path: str = 'models/timeseries.pkl'):
    processed = Path(processed_path)
    if not processed.exists():
        preprocess()
    df = pd.read_csv(processed_path, parse_dates=['datetime'])
    ts = df.set_index('datetime')['demand'].asfreq('h').fillna(method='ffill')
    # Simple SARIMAX fit
    if SARIMAX is None:
        raise ImportError('statsmodels is required to train timeseries models. Install via requirements.txt')
    model = SARIMAX(ts, order=(1,0,1), seasonal_order=(0,1,1,24), enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    Path('models').mkdir(parents=True, exist_ok=True)
    joblib.dump(res, model_path)
    return {'aic': float(res.aic), 'model_path': model_path}


if __name__ == '__main__':
    print('Training timeseries model...')
    print(train_timeseries())
