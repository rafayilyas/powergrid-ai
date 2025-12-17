import pandas as pd
from pathlib import Path
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    SARIMAX = None

# Handle import whether running as module or script
try:
    from training.preprocess import preprocess
except ImportError:
    from preprocess import preprocess


def train_timeseries(processed_path: str = 'data/processed/processed.csv', model_path: str = 'models/timeseries.pkl'):
    processed = Path(processed_path)
    
    # 1. Ensure Data Exists
    if not processed.exists():
        print("Processed file not found. Running preprocessing first...")
        preprocess()

    if SARIMAX is None:
        raise ImportError('statsmodels is required to train timeseries models. Please install it.')

    print("Loading data for Time Series...")
    # 2. Load and Set Index
    df = pd.read_csv(processed_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Set datetime as index and ensure it is sorted
    df = df.set_index('datetime').sort_index()
    
    # 3. Prepare Time Series Object
    # 'asfreq' ensures we have a strict hourly frequency, filling gaps if necessary
    ts = df['demand'].asfreq('h')
    
    # Fill missing values (forward fill is standard for time series)
    # Note: .fillna(method='ffill') is deprecated, using .ffill() instead
    ts = ts.ffill()

    # OPTIONAL: Limit training data size
    # SARIMAX is very slow on large datasets (30k+ points). 
    # We take the last 10,000 hours (over a year) which is plenty for a pattern.
    if len(ts) > 10000:
        print(f"Dataset is large ({len(ts)} hours). Training on the last 10,000 hours to save time...")
        ts = ts.iloc[-10000:]

    print("Fitting SARIMAX model (this might take a minute)...")
    
    # 4. Define and Fit Model
    # order=(p,d,q): (1,0,1) is a standard baseline
    # seasonal_order=(P,D,Q,s): (0,1,1,24) looks for daily (24h) seasonality
    model = SARIMAX(ts, 
                    order=(1, 0, 1), 
                    seasonal_order=(0, 1, 1, 24), 
                    enforce_stationarity=False, 
                    enforce_invertibility=False)
    
    res = model.fit(disp=False)
    
    # 5. Save Model
    Path('models').mkdir(parents=True, exist_ok=True)
    joblib.dump(res, model_path)
    
    print("Time Series model saved.")
    return {'aic': float(res.aic), 'model_path': model_path}


if __name__ == '__main__':
    print('Training timeseries model...')
    print(train_timeseries())