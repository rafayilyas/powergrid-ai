import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

# Handle import whether running as module or script
try:
    from training.preprocess import preprocess
except ImportError:
    from preprocess import preprocess

def train_regression(processed_path: str = 'data/processed/processed.csv', model_path: str = 'models/regression.pkl'):
    processed = Path(processed_path)
    
    # 1. Ensure Data Exists
    if not processed.exists():
        print("Processed file not found. Running preprocessing first...")
        preprocess()

    # 2. Load Data
    df = pd.read_csv(processed_path)
    
    # 3. Feature Engineering (The Fix)
    # We must regenerate 'hour' and 'dayofweek' from the datetime column
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek

    features = ['hour', 'temperature', 'voltage', 'dayofweek']
    
    # Clean data
    df = df.dropna(subset=features + ['demand'])

    # 4. Prepare Training Data
    X = df[features]
    y = df['demand']

    # 5. Train Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Fitting Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 6. Evaluate and Save
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    Path('models').mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    
    return {'rmse': float(rmse), 'model_path': model_path}

if __name__ == '__main__':
    print('Training regression model...')
    print(train_regression())