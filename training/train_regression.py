import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
from training.preprocess import preprocess


def train_regression(processed_path: str = 'data/processed/processed.csv', model_path: str = 'models/regression.pkl'):
    processed = Path(processed_path)
    if not processed.exists():
        preprocess()
    df = pd.read_csv(processed_path)
    features = ['hour', 'temperature', 'voltage', 'dayofweek']
    df = df.dropna(subset=features + ['demand'])
    X = df[features]
    y = df['demand']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    Path('models').mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return {'rmse': float(rmse), 'model_path': model_path}


if __name__ == '__main__':
    print('Training regression model...')
    print(train_regression())
