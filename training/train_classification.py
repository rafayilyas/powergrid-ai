import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Try importing based on folder structure (training.preprocess) or local file (preprocess)
try:
    from training.preprocess import preprocess
except ImportError:
    from preprocess import preprocess

def train_classification(processed_path: str = 'data/processed/processed.csv', model_path: str = 'models/classifier.pkl'):
    processed = Path(processed_path)
    
    # 1. Run Preprocessing if file is missing
    if not processed.exists():
        print("Processed file not found. Running preprocessing first...")
        preprocess()

    # 2. Load Data
    df = pd.read_csv(processed_path)
    
    # 3. Feature Engineering (Critical Step)
    # The processed.csv only has 'datetime', 'demand', 'temperature', 'voltage'.
    # We must re-create 'hour' and 'dayofweek' from the 'datetime' column.
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek

    features = ['hour', 'temperature', 'voltage', 'dayofweek']
    
    # Clean up any potential NaNs
    df = df.dropna(subset=features + ['demand'])

    # 4. Create Target Label
    # Label as high risk if demand is above the 75th percentile
    thresh = df['demand'].quantile(0.75)
    df['high_risk'] = (df['demand'] > thresh).astype(int)

    X = df[features]
    y = df['high_risk']

    # 5. Train Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 6. Evaluate and Save
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    Path('models').mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)

    return {
        'accuracy': float(acc), 
        'model_path': model_path, 
        'threshold': float(thresh),
        'training_samples': len(X_train)
    }

if __name__ == '__main__':
    print('Training classifier model...')
    print(train_classification())