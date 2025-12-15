import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from training.preprocess import preprocess


def train_classification(processed_path: str = 'data/processed/processed.csv', model_path: str = 'models/classifier.pkl'):
    processed = Path(processed_path)
    if not processed.exists():
        preprocess()
    df = pd.read_csv(processed_path)
    features = ['hour', 'temperature', 'voltage', 'dayofweek']
    df = df.dropna(subset=features + ['demand'])
    # Label as high risk if demand above 75th percentile
    thresh = df['demand'].quantile(0.75)
    df['high_risk'] = (df['demand'] > thresh).astype(int)
    X = df[features]
    y = df['high_risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    Path('models').mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    return {'accuracy': float(acc), 'model_path': model_path, 'threshold': float(thresh)}


if __name__ == '__main__':
    print('Training classifier model...')
    print(train_classification())
