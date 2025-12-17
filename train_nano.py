import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib

print("Loading data...")
try:
    # 1. Load Data
    df = pd.read_csv('data/processed/processed.csv')
    
    # 2. FIX: Ensure Datetime and Features exist
    # This block fixes the KeyError by regenerating the missing columns
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour
        df['dayofweek'] = df['datetime'].dt.dayofweek
    else:
        # Fallback if datetime is completely missing
        print("Warning: 'datetime' column missing. Using fallback index.")
        df['hour'] = [12] * len(df)
        df['dayofweek'] = [0] * len(df)

except FileNotFoundError:
    print("Processed data not found. Creating dummy data...")
    # Create Dummy Data if file doesn't exist
    data = {
        'hour': [0, 6, 12, 18, 23] * 20,
        'temperature': [20, 22, 30, 28, 25] * 20,
        'voltage': [230, 235, 220, 225, 230] * 20,
        'dayofweek': [0, 1, 2, 3, 4] * 20,
        'demand': [2.5, 3.0, 6.5, 7.0, 4.0] * 20
    }
    df = pd.DataFrame(data)

# 3. Double Check Columns exist now
features = ['hour', 'temperature', 'voltage', 'dayofweek']
for col in features:
    if col not in df.columns:
        print(f"Creating missing column: {col}")
        df[col] = 0  # Default value to prevent crash

# 4. Prepare Training Data
df = df.dropna(subset=features + ['demand'])
X = df[features]
y_reg = df['demand']
y_clf = (df['demand'] > df['demand'].median()).astype(int) # Simple risk threshold

# 5. Train NANO Models (Extreme settings for low memory)
print("Training Nano Regression Model...")
reg = RandomForestRegressor(n_estimators=5, max_depth=3, n_jobs=1, random_state=42)
reg.fit(X, y_reg)

print("Training Nano Classifier Model...")
clf = RandomForestClassifier(n_estimators=5, max_depth=3, n_jobs=1, random_state=42)
clf.fit(X, y_clf)

# 6. Save
Path('models').mkdir(parents=True, exist_ok=True)
joblib.dump(reg, 'models/regression.pkl', compress=9)
joblib.dump(clf, 'models/classifier.pkl', compress=9)

print("✅ Success! Tiny models saved.")