import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Create models directory
if not os.path.exists('models'):
    os.makedirs('models')

print("Generating realistic grid data...")

# 1. Generate Synthetic Data (5000 samples)
np.random.seed(42)
N = 5000

# Random Inputs
hours = np.random.randint(0, 24, N)
days = np.random.randint(0, 7, N) # 0=Mon, 6=Sun
# Temperature (Warmer in afternoon)
temps = 20 + (10 * np.sin((hours - 6) * np.pi / 12)).clip(0) + np.random.normal(0, 2, N)

# --- Simulate Physics ---
# Base Demand
demand = 2.0 
# Hour Effect: Peak at 19:00 (7 PM)
demand += 2.5 * np.exp(-(hours - 19)**2 / 8) 
# Temp Effect: High AC usage if temp > 25
demand += 0.15 * np.maximum(0, temps - 25)
# Day Effect: Lower demand on weekends (Sat=5, Sun=6)
demand -= 0.5 * (days >= 5)
# Add Random Noise
demand += np.random.normal(0, 0.2, N)

# Voltage Effect: High Demand CAUSES Voltage Drop
# Base 240V minus drop proportional to demand
voltage = 240 - (2.5 * demand) + np.random.normal(0, 1, N)

# Create DataFrame
df = pd.DataFrame({
    'hour': hours,
    'temperature': temps,
    'voltage': voltage,
    'dayofweek': days,
    'demand': demand
})

# 2. Train Regression Model (Predicts Value)
print("Training Regression Model...")
features = ['hour', 'temperature', 'voltage', 'dayofweek']
X = df[features]
y_reg = df['demand']

reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X, y_reg)
joblib.dump(reg_model, 'models/regression.pkl')

# 3. Train Classification Model (Predicts Risk Level)
print("Training Classification Model (3 Levels)...")
# Define Levels: Bottom 33% = Low, Middle 33% = Moderate, Top 33% = High
q33 = df['demand'].quantile(0.33)
q66 = df['demand'].quantile(0.66)

def get_risk_label(d):
    if d < q33: return 0      # Low
    elif d < q66: return 1    # Moderate
    else: return 2            # High

y_clf = df['demand'].apply(get_risk_label)

clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X, y_clf)
joblib.dump(clf_model, 'models/classifier.pkl')

print("✅ Models updated! High Demand now correlates with Low Voltage.")
print(f"  - Low Risk Cutoff: < {q33:.2f} kW")
print(f"  - High Risk Cutoff: > {q66:.2f} kW")