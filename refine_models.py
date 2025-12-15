import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Setup
if not os.path.exists('models'):
    os.makedirs('models')

print("Refining model sensitivity...")

# 1. Generate Synthetic Data (Robust Distribution)
np.random.seed(42)
N = 10000

# Random Inputs
hours = np.random.randint(0, 24, N)
days = np.random.randint(0, 7, N)
# Broader temp range
temps = 20 + (10 * np.sin((hours - 6) * np.pi / 12)).clip(0) + np.random.normal(0, 3, N)

# --- Physics Simulation ---
# Demand is driven by Hour and Temp
demand = 1.5
demand += 2.0 * np.exp(-(hours - 19)**2 / 10) # Peak at 7PM
demand += 0.2 * np.maximum(0, temps - 22) # AC kicks in at 22C
demand -= 0.3 * (days >= 5) # Weekends lower
demand += np.random.normal(0, 0.3, N) # Noise

# Voltage drops as Demand rises (Grid Physics)
# We tune this so 230V is "average" load
# 242V = No Load, 220V = Extreme Load
voltage = 242 - (3.0 * demand) + np.random.normal(0, 0.5, N)

df = pd.DataFrame({
    'hour': hours, 'temperature': temps, 'voltage': voltage, 
    'dayofweek': days, 'demand': demand
})

# 2. Define Risk Zones based on VOLTAGE (User Friendly Bands)
# High Risk: Voltage sagging below 228V
# Moderate: Voltage holding steady at 228V - 236V (Includes 230V!)
# Low Risk: Voltage high at > 236V
def get_risk_label(row):
    v = row['voltage']
    if v < 228: return 2      # High Risk
    elif v <= 236: return 1   # Moderate Risk
    else: return 0            # Low Risk

df['risk'] = df.apply(get_risk_label, axis=1)

# 3. Train Models
X = df[['hour', 'temperature', 'voltage', 'dayofweek']]

# Regression (Predicts Value)
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X, df['demand'])
joblib.dump(reg, 'models/regression.pkl')

# Classification (Predicts Label)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, df['risk'])
joblib.dump(clf, 'models/classifier.pkl')

print("✅ Models Refined!")
print("  - Low Risk Input:    > 236 V")
print("  - Moderate Risk Input: 228 V - 236 V (Try 230 V)")
print("  - High Risk Input:   < 228 V")