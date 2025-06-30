#!/usr/bin/env python3
"""
Test XGBoost Infinite Value Fix
==============================

Minimal test to verify the infinite value handling works.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

# Create test data with infinite values
print("Creating test data with infinite values...")
np.random.seed(42)

# Test features with some infinite values
X = pd.DataFrame({
    'feat1': [1, 2, np.inf, 4, 5],
    'feat2': [2, 3, 4, -np.inf, 6],
    'feat3': [1, np.inf, 3, 4, 5]
})

# Test target with some infinite values  
y = pd.Series([1.0, 2.0, np.inf, 4.0, 5.0])

print("Original data:")
print("X:", X.values)
print("y:", y.values)

# Test the cleaning approach
print("\nTesting data cleaning...")

# Clean features
X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
print("X_clean:", X_clean.values)

# Clean target
y_clean = y.replace([np.inf, -np.inf], np.nan)
print("y_clean:", y_clean.values)

# Remove problematic rows
valid_mask = np.isfinite(X_clean.values).all(axis=1) & np.isfinite(y_clean.values)
print("valid_mask:", valid_mask)

X_final = X_clean[valid_mask]
y_final = y_clean[valid_mask]

print("X_final:", X_final.values)
print("y_final:", y_final.values)

# Test XGBoost
print("\nTesting XGBoost...")
try:
    model = xgb.XGBRegressor(n_estimators=10, verbosity=0)
    model.fit(X_final, y_final)
    pred = model.predict(X_final)
    print("✅ XGBoost SUCCESS!")
    print("Predictions:", pred)
except Exception as e:
    print("❌ XGBoost FAILED:", str(e))
    print("Falling back to RandomForest...")
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_final, y_final)
    pred = model.predict(X_final)
    print("✅ RandomForest SUCCESS!")
    print("Predictions:", pred)

print("\n🎉 Test completed!")