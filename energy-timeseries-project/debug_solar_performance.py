#!/usr/bin/env python3
"""
Debug Script: Compare feature engineering between Notebook 05 and 10
Identifies why Solar R² drops from 0.98 to 0.83
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

print("=" * 80)
print("SOLAR PERFORMANCE DEBUGGING: Notebook 05 vs 10")
print("=" * 80)

# Load preprocessed data (used in Notebook 05)
data_dir = Path('data/processed')
train_df = pd.read_csv(data_dir / 'solar_train.csv', parse_dates=['timestamp'])
test_df = pd.read_csv(data_dir / 'solar_test.csv', parse_dates=['timestamp'])

print(f"\n1. Data Shapes:")
print(f"   Train: {train_df.shape}")
print(f"   Test:  {test_df.shape}")

# Check features in preprocessed data
feature_cols_nb05 = [col for col in train_df.columns if col not in ['timestamp', 'value']]
print(f"\n2. Features in Notebook 05 (from preprocessing):")
print(f"   Count: {len(feature_cols_nb05)}")
print(f"   Features: {feature_cols_nb05}")

# Notebook 10 features (from the code)
nb10_features = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear',
                 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                 'lag_24', 'lag_48', 'lag_168', 
                 'rolling_mean_24', 'rolling_std_24']
print(f"\n3. Features in Notebook 10 (from multi-series):")
print(f"   Count: {len(nb10_features)}")
print(f"   Features: {nb10_features}")

# Find differences
missing_in_nb10 = set(feature_cols_nb05) - set(nb10_features)
extra_in_nb10 = set(nb10_features) - set(feature_cols_nb05)

print(f"\n4. Feature Differences:")
print(f"   Missing in Notebook 10: {missing_in_nb10}")
print(f"   Extra in Notebook 10: {extra_in_nb10}")

# Reproduce Notebook 05 approach
print(f"\n5. Reproducing Notebook 05 Performance:")
X_train_nb05 = train_df[feature_cols_nb05].values
y_train_nb05 = train_df['value'].values
X_test_nb05 = test_df[feature_cols_nb05].values
y_test_nb05 = test_df['value'].values

model_nb05 = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    random_state=42,
    n_jobs=-1
)

print("   Training with Notebook 05 features...")
model_nb05.fit(X_train_nb05, y_train_nb05, verbose=False)
pred_nb05 = model_nb05.predict(X_test_nb05)
r2_nb05 = r2_score(y_test_nb05, pred_nb05)
mae_nb05 = mean_absolute_error(y_test_nb05, pred_nb05)

print(f"   ✅ Notebook 05 Performance:")
print(f"      R²:  {r2_nb05:.6f}")
print(f"      MAE: {mae_nb05:.2f} MW")

# Simulate Notebook 10 approach (limited features)
print(f"\n6. Simulating Notebook 10 Approach:")

# Check which nb10 features exist in our data
available_nb10_features = [f for f in nb10_features if f in feature_cols_nb05]
print(f"   Available features from nb10 list: {len(available_nb10_features)}/{len(nb10_features)}")
print(f"   Available: {available_nb10_features}")

if len(available_nb10_features) > 0:
    X_train_nb10 = train_df[available_nb10_features].values
    X_test_nb10 = test_df[available_nb10_features].values
    
    model_nb10 = XGBRegressor(
        n_estimators=1000,  # nb10 uses more estimators
        learning_rate=0.05,
        max_depth=6,  # nb10 uses max_depth=6
        random_state=42,
        n_jobs=-1
    )
    
    print("   Training with Notebook 10 features...")
    model_nb10.fit(X_train_nb10, y_train_nb05, verbose=False)
    pred_nb10 = model_nb10.predict(X_test_nb10)
    r2_nb10 = r2_score(y_test_nb05, pred_nb10)
    mae_nb10 = mean_absolute_error(y_test_nb05, pred_nb10)
    
    print(f"   ⚠️  Notebook 10 Performance:")
    print(f"      R²:  {r2_nb10:.6f}")
    print(f"      MAE: {mae_nb10:.2f} MW")
    
    print(f"\n7. Performance Drop Analysis:")
    r2_drop = (r2_nb05 - r2_nb10) / r2_nb05 * 100
    mae_increase = (mae_nb10 - mae_nb05) / mae_nb05 * 100
    print(f"   R² dropped:    {r2_drop:.2f}%")
    print(f"   MAE increased: {mae_increase:.2f}%")

# Check data splits
print(f"\n8. Data Split Analysis:")
print(f"   Train period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
print(f"   Test period:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
print(f"   Test size:    {len(test_df)} hours = {len(test_df)/24:.1f} days")

# Notebook 10 uses 30 days for testing
nb10_test_size = 30 * 24
print(f"   Notebook 10 test size: {nb10_test_size} hours = 30 days")
if len(test_df) != nb10_test_size:
    print(f"   ⚠️  MISMATCH: Notebook 05 uses {len(test_df)/24:.1f} days, Notebook 10 uses 30 days")

print("\n" + "=" * 80)
print("ROOT CAUSE ANALYSIS")
print("=" * 80)

print("\n🔍 Identified Issues:")

if missing_in_nb10:
    print(f"\n1. ❌ Feature Mismatch:")
    print(f"   Notebook 10 is missing {len(missing_in_nb10)} features that Notebook 05 uses")
    print(f"   Missing features: {missing_in_nb10}")
    print(f"   → This is likely the MAIN cause of performance drop")

if len(test_df) != nb10_test_size:
    print(f"\n2. ⚠️  Different Test Splits:")
    print(f"   Notebook 05: {len(test_df)/24:.1f} days")
    print(f"   Notebook 10: 30 days")
    print(f"   → Different test data makes comparison unfair")

print(f"\n3. ℹ️  Hyperparameter Differences:")
print(f"   Notebook 05: n_estimators=500, max_depth=7")
print(f"   Notebook 10: n_estimators=1000, max_depth=6")
print(f"   → Minor impact expected")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

print("\n💡 Solutions:")
print("\n1. Update Notebook 10 to use ALL features from preprocessing:")
print("   - Use the same feature list as Notebook 05")
print("   - Should restore R² to ~0.98")

print("\n2. Standardize test splits:")
print("   - Use consistent test period across all notebooks")
print("   - Consider using same dates for fair comparison")

print("\n3. Create a unified feature engineering function:")
print("   - Centralize in src/data/preprocessing.py")
print("   - Ensure consistency across all notebooks")

print("\n" + "=" * 80)
