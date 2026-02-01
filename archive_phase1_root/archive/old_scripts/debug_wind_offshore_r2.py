#!/usr/bin/env python3
"""
Deep dive: Why does Wind Offshore get R² = 0 in Multi-Series?
Test the actual multi-series pipeline on Wind Offshore
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

def create_features_updated(df):
    """Updated feature engineering from fixed Notebook 10"""
    df = df.copy()
    
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['dayofyear'] = df['timestamp'].dt.dayofyear
    df['weekofyear'] = df['timestamp'].dt.isocalendar().week.astype('int64')
    df['quarter'] = df['timestamp'].dt.quarter
    df['hour'] = df['timestamp'].dt.hour
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_start'] = df['timestamp'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['timestamp'].dt.is_month_end.astype(int)
    
    df['lag_1'] = df['value'].shift(1)
    df['lag_2'] = df['value'].shift(2)
    df['lag_3'] = df['value'].shift(3)
    df['lag_24'] = df['value'].shift(24)
    df['lag_48'] = df['value'].shift(48)
    df['lag_168'] = df['value'].shift(168)
    
    df['rolling_24_mean'] = df['value'].shift(1).rolling(window=24).mean()
    df['rolling_24_std'] = df['value'].shift(1).rolling(window=24).std()
    df['rolling_24_min'] = df['value'].shift(1).rolling(window=24).min()
    df['rolling_24_max'] = df['value'].shift(1).rolling(window=24).max()
    
    df['rolling_168_mean'] = df['value'].shift(1).rolling(window=168).mean()
    df['rolling_168_std'] = df['value'].shift(1).rolling(window=168).std()
    df['rolling_168_min'] = df['value'].shift(1).rolling(window=168).min()
    df['rolling_168_max'] = df['value'].shift(1).rolling(window=168).max()
    
    df = df.dropna()
    return df

print("=" * 80)
print("WIND OFFSHORE R²=0 DEBUGGING")
print("=" * 80)

# Load raw data
raw_file = Path('data/raw/wind_offshore_2022-01-01_2024-12-31_hour.csv')
df = pd.read_csv(raw_file, parse_dates=['timestamp'])

print(f"\n1. Raw Data:")
print(f"   Shape: {df.shape}")
print(f"   Missing: {df['value'].isna().sum()}")

# Interpolate missing values
df['value'] = df['value'].interpolate(method='linear')

print(f"\n2. After interpolation:")
print(f"   Missing: {df['value'].isna().sum()}")
print(f"   Mean: {df['value'].mean():.2f}")
print(f"   Std:  {df['value'].std():.2f}")

# Apply feature engineering
print(f"\n3. Feature Engineering...")
df_features = create_features_updated(df)

print(f"   Shape after features: {df_features.shape}")
print(f"   Rows lost to NaN: {len(df) - len(df_features)}")

# Define features
features = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 
            'quarter', 'hour', 'month_sin', 'month_cos', 'dayofweek_sin', 
            'dayofweek_cos', 'hour_sin', 'hour_cos', 'is_weekend', 
            'is_month_start', 'is_month_end', 'lag_1', 'lag_2', 'lag_3', 
            'lag_24', 'lag_48', 'lag_168', 'rolling_24_mean', 'rolling_24_std', 
            'rolling_24_min', 'rolling_24_max', 'rolling_168_mean', 
            'rolling_168_std', 'rolling_168_min', 'rolling_168_max']

X = df_features[features]
y = df_features['value']

# Train/test split (last 30 days)
TEST_DAYS = 30
split_idx = len(df_features) - (TEST_DAYS * 24)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\n4. Train/Test Split:")
print(f"   Train: {len(X_train)} samples")
print(f"   Test:  {len(X_test)} samples ({TEST_DAYS} days)")

# Check for issues
print(f"\n5. Data Quality Checks:")
print(f"   Train target - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
print(f"   Test target  - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}")
print(f"   Train variance: {y_train.var():.2f}")
print(f"   Test variance:  {y_test.var():.2f}")

if y_test.std() < 0.1:
    print(f"\n   ⚠️  TEST DATA IS CONSTANT!")
    print(f"   This causes R² = 0")

# Train model
print(f"\n6. Training XGBoost...")
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train, verbose=False)
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n7. RESULTS:")
print(f"   MAE: {mae:.2f} MW")
print(f"   R²:  {r2:.6f}")

# Analyze predictions
print(f"\n8. Prediction Analysis:")
print(f"   Actual - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}")
print(f"   Pred   - Mean: {y_pred.mean():.2f}, Std: {y_pred.std():.2f}")
print(f"   Difference in mean: {abs(y_test.mean() - y_pred.mean()):.2f}")

# Compare to naive baseline (predict mean)
y_mean = np.full_like(y_pred, y_train.mean())
mae_baseline = mean_absolute_error(y_test, y_mean)
r2_baseline = r2_score(y_test, y_mean)

print(f"\n9. Baseline Comparison (Predict Mean):")
print(f"   Baseline MAE: {mae_baseline:.2f} MW")
print(f"   Baseline R²:  {r2_baseline:.6f}")
print(f"   Model improvement: {mae_baseline - mae:.2f} MW ({(mae_baseline - mae)/mae_baseline*100:.2f}%)")

if r2 < 0.01 and r2 > -0.01:
    print(f"\n⚠️  R² ≈ 0: Model is no better than predicting the mean")
    print(f"   Possible causes:")
    print(f"   1. Test period has different distribution than train")
    print(f"   2. Features don't capture wind patterns well")
    print(f"   3. Data quality issue in specific time period")
    
    # Check test period
    test_dates = df_features['timestamp'].iloc[split_idx:]
    print(f"\n   Test period: {test_dates.min()} to {test_dates.max()}")
    
    # Check if test data is anomalous
    print(f"\n   Checking test period characteristics:")
    test_zero_pct = (y_test == 0).sum() / len(y_test) * 100
    print(f"   Zero values in test: {test_zero_pct:.2f}%")
    
    train_zero_pct = (y_train == 0).sum() / len(y_train) * 100
    print(f"   Zero values in train: {train_zero_pct:.2f}%")
    
    if abs(test_zero_pct - train_zero_pct) > 20:
        print(f"\n   ❌ DISTRIBUTION SHIFT DETECTED!")
        print(f"   Test data has significantly different characteristics")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

if r2 < 0.01:
    print("\n💡 Wind Offshore needs special treatment:")
    print("   1. Use different test period (not just last 30 days)")
    print("   2. Consider seasonality more carefully")
    print("   3. Add weather-related features if available")
    print("   4. Use ensemble of models trained on different periods")
    print("   5. Accept that offshore wind is inherently difficult to predict")

print("\n" + "=" * 80)
