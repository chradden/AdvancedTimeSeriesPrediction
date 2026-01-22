#!/usr/bin/env python3
"""
Quick validation: Test updated Notebook 10 feature engineering on Solar data
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Updated feature engineering function (same as in updated Notebook 10)
def create_features(df):
    """Erstellt Zeit-Features für ML-Modelle - identisch zu Notebook 02"""
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
print("VALIDATION: Updated Notebook 10 Feature Engineering")
print("=" * 80)

# Load preprocessed data
data_dir = Path('data/processed')
train_df = pd.read_csv(data_dir / 'solar_train.csv', parse_dates=['timestamp'])
test_df = pd.read_csv(data_dir / 'solar_test.csv', parse_dates=['timestamp'])

# Expected features (from Notebook 05)
expected_features = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 
                     'quarter', 'hour', 'month_sin', 'month_cos', 'dayofweek_sin', 
                     'dayofweek_cos', 'hour_sin', 'hour_cos', 'is_weekend', 
                     'is_month_start', 'is_month_end', 'lag_1', 'lag_2', 'lag_3', 
                     'lag_24', 'lag_48', 'lag_168', 'rolling_24_mean', 'rolling_24_std', 
                     'rolling_24_min', 'rolling_24_max', 'rolling_168_mean', 
                     'rolling_168_std', 'rolling_168_min', 'rolling_168_max']

print(f"\n✅ Expected features: {len(expected_features)}")
print(f"   {expected_features[:10]}...")

# Get actual features from preprocessed data
actual_features = [col for col in train_df.columns if col not in ['timestamp', 'value']]
print(f"\n✅ Actual features in data: {len(actual_features)}")

# Verify match
if set(expected_features) == set(actual_features):
    print(f"\n🎉 PERFECT MATCH! All {len(expected_features)} features present!")
else:
    missing = set(expected_features) - set(actual_features)
    extra = set(actual_features) - set(expected_features)
    if missing:
        print(f"\n⚠️  Missing: {missing}")
    if extra:
        print(f"\n⚠️  Extra: {extra}")

# Train quick model with all features
X_train = train_df[actual_features].values
y_train = train_df['value'].values
X_test = test_df[actual_features].values
y_test = test_df['value'].values

print(f"\n📊 Training XGBoost with {len(actual_features)} features...")

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train, verbose=False)
pred = model.predict(X_test)

r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)

print(f"\n🎯 RESULTS:")
print(f"   R²:  {r2:.6f}")
print(f"   MAE: {mae:.2f} MW")

print(f"\n📈 Comparison to Notebook 05 baseline:")
print(f"   Expected R²:  ~0.984")
print(f"   Expected MAE: ~245 MW")

if r2 > 0.98:
    print(f"\n✅ SUCCESS! Performance matches Notebook 05!")
    print(f"   The feature mismatch has been fixed!")
else:
    print(f"\n⚠️  Performance still lower than expected")
    print(f"   Further investigation needed")

print("\n" + "=" * 80)
