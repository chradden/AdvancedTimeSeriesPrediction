#!/usr/bin/env python3
"""
Quick end-to-end test of the updated Notebook 10 logic on all datasets
Tests both the fixed feature engineering AND smart test splits
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

def create_features_updated(df):
    """Updated feature engineering - all 31 features"""
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
print("QUICK TEST: Updated Notebook 10 Logic")
print("=" * 80)

# Test configuration from updated notebook
TEST_PERIODS = {
    'solar': {'start': '2024-07-01', 'end': '2024-07-30'},
    'wind_offshore': {'start': '2022-10-01', 'end': '2022-10-30'},
}

features = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 
            'quarter', 'hour', 'month_sin', 'month_cos', 'dayofweek_sin', 
            'dayofweek_cos', 'hour_sin', 'hour_cos', 'is_weekend', 
            'is_month_start', 'is_month_end', 'lag_1', 'lag_2', 'lag_3', 
            'lag_24', 'lag_48', 'lag_168', 'rolling_24_mean', 'rolling_24_std', 
            'rolling_24_min', 'rolling_24_max', 'rolling_168_mean', 
            'rolling_168_std', 'rolling_168_min', 'rolling_168_max']

results = []

for source in ['solar', 'wind_offshore']:
    print(f"\n{'='*50}")
    print(f"Testing: {source.upper()}")
    print(f"{'='*50}")
    
    # Load data
    raw_file = Path(f'data/raw/{source}_2022-01-01_2024-12-31_hour.csv')
    if not raw_file.exists():
        print(f"  ❌ File not found: {raw_file}")
        continue
    
    df = pd.read_csv(raw_file, parse_dates=['timestamp'])
    df['value'] = df['value'].interpolate(method='linear')
    
    # Feature engineering
    df_features = create_features_updated(df)
    print(f"  Features created: {df_features.shape}")
    
    # Smart split
    test_period = TEST_PERIODS[source]
    test_mask = (df_features['timestamp'] >= test_period['start']) & \
                (df_features['timestamp'] <= test_period['end'])
    train_mask = ~test_mask
    
    X_train = df_features[features][train_mask]
    X_test = df_features[features][test_mask]
    y_train = df_features['value'][train_mask]
    y_test = df_features['value'][test_mask]
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)} ({len(X_test)/24:.1f} days)")
    print(f"  Test period: {test_period['start']} to {test_period['end']}")
    print(f"  Test data - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}")
    
    # Quick quality check
    if y_test.std() < 1.0:
        print(f"  ⚠️  WARNING: Test data has low variance!")
        continue
    
    # Train quick model
    print(f"  Training XGBoost...")
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train, verbose=False)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n  RESULTS:")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²:  {r2:.6f}")
    
    results.append({
        'Dataset': source,
        'MAE': mae,
        'R²': r2,
        'Test_Period': f"{test_period['start']} to {test_period['end']}"
    })
    
    # Evaluate
    if r2 > 0.5:
        print(f"  ✅ EXCELLENT: R² > 0.5")
    elif r2 > 0.3:
        print(f"  ✅ GOOD: R² > 0.3")
    elif r2 > 0.0:
        print(f"  ⚠️  MODERATE: R² > 0, but could be better")
    else:
        print(f"  ❌ FAILED: R² ≤ 0")

print(f"\n\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}\n")

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print(f"\n\n🎯 KEY FINDINGS:")

solar_r2 = results_df[results_df['Dataset'] == 'solar']['R²'].values[0] if 'solar' in results_df['Dataset'].values else 0
wind_r2 = results_df[results_df['Dataset'] == 'wind_offshore']['R²'].values[0] if 'wind_offshore' in results_df['Dataset'].values else 0

if solar_r2 > 0.95:
    print(f"\n✅ Solar: R² = {solar_r2:.4f} - Matches Notebook 05 baseline!")
else:
    print(f"\n⚠️  Solar: R² = {solar_r2:.4f} - Below expected 0.98")

if wind_r2 > 0.3:
    print(f"✅ Wind Offshore: R² = {wind_r2:.4f} - FIXED! (was 0.00)")
elif wind_r2 > 0:
    print(f"⚠️  Wind Offshore: R² = {wind_r2:.4f} - Improved, but still challenging")
else:
    print(f"❌ Wind Offshore: R² = {wind_r2:.4f} - Still problematic")

print(f"\n{'='*80}")
