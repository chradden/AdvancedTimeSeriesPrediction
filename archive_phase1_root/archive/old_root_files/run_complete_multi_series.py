#!/usr/bin/env python3
"""
Execute complete Multi-Series Analysis with all fixes
Simulates the full Notebook 10 pipeline
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def create_features(df):
    """Complete feature engineering - 31 features"""
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

def evaluate_model(y_true, y_pred, model_name, dataset_name):
    """Calculate metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    return {
        'Dataset': dataset_name,
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }

print("=" * 80)
print("COMPLETE MULTI-SERIES ANALYSIS")
print("=" * 80)

# Configuration
DATA_SOURCES = ['solar', 'wind_onshore', 'wind_offshore', 'consumption', 'price_day_ahead']

TEST_PERIODS = {
    'solar': {'start': '2024-07-01', 'end': '2024-07-30'},
    'wind_onshore': {'start': '2023-11-01', 'end': '2023-11-30'},
    'wind_offshore': {'start': '2022-10-01', 'end': '2022-10-30'},
    'consumption': {'start': '2024-01-01', 'end': '2024-01-30'},
    'price_day_ahead': {'start': '2024-06-01', 'end': '2024-06-30'}
}

features = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 
            'quarter', 'hour', 'month_sin', 'month_cos', 'dayofweek_sin', 
            'dayofweek_cos', 'hour_sin', 'hour_cos', 'is_weekend', 
            'is_month_start', 'is_month_end', 'lag_1', 'lag_2', 'lag_3', 
            'lag_24', 'lag_48', 'lag_168', 'rolling_24_mean', 'rolling_24_std', 
            'rolling_24_min', 'rolling_24_max', 'rolling_168_mean', 
            'rolling_168_std', 'rolling_168_min', 'rolling_168_max']

all_results = []

for source in DATA_SOURCES:
    print(f"\n{'='*80}")
    print(f"📊 Analyzing: {source.upper()}")
    print(f"{'='*80}")
    
    # Load data
    raw_file = Path(f'data/raw/{source}_2022-01-01_2024-12-31_hour.csv')
    if not raw_file.exists():
        print(f"⚠️ File not found: {raw_file}")
        continue
    
    df = pd.read_csv(raw_file, parse_dates=['timestamp'])
    print(f"Loaded: {len(df)} hours")
    
    # Handle missing values
    missing = df['value'].isna().sum()
    if missing > 0:
        df['value'] = df['value'].interpolate(method='linear')
        print(f"Interpolated {missing} missing values")
    
    # Feature engineering
    df_features = create_features(df)
    print(f"Features created: {df_features.shape}")
    
    # Smart train/test split
    test_period = TEST_PERIODS[source]
    test_mask = (df_features['timestamp'] >= test_period['start']) & \
                (df_features['timestamp'] <= test_period['end'])
    train_mask = ~test_mask
    
    X_train = df_features[features][train_mask]
    X_test = df_features[features][test_mask]
    y_train = df_features['value'][train_mask]
    y_test = df_features['value'][test_mask]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)} ({len(X_test)/24:.1f} days)")
    print(f"Test period: {test_period['start']} to {test_period['end']}")
    
    # Quality check
    if y_test.std() < 1.0:
        print(f"⚠️ WARNING: Low test variance (Std={y_test.std():.2f})")
    
    # Train XGBoost
    print("Training XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=42
    )
    
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    y_pred_xgb = xgb_model.predict(X_test)
    res_xgb = evaluate_model(y_test, y_pred_xgb, 'XGBoost', source)
    all_results.append(res_xgb)
    print(f"  XGBoost R²: {res_xgb['R2']:.4f}, MAE: {res_xgb['MAE']:.2f}")
    
    # Train LightGBM
    print("Training LightGBM...")
    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
    
    lgb_model.fit(X_train, y_train)
    y_pred_lgb = lgb_model.predict(X_test)
    res_lgb = evaluate_model(y_test, y_pred_lgb, 'LightGBM', source)
    all_results.append(res_lgb)
    print(f"  LightGBM R²: {res_lgb['R2']:.4f}, MAE: {res_lgb['MAE']:.2f}")

# Results
print(f"\n\n{'='*80}")
print("FINAL RESULTS")
print(f"{'='*80}\n")

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('R2', ascending=False)

print(results_df.to_string(index=False))

# Save results
output_file = Path('results/metrics/multi_series_comparison_UPDATED.csv')
results_df.to_csv(output_file, index=False)
print(f"\n✅ Results saved to: {output_file}")

# Summary statistics
print(f"\n\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

# Best model per dataset
best_per_dataset = results_df.loc[results_df.groupby('Dataset')['R2'].idxmax()]

print(f"\n🏆 BEST MODEL PER DATASET:\n")
for _, row in best_per_dataset.iterrows():
    emoji = "🟢" if row['R2'] > 0.9 else "🟡" if row['R2'] > 0.7 else "🟠" if row['R2'] > 0.5 else "🔴"
    print(f"{emoji} {row['Dataset']:15s} - {row['Model']:10s} R²={row['R2']:.4f}, MAE={row['MAE']:8.2f}")

# Overall stats
avg_r2 = results_df['R2'].mean()
median_r2 = results_df['R2'].median()

print(f"\n📊 OVERALL PERFORMANCE:")
print(f"   Average R²:  {avg_r2:.4f}")
print(f"   Median R²:   {median_r2:.4f}")
print(f"   Best R²:     {results_df['R2'].max():.4f} ({results_df.loc[results_df['R2'].idxmax(), 'Dataset']})")
print(f"   Worst R²:    {results_df['R2'].min():.4f} ({results_df.loc[results_df['R2'].idxmin(), 'Dataset']})")

# Model comparison
print(f"\n🥊 MODEL BATTLE:")
xgb_wins = (results_df[results_df['Model'] == 'XGBoost'].groupby('Dataset')['R2'].max() > 
            results_df[results_df['Model'] == 'LightGBM'].groupby('Dataset')['R2'].max()).sum()
lgb_wins = len(DATA_SOURCES) - xgb_wins

print(f"   XGBoost wins:  {xgb_wins}/{len(DATA_SOURCES)} datasets")
print(f"   LightGBM wins: {lgb_wins}/{len(DATA_SOURCES)} datasets")

print(f"\n{'='*80}")
print("✅ ANALYSIS COMPLETE!")
print(f"{'='*80}")
