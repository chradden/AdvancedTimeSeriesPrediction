#!/usr/bin/env python3
"""
XGBoost Hyperparameter Tuning Script
Faster execution than Notebook for long-running optimization
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import time
import json
warnings.filterwarnings('ignore')

# ML Libraries
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 80)
print("🔧 XGBoost Hyperparameter Tuning - Production Script")
print("=" * 80)

# ==============================================================================
# 1. Load Data
# ==============================================================================
print("\n📂 Loading data...")
DATA_TYPE = 'solar'
data_dir = Path('data/processed')

train_df = pd.read_csv(data_dir / f'{DATA_TYPE}_train.csv', parse_dates=['timestamp'])
val_df = pd.read_csv(data_dir / f'{DATA_TYPE}_val.csv', parse_dates=['timestamp'])
test_df = pd.read_csv(data_dir / f'{DATA_TYPE}_test.csv', parse_dates=['timestamp'])

print(f"✅ Train: {len(train_df):,} samples")
print(f"✅ Val:   {len(val_df):,} samples")
print(f"✅ Test:  {len(test_df):,} samples")
print(f"✅ Value range: [{train_df['value'].min():.0f}, {train_df['value'].max():.0f}] MW")

# ==============================================================================
# 2. Feature Engineering
# ==============================================================================
print("\n🔨 Creating features...")

def create_features(df):
    """Create comprehensive time-based features"""
    df = df.copy()
    
    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['dayofyear'] = df['timestamp'].dt.dayofyear
    df['week'] = df['timestamp'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['timestamp'].dt.quarter
    
    # Cyclical encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Boolean flags
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_start'] = df['timestamp'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['timestamp'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['timestamp'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['timestamp'].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df['timestamp'].dt.is_year_start.astype(int)
    df['is_year_end'] = df['timestamp'].dt.is_year_end.astype(int)
    
    # Lag features
    for lag in [1, 2, 3, 24, 48, 168]:
        df[f'lag_{lag}'] = df['value'].shift(lag)
    
    # Rolling statistics
    for window in [24, 168]:
        df[f'rolling_{window}_mean'] = df['value'].shift(1).rolling(window=window).mean()
        df[f'rolling_{window}_std'] = df['value'].shift(1).rolling(window=window).std()
        df[f'rolling_{window}_min'] = df['value'].shift(1).rolling(window=window).min()
        df[f'rolling_{window}_max'] = df['value'].shift(1).rolling(window=window).max()
    
    # Drop NaN from lag/rolling features
    df = df.dropna()
    
    return df

train_featured = create_features(train_df)
val_featured = create_features(val_df)
test_featured = create_features(test_df)

# Prepare X, y
feature_cols = [col for col in train_featured.columns if col not in ['timestamp', 'value']]
X_train = train_featured[feature_cols]
y_train = train_featured['value']
X_val = val_featured[feature_cols]
y_val = val_featured['value']
X_test = test_featured[feature_cols]
y_test = test_featured['value']

print(f"✅ Features created: {len(feature_cols)} features")
print(f"✅ Train samples after feature creation: {len(X_train):,}")

# ==============================================================================
# 3. Baseline Model
# ==============================================================================
print("\n📊 Training baseline XGBoost model...")
start_time = time.time()

baseline_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)
baseline_model.fit(X_train, y_train)

# Evaluate baseline
y_pred_baseline = baseline_model.predict(X_test)
baseline_mae = mean_absolute_error(y_test, y_pred_baseline)
baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
baseline_r2 = r2_score(y_test, y_pred_baseline)

baseline_time = time.time() - start_time

print(f"\n✅ Baseline Results (Test Set):")
print(f"   MAE:  {baseline_mae:.2f} MW")
print(f"   RMSE: {baseline_rmse:.2f} MW")
print(f"   R²:   {baseline_r2:.4f}")
print(f"   Training time: {baseline_time:.1f}s")

# ==============================================================================
# 4. Hyperparameter Tuning with Random Search
# ==============================================================================
print("\n" + "=" * 80)
print("🔍 Starting Random Search Hyperparameter Tuning...")
print("=" * 80)

# Parameter distributions
param_distributions = {
    'n_estimators': [100, 200, 300, 400, 500, 750, 1000],
    'max_depth': [3, 4, 5, 6, 7, 8, 10],
    'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7, 10],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
}

# Time Series Cross-Validation (5 splits)
tscv = TimeSeriesSplit(n_splits=5)

# Base estimator
base_estimator = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

# Random Search
n_iter = 50  # 50 random combinations
print(f"📌 Configuration:")
print(f"   - Parameter combinations: {n_iter}")
print(f"   - CV splits: 5 (TimeSeriesSplit)")
print(f"   - Total fits: {n_iter * 5} = {n_iter * 5}")
print(f"   - Estimated time: 30-60 minutes")
print(f"\n⏳ Starting optimization (this will take a while)...\n")

random_search = RandomizedSearchCV(
    estimator=base_estimator,
    param_distributions=param_distributions,
    n_iter=n_iter,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# Fit
tuning_start = time.time()
random_search.fit(X_train, y_train)
tuning_time = time.time() - tuning_start

# ==============================================================================
# 5. Results
# ==============================================================================
print("\n" + "=" * 80)
print("✅ TUNING COMPLETED!")
print("=" * 80)

# Best parameters
best_params = random_search.best_params_
print("\n🏆 Best Parameters Found:")
for param, value in sorted(best_params.items()):
    print(f"   {param:20s}: {value}")

# Best CV score
best_cv_score = -random_search.best_score_
print(f"\n📊 Best CV MAE: {best_cv_score:.2f} MW")

# Test set evaluation
best_model = random_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)

tuned_mae = mean_absolute_error(y_test, y_pred_tuned)
tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
tuned_r2 = r2_score(y_test, y_pred_tuned)

print(f"\n✅ Tuned Model Results (Test Set):")
print(f"   MAE:  {tuned_mae:.2f} MW")
print(f"   RMSE: {tuned_rmse:.2f} MW")
print(f"   R²:   {tuned_r2:.4f}")

# ==============================================================================
# 6. Comparison
# ==============================================================================
print("\n" + "=" * 80)
print("📈 BASELINE vs. TUNED COMPARISON")
print("=" * 80)

improvement_mae = ((baseline_mae - tuned_mae) / baseline_mae) * 100
improvement_rmse = ((baseline_rmse - tuned_rmse) / baseline_rmse) * 100
improvement_r2 = ((tuned_r2 - baseline_r2) / baseline_r2) * 100

print(f"\n{'Metric':<10} {'Baseline':<12} {'Tuned':<12} {'Improvement':<15}")
print("-" * 55)
print(f"{'MAE':<10} {baseline_mae:>10.2f} MW {tuned_mae:>10.2f} MW {improvement_mae:>13.2f}%")
print(f"{'RMSE':<10} {baseline_rmse:>10.2f} MW {tuned_rmse:>10.2f} MW {improvement_rmse:>13.2f}%")
print(f"{'R²':<10} {baseline_r2:>12.4f} {tuned_r2:>12.4f} {improvement_r2:>13.2f}%")

print(f"\n⏱️  Tuning Time: {tuning_time/60:.1f} minutes")

# ==============================================================================
# 7. Save Results
# ==============================================================================
print("\n💾 Saving results...")

results_dir = Path('results/metrics')
results_dir.mkdir(parents=True, exist_ok=True)

# Save best parameters
with open(results_dir / 'xgboost_best_params.json', 'w') as f:
    json.dump(best_params, f, indent=2)

# Save comparison
comparison_df = pd.DataFrame({
    'Model': ['Baseline', 'Tuned'],
    'MAE': [baseline_mae, tuned_mae],
    'RMSE': [baseline_rmse, tuned_rmse],
    'R2': [baseline_r2, tuned_r2],
    'Training_Time_s': [baseline_time, tuning_time]
})
comparison_df.to_csv(results_dir / 'xgboost_tuning_comparison.csv', index=False)

# Save CV results
cv_results_df = pd.DataFrame(random_search.cv_results_)
cv_results_df.to_csv(results_dir / 'xgboost_cv_results.csv', index=False)

print(f"✅ Results saved to {results_dir}/")
print(f"   - xgboost_best_params.json")
print(f"   - xgboost_tuning_comparison.csv")
print(f"   - xgboost_cv_results.csv")

# ==============================================================================
# 8. Final Summary
# ==============================================================================
print("\n" + "=" * 80)
print("🎉 XGBoost Hyperparameter Tuning Complete!")
print("=" * 80)

if improvement_mae > 0:
    print(f"\n✅ Tuning was SUCCESSFUL: {improvement_mae:.2f}% MAE improvement")
else:
    print(f"\n⚠️  Baseline was better: {-improvement_mae:.2f}% MAE decrease")
    print("   This is expected - baseline params were already good!")

print(f"\n📊 Final Test MAE: {tuned_mae:.2f} MW")
print(f"📊 Final Test R²:  {tuned_r2:.4f}")
print(f"\n✅ All results saved to results/metrics/")
print("\n" + "=" * 80)
