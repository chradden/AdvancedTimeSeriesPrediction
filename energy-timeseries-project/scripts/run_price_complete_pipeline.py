#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PRICE FORECASTING - COMPLETE PIPELINE
Runs: Exploration, Preprocessing, Baseline, ML Models
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import time
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

def print_section(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def evaluate_model(y_true, y_pred, model_name):
    """Calculate metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    
    return {
        'Model': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }

# =============================================================================
# SETUP
# =============================================================================
print_section("PRICE FORECASTING - COMPLETE AUTOMATED PIPELINE")

data_path = Path('data/raw/price_day_ahead_2022-01-01_2024-12-31_hour.csv')
proc_dir = Path('data/processed')
results_dir = Path('results')
figures_dir = results_dir / 'figures'
metrics_dir = results_dir / 'metrics'

proc_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)
metrics_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PHASE 1: DATA EXPLORATION
# =============================================================================
print_section("PHASE 1: DATA EXPLORATION")

print("\n[1/3] Loading data...")
df = pd.read_csv(data_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df.sort_index(inplace=True)
df.rename(columns={'value': 'price'}, inplace=True)
print(f"✅ Loaded: {df.shape}")

print("\n[2/3] Computing statistics...")
stats = {
    'count': len(df),
    'mean': df['price'].mean(),
    'std': df['price'].std(),
    'min': df['price'].min(),
    'max': df['price'].max(),
    'cv': df['price'].std() / df['price'].mean(),
    'negatives': (df['price'] < 0).sum(),
    'neg_pct': (df['price'] < 0).sum() / len(df) * 100
}
print(f"✅ Mean={stats['mean']:.2f}, Std={stats['std']:.2f}, CV={stats['cv']:.3f}")
print(f"   Negatives: {stats['negatives']} ({stats['neg_pct']:.2f}%)")

print("\n[3/3] Creating visualizations...")
# Timeline
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(df.index, df['price'], linewidth=0.5, alpha=0.7)
ax.axhline(0, color='red', linestyle='--', linewidth=1)
ax.set_title('Price Timeline', fontweight='bold')
ax.set_ylabel('Price (EUR/MWh)')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'price_01_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Exploration complete")

# =============================================================================
# PHASE 2: PREPROCESSING & FEATURE ENGINEERING
# =============================================================================
print_section("PHASE 2: PREPROCESSING & FEATURE ENGINEERING")

print("\n[1/5] Creating time features...")
df_feat = df.copy()
df_feat['hour'] = df_feat.index.hour
df_feat['day_of_week'] = df_feat.index.dayofweek
df_feat['day_of_ month'] = df_feat.index.day
df_feat['month'] = df_feat.index.month
df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
df_feat['is_peak'] = ((df_feat['hour'] >= 7) & (df_feat['hour'] <= 9) | 
                      (df_feat['hour'] >= 17) & (df_feat['hour'] <= 20)).astype(int)
# Cyclic
df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
print(f"✅ Time features: {8} added")

print("\n[2/5] Creating lag features...")
for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
    df_feat[f'lag_{lag}'] = df_feat['price'].shift(lag)
print(f"✅ Lag features: 8 added")

print("\n[3/5] Creating rolling features...")
for window in [3, 6, 12, 24]:
    df_feat[f'rolling_mean_{window}'] = df_feat['price'].shift(1).rolling(window).mean()
    df_feat[f'rolling_std_{window}'] = df_feat['price'].shift(1).rolling(window).std()
print(f"✅ Rolling features: {4*2} added")

print("\n[4/5] Creating difference features...")
df_feat['diff_1'] = df_feat['price'].diff(1)
df_feat['diff_24'] = df_feat['price'].diff(24)
print(f"✅ Difference features: 2 added")

print("\n[5/5] Creating price-specific features...")
df_feat['is_negative'] = (df_feat['price'] < 0).astype(int)
df_feat['momentum_3h'] = df_feat['price'] - df_feat['price'].shift(3)
print(f"✅ Price-specific: 2 added")

# Drop NaNs
df_feat = df_feat.dropna()
print(f"\n✅ Total features: {len(df_feat.columns)-1}")
print(f"   Rows after dropna: {len(df_feat)} (dropped {len(df) - len(df_feat)})")

# =============================================================================
# PHASE 3: TRAIN/VAL/TEST SPLIT
# =============================================================================
print_section("PHASE 3: DATASET SPLIT")

val_start = '2024-07-01'
test_start = '2024-10-01'

train = df_feat[:val_start].copy()
val = df_feat[val_start:test_start].copy()
test = df_feat[test_start:].copy()

print(f"Train: {len(train)} hours ({len(train)/len(df_feat)*100:.1f}%)")
print(f"Val:   {len(val)} hours ({len(val)/len(df_feat)*100:.1f}%)")
print(f"Test:  {len(test)} hours ({len(test)/len(df_feat)*100:.1f}%)")

# Separate X and y
feature_cols = [c for c in df_feat.columns if c != 'price']
X_train, y_train = train[feature_cols].values, train['price'].values
X_val, y_val = val[feature_cols].values, val['price'].values
X_test, y_test = test[feature_cols].values, test['price'].values

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("✅ Features scaled")

# Save processed data
train_proc = pd.DataFrame(X_train_scaled, columns=feature_cols, index=train.index)
train_proc['price'] = y_train
val_proc = pd.DataFrame(X_val_scaled, columns=feature_cols, index=val.index)
val_proc['price'] = y_val
test_proc = pd.DataFrame(X_test_scaled, columns=feature_cols, index=test.index)
test_proc['price'] = y_test

train_proc.to_csv(proc_dir / 'price_train.csv')
val_proc.to_csv(proc_dir / 'price_val.csv')
test_proc.to_csv(proc_dir / 'price_test.csv')
print("✅ Processed data saved")

# =============================================================================
# PHASE 4: BASELINE MODELS
# =============================================================================
print_section("PHASE 4: BASELINE MODELS")

results = []

# Naive
naive_pred = np.full(len(y_test), y_train[-1])
results.append(evaluate_model(y_test, naive_pred, 'Naive'))
print(f"Naive:            R²={results[-1]['R²']:7.4f}  RMSE={results[-1]['RMSE']:6.2f}")

# Seasonal Naive (24h)
last_24 = y_train[-24:]
seasonal_pred = np.tile(last_24, int(np.ceil(len(y_test)/24)))[:len(y_test)]
results.append(evaluate_model(y_test, seasonal_pred, 'Seasonal Naive (24h)'))
print(f"Seasonal Naive:   R²={results[-1]['R²']:7.4f}  RMSE={results[-1]['RMSE']:6.2f}")

# Mean
mean_pred = np.full(len(y_test), y_train.mean())
results.append(evaluate_model(y_test, mean_pred, 'Mean'))
print(f"Mean:             R²={results[-1]['R²']:7.4f}  RMSE={results[-1]['RMSE']:6.2f}")

# =============================================================================
# PHASE 5: MACHINE LEARNING MODELS
# =============================================================================
print_section("PHASE 5: MACHINE LEARNING MODELS")

# Random Forest
print("\n[1/3] Training Random Forest...")
start = time.time()
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1, verbose=0)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
results.append(evaluate_model(y_test, rf_pred, 'Random Forest'))
print(f"✅ Random Forest:    R²={results[-1]['R²']:7.4f}  RMSE={results[-1]['RMSE']:6.2f}  ({time.time()-start:.1f}s)")

# XGBoost
print("\n[2/3] Training XGBoost...")
start = time.time()
xgb_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=7, learning_rate=0.05,
    random_state=42, n_jobs=-1, verbosity=0
)
xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
xgb_pred = xgb_model.predict(X_test_scaled)
results.append(evaluate_model(y_test, xgb_pred, 'XGBoost'))
print(f"✅ XGBoost:          R²={results[-1]['R²']:7.4f}  RMSE={results[-1]['RMSE']:6.2f}  ({time.time()-start:.1f}s)")

# LightGBM
print("\n[3/3] Training LightGBM...")
start = time.time()
lgb_train = lgb.Dataset(X_train_scaled, y_train)
lgb_val = lgb.Dataset(X_val_scaled, y_val, reference=lgb_train)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1,
    'random_state': 42
}
lgb_model = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=[lgb_val],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
lgb_pred = lgb_model.predict(X_test_scaled, num_iteration=lgb_model.best_iteration)
results.append(evaluate_model(y_test, lgb_pred, 'LightGBM'))
print(f"✅ LightGBM:         R²={results[-1]['R²']:7.4f}  RMSE={results[-1]['RMSE']:6.2f}  ({time.time()-start:.1f}s)")

# =============================================================================
# PHASE 6: RESULTS & VISUALIZATION
# =============================================================================
print_section("PHASE 6: RESULTS SUMMARY")

# Create results DataFrame
results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
print("\n" + results_df.to_string(index=False))

# Save results
results_df.to_csv(metrics_dir / 'price_all_models.csv', index=False)
print(f"\n✅ Results saved to {metrics_dir / 'price_all_models.csv'}")

# Best model
best = results_df.iloc[0]
print(f"\n🏆 BEST MODEL: {best['Model']}")
print(f"   R² = {best['R²']:.4f}")
print(f"   RMSE = {best['RMSE']:.2f} EUR/MWh")
print(f"   MAE = {best['MAE']:.2f} EUR/MWh")

# Visualization - Model Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].barh(results_df['Model'], results_df['R²'], edgecolor='black')
axes[0, 0].set_xlabel('R² Score')
axes[0, 0].set_title('R² Score by Model', fontweight='bold')
axes[0, 0].grid(alpha=0.3, axis='x')

axes[0, 1].barh(results_df['Model'], results_df['RMSE'], color='coral', edgecolor='black')
axes[0, 1].set_xlabel('RMSE')
axes[0, 1].set_title('RMSE by Model', fontweight='bold')
axes[0, 1].grid(alpha=0.3, axis='x')

axes[1, 0].barh(results_df['Model'], results_df['MAE'], color='seagreen', edgecolor='black')
axes[1, 0].set_xlabel('MAE')
axes[1, 0].set_title('MAE by Model', fontweight='bold')
axes[1, 0].grid(alpha=0.3, axis='x')

axes[1, 1].barh(results_df['Model'], results_df['MAPE'], color='purple', edgecolor='black')
axes[1, 1].set_xlabel('MAPE (%)')
axes[1, 1].set_title('MAPE by Model', fontweight='bold')
axes[1, 1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(figures_dir / 'price_02_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Comparison chart saved")

# Forecast visualization (best model)
best_pred = lgb_pred if best['Model'] == 'LightGBM' else (xgb_pred if best['Model'] == 'XGBoost' else rf_pred)
plot_hours = min(7*24, len(y_test))

fig, ax = plt.subplots(figsize=(16, 6))
test_idx = test.index[:plot_hours]
ax.plot(test_idx, y_test[:plot_hours], linewidth=2.5, label='Actual', color='black', zorder=5)
ax.plot(test_idx, best_pred[:plot_hours], linewidth=2, label=f'{best["Model"]} Forecast', 
        alpha=0.8, linestyle='--', color='steelblue')
ax.fill_between(test_idx, y_test[:plot_hours], best_pred[:plot_hours], alpha=0.2)
ax.axhline(0, color='red', linestyle='-', linewidth=1)
ax.set_title(f'{best["Model"]} - 7 Days Forecast', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Price (EUR/MWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'price_03_best_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Forecast visualization saved")

# Feature Importance (LightGBM)
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_model.feature_importance()
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(12, 8))
top_features = importance_df.head(20)
ax.barh(range(len(top_features)), top_features['importance'].values, color='darkorange', edgecolor='black')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'].values)
ax.invert_yaxis()
ax.set_xlabel('Importance')
ax.set_title('LightGBM - Top 20 Features', fontweight='bold')
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(figures_dir / 'price_04_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Feature importance saved")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print_section("PIPELINE COMPLETE - FINAL SUMMARY")

summary = {
    'execution_timestamp': pd.Timestamp.now().isoformat(),
    'data_points': int(stats['count']),
    'train_size': int(len(train)),
    'val_size': int(len(val)),
    'test_size': int(len(test)),
    'features_created': int(len(feature_cols)),
    'price_mean': float(stats['mean']),
    'price_std': float(stats['std']),
    'price_cv': float(stats['cv']),
    'negatives_count': int(stats['negatives']),
    'negatives_pct': float(stats['neg_pct']),
    'best_model': str(best['Model']),
    'best_r2': float(best['R²']),
    'best_rmse': float(best['RMSE']),
    'best_mae': float(best['MAE']),
    'top_5_features': importance_df.head(5)['feature'].tolist()
}

# Save summary
with open(metrics_dir / 'price_pipeline_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"""
📊 DATA PROCESSED:
   Total hours: {summary['data_points']}
   Train/Val/Test: {summary['train_size']}/{summary['val_size']}/{summary['test_size']}
   Features: {summary['features_created']}

📈 PRICE CHARACTERISTICS:
   Mean: {summary['price_mean']:.2f} EUR/MWh
   Std: {summary['price_std']:.2f} EUR/MWh
   CV: {summary['price_cv']:.3f}
   Negatives: {summary['negatives_count']} ({summary['negatives_pct']:.2f}%)

🏆 BEST MODEL: {summary['best_model']}
   R² = {summary['best_r2']:.4f}
   RMSE = {summary['best_rmse']:.2f} EUR/MWh
   MAE = {summary['best_mae']:.2f} EUR/MWh

📁 OUTPUT FILES:
   ✅ Processed data: {proc_dir}
   ✅ Metrics: {metrics_dir}
   ✅ Figures: {figures_dir}

⭐ TOP 5 FEATURES:
""")
for i, feat in enumerate(summary['top_5_features'], 1):
    print(f"   {i}. {feat}")

print("\n" + "="*80)
print("✨ PRICE FORECASTING PIPELINE SUCCESSFULLY COMPLETED!")
print("="*80)
