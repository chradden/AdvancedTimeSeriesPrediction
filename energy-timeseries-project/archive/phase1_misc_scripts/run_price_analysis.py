#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Price Analysis - Simplified Automated Execution
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

print("="*80)
print("PRICE ANALYSIS - AUTOMATED EXECUTION")
print("="*80)

# Paths
data_path = Path('data/raw/price_day_ahead_2022-01-01_2024-12-31_hour.csv')
results_dir = Path('results')
figures_dir = results_dir / 'figures'
metrics_dir = results_dir / 'metrics'
figures_dir.mkdir(parents=True, exist_ok=True)
metrics_dir.mkdir(parents=True, exist_ok=True)

# Load data
print("\n[1/7] Loading data...")
df = pd.read_csv(data_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df=df.sort_index()
df.rename(columns={'value': 'price'}, inplace=True)
print(f"✅ Loaded: {df.shape} from {df.index.min()} to {df.index.max()}")

# Basic statistics
print("\n[2/7] Computing statistics...")
stats = {
    'count': len(df),
    'mean': df['price'].mean(),
    'std': df['price'].std(),
    'min': df['price'].min(),
    'max': df['price'].max(),
    'median': df['price'].median(),
    'negative_count': (df['price'] < 0).sum(),
    'negative_pct': (df['price'] < 0).sum() / len(df) * 100,
    'zero_count': (df['price'] == 0).sum()
}
print(f"✅ Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}, CV: {stats['std']/stats['mean']:.3f}")
print(f"   Negative: {stats['negative_count']} ({stats['negative_pct']:.2f}%)")

# Timeline plot
print("\n[3/7] Creating timeline visualization...")
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(df.index, df['price'], linewidth=0.5, alpha=0.7, color='steelblue')
ax.axhline(0, color='red', linestyle='--', linewidth=1)
ax.set_title('Electricity Price - Full Timeline', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Price (EUR/MWh)')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'price_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Timeline saved")

# Distribution
print("\n[4/7] Creating distribution plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['price'], bins=100, edgecolor='black', alpha=0.7)
axes[0].axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.2f}")
axes[0].axvline(0, color='black', linestyle='-', linewidth=1)
axes[0].set_title('Price Distribution')
axes[0].set_xlabel('Price (EUR/MWh)')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].boxplot(df['price'])
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_title('Box Plot')
axes[1].set_ylabel('Price (EUR/MWh)')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'price_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Distribution saved")

# Hourly patterns
print("\n[5/7] Analyzing temporal patterns...")
df['hour'] = df.index.hour
df['dow'] = df.index.dayofweek
hourly_mean = df.groupby('hour')['price'].mean()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(hourly_mean.index, hourly_mean.values, marker='o', linewidth=2, markersize=6)
ax.set_title('Average Price by Hour', fontweight='bold')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Average Price (EUR/MWh)')
ax.grid(alpha=0.3)
ax.set_xticks(range(24))
plt.tight_layout()
plt.savefig(figures_dir / 'price_hourly_pattern.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Peak hour: {hourly_mean.idxmax()}:00 ({hourly_mean.max():.2f} EUR/MWh)")

# Train/Val/Test split
print("\n[6/7] Creating dataset split...")
val_start = '2024-07-01'
test_start = '2024-10-01'

train = df[:val_start]
val = df[val_start:test_start]
test = df[test_start:]

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(train.index, train['price'], linewidth=0.8, label='Train', alpha=0.7)
ax.plot(val.index, val['price'], linewidth=0.8, label='Validation', alpha=0.7)
ax.plot(test.index, test['price'], linewidth=0.8, label='Test', alpha=0.7)
ax.axvline(pd.to_datetime(val_start), color='orange', linestyle='--', linewidth=2)
ax.axvline(pd.to_datetime(test_start), color='red', linestyle='--', linewidth=2)
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.set_title('Train/Val/Test Split')
ax.set_xlabel('Date')
ax.set_ylabel('Price (EUR/MWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'price_split.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# Save summary
print("\n[7/7] Saving exploration summary...")
summary_df = pd.DataFrame([stats])
summary_df.to_csv(metrics_dir / 'price_exploration_summary.csv', index=False)
print("✅ Summary saved")

# Final report
print("\n" + "="*80)
print("EXPLORATION COMPLETE")
print("="*80)
print(f"📊 Data: {stats['count']} hours from {df.index.min().date()} to {df.index.max().date()}")
print(f"📈 Price range: [{stats['min']:.2f}, {stats['max']:.2f}] EUR/MWh")
print(f"🔴 Negatives: {stats['negative_count']} ({stats['negative_pct']:.2f}%)")
print(f"📁 Figures saved to: {figures_dir}")
print(f"📁 Metrics saved to: {metrics_dir}")
print("="*80)
