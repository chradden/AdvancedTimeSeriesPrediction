#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Price Data Exploration - Automated Execution Script
Extracted from 01_price_data_exploration.ipynb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

print("="*80)
print("PRICE DATA EXPLORATION - AUTOMATED EXECUTION")
print("="*80)

# Define paths
data_path = Path('data/raw/price_day_ahead_2022-01-01_2024-12-31_hour.csv')
results_dir = Path('results/figures')
results_dir.mkdir(parents=True, exist_ok=True)

# 1. Load Data
print("\n1. Loading data...")
df = pd.read_csv(data_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df.sort_index(inplace=True)
df.rename(columns={'value': 'price'}, inplace=True)
print(f"✅ Data loaded: {df.shape}")
print(f"   Date range: {df.index.min()} to {df.index.max()}")
print(f"   Total hours: {len(df)}")

# 2. Data Quality Checks
print("\n2. Data Quality Checks...")
missing = df.isnull().sum().sum()
zeros = (df['price'] == 0).sum()
negatives = (df['price'] < 0).sum()
print(f"   Missing values: {missing}")
print(f"   Zero values: {zeros}")
print(f"   Negative prices: {negatives} ({negatives/len(df)*100:.2f}%)")
if negatives > 0:
    print(f"   Min price: {df['price'].min():.2f} EUR/MWh")

# 3. Statistical Summary
print("\n3. Statistical Summary:")
print(df['price'].describe())
cv = df['price'].std() / df['price'].mean()
print(f"   Coefficient of Variation: {cv:.3f}")
print(f"   Skewness: {df['price'].skew():.3f}")
print(f"   Kurtosis: {df['price'].kurtosis():.3f}")

# 4. Full Timeline
print("\n4. Creating visualizations...")
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(df.index, df['price'], linewidth=0.5, alpha=0.7)
ax.axhline(0, color='red', linestyle='--', linewidth=1, label='Zero line')
ax.set_title('Electricity Price (Day-Ahead) - Full Timeline (2022-2024)', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Price (EUR/MWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(results_dir / 'price_full_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Full timeline saved")

# 5. Yearly timelines
fig, axes = plt.subplots(3, 1, figsize=(16, 10))
for i, (year, color) in enumerate([(2022, 'steelblue'), (2023, 'darkorange'), (2024, 'seagreen')]):
    df_year = df[str(year)]
    axes[i].plot(df_year.index, df_year['price'], linewidth=0.8, color=color)
    axes[i].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[i].set_title(f'{year}: Price Timeline', fontweight='bold')
    axes[i].set_ylabel('Price (EUR/MWh)')
    axes[i].grid(alpha=0.3)
axes[2].set_xlabel('Date')
plt.tight_layout()
plt.savefig(results_dir / 'price_yearly_timelines.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Yearly timelines saved")

# 6. Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['price'], bins=100, edgecolor='black', alpha=0.7)
axes[0].axvline(df['price'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["price"].mean():.2f}')
axes[0].axvline(df['price'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["price"].median():.2f}')
axes[0].axvline(0, color='black', linestyle='-', linewidth=1, label='Zero')
axes[0].set_title('Price Distribution', fontweight='bold')
axes[0].set_xlabel('Price (EUR/MWh)')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].boxplot(df['price'], vert=True)
axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
axes[1].set_title('Price Box Plot', fontweight='bold')
axes[1].set_ylabel('Price (EUR/MWh)')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(results_dir / 'price_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Distribution plots saved")

# 7. Temporal patterns
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# By hour
hourly_avg = df.groupby('hour')['price'].mean()
hourly_std = df.groupby('hour')['price'].std()
axes[0].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=6)
axes[0].fill_between(hourly_avg.index, 
                       hourly_avg.values - hourly_std.values, 
                       hourly_avg.values + hourly_std.values, 
                       alpha=0.3)
axes[0].set_title('Average Price by Hour of Day', fontweight='bold')
axes[0].set_xlabel('Hour')
axes[0].set_ylabel('Average Price (EUR/MWh)')
axes[0].grid(alpha=0.3)
axes[0].set_xticks(range(24))

# By day of week
dow_avg = df.groupby('day_of_week')['price'].mean()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
axes[1].bar(range(7), dow_avg.values, alpha=0.7, edgecolor='black')
axes[1].set_title('Average Price by Day of Week', fontweight='bold')
axes[1].set_xlabel('Day of Week')
axes[1].set_ylabel('Average Price (EUR/MWh)')
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(days)
axes[1].grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(results_dir / 'price_temporal_patterns.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Temporal patterns saved")

# 8. Seasonal pattern
fig, ax = plt.subplots(figsize=(12, 5))
monthly_avg = df.groupby('month')['price'].mean()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.bar(range(1, 13), monthly_avg.values, alpha=0.7, edgecolor='black')
ax.set_title('Average Price by Month', fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Average Price (EUR/MWh)')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(months)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(results_dir / 'price_seasonal_pattern.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Seasonal pattern saved")

# 9. Autocorrelation
try:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    plot_acf(df['price'].dropna(), lags=168, ax=axes[0])
    axes[0].set_title('Autocorrelation Function (ACF)', fontweight='bold')
    plot_pacf(df['price'].dropna(), lags=168, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation Function (PACF)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'price_autocorrelation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Autocorrelation plots saved")
except Exception as e:
    print(f"   ⚠️ Autocorrelation plot skipped: {e}")

# 10. Spikes Analysis
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR

spikes_low = df[df['price'] < lower_bound]
spikes_high = df[df['price'] > upper_bound]

print(f"\n5. Spike Analysis:")
print(f"   Low spikes (< {lower_bound:.2f}): {len(spikes_low)} ({len(spikes_low)/len(df)*100:.2f}%)")
print(f"   High spikes (> {upper_bound:.2f}): {len(spikes_high)} ({len(spikes_high)/len(df)*100:.2f}%)")

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(df.index, df['price'], linewidth=0.5, alpha=0.5, label='Price')
ax.scatter(spikes_low.index, spikes_low['price'], color='blue', s=20, label=f'Low spikes ({len(spikes_low)})', zorder=5)
ax.scatter(spikes_high.index, spikes_high['price'], color='red', s=20, label=f'High spikes ({len(spikes_high)})', zorder=5)
ax.axhline(upper_bound, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(lower_bound, color='blue', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.set_title('Price Spikes Detection (3×IQR method)', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Price (EUR/MWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(results_dir / 'price_spikes.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Spike detection plot saved")

# 11. Volatility
df['rolling_std_24h'] = df['price'].rolling(window=24).std()
df['rolling_std_168h'] = df['price'].rolling(window=168).std()

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(df.index, df['rolling_std_24h'], linewidth=1, label='24h Rolling Std', alpha=0.7)
ax.plot(df.index, df['rolling_std_168h'], linewidth=1.5, label='168h Rolling Std', alpha=0.7)
ax.set_title('Price Volatility Over Time', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Standard Deviation (EUR/MWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(results_dir / 'price_volatility.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Volatility plot saved")

# 12. Train/Val/Test Split
val_start = '2024-07-01'
test_start = '2024-10-01'

train = df[:val_start]
val = df[val_start:test_start]
test = df[test_start:]

print(f"\n6. Dataset Split:")
print(f"   Train: {train.index.min()} to {train.index.max()} ({len(train)} hours, {len(train)/len(df)*100:.1f}%)")
print(f"   Val:   {val.index.min()} to {val.index.max()} ({len(val)} hours, {len(val)/len(df)*100:.1f}%)")
print(f"   Test:  {test.index.min()} to {test.index.max()} ({len(test)} hours, {len(test)/len(df)*100:.1f}%)")

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(train.index, train['price'], linewidth=0.8, label='Train', alpha=0.7)
ax.plot(val.index, val['price'], linewidth=0.8, label='Validation', alpha=0.7)
ax.plot(test.index, test['price'], linewidth=0.8, label='Test', alpha=0.7)
ax.axvline(pd.to_datetime(val_start), color='orange', linestyle='--', linewidth=2)
ax.axvline(pd.to_datetime(test_start), color='red', linestyle='--', linewidth=2)
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.set_title('Train/Val/Test Split', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Price (EUR/MWh)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(results_dir / 'price_train_val_test_split.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Split visualization saved")

# Final Summary
print("\n" + "="*80)
print("PRICE DATA EXPLORATION - SUMMARY")
print("="*80)
print(f"\n📊 DATA COMPLETENESS:")
print(f"   Date range: {df.index.min()} to {df.index.max()}")
print(f"   Total hours: {len(df)}")
print(f"   Missing values: {missing}")

print(f"\n📈 STATISTICAL PROPERTIES:")
print(f"   Mean: {df['price'].mean():.2f} EUR/MWh")
print(f"   Median: {df['price'].median():.2f} EUR/MWh")
print(f"   Std Dev: {df['price'].std():.2f} EUR/MWh")
print(f"   CV: {cv:.3f}")
print(f"   Min: {df['price'].min():.2f} EUR/MWh")
print(f"   Max: {df['price'].max():.2f} EUR/MWh")

print(f"\n🔴 SPECIAL CHARACTERISTICS:")
print(f"   Negative prices: {negatives} ({negatives/len(df)*100:.2f}%)")
print(f"   High spikes: {len(spikes_high)}")
print(f"   Low spikes: {len(spikes_low)}")

print(f"\n📅 TEMPORAL PATTERNS:")
print(f"   Peak hour: {hourly_avg.idxmax()}:00 ({hourly_avg.max():.2f} EUR/MWh)")
print(f"   Low hour: {hourly_avg.idxmin()}:00 ({hourly_avg.min():.2f} EUR/MWh)")

print(f"\n✅ Data exploration complete!")
print(f"   Figures saved to: {results_dir}")
print("="*80)
