#!/usr/bin/env python3
"""
Quick validation: Test if Wind Offshore fix works
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("WIND OFFSHORE FIX VALIDATION")
print("=" * 80)

# Load Wind Offshore data
raw_file = Path('data/raw/wind_offshore_2022-01-01_2024-12-31_hour.csv')
df = pd.read_csv(raw_file, parse_dates=['timestamp'])

# Test period from updated config
TEST_PERIOD = {'start': '2023-07-01', 'end': '2023-07-30'}

print(f"\n1. Data Overview:")
print(f"   Total period: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"   Total hours: {len(df)}")

# Check old problematic test period (last 30 days)
df_sorted = df.sort_values('timestamp')
last_30_days = df_sorted.tail(30 * 24)
print(f"\n2. OLD Test Period (Last 30 days):")
print(f"   Period: {last_30_days['timestamp'].min()} to {last_30_days['timestamp'].max()}")
print(f"   Mean: {last_30_days['value'].mean():.2f} MW")
print(f"   Std:  {last_30_days['value'].std():.2f} MW")
print(f"   Zero %: {(last_30_days['value'] == 0).sum() / len(last_30_days) * 100:.1f}%")
if last_30_days['value'].std() < 1:
    print(f"   ❌ PROBLEMATIC: Constant or near-constant data!")

# Check new test period
test_mask = (df['timestamp'] >= TEST_PERIOD['start']) & (df['timestamp'] <= TEST_PERIOD['end'])
new_test_data = df[test_mask]

print(f"\n3. NEW Test Period (Smart Split):")
print(f"   Period: {TEST_PERIOD['start']} to {TEST_PERIOD['end']}")
print(f"   Hours: {len(new_test_data)}")
print(f"   Mean: {new_test_data['value'].mean():.2f} MW")
print(f"   Std:  {new_test_data['value'].std():.2f} MW")
print(f"   Zero %: {(new_test_data['value'] == 0).sum() / len(new_test_data) * 100:.1f}%")

if new_test_data['value'].std() > 100:
    print(f"   ✅ GOOD: Sufficient variance for modeling!")
else:
    print(f"   ⚠️  STILL PROBLEMATIC: Low variance")

# Distribution comparison
print(f"\n4. Distribution Comparison:")
train_mask = ~test_mask
train_data = df[train_mask]

print(f"   Train - Mean: {train_data['value'].mean():.2f}, Std: {train_data['value'].std():.2f}")
print(f"   Test  - Mean: {new_test_data['value'].mean():.2f}, Std: {new_test_data['value'].std():.2f}")

mean_diff = abs(train_data['value'].mean() - new_test_data['value'].mean())
std_ratio = new_test_data['value'].std() / train_data['value'].std()

print(f"\n   Mean difference: {mean_diff:.2f} MW")
print(f"   Std ratio (test/train): {std_ratio:.3f}")

if std_ratio > 0.3 and std_ratio < 3.0:
    print(f"   ✅ Distributions are compatible!")
else:
    print(f"   ⚠️  Large distribution shift")

# Expected R² prediction
print(f"\n5. Expected Outcome:")
if new_test_data['value'].std() > 100 and std_ratio > 0.3:
    print(f"   ✅ Wind Offshore should now achieve R² > 0.3")
    print(f"   Expected R²: 0.4 - 0.6 (wind is inherently difficult)")
else:
    print(f"   ⚠️  May still have issues, but better than R²=0")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if new_test_data['value'].std() > last_30_days['value'].std() * 10:
    print("\n🎉 SUCCESS! New test period is much better!")
    print("   Old test period: Std = {:.2f} (nearly constant)".format(last_30_days['value'].std()))
    print("   New test period: Std = {:.2f} (good variance)".format(new_test_data['value'].std()))
    print("\n   Notebook 10 should now produce meaningful results for Wind Offshore!")
else:
    print("\n⚠️  Improvement, but may need further tuning")

print("\n" + "=" * 80)
