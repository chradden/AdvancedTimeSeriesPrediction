#!/usr/bin/env python3
"""
Find best test period for Wind Offshore
"""

import pandas as pd
import numpy as np
from pathlib import Path

raw_file = Path('data/raw/wind_offshore_2022-01-01_2024-12-31_hour.csv')
df = pd.read_csv(raw_file, parse_dates=['timestamp'])

print("=" * 80)
print("FINDING BEST TEST PERIOD FOR WIND OFFSHORE")
print("=" * 80)

# Check each month
monthly_stats = []

for year in [2022, 2023, 2024]:
    for month in range(1, 13):
        month_mask = (df['timestamp'].dt.year == year) & (df['timestamp'].dt.month == month)
        month_data = df[month_mask]
        
        if len(month_data) > 0:
            stats = {
                'year': year,
                'month': month,
                'name': f"{year}-{month:02d}",
                'count': len(month_data),
                'mean': month_data['value'].mean(),
                'std': month_data['value'].std(),
                'zero_pct': (month_data['value'] == 0).sum() / len(month_data) * 100,
                'min': month_data['value'].min(),
                'max': month_data['value'].max()
            }
            monthly_stats.append(stats)

df_stats = pd.DataFrame(monthly_stats)

# Sort by std (we want good variance)
df_stats = df_stats.sort_values('std', ascending=False)

print(f"\n📊 Monthly Statistics (sorted by variance):\n")
print(df_stats.to_string(index=False))

# Find best candidates (high variance, low zero percentage)
good_periods = df_stats[(df_stats['std'] > 1000) & (df_stats['zero_pct'] < 50)]

print(f"\n\n✅ BEST TEST PERIODS (High variance, <50% zeros):\n")
if len(good_periods) > 0:
    print(good_periods[['name', 'mean', 'std', 'zero_pct', 'count']].head(10).to_string(index=False))
    
    # Recommend top 3
    print(f"\n\n🎯 TOP 3 RECOMMENDATIONS:")
    for i, (_, row) in enumerate(good_periods.head(3).iterrows(), 1):
        print(f"\n{i}. {row['name']}")
        print(f"   Mean: {row['mean']:.0f} MW, Std: {row['std']:.0f} MW")
        print(f"   Zero %: {row['zero_pct']:.1f}%")
        print(f"   Hours: {row['count']}")
else:
    print("❌ No periods with good characteristics found!")
    print("\nShowing least problematic periods:")
    print(df_stats[['name', 'mean', 'std', 'zero_pct']].head(5).to_string(index=False))

print("\n" + "=" * 80)
