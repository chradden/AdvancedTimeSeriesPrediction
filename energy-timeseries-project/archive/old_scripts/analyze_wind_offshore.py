#!/usr/bin/env python3
"""
Analyze Wind Offshore data to understand why R² = 0.00
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 80)
print("WIND OFFSHORE DATA ANALYSIS")
print("=" * 80)

# Load raw data
raw_file = Path('data/raw/wind_offshore_2022-01-01_2024-12-31_hour.csv')

if not raw_file.exists():
    print(f"\n❌ File not found: {raw_file}")
    print("   Checking available files...")
    for f in Path('data/raw').glob('wind_offshore*.csv'):
        print(f"   Found: {f.name}")
else:
    print(f"\n✅ Loading: {raw_file.name}")
    
    df = pd.read_csv(raw_file, parse_dates=['timestamp'])
    
    print(f"\n1. Basic Information:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Total hours: {len(df)}")
    
    print(f"\n2. Missing Values:")
    missing = df['value'].isna().sum()
    missing_pct = missing / len(df) * 100
    print(f"   Missing: {missing} ({missing_pct:.2f}%)")
    
    print(f"\n3. Value Statistics:")
    print(f"   Mean:   {df['value'].mean():.2f}")
    print(f"   Median: {df['value'].median():.2f}")
    print(f"   Std:    {df['value'].std():.2f}")
    print(f"   Min:    {df['value'].min():.2f}")
    print(f"   Max:    {df['value'].max():.2f}")
    print(f"   Range:  {df['value'].max() - df['value'].min():.2f}")
    
    # Check for zero values
    zero_count = (df['value'] == 0).sum()
    zero_pct = zero_count / len(df) * 100
    print(f"\n4. Zero Values:")
    print(f"   Zeros: {zero_count} ({zero_pct:.2f}%)")
    
    # Check variance
    variance = df['value'].var()
    print(f"\n5. Variance Analysis:")
    print(f"   Variance: {variance:.2f}")
    if variance < 1:
        print(f"   ⚠️  VERY LOW VARIANCE!")
        print(f"   This could cause R² = 0 (model predicts mean)")
    
    # Check for constant values
    unique_values = df['value'].nunique()
    print(f"\n6. Unique Values:")
    print(f"   Count: {unique_values}")
    if unique_values < 100:
        print(f"   Top 20 most common values:")
        print(df['value'].value_counts().head(20))
    
    # Check temporal distribution
    print(f"\n7. Temporal Analysis:")
    df_clean = df.dropna()
    
    if len(df_clean) > 0:
        # Group by month
        monthly = df_clean.groupby(df_clean['timestamp'].dt.month)['value'].agg(['mean', 'std', 'count'])
        print(f"\n   Monthly Statistics:")
        print(monthly)
        
        # Check if data is actually constant
        if df_clean['value'].std() < 0.1:
            print(f"\n   ⚠️  CRITICAL: Data is almost constant!")
            print(f"   Standard deviation: {df_clean['value'].std():.6f}")
            print(f"   This explains R² = 0.00")
        
        # Plot data
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Time series
        axes[0].plot(df['timestamp'], df['value'], alpha=0.7, linewidth=0.5)
        axes[0].set_title('Wind Offshore - Time Series', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Value (MW)')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram
        axes[1].hist(df_clean['value'], bins=100, alpha=0.7, edgecolor='black')
        axes[1].set_title('Wind Offshore - Value Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Value (MW)')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('results/figures/wind_offshore_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\n   ✅ Saved visualization: results/figures/wind_offshore_analysis.png")
        
    else:
        print(f"\n   ⚠️  No valid data after removing NaN!")
    
    print(f"\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    if variance < 100:
        print("\n❌ PROBLEM IDENTIFIED: Very low variance in data")
        print("   Possible causes:")
        print("   1. Data source issue (wrong measurement)")
        print("   2. All values are nearly constant")
        print("   3. Data preprocessing error")
        print("\n   RECOMMENDATION:")
        print("   - Check SMARD data source")
        print("   - Verify correct filter/category")
        print("   - Consider using different time period")
        print("   - Contact SMARD API for data validation")
    elif missing_pct > 50:
        print("\n⚠️  HIGH MISSING DATA RATE")
        print("   Over 50% of data is missing")
        print("   Interpolation may not be reliable")
    else:
        print("\n✅ Data looks reasonable")
        print("   Further model-specific investigation needed")
    
    print("\n" + "=" * 80)
