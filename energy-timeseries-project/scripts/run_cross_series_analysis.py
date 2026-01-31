#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CROSS-SERIES ANALYSIS
Compare Price, Solar, Wind Onshore, Consumption (and Offshore if available)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

plt.style.use('seaborn-v0_8-darkgrid')

metrics_dir = Path("results/metrics")
figures_dir = Path("results/figures")
figures_dir.mkdir(parents=True, exist_ok=True)

# 1. Load Summary Data
def load_summary(energy_type):
    f = metrics_dir / f"{energy_type}_all_models_extended.csv"
    if f.exists():
        df = pd.read_csv(f)
        df['Energy Type'] = type_name(energy_type)
        return df
    return None

def type_name(t):
    return {
        'price': 'Price',
        'solar': 'Solar',
        'wind_onshore': 'Wind Onshore',
        'consumption': 'Consumption'
    }.get(t, t)

types = ['price', 'solar', 'wind_onshore', 'consumption']
dfs = []

for t in types:
    d = load_summary(t)
    if d is not None:
        dfs.append(d)

if not dfs:
    print("No extended results found.")
    exit(1)

all_results = pd.concat(dfs, ignore_index=True)

# 2. Add Deep Learning Results if available (from separate run)
lstm_file = metrics_dir / "all_lstm_results.csv"
if lstm_file.exists():
    lstm_df = pd.read_csv(lstm_file)
    # Map type names to match
    lstm_df['Energy Type'] = lstm_df['type'] # Already formatted in script
    # Columns slightly different
    lstm_df['Category'] = 'Deep Learning'
    # Concat
    cols = ['Energy Type', 'Model', 'Category', 'R²', 'RMSE', 'MAE']
    all_results = pd.concat([all_results, lstm_df[cols]], ignore_index=True)

# 3. Best Model per Type
best_models = all_results.loc[all_results.groupby('Energy Type')['R²'].idxmax()]
best_models = best_models.sort_values('R²', ascending=False)

print("\n" + "="*80)
print("🏆 BEST MODEL PER ENERGY TYPE")
print("="*80)
print(best_models[['Energy Type', 'Model', 'Category', 'R²', 'RMSE']].to_string(index=False))

# 4. Visualization: R² Comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=best_models, x='Energy Type', y='R²', hue='Category', dodge=False)
plt.ylim(0.95, 1.005) # Zoom in on top performance
plt.title('Best Model Performance across Energy Types', fontweight='bold')
plt.ylabel('R² Score')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / "cross_series_01_best_r2.png")

# 5. Visualization: Complexity vs Performance
# Scale standard RMSE to MAPE-like logic (normalized RMSE)
# nRMSE = RMSE / Mean
means = {
    'Price': 80.0, # Approx from simple mean
    'Solar': 6640.0,
    'Wind Onshore': 6000.0, # Estimating
    'Consumption': 53000.0
}
# Update means from json summaries
for t in types:
    json_f = metrics_dir / f"{t}_extended_summary.json"
    if json_f.exists():
        with open(json_f, 'r') as f:
            j = json.load(f)
            # Find mean from data_points?? No, mean isn't in summary json sadly.
            # Using estimative means above is acceptable for chart.
            pass

# Create normalized RMSE
all_results['nRMSE'] = all_results.apply(lambda row: row['RMSE'] / means.get(row['Energy Type'], 1.0), axis=1)

plt.figure(figsize=(12, 8))
sns.scatterplot(data=all_results, x='Energy Type', y='R²', hue='Category', style='Category', s=100)
plt.title('All Models Comparison', fontweight='bold')
plt.ylim(0.8, 1.01)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(figures_dir / "cross_series_02_all_models.png")


print("\n✅ Cross-series analysis completed.")
print(f"   Figures saved to {figures_dir}")
