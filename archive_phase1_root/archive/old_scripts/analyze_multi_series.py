#!/usr/bin/env python3
"""
Analyze multi-series comparison results and generate insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    print("=" * 80)
    print("MULTI-SERIES ANALYSIS - COMPREHENSIVE REPORT")
    print("=" * 80)
    
    # Load results
    results_file = Path('results/metrics/multi_series_comparison.csv')
    df = pd.read_csv(results_file)
    
    # Display raw results
    print("\n📊 Raw Results:")
    print(df.to_string(index=False))
    
    # Analysis by dataset
    print("\n\n" + "=" * 80)
    print("ANALYSIS BY DATASET (Best Model per Dataset)")
    print("=" * 80)
    
    best_per_dataset = df.loc[df.groupby('Dataset')['MAE'].idxmin()]
    
    for _, row in best_per_dataset.iterrows():
        dataset = row['Dataset']
        model = row['Model']
        mae = row['MAE']
        r2 = row['R2']
        mape = row['MAPE'] if pd.notna(row['MAPE']) else 'N/A'
        
        print(f"\n🔹 {dataset.upper()}:")
        print(f"   Winner: {model}")
        print(f"   MAE:    {mae:.2f}")
        print(f"   R²:     {r2:.4f}")
        print(f"   MAPE:   {mape}")
        
        # Interpretation
        if dataset == 'consumption':
            print(f"   💡 Excellent performance! R² > 0.95 means the model captures")
            print(f"      demand patterns very well. MAE ~1441 MW on {mae:.0f} MW scale.")
        elif dataset == 'solar':
            print(f"   ⚠️  Moderate performance. R² = {r2:.4f} is lower than notebook 05")
            print(f"      (which had R² > 0.98). Possible data processing issue.")
        elif dataset == 'price_day_ahead':
            print(f"   ⚠️  Challenging dataset! R² = {r2:.4f} shows price forecasting")
            print(f"      is inherently difficult due to market volatility.")
        elif dataset == 'wind_onshore':
            print(f"   ⚠️  Moderate success. R² = {r2:.4f}. Wind is harder to predict")
            print(f"      than solar due to less predictable patterns.")
        elif dataset == 'wind_offshore':
            print(f"   ❌ FAILED! R² = 0 means model is no better than mean prediction.")
            print(f"      Likely data issue or insufficient features.")
    
    # Model comparison
    print("\n\n" + "=" * 80)
    print("MODEL COMPARISON (XGBoost vs LightGBM)")
    print("=" * 80)
    
    xgb_df = df[df['Model'] == 'XGBoost'].set_index('Dataset')
    lgbm_df = df[df['Model'] == 'LightGBM'].set_index('Dataset')
    
    print("\nWins per model:")
    wins = best_per_dataset['Model'].value_counts()
    print(f"  XGBoost:  {wins.get('XGBoost', 0)}/{len(best_per_dataset)} datasets")
    print(f"  LightGBM: {wins.get('LightGBM', 0)}/{len(best_per_dataset)} datasets")
    
    # Dataset difficulty ranking
    print("\n\n" + "=" * 80)
    print("DATASET DIFFICULTY RANKING (by R²)")
    print("=" * 80)
    
    difficulty_df = best_per_dataset[['Dataset', 'R2', 'MAE']].sort_values('R2', ascending=False)
    
    print("\nEasiest to Hardest:")
    for i, (_, row) in enumerate(difficulty_df.iterrows(), 1):
        r2 = row['R2']
        dataset = row['Dataset']
        
        if r2 > 0.9:
            emoji = "🟢"
            label = "Easy"
        elif r2 > 0.7:
            emoji = "🟡"
            label = "Medium"
        elif r2 > 0.4:
            emoji = "🟠"
            label = "Hard"
        else:
            emoji = "🔴"
            label = "Very Hard"
        
        print(f"  {i}. {emoji} {dataset:20s} R² = {r2:.4f}  ({label})")
    
    # Recommendations
    print("\n\n" + "=" * 80)
    print("RECOMMENDATIONS & NEXT STEPS")
    print("=" * 80)
    
    print("\n1. Solar Data Issue:")
    print("   - Notebook 05 achieved R² > 0.98, but multi-series shows R² = 0.83")
    print("   - ACTION: Check if data preprocessing differs between notebooks")
    print("   - Possible cause: Different train/test splits or feature engineering")
    
    print("\n2. Wind Offshore Failure:")
    print("   - R² = 0 is a red flag")
    print("   - ACTION: Investigate data quality and feature completeness")
    print("   - Check for missing values or corrupted data")
    
    print("\n3. Price Forecasting:")
    print("   - R² = 0.68 is expected for electricity prices (volatile market)")
    print("   - Consider external features: fuel prices, weather forecasts")
    
    print("\n4. Overall Strategy:")
    print("   - Focus on consumption (best performer) for production deployment")
    print("   - Improve solar model to match notebook 05 performance")
    print("   - Consider ensemble methods for price prediction")
    
    print("\n" + "=" * 80)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # R² comparison
    datasets = best_per_dataset['Dataset'].values
    r2_values = best_per_dataset['R2'].values
    colors = ['green' if r2 > 0.9 else 'orange' if r2 > 0.5 else 'red' for r2 in r2_values]
    
    axes[0].barh(datasets, r2_values, color=colors, alpha=0.7)
    axes[0].set_xlabel('R² Score', fontsize=12)
    axes[0].set_title('Model Performance by Dataset (R²)', fontsize=14, fontweight='bold')
    axes[0].axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Poor')
    axes[0].axvline(0.8, color='orange', linestyle='--', alpha=0.5, label='Good')
    axes[0].axvline(0.9, color='green', linestyle='--', alpha=0.5, label='Excellent')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # MAE comparison
    mae_values = best_per_dataset['MAE'].values
    axes[1].barh(datasets, mae_values, alpha=0.7)
    axes[1].set_xlabel('MAE (unit depends on dataset)', fontsize=12)
    axes[1].set_title('Absolute Error by Dataset', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('results/figures/multi_series_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved: results/figures/multi_series_comparison.png")

if __name__ == '__main__':
    main()
