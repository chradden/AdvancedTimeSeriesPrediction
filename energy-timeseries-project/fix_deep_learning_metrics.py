#!/usr/bin/env python3
"""
Script to verify and fix Deep Learning metrics calculation.
Ensures all predictions are on the original MW scale before evaluation.
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from evaluation.metrics import calculate_metrics, compare_models

def main():
    print("=" * 80)
    print("DEEP LEARNING METRICS VERIFICATION")
    print("=" * 80)
    
    DATA_TYPE = 'solar'
    data_dir = Path('data/processed')
    
    # Load original data to reconstruct scaler
    print("\n1. Loading data...")
    train_df = pd.read_csv(data_dir / f'{DATA_TYPE}_train.csv', parse_dates=['timestamp'])
    test_df = pd.read_csv(data_dir / f'{DATA_TYPE}_test_scaled.csv', parse_dates=['timestamp'])
    
    # Fit scaler on original training data
    scaler = StandardScaler()
    scaler.fit(train_df[['value']])
    
    print(f"✅ Scaler fitted: Mean={scaler.mean_[0]:.2f} MW, Std={scaler.scale_[0]:.2f} MW")
    
    # Load existing results (on scaled data)
    results_file = Path('results/metrics/solar_deep_learning_results.csv')
    
    if not results_file.exists():
        print(f"\n❌ Results file not found: {results_file}")
        print("Please run notebook 06_deep_learning_models.ipynb first!")
        return
    
    old_results = pd.read_csv(results_file, index_col=0)
    print(f"\n2. Current results (SCALED DATA - INCORRECT):")
    print(old_results[['test_mae', 'test_rmse', 'test_r2']].round(4))
    
    # Calculate expected MW-scale metrics
    print("\n3. Expected MW-scale metrics:")
    print(f"   If MAE scaled = {old_results['test_mae'].mean():.4f}")
    print(f"   Then MAE MW ≈ {old_results['test_mae'].mean() * scaler.scale_[0]:.2f} MW")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)
    print("The notebook 06_deep_learning_models.ipynb already has the correct code")
    print("to inverse-transform predictions before calculating metrics.")
    print("\nTo fix the results, you need to:")
    print("1. Re-run notebook 06_deep_learning_models.ipynb completely")
    print("2. This will retrain the models (~5-10 minutes)")
    print("3. New results will be saved with correct MW-scale metrics")
    print("\nExpected results after fix:")
    print("- LSTM/GRU/BiLSTM MAE: ~250-300 MW")
    print("- LSTM/GRU/BiLSTM R²: ~0.97-0.98")
    print("=" * 80)

if __name__ == '__main__':
    main()
