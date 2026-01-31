#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run LSTM models for all 4 Energy Types:
1. Solar
2. Wind Onshore
3. Price
4. Consumption

Loads preprocessed data, trains LSTM, and saves results.
"""

import pandas as pd
import numpy as np
import time
import json
import warnings
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# Disable GPU if not available or just use CPU (simpler)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    print(f"✅ TensorFlow version: {tf.__version__}")
except ImportError:
    print("❌ TensorFlow not installed! Cannot run LSTM models.")
    exit(1)


def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    
    return {
        'Model': model_name,
        'Category': 'Deep Learning',
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }

def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(seq_length, len(X)):
        X_seq.append(X[i-seq_length:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

def run_lstm_for_type(energy_type, train_file, val_file, test_file):
    print(f"\n{'='*40}")
    print(f"Running LSTM for: {energy_type.upper()}")
    print(f"{'='*40}")
    
    # Load data
    try:
        train_df = pd.read_csv(train_file, index_col=0)
        val_df = pd.read_csv(val_file, index_col=0)
        test_df = pd.read_csv(test_file, index_col=0)
        
        target_col = energy_type if energy_type != 'price' else 'price'
        # Fix column names if needed
        if energy_type == 'wind_onshore': target_col = 'wind_power'
        if energy_type == 'solar': target_col = 'solar'
        if energy_type == 'consumption': target_col = 'consumption'
        
        # Check if target col exists
        if target_col not in train_df.columns:
            # Fallback
            cols = [c for c in train_df.columns if 'value' in c or 'price' in c or 'solar' in c or 'wind' in c or 'consumption' in c]
            if cols: target_col = cols[-1] 
            else: 
                print(f"❌ Target column '{target_col}' not found in {train_file}")
                return None

        feature_cols = [c for c in train_df.columns if c != target_col]
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_val = val_df[feature_cols].values
        y_val = val_df[target_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values
        
        print(f"Loaded train/val/test: {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}")
        
    except Exception as e:
        print(f"❌ Error loading data for {energy_type}: {e}")
        return None

    # Prepare sequences
    seq_length = 24
    print(f"Creating sequences (len={seq_length})...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)
    
    # Build Model
    model = Sequential([
        Input(shape=(seq_length, X_train_seq.shape[2])),
        LSTM(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train
    print("Training LSTM...")
    start_time = time.time()
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=30,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0
    )
    train_time = time.time() - start_time
    print(f"✅ Training complete in {train_time:.1f}s")
    
    # Predict
    y_pred = model.predict(X_test_seq, verbose=0).flatten()
    
    # Evaluate
    result = evaluate_model(y_test_seq, y_pred, 'LSTM')
    print(f"✅ Results: R²={result['R²']:.4f}, RMSE={result['RMSE']:.2f}")
    
    # Save generic deep learning results file
    out_file = Path(f"results/metrics/{energy_type}_deep_learning_results_fixed.csv")
    pd.DataFrame([result]).to_csv(out_file, index=False)
    print(f"Saved results to {out_file}")
    
    return result

# Main execution
processed_dir = Path("data/processed")
results = []

# 1. Price
res = run_lstm_for_type('price', 
                  processed_dir / 'price_train.csv', 
                  processed_dir / 'price_val.csv', 
                  processed_dir / 'price_test.csv')
if res: results.append({'type': 'Price', **res})

# 2. Solar
res = run_lstm_for_type('solar', 
                  processed_dir / 'solar_train.csv', 
                  processed_dir / 'solar_val.csv', 
                  processed_dir / 'solar_test.csv')
if res: results.append({'type': 'Solar', **res})

# 3. Wind Onshore
res = run_lstm_for_type('wind_onshore', 
                  processed_dir / 'wind_onshore_train.csv', 
                  processed_dir / 'wind_onshore_val.csv', 
                  processed_dir / 'wind_onshore_test.csv')
if res: results.append({'type': 'Wind Onshore', **res})

# 4. Consumption
res = run_lstm_for_type('consumption', 
                  processed_dir / 'consumption_train.csv', 
                  processed_dir / 'consumption_val.csv', 
                  processed_dir / 'consumption_test.csv')

# Correction for updated script logic
# The consumption pipeline was created by copying wind script. 
# Adapt script only changed content, filenames might be tricky if not careful.
# Let's assume standard names based on pipeline run. 
# Actually consumption pipeline probably saved as 'consumption_train.csv' IF the code was updated.
# But wait, adapter script REPLACED 'wind_onshore' with 'consumption' in CONTENT.
# So the file output lines: train_proc.to_csv(proc_dir / 'wind_onshore_train.csv')
# would become: train_proc.to_csv(proc_dir / 'consumption_train.csv')
# So filenames should be 'consumption_train.csv'.

# Verify file existence logic inside run_lstm_for_type can handle errors.

print("\n" + "="*80)
print("FINAL DEEP LEARNING SUMMARY")
print("="*80)
df_res = pd.DataFrame(results)
if not df_res.empty:
    print(df_res[['type', 'Model', 'R²', 'RMSE', 'MAE']].to_string(index=False))
    df_res.to_csv("results/metrics/all_lstm_results.csv", index=False)
else:
    print("No results generated.")
