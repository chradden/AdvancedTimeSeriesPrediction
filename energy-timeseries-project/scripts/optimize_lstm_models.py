#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM OPTIMIZATION - Systematisches Tuning für bessere Performance

Testet 4 Optimierungsstrategien:
1. Hyperparameter-Tuning (units, dropout, learning rate)
2. Sequence-Length Experimente (24h, 48h, 168h)
3. Mehr Epochen (50, 100 statt 20)
4. Architektur-Varianten (Bi-LSTM, Stacked LSTM, mit Attention)
"""

import sys
import time
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR / 'src'))

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Config
SERIES_TO_TEST = ['solar']  # Fokus auf Solar (wo LSTM am schwächsten war)
RANDOM_SEED = 42
QUICK_MODE = True  # Schnellere Tests
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

results_dir = PROJECT_DIR / 'results'
figures_dir = results_dir / 'figures'
metrics_dir = results_dir / 'metrics'

figures_dir.mkdir(parents=True, exist_ok=True)
metrics_dir.mkdir(parents=True, exist_ok=True)


def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def load_processed_data(series_name: str):
    """Lädt vorverarbeitete Train/Val/Test Daten"""
    data_dir = PROJECT_DIR / 'data' / 'processed'
    
    train_df = pd.read_csv(data_dir / f'{series_name}_train.csv')
    val_df = pd.read_csv(data_dir / f'{series_name}_val.csv')
    test_df = pd.read_csv(data_dir / f'{series_name}_test.csv')
    
    return train_df, val_df, test_df


def create_sequences(data, target, seq_length):
    """Erstellt Sequenzen für LSTM"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)


def build_standard_lstm(input_shape, units=64, dropout=0.2, learning_rate=0.001):
    """Standard LSTM"""
    model = keras.Sequential([
        layers.LSTM(units, return_sequences=True, input_shape=input_shape),
        layers.Dropout(dropout),
        layers.LSTM(units // 2),
        layers.Dropout(dropout),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


def build_bidirectional_lstm(input_shape, units=64, dropout=0.2, learning_rate=0.001):
    """Bi-Directional LSTM"""
    model = keras.Sequential([
        layers.Bidirectional(layers.LSTM(units, return_sequences=True), input_shape=input_shape),
        layers.Dropout(dropout),
        layers.Bidirectional(layers.LSTM(units // 2)),
        layers.Dropout(dropout),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


def build_stacked_lstm(input_shape, units=64, dropout=0.2, learning_rate=0.001):
    """Stacked LSTM (3 Layers)"""
    model = keras.Sequential([
        layers.LSTM(units, return_sequences=True, input_shape=input_shape),
        layers.Dropout(dropout),
        layers.LSTM(units, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(units // 2),
        layers.Dropout(dropout),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


def build_lstm_with_attention(input_shape, units=64, dropout=0.2, learning_rate=0.001):
    """LSTM mit Attention Mechanismus"""
    inputs = layers.Input(shape=input_shape)
    
    # LSTM Layer
    lstm_out = layers.LSTM(units, return_sequences=True)(inputs)
    lstm_out = layers.Dropout(dropout)(lstm_out)
    
    # Attention Mechanism
    attention = layers.Dense(1, activation='tanh')(lstm_out)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(units)(attention)
    attention = layers.Permute([2, 1])(attention)
    
    # Apply attention
    sent_representation = layers.Multiply()([lstm_out, attention])
    sent_representation = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(sent_representation)
    
    # Output
    output = layers.Dense(32, activation='relu')(sent_representation)
    output = layers.Dropout(dropout)(output)
    output = layers.Dense(1)(output)
    
    model = Model(inputs=inputs, outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


def evaluate_model(y_true, y_pred, model_name):
    """Berechnet Metriken"""
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


def train_and_evaluate(series_name, model_builder, model_name, seq_length, 
                       epochs, units, dropout, learning_rate):
    """Trainiert und evaluiert ein Modell"""
    print(f"\n  Testing: {model_name} | seq={seq_length} | epochs={epochs} | units={units}")
    
    # Daten laden
    train_df, val_df, test_df = load_processed_data(series_name)
    
    # Ermittle Value-Spalte (kann 'solar', 'price', etc. heißen)
    value_col = [c for c in train_df.columns if c in ['solar', 'price', 'value', 'wind_offshore', 
                                                        'wind_onshore', 'consumption']][0]
    
    feature_cols = [c for c in train_df.columns if c not in ['timestamp', value_col]]
    
    # Skalierung
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train = scaler_X.fit_transform(train_df[feature_cols])
    y_train = scaler_y.fit_transform(train_df[[value_col]])
    
    X_val = scaler_X.transform(val_df[feature_cols])
    y_val = scaler_y.transform(val_df[[value_col]])
    
    X_test = scaler_X.transform(test_df[feature_cols])
    y_test_orig = test_df[value_col].values
    
    # Sequenzen erstellen
    X_train_seq, y_train_seq = create_sequences(X_train, y_train.flatten(), seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val.flatten(), seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test, np.zeros(len(X_test)), seq_length)
    y_test_seq = y_test_orig[seq_length:]  # Anpassung für Sequence Offset
    
    # Modell bauen
    input_shape = (seq_length, X_train.shape[1])
    model = model_builder(input_shape, units=units, dropout=dropout, learning_rate=learning_rate)
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    
    # Training
    start = time.time()
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    train_time = time.time() - start
    
    # Vorhersage
    y_pred_scaled = model.predict(X_test_seq, verbose=0).flatten()
    
    # Rückskalierung
    y_pred_scaled_2d = y_pred_scaled.reshape(-1, 1)
    y_pred = scaler_y.inverse_transform(y_pred_scaled_2d).flatten()
    
    # Evaluation
    metrics = evaluate_model(y_test_seq, y_pred, model_name)
    metrics['Sequence_Length'] = seq_length
    metrics['Epochs_Actual'] = len(history.history['loss'])
    metrics['Epochs_Requested'] = epochs
    metrics['Units'] = units
    metrics['Dropout'] = dropout
    metrics['Learning_Rate'] = learning_rate
    metrics['Train_Time_Sec'] = train_time
    
    print(f"    ✅ R²={metrics['R²']:.4f} | RMSE={metrics['RMSE']:.2f} | Time={train_time:.1f}s")
    
    return metrics, history


# =============================================================================
# MAIN OPTIMIZATION LOOP
# =============================================================================
print_section("LSTM OPTIMIZATION - SYSTEMATIC TUNING")

all_results = []

for series_name in SERIES_TO_TEST:
    print_section(f"SERIES: {series_name.upper()}")
    
    # ==========================================================================
    # EXPERIMENT 1: Baseline (Standard LSTM mit Default-Parametern)
    # ==========================================================================
    print("\n[1/4] BASELINE - Standard LSTM")
    baseline_metrics, _ = train_and_evaluate(
        series_name=series_name,
        model_builder=build_standard_lstm,
        model_name='LSTM_Baseline',
        seq_length=24,
        epochs=30,
        units=64,
        dropout=0.2,
        learning_rate=0.001
    )
    baseline_metrics['Experiment'] = 'Baseline'
    baseline_metrics['Series'] = series_name
    all_results.append(baseline_metrics)
    
    # ==========================================================================
    # EXPERIMENT 2: Sequence Length Variation
    # ==========================================================================
    print("\n[2/4] SEQUENCE LENGTH EXPERIMENTS")
    seq_lengths = [48, 168] if QUICK_MODE else [48, 72, 168]
    for seq_len in seq_lengths:
        metrics, _ = train_and_evaluate(
            series_name=series_name,
            model_builder=build_standard_lstm,
            model_name=f'LSTM_Seq{seq_len}',
            seq_length=seq_len,
            epochs=30,
            units=64,
            dropout=0.2,
            learning_rate=0.001
        )
        metrics['Experiment'] = 'Sequence_Length'
        metrics['Series'] = series_name
        all_results.append(metrics)
    
    # ==========================================================================
    # EXPERIMENT 3: More Epochs
    # ==========================================================================
    print("\n[3/4] MORE EPOCHS EXPERIMENTS")
    epoch_counts = [50] if QUICK_MODE else [50, 100]
    for n_epochs in epoch_counts:
        metrics, _ = train_and_evaluate(
            series_name=series_name,
            model_builder=build_standard_lstm,
            model_name=f'LSTM_{n_epochs}Epochs',
            seq_length=24,
            epochs=n_epochs,
            units=64,
            dropout=0.2,
            learning_rate=0.001
        )
        metrics['Experiment'] = 'More_Epochs'
        metrics['Series'] = series_name
        all_results.append(metrics)
    
    # ==========================================================================
    # EXPERIMENT 4: Architecture Variants
    # ==========================================================================
    print("\n[4/4] ARCHITECTURE VARIANTS")
    
    # Bi-LSTM
    metrics, _ = train_and_evaluate(
        series_name=series_name,
        model_builder=build_bidirectional_lstm,
        model_name='Bi-LSTM',
        seq_length=24,
        epochs=50,
        units=64,
        dropout=0.2,
        learning_rate=0.001
    )
    metrics['Experiment'] = 'Architecture'
    metrics['Series'] = series_name
    all_results.append(metrics)
    
    # Stacked LSTM
    metrics, _ = train_and_evaluate(
        series_name=series_name,
        model_builder=build_stacked_lstm,
        model_name='Stacked_LSTM',
        seq_length=24,
        epochs=50,
        units=64,
        dropout=0.2,
        learning_rate=0.001
    )
    metrics['Experiment'] = 'Architecture'
    metrics['Series'] = series_name
    all_results.append(metrics)
    
    # LSTM with Attention
    metrics, _ = train_and_evaluate(
        series_name=series_name,
        model_builder=build_lstm_with_attention,
        model_name='LSTM_Attention',
        seq_length=24,
        epochs=50,
        units=64,
        dropout=0.2,
        learning_rate=0.001
    )
    metrics['Experiment'] = 'Architecture'
    metrics['Series'] = series_name
    all_results.append(metrics)
    
    # ==========================================================================
    # EXPERIMENT 5: Hyperparameter Tuning (Best Config)
    # ==========================================================================
    print("\n[5/5] HYPERPARAMETER TUNING")
    
    # Grid für schnelles Tuning
    if QUICK_MODE:
        param_grid = [
            {'units': 128, 'dropout': 0.3, 'lr': 0.0005},
            {'units': 96, 'dropout': 0.25, 'lr': 0.001}
        ]
    else:
        param_grid = [
            {'units': 128, 'dropout': 0.3, 'lr': 0.0005},
            {'units': 96, 'dropout': 0.25, 'lr': 0.001},
            {'units': 64, 'dropout': 0.15, 'lr': 0.002}
        ]
    
    for params in param_grid:
        metrics, _ = train_and_evaluate(
            series_name=series_name,
            model_builder=build_bidirectional_lstm,  # Nutze Bi-LSTM als Base
            model_name=f"BiLSTM_u{params['units']}_d{params['dropout']}_lr{params['lr']}",
            seq_length=48,  # Längere Sequenzen
            epochs=50,
            units=params['units'],
            dropout=params['dropout'],
            learning_rate=params['lr']
        )
        metrics['Experiment'] = 'Hyperparameter_Tuning'
        metrics['Series'] = series_name
        all_results.append(metrics)


# =============================================================================
# RESULTS ANALYSIS
# =============================================================================
print_section("OPTIMIZATION RESULTS")

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(['Series', 'R²'], ascending=[True, False])

# Speichern
results_df.to_csv(metrics_dir / 'lstm_optimization_results.csv', index=False)
print(f"✅ Results saved to: lstm_optimization_results.csv")

# Beste Modelle pro Serie
print("\n🏆 BEST MODELS PER SERIES:")
for series in SERIES_TO_TEST:
    series_results = results_df[results_df['Series'] == series]
    best = series_results.iloc[0]
    baseline = series_results[series_results['Experiment'] == 'Baseline'].iloc[0]
    
    improvement = ((best['R²'] - baseline['R²']) / abs(baseline['R²'])) * 100
    
    print(f"\n{series.upper()}:")
    print(f"  Baseline:     R²={baseline['R²']:.4f} | RMSE={baseline['RMSE']:.2f}")
    print(f"  Best Model:   {best['Model']}")
    print(f"                R²={best['R²']:.4f} | RMSE={best['RMSE']:.2f}")
    print(f"  Improvement:  {improvement:+.2f}% R²")
    print(f"  Config:       seq={int(best['Sequence_Length'])}, epochs={int(best['Epochs_Actual'])}, " +
          f"units={int(best['Units'])}, dropout={best['Dropout']:.2f}, lr={best['Learning_Rate']:.4f}")

# Visualisierung
print("\n📊 Creating visualizations...")

for series in SERIES_TO_TEST:
    series_results = results_df[results_df['Series'] == series]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'LSTM Optimization Results - {series.upper()}', fontsize=16, fontweight='bold')
    
    # 1. R² by Experiment
    ax = axes[0, 0]
    exp_order = ['Baseline', 'Sequence_Length', 'More_Epochs', 'Architecture', 'Hyperparameter_Tuning']
    series_results['Experiment'] = pd.Categorical(series_results['Experiment'], categories=exp_order, ordered=True)
    series_results_sorted = series_results.sort_values('Experiment')
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(series_results_sorted)))
    ax.barh(range(len(series_results_sorted)), series_results_sorted['R²'], color=colors)
    ax.set_yticks(range(len(series_results_sorted)))
    ax.set_yticklabels(series_results_sorted['Model'], fontsize=8)
    ax.set_xlabel('R² Score')
    ax.set_title('R² Score by Model')
    ax.grid(alpha=0.3, axis='x')
    
    # 2. RMSE by Experiment
    ax = axes[0, 1]
    ax.barh(range(len(series_results_sorted)), series_results_sorted['RMSE'], color=colors)
    ax.set_yticks(range(len(series_results_sorted)))
    ax.set_yticklabels(series_results_sorted['Model'], fontsize=8)
    ax.set_xlabel('RMSE')
    ax.set_title('RMSE by Model')
    ax.grid(alpha=0.3, axis='x')
    
    # 3. R² vs Training Time
    ax = axes[1, 0]
    scatter = ax.scatter(series_results['Train_Time_Sec'], series_results['R²'], 
                        c=series_results['Sequence_Length'], cmap='coolwarm', s=100, alpha=0.7)
    ax.set_xlabel('Training Time (seconds)')
    ax.set_ylabel('R² Score')
    ax.set_title('Performance vs Training Time')
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Sequence Length')
    
    # 4. Experiment Comparison
    ax = axes[1, 1]
    exp_stats = series_results.groupby('Experiment')['R²'].agg(['mean', 'max'])
    exp_stats = exp_stats.reindex(exp_order)
    
    x = np.arange(len(exp_stats))
    width = 0.35
    ax.bar(x - width/2, exp_stats['mean'], width, label='Mean R²', alpha=0.7)
    ax.bar(x + width/2, exp_stats['max'], width, label='Max R²', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_stats.index, rotation=45, ha='right')
    ax.set_ylabel('R² Score')
    ax.set_title('Mean vs Max R² by Experiment Type')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(figures_dir / f'lstm_optimization_{series}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: lstm_optimization_{series}.png")

print_section("✨ LSTM OPTIMIZATION COMPLETE!")
print(f"\n📁 Results: {metrics_dir / 'lstm_optimization_results.csv'}")
print(f"📊 Plots: {figures_dir / 'lstm_optimization_*.png'}")
