"""
Ensemble Methods für Solar Power Forecasting
============================================

Kombiniert die besten Modelle:
- XGBoost (Tuned)
- LSTM
- Chronos-T5-Small

Ensemble Strategien:
1. Simple Average
2. Weighted Average (performance-based)
3. Optimized Weights (grid search)
4. Stacking Meta-Learner
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import joblib
import torch
import torch.nn as nn
import xgboost as xgb
from chronos import ChronosPipeline
import matplotlib.pyplot as plt
import seaborn as sns


class LSTMModel(nn.Module):
    """LSTM Model Architecture"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


def load_data():
    """Lade Test-Daten"""
    print("📊 Lade Daten...")
    
    X_test = pd.read_csv('data/processed/solar_test.csv', index_col=0, parse_dates=True)
    y_test = X_test['generation_solar']
    X_test_features = X_test.drop('generation_solar', axis=1)
    
    X_test_scaled = pd.read_csv('data/processed/solar_test_scaled.csv', index_col=0, parse_dates=True)
    X_test_scaled_features = X_test_scaled.drop('generation_solar', axis=1)
    
    print(f"✅ Test Samples: {len(X_test)}")
    return X_test_features, y_test, X_test_scaled_features


def get_xgboost_predictions(X_test, y_test):
    """XGBoost Predictions"""
    print("\n🌲 XGBoost Predictions...")
    
    model = xgb.XGBRegressor()
    model.load_model('results/models/xgboost_tuned_solar.json')
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"   MAE: {mae:.2f} MW, R²: {r2:.4f}")
    return predictions, mae, r2


def get_lstm_predictions(X_test_scaled, y_test):
    """LSTM Predictions"""
    print("\n🧠 LSTM Predictions...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model laden
    model = LSTMModel(input_size=X_test_scaled.shape[1]).to(device)
    model.load_state_dict(torch.load('results/models/lstm_solar_best.pth', map_location=device))
    model.eval()
    
    # Scaler laden
    scaler_y = joblib.load('results/models/scaler_y_solar.pkl')
    
    # Sequenzen erstellen
    sequence_length = 24
    X_sequences = []
    y_sequences = []
    
    for i in range(sequence_length, len(X_test_scaled)):
        X_sequences.append(X_test_scaled.iloc[i-sequence_length:i].values)
        y_sequences.append(y_test.iloc[i])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    # Predictions
    X_tensor = torch.FloatTensor(X_sequences).to(device)
    
    with torch.no_grad():
        pred_scaled = model(X_tensor).cpu().numpy().flatten()
        predictions = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    
    mae = mean_absolute_error(y_sequences, predictions)
    r2 = r2_score(y_sequences, predictions)
    
    print(f"   MAE: {mae:.2f} MW, R²: {r2:.4f}")
    return predictions, mae, r2, y_sequences


def get_chronos_predictions(y_test):
    """Chronos Predictions"""
    print("\n🤖 Chronos Predictions...")
    
    # Model laden
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    # Historische Daten
    train_data = pd.read_csv('data/processed/solar_train.csv', index_col=0, parse_dates=True)
    val_data = pd.read_csv('data/processed/solar_val.csv', index_col=0, parse_dates=True)
    historical_data = pd.concat([train_data, val_data])
    historical_series = torch.tensor(historical_data['generation_solar'].values)
    
    # Forecast
    forecast = pipeline.predict(
        context=historical_series,
        prediction_length=len(y_test),
        num_samples=20
    )
    
    predictions = forecast[0].median(dim=0).values.numpy()
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"   MAE: {mae:.2f} MW, R²: {r2:.4f}")
    return predictions, mae, r2


def align_predictions(xgb_pred, lstm_pred, chronos_pred, y_test):
    """Align alle Predictions auf gleiche Länge"""
    min_len = min(len(xgb_pred), len(lstm_pred), len(chronos_pred))
    
    return (xgb_pred[-min_len:], 
            lstm_pred[-min_len:], 
            chronos_pred[-min_len:], 
            y_test.values[-min_len:])


def simple_average_ensemble(xgb_pred, lstm_pred, chronos_pred, y_test):
    """Simple Average Ensemble"""
    ensemble = (xgb_pred + lstm_pred + chronos_pred) / 3
    mae = mean_absolute_error(y_test, ensemble)
    r2 = r2_score(y_test, ensemble)
    
    return ensemble, mae, r2


def weighted_average_ensemble(xgb_pred, lstm_pred, chronos_pred, y_test, 
                              xgb_r2, lstm_r2, chronos_r2):
    """Weighted Average basierend auf R² Scores"""
    total_r2 = xgb_r2 + lstm_r2 + chronos_r2
    w_xgb = xgb_r2 / total_r2
    w_lstm = lstm_r2 / total_r2
    w_chronos = chronos_r2 / total_r2
    
    ensemble = (w_xgb * xgb_pred + w_lstm * lstm_pred + w_chronos * chronos_pred)
    mae = mean_absolute_error(y_test, ensemble)
    r2 = r2_score(y_test, ensemble)
    
    return ensemble, mae, r2, (w_xgb, w_lstm, w_chronos)


def optimized_weights_ensemble(xgb_pred, lstm_pred, chronos_pred, y_test):
    """Grid Search für optimale Gewichte"""
    print("\n🔍 Suche optimale Gewichte...")
    
    best_mae = float('inf')
    best_weights = None
    
    for w1 in np.arange(0, 1.1, 0.1):
        for w2 in np.arange(0, 1.1 - w1, 0.1):
            w3 = 1.0 - w1 - w2
            if w3 < 0 or w3 > 1:
                continue
                
            ensemble = w1 * xgb_pred + w2 * lstm_pred + w3 * chronos_pred
            mae = mean_absolute_error(y_test, ensemble)
            
            if mae < best_mae:
                best_mae = mae
                best_weights = (w1, w2, w3)
    
    ensemble = (best_weights[0] * xgb_pred + 
                best_weights[1] * lstm_pred + 
                best_weights[2] * chronos_pred)
    
    r2 = r2_score(y_test, ensemble)
    
    return ensemble, best_mae, r2, best_weights


def stacking_ensemble(xgb_pred, lstm_pred, chronos_pred, y_test):
    """Stacking mit Meta-Learner"""
    print("\n🎯 Training Stacking Meta-Learner...")
    
    # Split für Meta-Learner
    meta_split = len(xgb_pred) // 2
    
    X_meta_train = np.column_stack([
        xgb_pred[:meta_split],
        lstm_pred[:meta_split],
        chronos_pred[:meta_split]
    ])
    y_meta_train = y_test[:meta_split]
    
    X_meta_test = np.column_stack([
        xgb_pred[meta_split:],
        lstm_pred[meta_split:],
        chronos_pred[meta_split:]
    ])
    y_meta_test = y_test[meta_split:]
    
    # Ridge Regression
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_meta_train, y_meta_train)
    
    ensemble = meta_model.predict(X_meta_test)
    mae = mean_absolute_error(y_meta_test, ensemble)
    r2 = r2_score(y_meta_test, ensemble)
    
    return ensemble, mae, r2, meta_model.coef_, y_meta_test


def create_visualizations(results, xgb_pred, lstm_pred, ensemble_opt, y_test):
    """Erstelle Visualisierungen"""
    print("\n📊 Erstelle Visualisierungen...")
    
    # Performance Vergleich
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].barh(results['Model'], results['MAE'], color='steelblue')
    axes[0].set_xlabel('MAE (MW)', fontsize=12)
    axes[0].set_title('Mean Absolute Error - Ensemble Comparison', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    axes[1].barh(results['Model'], results['R²'], color='coral')
    axes[1].set_xlabel('R² Score', fontsize=12)
    axes[1].set_title('R² Score - Ensemble Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xlim([0.95, 1.0])
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/ensemble_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✅ Performance Vergleich gespeichert")
    
    # Zeitreihen Vergleich
    days_to_plot = 7 * 24
    plot_start = -min(days_to_plot, len(y_test))
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    time_index = range(len(y_test[plot_start:]))
    
    ax.plot(time_index, y_test[plot_start:], label='Actual', linewidth=2, color='black', alpha=0.7)
    ax.plot(time_index, xgb_pred[plot_start:], label='XGBoost', linewidth=1.5, alpha=0.7)
    ax.plot(time_index, lstm_pred[plot_start:], label='LSTM', linewidth=1.5, alpha=0.7)
    ax.plot(time_index, ensemble_opt[plot_start:], label='Ensemble (Optimized)', 
            linewidth=2, color='red', linestyle='--')
    
    ax.set_xlabel('Hours', fontsize=12)
    ax.set_ylabel('Solar Power (MW)', fontsize=12)
    ax.set_title('Ensemble vs Single Models - Last 7 Days', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/ensemble_timeseries_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✅ Zeitreihen Vergleich gespeichert")


def main():
    """Main Ensemble Pipeline"""
    print("="*80)
    print("ENSEMBLE METHODS FOR SOLAR POWER FORECASTING")
    print("="*80)
    
    # 1. Daten laden
    X_test, y_test, X_test_scaled = load_data()
    
    # 2. Basis-Model Predictions
    xgb_pred, xgb_mae, xgb_r2 = get_xgboost_predictions(X_test, y_test)
    lstm_pred, lstm_mae, lstm_r2, y_lstm = get_lstm_predictions(X_test_scaled, y_test)
    chronos_pred, chronos_mae, chronos_r2 = get_chronos_predictions(y_test)
    
    # 3. Predictions alignen
    xgb_aligned, lstm_aligned, chronos_aligned, y_aligned = align_predictions(
        xgb_pred, lstm_pred, chronos_pred, y_test
    )
    
    # 4. Ensemble Methods
    print("\n" + "="*80)
    print("ENSEMBLE STRATEGIES")
    print("="*80)
    
    # Simple Average
    _, simple_mae, simple_r2 = simple_average_ensemble(
        xgb_aligned, lstm_aligned, chronos_aligned, y_aligned
    )
    print(f"\n✅ Simple Average: MAE={simple_mae:.2f} MW, R²={simple_r2:.4f}")
    
    # Weighted Average
    _, weighted_mae, weighted_r2, weights = weighted_average_ensemble(
        xgb_aligned, lstm_aligned, chronos_aligned, y_aligned,
        xgb_r2, lstm_r2, chronos_r2
    )
    print(f"✅ Weighted Average: MAE={weighted_mae:.2f} MW, R²={weighted_r2:.4f}")
    print(f"   Weights: XGBoost={weights[0]:.3f}, LSTM={weights[1]:.3f}, Chronos={weights[2]:.3f}")
    
    # Optimized Weights
    ensemble_opt, opt_mae, opt_r2, best_weights = optimized_weights_ensemble(
        xgb_aligned, lstm_aligned, chronos_aligned, y_aligned
    )
    print(f"✅ Optimized Weights: MAE={opt_mae:.2f} MW, R²={opt_r2:.4f}")
    print(f"   Best Weights: XGBoost={best_weights[0]:.2f}, LSTM={best_weights[1]:.2f}, Chronos={best_weights[2]:.2f}")
    
    # Stacking
    _, stack_mae, stack_r2, coefs, _ = stacking_ensemble(
        xgb_aligned, lstm_aligned, chronos_aligned, y_aligned
    )
    print(f"✅ Stacking Meta-Learner: MAE={stack_mae:.2f} MW, R²={stack_r2:.4f}")
    print(f"   Coefficients: XGBoost={coefs[0]:.3f}, LSTM={coefs[1]:.3f}, Chronos={coefs[2]:.3f}")
    
    # 5. Ergebnisse zusammenfassen
    results = pd.DataFrame([
        {'Model': 'XGBoost (Single)', 'MAE': xgb_mae, 'R²': xgb_r2},
        {'Model': 'LSTM (Single)', 'MAE': lstm_mae, 'R²': lstm_r2},
        {'Model': 'Chronos (Single)', 'MAE': chronos_mae, 'R²': chronos_r2},
        {'Model': 'Simple Average', 'MAE': simple_mae, 'R²': simple_r2},
        {'Model': 'Weighted Average', 'MAE': weighted_mae, 'R²': weighted_r2},
        {'Model': 'Optimized Weights', 'MAE': opt_mae, 'R²': opt_r2},
        {'Model': 'Stacking', 'MAE': stack_mae, 'R²': stack_r2},
    ])
    
    results = results.sort_values('MAE')
    results['MAPE'] = (results['MAE'] / y_test.mean()) * 100
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(results.to_string(index=False))
    print("="*80)
    
    # 6. Speichern
    results.to_csv('results/metrics/ensemble_methods_comparison.csv', index=False)
    print("\n✅ Ergebnisse gespeichert: results/metrics/ensemble_methods_comparison.csv")
    
    # 7. Visualisierungen
    create_visualizations(results, xgb_aligned, lstm_aligned, ensemble_opt, y_aligned)
    
    # 8. Zusammenfassung
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if opt_mae < xgb_mae:
        improvement = ((xgb_mae - opt_mae) / xgb_mae) * 100
        print(f"🎉 Ensemble verbessert XGBoost um {improvement:.2f}%")
        print(f"🏆 Best Model: Optimized Ensemble (MAE={opt_mae:.2f} MW)")
    else:
        print("⚠️ XGBoost bleibt das beste Einzelmodell")
        print(f"🏆 Best Model: XGBoost (MAE={xgb_mae:.2f} MW)")
    
    print("\n✨ Ensemble Methods Analyse abgeschlossen!")


if __name__ == '__main__':
    main()
