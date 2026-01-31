#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WIND OFFSHORE FORECASTING - EXTENDED PIPELINE (9 Phases)
Inhaltlich angelehnt an die 9 Solar-Notebooks:
01 Exploration, 02 Preprocessing, 03 Baselines, 04 Statistical,
05 ML Trees, 06 Deep Learning, 07 Generative, 08 Advanced, 09 Comparison
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR / 'src'))

from data.preprocessing import TimeSeriesPreprocessor, train_test_split_temporal
from evaluation.metrics import calculate_metrics, compare_models
from models.baseline import (
    NaiveForecaster, SeasonalNaiveForecaster,
    MovingAverageForecaster, DriftForecaster, MeanForecaster
)

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Runtime flags (safe defaults for CI/containers)
RUN_AUTO_ARIMA = True
RUN_ADVANCED_MODELS = False  # N-BEATS/N-HiTS überspringen (zu langsam)


def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def evaluate_simple(y_true, y_pred, model_name: str) -> dict:
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


def add_metrics(results_dict, model_name, y_true, y_pred, y_train=None, seasonality=24):
    metrics = calculate_metrics(
        y_true,
        y_pred,
        y_train=y_train,
        seasonality=seasonality,
        prefix='test_'
    )
    results_dict[model_name] = metrics
    return metrics


# =============================================================================
# SETUP
# =============================================================================
print_section("WIND OFFSHORE FORECASTING - EXTENDED PIPELINE (9 PHASES)")

DATA_TYPE = 'wind_offshore'
data_path = PROJECT_DIR / 'data' / 'raw' / 'wind_offshore_2022-01-01_2024-12-31_hour.csv'

proc_dir = PROJECT_DIR / 'data' / 'processed'
results_dir = PROJECT_DIR / 'results'
figures_dir = results_dir / 'figures'
metrics_dir = results_dir / 'metrics'

proc_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)
metrics_dir.mkdir(parents=True, exist_ok=True)

all_results = []


# =============================================================================
# PHASE 1: DATA EXPLORATION
# =============================================================================
print_section("PHASE 1: DATA EXPLORATION")

print("\n[1/3] Loading data...")
df = pd.read_csv(data_path, parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
print(f"✅ Loaded: {df.shape}")

print("\n[2/3] Computing statistics...")
stats = {
    'count': len(df),
    'mean': df['value'].mean(),
    'std': df['value'].std(),
    'min': df['value'].min(),
    'max': df['value'].max(),
    'cv': df['value'].std() / (df['value'].mean() + 1e-8)
}
print(f"✅ Mean={stats['mean']:.2f}, Std={stats['std']:.2f}, CV={stats['cv']:.3f}")

print("\n[3/3] Creating visualizations...")
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(df['timestamp'], df['value'], linewidth=0.5, alpha=0.7)
ax.set_title('Wind Offshore Timeline', fontweight='bold')
ax.set_ylabel('MW')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'wind_offshore_01_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Exploration complete")


# =============================================================================
# PHASE 2: PREPROCESSING & FEATURE ENGINEERING
# =============================================================================
print_section("PHASE 2: PREPROCESSING & FEATURE ENGINEERING")

print("\n[1/4] Handling missing values...")
prep = TimeSeriesPreprocessor()
missing_before = df['value'].isna().sum()
df_clean = prep.handle_missing_values(df, value_col='value', method='interpolate')
missing_after = df_clean['value'].isna().sum()
print(f"✅ Missing before: {missing_before}, after: {missing_after}")

# CRITICAL FIX: Wind Offshore hatte 9-monatigen Stillstand (Apr 2023 - Jan 2024)
# Nutze nur Daten VOR dem Stillstand für saubere Modellierung
print("\n⚠️  WICHTIG: Erkenne Stillstandsperiode...")
zeros = df_clean['value'] == 0
zero_count = zeros.sum()
print(f"   Nullwerte gefunden: {zero_count} ({zero_count/len(df_clean)*100:.1f}%)")
if zero_count > 100:  # Signifikante Stillstandsperiode
    first_zero_idx = df_clean[zeros].index[0]
    cutoff_date = df_clean.loc[first_zero_idx, 'timestamp']
    print(f"   Stillstand ab: {cutoff_date}")
    print(f"   ➡️  Nutze nur Daten VOR Stillstand für Training")
    df_clean = df_clean[df_clean['timestamp'] < cutoff_date].reset_index(drop=True)
    print(f"   Verbleibende Daten: {len(df_clean)}")

print("\n[2/4] Outlier analysis (IQR) - auf bereinigte Daten...")
q1 = df_clean['value'].quantile(0.25)
q3 = df_clean['value'].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
outliers = ((df_clean['value'] < lower) | (df_clean['value'] > upper)).sum()
print(f"✅ Outliers (IQR): {outliers} ({outliers / len(df_clean) * 100:.2f}%)")

print("\n[3/4] Creating time, lag, rolling features...")
df_features = prep.create_time_features(df_clean, timestamp_col='timestamp')
df_features = prep.create_lag_features(
    df_features,
    value_col='value',
    lags=[1, 2, 3, 24, 48, 168]
)
df_features = prep.create_rolling_features(
    df_features,
    value_col='value',
    windows=[24, 168],
    functions=['mean', 'std', 'min', 'max']
)

df_complete = df_features.dropna().reset_index(drop=True)
print(f"✅ Features created. Rows after dropna: {len(df_complete)}")

print("\n[4/4] Train/Val/Test split (chronological)...")
train_df, val_df, test_df = train_test_split_temporal(
    df_complete,
    test_size=0.15,
    val_size=0.15
)
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(train_df['timestamp'], train_df['value'], label='Train', linewidth=0.5)
ax.plot(val_df['timestamp'], val_df['value'], label='Validation', linewidth=0.5)
ax.plot(test_df['timestamp'], test_df['value'], label='Test', linewidth=0.5)
ax.set_title('Train/Validation/Test Split', fontweight='bold')
ax.set_xlabel('Zeit')
ax.set_ylabel('MW')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'wind_offshore_02_split.png', dpi=150, bbox_inches='tight')
plt.close()

feature_cols = [c for c in df_complete.columns if c not in ['timestamp', 'value']]

print("\nScaling features (StandardScaler)...")
scaler_prep = TimeSeriesPreprocessor()
train_scaled, val_scaled = scaler_prep.scale_data(
    train_df,
    val_df,
    feature_cols=feature_cols + ['value'],
    method='standard'
)
_, test_scaled = scaler_prep.scale_data(
    train_df,
    test_df,
    feature_cols=feature_cols + ['value'],
    method='standard'
)

train_df.to_csv(proc_dir / f'{DATA_TYPE}_train.csv', index=False)
val_df.to_csv(proc_dir / f'{DATA_TYPE}_val.csv', index=False)
test_df.to_csv(proc_dir / f'{DATA_TYPE}_test.csv', index=False)

train_scaled.to_csv(proc_dir / f'{DATA_TYPE}_train_scaled.csv', index=False)
val_scaled.to_csv(proc_dir / f'{DATA_TYPE}_val_scaled.csv', index=False)
test_scaled.to_csv(proc_dir / f'{DATA_TYPE}_test_scaled.csv', index=False)
print("✅ Processed datasets saved")


# =============================================================================
# PHASE 3: BASELINE MODELS
# =============================================================================
print_section("PHASE 3: BASELINE MODELS")

y_train = train_df['value'].values
y_val = val_df['value'].values
y_test = test_df['value'].values

baseline_results = {}

models = {
    'Naive': NaiveForecaster(),
    'Seasonal Naive (24h)': SeasonalNaiveForecaster(seasonality=24),
    'Moving Average (168h)': MovingAverageForecaster(window=168),
    'Drift': DriftForecaster(),
    'Mean': MeanForecaster()
}

for name, model in models.items():
    model.fit(y_train)
    pred = model.predict(steps=len(y_test))
    add_metrics(baseline_results, name, y_test, pred, y_train=y_train, seasonality=24)
    print(f"✅ {name} done")

baseline_df = compare_models(baseline_results, sort_by='test_rmse')
baseline_df.to_csv(metrics_dir / f'{DATA_TYPE}_baseline_results.csv')

for name in baseline_results:
    all_results.append({'Category': 'Baseline', 'Model': name, **baseline_results[name]})


# =============================================================================
# PHASE 4: STATISTICAL MODELS
# =============================================================================
print_section("PHASE 4: STATISTICAL MODELS")

stat_results = {}

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from pmdarima import auto_arima

    y_train_series = train_df.set_index('timestamp')['value']
    y_test_series = test_df.set_index('timestamp')['value']

    if RUN_AUTO_ARIMA:
        print("\n[1/3] Auto-ARIMA parameter search (sampled)...")
        sample_size = min(1000, len(y_train_series))
        y_train_sample = y_train_series[-sample_size:]
        auto_model = auto_arima(
            y_train_sample,
            seasonal=True,
            m=24,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            n_fits=10
        )
        best_order = auto_model.order
        best_seasonal_order = auto_model.seasonal_order
        print(f"✅ Best ARIMA: {best_order}, Seasonal: {best_seasonal_order}")
    else:
        print("\n[1/3] Auto-ARIMA übersprungen (RUN_AUTO_ARIMA=False).")
        best_order = (1, 0, 1)
        best_seasonal_order = (1, 0, 1, 24)

    print("\n[2/3] SARIMA...")
    sarima_model = SARIMAX(
        y_train_series,
        order=best_order,
        seasonal_order=best_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sarima_fit = sarima_model.fit(disp=False)
    sarima_pred = sarima_fit.forecast(steps=len(y_test_series))
    add_metrics(stat_results, 'SARIMA', y_test_series.values, sarima_pred.values, y_train=y_train)
    print("✅ SARIMA done")

    print("\n[3/3] ARIMA + ETS...")
    arima_model = ARIMA(y_train_series, order=best_order)
    arima_fit = arima_model.fit()
    arima_pred = arima_fit.forecast(steps=len(y_test_series))
    add_metrics(stat_results, 'ARIMA', y_test_series.values, arima_pred.values, y_train=y_train)

    ets_model = ExponentialSmoothing(
        y_train_series,
        trend='add',
        seasonal='add',
        seasonal_periods=24
    )
    ets_fit = ets_model.fit()
    ets_pred = ets_fit.forecast(steps=len(y_test_series))
    add_metrics(stat_results, 'ETS', y_test_series.values, ets_pred.values, y_train=y_train)
    print("✅ ARIMA + ETS done")

except Exception as e:
    print(f"⚠️ Statistical models skipped: {e}")

if stat_results:
    stat_df = compare_models(stat_results, sort_by='test_rmse')
    stat_df.to_csv(metrics_dir / f'{DATA_TYPE}_statistical_results.csv')
    for name in stat_results:
        all_results.append({'Category': 'Statistical', 'Model': name, **stat_results[name]})


# =============================================================================
# PHASE 5: ML TREE MODELS
# =============================================================================
print_section("PHASE 5: MACHINE LEARNING (TREE MODELS)")

ml_results = {}

X_train = train_df[feature_cols].values
X_val = val_df[feature_cols].values
X_test = test_df[feature_cols].values

try:
    from sklearn.ensemble import RandomForestRegressor

    print("\n[1/4] Random Forest...")
    start = time.time()
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    add_metrics(ml_results, 'Random Forest', y_test, rf_pred, y_train=y_train)
    print(f"✅ RF done ({time.time() - start:.1f}s)")

    print("\n[2/4] XGBoost...")
    import xgboost as xgb
    start = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_pred = xgb_model.predict(X_test)
    add_metrics(ml_results, 'XGBoost', y_test, xgb_pred, y_train=y_train)
    print(f"✅ XGBoost done ({time.time() - start:.1f}s)")

    print("\n[3/4] LightGBM...")
    import lightgbm as lgb
    start = time.time()
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'verbose': -1,
        'random_state': 42
    }
    lgb_model = lgb.train(
        params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    lgb_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    add_metrics(ml_results, 'LightGBM', y_test, lgb_pred, y_train=y_train)
    print(f"✅ LightGBM done ({time.time() - start:.1f}s)")

    print("\n[4/4] CatBoost...")
    from catboost import CatBoostRegressor
    start = time.time()
    cat_model = CatBoostRegressor(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        loss_function='RMSE',
        random_seed=42,
        verbose=False
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    cat_pred = cat_model.predict(X_test)
    add_metrics(ml_results, 'CatBoost', y_test, cat_pred, y_train=y_train)
    print(f"✅ CatBoost done ({time.time() - start:.1f}s)")

except Exception as e:
    print(f"⚠️ ML Tree models skipped: {e}")

if ml_results:
    ml_df = compare_models(ml_results, sort_by='test_rmse')
    ml_df.to_csv(metrics_dir / f'{DATA_TYPE}_ml_tree_results.csv')
    for name in ml_results:
        all_results.append({'Category': 'ML Trees', 'Model': name, **ml_results[name]})


# =============================================================================
# PHASE 6: DEEP LEARNING MODELS
# =============================================================================
print_section("PHASE 6: DEEP LEARNING (RNN/LSTM/GRU)")

deep_results = {}

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.preprocessing import StandardScaler

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")

    SEQ_LENGTH = 24

    def create_sequences(values, seq_length):
        X, y = [], []
        for i in range(len(values) - seq_length):
            X.append(values[i:i + seq_length])
            y.append(values[i + seq_length])
        return np.array(X), np.array(y)

    train_vals = train_scaled['value'].values
    val_vals = val_scaled['value'].values
    test_vals = test_scaled['value'].values

    X_train_seq, y_train_seq = create_sequences(train_vals, SEQ_LENGTH)
    X_val_seq, y_val_seq = create_sequences(val_vals, SEQ_LENGTH)
    X_test_seq, y_test_seq = create_sequences(test_vals, SEQ_LENGTH)

    X_train_seq = X_train_seq.reshape(-1, SEQ_LENGTH, 1)
    X_val_seq = X_val_seq.reshape(-1, SEQ_LENGTH, 1)
    X_test_seq = X_test_seq.reshape(-1, SEQ_LENGTH, 1)

    class TimeSeriesDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_loader = DataLoader(TimeSeriesDataset(X_train_seq, y_train_seq), batch_size=64, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val_seq, y_val_seq), batch_size=64, shuffle=False)
    test_loader = DataLoader(TimeSeriesDataset(X_test_seq, y_test_seq), batch_size=64, shuffle=False)

    class RNNRegressor(nn.Module):
        def __init__(self, rnn_type='LSTM', hidden_size=64, num_layers=1, dropout=0.1):
            super().__init__()
            if rnn_type == 'GRU':
                self.rnn = nn.GRU(1, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
            elif rnn_type == 'RNN':
                self.rnn = nn.RNN(1, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
            else:
                self.rnn = nn.LSTM(1, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.rnn(x)
            out = out[:, -1, :]
            return self.fc(out).squeeze(-1)

    def train_model(model, epochs=6):
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
        return model

    def predict_model(model):
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                out = model(xb).detach().cpu().numpy()
                preds.append(out)
        return np.concatenate(preds)

    scaler = StandardScaler()
    scaler.fit(train_df[['value']])
    y_test_true = test_df['value'].values[SEQ_LENGTH:]

    print("\n[1/3] Simple RNN...")
    rnn_model = train_model(RNNRegressor(rnn_type='RNN', hidden_size=48))
    rnn_pred_scaled = predict_model(rnn_model)
    rnn_pred = scaler.inverse_transform(rnn_pred_scaled.reshape(-1, 1)).flatten()
    add_metrics(deep_results, 'Simple RNN', y_test_true, rnn_pred, y_train=y_train)
    print("✅ Simple RNN done")

    print("\n[2/3] LSTM...")
    lstm_model = train_model(RNNRegressor(rnn_type='LSTM', hidden_size=64))
    lstm_pred_scaled = predict_model(lstm_model)
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()
    add_metrics(deep_results, 'LSTM', y_test_true, lstm_pred, y_train=y_train)
    print("✅ LSTM done")

    print("\n[3/3] GRU...")
    gru_model = train_model(RNNRegressor(rnn_type='GRU', hidden_size=64))
    gru_pred_scaled = predict_model(gru_model)
    gru_pred = scaler.inverse_transform(gru_pred_scaled.reshape(-1, 1)).flatten()
    add_metrics(deep_results, 'GRU', y_test_true, gru_pred, y_train=y_train)
    print("✅ GRU done")

except Exception as e:
    print(f"⚠️ Deep Learning skipped: {e}")

if deep_results:
    deep_df = compare_models(deep_results, sort_by='test_rmse')
    deep_df.to_csv(metrics_dir / f'{DATA_TYPE}_deep_learning_results.csv')
    for name in deep_results:
        all_results.append({'Category': 'Deep Learning', 'Model': name, **deep_results[name]})


# =============================================================================
# PHASE 7: GENERATIVE MODELS
# =============================================================================
print_section("PHASE 7: GENERATIVE MODELS (Overview)")
print("Generative Modelle (Autoencoder, VAE, GAN, DeepAR) sind in Notebooks enthalten.")
print("Für die automatisierte Pipeline werden sie aus Laufzeitgründen übersprungen.")


# =============================================================================
# PHASE 8: ADVANCED MODELS
# =============================================================================
print_section("PHASE 8: ADVANCED MODELS (N-BEATS, N-HiTS, TFT)")

advanced_results = {}

if RUN_ADVANCED_MODELS:
    try:
        from darts import TimeSeries
        from darts.models import NBEATSModel, NHiTSModel
        from darts.dataprocessing.transformers import Scaler

        print("✅ Darts verfügbar - starte N-BEATS/N-HiTS (kurz)...")
        train_series = TimeSeries.from_dataframe(train_df, time_col='timestamp', value_cols='value', freq='H')
        val_series = TimeSeries.from_dataframe(val_df, time_col='timestamp', value_cols='value', freq='H')
        test_series = TimeSeries.from_dataframe(test_df, time_col='timestamp', value_cols='value', freq='H')

        scaler_adv = Scaler()
        train_scaled_series = scaler_adv.fit_transform(train_series)
        val_scaled_series = scaler_adv.transform(val_series)

        print("\n[1/2] N-BEATS...")
        nbeats = NBEATSModel(
            input_chunk_length=24,
            output_chunk_length=24,
            n_epochs=10,
            random_state=42,
            batch_size=64,
            generic_architecture=True,
            pl_trainer_kwargs={"enable_progress_bar": False}
        )
        nbeats.fit(train_scaled_series, val_series=val_scaled_series, verbose=False)
        nbeats_pred = nbeats.predict(n=len(test_series))
        nbeats_pred = scaler_adv.inverse_transform(nbeats_pred).values().flatten()
        add_metrics(advanced_results, 'N-BEATS', y_test, nbeats_pred, y_train=y_train)
        print("✅ N-BEATS done")

        print("\n[2/2] N-HiTS...")
        nhits = NHiTSModel(
            input_chunk_length=24,
            output_chunk_length=24,
            n_epochs=10,
            random_state=42,
            batch_size=64,
            pl_trainer_kwargs={"enable_progress_bar": False}
        )
        nhits.fit(train_scaled_series, val_series=val_scaled_series, verbose=False)
        nhits_pred = nhits.predict(n=len(test_series))
        nhits_pred = scaler_adv.inverse_transform(nhits_pred).values().flatten()
        add_metrics(advanced_results, 'N-HiTS', y_test, nhits_pred, y_train=y_train)
        print("✅ N-HiTS done")

    except Exception as e:
        print(f"⚠️ Advanced models skipped: {e}")
else:
    print("⏭️  Advanced models (N-BEATS/N-HiTS) übersprungen (RUN_ADVANCED_MODELS=False)")
    print("   Für schnelleres Testing fokussieren wir auf Baselines, Statistical & ML Trees.")

if advanced_results:
    advanced_df = compare_models(advanced_results, sort_by='test_rmse')
    advanced_df.to_csv(metrics_dir / f'{DATA_TYPE}_advanced_results.csv')
    for name in advanced_results:
        all_results.append({'Category': 'Advanced', 'Model': name, **advanced_results[name]})


# =============================================================================
# PHASE 9: FINAL COMPARISON
# =============================================================================
print_section("PHASE 9: FINAL MODEL COMPARISON")

if all_results:
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('test_rmse', ascending=True)
    results_df.to_csv(metrics_dir / f'{DATA_TYPE}_all_models_extended.csv', index=False)

    print("\n" + results_df[['Category', 'Model', 'test_rmse', 'test_r2']].to_string(index=False))
    best = results_df.iloc[0]
    print(f"\n🥇 BEST MODEL: {best['Model']} ({best['Category']})")
    print(f"   RMSE = {best['test_rmse']:.2f}")
    print(f"   R²   = {best['test_r2']:.4f}")

    # Visualization
    fig, ax = plt.subplots(figsize=(14, 10))
    y_pos = np.arange(len(results_df))
    rmse_values = results_df['test_rmse'].values
    model_names = results_df['Category'] + " - " + results_df['Model']

    colors = []
    for cat in results_df['Category']:
        if cat == 'Baseline':
            colors.append('lightcoral')
        elif cat == 'Statistical':
            colors.append('lightskyblue')
        elif cat == 'ML Trees':
            colors.append('lightgreen')
        elif cat == 'Deep Learning':
            colors.append('plum')
        else:
            colors.append('gold')

    bars = ax.barh(y_pos, rmse_values, color=colors, edgecolor='black', alpha=0.85)
    bars[0].set_color('darkgreen')
    bars[0].set_alpha(1.0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('RMSE (Test Set)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison - RMSE', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(figures_dir / 'wind_offshore_09_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✅ Comparison chart saved")
else:
    print("⚠️ No results available for comparison.")
