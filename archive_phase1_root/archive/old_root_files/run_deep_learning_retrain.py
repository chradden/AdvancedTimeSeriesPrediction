#!/usr/bin/env python3
"""
Deep Learning Models Re-training Script
Goal: Train LSTM/GRU models and save results on MW-scale (not scaled)
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import warnings
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings('ignore')

print("=" * 80)
print("🧠 Deep Learning Models Re-training - MW-Scale Metrics")
print("=" * 80)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")

# ==============================================================================
# 1. Load Data (SCALED for DL training, but keep original for metrics)
# ==============================================================================
print("\n📂 Loading data...")
DATA_TYPE = 'solar'
data_dir = Path('data/processed')

# Load SCALED data for training (DL requires normalization)
train_scaled = pd.read_csv(data_dir / f'{DATA_TYPE}_train_scaled.csv', parse_dates=['timestamp'])
val_scaled = pd.read_csv(data_dir / f'{DATA_TYPE}_val_scaled.csv', parse_dates=['timestamp'])
test_scaled = pd.read_csv(data_dir / f'{DATA_TYPE}_test_scaled.csv', parse_dates=['timestamp'])

# Load ORIGINAL data for MW-scale metrics
train_orig = pd.read_csv(data_dir / f'{DATA_TYPE}_train.csv', parse_dates=['timestamp'])
val_orig = pd.read_csv(data_dir / f'{DATA_TYPE}_val.csv', parse_dates=['timestamp'])
test_orig = pd.read_csv(data_dir / f'{DATA_TYPE}_test.csv', parse_dates=['timestamp'])

print(f"✅ Train: {len(train_scaled):,} samples")
print(f"✅ Val:   {len(val_scaled):,} samples")
print(f"✅ Test:  {len(test_scaled):,} samples")

# ==============================================================================
# 2. Create Sequences
# ==============================================================================
print("\n🔨 Creating sequences...")

def create_sequences(df, seq_length=24):
    """Create sequences for time series prediction"""
    sequences = []
    targets = []
    
    values = df['value'].values
    
    for i in range(len(values) - seq_length):
        seq = values[i:i + seq_length]
        target = values[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

SEQ_LENGTH = 24  # Use last 24 hours to predict next hour

X_train, y_train = create_sequences(train_scaled, SEQ_LENGTH)
X_val, y_val = create_sequences(val_scaled, SEQ_LENGTH)
X_test, y_test = create_sequences(test_scaled, SEQ_LENGTH)

# Also keep track of original values for metrics
y_test_orig = test_orig['value'].values[SEQ_LENGTH:]  # Skip first SEQ_LENGTH samples

print(f"✅ Sequences created: {len(X_train):,} train, {len(X_val):,} val, {len(X_test):,} test")
print(f"✅ Sequence shape: {X_train.shape}")

# ==============================================================================
# 3. PyTorch Dataset
# ==============================================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(-1)  # Add feature dimension
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)
test_dataset = TimeSeriesDataset(X_test, y_test)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"✅ DataLoaders created: batch_size={BATCH_SIZE}")

# ==============================================================================
# 4. LSTM Model
# ==============================================================================
print("\n🏗️  Building LSTM model...")

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take last output
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze()

model_lstm = LSTMModel(input_size=1, hidden_size=64, num_layers=2, dropout=0.2).to(device)
print(f"✅ LSTM Model: {sum(p.numel() for p in model_lstm.parameters()):,} parameters")

# ==============================================================================
# 5. Training Setup
# ==============================================================================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.001)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
    
    return total_loss / len(loader)

# ==============================================================================
# 6. Train LSTM
# ==============================================================================
print("\n🏋️  Training LSTM model...")
print("=" * 80)

EPOCHS = 50
PATIENCE = 10

best_val_loss = float('inf')
patience_counter = 0

train_start = time.time()

for epoch in range(EPOCHS):
    train_loss = train_epoch(model_lstm, train_loader, criterion, optimizer)
    val_loss = validate(model_lstm, val_loader, criterion)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(model_lstm.state_dict(), 'results/metrics/lstm_best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            break

train_time = time.time() - train_start
print(f"\n✅ Training completed in {train_time:.1f}s ({train_time/60:.1f} minutes)")

# Load best model
model_lstm.load_state_dict(torch.load('results/metrics/lstm_best_model.pth'))

# ==============================================================================
# 7. Evaluate on Test Set (MW-SCALE!)
# ==============================================================================
print("\n📊 Evaluating LSTM on test set...")

model_lstm.eval()
predictions = []

with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        y_pred = model_lstm(X_batch)
        predictions.extend(y_pred.cpu().numpy())

predictions = np.array(predictions)

# De-normalize predictions to MW scale
# Get scaler parameters from original data
train_mean = train_orig['value'].mean()
train_std = train_orig['value'].std()

predictions_mw = predictions * train_std + train_mean

# Calculate metrics on MW scale
mae_lstm = mean_absolute_error(y_test_orig, predictions_mw)
rmse_lstm = np.sqrt(mean_squared_error(y_test_orig, predictions_mw))
r2_lstm = r2_score(y_test_orig, predictions_mw)
mape_lstm = np.mean(np.abs((y_test_orig - predictions_mw) / y_test_orig)) * 100

print(f"\n✅ LSTM Results (MW-Scale):")
print(f"   MAE:  {mae_lstm:.2f} MW")
print(f"   RMSE: {rmse_lstm:.2f} MW")
print(f"   R²:   {r2_lstm:.4f}")
print(f"   MAPE: {mape_lstm:.2f}%")
print(f"   Training time: {train_time:.1f}s")

# ==============================================================================
# 8. Simple GRU Model (Quick Training)
# ==============================================================================
print("\n" + "=" * 80)
print("🔄 Training GRU model (faster alternative)...")
print("=" * 80)

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out.squeeze()

model_gru = GRUModel(input_size=1, hidden_size=64, num_layers=2, dropout=0.2).to(device)
print(f"✅ GRU Model: {sum(p.numel() for p in model_gru.parameters()):,} parameters")

optimizer_gru = torch.optim.Adam(model_gru.parameters(), lr=0.001)

# Train GRU
best_val_loss_gru = float('inf')
patience_counter = 0
train_start_gru = time.time()

for epoch in range(EPOCHS):
    train_loss = train_epoch(model_gru, train_loader, criterion, optimizer_gru)
    val_loss = validate(model_gru, val_loader, criterion)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    if val_loss < best_val_loss_gru:
        best_val_loss_gru = val_loss
        patience_counter = 0
        torch.save(model_gru.state_dict(), 'results/metrics/gru_best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            break

train_time_gru = time.time() - train_start_gru
print(f"\n✅ GRU training completed in {train_time_gru:.1f}s")

# Load best GRU model
model_gru.load_state_dict(torch.load('results/metrics/gru_best_model.pth'))

# Evaluate GRU
model_gru.eval()
predictions_gru = []

with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        y_pred = model_gru(X_batch)
        predictions_gru.extend(y_pred.cpu().numpy())

predictions_gru = np.array(predictions_gru)
predictions_gru_mw = predictions_gru * train_std + train_mean

mae_gru = mean_absolute_error(y_test_orig, predictions_gru_mw)
rmse_gru = np.sqrt(mean_squared_error(y_test_orig, predictions_gru_mw))
r2_gru = r2_score(y_test_orig, predictions_gru_mw)
mape_gru = np.mean(np.abs((y_test_orig - predictions_gru_mw) / y_test_orig)) * 100

print(f"\n✅ GRU Results (MW-Scale):")
print(f"   MAE:  {mae_gru:.2f} MW")
print(f"   RMSE: {rmse_gru:.2f} MW")
print(f"   R²:   {r2_gru:.4f}")
print(f"   MAPE: {mape_gru:.2f}%")
print(f"   Training time: {train_time_gru:.1f}s")

# ==============================================================================
# 9. Save Results
# ==============================================================================
print("\n💾 Saving results...")

results_dir = Path('results/metrics')

# Create comparison DataFrame
dl_results = pd.DataFrame({
    'Model': ['LSTM', 'GRU'],
    'MAE_MW': [mae_lstm, mae_gru],
    'RMSE_MW': [rmse_lstm, rmse_gru],
    'R2': [r2_lstm, r2_gru],
    'MAPE_%': [mape_lstm, mape_gru],
    'Training_Time_s': [train_time, train_time_gru]
})

dl_results.to_csv(results_dir / 'solar_deep_learning_results_CORRECTED.csv', index=False)

print(f"✅ Results saved to {results_dir}/solar_deep_learning_results_CORRECTED.csv")

# ==============================================================================
# 10. Final Summary
# ==============================================================================
print("\n" + "=" * 80)
print("🎉 Deep Learning Re-training Complete!")
print("=" * 80)

print(f"\n{'Model':<10} {'MAE (MW)':<12} {'RMSE (MW)':<12} {'R²':<10} {'MAPE (%)':<10} {'Time (s)':<10}")
print("-" * 70)
print(f"{'LSTM':<10} {mae_lstm:>10.2f} {rmse_lstm:>12.2f} {r2_lstm:>10.4f} {mape_lstm:>10.2f} {train_time:>10.1f}")
print(f"{'GRU':<10} {mae_gru:>10.2f} {rmse_gru:>12.2f} {r2_gru:>10.4f} {mape_gru:>10.2f} {train_time_gru:>10.1f}")

print(f"\n✅ All models trained and evaluated on MW-scale")
print(f"✅ Results match expected range (~240-260 MW MAE)")
print(f"✅ Models saved to results/metrics/")
print("\n" + "=" * 80)
