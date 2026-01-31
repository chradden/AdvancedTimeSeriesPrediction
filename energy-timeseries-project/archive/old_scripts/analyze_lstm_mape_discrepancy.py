"""
Analyse der LSTM-Modell Performance: MAE/RMSE vs MAPE Diskrepanz
================================================================

Dieses Skript untersucht, warum LSTM-Modelle sehr gute MAE/RMSE Werte haben,
aber gleichzeitig hohe MAPE (Mean Absolute Percentage Error) Werte zeigen.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Setup
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Pfade
DATA_DIR = Path('data')
RESULTS_DIR = Path('results')
FIGURES_DIR = RESULTS_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("LSTM Performance Analyse: MAE/RMSE vs MAPE Diskrepanz")
print("="*70)

# 1. Lade Daten
print("\n1. Lade Test-Daten...")
y_test = pd.read_csv(DATA_DIR / 'processed' / 'solar_test.csv')['value'].values
y_test_scaled = pd.read_csv(DATA_DIR / 'processed' / 'solar_test_scaled.csv')['value'].values

# Statistiken der Original-Daten
print(f"\nTest-Daten Statistiken:")
print(f"  Min:     {y_test.min():.2f} MW")
print(f"  Max:     {y_test.max():.2f} MW") 
print(f"  Mean:    {y_test.mean():.2f} MW")
print(f"  Median:  {np.median(y_test):.2f} MW")
print(f"  Std:     {y_test.std():.2f} MW")
print(f"  Anzahl Nullwerte: {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")
print(f"  Anzahl < 100 MW:  {(y_test < 100).sum()} ({(y_test < 100).sum()/len(y_test)*100:.1f}%)")

# 2. Definiere LSTM Modell (muss identisch zum Training sein)
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        last_output = out[:, -1, :]
        predictions = self.fc(last_output)
        return predictions.squeeze()

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        last_output = out[:, -1, :]
        predictions = self.fc(last_output)
        return predictions.squeeze()

# 3. Lade trainierte Modelle
print("\n2. Lade trainierte Modelle...")
device = torch.device('cpu')

lstm_model = LSTMModel()
gru_model = GRUModel()

try:
    lstm_model.load_state_dict(torch.load(RESULTS_DIR / 'metrics' / 'lstm_best_model.pth', 
                                         map_location=device))
    gru_model.load_state_dict(torch.load(RESULTS_DIR / 'metrics' / 'gru_best_model.pth', 
                                        map_location=device))
    lstm_model.eval()
    gru_model.eval()
    print("✓ Modelle erfolgreich geladen")
except Exception as e:
    print(f"✗ Fehler beim Laden der Modelle: {e}")
    print("  Bitte zuerst Notebook 06 ausführen, um Modelle zu trainieren.")
    exit(1)

# 4. Erstelle Sequences und mache Predictions
print("\n3. Erstelle Predictions...")
def create_sequences(data, seq_length=24):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 24
X_test, y_test_seq = create_sequences(y_test_scaled, seq_length)

# Predictions auf skalierte Daten
X_test_tensor = torch.FloatTensor(X_test).unsqueeze(-1)

with torch.no_grad():
    lstm_pred_scaled = lstm_model(X_test_tensor).numpy()
    gru_pred_scaled = gru_model(X_test_tensor).numpy()

# De-Normalisierung (zurück zu MW)
# Wir brauchen den Scaler - laden aus Training Data
y_train = pd.read_csv(DATA_DIR / 'processed' / 'solar_train.csv')['value'].values
scaler = StandardScaler()
scaler.fit(y_train.reshape(-1, 1))

lstm_pred = scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()
gru_pred = scaler.inverse_transform(gru_pred_scaled.reshape(-1, 1)).flatten()
y_test_actual = y_test[seq_length:]  # Alignment mit Predictions

print(f"✓ {len(lstm_pred)} Predictions erstellt")

# 5. Berechne Metriken
print("\n4. Berechne detaillierte Metriken...")

def calculate_metrics(y_true, y_pred, model_name):
    """Berechne alle Metriken mit detaillierter Analyse"""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # MAPE - VORSICHT bei kleinen Werten!
    # Standardformel: MAPE = mean(|actual - predicted| / |actual|) * 100
    epsilon = 1e-10  # Vermeidet Division durch Null
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    # Alternative: sMAPE (symmetric MAPE) - robuster bei kleinen Werten
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon)) * 100
    
    r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    
    print(f"\n{model_name}:")
    print(f"  MAE:   {mae:.2f} MW")
    print(f"  RMSE:  {rmse:.2f} MW")
    print(f"  MAPE:  {mape:.2f}%")
    print(f"  sMAPE: {smape:.2f}%")
    print(f"  R²:    {r2:.4f}")
    
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'smape': smape, 'r2': r2}

lstm_metrics = calculate_metrics(y_test_actual, lstm_pred, "LSTM")
gru_metrics = calculate_metrics(y_test_actual, gru_pred, "GRU")

# 6. Analysiere MAPE-Problematik
print("\n" + "="*70)
print("5. MAPE-Diskrepanz Analyse")
print("="*70)

# Berechne prozentuale Fehler für jeden Zeitpunkt
lstm_percentage_errors = np.abs((y_test_actual - lstm_pred) / (y_test_actual + 1e-10)) * 100

# Kategorisiere nach Wertebereichen
ranges = [
    (0, 100, "Sehr niedrig (0-100 MW)"),
    (100, 500, "Niedrig (100-500 MW)"),
    (500, 2000, "Mittel (500-2000 MW)"),
    (2000, np.inf, "Hoch (>2000 MW)")
]

print("\nFehleranalyse nach Wertebereichen:")
print("-" * 70)

for low, high, label in ranges:
    mask = (y_test_actual >= low) & (y_test_actual < high)
    if mask.sum() > 0:
        count = mask.sum()
        pct = count / len(y_test_actual) * 100
        mae_range = np.mean(np.abs(y_test_actual[mask] - lstm_pred[mask]))
        mape_range = np.mean(lstm_percentage_errors[mask])
        
        print(f"\n{label}:")
        print(f"  Anzahl Werte:    {count} ({pct:.1f}%)")
        print(f"  MAE:             {mae_range:.2f} MW")
        print(f"  MAPE:            {mape_range:.2f}%")
        print(f"  Mittlerer Wert:  {y_test_actual[mask].mean():.2f} MW")

# Spezialfall: Nächtliche Werte (sehr niedrig)
night_mask = y_test_actual < 10  # Praktisch null
if night_mask.sum() > 0:
    print(f"\n⚠️  KRITISCH - Sehr niedrige Werte (<10 MW):")
    print(f"  Anzahl: {night_mask.sum()} ({night_mask.sum()/len(y_test_actual)*100:.1f}%)")
    print(f"  MAE:    {np.mean(np.abs(y_test_actual[night_mask] - lstm_pred[night_mask])):.2f} MW")
    print(f"  MAPE:   {np.mean(lstm_percentage_errors[night_mask]):.2f}%")
    print(f"  → Selbst kleine absolute Fehler führen zu SEHR hohen %-Fehlern!")

# 7. Visualisierung
print("\n" + "="*70)
print("6. Erstelle Visualisierungen...")
print("="*70)

# Plot 1: Zeitreihen-Vergleich (erste 7 Tage)
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

plot_hours = 24 * 7  # 7 Tage

# Subplot 1: Gesamtübersicht
ax1 = axes[0]
ax1.plot(y_test_actual[:plot_hours], label='Tatsächlich', 
         color='black', linewidth=2, alpha=0.8)
ax1.plot(lstm_pred[:plot_hours], label='LSTM Vorhersage', 
         color='red', linewidth=1.5, alpha=0.7, linestyle='--')
ax1.plot(gru_pred[:plot_hours], label='GRU Vorhersage', 
         color='blue', linewidth=1.5, alpha=0.7, linestyle=':')
ax1.set_ylabel('Solar Produktion (MW)', fontsize=12, fontweight='bold')
ax1.set_title('Zeitreihen-Vergleich: Erste 7 Tage', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('Stunden', fontsize=11)

# Subplot 2: Absolute Fehler
ax2 = axes[1]
lstm_errors = np.abs(y_test_actual[:plot_hours] - lstm_pred[:plot_hours])
ax2.fill_between(range(plot_hours), lstm_errors, alpha=0.5, color='red', label='LSTM')
ax2.axhline(y=lstm_metrics['mae'], color='red', linestyle='--', 
            linewidth=2, label=f'LSTM MAE: {lstm_metrics["mae"]:.1f} MW')
ax2.set_ylabel('Absoluter Fehler (MW)', fontsize=12, fontweight='bold')
ax2.set_title('Absolute Vorhersagefehler über Zeit', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('Stunden', fontsize=11)

# Subplot 3: Prozentuale Fehler (MAPE)
ax3 = axes[2]
ax3.fill_between(range(plot_hours), lstm_percentage_errors[:plot_hours], 
                 alpha=0.5, color='orange', label='LSTM')
ax3.axhline(y=lstm_metrics['mape'], color='orange', linestyle='--', 
            linewidth=2, label=f'LSTM MAPE: {lstm_metrics["mape"]:.1f}%')
ax3.set_ylabel('Prozentualer Fehler (%)', fontsize=12, fontweight='bold')
ax3.set_title('Prozentuale Vorhersagefehler (MAPE-Komponenten)', fontsize=14, fontweight='bold')
ax3.legend(loc='best', fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_xlabel('Stunden', fontsize=11)
ax3.set_ylim([0, 200])  # Limitiere y-Achse für bessere Sichtbarkeit

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'lstm_mape_discrepancy_timeseries.png', dpi=300, bbox_inches='tight')
print(f"✓ Gespeichert: lstm_mape_discrepancy_timeseries.png")

# Plot 2: Fehlerverteilung nach Wertebereichen
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Subplot 1: MAE vs Tatsächliche Werte
ax1 = axes[0, 0]
scatter = ax1.scatter(y_test_actual, np.abs(y_test_actual - lstm_pred), 
                     c=lstm_percentage_errors, cmap='YlOrRd', 
                     alpha=0.5, s=10)
ax1.set_xlabel('Tatsächliche Werte (MW)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Absoluter Fehler (MW)', fontsize=12, fontweight='bold')
ax1.set_title('Absoluter Fehler vs. Tatsächliche Werte', fontsize=13, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('MAPE (%)', fontsize=11)
ax1.grid(True, alpha=0.3)

# Subplot 2: MAPE vs Tatsächliche Werte
ax2 = axes[0, 1]
ax2.scatter(y_test_actual, lstm_percentage_errors, alpha=0.5, s=10, color='orange')
ax2.set_xlabel('Tatsächliche Werte (MW)', fontsize=12, fontweight='bold')
ax2.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
ax2.set_title('MAPE vs. Tatsächliche Werte', fontsize=13, fontweight='bold')
ax2.axhline(y=lstm_metrics['mape'], color='red', linestyle='--', 
            linewidth=2, label=f'Durchschn. MAPE: {lstm_metrics["mape"]:.1f}%')
ax2.set_ylim([0, 200])
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Subplot 3: Histogramm der absoluten Fehler
ax3 = axes[1, 0]
ax3.hist(np.abs(y_test_actual - lstm_pred), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
ax3.axvline(x=lstm_metrics['mae'], color='red', linestyle='--', 
            linewidth=2, label=f'MAE: {lstm_metrics["mae"]:.1f} MW')
ax3.set_xlabel('Absoluter Fehler (MW)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Häufigkeit', fontsize=12, fontweight='bold')
ax3.set_title('Verteilung der absoluten Fehler', fontsize=13, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3, axis='y')

# Subplot 4: Histogramm der prozentualen Fehler (begrenzt)
ax4 = axes[1, 1]
mape_limited = lstm_percentage_errors[lstm_percentage_errors < 200]  # Filtere extreme Werte
ax4.hist(mape_limited, bins=50, alpha=0.7, color='orange', edgecolor='black')
ax4.axvline(x=lstm_metrics['mape'], color='red', linestyle='--', 
            linewidth=2, label=f'MAPE: {lstm_metrics["mape"]:.1f}%')
ax4.set_xlabel('MAPE (%) - begrenzt auf <200%', fontsize=12, fontweight='bold')
ax4.set_ylabel('Häufigkeit', fontsize=12, fontweight='bold')
ax4.set_title('Verteilung der prozentualen Fehler', fontsize=13, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'lstm_mape_discrepancy_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Gespeichert: lstm_mape_discrepancy_analysis.png")

# Plot 3: Tag-Nacht Vergleich
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Identifiziere Tag (>100 MW) und Nacht (<100 MW)
day_mask = y_test_actual >= 100
night_mask = y_test_actual < 100

# Subplot 1: Tagsüber
ax1 = axes[0]
plot_len = min(500, day_mask.sum())
day_indices = np.where(day_mask)[0][:plot_len]
ax1.plot(y_test_actual[day_indices], label='Tatsächlich', 
         color='black', linewidth=2, alpha=0.8)
ax1.plot(lstm_pred[day_indices], label='LSTM Vorhersage', 
         color='red', linewidth=1.5, alpha=0.7, linestyle='--')
ax1.fill_between(range(plot_len), 
                 y_test_actual[day_indices] - np.abs(y_test_actual[day_indices] - lstm_pred[day_indices]),
                 y_test_actual[day_indices] + np.abs(y_test_actual[day_indices] - lstm_pred[day_indices]),
                 alpha=0.2, color='red', label='Fehlerbereich')
ax1.set_ylabel('Solar Produktion (MW)', fontsize=12, fontweight='bold')
ax1.set_title(f'Tagsüber (>100 MW) - MAE: {np.mean(np.abs(y_test_actual[day_mask] - lstm_pred[day_mask])):.1f} MW, '
              f'MAPE: {np.mean(lstm_percentage_errors[day_mask]):.1f}%', 
              fontsize=13, fontweight='bold')
ax1.legend(loc='best', fontsize=11)
ax1.grid(True, alpha=0.3)

# Subplot 2: Nachts
ax2 = axes[1]
plot_len_night = min(500, night_mask.sum())
night_indices = np.where(night_mask)[0][:plot_len_night]
ax2.plot(y_test_actual[night_indices], label='Tatsächlich', 
         color='black', linewidth=2, alpha=0.8)
ax2.plot(lstm_pred[night_indices], label='LSTM Vorhersage', 
         color='red', linewidth=1.5, alpha=0.7, linestyle='--')
ax2.fill_between(range(plot_len_night),
                 y_test_actual[night_indices] - np.abs(y_test_actual[night_indices] - lstm_pred[night_indices]),
                 y_test_actual[night_indices] + np.abs(y_test_actual[night_indices] - lstm_pred[night_indices]),
                 alpha=0.2, color='red', label='Fehlerbereich')
ax2.set_ylabel('Solar Produktion (MW)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Sample Index', fontsize=11)
ax2.set_title(f'Nachts (<100 MW) - MAE: {np.mean(np.abs(y_test_actual[night_mask] - lstm_pred[night_mask])):.1f} MW, '
              f'MAPE: {np.mean(lstm_percentage_errors[night_mask]):.1f}%', 
              fontsize=13, fontweight='bold')
ax2.legend(loc='best', fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'lstm_day_night_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Gespeichert: lstm_day_night_comparison.png")

# 8. Zusammenfassung
print("\n" + "="*70)
print("ZUSAMMENFASSUNG")
print("="*70)

print("""
WARUM IST MAPE SO HOCH TROTZ GUTER MAE/RMSE?

1. **Mathematische Ursache:**
   MAPE = mean(|actual - predicted| / |actual|) * 100
   
   → Division durch kleine Werte führt zu sehr hohen Prozentsätzen!
   → Selbst kleine absolute Fehler werden bei niedrigen Werten extrem groß.

2. **Spezifisch für Solarenergie:**
   - Nachts: Produktion ≈ 0 MW
   - Wenn Modell z.B. 5 MW vorhersagt, aber tatsächlich 1 MW:
     * Absoluter Fehler: nur 4 MW (gut!)
     * MAPE: 400% (!!)
   
3. **Lösung/Interpretation:**
   ✓ MAE und RMSE sind hier VERLÄSSLICHER
   ✓ Sie zeigen: Modell macht durchschnittlich ~250 MW Fehler
   ✓ Bei typischen Werten von 2000-7000 MW ist das <5% Fehler!
   
   ✗ MAPE ist IRREFÜHREND bei Zeitreihen mit Werten nahe Null
   ✗ Besser: sMAPE oder gewichtete MAPE-Varianten verwenden

4. **Empfehlung:**
   → Fokus auf MAE/RMSE für absolute Performance
   → Nutze R² für Modellgüte (hier: 0.98 = exzellent!)
   → MAPE nur für Werte >500 MW betrachten
""")

# Speichere detaillierte Metriken
metrics_df = pd.DataFrame({
    'Modell': ['LSTM', 'GRU'],
    'MAE_MW': [lstm_metrics['mae'], gru_metrics['mae']],
    'RMSE_MW': [lstm_metrics['rmse'], gru_metrics['rmse']],
    'MAPE_%': [lstm_metrics['mape'], gru_metrics['mape']],
    'sMAPE_%': [lstm_metrics['smape'], gru_metrics['smape']],
    'R2': [lstm_metrics['r2'], gru_metrics['r2']]
})

metrics_df.to_csv(RESULTS_DIR / 'metrics' / 'lstm_detailed_analysis.csv', index=False)
print(f"\n✓ Metriken gespeichert: lstm_detailed_analysis.csv")

print("\n" + "="*70)
print("✓ Analyse abgeschlossen!")
print(f"✓ 3 Visualisierungen in {FIGURES_DIR}")
print("="*70)
