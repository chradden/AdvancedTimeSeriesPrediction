# LSTM MAPE-Diskrepanz: Analyse und Erklärung

## Zusammenfassung der Erkenntnisse

### Das "Problem"
Der Kommentator hat beobachtet:
- ✓ LSTM-Modelle haben **sehr gute MAE und RMSE** Werte
- ✗ Aber **hohe MAPE** (Mean Absolute Percentage Error) Werte
- 💡 Empfehlung: Zeitreihen plotten, um die Diskrepanz zu verstehen

---

## Analyse-Ergebnisse

### 1. Die tatsächlichen Metriken (aktualisiert)

**LSTM-Modell auf Solar-Daten:**
- **MAE**: 251.53 MW
- **RMSE**: 377.19 MW
- **MAPE**: **3.48%** ← SEHR GUT!
- **R²**: 0.9822 ← EXZELLENT!

### 2. Das frühere Problem (gelöst)

**Alte CSV-Datei** (`solar_deep_learning_results.csv`):
```
LSTM: test_mape=61.85%  ← HOCH!
```

**Neue CSV-Datei** (`solar_deep_learning_results_CORRECTED.csv`):
```
LSTM: MAPE=3.48%  ← GUT!
```

**Was war das Problem?**
- Die alten Metriken wurden auf **normalisierten/skalierten Daten** berechnet
- MAPE = |actual_scaled - predicted_scaled| / |actual_scaled| × 100
- Bei skalierten Werten zwischen 0-1: Selbst kleine Fehler → hohe %
- Nach **Rücktransformation zu MW**: MAPE ist korrekt niedrig

---

## Visualisierungen erstellt

### 1. `lstm_mape_discrepancy_timeseries.png`
Zeigt über 7 Tage:
- Tatsächliche vs. vorhergesagte Werte
- Absolute Fehler über Zeit (MAE ~251 MW)
- Prozentuale Fehler über Zeit (MAPE ~3.5%)

**Erkenntnis:** Das Modell folgt der Zeitreihe sehr genau!

### 2. `lstm_mape_discrepancy_analysis.png`
4 Subplots:
- Scatter: Absoluter Fehler vs. tatsächliche Werte
- Scatter: MAPE vs. tatsächliche Werte
- Histogramm: Verteilung absoluter Fehler
- Histogramm: Verteilung prozentualer Fehler

**Erkenntnis:** Fehler sind gleichmäßig verteilt, keine systematischen Probleme!

### 3. `lstm_day_night_comparison.png`
Vergleich Tag vs. Nacht:
- Tagsüber (>100 MW): MAE ~251 MW, MAPE ~3.5%
- Nachts (<100 MW): keine Werte in diesem Datensatz!

**Wichtig:** Diese Solar-Daten haben KEINE Nullwerte (nur 2831-13638 MW)
→ Deshalb ist MAPE hier verlässlich!

---

## Warum kann MAPE problematisch sein?

### Mathematisches Problem
```
MAPE = mean(|actual - predicted| / |actual|) × 100
```

**Problem bei kleinen Werten:**
- Wenn actual = 1 MW, predicted = 5 MW
- Absoluter Fehler = 4 MW (klein)
- MAPE = 400% (RIESIG!)

### Typisches Szenario für Solarenergie
Bei Datensätzen MIT nächtlichen Werten:
- Nachts: Solar ≈ 0 MW
- Kleine absolute Fehler → extreme MAPE-Werte
- MAPE wird durch Nachtwerte dominiert

### Dein Projekt
**KEINE nächtlichen Werte im Datensatz!**
- Min: 2831 MW
- Max: 13638 MW
- Mean: 8631 MW
→ MAPE ist hier ein **valider Indikator**

---

## Interpretation des Kommentars

Der Kommentator hatte wahrscheinlich die **alte CSV-Datei** (`solar_deep_learning_results.csv`) gesehen mit:
```
LSTM: test_mape=61.85%
```

Dies war ein **Berechnungsfehler** (Metriken auf skalierten Daten).

**Nach Korrektur:**
- MAPE: 3.48% ← SEHR GUT!
- Das Modell macht im Schnitt nur 3.5% Fehler
- Bei einem Durchschnittswert von 8631 MW = ~300 MW Fehler
- Stimmt mit MAE von 251 MW überein ✓

---

## Empfehlungen

### ✓ Was du bereits richtig machst:
1. **Metriken korrekt berechnet** (nach De-Normalisierung)
2. **R² = 0.982** ist exzellent
3. **Plots erstellt** (in results/figures)

### 💡 Zusätzliche Empfehlungen:

#### 1. Multi-Metrik Ansatz
Nutze immer mehrere Metriken:
- **MAE/RMSE**: Absolute Performance in MW
- **R²**: Modellgüte (Varianz erklärt)
- **MAPE/sMAPE**: Relative Performance
- **MASE**: Vergleich zu naivem Baseline-Modell

#### 2. Wenn nächtliche Werte dabei wären:
```python
# Filtere Werte > Schwellwert für MAPE
threshold = 100  # MW
mask = y_actual > threshold
mape_filtered = np.mean(np.abs((y_actual[mask] - y_pred[mask]) / y_actual[mask])) * 100
```

#### 3. Nutze sMAPE (symmetric MAPE)
```python
smape = np.mean(2 * np.abs(y_actual - y_pred) / (np.abs(y_actual) + np.abs(y_pred))) * 100
```
→ Robuster gegen kleine Werte!

---

## Fazit

**Für deinen Kommentator:**

> Vielen Dank für den Hinweis! Ich habe die Zeitreihen analysiert und visualisiert.
> 
> **Ergebnis:** Die ursprünglich hohe MAPE (61%) war ein Berechnungsfehler auf 
> skalierten Daten. Nach Korrektur:
> 
> - **MAPE: 3.48%** (sehr gut!)
> - **MAE: 251 MW** bei durchschnittlich 8631 MW
> - **R²: 0.982** (exzellente Modellgüte)
> 
> Die Visualisierungen zeigen, dass das LSTM-Modell die Zeitreihe sehr präzise 
> lernt und extrapoliert. Es gibt keine systematischen Abweichungen.
> 
> Die Plots sind unter `results/figures/lstm_*.png` zu finden.

**Technisch:**
Die MAPE-Diskrepanz existiert in diesem Datensatz nicht mehr. Bei Solarenergie-
Daten MIT nächtlichen Werten (≈0 MW) wäre MAPE tatsächlich problematisch, aber 
dieser Datensatz enthält nur Tageswerte (2831-13638 MW), daher ist MAPE hier 
ein valider Indikator.

---

## Dateien

**Analyse-Skript:**
- `analyze_lstm_mape_discrepancy.py`

**Visualisierungen:**
- `results/figures/lstm_mape_discrepancy_timeseries.png`
- `results/figures/lstm_mape_discrepancy_analysis.png`
- `results/figures/lstm_day_night_comparison.png`

**Metriken:**
- `results/metrics/lstm_detailed_analysis.csv`
- `results/metrics/solar_deep_learning_results_CORRECTED.csv` (korrigiert)
