# Phase 2: Systematische Modell-Evaluation - Zusammenfassung

**Datum**: 31. Januar 2026  
**Status**: ✅ Abgeschlossen

---

## 🎯 Zielsetzung

Systematische Evaluation aller verfügbaren Modelle auf **5 Zeitreihen**:
- Solar
- Wind Offshore
- Wind Onshore
- Price
- Consumption

Jede Zeitreihe durchlief **9 Phasen** mit insgesamt **~17 Modellen**.

---

## 📊 Ergebnisse pro Zeitreihe

### 1. ☀️ Solar

**Best Model**: Random Forest (ML Tree)

| Metrik | Wert |
|--------|------|
| **R²** | 0.9994 |
| **RMSE** | 122.97 MW |
| **MAE** | 39.09 MW |
| **MAPE** | 3.16% |

**Top 5 Features**: diff_1, lag_1, diff_24, lag_24, rolling_std_3

**Modelle getestet**: 7 (Naive, Seasonal Naive, Mean, Random Forest, XGBoost, LightGBM, LSTM)

---

### 2. 🌊 Wind Offshore

**Best Model**: GRU (Deep Learning) ✅

| Metrik | Wert |
|--------|------|
| **R²** | 0.9119 |
| **RMSE** | 44.72 MW |
| **MAE** | - |

**⚠️ KRITISCHER FIX IMPLEMENTIERT**:
- **Problem**: 9-monatige Stillstandsperiode (Apr 2023 - Jan 2024)
  - 7.081 Nullwerte (38.7% der Daten)
  - Verursachte Datenleck → R²=1.0/0.0 Fehler
- **Lösung**: Nur Daten VOR Stillstand nutzen (11.231 → 11.063 Datenpunkte)
- **Ergebnis**: Realistische Scores, Deep Learning übertrifft ML Trees

**Top 3 Models**:
1. GRU: R²=0.9119
2. LSTM: R²=0.9096  
3. Simple RNN: R²=0.9036

**Modelle getestet**: 14 (Baselines, Statistical, ML Trees, Deep Learning)

---

### 3. 💨 Wind Onshore

**Best Model**: Random Forest (ML Tree)

| Metrik | Wert |
|--------|------|
| **R²** | 0.9997 |
| **RMSE** | 33.96 MW |
| **MAE** | 13.10 MW |
| **MAPE** | 2.24% |

**Top 5 Features**: diff_1, lag_1, diff_24, lag_24, lag_2

**Modelle getestet**: 7 (Naive, Seasonal Naive, Mean, Random Forest, XGBoost, LightGBM, LSTM)

---

### 4. 💰 Price

**Best Model**: LightGBM (ML Tree)

| Metrik | Wert |
|--------|------|
| **R²** | 0.9800 |
| **RMSE** | 9.99 EUR/MWh |
| **MAE** | 1.73 EUR/MWh |
| **MAPE** | 4.58% |

**Top 5 Features**: diff_1, lag_1, momentum_3h, diff_24, rolling_std_3

**Modelle getestet**: 7 (Naive, Seasonal Naive, Mean, Random Forest, XGBoost, LightGBM, LSTM)

---

### 5. 🏭 Consumption

**Best Model**: Random Forest (ML Tree)

| Metrik | Wert |
|--------|------|
| **R²** | 0.9999 |
| **RMSE** | 104.44 MW |
| **MAE** | 57.56 MW |
| **MAPE** | 0.10% |

**Top 5 Features**: lag_1, diff_1, lag_168, diff_24, rolling_std_3

**Modelle getestet**: 7 (Naive, Seasonal Naive, Mean, Random Forest, XGBoost, LightGBM, LSTM)

---

## 🏆 Gesamtvergleich (ALLE 5 Zeitreihen)

| Zeitreihe | Best Model | R² | RMSE | MAE | Status |
|-----------|-----------|-----|------|-----|--------|
| **Consumption** 🏭 | Random Forest | **0.9999** | 104.44 MW | 57.56 MW | ✅ |
| **Wind Onshore** 💨 | Random Forest | **0.9997** | 33.96 MW | 13.10 MW | ✅ |
| **Solar** ☀️ | Random Forest | **0.9994** | 122.97 MW | 39.09 MW | ✅ |
| **Price** 💰 | LightGBM | **0.9800** | 9.99 €/MWh | 1.73 €/MWh | ✅ |
| **Wind Offshore** 🌊 | GRU | **0.9119** | 44.72 MW | - | ✅ **GEFIXT** |

**Durchschnitt (ALLE 5)**: R² = **0.9782** 🎉

---

## 🔍 Wichtigste Erkenntnisse

### ✅ Was funktioniert hervorragend:

1. **Random Forest dominiert** bei strukturierten Zeitreihen (3 von 5 Best Models)
2. **Deep Learning (GRU/LSTM) übertrifft** bei Wind Offshore (weniger strukturiert)
3. **Tree-basierte ML-Modelle** (RF, XGBoost, LightGBM) sind sehr robust für strukturierte Daten
4. **Feature Engineering** ist entscheidend:
   - `lag_1`, `diff_1` (kurzfristige Abhängigkeit)
   - `lag_24`, `diff_24` (Tagesmuster)
   - `rolling_std_3` (Volatilität)
   - `lag_168` (Wochenmuster, bei Consumption)

5. **Konsistente Top-Features** über alle Zeitreihen:
   - Diff-Features (Änderungsrate)
   - Lag-Features (Vergangenheitswerte)
   - Rolling Statistics (Volatilität)

6. **Datenqualität** ist kritisch:
   - Wind Offshore: 9-monatige Stillstandsperiode musste ausgeschlossen werden
   - Signifikante Nullwerte können Datenlecks verursachen

### ⚠️ Was gelernt wurde:

1. **LSTM nicht immer optimal**:
   - Solar: R² = 0.86 vs. RF 0.9994
   - Wind Onshore: R² = 0.90 vs. RF 0.9997
   - Price: R² = 0.57 vs. LightGBM 0.98
   - Consumption: R² = 0.45 vs. RF 0.9999
   - **ABER**: Bei Wind Offshore (GRU R²=0.91) besser als ML Trees!

2. **Baseline-Modelle** schlecht bis negativ:
   - Naive, Seasonal Naive, Mean: R² oft negativ
   - Nur bei strukturierten Daten (Consumption) funktioniert Seasonal Naive (R²=0.39)

3. **Datenqualität-Probleme** kritisch:
   - Wind Offshore: 9-monatige Stillstandsperiode (7.081 Nullwerte = 38.7%)
   - Verursachte massiven Datenleck (R²=1.0/0.0)
   - Fix: Nur Daten VOR Stillstand nutzen → realistische Ergebnisse

---

## 📁 Generierte Outputs

Für jede Zeitreihe:

### Metriken (CSV):
- `results/metrics/{serie}_all_models_extended.csv`
- `results/metrics/{serie}_extended_summary.json`

### Visualisierungen (PNG):
- `results/figures/{serie}_extended_01_timeline.png`
- `results/figures/{serie}_extended_09_final_comparison.png`
- `results/figures/{serie}_extended_feature_importance.png`

---

## 🔄 Nächste Schritte

### ✅ Priorität 1: Wind Offshore Debug
- [x] Preprocessing-Code überprüft
- [x] Stillstandsperiode identifiziert (9 Monate)
- [x] Feature-Leak gefixed (nur Daten vor Stillstand)
- [x] Pipeline-Fix implementiert
- [x] Erneut ausgeführt → **R²=0.9119 ✅**

### Priorität 2: LSTM-Optimierung (Optional)
- [ ] Hyperparameter-Tuning
- [ ] Sequence-Length experimentieren
- [ ] Mehr Epochen trainieren
- [ ] Architektur anpassen (Bi-LSTM, Attention)

### Priorität 3: Multivariate Ansätze
- [ ] Cross-Series Features (z.B. Wind → Solar)
- [ ] External Features (Wetter, Feiertage)
- [ ] VAR/VECM Modelle

### Priorität 4: Ensemble-Methoden
- [ ] Stacking (RF + LightGBM + XGBoost)
- [ ] Weighted Averaging
- [ ] Blending

---

## 💡 Empfehlungen für Produktion

1. **Nutze Random Forest** für:
   - Solar (R² = 0.9994)
   - Wind Onshore (R² = 0.9997)
   - Consumption (R² = 0.9999)

2. **Nutze LightGBM** für:
   - Price (R² = 0.9800, schneller als RF)

3. **Nutze Deep Learning (GRU)** für:
   - Wind Offshore (R² = 0.9119, besser als ML Trees bei weniger strukturierten Daten)

4. **Feature Set**:
   - Minimum: lag_1, diff_1, lag_24, diff_24
   - Empfohlen: + rolling_std_3, lag_168, momentum

5. **Monitoring**:
   - Überwache MAPE < 5% für gute Performance
   - Re-train bei Drift (> 10% MAPE-Anstieg)

6. **Datenqualität**:
   - Prüfe auf längere Stillstandsperioden
   - Exkludiere oder markiere als Feature
   - Vermeide Datenlecks durch lag-Features während Nullperioden

---

## 📊 Laufzeiten

| Pipeline | Dauer | Modelle |
|----------|-------|---------|
| Solar | ~2 Min | 7 |
| Wind Offshore | ~8 Min | 14 (ohne Advanced) |
| Wind Onshore | ~3 Min | 7 |
| Price | ~2 Min | 7 |
| Consumption | ~2 Min | 7 |
| **Gesamt** | **~17 Min** | **42** |

---

**Fazit**: Systematische Evaluation abgeschlossen! **Random Forest dominiert** strukturierte Zeitreihen (Solar, Wind Onshore, Consumption), **Deep Learning (GRU) übertrifft** bei weniger strukturierten Daten (Wind Offshore), **LightGBM optimal** für Price. Wind Offshore Fix zeigt Wichtigkeit von Datenqualitäts-Checks. Durchschnittlicher R²=0.9782 über alle 5 Zeitreihen.
