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

**Status**: ⚠️ **PROBLEM ERKANNT**

**Best Model**: Naive (FEHLER!)

| Metrik | Wert |
|--------|------|
| **R²** | 1.0000 ❌ |
| **RMSE** | 0.00 ❌ |

**Problem**: 
- Alle Baseline-Modelle zeigen R²=1.0, RMSE=0.0
- Alle ML-Modelle zeigen R²=0.0
- **Verdacht**: Datenleck oder Skalierungsfehler im Preprocessing

**Modelle getestet**: 17 (Baselines, Statistical, ML Trees, Deep Learning, Advanced)

**Nächste Schritte**: 
1. Preprocessing überprüfen (Skalierung, Feature-Leak)
2. Train/Val/Test Split validieren
3. Pipeline debuggen

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

## 🏆 Gesamtvergleich (ohne Wind Offshore)

| Zeitreihe | Best Model | R² | RMSE | MAE | Features |
|-----------|-----------|-----|------|-----|----------|
| **Consumption** 🏭 | Random Forest | **0.9999** | 104.44 MW | 57.56 MW | 27 |
| **Wind Onshore** 💨 | Random Forest | **0.9997** | 33.96 MW | 13.10 MW | 27 |
| **Solar** ☀️ | Random Forest | **0.9994** | 122.97 MW | 39.09 MW | 27 |
| **Price** 💰 | LightGBM | **0.9800** | 9.99 EUR/MWh | 1.73 EUR/MWh | 28 |
| **Wind Offshore** 🌊 | ❌ FEHLER | ❌ 1.0000 | ❌ 0.00 | - | - |

**Durchschnitt (4 erfolgreiche)**: R² = **0.9973**

---

## 🔍 Wichtigste Erkenntnisse

### ✅ Was funktioniert hervorragend:

1. **Random Forest dominiert** (3 von 4 Best Models)
2. **Tree-basierte ML-Modelle** (RF, XGBoost, LightGBM) sind sehr robust
3. **Feature Engineering** ist entscheidend:
   - `lag_1`, `diff_1` (kurzfristige Abhängigkeit)
   - `lag_24`, `diff_24` (Tagesmuster)
   - `rolling_std_3` (Volatilität)
   - `lag_168` (Wochenmuster, bei Consumption)

4. **Konsistente Top-Features** über alle Zeitreihen:
   - Diff-Features (Änderungsrate)
   - Lag-Features (Vergangenheitswerte)
   - Rolling Statistics (Volatilität)

### ⚠️ Was nicht funktioniert:

1. **LSTM** deutlich schlechter als ML-Modelle:
   - Solar: R² = 0.86 vs. RF 0.9994
   - Wind Onshore: R² = 0.90 vs. RF 0.9997
   - Price: R² = 0.57 vs. LightGBM 0.98
   - Consumption: R² = 0.45 vs. RF 0.9999

2. **Baseline-Modelle** schlecht bis negativ:
   - Naive, Seasonal Naive, Mean: R² oft negativ
   - Nur bei strukturierten Daten (Consumption) funktioniert Seasonal Naive (R²=0.39)

3. **Wind Offshore Pipeline** komplett fehlerhaft

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

### Priorität 1: Wind Offshore Debug
- [ ] Preprocessing-Code überprüfen
- [ ] Skalierung validieren
- [ ] Feature-Leak identifizieren
- [ ] Pipeline-Fix implementieren
- [ ] Erneut ausführen

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

3. **Feature Set**:
   - Minimum: lag_1, diff_1, lag_24, diff_24
   - Empfohlen: + rolling_std_3, lag_168, momentum

4. **Monitoring**:
   - Überwache MAPE < 5% für gute Performance
   - Re-train bei Drift (> 10% MAPE-Anstieg)

---

## 📊 Laufzeiten

| Pipeline | Dauer | Modelle |
|----------|-------|---------|
| Solar | ~2 Min | 7 |
| Wind Offshore | ~8 Min | 17 |
| Wind Onshore | ~3 Min | 7 |
| Price | ~2 Min | 7 |
| Consumption | ~2 Min | 7 |
| **Gesamt** | **~17 Min** | **45** |

---

**Fazit**: Systematische Evaluation zeigt klare Dominanz von Tree-basierten ML-Modellen (Random Forest, LightGBM). Deep Learning (LSTM) deutlich unterlegen bei diesen strukturierten Zeitreihen. Wind Offshore Pipeline benötigt dringend Debug.
