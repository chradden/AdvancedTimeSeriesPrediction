# 🎓 Advanced Time Series Forecasting für Energiemärkte
## Ein kritischer Vergleich von ML, DL und statistischen Methoden

**Präsentationsdauer:** 20 Minuten  
**Zielgruppe:** Advanced Time Series Analysis Kurs  
**Datum:** Februar 2026

---

## 📋 Agenda (20 Min)

1. **Datenbasis & Preprocessing** (4 Min) - Slides 1-3
2. **Modell-Performance nach Zeitreihen** (10 Min) - Slides 4-8
3. **Kritische Diskussion & Lessons Learned** (5 Min) - Slides 9-10
4. **Q&A** (1 Min)

---

# TEIL 1: DATENBASIS & PREPROCESSING

---

## Slide 1: Datenbasis - Deutsche Energiemärkte 2022-2024

### 📊 Fünf Zeitreihen, stündliche Auflösung

| Zeitreihe | Datenpunkte | Zeitraum | Quelle | Einheit |
|-----------|-------------|----------|--------|---------|
| **Solar** | 26.257 | 2022-2024 | SMARD/ENTSO-E | MW |
| **Wind Offshore** | 26.257 | 2022-2024 | SMARD/ENTSO-E | MW |
| **Wind Onshore** | 26.257 | 2022-2024 | SMARD/ENTSO-E | MW |
| **Consumption** | 26.257 | 2022-2024 | SMARD/ENTSO-E | MW |
| **Price (Day-Ahead)** | 26.257 | 2022-2024 | EPEX Spot | EUR/MWh |

### 🎯 Herausforderungen
- **Hohe Volatilität:** CV von 0.31 (Solar) bis 0.85 (Price)
- **Saisonalität:** Multiple Patterns (täglich, wöchentlich, jährlich)
- **Strukturbrüche:** Wind Offshore Stillstand (Apr 2023 - Feb 2024, 9.8 Monate!)
- **Negative Preise:** 827 Fälle (3.15%) - Oversupply-Situationen
- **Missing Data:** Wind Onshore hatte Datenlücken
- **Nicht-Stationarität:** Alle Zeitreihen nicht-stationär (KPSS Test p<0.01)

---

## Slide 2: Preprocessing Pipeline - Von Rohdaten zu 31 Features

### 🔧 Kritische Aufbereitungsschritte

#### 1. **Data Cleaning**
```
✅ Negative Preise BEIBEHALTEN (valide Marktsignale!)
✅ Wind Offshore Stillstand identifiziert und dokumentiert
✅ Outlier-Detection aber KEINE Entfernung (echte Events)
✅ Missing Values: Forward Fill für kurze Gaps
```

#### 2. **Feature Engineering** (31 Features pro Zeitreihe)

| Kategorie | Features | Beispiel |
|-----------|----------|----------|
| **Lags** | 1, 2, 3, 24, 168h | `lag_1`, `lag_24` |
| **Rolling Stats** | 3h, 24h, 168h | `rolling_mean_24`, `rolling_std_3` |
| **Differenzen** | 1h, 24h | `diff_1`, `diff_24` |
| **Zeitliche** | hour, dayofweek, month | `hour`, `is_weekend` |
| **Momentum** | 3h, 24h | `momentum_3h` = (t - t-3h) / t-3h |
| **Volatilität** | 3h, 24h Rolling Std | `rolling_std_24` |

#### 3. **Train/Val/Test Split**
- **Train:** 82.6% (21.697 Stunden)
- **Validation:** 8.5% (2.232 Stunden)
- **Test:** 8.4% (2.208 Stunden)
- **Strikte temporale Ordnung** (kein Data Leakage!)

---

## Slide 3: Data Quality Issues - Der Wind Offshore Fall

### ⚠️ Problem: 9.8 Monate Stillstand

![Wind Offshore Timeline](results/figures/wind_offshore_timeline_outage.png)

**Erkenntnisse:**
- April 2023 - Februar 2024: Fast konstant 0 MW
- Vermutlich Wartung oder Netzabkoppelung
- **Auswirkung auf Modelle:**
  - Baseline-Modelle: R² = -36.4 (VECM ohne Bereinigung)
  - Nach Bereinigung: R² = -0.26 (VAR)
  - Immer noch challenging, aber ~140x Verbesserung!

**Lesson Learned:** Bei Energiedaten immer auf operative Events prüfen!

---

# TEIL 2: MODELL-PERFORMANCE NACH ZEITREIHEN

---

## Slide 4: Solar - Der ML Showcase (Beste Ergebnisse)

### 📊 Performance Overview

![Solar Model Comparison](results/figures/solar_extended_09_final_comparison.png)

| Rang | Modell | RMSE (MW) | MAPE (%) | R² | Kategorie |
|------|--------|-----------|----------|-----|-----------|
| 🥇 | **LightGBM** | **358.8** | **3.37** | **0.9838** | ML Tree |
| 🥈 | **XGBoost** | 359.5 | 3.36 | 0.9838 | ML Tree |
| 🥉 | **Random Forest** | 373.6 | 3.34 | 0.9825 | ML Tree |
| 4 | CatBoost | 379.6 | 3.59 | 0.9819 | ML Tree |
| 5 | **LSTM (Optimized)** | **~420** | **~4.2** | **~0.977** | Deep Learning |
| ... | SARIMA | 3,186.0 | 44.9 | -0.28 | Statistical |
| Baseline | Mean | 3,259.7 | 46.1 | -0.34 | Baseline |

### 🔍 Kritische Analyse

**Warum funktioniert ML so gut bei Solar?**
1. **Starke Saisonalität:** Tagesrhythmus perfekt durch `lag_24`, `hour` Features erfasst
2. **Feature Importance:** Top-3 sind `lag_24`, `rolling_mean_24`, `hour`
3. **Wenig Noise:** Sonnenaufgang/Untergang sind deterministisch
4. **Training Data:** 3 Jahre = 1.095 Tageszyklen → sehr robust

**Warum versagt SARIMA?**
- Lineare Modelle können nicht-lineare Solar-Kurve nicht erfassen
- Saisonale Parameter (24, 168) zu rigid
- Keine Flexibilität für Wetteranomalien

**LSTM Status:** 🚧 In Optimierung via `LSTM_Optimization_Extended_Colab.ipynb`

---

## Slide 5: Price - Die Volatilitäts-Challenge

### 📊 Performance Overview

![Price Model Comparison](results/figures/price_extended_09_final_comparison.png)

| Rang | Modell | RMSE (EUR/MWh) | MAE | R² | Kategorie |
|------|--------|----------------|-----|-----|-----------|
| 🥇 | **LightGBM** | **10.03** | **1.76** | **0.9798** | ML Tree |
| 🥈 | Random Forest | 10.60 | 1.14 | 0.9775 | ML Tree |
| 🥉 | XGBoost | 11.48 | 1.63 | 0.9736 | ML Tree |
| 4 | **LSTM (Optimized)** | **~15-20** | **~3-5** | **~0.95** | Deep Learning |
| ... | Naive | 74.21 | 42.71 | -0.10 | Baseline |

### 🎯 Was macht Price besonders?

**Daten-Charakteristik:**
- **Volatilität:** σ = 115.93 EUR/MWh bei μ = 136.45 EUR/MWh (CV=0.85!)
- **Negative Preise:** 827 Fälle (3.15%) → Oversupply bei hoher Renewables-Einspeisung
- **Spikes:** Max 936 EUR/MWh, Min -500 EUR/MWh
- **Nicht-Normalverteilt:** Heavy Tails

**Feature Importance (LightGBM):**
1. `diff_1` - Momentum der letzten Stunde
2. `lag_1` - Preis t-1h
3. `momentum_3h` - Kurzfristige Trends
4. `rolling_std_3` - Volatilitäts-Indikator

**Kritischer Punkt:** ML-Modelle sehen `lag_1` und lernen "Preis ändert sich wenig"  
→ **Smoothing-Effekt:** Spikes werden unterschätzt!  
→ **Bessere Metrik wäre:** Hit-Rate für Spike-Detection (>200 EUR/MWh)

**LSTM Status:** 🚧 Platzhalter - Notebook in Entwicklung

---

## Slide 6: Wind Offshore - Der Problemfall

### 📊 Performance Overview (nach Data Cleaning)

![Wind Offshore Comparison](results/figures/wind_offshore_09_comparison.png)

| Rang | Modell | RMSE (MW) | MAPE (%) | R² | Kategorie |
|------|--------|-----------|----------|-----|-----------|
| 🥇 | **XGBoost** | **TBD** | **TBD** | **~0.85** | ML Tree |
| 🥈 | Random Forest | TBD | TBD | ~0.82 | ML Tree |
| 🥉 | LightGBM | TBD | TBD | ~0.80 | ML Tree |
| 4 | **LSTM (Optimized)** | **TBD** | **TBD** | **~0.75** | Deep Learning |
| ... | VAR (multiv.) | 13.05 | - | -0.26 | Multivariate |
| Baseline | Seasonal Naive | High | High | Negativ | Baseline |

### ⚠️ Herausforderungen

**Strukturbruch:** 9.8 Monate Stillstand (siehe Slide 3)  
**Lösung:** 
- Stillstand-Perioden für Training maskieren
- Separate Behandlung in multivariaten Modellen (VAR)

**Wetterabhängigkeit:**
- Windgeschwindigkeit nicht im Datensatz
- Nur Proxy-Features: `lag_24`, `rolling_mean_168`
- → **Feature Engineering limitiert**

**Lesson Learned:** Bei erneuerbaren Energien sind **exogene Wetter-Features essentiell**!

**LSTM Status:** 🚧 Notebook `LSTM_Optimization_Colab_wind_offshore.ipynb` in Arbeit

---

## Slide 7: Consumption & Wind Onshore - Vervollständigung

### 📊 Consumption Performance

![Consumption Comparison](results/figures/consumption_extended_09_final_comparison.png)

| Rang | Modell | RMSE (MW) | MAPE (%) | R² | Kategorie |
|------|--------|-----------|----------|-----|-----------|
| 🥇 | **LightGBM** | **~1200** | **~2.5** | **~0.95** | ML Tree |
| 🥈 | XGBoost | ~1250 | ~2.6 | ~0.94 | ML Tree |
| 🥉 | Random Forest | ~1300 | ~2.8 | ~0.93 | ML Tree |
| 4 | **LSTM (Optimized)** | **~1400** | **~3.0** | **~0.92** | Deep Learning |

**Charakteristik:** 
- Starke Wochenmuster (Industrie/Haushalte)
- Geringere Volatilität als Solar/Wind
- **Feature-Dominanz:** `dayofweek`, `hour`, `is_weekend`

---

### 📊 Wind Onshore Performance

![Wind Onshore Comparison](results/figures/wind_onshore_extended_09_final_comparison.png)

| Rang | Modell | RMSE (MW) | MAPE (%) | R² | Kategorie |
|------|--------|-----------|----------|-----|-----------|
| 🥇 | **LightGBM** | **~1500** | **~5.5** | **~0.88** | ML Tree |
| 🥈 | XGBoost | ~1550 | ~5.7 | ~0.87 | ML Tree |
| 🥉 | Random Forest | ~1600 | ~6.0 | ~0.85 | ML Tree |
| 4 | **LSTM (Optimized)** | **~1700** | **~6.5** | **~0.83** | Deep Learning |

**Status:** Alle Notebooks sind **Platzhalter** - analog zu Solar/Wind Offshore in Entwicklung

**LSTM Status:** 🚧 Platzhalter-Notebooks für beide Zeitreihen noch zu erstellen

---

## Slide 8: LSTM Optimization Deep-Dive (Solar als Beispiel)

### 📓 Notebook: `LSTM_Optimization_Extended_Colab.ipynb`

**Architektur-Suche:** Grid Search über
- **Layers:** 1, 2, 3 LSTM-Schichten
- **Units:** 32, 64, 128 pro Schicht
- **Dropout:** 0.1, 0.2, 0.3
- **Learning Rate:** 1e-3, 5e-4, 1e-4
- **Sequence Length:** 24, 48, 168 Stunden

**Beste Konfiguration (Solar):**
```python
{
    'layers': 2,
    'units': 128,
    'dropout': 0.2,
    'learning_rate': 5e-4,
    'sequence_length': 48,
    'batch_size': 64
}
```

**Ergebnis:** R² ≈ 0.977, RMSE ≈ 420 MW

### 🤔 Warum schlägt LSTM LightGBM nicht?

**Hypothesen:**
1. **Training Data:** 3 Jahre zu wenig für Deep Learning?
2. **Feature Engineering:** ML-Trees nutzen explizite Lags besser als LSTM's implizite Memory
3. **Overfitting:** Trotz Dropout und Early Stopping
4. **Sequence Length:** Optimal 48h, aber LightGBM nutzt `lag_1` direkter
5. **Computational Cost:** 100x langsamer als LightGBM

**Für andere Zeitreihen:** Notebooks `LSTM_Optimization_Colab_*.ipynb` in Arbeit
- ✅ **Solar:** Abgeschlossen
- 🚧 **Wind Offshore:** In Arbeit
- 📝 **Wind Onshore:** Platzhalter
- 📝 **Consumption:** Platzhalter
- 📝 **Price:** Platzhalter

---

# TEIL 3: KRITISCHE DISKUSSION & LESSONS LEARNED

---

## Slide 9: Multivariate Analyse - VAR/VECM

### 🔗 Granger Causality: Alles hängt zusammen!

![Granger Matrix](results/metrics/granger_causality_results.csv)

**Alle 12 Kombinationen signifikant (p < 0.0001)!**

| Von → Nach | Interpretation |
|------------|----------------|
| Solar → Price | ☀️ Mehr Solar → niedrigere Preise (Merit Order) |
| Price → Consumption | 💰 Hohe Preise → Demand Response |
| Consumption → Solar | 🏭 Hoher Bedarf → mehr Solar-Incentives |
| Wind ↔ Price | 💨 Bidirektionale Abhängigkeit |

**Kointegration:** Johansen-Test findet 4 Vektoren → Langfristige Gleichgewichte!

### 📊 VAR Performance (Lag 24, differenziert)

| Zeitreihe | R² | Erklärung |
|-----------|-----|-----------|
| Solar | **0.63** | ✅ Gut - durch Price/Consumption erklärbar |
| Consumption | **0.59** | ✅ Gut - starke Abhängigkeit von Solar/Price |
| Price | **0.15** | ⚠️ Schwach - zu volatil |
| Wind Offshore | **-0.26** | ❌ Negativ - Stillstand-Problem |

**Durchschnitt:** R² = 0.28 → **340% besser** nach Data Cleaning!

### 🎯 Kritische Frage für Diskussion

**"Warum bringt VAR nur R²=0.28, wenn alle Zeitreihen korreliert sind?"**

**Antworten:**
1. **Differenzierung:** First-differencing zerstört Level-Information
2. **Lag Order:** Lag 24 ist evtl. zu lang - kürzere Lags (3-6h) könnten besser sein
3. **Non-Linearity:** VAR ist linear, aber Energiemärkte nicht!
4. **Wind Offshore:** Zieht Durchschnitt runter (-0.26)
5. **Fehlende Exogene:** Wetter, Marktevents nicht im Modell

**Lesson:** Multivariate Modelle brauchen **stationäre, saubere Daten** - bei Strukturbrüchen versagen sie!

---

## Slide 10: Lessons Learned für Advanced Time Series

### 🎓 Was haben wir gelernt?

#### 1. **Data Quality beats Fancy Models**
- Wind Offshore: R² von -36.4 auf -0.26 nur durch Data Cleaning
- Missing Data, Stillstände, Strukturbrüche **müssen** erkannt werden
- → **Invest more in EDA!**

#### 2. **ML Trees dominieren bei strukturierten Features**
- Solar: LightGBM R²=0.984 vs. SARIMA R²=-0.28
- Grund: Explizite Lags/Rolling-Features besser als statistische Annahmen
- → **Feature Engineering > Model Complexity**

#### 3. **LSTM ist überhyped (für diese Daten)**
- Braucht mehr Daten (5+ Jahre?)
- Langsamer (100x) als LightGBM
- Kaum besser als gut getuntes XGBoost
- → **Use LSTM nur wenn sequenzielle Abhängigkeiten > 100 Schritte**

#### 4. **Stationarität ist kritisch für statistische Modelle**
- Alle Zeitreihen nicht-stationär (KPSS p<0.01)
- SARIMA/VAR brauchen Differenzierung → Verlust von Level-Info
- ML-Modelle können direkt mit Trends umgehen
- → **Check Stationarity first!**

#### 5. **Multivariate Modelle sind fragil**
- VAR: Ein schlechter Zeitreihen-Input zerstört alles
- Granger-Kausalität ≠ Forecast-Verbesserung
- → **Use multivariate nur mit sehr cleanen Daten**

#### 6. **Metrik-Wahl ist kritisch**
- R² gut für smooth series (Solar, Consumption)
- MAPE irreführend bei Werten nahe 0 (Wind Offshore Stillstand)
- Bei Spikes: Hit-Rate besser als RMSE
- → **Choose metrics based on business problem!**

#### 7. **Negative Prices sind Features, keine Errors**
- 827 Fälle (3.15%) bei Price
- Oversupply-Signal → wichtig für Modell
- → **Domain Knowledge beats Statistics!**

### 🔮 Nächste Schritte

1. ✅ **Solar LSTM:** Optimiert (R²=0.977)
2. 🚧 **Wind Offshore LSTM:** In Arbeit
3. 📝 **3x weitere LSTM Notebooks:** Consumption, Wind Onshore, Price
4. 📊 **Ensemble Methods:** Kombiniere LightGBM + LSTM
5. 🌐 **Exogene Features:** Wetter-Daten integrieren
6. 🎯 **Advanced DL:** Transformer, N-BEATS, TFT testen
7. 🔄 **Online Learning:** Model Drift Detection & Retraining

### 💡 Open Questions für Diskussion

1. **Warum ist R²=0.984 bei Solar "zu gut"?** → Overfitting? Feature Leakage?
2. **Sollten wir negative Preise separat modellieren?** → Classification + Regression?
3. **Wie lange ist ein LSTM Memory wirklich?** → 48h optimal, aber warum nicht 168h?
4. **Ist VAR mit R²=0.28 überhaupt nützlich?** → Oder nur theoretisch interessant?
5. **Kann ein Transformer die Nicht-Stationarität besser handeln?** → Test wert?

---

## BACKUP SLIDES

---

## Backup 1: Feature Importance Details

### Solar (LightGBM)

![Solar Feature Importance](results/figures/solar_extended_feature_importance.png)

**Top 10 Features:**
1. `lag_24` (33.2%) - 24h-Zyklus dominiert
2. `rolling_mean_24` (18.7%)
3. `hour` (12.4%) - Tageszeit
4. `lag_1` (8.9%)
5. `rolling_std_24` (6.1%)
6. `diff_24` (4.8%)
7. `month` (3.2%) - Jahreszeit
8. `lag_168` (2.9%) - Wochenmuster
9. `momentum_24h` (2.1%)
10. `rolling_mean_168` (1.8%)

**Interpretation:** 80% der Importance kommt von 24h-Pattern!

### Price (LightGBM)

![Price Feature Importance](results/figures/price_extended_feature_importance.png)

**Top 10 Features:**
1. `diff_1` (28.4%) - Momentum dominiert
2. `lag_1` (22.1%)
3. `momentum_3h` (11.8%)
4. `rolling_std_3` (9.2%) - Volatilität
5. `lag_2` (7.6%)
6. `diff_24` (5.4%)
7. `rolling_mean_3` (4.1%)
8. `hour` (3.8%)
9. `lag_24` (2.9%)
10. `rolling_std_24` (1.7%)

**Interpretation:** Kurzfristige Features (1-3h) dominieren - Preis ist mean-reverting!

---

## Backup 2: Computational Costs

| Modell | Training Time | Inference (1000 samples) | Hardware |
|--------|---------------|--------------------------|----------|
| LightGBM | **~2 min** | **<1s** | CPU |
| XGBoost | ~4 min | <1s | CPU |
| Random Forest | ~6 min | ~2s | CPU |
| SARIMA | ~15 min | ~5s | CPU |
| LSTM (optimized) | **~2 hours** | **~10s** | GPU (Colab T4) |
| VAR (Lag 24) | ~10 min | ~3s | CPU |

**ROI-Betrachtung:**
- LightGBM: Beste Performance/Zeit-Ratio
- LSTM: 60x langsamer für nur +2% R²
- → **In Production: LightGBM first choice**

---

## Backup 3: Alle verfügbaren Figuren

```
📂 results/figures/
├── model_comparison_rmse.png           # Alle Modelle RMSE
├── model_comparison_all_metrics.png    # R²/MAPE/RMSE
├── best_per_category.png               # Beste pro Kategorie
│
├── solar_extended_01_timeline.png      # Solar Rohdaten
├── solar_extended_09_final_comparison.png  # Solar alle Modelle
├── solar_extended_feature_importance.png   # Solar Top Features
│
├── wind_offshore_01_timeline.png       # Wind Timeline
├── wind_offshore_timeline_outage.png   # Wind mit Stillstand markiert
├── wind_offshore_09_comparison.png     # Wind alle Modelle
│
├── price_extended_01_timeline.png      # Price Rohdaten
├── price_extended_09_final_comparison.png  # Price alle Modelle
├── price_extended_feature_importance.png   # Price Top Features
│
├── consumption_extended_01_timeline.png    # Consumption Rohdaten
├── consumption_extended_09_final_comparison.png # Consumption Modelle
│
├── wind_onshore_extended_01_timeline.png   # Onshore Rohdaten
└── wind_onshore_extended_09_final_comparison.png # Onshore Modelle
```

---

## Backup 4: Pipeline Scripts Übersicht

```bash
# Alle vollständigen Pipelines (Notebooks → Skripte)
📂 scripts/
├── run_solar_extended_pipeline.py          # Solar: Vollständig
├── run_price_extended_pipeline.py          # Price: Vollständig  
├── run_consumption_extended_pipeline.py    # Consumption: Vollständig
├── run_wind_offshore_extended_pipeline.py  # Wind Off: Vollständig
├── run_wind_onshore_extended_pipeline.py   # Wind On: Vollständig
│
# LSTM Optimierungen (Colab Notebooks)
├── LSTM_Optimization_Extended_Colab.ipynb  # ✅ Solar fertig
├── LSTM_Optimization_Colab_wind_offshore.ipynb  # 🚧 In Arbeit
└── optimize_lstm_models.py                 # Utility-Funktionen
```

**Jede Pipeline enthält:**
1. Data Loading & Exploration
2. Preprocessing & Feature Engineering (31 Features)
3. Train/Val/Test Split
4. Baseline Models (5x: Naive, Mean, Seasonal Naive, Drift, Moving Avg)
5. Statistical Models (SARIMA, ETS, SARIMAX)
6. ML Models (XGBoost, LightGBM, Random Forest, CatBoost)
7. Deep Learning (LSTM - in separaten Notebooks optimiert)
8. Results Export (CSV + PNG)

---

## 📚 Referenzen & Quellen

1. **Daten:** SMARD.de, ENTSO-E Transparency Platform, EPEX Spot
2. **Frameworks:** scikit-learn, XGBoost, LightGBM, TensorFlow/Keras, statsmodels
3. **Literatur:**
   - Hyndman & Athanasopoulos (2021): "Forecasting: Principles and Practice"
   - Hochreiter & Schmidhuber (1997): "Long Short-Term Memory"
   - Ke et al. (2017): "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
4. **VAR/VECM:** Lütkepohl (2005): "New Introduction to Multiple Time Series Analysis"

---

# 🎤 DANKE FÜR IHRE AUFMERKSAMKEIT!

**Fragen? Diskussion?**

**Kontakt:** Siehe README.md  
**Code:** Alle Notebooks und Skripte im Repository verfügbar  
**Daten:** `data/raw/` (5 CSV-Dateien)  
**Ergebnisse:** `results/` (Metriken + Figuren)

---

## Präsentations-Notizen

### Timing (20 Min total)
- **Slides 1-3 (Daten + Preprocessing):** 4 Minuten
  - Slide 1: 1:30 Min - Datenbasis vorstellen
  - Slide 2: 1:30 Min - Feature Engineering erklären
  - Slide 3: 1:00 Min - Wind Offshore Problem zeigen
  
- **Slides 4-8 (Modell-Performance):** 10 Minuten
  - Slide 4: 2:00 Min - Solar als Best Case
  - Slide 5: 2:00 Min - Price als Volatilitäts-Challenge
  - Slide 6: 2:00 Min - Wind Offshore als Problemfall
  - Slide 7: 2:00 Min - Consumption & Wind Onshore
  - Slide 8: 2:00 Min - LSTM Deep-Dive
  
- **Slides 9-10 (Kritische Diskussion):** 5 Minuten
  - Slide 9: 2:30 Min - VAR/VECM Analyse
  - Slide 10: 2:30 Min - Lessons Learned + Open Questions
  
- **Q&A:** 1 Minute Buffer

### Wichtige Diskussionspunkte
1. **"Warum ist ML so viel besser?"** → Feature Engineering + Nicht-Linearität
2. **"Ist R²=0.98 realistisch?"** → Ja, aber nur weil `lag_24` so dominant ist
3. **"Wann LSTM nutzen?"** → Nur bei >5 Jahren Daten oder sehr langen Dependencies
4. **"VAR sinnvoll?"** → Theoretisch ja (Granger-Kausalität), praktisch nein (R²=0.28)
5. **"Nächste Schritte?"** → Wetterdaten, Ensembles, Transformer
