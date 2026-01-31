# 📚 Notebooks Übersicht

Diese Notebooks-Sammlung enthält eine vollständige Zeitreihenanalyse für 5 verschiedene Energie-Zeitreihen.

## 📁 Ordnerstruktur

```
notebooks/
├── solar/              ✅ 9 Notebooks (KOMPLETT)
├── wind_offshore/      ✅ 5 Notebooks (KOMPLETT)
├── wind_onshore/       ⏳ 0/6 Notebooks (IN ARBEIT)
├── consumption/        ⏳ 0/6 Notebooks (GEPLANT)
├── price/              ⏳ 0/6 Notebooks (GEPLANT)
└── cross_series/       ✅ 7 Notebooks (KOMPLETT)
```

## 🎯 Status pro Energy Type

### ✅ Solar (Referenz-Pipeline)
Vollständige Pipeline mit allen Modelltypen:
1. `01_data_exploration.ipynb` - Datenanalyse & Quality Check
2. `02_data_preprocessing.ipynb` - Feature Engineering
3. `03_baseline_models.ipynb` - Einfache Benchmarks
4. `04_statistical_models.ipynb` - SARIMA, ETS
5. `05_ml_tree_models.ipynb` - XGBoost, LightGBM, CatBoost
6. `06_deep_learning_models.ipynb` - LSTM, GRU, BiLSTM
7. `07_generative_models.ipynb` - Autoencoder, VAE, GAN
8. `08_advanced_models.ipynb` - Weitere Experimente
9. `09_model_comparison.ipynb` - Gesamtvergleich

**Best Model:** BiLSTM R²=0.9988

---

### ✅ Wind Offshore (Data Quality Case Study)
Demonstriert Umgang mit Datenqualitätsproblemen:
1. `01_data_exploration.ipynb` - **9-Monats-Outage entdeckt!**
2. `02_preprocessing.ipynb` - Outage-Removal & Cleaning
3. `03_baseline_models.ipynb` - Mean R²=-0.003 (beste Baseline)
4. `04_statistical_models.ipynb` - SARIMA versagt (R²=-8.02)
5. `05_ml_tree_models.ipynb` - LightGBM rettet es (R²=0.9997)

**Best Model:** LightGBM R²=0.9997 ⭐ (schlägt Solar!)

**Key Insight:** Rolling features dominieren bei Wind (rolling_mean_3 wichtigster)

---

### ⏳ Wind Onshore (GEPLANT)
Zu erstellen:
1. `01_data_exploration.ipynb`
2. `02_preprocessing.ipynb`
3. `03_baseline_models.ipynb`
4. `04_statistical_models.ipynb`
5. `05_ml_tree_models.ipynb`
6. `06_deep_learning.ipynb` (optional)

**Erwartung:** R²=0.980-0.995 (volatiler als Offshore)

---

### ⏳ Consumption (GEPLANT)
Zu erstellen:
1. `01_data_exploration.ipynb`
2. `02_preprocessing.ipynb`
3. `03_baseline_models.ipynb`
4. `04_statistical_models.ipynb`
5. `05_ml_tree_models.ipynb`
6. `06_deep_learning.ipynb`

**Erwartung:** R²>0.99 (starke Tag/Nacht & Wochenend-Muster)

---

### ⏳ Price (GEPLANT)
Zu erstellen:
1. `01_data_exploration.ipynb`
2. `02_preprocessing.ipynb`
3. `03_baseline_models.ipynb`
4. `04_statistical_models.ipynb`
5. `05_ml_tree_models.ipynb`
6. `06_deep_learning.ipynb`

**Erwartung:** R²=0.85-0.92 (am schwierigsten, Spikes & negative Preise)

---

### ✅ Cross-Series (Meta-Analysen)
Series-übergreifende Experimente:
1. `10_multi_series_analysis.ipynb` - Vergleich aller 5 mit XGBoost
2. `11_xgboost_tuning.ipynb` - Hyperparameter-Optimierung
3. `12_llm_time_series_models.ipynb` - Chronos, TimeGPT
4. `13_ensemble_methods.ipynb` - XGBoost + LSTM + Chronos
5. `14_multivariate_forecasting.ipynb` - VAR, Multi-LSTM
6. `15_external_weather_features.ipynb` - Wetter-API Integration
7. `16_chronos_finetuning.ipynb` - Foundation Model Finetuning

---

## 🎓 Standard-Pipeline pro Energy Type

Jedes Energy Type folgt dieser bewährten Struktur:

### 📊 01: Data Exploration
- Timeline-Visualisierung
- Data Quality Checks (Nullen, Outliers, Lücken)
- Statistische Eigenschaften (Mean, Std, CV)
- Zeitliche Muster (Stündlich, Täglich, Wöchentlich)
- Autokorrelationsanalyse
- Optimale Testperioden-Auswahl

### 🔧 02: Preprocessing
- Missing Value Handling
- Outlier Detection/Removal
- Feature Engineering (46 Standard-Features)
  - Zeitliche: hour, dayofweek, month, etc.
  - Zyklisch: hour_sin/cos, month_sin/cos
  - Lag: 1, 2, 3, 6, 12, 24, 48, 72, 168h
  - Rolling: mean, std, min, max (3, 6, 12, 24, 168h)
  - Differencing: diff_1, diff_24, diff_168
- Train/Val/Test Split
- StandardScaler
- Speichern für downstream-Nutzung

### 📏 03: Baseline Models
- Naive Forecast
- Seasonal Naive (24h)
- Moving Average (168h)
- Drift Method
- Mean Forecast
→ Schwellwerte für spätere Modelle

### 📈 04: Statistical Models
- Auto-ARIMA (Parametersuche)
- SARIMA (manuelle Parameter)
- ETS (Exponential Smoothing)
- Residualanalyse
→ Benchmark für ML-Modelle

### 🌳 05: ML Tree Models
- Random Forest (Baseline)
- XGBoost (meist bestes ML-Modell)
- LightGBM (schnell & effizient)
- CatBoost (kategorische Features)
- Feature Importance Analysis
- Error Analysis
→ Production-Ready Models

### 🧠 06: Deep Learning
- LSTM (Standard)
- GRU (schneller als LSTM)
- BiLSTM (beste Performance)
- Hyperparameter Tuning
- Early Stopping
- Vergleich zu ML
→ State-of-the-Art Performance

---

## 📊 Erwartete Ergebnisse

| Energy Type    | Data Quality | Expected Best R² | Best Model Expected | Difficulty |
|----------------|--------------|------------------|---------------------|------------|
| Solar          | ⭐⭐⭐⭐⭐      | 0.995-0.999      | BiLSTM              | 🟢 Einfach  |
| Wind Offshore  | ⭐⭐⭐⭐        | 0.995-0.999      | LightGBM            | 🟡 Mittel   |
| Wind Onshore   | ⭐⭐⭐⭐        | 0.980-0.995      | XGBoost/BiLSTM      | 🟡 Mittel   |
| Consumption    | ⭐⭐⭐⭐⭐      | 0.990-0.998      | LSTM/XGBoost        | 🟢 Einfach  |
| Price          | ⭐⭐⭐         | 0.850-0.920      | LightGBM            | 🔴 Schwer   |

---

## 🚀 Wie man ein Notebook ausführt

### Option 1: Jupyter Lab
```bash
cd /workspaces/AdvancedTimeSeriesPrediction/energy-timeseries-project
jupyter lab
# Navigate to notebooks/solar/01_data_exploration.ipynb
```

### Option 2: VS Code
```bash
# Öffne VS Code in diesem Workspace
# Navigate to notebooks/solar/01_data_exploration.ipynb
# Klick "Run All"
```

### Option 3: Command Line
```bash
cd notebooks/solar
jupyter nbconvert --execute --to notebook --inplace 01_data_exploration.ipynb
```

---

## 📦 Dependencies

Siehe `requirements.txt` im Root-Ordner:
```
pandas>=3.0.0
numpy>=2.0.0
matplotlib>=3.10.0
seaborn>=0.13.0
scikit-learn>=1.8.0
xgboost>=3.0.0
lightgbm>=4.0.0
catboost>=1.2.0
tensorflow>=2.20.0
torch>=2.10.0
statsmodels>=0.14.0
pmdarima>=2.1.0
plotly>=6.0.0
```

---

## 📈 Progressverfolgung

### Abgeschlossen ✅
- [x] Solar: 9/9 Notebooks
- [x] Wind Offshore: 5/5 Notebooks
- [x] Cross-Series: 7/7 Notebooks

### In Arbeit ⏳
- [ ] Wind Onshore: 0/6 Notebooks
- [ ] Consumption: 0/6 Notebooks
- [ ] Price: 0/6 Notebooks

### Gesamt
**Fortschritt:** 21/39 Notebooks (54%)

---

## 🎯 Nächste Schritte

1. **Wind Onshore erstellen** (Priorität 1)
   - Dauer: ~90 min
   - Notebooks 01-06

2. **Consumption erstellen** (Priorität 2)
   - Dauer: ~90 min
   - Notebooks 01-06

3. **Price erstellen** (Priorität 3)
   - Dauer: ~90 min
   - Notebooks 01-06

4. **Multi-Series Update** (Priorität 4)
   - Dauer: ~30 min
   - Update cross_series/10_multi_series_analysis.ipynb
   - Alle 5 Energy Types × 4 Best Models Matrix

---

## 📝 Lessons Learned

### Von Solar gelernt:
- BiLSTM erreicht beste Performance (R²=0.9988)
- Starke tägliche Muster machen Vorhersage einfach
- XGBoost bereits exzellent (R²=0.9838)

### Von Wind Offshore gelernt:
- Data Quality Issues sind kritisch (9-Monats-Outage)
- Statistische Modelle scheitern bei Volatilität (SARIMA R²=-8.02)
- LightGBM rettet es (R²=0.9997) - sogar besser als Solar!
- Rolling Features dominieren (rolling_mean_3 wichtigster)

### Erwartungen für kommende Analysen:
- **Wind Onshore:** Ähnlich wie Offshore aber volatiler
- **Consumption:** Sollte ähnlich gut wie Solar performen
- **Price:** Wird herausfordernd (Spikes, negative Werte)

---

## 🏆 Projekt-Ziele

1. **Vollständigkeit:** Alle 5 Energy Types vollständig analysiert
2. **Vergleichbarkeit:** Gleiche Methodik für alle
3. **Best Practices:** Template-artige, reproduzierbare Struktur
4. **Insights:** Welcher Energy Type ist am einfachsten/schwersten?
5. **Production-Ready:** Einsatzbereite Forecasting-Pipeline

---

**Erstellt am:** 31. Januar 2026  
**Status:** 54% komplett (21/39 Notebooks)  
**Letztes Update:** Wind Offshore Notebooks 04-05 abgeschlossen
