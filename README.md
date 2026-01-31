# Advanced Time Series Prediction

## 🎯 Aktueller Fokus: Umfassende Modell-Evaluation

Wir testen systematisch **alle verfügbaren Modelle** auf **5 Zeitreihen**:

1. **Solar** - Solarenergie-Erzeugung
2. **Wind Offshore** - Offshore-Windenergie  
3. **Wind Onshore** - Onshore-Windenergie
4. **Price** - Strompreise (Day-Ahead)
5. **Consumption** - Stromverbrauch

---

## 🚀 Quick Start

### Pipeline für eine Zeitreihe ausführen:

```bash
cd energy-timeseries-project

# Solar
python scripts/run_solar_extended_pipeline.py

# Wind Offshore
python scripts/run_wind_offshore_extended_pipeline.py

# Wind Onshore
python scripts/run_wind_onshore_extended_pipeline.py

# Price
python scripts/run_price_extended_pipeline.py

# Consumption
python scripts/run_consumption_extended_pipeline.py
```

### Alle Pipelines nacheinander:

```bash
cd energy-timeseries-project
for pipeline in scripts/run_*_extended_pipeline.py; do
    echo "Running $pipeline..."
    python "$pipeline"
done
```

---

## 📊 Was wird getestet?

Jede Pipeline durchläuft **9 Phasen**:

1. **Exploration** - Datenanalyse, Stationarität, Saisonalität
2. **Preprocessing** - Bereinigung, Feature Engineering
3. **Baselines** - Naive, Seasonal Naive, Moving Average, Drift, Mean
4. **Statistical** - ARIMA, SARIMA, Auto-ARIMA, ETS
5. **ML Trees** - Random Forest, XGBoost, LightGBM, CatBoost
6. **Deep Learning** - Simple RNN, LSTM, GRU
7. **Generative** - N-BEATS, N-HiTS
8. **Advanced** - Weitere fortgeschrittene Methoden
9. **Comparison** - Modellvergleich & Visualisierung

---

## 📁 Projektstruktur

```
energy-timeseries-project/
├── scripts/                           # 🎯 5 Pipeline-Skripte
│   ├── run_solar_extended_pipeline.py
│   ├── run_wind_offshore_extended_pipeline.py
│   ├── run_wind_onshore_extended_pipeline.py
│   ├── run_price_extended_pipeline.py
│   └── run_consumption_extended_pipeline.py
├── src/                               # Wiederverwendbare Module
│   ├── data/preprocessing.py
│   ├── models/baseline.py, statistical.py, ml_models.py, deep_learning.py
│   └── evaluation/metrics.py
├── data/
│   ├── raw/                          # Original CSVs
│   └── processed/                    # Verarbeitete Daten
├── results/                          # Metriken & Visualisierungen
│   ├── metrics/                      # CSV mit Modell-Scores
│   └── figures/                      # PNG-Plots
├── requirements.txt                  # Python Dependencies
└── archive/                          # 📦 Alte Entwicklungsartefakte
    ├── phase1_notebooks/             # Jupyter Notebooks
    ├── phase1_api_monitoring/        # API, Grafana, Docker
    ├── phase1_misc_scripts/          # Alte Scripts, Docs
    ├── old_scripts/                  # Debug-Scripts
    ├── old_docs/                     # Session-Logs
    └── old_root_files/               # Veraltete Root-Skripte
```

---

## 📈 Ergebnisse

Nach Ausführung einer Pipeline:
- **Metriken**: `results/metrics/{series}_all_models_extended.csv`
- **Visualisierungen**: `results/figures/{series}/`
  - Timeline-Plot
  - Train/Val/Test Split
  - Modellvergleich

---

## 🔄 Nächste Schritte

1. ✅ **Aktuell**: Alle 5 Zeitreihen mit allen Modellen testen
2. ⏳ **Dann**: Multivariate Methoden evaluieren
3. ⏳ **Später**: Ensemble-Methoden, Cross-Series Analysis

---

## 📚 Archiv

Alte Entwicklungsartefakte wurden archiviert:
- `energy-timeseries-project/archive/` - Notebooks, API, Monitoring, alte Scripts
- `archive_root/` & `archive_phase1_root/` - Alte Root-Dokumentationen

---

## 🛠 Tech Stack

- **Python**: 3.12
- **ML**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Deep Learning**: PyTorch, TensorFlow/Keras
- **Time Series**: statsmodels, pmdarima, Darts, NeuralForecast
- **Data**: pandas, numpy, scipy
- **Viz**: matplotlib, seaborn, plotly

---

**Status**: Aktive Entwicklung | **Stand**: Januar 2026
