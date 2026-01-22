# Energie-Zeitreihen-Analyse & -Vorhersage

**Projektarbeit: Advanced Time Series Prediction**

> Anwendung verschiedener Zeitreihen-Vorhersagemethoden auf deutsche Energiedaten zur Identifikation der optimalen Methode

## 📊 Übersicht

Dieses Projekt analysiert Energiezeitreihen der deutschen Stromversorgung mit verschiedenen State-of-the-Art Forecasting-Methoden:

- **Datenquelle:** [SMARD](https://www.smard.de/home) (Bundesnetzagentur)
- **Zeitreihen:** Stromerzeugung (Solar, Wind), Verbrauch, Preise
- **Zeitraum:** 2022-2024 (3 Jahre stündliche Daten)
- **Ziel:** Vergleich von statistischen, ML- und Deep-Learning-Modellen

## 🗂️ Projektstruktur

```
energy-timeseries-project/
├── data/
│   ├── raw/              # Rohdaten von SMARD API (gecached)
│   ├── processed/        # Aufbereitete Daten
│   └── external/         # Zusätzliche Daten (Wetter, Feiertage)
├── notebooks/
│   ├── 01_data_exploration.ipynb       # ✅ Explorative Datenanalyse
│   ├── 02_data_preprocessing.ipynb     # ✅ Datenaufbereitung
│   ├── 03_baseline_models.ipynb        # ✅ Baseline-Modelle
│   ├── 04_statistical_models.ipynb     # ✅ SARIMA, ETS
│   ├── 05_ml_tree_models.ipynb         # ✅ XGBoost, LightGBM
│   ├── 06_deep_learning_models.ipynb   # ✅ LSTM, GRU
│   ├── 07_generative_models.ipynb      # ✅ VAE, GAN, DeepAR (Week08)
│   ├── 08_advanced_models.ipynb        # ✅ TFT, N-BEATS
│   └── 09_model_comparison.ipynb       # ✅ Model-Vergleich
├── src/
│   ├── data/             # Daten-Loading & Preprocessing
│   ├── models/           # Model-Implementierungen
│   ├── visualization/    # Plotting-Funktionen
│   └── evaluation/       # Metriken & Evaluation
├── results/
│   ├── figures/          # Plots & Visualisierungen
│   └── metrics/          # Performance-Metriken
├── README.md
├── requirements.txt
└── PROJEKTPLAN_ENERGIE_ZEITREIHEN.md   # Detaillierter Plan
```

## 🚀 Quick Start

### 1. Installation

```bash
# Repository klonen
cd c:\Users\Christian\Coding\AdvancedTimeSeriesPrediction
cd energy-timeseries-project

# Virtual Environment erstellen (empfohlen)
python -m venv venv
.\venv\Scripts\activate  # Windows

# Dependencies installieren
pip install -r requirements.txt

# Jupyter Notebook starten
jupyter notebook
```

### 2. Erste Schritte

Öffne `notebooks/01_data_exploration.ipynb` und führe die Zellen aus!

Das Notebook wird:
- ✅ Daten von SMARD API laden (automatisches Caching)
- ✅ Explorative Datenanalyse durchführen
- ✅ Saisonalität & Trends visualisieren
- ✅ Stationaritätstests durchführen

## 📈 Geplante Modelle

### Baseline
- ✅ Naive Forecast
- ✅ Seasonal Naive
- ✅ Moving Average

### Statistische Modelle
- ✅ ARIMA
- ✅ SARIMA
- ✅ SARIMAX (mit exogenen Variablen)
- ✅ ETS (Exponential Smoothing)

### Machine Learning
- ✅ XGBoost
- ✅ LightGBM
- ✅ CatBoost
- ✅ Random Forest

### Deep Learning - Grundlagen
- ✅ LSTM (Long Short-Term Memory)
- ✅ GRU (Gated Recurrent Unit)
- ✅ Bidirectional LSTM

### Generative Models (Week08) ✅
- ✅ Autoencoders für Anomalie-Erkennung
- ✅ VAEs (Variational Autoencoders)
- ✅ GANs (Generative Adversarial Networks)
- ✅ DeepAR (Probabilistische Vorhersagen)

### Deep Learning - Advanced
- ✅ Temporal Fusion Transformer (TFT)
- ✅ N-BEATS
- ✅ N-HiTS
- 🔄 xLSTM (optional)

### Generative Models (Week08) ✅
- ✅ Autoencoders für Anomalie-Erkennung
- ✅ VAEs (Variational Autoencoders)
- ✅ GANs (Generative Adversarial Networks)
- ✅ DeepAR (Probabilistische Vorhersagen)

### Cutting Edge (optional)
- 🔄 Time Series Foundation Models
- 🔄 Graph Neural Networks

## 📊 Evaluation-Metriken

Alle Modelle werden verglichen anhand von:

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R² Score**
- **Trainingszeit**
- **Inferenzzeit**

## 🔧 Verwendete Tools & Libraries

### Daten & Preprocessing
- `pandas`, `numpy`, `scipy`
- `sklearn.preprocessing`
- `holidays` (deutsche Feiertage)

### Visualisierung
- `matplotlib`, `seaborn`, `plotly`

### Statistische Modelle
- `statsmodels` (ARIMA, SARIMA)
- `pmdarima` (auto_arima)

### Machine Learning
- `scikit-learn`
- `xgboost`, `lightgbm`, `catboost`

### Deep Learning
- `pytorch`, `tensorflow`
- `darts` (Forecasting-Framework)
- `pytorch-forecasting` (TFT)
- `neuralforecast` (N-BEATS, N-HiTS)

### Optimierung
- `optuna` (Hyperparameter-Tuning)

## 📚 Datenquellen

### Primäre Quelle: SMARD
API der Bundesnetzagentur: https://www.smard.de/home

**Verfügbare Zeitreihen:**
- ✅ Photovoltaik-Erzeugung
- ✅ Wind Onshore
- ✅ Wind Offshore
- ✅ Stromverbrauch Deutschland
- ✅ Day-Ahead Strompreise
- ✅ Gesamterzeugung
- Und weitere...

**Auflösung:** Stündlich, täglich, wöchentlich, monatlich

### Alternative Quellen
- [Energy-Charts](https://www.energy-charts.info/?l=de&c=DE) (Fraunhofer ISE)
- [Bundesnetzagentur Datenportal](https://www.bundesnetzagentur.de/DE/Fachthemen/Datenportal/start.html)

### Externe Daten (optional)
- Wetterdaten: [Open-Meteo](https://open-meteo.com/)
- Feiertage: Python `holidays` Library

## 🎯 Projektziele

1. **Datenverständnis:** Tiefe explorative Analyse der Energiedaten
2. **Methodenvergleich:** Systematischer Vergleich verschiedener Ansätze
3. **Best Practice:** Reproduzierbare, gut dokumentierte Analyse
4. **Praktische Relevanz:** Erkenntnisse für den Energiesektor
5. **Technische Tiefe:** Anwendung fortgeschrittener Methoden

## 📝 Nächste Schritte

- [x] Projektstruktur erstellen
- [x] SMARD API-Integration
- [x] Erstes Explorations-Notebook
- [x] Datenaufbereitung & Feature Engineering
- [x] Train/Test/Validation Split
- [x] Baseline-Modelle implementieren
- [x] Statistische Modelle (SARIMA)
- [x] ML-Modelle (XGBoost etc.)
- [x] Deep Learning (LSTM, GRU)
- [x] Advanced Models (N-BEATS, TFT)
- [x] Model-Comparison & Ensembles
- [x] Alle 8 Notebooks erstellt
- [ ] Notebooks ausführen und Ergebnisse generieren
- [ ] Hyperparameter-Tuning
- [ ] Finale Dokumentation & Visualisierung

## 🔗 Referenzen

### Kursmaterial
- [TimeSeriesPrediction Repository](../TimeSeriesPrediction)
- Week02: SARIMA, Week04: Trees, Week05: LSTM
- Week08: VAEs, GANs, DeepAR ✅
- Week09: Transformers, Week10: N-BEATS

### Erfolgreiche Projekte
- [Energy Timeseries Project](https://github.com/Timson1235/energy-timeseries-project) (VDE Prize Winner)
- [Solar Prediction](https://github.com/AnnaValentinaHirsch/solar-prediction)
- [German Energy Analysis](https://github.com/worldmansist/German-energy-Time-Series-analysis-)



## 📄 Lizenz

Dieses Projekt ist für Bildungszwecke erstellt.

---

**Status:** ✅ Alle 9 Notebooks erstellt - Bereit zum Ausführen!  
**Letzte Aktualisierung:** 2026-01-21
