# Projektplan: Energie-Zeitreihenanalyse und -Vorhersage

**Ziel:** Anwendung verschiedener Zeitreihen-Methoden auf Energiedaten zur Identifikation der optimalen Vorhersagemethode

**Zeitrahmen:** ca. 6-8 Wochen

---

## Phase 1: Vorbereitung & Datenakquise (Woche 1)

### 1.1 Datenquellenauswahl
Wähle **eine** der drei vorgeschlagenen Energiequellen:

- **[energy-charts.info](https://www.energy-charts.info/?l=de&c=DE)** - Fraunhofer ISE, sehr umfassend
- **[SMARD](https://www.smard.de/home)** - Bundesnetzagentur, gut strukturiert, API verfügbar
- **[Bundesnetzagentur Datenportal](https://www.bundesnetzagentur.de/DE/Fachthemen/Datenportal/start.html)** - Offiziell, detailliert

**Empfehlung:** Starte mit **SMARD** - hat gute API-Dokumentation und ist für Automatisierung geeignet.

### 1.2 Datenauswahl
Wähle eine **spezifische Zeitreihe**, z.B.:
- Stromerzeugung (Solar, Wind, konventionell)
- Stromverbrauch (Deutschland gesamt oder regional)
- Strompreise (Day-Ahead Market)
- CO2-Emissionen

**Wichtig:** Wähle Daten mit **mindestens 2-3 Jahren Historie** in hoher Auflösung (stündlich oder täglich).

### 1.3 Projektstruktur erstellen
```
energy-timeseries-project/
├── data/
│   ├── raw/              # Rohdaten
│   ├── processed/        # Aufbereitete Daten
│   └── external/         # Zusätzliche Daten (Wetter, Feiertage)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_statistical_models.ipynb
│   ├── 05_ml_tree_models.ipynb
│   ├── 06_deep_learning_models.ipynb
│   ├── 07_advanced_models.ipynb
│   └── 08_model_comparison.ipynb
├── src/
│   ├── data/             # Daten-Loading und -Preprocessing
│   ├── models/           # Model-Klassen
│   ├── visualization/    # Plotting-Funktionen
│   └── evaluation/       # Metriken und Evaluation
├── results/
│   ├── figures/
│   └── metrics/
├── README.md
└── requirements.txt
```

### 1.4 Erfolgreiche Referenzprojekte anschauen
- [energy-timeseries-project](https://github.com/Timson1235/energy-timeseries-project) (Winter 2024)
- [solar-prediction](https://github.com/AnnaValentinaHirsch/solar-prediction) (Summer 2023)
- [German-energy-Time-Series-analysis](https://github.com/worldmansist/German-energy-Time-Series-analysis-) (Summer 2025)

---

## Phase 2: Explorative Datenanalyse (Woche 2)

### 2.1 Daten laden und sichten
- [ ] API-Zugriff / Download implementieren
- [ ] Datenformat verstehen
- [ ] Zeitraum festlegen (z.B. 2020-2025)

### 2.2 Datenqualität prüfen
- [ ] Fehlende Werte identifizieren
- [ ] Ausreißer erkennen
- [ ] Datentypen validieren

### 2.3 Visualisierung & statistische Analyse
**Basierend auf: Week01_Data_Pre-Processing**

- [ ] Zeitreihen-Plot (gesamter Zeitraum)
- [ ] Saisonalität identifizieren (täglich, wöchentlich, jährlich)
- [ ] Trend-Analyse
- [ ] ACF/PACF-Plots (Autokorrelation)
- [ ] Stationaritäts-Test (ADF-Test, KPSS-Test)
- [ ] Verteilungsanalyse (Histogramm, Q-Q-Plot)

### 2.4 Feature Engineering (erste Ideen)
**Basierend auf: Week03_Time_Series_Features**

- [ ] Zeitbasierte Features (Stunde, Wochentag, Monat, Jahreszeit)
- [ ] Lag-Features (t-1, t-24, t-168 für stündliche Daten)
- [ ] Rolling-Statistics (Mean, Std über verschiedene Fenster)
- [ ] Feiertage als binäre Features

---

## Phase 3: Datenaufbereitung & Train/Test-Split (Woche 2-3)

### 3.1 Data Cleaning
**Basierend auf: Week01_Data_Pre-Processing**

- [ ] Fehlende Werte behandeln (Interpolation, Forward Fill)
- [ ] Ausreißer behandeln (Clipping, Smoothing)
- [ ] Zeitzone-Harmonisierung

### 3.2 Feature Engineering (vollständig)
**Basierend auf: Week03_Time_Series_Features**

- [ ] Alle relevanten Features erstellen
- [ ] Feature-Selektion (Korrelations-Analyse, Feature Importance)

### 3.3 Normalisierung/Skalierung
- [ ] MinMax-Scaler oder StandardScaler anwenden
- [ ] **Wichtig:** Nur auf Trainingsdaten fitten!

### 3.4 Train/Test/Validation-Split
- [ ] **Chronologischer Split** (nicht random!)
- [ ] Beispiel: 70% Train, 15% Validation, 15% Test
- [ ] Alternative: Rolling Window Cross-Validation

---

## Phase 4: Baseline-Modelle (Woche 3)

### 4.1 Naive Modelle
Zur Orientierung und Vergleich:
- [ ] **Naive Forecast** (letzter Wert wird wiederholt)
- [ ] **Seasonal Naive** (Wert von vor einer Saison)
- [ ] **Moving Average** (einfacher gleitender Durchschnitt)

### 4.2 Evaluation-Metriken definieren
- [ ] MAE (Mean Absolute Error)
- [ ] RMSE (Root Mean Squared Error)
- [ ] MAPE (Mean Absolute Percentage Error)
- [ ] R² Score
- [ ] Optional: Pinball Loss (für probabilistische Vorhersagen)

---

## Phase 5: Statistische Modelle (Woche 3-4)

**Basierend auf: Week02_(S)ARIMA(X) + GARCH**

### 5.1 ARIMA-Familie
- [ ] **ARIMA** (AutoRegressive Integrated Moving Average)
  - Grid-Search für (p, d, q) Parameter
  - AIC/BIC zur Modellauswahl
- [ ] **SARIMA** (Seasonal ARIMA)
  - Berücksichtigung saisonaler Komponenten
- [ ] **SARIMAX** (mit exogenen Variablen)
  - Z.B. Wetterdaten, Feiertage als exogene Variablen

### 5.2 Exponential Smoothing
- [ ] **ETS** (Error, Trend, Seasonality)
- [ ] **Holt-Winters** Methode

### 5.3 Optional: Volatilitätsmodelle
- [ ] **GARCH** (falls Volatilität wichtig ist, z.B. bei Preisen)

---

## Phase 6: Machine Learning - Tree-based Models (Woche 4)

**Basierend auf: Week04_Trees**

### 6.1 Einzelne Bäume
- [ ] **Decision Tree Regressor** (Baseline)

### 6.2 Ensemble-Modelle
- [ ] **Random Forest**
  - Hyperparameter-Tuning (n_estimators, max_depth, min_samples_split)
- [ ] **XGBoost**
  - Grid-Search oder RandomSearch
  - Feature Importance analysieren
- [ ] **LightGBM**
  - Besonders effizient bei großen Datenmengen
- [ ] **CatBoost**
  - Gut bei kategorischen Features

**Feature Engineering ist hier entscheidend!** Lag-Features, Rolling-Features, etc.

---

## Phase 7: Deep Learning - Sequenzmodelle (Woche 5)

### 7.1 Recurrent Neural Networks
**Basierend auf: Week05_RNNs_LSTM_GRU**

- [ ] **Simple RNN** (Baseline)
- [ ] **LSTM** (Long Short-Term Memory)
  - Multi-Layer LSTM
  - Bidirectional LSTM
- [ ] **GRU** (Gated Recurrent Unit)
  - Oft schneller als LSTM

**Hyperparameter:**
- Hidden units, Anzahl Layer, Dropout, Learning Rate, Batch Size

### 7.2 State-Space Models
**Basierend auf: Week06_State-Space-Models+LMU**

- [ ] **State-Space Models**
- [ ] **LMU** (Legendre Memory Unit)

---

## Phase 8: Advanced Deep Learning (Woche 6)

### 8.1 Attention-basierte Modelle
**Basierend auf: Week09_Transformers+TFTs**

- [ ] **Transformer** für Zeitreihen
- [ ] **TFT** (Temporal Fusion Transformer)
  - Besonders gut für multivariate Zeitreihen mit exogenen Variablen
  - Interpretierbare Attention-Mechanismen

### 8.2 Spezialisierte Architekturen
**Basierend auf: Week10_NBEATS+NHITS+xLSTMs**

- [ ] **N-BEATS** (Neural Basis Expansion Analysis)
  - Interpretierbare Trend- und Saisonalitätskomponenten
- [ ] **N-HiTS** (Hierarchical Interpolation)
  - Multi-Rate Sampling für lange Horizonte
- [ ] Optional: **xLSTM** (Extended LSTM)

### 8.3 Optional: Graph Neural Networks
**Basierend auf: Week11_Graphs+Networks** (falls räumliche Komponente relevant)

---

## Phase 9: Cutting-Edge (Optional, Woche 6-7)

### 9.1 Generative Models
**Basierend auf: Week08_Generative_Architectures_VAEs_GANs** ✅ 

Generative Modelle für Zeitreihen - besonders relevant für:
- Anomalie-Erkennung in Energiedaten
- Probabilistische Vorhersagen
- Szenario-Generierung

#### Autoencoders & Anomalie-Erkennung
- [ ] **Autoencoders** für Anomalie-Detektion
  - Reconstruction Error als Anomalie-Score
  - Besonders nützlich für Erkennung ungewöhnlicher Verbrauchsmuster

#### VAEs (Variational Autoencoders)
- [ ] **VAE** für probabilistische Modellierung
  - Latent Space Repräsentation
  - Generierung synthetischer Zeitreihen
  - Nützlich für: Data Augmentation, Worst-Case Szenarien

#### GANs (Generative Adversarial Networks)
- [ ] **Time Series GAN**
  - Generator + Discriminator Architektur
  - Wasserstein GAN für stabileres Training
  - Anwendung: Synthetische Energiedaten für Stresstests

#### DeepAR
- [ ] **DeepAR** (Amazon's probabilistisches Modell)
  - Autoregressive RNN mit probabilistischen Outputs
  - Quantile-Vorhersagen (P10, P50, P90)
  - Sehr gut für Unsicherheitsquantifizierung

**Praktische Anwendung im Energiesektor:**
- Erkennung von Netzschwankungen
- Generierung von Szenarien für Kapazitätsplanung
- Unsicherheitsquantifizierung für Risikoanalyse

### 9.2 Time Series LLMs
**Basierend auf: Week12_TimeSeriesLLMs**

Falls Zeit bleibt, kannst du experimentieren mit:
- [ ] Foundation Models für Zeitreihen (z.B. TimeGPT, Chronos)
- [ ] Pre-trained Models fine-tunen

---

## Phase 10: Model Comparison & Ensembles (Woche 7)

### 10.1 Umfassende Evaluation
- [ ] Alle Modelle auf Test-Set evaluieren
- [ ] Metriken-Tabelle erstellen (MAE, RMSE, MAPE, R², Trainingszeit)
- [ ] Residuen-Analyse für beste Modelle

### 10.2 Visualisierung
- [ ] Vorhersage vs. Actual für verschiedene Zeiträume
- [ ] Error-Distribution
- [ ] Feature Importance (für Tree- und TFT-Modelle)

### 10.3 Ensemble-Methoden
- [ ] **Simple Average** der besten 3-5 Modelle
- [ ] **Weighted Average** (basierend auf Validation-Performance)
- [ ] **Stacking** (Meta-Learner)

### 10.4 Horizont-Analyse
- [ ] Wie performen Modelle über verschiedene Vorhersage-Horizonte?
  - 1-Stunde, 24-Stunden, 7-Tage

---

## Phase 11: Dokumentation & Präsentation (Woche 8)

### 11.1 README.md erstellen
- [ ] Projektübersicht
- [ ] Datenquellen
- [ ] Installationsanleitung
- [ ] Reproduzierbarkeit (requirements.txt, seed-setting)
- [ ] Hauptergebnisse
- [ ] Visualisierungen einbinden

### 11.2 Jupyter Notebooks aufräumen
- [ ] Narrative Struktur
- [ ] Markdown-Erklärungen
- [ ] Code-Kommentare
- [ ] Klare Outputs und Visualisierungen

### 11.3 Ergebniszusammenfassung
Erstelle eine Tabelle wie:

| Modell | MAE | RMSE | MAPE | R² | Training Time | Inference Time |
|--------|-----|------|------|----|--------------|--------------------|
| Naive | ... | ... | ... | ... | ... | ... |
| SARIMA | ... | ... | ... | ... | ... | ... |
| XGBoost | ... | ... | ... | ... | ... | ... |
| LSTM | ... | ... | ... | ... | ... | ... |
| TFT | ... | ... | ... | ... | ... | ... |
| N-BEATS | ... | ... | ... | ... | ... | ... |
| **Ensemble** | ... | ... | ... | ... | ... | ... |

### 11.4 Visualisierungen
- [ ] Beste Vorhersagen visualisieren
- [ ] Feature Importance
- [ ] Learning Curves
- [ ] Residual Plots

### 11.5 Lessons Learned & Diskussion
- [ ] Was hat gut funktioniert?
- [ ] Wo waren Herausforderungen?
- [ ] Was sind praktische Implikationen für den Energiesektor?
- [ ] Limitierungen & Future Work

---

## Empfohlene Tools & Libraries

### Daten & Preprocessing
```python
pandas, numpy, scipy
sklearn.preprocessing
holidays  # für Feiertage
```

### Visualisierung
```python
matplotlib, seaborn, plotly
```

### Statistische Modelle
```python
statsmodels  # ARIMA, SARIMA, SARIMAX
pmdarima  # auto_arima
```

### Machine Learning
```python
scikit-learn
xgboost
lightgbm
catboost
```

### Deep Learning
```python
tensorflow / keras
pytorch
pytorch-forecasting  # TFT implementation
darts  # umfassendes Forecasting-Framework
neuralforecast  # N-BEATS, N-HiTS
```

### Evaluation & Utils
```python
sklearn.metrics
optuna  # für Hyperparameter-Tuning
mlflow  # für Experiment-Tracking (optional)
```

---

## Priorisierung - Falls Zeit knapp wird

### Must-Have (Minimum Viable Project)
1. ✅ Explorative Datenanalyse
2. ✅ Baseline-Modelle (Naive, MA)
3. ✅ SARIMA(X)
4. ✅ XGBoost oder LightGBM
5. ✅ LSTM
6. ✅ Model Comparison
7. ✅ Dokumentation

### Should-Have (Gutes Projekt)
- TFT (Temporal Fusion Transformer)
- N-BEATS
- Ensemble-Methoden
- Hyperparameter-Tuning
- Mehrere Vorhersage-Horizonte

### Nice-to-Have (Exzellentes Projekt)
- Graph Neural Networks (falls räumliche Komponente)
- Time Series LLMs
- Online-Dashboard (Streamlit/Dash)
- Deployment-ready Code
- Umfassende Ablation Studies

---

## Nächste Schritte - Heute starten!

### Schritt 1: Datenquelle identifizieren
- [ ] Öffne [SMARD](https://www.smard.de/home)
- [ ] Schaue dir die verfügbaren Daten an
- [ ] Entscheide: Stromerzeugung, Verbrauch oder Preise?

### Schritt 2: Projektstruktur erstellen
```bash
cd c:\Users\Christian\Coding\AdvancedTimeSeriesPrediction
mkdir energy-timeseries-project
cd energy-timeseries-project
# Erstelle Ordnerstruktur (siehe oben)
```

### Schritt 3: Erstes Notebook
- [ ] Erstelle `notebooks/01_data_exploration.ipynb`
- [ ] Lade erste Daten herunter
- [ ] Erstelle ersten Plot

---

## Tipps für Erfolg

1. **Starte einfach:** Baseline-Modelle sind wichtig für Vergleich
2. **Iteriere schnell:** Nicht zu lange an einem Modell hängen
3. **Dokumentiere kontinuierlich:** Nicht alles am Ende
4. **Nutze vorhandenen Code:** Schaue dir Beispiel-Notebooks im Repo an
5. **Reproduzierbarkeit:** Seeds setzen, requirements.txt pflegen
6. **Visualisiere viel:** Bilder sagen mehr als Zahlen
7. **Frage dich:** Was ist die praktische Relevanz meiner Ergebnisse?

---

## Ressourcen

- **Kurs-Material:** `TimeSeriesPrediction/Week*` Ordner
- **Frühere Projekte:** Siehe `Projects/README.md`
- **Datenquellen:** SMARD, energy-charts, Bundesnetzagentur
- **Frameworks:** `darts`, `pytorch-forecasting`, `neuralforecast`

---

**Viel Erfolg! 🚀**
