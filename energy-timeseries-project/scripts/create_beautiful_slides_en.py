#!/usr/bin/env python3
"""Creates a beautiful HTML presentation with proper Markdown rendering - English Version"""

from pathlib import Path
import markdown
from markdown.extensions.tables import TableExtension
from markdown.extensions.fenced_code import FencedCodeExtension
import re

# Read the Markdown file and translate headers
input_file = Path("../VORTRAG_ADVANCED_TIME_SERIES.md")
with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Translate the main headers and content to English
translations = {
    # Main title
    "🎓 Advanced Time Series Forecasting für Energiemärkte": "🎓 Advanced Time Series Forecasting for Energy Markets",
    "Ein kritischer Vergleich von ML, DL und statistischen Methoden": "A Critical Comparison of ML, DL, and Statistical Methods",
    
    # Sections
    "Präsentationsdauer:": "Duration:",
    "Minuten": "Minutes",
    "Zielgruppe:": "Audience:",
    "Advanced Time Series Analysis Kurs": "Advanced Time Series Analysis Course",
    "Datum:": "Date:",
    "Februar": "February",
    
    # Agenda
    "Agenda": "Agenda",
    "TEIL 1: DATENBASIS & PREPROCESSING": "PART 1: DATA BASIS & PREPROCESSING",
    "TEIL 2: MODELL-PERFORMANCE NACH ZEITREIHEN": "PART 2: MODEL PERFORMANCE BY TIME SERIES",
    "TEIL 3: KRITISCHE DISKUSSION & LESSONS LEARNED": "PART 3: CRITICAL DISCUSSION & LESSONS LEARNED",
    
    # Slide sections
    "Datenbasis & Preprocessing": "Data Basis & Preprocessing",
    "Modell-Performance nach Zeitreihen": "Model Performance by Time Series",
    "Kritische Diskussion & Lessons Learned": "Critical Discussion & Lessons Learned",
    
    # Common terms
    "Slide": "Slide",
    "Datenbasis": "Data Basis",
    "Deutsche Energiemärkte": "German Energy Markets",
    "Fünf Zeitreihen, stündliche Auflösung": "Five Time Series, Hourly Resolution",
    "Zeitreihe": "Time Series",
    "Datenpunkte": "Data Points",
    "Zeitraum": "Period",
    "Quelle": "Source",
    "Einheit": "Unit",
    "Zeitreihen-Übersicht": "Time Series Overview",
    
    # Slide 3 - Model Portfolio
    "Modell-Portfolio": "Model Portfolio",
    "Modelle im Benchmark": "Models in Benchmark",
    "Getestete Modellarchitekturen": "Tested Model Architectures",
    "Wir haben": "We tested",
    "verschiedene Modelle": "different models",
    "über": "across",
    "Zeitreihen": "time series",
    "getestet": "tested",
    "Experimente": "experiments",
    "Modell-Kategorien": "Model Categories",
    
    # Model categories
    "Statistische Baseline-Modelle": "Statistical Baseline Models",
    "Klassische Zeitreihenanalyse": "Classical Time Series Analysis",
    "Univariat, Einfach": "Univariate, Simple",
    "Machine Learning Tree Models": "Machine Learning Tree Models",
    "Standard Python Pipeline": "Standard Python Pipeline",
    "Deep Learning Models - Standard": "Deep Learning Models - Standard",
    "Deep Learning Models - Generative": "Deep Learning Models - Generative",
    "Deep Learning Models - State-of-the-Art": "Deep Learning Models - State-of-the-Art",
    "Multivariate Modelle": "Multivariate Models",
    
    # Baseline models
    "Naive": "Naive",
    "Last Value": "Last Value",
    "Letzte Beobachtung wird fortgeschrieben": "Last observation carried forward",
    "Seasonal Naive": "Seasonal Naive",
    "Seasonal Last Value": "Seasonal Last Value",
    "Letzter Saisonzyklus": "Last seasonal cycle",
    "wird wiederholt": "is repeated",
    "Mean": "Mean",
    "Historical Average": "Historical Average",
    "Mittelwert der Trainings-Daten": "Mean of training data",
    "Seasonal ARIMA": "Seasonal ARIMA",
    "Saisonalität": "Seasonality",
    "Einfachste Baselines": "Simplest baselines",
    "Zeigen wie viel Komplexität bringt": "Show how much complexity brings",
    "Beschreibung": "Description",
    "Baseline Models": "Baseline Models",
    
    # Performance tables
    "Statistical": "Statistical",
    
    # Multivariate models
    "Vector Autoregression": "Vector Autoregression",
    "Vector Error Correction": "Vector Error Correction",
    "Nutzen Kausalität zwischen Zeitreihen": "Leverage causality between time series",
    "Resultat:": "Result:",
    "Schlechter als univariat": "Worse than univariate",
    
    # Table headers
    "Modell": "Model",
    "Typ": "Type",
    "Annahmen": "Assumptions",
    "Training Umgebung": "Training Environment",
    "Stärken": "Strengths",
    "Architektur": "Architecture",
    "Parameter": "Parameters",
    "Training Zeit": "Training Time",
    "Use Case": "Use Case",
    "Komplexität": "Complexity",
    "Paper": "Paper",
    "Spezialisierung": "Specialization",
    
    # Model types
    "Univariate Time Series": "Univariate Time Series",
    "Multivariate Vector AR": "Multivariate Vector AR",
    "Kointegration": "Cointegration",
    "Gradient Boosting": "Gradient Boosting",
    "Lokal (CPU)": "Local (CPU)",
    "Ensemble": "Ensemble",
    "Recurrent": "Recurrent",
    "vereinfacht": "simplified",
    "Bidirektional": "Bidirectional",
    "Encoder-Decoder": "Encoder-Decoder",
    "Variational": "Variational",
    
    # Assumptions and features
    "Stationarität": "Stationarity",
    "Linearität": "Linearity",
    "Lag-Struktur": "Lag structure",
    "Langfristige Gleichgewichte": "Long-term equilibria",
    "Feature-rich": "Feature-rich",
    "robust": "robust",
    "Schnell": "Fast",
    "memory-effizient": "memory-efficient",
    "Chaos-resistent": "Chaos-resistant",
    "keine Hyperparameter": "no hyperparameters",
    "Kategorische Features": "Categorical features",
    "Sequenzen": "Sequences",
    "Unidirektional": "Unidirectional",
    "schneller": "faster",
    "Symmetrische Patterns": "Symmetric patterns",
    "Feature Learning": "Feature Learning",
    "Probabilistisch": "Probabilistic",
    "Univariate Decomposition": "Univariate Decomposition",
    "Hierarchical Interpolation": "Hierarchical Interpolation",
    "Probabilistic Forecasting": "Probabilistic Forecasting",
    
    # Descriptions
    "Zweck:": "Purpose:",
    "Benchmark für ML/DL": "Benchmark for ML/DL",
    "Zeigen wie viel Komplexität bringt": "Show how much complexity brings",
    "Zeigen": "Show",
    "wie viel": "how much",
    "Komplexität bringt": "complexity brings",
    "Features:": "Features:",
    "engineered features": "engineered features",
    "lags": "lags",
    "rolling stats": "rolling stats",
    "temporal": "temporal",
    "Erwartung:": "Expectation:",
    "SOTA-Modelle sollten gewinnen": "SOTA models should win",
    "sollten gewinnen": "should win",
    "Tatsächlich:": "Actually:",
    "Alle negativ": "All negative",
    
    # Key insights
    "Wichtige Erkenntnisse": "Key Insights",
    "SOTA ≠ Best Performance": "SOTA ≠ Best Performance",
    "Alle 5 Zeitreihen negativ": "All 5 time series negative",
    "GPU ≠ Bessere Ergebnisse": "GPU ≠ Better Results",
    "schlägt": "beats",
    "Komplexität ≠ Accuracy": "Complexity ≠ Accuracy",
    "Training Time Paradox": "Training Time Paradox",
    "Schnellste Modelle": "Fastest models",
    "oft besser als langsamste": "often better than slowest",
    "Key Lesson:": "Key Lesson:",
    "Benchmarke IMMER selbst": "ALWAYS benchmark yourself",
    "Papers ≠ Production Reality": "Papers ≠ Production Reality",
    
    # More translations needed
    "Zeitreihen-Übersicht": "Time Series Overview",
    "für einzelne": "for individual",
    "Modele": "models",
    "profitieren massiv von": "benefit massively from",
    "sollten": "should",
    "gewinnen": "win",
    "negativee": "negative",
    "Fastste": "Fastest",
    "zeigt": "shows",
    "versagen spektakulär": "fail spectacularly",
    "separat modelliert": "modeled separately",
    "werden": "be",
    "signifikant": "significant",
    "auch nach Kontrolle für": "even after controlling for",
    "Tageszeit": "time of day",
    "Daysszeit": "time of day",
    "zu lang für kurzfristige": "too long for short-term",
    "Dynamik": "dynamics",
    "Univariatee": "Univariate",
    "mit guten": "with good",
    "für": "for",
    "Price-Prognose": "price prediction",
    "bettere": "better",
    "bauen": "build",
    "Alle": "All",
    "negative": "negative",
    "Default": "default",
    "neue": "new",
    "Warum versagen": "Why do fail",
    "SO konsisent": "SO consistently",
    "Fundamental fthanch": "Fundamentally wrong",
    "Energy": "energy",
    
    # More comprehensive translations
    "Herausforderungen": "Challenges",
    "Zeitreihen-Übersicht": "Time Series Overview",
    "Time Seriesn-Übersicht": "Time Series Overview",
    "Hohe Volatilität:": "High Volatility:",
    "Saisonalität:": "Seasonality:",
    "Strukturbrüche:": "Structural Breaks:",
    "Stillstand": "Outage",
    "Monate": "Months",
    "Negative Preise:": "Negative Prices:",
    "Fälle": "Cases",
    "Missing Data:": "Missing Data:",
    "hatte Datenlücken": "had data gaps",
    "Nicht-Stationarität:": "Non-Stationarity:",
    
    # Fix typos and compound words
    "Univariatee": "Univariate",
    "Complexity bringt": "complexity brings",
    "bettere": "better",
    "build": "build",
    "versagt konsisent bei energy": "consistently fails with energy",
    "versagt konsisent": "consistently fails",
    "konsisent": "consistently",
    "fail SOTA-models SO konsisent": "do SOTA models fail SO consistently",
    "Why do fail": "Why do",
    "SOTA-models": "SOTA models",
    "Fundamental fthanch for energy": "Fundamentally wrong for energy",
    "fthanch": "wrong",
    "energy Data": "energy data",
    "Optimierung": "optimization",
    "Alle Zeitreihen nicht-stationär": "All time series non-stationary",
    
    # Preprocessing
    "Preprocessing Pipeline": "Preprocessing Pipeline",
    "Von Rohdaten zu 31 Features": "From Raw Data to 31 Features",
    "Kritische Aufbereitungsschritte": "Critical Processing Steps",
    "Data Cleaning": "Data Cleaning",
    "Feature Engineering": "Feature Engineering",
    "Features pro Zeitreihe": "Features per Time Series",
    "Kategorien:": "Categories:",
    "Lags": "Lags",
    "Rolling Statistics": "Rolling Statistics",
    "Differenzen": "Differences",
    "Zeitliche Features": "Temporal Features",
    "Momentum": "Momentum",
    "Volatilität": "Volatility",
    "Warum so viele?": "Why so many?",
    "profitieren massiv von Features": "benefit massively from features",
    "der Performance": "of performance",
    "nutzt nur Rohdaten": "uses only raw data",
    "Train/Val/Test Split": "Train/Val/Test Split",
    "Temporale Trennung": "Temporal Split",
    "KEINE Random-Shuffle bei Zeitreihen!": "NO Random-Shuffle for time series!",
    "Wichtig:": "Important:",
    "für Production-Deployment": "for Production Deployment",
    
    # Data Quality
    "Data Quality Issues": "Data Quality Issues",
    "Der Wind Offshore Problemfall": "The Wind Offshore Problem Case",
    "Strukturbruch:": "Structural Break:",
    "Problem:": "Problem:",
    "Periode:": "Period:",
    "bis": "to",
    "Dauer:": "Duration:",
    "Tage": "Days",
    "Impact:": "Impact:",
    "der Daten sind Nullen oder Missing": "of data are zeros or missing",
    "Grund:": "Reason:",
    "Wartung oder technische Probleme": "Maintenance or technical problems",
    "nicht dokumentiert": "not documented",
    "Lösungsstrategien": "Solution Strategies",
    "Ignoriere Stillstand im Training": "Ignore Outage in Training",
    "Maskiere Nullen im Training-Set": "Mask zeros in training set",
    "Risiko:": "Risk:",
    "Modell kann keine Stillstände vorhersagen!": "Model cannot predict outages!",
    "Separate Outage-Prediction": "Separate Outage Prediction",
    "Binary Classifier": "Binary Classifier",
    "Läuft Anlage?": "Is plant running?",
    "Falls Ja": "If Yes",
    "Regressionsmodell für": "Regression model for",
    "Besser für Production!": "Better for production!",
    "Unsere Wahl:": "Our Choice:",
    "für Testing": "for testing",
    "für Production": "for production",
    "Impact auf Modelle": "Impact on Models",
    "mit Stillstand": "with outage",
    "bereinigt": "cleaned",
    "Punkte": "points",
    "Key Lesson:": "Key Lesson:",
    "Data Quality > Model Complexity!": "Data Quality > Model Complexity!",
    
    # Performance
    "Performance Overview": "Performance Overview",
    "Der DL Showcase": "The DL Showcase",
    "Beste Ergebnisse": "Best Results",
    "Charakteristik:": "Characteristics:",
    "Symmetrische Tagesverläufe": "Symmetric daily patterns",
    "Winter-Sommer-Kontrast": "Winter-summer contrast",
    "Standard-Pipeline": "Standard Pipeline",
    "Rang": "Rank",
    "Modell": "Model",
    "Kategorie": "Category",
    "Training Zeit": "Training Time",
    "Key Insights": "Key Insights",
    "Warum DL gewinnt:": "Why DL wins:",
    "Bidirektionale Architektur erfasst": "Bidirectional architecture captures",
    "Sequenzielle Muster optimal für tägliche Zyklen": "Sequential patterns optimal for daily cycles",
    "GPU-beschleunigt:": "GPU-accelerated:",
    "Training": "Training",
    "Archetyp": "Archetype",
    "Deterministisch-Symmetrisch": "Deterministic-Symmetric",
    
    # Wind Onshore
    "ML Dominanz trotz Chaos": "ML Dominance Despite Chaos",
    "Kontinuierlicher Betrieb": "Continuous operation",
    "nur": "only",
    "Nullwerte": "zero values",
    "hohe Volatilität": "high volatility",
    "DOMINANZ": "DOMINANCE",
    "Kritische Analyse": "Critical Analysis",
    "Gap zugunsten ML": "gap in favor of ML",
    "Warum ML gewinnt:": "Why ML wins:",
    "Wind ist fundamental stochastisch": "Wind is fundamentally stochastic",
    "Schmetterlingseffekt": "Butterfly effect",
    "Schwache sequenzielle Patterns": "Weak sequential patterns",
    "findet wenig": "finds little",
    "mittelt": "averages",
    "Trees": "Trees",
    "robust gegen Chaos": "robust against chaos",
    "dominiert Sequences": "dominates sequences",
    "Stochastisch-Chaotisch": "Stochastic-Chaotic",
    
    # Wind Offshore
    "Der Problemfall gelöst!": "The Problem Case Solved!",
    "valide Datenpunkte": "valid data points",
    "nach Data Cleaning": "after data cleaning",
    "NEUE ERGEBNISSE": "NEW RESULTS",
    "Alle 8 DL-Modelle getestet": "All 8 DL models tested",
    "beste Wahl": "best choice",
    "aber": "but",
    "zeigt massive Herausforderungen": "shows massive challenges",
    "Warum ist": "Why is",
    "so niedrig?": "so low?",
    "Datenverlust:": "Data Loss:",
    "valide Punkte": "valid points",
    "Outage fragmentiert Training-Daten": "Outage fragments training data",
    "Wetterabhängigkeit:": "Weather Dependency:",
    "Windgeschwindigkeit fehlt": "Wind speed missing",
    "nur Proxy-Features": "only proxy features",
    "Chaotische Physik:": "Chaotic Physics:",
    "Offshore-Wind noch unvorhersehbarer als Onshore": "Offshore wind even more unpredictable than onshore",
    "besser": "better",
    "Vergleich zu": "Comparison to",
    "Metrik": "Metric",
    "Interpretation": "Interpretation",
    "Bestes DL": "Best DL",
    "Bestes ML": "Best ML",
    "durch Outage": "due to outage",
    "durch Datenverlust": "due to data loss",
    "mehr": "more",
    "Trainierbare Punkte": "Trainable points",
    "Daten": "data",
    "Key Insight:": "Key Insight:",
    "ist": "is",
    "nicht unlösbar": "not unsolvable",
    "massiv schwerer": "massively harder",
    "als": "than",
    "schlägt": "beats",
    "auch hier": "here too",
    "wie bei": "as with",
    "SOTA-Modelle versagen spektakulär": "SOTA models fail spectacularly",
    "Lesson Learned:": "Lesson Learned:",
    "Bei erneuerbaren Energien sind": "For renewable energies",
    "exogene Wetter-Features essentiell": "exogenous weather features are essential",
    "Strukturbrüche müssen": "Structural breaks must be",
    "separat modelliert": "modeled separately",
    "werden": "be",
    "ist robuster als": "is more robust than",
    "bei fragmentierten Daten": "with fragmented data",
    "Fragmentiert-Chaotisch": "Fragmented-Chaotic",
    
    # Consumption
    "GRU übertrifft Bi-LSTM!": "GRU Outperforms Bi-LSTM!",
    "Stabile Muster": "Stable patterns",
    "niedrigste Volatilität": "lowest volatility",
    "klare Wochen-/Tageszyklen": "clear weekly/daily cycles",
    "Überraschung:": "Surprise:",
    "absolut": "absolute",
    "schneller": "faster",
    "Warum?": "Why?",
    "Wochenmuster sind unidirektional": "Weekly patterns are unidirectional",
    "Einfacher": "Simpler",
    "Gates statt": "gates instead of",
    "weniger Overfitting": "less overfitting",
    "Vorteile": "advantages",
    "Symmetrie": "symmetry",
    "hier nicht relevant": "not relevant here",
    "Strukturiert-Sequenziell": "Structured-Sequential",
    
    # Price
    "ML dominiert volatile Märkte": "ML Dominates Volatile Markets",
    "Hohe Volatilität": "High volatility",
    "negative Preise": "negative prices",
    "STARK": "STRONG",
    "Warum ML gewinnt:": "Why ML wins:",
    "Spikes dominieren": "spikes dominate",
    "erfasst Spikes besser": "captures spikes better",
    "DL glättet zu stark": "DL smooths too much",
    "unterschätzt Extrema": "underestimates extrema",
    "Volatil-Strukturiert": "Volatile-Structured",
    
    # Model Comparison
    "Modell-Architektur Vergleich": "Model Architecture Comparison",
    "Zeitreihen Analyse": "Time Series Analysis",
    "Performance-Matrix:": "Performance Matrix:",
    "Cross-Series Vergleich": "Cross-Series Comparison",
    "Architektur": "Architecture",
    "Best Use Case": "Best Use Case",
    "Symmetrische Patterns": "Symmetric Patterns",
    "Unidirektional/Volatil": "Unidirectional/Volatile",
    "Standard-Sequences": "Standard Sequences",
    "Chaotische Daten": "Chaotic Data",
    "Universell stark": "Universally strong",
    "Feature-rich": "Feature-rich",
    "Die 5 Zeitreihen-Archetypen": "The 5 Time Series Archetypes",
    "Starke Tageszyklen": "Strong daily cycles",
    "symmetrische Gradienten": "symmetric gradients",
    "Bidirektionalität nutzt Symmetrie": "Bidirectionality uses symmetry",
    "Wochenmuster": "Weekly patterns",
    "unidirektionale Sequenzen": "unidirectional sequences",
    "schneller als": "faster than",
    "Schwache Patterns": "Weak patterns",
    "hohe Stochastizität": "high stochasticity",
    "Ensemble mittelt Chaos": "Ensemble averages chaos",
    "Spikes": "Spikes",
    "negative Werte": "negative values",
    "Features > Sequences": "Features > Sequences",
    "Strukturbrüche": "Structural breaks",
    "Datenverlust": "data loss",
    "Beide schwach": "Both weak",
    "Entscheidungsbaum": "Decision Tree",
    "START:": "START:",
    "Analysiere deine Zeitreihe": "Analyze your time series",
    "Hat sie STRUKTURBRÜCHE": "Does it have STRUCTURAL BREAKS",
    "Missing": "missing",
    "Ja": "Yes",
    "robuster als LSTM": "more robust than LSTM",
    "Ist sie SYMMETRISCH": "Is it SYMMETRIC",
    "auf/ab gleich": "up/down equal",
    "z.B.": "e.g.",
    "Ist sie UNIDIREKTIONAL sequenziell?": "Is it UNIDIRECTIONAL sequential?",
    "Ist sie VOLATIL": "Is it VOLATILE",
    "DL versagt": "DL fails",
    "Ist sie CHAOTISCH": "Is it CHAOTIC",
    "NIEMALS": "NEVER",
    "nutzen": "use",
    "Bei uns IMMER negativ": "Always negative for us",
    
    # Discussion
    "Energiemarkt-Dynamik": "Energy Market Dynamics",
    "Was treibt was?": "What drives what?",
    "Die ökonomische Perspektive:": "The economic perspective:",
    "zeigt Marktmechanismen": "shows market mechanisms",
    "Alle 12 Kombinationen signifikant": "All 12 combinations significant",
    "Was bedeutet das wirtschaftlich?": "What does this mean economically?",
    "stärkster Effekt": "strongest effect",
    "Merit Order Effekt in Aktion:": "Merit Order Effect in Action:",
    "Sonniger Tag": "Sunny day",
    "ins Netz": "into the grid",
    "hat Grenzkosten": "has marginal costs",
    "verdrängt teure Gaskraftwerke": "displaces expensive gas plants",
    "Preis fällt von": "Price drops from",
    "auf": "to",
    "Real-World Impact:": "Real-World Impact:",
    "An sonnigen Sommertagen:": "On sunny summer days:",
    "möglich": "possible",
    "Aber:": "But:",
    "Prognose schwierig": "forecasting difficult",
    "weil non-linear": "because non-linear",
    "Schwellenwert-Effekt": "threshold effect",
    "Demand Response": "Demand Response",
    "Die Marktreaktion:": "The market reaction:",
    "Hoher Preis": "High price",
    "Industrie schaltet ab": "Industry shuts down",
    "Niedriger Preis": "Low price",
    "Zusätzliche Nachfrage": "Additional demand",
    "Beispiel": "Example",
    "Aluminium-Schmelze:": "Aluminum smelter:",
    "Flexibler Stromverbrauch": "Flexible power consumption",
    "Produktion runter": "Production down",
    "sinkt": "decreases",
    "Produktion hoch": "Production up",
    "steigt": "increases",
    "Korrelation:": "Correlation:",
    "negativ": "negative",
    "drückt Nachfrage": "suppresses demand",
    "Warum steigt Konsum bei hoher": "Why does consumption increase with high",
    "Einspeisung?": "feed-in?",
    "Hypothese": "Hypothesis",
    "Preissignal": "Price signal",
    "über": "via",
    "als Mediator": "as mediator",
    "Indirekte Kausalität:": "Indirect causality:",
    "Tageszeit-Effekt": "Time-of-day effect",
    "Solar peak": "Solar peak",
    "Uhr": "o'clock",
    "Industrielle Spitze": "Industrial peak",
    "Scheinkorrelation:": "Spurious correlation:",
    "Beide folgen Tagesrhythmus": "Both follow daily rhythm",
    "Smart Grid Response": "Smart Grid Response",
    "Intelligente Verbraucher": "Smart consumers",
    "Wärmepumpen": "Heat pumps",
    "E-Autos": "E-cars",
    "Laden automatisch bei hoher Renewable-Einspeisung": "Charge automatically with high renewable feed-in",
    "Reale Kausalität:": "Real causality:",
    "Solar-Forecast": "Solar forecast",
    "Consumption-Planung": "Consumption planning",
    "Test mit VAR:": "Test with VAR:",
    "ist signifikant": "is significant",
    "auch nach Kontrolle für Tageszeit": "even after controlling for time of day",
    "Hybride Erklärung:": "Hybrid explanation:",
    "Tageszeit": "Time of day",
    "Smart Response": "Smart response",
    "Bidirektional": "Bidirectional",
    "Komplexe Wechselwirkung:": "Complex interaction:",
    "Windreiche Nacht": "Windy night",
    "Überangebot": "Oversupply",
    "kann negativ werden": "can become negative",
    "Maximum": "Maximum",
    "Scheinbar paradox:": "Seemingly paradoxical:",
    "Wie kann Preis Wind beeinflussen?": "How can price influence wind?",
    "Erklärung:": "Explanation:",
    "Curtailment": "Curtailment",
    "Abregelung": "Curtailment",
    "Windparks werden abgeschaltet": "Wind farms are shut down",
    "Gemessene Wind-Einspeisung sinkt": "Measured wind feed-in decreases",
    "obwohl Wind physisch stark ist": "although wind is physically strong",
    "Ökonomische Entscheidung": "Economic decision",
    "nicht meteorologisch": "not meteorological",
    "Lesson:": "Lesson:",
    "Granger-Kausalität": "Granger causality",
    "physikalische Kausalität": "physical causality",
    "Kointegration:": "Cointegration:",
    "Langfristige Gleichgewichte": "Long-term equilibria",
    "4 Kointegrationsvektoren gefunden": "4 cointegration vectors found",
    "Was bedeutet das?": "What does this mean?",
    "Vereinfachtes Beispiel:": "Simplified example:",
    "Langfristiger Zusammenhang:": "Long-term relationship:",
    "Interpretation:": "Interpretation:",
    "Was sagt uns das?": "What does this tell us?",
    "Kurzfristig:": "Short-term:",
    "Preise schwanken wild": "Prices fluctuate wildly",
    "Volatilität": "Volatility",
    "Langfristig:": "Long-term:",
    "Es gibt Gleichgewichte": "Equilibria exist",
    "Regression to Mean": "Regression to mean",
    "Praktisch:": "Practically:",
    "Für Day-Ahead-Forecasts": "For day-ahead forecasts",
    "Kointegration hilft wenig": "cointegration helps little",
    "VAR-Modell:": "VAR Model:",
    "Kann man Kausalität nutzen?": "Can we use causality?",
    "Ernüchternde Ergebnisse:": "Sobering results:",
    "Univariat": "Univariate",
    "Best": "Best",
    "Multivariat": "Multivariate",
    "Warum hilft Kausalität nicht beim Forecasting?": "Why doesn't causality help with forecasting?",
    "VAR ist linear": "VAR is linear",
    "Märkte sind nicht-linear": "markets are non-linear",
    "Merit Order:": "Merit Order:",
    "Stufen-Funktion": "Step function",
    "keine Gerade": "not a line",
    "Schwellenwert-Effekt bei negativen Preisen": "Threshold effect with negative prices",
    "VAR erfasst das nicht": "VAR doesn't capture this",
    "Lag 24 zu lang für kurzfristige Dynamik": "Lag 24 too long for short-term dynamics",
    "Price-Spikes entstehen in Minuten": "Price spikes occur in minutes",
    "VAR mit 24h-Lag ist zu träge": "VAR with 24h lag is too sluggish",
    "Braucht kürzere Lags": "Needs shorter lags",
    "aber dann fehlt Saisonalität": "but then seasonality is missing",
    "Fehlende exogene Faktoren": "Missing exogenous factors",
    "Wetter": "Weather",
    "dominant für Solar/Wind": "dominant for solar/wind",
    "Marktevents": "Market events",
    "Kraftwerksausfälle": "Power plant outages",
    "Policy": "Policy",
    "CO2-Preis-Änderungen": "CO2 price changes",
    "Kritischer Insight:": "Critical insight:",
    "ist DESKRIPTIV": "is DESCRIPTIVE",
    "zeigt Zusammenhänge": "shows relationships",
    "Aber nicht PRÄDIKTIV": "But not PREDICTIVE",
    "hilft nicht beim Forecasting": "doesn't help with forecasting",
    "Univariate Modelle mit guten Features": "Univariate models with good features",
    "schlagen VAR": "beat VAR",
    "Praktische Implikationen für": "Practical implications for",
    "Energy Trading": "Energy Trading",
    "Was haben wir gelernt?": "What did we learn?",
    "Merit Order funktioniert!": "Merit Order works!",
    "hoch": "high",
    "runter": "down",
    "Für Trader:": "For traders:",
    "Monitor Solar-Forecast für Price-Prognose": "Monitor solar forecast for price prediction",
    "ist real": "is real",
    "Für Grid Operators:": "For grid operators:",
    "Preissignale steuern Nachfrage": "Price signals control demand",
    "ist ökonomisch": "is economic",
    "nicht physisch": "not physical",
    "Für Policy:": "For policy:",
    "Speicher-Incentives reduzieren Curtailment": "Storage incentives reduce curtailment",
    "ist nicht die Lösung": "is not the solution",
    "Non-Linearity": "Non-linearity",
    "fehlende Exogene": "missing exogenous",
    "Besser:": "Better:",
    "exogene Features": "exogenous features",
    "Alternativ:": "Alternatively:",
    "ML-basierte Multivariate": "ML-based multivariate",
    "mit Cross-Series-Lags": "with cross-series lags",
    "zeigt langfristige Trends": "shows long-term trends",
    "Für strategische Planung": "For strategic planning",
    "Investitionen": "Investments",
    "Nicht für operatives Forecasting": "Not for operational forecasting",
    "Day-Ahead": "Day-ahead",
    "Key Takeaway:": "Key Takeaway:",
    "Kausalität verstehen": "Understand causality",
    "bessere Features bauen": "build better features",
    "bessere univariate Modelle": "better univariate models",
    "Nicht:": "Not:",
    "schlechte Forecasts": "poor forecasts",
    
    # Lessons Learned
    "Lessons Learned für": "Lessons Learned for",
    "Advanced Time Series": "Advanced Time Series",
    "Was haben wir aus 5 Zeitreihen gelernt?": "What did we learn from 5 time series?",
    "Data Quality beats Fancy Models": "Data Quality beats Fancy Models",
    "von": "from",
    "nur durch Data Cleaning": "only through data cleaning",
    "Invest more in EDA than Model Tuning!": "Invest more in EDA than Model Tuning!",
    "Deep Learning ist NICHT universell": "Deep Learning is NOT universal",
    "Archetypen validiert": "archetypes validated",
    "Pattern erkannt:": "Pattern recognized:",
    "Je schwächer ML": "The weaker ML",
    "desto mehr hilft DL": "the more DL helps",
    "ist der unterschätzte Champion!": "is the underrated champion!",
    "Der stille Gewinner bei Chaos": "The silent winner with chaos",
    "besser als JEDES DL-Modell": "better than ANY DL model",
    "Robust gegen Stochastizität": "Robust against stochasticity",
    "kein GPU nötig": "no GPU needed",
    "als First Choice": "as first choice",
    "State-of-the-Art": "State-of-the-Art",
    "versagt konsistent bei Energy Data": "consistently fails with energy data",
    "Alle SOTA-Modelle negativ": "All SOTA models negative",
    "SOTA": "SOTA",
    "Production-Ready": "Production-ready",
    "Immer selbst benchmarken": "Always benchmark yourself",
    "Bidirektionalität hilft nur bei Symmetrie": "Bidirectionality only helps with symmetry",
    "symmetrisch": "symmetric",
    "sequenziell": "sequential",
    "fragmentiert": "fragmented",
    "Pattern-Typ bestimmt Architektur-Wahl": "Pattern type determines architecture choice",
    "DL-ROI korreliert negativ mit ML-Performance": "DL ROI correlates negatively with ML performance",
    "schwach": "weak",
    "Vorteil groß": "advantage large",
    "stark": "strong",
    "Vorteil klein": "advantage small",
    "perfekt": "perfect",
    "Wenn ML schon gut ist": "If ML is already good",
    "bringt DL wenig": "DL brings little",
    "Strukturbrüche brauchen separate Behandlung": "Structural breaks need separate treatment",
    "Outage-Periode zerstört Training": "Outage period destroys training",
    "Lösung:": "Solution:",
    "läuft": "running",
    "wie viel": "how much",
    "Domain Knowledge > Algorithmen": "Domain knowledge > algorithms",
    "Training Zeit": "Training time",
    "Performance": "Performance",
    "Schnell iterieren": "Iterate quickly",
    "langsames": "slow",
    "Perfect Model": "perfect model",
    "Exogene Features sind kritisch bei Renewables": "Exogenous features are critical for renewables",
    "ohne Windgeschwindigkeit": "without wind speed",
    "Erwartung:": "Expectation:",
    "mit Weather-APIs": "with weather APIs",
    "Investiere in Data Sourcing": "Invest in data sourcing",
    "Nächste Schritte": "Next Steps",
    "Alle 5 Zeitreihen getestet": "All 5 time series tested",
    "DL-Archetypen validiert": "DL archetypes validated",
    "GRU-First Strategy": "GRU-First Strategy",
    "GRU als Default für neue Zeitreihen": "GRU as default for new time series",
    "Ensemble:": "Ensemble:",
    "temporal": "temporal",
    "features": "features",
    "ACF-Based Routing:": "ACF-Based Routing:",
    "Automatische Modellwahl": "Automatic model selection",
    "Exogene Features:": "Exogenous Features:",
    "Wetter-APIs integrieren": "Integrate weather APIs",
    "Solar-Irradiance": "Solar irradiance",
    "Production:": "Production:",
    "SOTA-Debug:": "SOTA-Debug:",
    "Kann man": "Can we",
    "retten?": "rescue?",
    "evtl. nicht lohnend": "possibly not worthwhile",
    "Open Questions für Diskussion": "Open Questions for Discussion",
    "Warum ist GRU so viel besser": "Why is GRU so much better",
    "Einfachheit = Robustheit?": "Simplicity = robustness?",
    "Warum versagen SOTA-Modelle SO konsistent?": "Why do SOTA models fail SO consistently?",
    "Univariate Optimierung vs Feature-Rich Energy Data?": "Univariate optimization vs feature-rich energy data?",
    "Fundamental falsch für Energy?": "Fundamentally wrong for energy?",
    "Kann man": "Can we bring",
    "auf": "to",
    "bringen?": "?",
    "Windgeschwindigkeit": "Wind speed",
    "Richtung": "Direction",
    "Hybrid-Modell": "Hybrid model",
    "Regressor": "Regressor",
    "Ensemble": "Ensemble",
    "lernt temporal": "learns temporal",
    "lernt features": "learns features",
    "Unterschiedliche Fehler": "Different errors",
    "Kombination?": "Combination?",
    "Transfer Learning zwischen Archetypen?": "Transfer learning between archetypes?",
    "andere PV": "other PV",
    "andere Länder": "other countries",
    "Zwischen Archetypen?": "Between archetypes?",
    "zu unterschiedlich": "too different",
    
    # References
    "Referenzen & Quellen": "References & Sources",
    "Frameworks:": "Frameworks:",
    "Literatur:": "Literature:",
    "Forecasting: Principles and Practice": "Forecasting: Principles and Practice",
    "Long Short-Term Memory": "Long Short-Term Memory",
    
    # Final
    "DANKE FÜR IHRE AUFMERKSAMKEIT!": "THANK YOU FOR YOUR ATTENTION!",
    "Fragen?": "Questions?",
    "Diskussion?": "Discussion?",
    "Keine Universallösung": "No universal solution",
    "Praktischer Rat:": "Practical advice:",
    "Teste": "Test",
    "in dieser Reihenfolge": "in this order",
    "Wichtigste Lektion:": "Most important lesson:",
    "durch Cleaning": "through cleaning",
}

# Apply translations
content_en = content
for de, en in translations.items():
    content_en = content_en.replace(de, en)

# Split into slides
slides_raw = content_en.split('\n---\n')

# Markdown parser with extensions
md = markdown.Markdown(extensions=['tables', 'fenced_code', 'nl2br'])

# Convert each slide and fix image paths
slides_html = []
for i, slide_md in enumerate(slides_raw):
    # Fix image paths: results/figures/ -> figures/
    slide_md_fixed = re.sub(r'!\[([^\]]*)\]\(results/figures/', r'![\1](figures/', slide_md)
    
    slide_html = md.convert(slide_md_fixed)
    md.reset()
    slides_html.append(slide_html)

# HTML Template
html_output = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Time Series Forecasting - Presentation</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            overflow: hidden;
        }}
        .slide {{
            display: none;
            width: 100vw;
            height: 100vh;
            padding: 50px 80px;
            background: white;
            overflow-y: auto;
        }}
        .slide.active {{ display: block; }}
        
        /* Typography */
        .slide h1 {{ 
            color: #667eea; 
            font-size: 2.5em; 
            margin-bottom: 25px;
            border-bottom: 4px solid #764ba2;
            padding-bottom: 15px;
            font-weight: 700;
        }}
        .slide h2 {{ 
            color: #764ba2; 
            font-size: 2em; 
            margin: 25px 0 15px;
            font-weight: 600;
        }}
        .slide h3 {{ 
            color: #667eea; 
            font-size: 1.5em; 
            margin: 20px 0 10px;
            font-weight: 600;
        }}
        .slide h4 {{
            color: #555;
            font-size: 1.2em;
            margin: 15px 0 10px;
            font-weight: 600;
        }}
        
        /* Tables */
        .slide table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            font-size: 0.9em;
        }}
        .slide th, .slide td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        .slide th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}
        .slide tbody tr:hover {{ 
            background: #f8f9fa;
            transition: background 0.2s;
        }}
        .slide tbody tr:nth-child(even) {{
            background: #f9f9f9;
        }}
        
        /* Code */
        .slide code {{
            background: #f4f4f4;
            padding: 3px 8px;
            border-radius: 4px;
            font-family: 'SF Mono', Monaco, 'Courier New', monospace;
            font-size: 0.9em;
            color: #d63384;
        }}
        .slide pre {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 20px 0;
            font-size: 0.85em;
            line-height: 1.5;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        }}
        .slide pre code {{
            background: none;
            color: inherit;
            padding: 0;
        }}
        
        /* Lists */
        .slide ul, .slide ol {{
            margin-left: 40px;
            line-height: 1.8;
            margin-bottom: 20px;
        }}
        .slide li {{
            margin-bottom: 10px;
        }}
        .slide li::marker {{
            color: #667eea;
            font-weight: bold;
        }}
        
        /* Text formatting */
        .slide strong {{ 
            color: #764ba2; 
            font-weight: 600;
        }}
        .slide em {{
            color: #667eea;
            font-style: italic;
        }}
        .slide p {{
            margin-bottom: 15px;
            line-height: 1.7;
            color: #333;
        }}
        
        /* Images */
        .slide img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
            margin: 25px 0;
        }}
        
        /* Blockquotes */
        .slide blockquote {{
            border-left: 5px solid #667eea;
            padding-left: 20px;
            margin: 25px 0;
            color: #555;
            font-style: italic;
            background: #f8f9fa;
            padding: 15px 20px;
            border-radius: 0 8px 8px 0;
        }}
        
        /* Navigation Controls */
        .controls {{
            position: fixed;
            bottom: 40px;
            right: 40px;
            display: flex;
            gap: 15px;
            z-index: 1000;
        }}
        .controls button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 16px;
            border-radius: 50px;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
            transition: all 0.3s ease;
            font-weight: 600;
        }}
        .controls button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 6px 30px rgba(118, 75, 162, 0.6);
        }}
        .controls button:active {{
            transform: translateY(-1px);
        }}
        
        /* Slide Counter */
        .slide-number {{
            position: fixed;
            bottom: 105px;
            right: 40px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 12px 24px;
            border-radius: 30px;
            font-size: 16px;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }}
        
        /* Progress Bar */
        .progress-bar {{
            position: fixed;
            top: 0;
            left: 0;
            width: 0%;
            height: 4px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
            z-index: 1001;
        }}
        
        /* Responsive */
        @media (max-width: 1024px) {{
            .slide {{
                padding: 30px 40px;
                font-size: 0.9em;
            }}
            .slide h1 {{ font-size: 2em; }}
            .slide h2 {{ font-size: 1.6em; }}
            .slide table {{ font-size: 0.8em; }}
        }}
    </style>
</head>
<body>
    <div class="progress-bar" id="progress"></div>
    
    <div id="slides">
"""

# Add all slides
for slide_html in slides_html:
    html_output += f'        <div class="slide">{slide_html}</div>\n'

html_output += f"""    </div>
    
    <div class="controls">
        <button onclick="prevSlide()" title="Previous Slide (←)">◀ Back</button>
        <button onclick="nextSlide()" title="Next Slide (→)">Next ▶</button>
    </div>
    
    <div class="slide-number">
        <span id="current">1</span> / <span id="total">{len(slides_html)}</span>
    </div>
    
    <script>
        let currentSlide = 0;
        const slides = document.querySelectorAll('.slide');
        const totalSlides = slides.length;
        
        document.getElementById('total').textContent = totalSlides;
        updateProgress();
        
        function showSlide(n) {{
            slides[currentSlide].classList.remove('active');
            currentSlide = (n + totalSlides) % totalSlides;
            slides[currentSlide].classList.add('active');
            document.getElementById('current').textContent = currentSlide + 1;
            updateProgress();
        }}
        
        function updateProgress() {{
            const progress = ((currentSlide + 1) / totalSlides) * 100;
            document.getElementById('progress').style.width = progress + '%';
        }}
        
        function nextSlide() {{ showSlide(currentSlide + 1); }}
        function prevSlide() {{ showSlide(currentSlide - 1); }}
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'Enter') {{
                e.preventDefault();
                nextSlide();
            }}
            if (e.key === 'ArrowLeft' || e.key === 'Backspace') {{
                e.preventDefault();
                prevSlide();
            }}
            if (e.key === 'Home') {{
                e.preventDefault();
                showSlide(0);
            }}
            if (e.key === 'End') {{
                e.preventDefault();
                showSlide(totalSlides - 1);
            }}
        }});
        
        // Touch/Swipe support
        let touchStartX = 0;
        let touchEndX = 0;
        
        document.addEventListener('touchstart', e => {{
            touchStartX = e.changedTouches[0].screenX;
        }});
        
        document.addEventListener('touchend', e => {{
            touchEndX = e.changedTouches[0].screenX;
            handleSwipe();
        }});
        
        function handleSwipe() {{
            if (touchEndX < touchStartX - 50) nextSlide();
            if (touchEndX > touchStartX + 50) prevSlide();
        }}
        
        // Show first slide
        showSlide(0);
    </script>
</body>
</html>"""

# Save file
output_file = Path("presentation_output/presentation_beautiful_en.html")
output_file.parent.mkdir(exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_output)

print(f"✅ Beautiful presentation created: {output_file}")
print(f"📊 {len(slides_html)} slides generated")
