# Archiv-Verzeichnis

Dieses Verzeichnis enthält **veraltete Entwicklungspfade**, die nicht mehr für die produktive Nutzung benötigt werden, aber für die Projekt-Historie bewahrt wurden.

## 📁 Struktur

### `old_scripts/`
Debug-, Analyse- und Validierungs-Skripte aus früheren Entwicklungsphasen:

**Debug-Skripte:**
- `debug_solar_performance.py` - Solar-Performance-Debugging
- `debug_wind_offshore_r2.py` - Wind-Offshore R²-Analyse

**Analyse-Skripte:**
- `analyze_lstm_mape_discrepancy.py` - LSTM-MAPE-Diskrepanzen
- `analyze_multi_series.py` - Multi-Serien-Analyse
- `analyze_wind_offshore.py` - Wind-Offshore-Analyse

**Fix-Skripte:**
- `fix_deep_learning_metrics.py` - Metriken-Korrekturen
- `fix_wind_refs.py` - Referenz-Korrekturen

**Validierungs-Skripte:**
- `validate_notebook10_fix.py` - Notebook-10-Validierung
- `validate_wind_offshore_fix.py` - Wind-Offshore-Validierung

**Test-Skripte:**
- `quick_test_nb10_fixes.py` - Schnelltest für Notebook-Fixes
- `find_best_wind_offshore_period.py` - Perioden-Optimierung

**Adaptions-Skripte:**
- `adapt_consumption.py` - Consumption-Anpassungen
- `adapt_solar.py` - Solar-Anpassungen

### `old_docs/`
Session-Logs und temporäre Dokumentationen aus der Entwicklungsphase:

- `LSTM_MAPE_ANALYSE.md` - LSTM-MAPE-Analyse
- `MODEL_DRIFT_FIX.md` - Model-Drift-Korrekturen
- `SESSION_2_DEBUGGING.md` - Session-2-Debugging
- `SESSION_3_OPTIMIZATIONS.md` - Session-3-Optimierungen
- `SESSION_5_EXTENSIONS.md` - Session-5-Erweiterungen
- `SESSION_SUMMARY_2026-01-22.md` - Session-Zusammenfassung
- `WHATS_NEW_SESSION_5.md` - Neuerungen Session 5

### `old_root_files/`
Veraltete Haupt-Skripte, die durch die Extended Pipelines ersetzt wurden:

- `run_chronos_forecasting.py` - Chronos-Foundation-Model-Experiment
- `run_complete_multi_series.py` - Alte Multi-Serien-Pipeline
- `run_deep_learning_retrain.py` - Deep-Learning-Retraining
- `run_ensemble_methods.py` - Ensemble-Methoden
- `run_ensemble_simple.py` - Vereinfachte Ensembles
- `run_xgboost_tuning.py` - XGBoost-Hyperparameter-Tuning

## 🔄 Warum archiviert?

Diese Dateien wurden archiviert, weil:

1. **Ersetzt durch bessere Alternativen**
   - Extended Pipelines (`scripts/run_*_extended_pipeline.py`) bieten strukturierte, reproduzierbare Workflows
   - Notebooks bieten interaktive Analysen

2. **Spezifische temporäre Probleme gelöst**
   - Debug-Skripte lösten spezifische Bugs (z.B. LSTM-MAPE-Diskrepanzen)
   - Diese Probleme sind nun behoben

3. **Entwicklungs-Historie**
   - Session-Logs dokumentierten den Entwicklungsprozess
   - Für produktive Nutzung nicht mehr relevant

## ✅ Aktuelle Alternativen

**Statt alter Debug-Skripte:**
- Nutze `scripts/run_*_extended_pipeline.py` für reproduzierbare Analysen

**Statt Session-Logs:**
- Siehe `docs/FINAL_PROJECT_SUMMARY.md` für Gesamtübersicht
- Siehe `docs/PROJECT_COMPLETION_REPORT.md` für Abschlussbericht

**Statt alter Root-Skripte:**
- `run_chronos_forecasting.py` → Experimentell, bei Bedarf aus Archiv holen
- `run_complete_multi_series.py` → Nutze individuelle Extended Pipelines
- Ensemble-Methoden → In Notebooks integriert

## 🗑️ Kann ich das Archiv löschen?

**Nein, nicht empfohlen.** Das Archiv:
- Dokumentiert den Entwicklungsprozess
- Kann für spezielle Analysen nützlich sein
- Nimmt wenig Speicherplatz ein

Bei Bedarf können einzelne Skripte reaktiviert werden.

---

**Archiviert:** 31. Januar 2026
