# Phase 1 Archiv

## 📦 Was ist hier?

Alle Dateien aus der **ersten Entwicklungsphase** (Notebooks, API, Monitoring, alte Dokumentationen).

## 🎯 Warum archiviert?

**Neuer Fokus**: Systematische Modell-Evaluation mit automatisierten Pipelines für alle 5 Zeitreihen.

Die alten Notebooks und API-Implementierungen waren explorativ und nicht mehr Teil der aktuellen Strategie.

## 📁 Archiv-Struktur

### `phase1_notebooks/`
- 16 Jupyter Notebooks (Solar, Wind, Price, Cross-Series)
- Explorative Analysen und manuelle Modell-Tests
- ➡️ **Ersetzt durch**: `scripts/run_*_extended_pipeline.py`

### `phase1_api_monitoring/`
- FastAPI Server (`api.py`, `api_simple.py`)
- Grafana/Prometheus Monitoring
- Docker Setup
- ➡️ **Status**: Noch nicht wieder implementiert (kommt später)

### `phase1_misc_scripts/`
- Debug-Scripts (`debug_*.py`, `analyze_*.py`, `fix_*.py`)
- Alte Dokumentationen (`docs/`, `catboost_info/`)
- Test-Scripts
- ➡️ **Status**: Obsolet, durch Pipelines ersetzt

### `old_scripts/`
- Debug/Analyse-Scripts aus früheren Sessions
- ➡️ **Status**: Obsolet

### `old_docs/`
- Session-Logs (SESSION_2_DEBUGGING.md, SESSION_3_OPTIMIZATIONS.md, etc.)
- Temporäre Dokumentationen
- ➡️ **Status**: Historisch, nicht mehr relevant

### `old_root_files/`
- Veraltete Root-Level Scripts (`run_chronos_forecasting.py`, etc.)
- ➡️ **Status**: Ersetzt durch extended pipelines

## 🚀 Aktuelle Strategie

**Siehe:** `/README.md` (Root) und `/scripts/`

**5 Pipelines**:
1. `run_solar_extended_pipeline.py`
2. `run_wind_offshore_extended_pipeline.py`
3. `run_wind_onshore_extended_pipeline.py`
4. `run_price_extended_pipeline.py`
5. `run_consumption_extended_pipeline.py`

Jede Pipeline:
- 9 Phasen (Exploration → Preprocessing → Baselines → Statistical → ML Trees → Deep Learning → Generative → Advanced → Comparison)
- Automatisiert
- Reproduzierbar
- Vollständige Metriken & Visualisierungen

---

**Stand**: Januar 2026 | **Archiviert**: 31. Januar 2026
