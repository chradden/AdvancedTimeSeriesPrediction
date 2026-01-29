# 🚀 Monitoring Stack Setup & Persistierung

## Status 2026-01-29: ✅ Fully Configured

Die Monitoring-Stack ist jetzt **vollständig konfiguriert** und startet automatisch beim nächsten Neustart.

### Was wurde konfiguriert:

#### 1. **Docker Compose** (docker-compose.yml)
- ✅ Prometheus jetzt im Standard-Setup (nicht nur als Profil)
- ✅ Grafana jetzt im Standard-Setup (nicht nur als Profil)
- ✅ Healthchecks für alle Services hinzugefügt
- ✅ Dependency-Management: API wartet auf Prometheus
- ✅ Volume Persistence für Grafana-Daten

#### 2. **API Monitoring Integration** (api_simple.py)
- ✅ Background-Tasks für Dummy-Actuals
- ✅ Baselines für alle 5 Energietypen beim Start
- ✅ Drift-Detection läuft kontinuierlich
- ✅ Data-Quality-Checks bei jedem Forecast

#### 3. **Grafana Provisioning** (monitoring/grafana-provisioning/)
- ✅ Prometheus-Datasource auto-provisioned
- ✅ Dashboard mit allen 7 Charts auto-deployed
- ✅ Keine manuellen Clicks nötig!

#### 4. **Startup Script** (start_monitoring.sh)
- ✅ One-Click Start für Codespaces
- ✅ Gesundheitsprüfungen für alle Services
- ✅ Hilfreiche Output mit Port-Information

#### 5. **Dokumentation** (docs/)
- ✅ GRAFANA_DASHBOARD_GUIDE_DE.md - Komplette Erklärung für Anfänger
- ✅ QUICKSTART.md - Aktualisiert mit Monitoring-Profil

---

## 🔄 Nächster Neustart (Neuer Codespace)

### Schritt 1: Repository klonen
```bash
git clone https://github.com/chradden/AdvancedTimeSeriesPrediction.git
cd AdvancedTimeSeriesPrediction/energy-timeseries-project
```

### Schritt 2: Start-Script ausführen
```bash
./start_monitoring.sh
```

**Oder klassisch mit Docker:**
```bash
docker compose up
```

Das war's! ✨

- API läuft auf Port 8000
- Prometheus läuft auf Port 9090
- Grafana läuft auf Port 3000
- Baselines sind gesetzt
- Dashboard ist auto-deployed
- Dummy-Actuals werden generiert

---

## 📊 Was bleibt erhalten?

### Grafana Volumes
```yaml
volumes:
  grafana-storage:  # Alle Dashboards, Datenquellen, User-Einstellungen
```

Beim Neustart:
- ✅ Das Dashboard bleibt erhalten
- ✅ Admin-Passwort "admin" bleibt
- ✅ Alle Konfigurationen bleiben

### API Monitoring State
- ✅ Baselines werden beim Startup neu gesetzt
- ✅ Background-Tasks starten automatisch
- ✅ Metriken akkumulieren neu

---

## 🐳 Docker Compose Commands

```bash
# Alles starten (mit Monitoring)
docker compose up

# Alles stoppen
docker compose down

# Mit Volume-Cleanup (Neustart)
docker compose down -v

# Nur API (ohne Monitoring)
docker compose up -d api

# Logs anschauen
docker compose logs -f grafana
docker compose logs -f api
docker compose logs -f prometheus

# Status prüfen
docker compose ps
```

---

## 🔍 Troubleshooting beim Neustart

### Problem: Grafana zeigt keine Daten
```bash
# Lösung 1: Seite neu laden (F5)
# Lösung 2: Timebereich ändern (z.B. "Last 1 hour")
# Lösung 3: Refresh klicken
# Lösung 4: Container-Logs prüfen
docker compose logs grafana | tail -50
```

### Problem: API startet nicht
```bash
# Logs prüfen
docker compose logs api

# Container neu bauen
docker compose up -d --build api

# Docker-Cache löschen
docker system prune -a
docker compose down -v
docker compose up
```

### Problem: Prometheus hat keine Metriken
```bash
# Prometheus UI öffnen: http://localhost:9090
# Targets prüfen: http://localhost:9090/targets
# API sollte grün sein (UP)

# Wenn rot: API nicht erreichbar
# Logs prüfen:
docker compose logs api | grep metrics
```

---

## 📈 Performance Notes

### Speicherverbrauch
- API Container: ~300-400 MB
- Grafana Container: ~100-150 MB
- Prometheus Container: ~100-200 MB
- **Total: ~600 MB** (akzeptabel für Development)

### Netzwerk
- Prometheus scrapped API alle 15 Sekunden
- Grafana refreshed Dashboard alle 30 Sekunden
- Background-Task erzeugt Dummy-Actuals alle 30 Sekunden

### Bei vielen Predictions
- MAE/MAPE werden nur für letzte 100 Predictions berechnet (Memory-effizient)
- Grafana speichert alles in SQLite (grafana-storage Volume)

---

## 🔐 Sicherheit

### Default Credentials
```
Grafana Admin: admin / admin
```

⚠️ **In Produktion ändern!**
```yaml
# In docker-compose.yml:
environment:
  - GF_SECURITY_ADMIN_PASSWORD=<sicheres_passwort>
```

### API
- Keine Authentifizierung (nur in Codespaces, nicht produktiv!)
- CORS: Alle Origins erlaubt (nur für Demo!)

---

## 📚 Dateien-Übersicht

```
energy-timeseries-project/
├── docker-compose.yml              # ✅ Monitoring im Default
├── start_monitoring.sh             # ✅ One-Click Start
├── api_simple.py                   # ✅ Mit Monitoring-Integration
├── monitoring/
│   ├── prometheus.yml              # ✅ Scrape-Config
│   ├── grafana-dashboard.json      # ✅ Auto-deployed Dashboard
│   └── grafana-provisioning/       # ✅ Auto-Provisioning
│       ├── dashboards/
│       │   └── dashboard.yml       # ✅ Dashboard-Provider
│       └── datasources/
│           └── datasource.yml      # ✅ Prometheus-Datasource
└── docs/
    └── GRAFANA_DASHBOARD_GUIDE_DE.md  # ✅ Dokumentation
```

---

## ✅ Checkliste für Produktion

- [ ] Grafana Admin-Passwort ändern
- [ ] CORS-Origins einschränken
- [ ] API Authentication hinzufügen (JWT, OAuth)
- [ ] Prometheus Retention-Policy setzen (nicht unbegrenzt speichern)
- [ ] Backups der Grafana-DB einrichten
- [ ] Monitoring Alerts konfigurieren
- [ ] HTTPS aktivieren
- [ ] Load Balancer vor der API

---

## 📞 Support

Wenn etwas nach Neustart nicht funktioniert:

1. **Alle Container neu starten:**
   ```bash
   docker compose down -v
   docker compose up
   ```

2. **Container-Logs prüfen:**
   ```bash
   docker compose logs
   ```

3. **Ports checken:**
   ```bash
   docker compose ps
   netstat -an | grep 8000
   ```

4. **Grafana manuell konfigurieren:**
   - UI: http://localhost:3000
   - Connection → Data Sources → Add Prometheus
   - URL: http://prometheus:9090

---

**Letzte Aktualisierung:** 2026-01-29  
**Status:** ✅ Production-Ready für Codespaces
