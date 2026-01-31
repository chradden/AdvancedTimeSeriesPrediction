# 📊 Grafana Dashboard - Erklärung für Anfänger

## 🎯 Überblick

Das Grafana Dashboard zeigt Echtzeit-Daten zur **Energie-Vorhersage und Modell-Performance**. Es überwacht, wie gut unsere KI-Modelle Stromproduktion und -verbrauch vorhersagen können.

### 📌 Wichtig: 2 Dashboards für 2 Aufgaben

| Dashboard | Localhost | Codespace | Zweck |
|-----------|-----------|-----------|-------|
| 📈 **Grafana** (dieses hier) | http://localhost:3000 | https://<codespace-name>-3000.app.github.dev | Performance-Metriken & Monitoring |
| 🎯 **API-UI** (zum Prognosen generieren) | http://localhost:8000/ui | https://<codespace-name>-8000.app.github.dev/ui | Live-Vorhersagen & Visualisierung |

**Tipp:** Beide Seite-an-Seite öffnen für den vollständigen Überblick!

👉 Siehe auch: [PREDICTIONS_AND_GRAFANA.md](PREDICTIONS_AND_GRAFANA.md) für die Integration

---

## 📈 Die Charts erklärt

### 1️⃣ **Prediction Count by Energy Type** (Oben links)

**Was wird angezeigt?**
- Anzahl der Vorhersagen pro Energietyp über die Zeit

**Die 5 Energietypen:**
- 🌞 **Solar** - Solarstrom-Erzeugung
- 💨 **Wind Offshore** - Windkraft auf dem Meer
- 💨 **Wind Onshore** - Windkraft an Land
- 🔋 **Consumption** - Stromverbrauch in Deutschland
- 💰 **Price** - Strompreise

**Was bedeutet es?**
- Eine steigende Linie = Das Modell wird häufiger verwendet
- Steile Anstiege = Viele Vorhersagen auf einmal
- Wichtig für: Systemauslastung verstehen

**Gut oder schlecht?**
- Konstant ansteigend = ✅ Normal
- Plötzliche Lücken = ⚠️ Modell könnte offline sein

---

### 2️⃣ **Model Drift Score** (Oben rechts)

**Was wird angezeigt?**
- "Ist unser Modell noch gut?" - Messwert zwischen 0 und 1

**Die Skala:**
- 🟢 **0.0 - 0.2** = Modell läuft super! (Vorhersagen sind genau)
- 🟡 **0.2 - 0.5** = Warnung - Performance lässt nach
- 🔴 **0.5 - 1.0** = Problem! Modell braucht Update

**Was bedeutet Drift?**
"Drift" = Das Modell verliert an Genauigkeit. Die Realität ändert sich (Jahreszeiten, neuer Trend), aber das Modell passt sich nicht an.

**Beispiel:**
- Modell trainiert im Sommer → sagt Solarstrom gut voraus
- Winter kommt → viel weniger Sonne → Vorhersagen werden falsch
- Model Drift Score steigt 📈 (Warnung!)

**Was tun?**
- Score > 0.5 = **Modell sollte neu trainiert werden**

---

### 3️⃣ **Prediction MAE (50 predictions window)** (Unten links)

**MAE = Mean Absolute Error** (Mittlerer absoluter Fehler)

**Was wird angezeigt?**
- Wie weit liegen die Vorhersagen von der Realität ab?
- Gemessen in MW (Megawatt) oder anderen Einheiten

**Beispiel:**
- Vorhersage: 5000 MW Solar
- Realität: 4950 MW Solar
- Fehler: 50 MW → geht in MAE ein

**Niedrig = Gut, Hoch = Schlecht:**
- Solar MAE 250 = ✅ Sehr gut!
- Wind MAE 500 = ⚠️ Könnte besser sein
- Consumption MAE 500 = ✅ Akzeptabel

**Die "50 predictions window":**
- Schaut nur die letzten 50 Vorhersagen an (nicht alle)
- Hilft, aktuelle Fehler zu sehen (nicht historische)

---

### 4️⃣ **Prediction MAPE (%)** (Unten rechts)

**MAPE = Mean Absolute Percentage Error** (Fehler in %)

**Was wird angezeigt?**
- Wie weit weg ist die Vorhersage? (in Prozent!)

**Beispiel:**
- Vorhersage: 1000 MW
- Realität: 900 MW
- Fehler: 100 MW = **10% MAPE** ← Das ist der Fehler in %

**Bewertung:**
- 🟢 **0-5%** = Exzellent
- 🟡 **5-10%** = Gut
- 🟠 **10-20%** = Akzeptabel
- 🔴 **>20%** = Schlecht, Modell braucht Update

**Warum Prozent?**
- MAE zeigt absolute Fehler
- MAPE zeigt relative Fehler (besser vergleichbar!)
- Beispiel: 100 MW Fehler bei 1000 MW = 10% (schlecht)
- Aber: 100 MW Fehler bei 50000 MW = 0.2% (super!)

---

### 5️⃣ **Data Quality Score** (Unten Mitte - Gauge)

**Was wird angezeigt?**
- Qualität der Eingangsdaten zwischen 0 und 1 (wie ein Tankometer)

**Die Skala:**
- 🟢 **0.9-1.0** = Beste Datenqualität (Tank voll)
- 🟡 **0.7-0.9** = Noch ok
- 🔴 **<0.7** = Warnung! Daten haben Probleme

**Was wird überprüft?**
- ✅ Fehlende Werte (NaN) - sollten <5% sein
- ✅ Null-Werte - sollten <5% sein
- ✅ Konsistenz der Daten

**Beispiel Problem:**
- Sensorausffall → viele fehlende Werte
- Data Quality Score sinkt → ⚠️ Warnung!

---

### 6️⃣ **Prediction Latency (p95)** (Unten Mitte-Rechts)

**Was wird angezeigt?**
- Wie schnell ist die Vorhersage? (in Sekunden)
- p95 = 95% der Vorhersagen sind schneller als dieser Wert

**Beispiel:**
- Latency p95 = 0.5 Sekunden
- Bedeutet: 95% der Vorhersagen sind in < 0.5 Sekunden fertig
- 5% sind langsamer (ok, sind Ausnahmen)

**Gut oder schlecht?**
- **< 0.1 Sekunden** = 🟢 Blitzschnell (optimal)
- **0.1 - 0.5 Sekunden** = 🟡 Ok, aber ausbaufähig
- **> 1 Sekunde** = 🔴 Zu langsam! Server überfordert?

**Praktisch:**
- Latency steigt → viele Vorhersagen gleichzeitig?
- CPU/RAM könnte Engpass sein

---

### 7️⃣ **API Request Rate** (Unten Rechts)

**Was wird angezeigt?**
- Wie viele API-Anfragen pro Minute kommen?
- Trend über die Zeit

**Beispiel:**
- 100 Anfragen/Minute → viele User nutzen das System
- Plötzlich 0 Anfragen → API ist down?

**Gut oder schlecht?**
- 🟢 Konstant = Normal, Lads sind gleichmäßig
- 📈 Steigend = Mehr Nutzer (System wird beliebter!)
- ❌ Absturz = Wahrscheinlich ein Problem

---

## 🎓 Kombiniert verstehen

### Szenario 1: Alles grün ✅
```
✅ Prediction Count: Ansteigend
✅ Model Drift: 0.1 (super!)
✅ MAE: 250 MW (gut)
✅ MAPE: 3% (exzellent)
✅ Data Quality: 0.95 (Sehr gut)
✅ Latency: 0.15s (schnell)
✅ API Requests: Konstant

→ System läuft PERFEKT!
```

### Szenario 2: Problem erkannt ⚠️
```
⚠️ Model Drift: 0.7 (hoch!)
⚠️ MAE: 1000 MW (zu hoch)
⚠️ MAPE: 25% (schlecht)

→ Modell braucht RETRAINING!
   - Jahreszeit hat sich geändert
   - Neue Wettermuster
   - Veraltete Trainingsdaten

→ MASSNAHME: Modell mit neuen Daten neu trainieren
```

### Szenario 3: Technisches Problem 🔴
```
🔴 Latency: 5 Sekunden (viel zu langsam!)
🔴 API Requests: Plötzlich 0
🔴 Data Quality: 0.3 (viele fehlende Daten)

→ Könnte sein:
   - Server überfordert
   - Datenquelle offline
   - Netzwerkprobleme

→ MASSNAHME: Server neu starten / Logs prüfen
```

---

## 🔄 Zeitliche Einstellungen

Im oben links findest du Einstellungen:

- **Last 15 minutes** = Letzte 15 Minuten anzeigen
- **Last 1 hour** = Letzte 60 Minuten
- **Last 24 hours** = Letzter Tag
- **Refresh 30s** = Grafana aktualisiert alle 30 Sekunden

**Tipp:** 
- Für Debugging: "Last 1 hour" + "Refresh 5s" wählen
- Für Monitoring: "Last 24 hours" nutzen

---

## 📊 Metriken-Zusammenfassung

| Chart | Einheit | Gut | Schlecht | Aktion |
|-------|---------|-----|----------|--------|
| Prediction Count | Anzahl | Ansteigend | 0 für lange Zeit | System check |
| Model Drift | 0-1 | <0.2 | >0.5 | Modell retrainieren |
| MAE | MW/€ | Baseline | 2x Baseline | Daten/Modell prüfen |
| MAPE | % | <5% | >20% | Daten/Modell prüfen |
| Data Quality | 0-1 | >0.9 | <0.7 | Datenquelle prüfen |
| Latency p95 | Sekunden | <0.1s | >1s | Server optimieren |
| API Requests | Anfragen/min | Stabil | Spitzen/Lücken | Kapazität planen |

---

## 🚀 So navigierst du

1. **Schnellcheck (5 Minuten):**
   - Model Drift Score anschauen → Ist das Modell ok?
   - Data Quality → Sind die Daten gut?
   - Latency → Läuft das System schnell?

2. **Tiefer Blick (15 Minuten):**
   - MAE/MAPE für jeden Energietyp prüfen
   - Trends in den Linien-Charts anschauen
   - Vergleiche mit gestern/vorgestern

3. **Problemsuche:**
   - Model Drift hoch? → Retraining starten
   - Latency hoch? → Server-Logs prüfen
   - API Requests 0? → System neu starten

---

## 💡 Häufige Fragen

**F: Warum ändern sich die Zahlen ständig?**
A: Das System generiert ständig neue Vorhersagen und vergleicht sie mit Realwerten. Das ist normal und gewünscht!

**F: Wo sehe ich die tatsächlichen Prognosen (mit Charts)?**
A: Im **API-UI Dashboard** unter http://localhost:8000/ui - dort kannst du Prognosen generieren und sofort visualisiert sehen!

**F: Was ist der Unterschied zwischen Grafana und API-UI?**
A: 
- **Grafana** = Performance-Monitoring (Model Drift, MAE, MAPE)
- **API-UI** = Aktuelle Vorhersagen generieren & visualisieren

**F: Was ist der Unterschied zwischen MAE und MAPE?**
A: MAE sagt dir "um wie viel MW", MAPE sagt dir "um wie viel %". Benutze MAPE für Vergleiche zwischen unterschiedlich großen Werten.

**F: Warum ist Drift plötzlich 1.0?**
A: Das Modell performt viel schlechter als am Anfang. Wahrscheinlich Jahreszeit oder Trend hat sich geändert. Zeit für Retraining!

**F: Kann das Dashboard über Nacht kaputt gehen?**
A: Nein! Es speichert alle Daten. Beim Neustart sind alle Metriken wieder da.

---

## 🔗 Quicklinks

- 📊 **API Prognose-Dashboard:** http://localhost:8000/ui
- 📈 **Grafana Monitoring:** http://localhost:3000
- 🔧 **API Dokumentation:** http://localhost:8000/docs
- 📚 **Integration Guide:** [PREDICTIONS_AND_GRAFANA.md](PREDICTIONS_AND_GRAFANA.md)

---

## 💬 Häufige Fragen

**F: Warum ändern sich die Zahlen ständig?**
A: Das System generiert ständig neue Vorhersagen und vergleicht sie mit Realwerten. Das ist normal und gewünscht!

**F: Was ist der Unterschied zwischen MAE und MAPE?**
A: MAE sagt dir "um wie viel MW", MAPE sagt dir "um wie viel %". Benutze MAPE für Vergleiche zwischen unterschiedlich großen Werten.

**F: Warum ist Drift plötzlich 1.0?**
A: Das Modell performt viel schlechter als am Anfang. Wahrscheinlich Jahreszeit oder Trend hat sich geändert. Zeit für Retraining!

**F: Kann das Dashboard über Nacht kaputt gehen?**
A: Nein! Es speichert alle Daten. Beim Neustart sind alle Metriken wieder da.
