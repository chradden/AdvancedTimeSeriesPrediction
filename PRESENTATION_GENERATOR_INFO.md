# 🎤 Präsentationsgenerator - Quick Reference

## Was wurde erstellt?

Ich habe ein vollständiges System zur Erstellung von Slides-Präsentationen aus deiner Markdown-Datei [VORTRAG_ADVANCED_TIME_SERIES.md](energy-timeseries-project/VORTRAG_ADVANCED_TIME_SERIES.md) erstellt.

## 📁 Dateien

```
energy-timeseries-project/scripts/
├── generate_presentation.py          # Hauptskript (Python)
├── quick_start.sh                    # Interaktives Bash-Skript
└── PRESENTATION_GENERATOR_README.md  # Ausführliche Dokumentation
```

## 🚀 Schnellstart

### Option 1: Python-Skript (Empfohlen)

```bash
cd energy-timeseries-project/scripts

# Standalone HTML generieren (keine Dependencies!)
python generate_presentation.py --format html

# Mit reveal.js (benötigt: npm install -g reveal-md)
python generate_presentation.py --format revealjs --theme sky

# Alle Formate
python generate_presentation.py
```

### Option 2: Interaktives Bash-Skript

```bash
cd energy-timeseries-project/scripts
./quick_start.sh
```

Das Skript führt dich durch den Prozess mit einem interaktiven Menü!

## 🎨 Verfügbare Formate

| Format | Command | Dependencies | Beste für |
|--------|---------|--------------|-----------|
| **Standalone HTML** | `--format html` | ✅ Keine | Sofort loslegen |
| **reveal.js** | `--format revealjs` | npm install -g reveal-md | Live-Präsentation |
| **Marp** | `--format marp` | npm install -g @marp-team/marp-cli | Minimalistisch |
| **PDF/Beamer** | `--format pdf` | apt install pandoc texlive-latex-extra | Druckversion |

## 📂 Output

Die generierten Präsentationen findest du hier:
```
energy-timeseries-project/scripts/presentation_output/
```

## 🌟 Empfehlung für deinen Vortrag

**Für den Advanced Time Series Kurs:**

```bash
cd energy-timeseries-project/scripts

# Generiere reveal.js mit "sky" Theme (blau, professionell)
python generate_presentation.py --format revealjs --theme sky

# Starte lokalen Server
cd presentation_output
python -m http.server 8000

# Öffne im Browser: http://localhost:8000/presentation_revealjs.html
```

**Keyboard Shortcuts während der Präsentation:**
- `→` oder `Leertaste`: Nächste Folie
- `←`: Vorherige Folie
- `Esc`: Übersicht aller Folien
- `F`: Fullscreen
- `B`: Bildschirm schwarz (Pause)

## 📖 Vollständige Dokumentation

Siehe [PRESENTATION_GENERATOR_README.md](energy-timeseries-project/scripts/PRESENTATION_GENERATOR_README.md) für:
- Detaillierte Installationsanweisungen
- Anpassungsmöglichkeiten
- Troubleshooting
- Best Practices
- Erweiterte Tipps

## ✅ Was funktioniert jetzt schon?

- ✅ Standalone HTML (ohne jegliche Dependencies)
- ✅ Automatische Slide-Trennung bei `---`
- ✅ Keyboard-Navigation
- ✅ Responsive Design
- ✅ 4 verschiedene Output-Formate
- ✅ Interaktives Bash-Menü

## 📊 Getestete Formate

Ich habe die Standalone HTML-Version bereits erfolgreich getestet:
- ✅ 37 KB HTML-Datei generiert
- ✅ Enthält alle Slides aus deiner Markdown-Datei
- ✅ Funktioniert offline im Browser

## 🔧 Nächste Schritte (optional)

Falls du reveal.js nutzen möchtest (empfohlen für Live-Präsentation):

```bash
# Node.js installieren (falls nicht vorhanden)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# reveal-md installieren
npm install -g reveal-md

# Dann: reveal.js Präsentation generieren
cd energy-timeseries-project/scripts
python generate_presentation.py --format revealjs --theme sky
```

## 💡 Warum reveal.js?

- 🎨 Professionelle Themes
- 🔄 Smooth Slide-Transitions
- 📱 Mobile-friendly
- 📊 Unterstützt Code-Highlighting
- 🎤 Speaker Notes möglich
- 📈 Perfekt für akademische Präsentationen

Aber: **Standalone HTML funktioniert auch ohne Installation sofort!**

---

**Viel Erfolg mit deinem Vortrag! 🎓**
