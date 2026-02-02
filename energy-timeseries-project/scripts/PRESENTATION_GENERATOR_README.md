# 🎤 Präsentationsgenerator für VORTRAG_ADVANCED_TIME_SERIES.md

Dieses Skript konvertiert die Markdown-Datei automatisch in verschiedene Präsentationsformate.

## 🚀 Quick Start

```bash
cd energy-timeseries-project/scripts
python generate_presentation.py
```

Das Skript erstellt automatisch **4 verschiedene Formate**:
- ✅ reveal.js (HTML, interaktiv) - **Empfohlen!**
- ✅ Marp (HTML, minimalistisch)
- ✅ PDF via Pandoc/Beamer
- ✅ Standalone HTML (ohne Dependencies)

## 📦 Installation der Dependencies

### Option 1: reveal.js (Empfohlen) 🏆

```bash
# Node.js installieren (falls nicht vorhanden)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# reveal-md installieren
npm install -g reveal-md
```

**Vorteile:**
- 🎨 Professionelle Themes (black, white, league, sky, night, serif)
- 🔄 Smooth Transitions
- 📱 Responsive Design
- ⌨️ Keyboard Navigation (Pfeiltasten, Leertaste)
- 🖱️ Touch-Support
- 📊 Code-Highlighting

### Option 2: Marp (Minimalistisch)

```bash
npm install -g @marp-team/marp-cli
```

**Vorteile:**
- 🎯 Einfach & schnell
- 📄 PDF-Export integriert
- 🎨 Custom CSS möglich

### Option 3: Pandoc + Beamer (PDF)

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y pandoc texlive-latex-extra texlive-fonts-recommended

# macOS
brew install pandoc basictex
```

**Vorteile:**
- 📄 Direkter PDF-Export
- 🖨️ Druckfreundlich
- 📊 LaTeX-Qualität

### Option 4: Standalone HTML (Keine Dependencies!)

Keine Installation nötig - funktioniert out-of-the-box!

**Vorteile:**
- 🚀 Sofort einsatzbereit
- 📦 Keine npm/Node.js erforderlich
- 🌐 Pure HTML/CSS/JS

**Nachteile:**
- ⚠️ Einfaches Markdown-Parsing (keine komplexen Features)
- 📊 Keine Bild-Vorschau (Pfade müssen manuell angepasst werden)

## 🎯 Verwendung

### Alle Formate generieren

```bash
python generate_presentation.py
```

### Nur ein spezifisches Format

```bash
# Nur reveal.js
python generate_presentation.py --format revealjs

# Nur Marp
python generate_presentation.py --format marp

# Nur PDF
python generate_presentation.py --format pdf

# Nur Standalone HTML
python generate_presentation.py --format html
```

### Mit Custom Theme (reveal.js)

```bash
python generate_presentation.py --format revealjs --theme sky
```

**Verfügbare Themes:**
- `black` (dunkel, default)
- `white` (hell, sauber)
- `league` (grau/orange)
- `beige` (warm)
- `sky` (blau)
- `night` (dunkelblau)
- `serif` (klassisch)
- `simple` (minimalistisch)
- `solarized` (Solarized-Farbschema)

### Custom Input/Output

```bash
python generate_presentation.py \
  --input ../VORTRAG_ADVANCED_TIME_SERIES.md \
  --output my_presentation
```

## 📂 Output-Struktur

```
presentation_output/
├── presentation_revealjs.html    # reveal.js (empfohlen für Live-Präsentation)
├── presentation_marp.html        # Marp (minimalistisch)
├── presentation_beamer.pdf       # PDF (Beamer)
└── presentation_standalone.html  # Standalone HTML (Fallback)
```

## 🎨 Präsentation öffnen

### reveal.js (empfohlen)

```bash
# In VS Code Simple Browser öffnen
# Oder im Terminal:
xdg-open presentation_output/presentation_revealjs.html

# Live-Server für beste Erfahrung
cd presentation_output
python -m http.server 8000
# Dann öffne: http://localhost:8000/presentation_revealjs.html
```

**Keyboard Shortcuts:**
- `→` oder `Space`: Nächste Slide
- `←`: Vorherige Slide
- `Esc`: Übersicht (Alle Slides)
- `F`: Fullscreen
- `S`: Speaker Notes (falls vorhanden)
- `B`: Bildschirm schwarz (Pause)

### Standalone HTML

```bash
xdg-open presentation_output/presentation_standalone.html
```

**Keyboard Shortcuts:**
- `→` oder `Space`: Nächste Slide
- `←`: Vorherige Slide

### PDF

```bash
xdg-open presentation_output/presentation_beamer.pdf
```

## 🛠️ Anpassungen

### Markdown-Struktur für optimale Slides

Die Datei nutzt bereits das richtige Format:

```markdown
# Titel (Hauptfolie)

---

## Slide 1: Titel

Inhalt...

---

## Slide 2: Nächster Titel

Mehr Inhalt...
```

**Wichtig:**
- `---` trennt Slides (horizontal)
- `----` kann für vertikale Slides genutzt werden (reveal.js only)
- `# Titel` für Hauptüberschriften
- `## Titel` für Slide-Titel
- `### Untertitel` für Untertitel

### Bilder einbinden

Stelle sicher, dass Bildpfade relativ zum Output-Verzeichnis korrekt sind:

```markdown
![Beschreibung](../results/figures/bild.png)
```

Oder kopiere Bilder in das Output-Verzeichnis:

```bash
cp -r ../results/figures presentation_output/
```

Dann in Markdown:

```markdown
![Beschreibung](figures/bild.png)
```

## 🎓 Best Practices

### 1. Verwende reveal.js für Live-Präsentationen

```bash
python generate_presentation.py --format revealjs --theme black
```

**Warum?**
- Professionelles Design
- Smooth Animations
- Interaktiv

### 2. Verwende PDF für Handouts

```bash
python generate_presentation.py --format pdf
```

**Warum?**
- Druckfreundlich
- Offline verfügbar
- Universell kompatibel

### 3. Teste lokal mit http.server

```bash
cd presentation_output
python -m http.server 8000
```

Öffne: http://localhost:8000

**Warum?**
- Bilder laden korrekt
- JavaScript funktioniert ohne CORS-Probleme
- Simuliert echten Webserver

### 4. Speaker View für Präsentationen

Bei reveal.js: Drücke `S` für Speaker Notes

Füge Speaker Notes hinzu:

```markdown
## Slide Titel

Öffentlicher Inhalt...

Note:
- Dies sind private Notizen
- Nur im Speaker View sichtbar
- Mit Timern und nächstem Slide-Preview
```

## 🐛 Troubleshooting

### "reveal-md not found"

```bash
# Node.js installieren
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# reveal-md neu installieren
npm install -g reveal-md
```

### "pandoc not found"

```bash
sudo apt-get update
sudo apt-get install -y pandoc texlive-latex-extra
```

### Bilder werden nicht angezeigt

**Option 1: HTTP-Server nutzen**
```bash
cd presentation_output
python -m http.server 8000
```

**Option 2: Bilder kopieren**
```bash
cp -r ../results/figures presentation_output/
```

**Option 3: Absolute Pfade (nicht empfohlen)**
```bash
# In Markdown verwende absolute Pfade
file:///workspaces/AdvancedTimeSeriesPrediction/energy-timeseries-project/results/figures/bild.png
```

### "Permission denied"

```bash
chmod +x generate_presentation.py
```

## 📊 Vergleich der Formate

| Format | Vorteile | Nachteile | Use Case |
|--------|----------|-----------|----------|
| **reveal.js** | ✅ Professionell<br>✅ Interaktiv<br>✅ Themes | ❌ Braucht npm | Live-Präsentation |
| **Marp** | ✅ Einfach<br>✅ Schnell | ❌ Weniger Features | Schnelle Slides |
| **PDF (Beamer)** | ✅ Druckbar<br>✅ Offline | ❌ Nicht interaktiv | Handouts |
| **Standalone** | ✅ No dependencies | ❌ Basic Features | Fallback |

## 🎯 Empfehlung für VORTRAG_ADVANCED_TIME_SERIES.md

**Für Live-Präsentation im Kurs:**
```bash
python generate_presentation.py --format revealjs --theme sky
cd presentation_output
python -m http.server 8000
```

Öffne: http://localhost:8000/presentation_revealjs.html

**Theme `sky`** passt perfekt:
- 🌟 Professionell aber nicht zu dunkel
- 📊 Gut für Charts/Tabellen
- 🎨 Blau-Töne passen zu Data Science

**Für Submission/Upload:**
```bash
python generate_presentation.py --format pdf
```

Reiche `presentation_beamer.pdf` ein.

## 💡 Erweiterte Tipps

### Custom CSS für reveal.js

Erstelle `custom.css`:

```css
.reveal h1 {
    color: #667eea;
    text-transform: uppercase;
}

.reveal section img {
    border: none;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
```

Nutze es:

```bash
reveal-md VORTRAG_ADVANCED_TIME_SERIES.md \
  --theme black \
  --css custom.css \
  --static presentation_output/presentation_custom.html
```

### Automatisches Reload bei Änderungen

```bash
# reveal-md mit Live-Reload
reveal-md VORTRAG_ADVANCED_TIME_SERIES.md --watch
```

Öffne: http://localhost:1948

Bearbeite Markdown → Browser aktualisiert automatisch!

### Export zu PowerPoint

```bash
# Mit Pandoc
pandoc VORTRAG_ADVANCED_TIME_SERIES.md -o presentation.pptx
```

**Achtung:** Formatierung kann verloren gehen!

## 📚 Weitere Ressourcen

- [reveal.js Dokumentation](https://revealjs.com/)
- [Marp Dokumentation](https://marp.app/)
- [Pandoc Manual](https://pandoc.org/MANUAL.html)
- [Markdown Syntax](https://www.markdownguide.org/)

## 🤝 Support

Bei Problemen:
1. Prüfe `--help`: `python generate_presentation.py --help`
2. Teste Standalone HTML (keine Dependencies)
3. Checke Installationen: `reveal-md --version`, `pandoc --version`

## 🎉 Viel Erfolg mit deiner Präsentation!
