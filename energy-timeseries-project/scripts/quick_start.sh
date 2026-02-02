#!/bin/bash
# Quick-Start-Skript für Präsentationsgenerierung
# Autor: Advanced Time Series Project

set -e  # Exit bei Fehler

echo "🎤 Advanced Time Series Präsentationsgenerator"
echo "=============================================="
echo ""

# Farben für Output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Prüfe, ob wir im richtigen Verzeichnis sind
if [ ! -f "generate_presentation.py" ]; then
    echo -e "${RED}❌ Fehler: generate_presentation.py nicht gefunden!${NC}"
    echo "Bitte führe das Skript im scripts/ Verzeichnis aus:"
    echo "  cd energy-timeseries-project/scripts"
    echo "  ./quick_start.sh"
    exit 1
fi

# Prüfe, ob die Markdown-Datei existiert
if [ ! -f "../VORTRAG_ADVANCED_TIME_SERIES.md" ]; then
    echo -e "${RED}❌ Fehler: VORTRAG_ADVANCED_TIME_SERIES.md nicht gefunden!${NC}"
    exit 1
fi

echo -e "${BLUE}📋 Prüfe installierte Tools...${NC}"
echo ""

# Funktion zum Prüfen von Befehlen
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✅ $1 gefunden${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  $1 nicht gefunden${NC}"
        return 1
    fi
}

# Prüfe Python
if ! check_command python3; then
    echo -e "${RED}❌ Python3 ist erforderlich!${NC}"
    exit 1
fi

# Prüfe optionale Tools
REVEAL_INSTALLED=false
MARP_INSTALLED=false
PANDOC_INSTALLED=false

check_command reveal-md && REVEAL_INSTALLED=true
check_command marp && MARP_INSTALLED=true
check_command pandoc && PANDOC_INSTALLED=true

echo ""
echo -e "${BLUE}📊 Verfügbare Formate:${NC}"
echo "  1) Standalone HTML (keine Dependencies) ✅"
$REVEAL_INSTALLED && echo "  2) reveal.js (interaktiv) ✅" || echo "  2) reveal.js (interaktiv) ❌ (npm install -g reveal-md)"
$MARP_INSTALLED && echo "  3) Marp (minimalistisch) ✅" || echo "  3) Marp (minimalistisch) ❌ (npm install -g @marp-team/marp-cli)"
$PANDOC_INSTALLED && echo "  4) PDF/Beamer ✅" || echo "  4) PDF/Beamer ❌ (sudo apt-get install pandoc texlive-latex-extra)"
echo ""

# Interaktive Auswahl
echo -e "${BLUE}🎯 Was möchtest du generieren?${NC}"
echo "  [1] Nur Standalone HTML (schnell, keine Dependencies)"
echo "  [2] reveal.js (empfohlen für Live-Präsentation)"
echo "  [3] Alle verfügbaren Formate"
echo "  [4] Custom (manuelle Auswahl)"
echo ""
read -p "Wähle eine Option [1-4]: " choice

case $choice in
    1)
        echo ""
        echo -e "${GREEN}🚀 Generiere Standalone HTML...${NC}"
        python3 generate_presentation.py --format html
        ;;
    2)
        if [ "$REVEAL_INSTALLED" = false ]; then
            echo -e "${RED}❌ reveal-md ist nicht installiert!${NC}"
            echo "Installation: npm install -g reveal-md"
            exit 1
        fi
        
        echo ""
        echo -e "${BLUE}🎨 Wähle ein Theme:${NC}"
        echo "  [1] black (dunkel, klassisch)"
        echo "  [2] white (hell, sauber)"
        echo "  [3] sky (blau, professionell) 🌟 empfohlen"
        echo "  [4] league (grau/orange)"
        echo "  [5] night (dunkelblau)"
        echo ""
        read -p "Wähle Theme [1-5]: " theme_choice
        
        case $theme_choice in
            1) THEME="black" ;;
            2) THEME="white" ;;
            3) THEME="sky" ;;
            4) THEME="league" ;;
            5) THEME="night" ;;
            *) THEME="sky" ;;
        esac
        
        echo ""
        echo -e "${GREEN}🚀 Generiere reveal.js mit Theme '$THEME'...${NC}"
        python3 generate_presentation.py --format revealjs --theme "$THEME"
        ;;
    3)
        echo ""
        echo -e "${GREEN}🚀 Generiere alle verfügbaren Formate...${NC}"
        python3 generate_presentation.py --format all
        ;;
    4)
        echo ""
        echo -e "${BLUE}📝 Gib deine Optionen ein (z.B. --format pdf --output my_slides):${NC}"
        read -p "Optionen: " custom_options
        python3 generate_presentation.py $custom_options
        ;;
    *)
        echo -e "${RED}❌ Ungültige Auswahl!${NC}"
        exit 1
        ;;
esac

# Prüfe Erfolg
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Präsentation erfolgreich generiert!${NC}"
    echo ""
    echo -e "${BLUE}📂 Output-Verzeichnis:${NC} presentation_output/"
    echo ""
    
    # Liste generierte Dateien
    if [ -d "presentation_output" ]; then
        echo -e "${BLUE}📄 Generierte Dateien:${NC}"
        ls -lh presentation_output/ | grep -v "^total" | grep -v "^d" | awk '{print "  " $9 " (" $5 ")"}'
        echo ""
    fi
    
    # Biete an, die Präsentation zu öffnen
    echo -e "${BLUE}🌐 Möchtest du die Präsentation öffnen?${NC}"
    
    if [ -f "presentation_output/presentation_revealjs.html" ]; then
        echo "  [1] reveal.js im Browser öffnen"
    fi
    if [ -f "presentation_output/presentation_standalone.html" ]; then
        echo "  [2] Standalone HTML im Browser öffnen"
    fi
    if [ -f "presentation_output/presentation_beamer.pdf" ]; then
        echo "  [3] PDF öffnen"
    fi
    echo "  [0] Nein, später"
    echo ""
    read -p "Wähle eine Option: " open_choice
    
    case $open_choice in
        1)
            if command -v xdg-open &> /dev/null; then
                xdg-open presentation_output/presentation_revealjs.html
            else
                echo "Öffne: file://$(pwd)/presentation_output/presentation_revealjs.html"
            fi
            ;;
        2)
            if command -v xdg-open &> /dev/null; then
                xdg-open presentation_output/presentation_standalone.html
            else
                echo "Öffne: file://$(pwd)/presentation_output/presentation_standalone.html"
            fi
            ;;
        3)
            if command -v xdg-open &> /dev/null; then
                xdg-open presentation_output/presentation_beamer.pdf
            else
                echo "Öffne: file://$(pwd)/presentation_output/presentation_beamer.pdf"
            fi
            ;;
        0)
            echo "OK, du findest die Dateien in: presentation_output/"
            ;;
    esac
    
    # Optional: Lokalen Server starten
    if [ -f "presentation_output/presentation_revealjs.html" ] || [ -f "presentation_output/presentation_standalone.html" ]; then
        echo ""
        echo -e "${BLUE}💡 Tipp: Für beste Ergebnisse, starte einen lokalen Server:${NC}"
        echo "  cd presentation_output"
        echo "  python3 -m http.server 8000"
        echo "  Dann öffne: http://localhost:8000"
        echo ""
        
        read -p "Lokalen Server jetzt starten? [y/N]: " start_server
        if [[ $start_server =~ ^[Yy]$ ]]; then
            echo ""
            echo -e "${GREEN}🚀 Starte Server auf http://localhost:8000${NC}"
            echo "Drücke Ctrl+C zum Beenden"
            cd presentation_output
            python3 -m http.server 8000
        fi
    fi
else
    echo ""
    echo -e "${RED}❌ Fehler bei der Generierung!${NC}"
    echo "Prüfe die Fehlermeldungen oben."
    exit 1
fi
