#!/usr/bin/env python3
"""
Skript zur Generierung einer Slides-Präsentation aus VORTRAG_ADVANCED_TIME_SERIES.md

Unterstützt mehrere Präsentations-Frameworks:
1. reveal.js (via reveal-md) - Empfohlen!
2. Marp - Minimalistische Alternative
3. Pandoc + Beamer - PDF-Export

Autor: Advanced Time Series Project
Datum: Februar 2026
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse


class PresentationGenerator:
    """Konvertiert Markdown zu Slides mit verschiedenen Tools"""
    
    def __init__(self, input_file: str, output_dir: str = "presentation_output"):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input-Datei nicht gefunden: {input_file}")
    
    def check_dependency(self, command: str) -> bool:
        """Prüft, ob ein Befehl verfügbar ist"""
        try:
            subprocess.run([command, "--version"], 
                          capture_output=True, 
                          check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def generate_revealjs(self, theme: str = "black", 
                          transition: str = "slide") -> Path:
        """
        Generiert reveal.js Präsentation mit reveal-md
        
        Installation: npm install -g reveal-md
        
        Args:
            theme: reveal.js Theme (black, white, league, beige, sky, night, serif, simple, solarized)
            transition: Slide-Übergang (none, fade, slide, convex, concave, zoom)
        
        Returns:
            Path zur generierten HTML-Datei
        """
        print("🎨 Generiere reveal.js Präsentation...")
        
        if not self.check_dependency("reveal-md"):
            print("❌ reveal-md nicht gefunden!")
            print("Installation: npm install -g reveal-md")
            return None
        
        output_file = self.output_dir / "presentation_revealjs.html"
        
        # reveal-md Kommando
        cmd = [
            "reveal-md",
            str(self.input_file),
            "--theme", theme,
            "--transition", transition,
            "--static", str(output_file),
            "--separator", "^---$",  # Horizontal separator
            "--separator-vertical", "^----$",  # Vertical separator (optional)
            "--highlight-theme", "monokai",  # Code-Highlighting
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ reveal.js Präsentation erstellt: {output_file}")
            return output_file
        except subprocess.CalledProcessError as e:
            print(f"❌ Fehler bei reveal.js Generierung: {e}")
            return None
    
    def generate_marp(self, theme: str = "default") -> Path:
        """
        Generiert Marp Präsentation
        
        Installation: npm install -g @marp-team/marp-cli
        
        Args:
            theme: Marp Theme (default, gaia, uncover)
        
        Returns:
            Path zur generierten HTML-Datei
        """
        print("🎨 Generiere Marp Präsentation...")
        
        if not self.check_dependency("marp"):
            print("❌ marp nicht gefunden!")
            print("Installation: npm install -g @marp-team/marp-cli")
            return None
        
        output_file = self.output_dir / "presentation_marp.html"
        
        # Marp erwartet Frontmatter für Themes
        # Wir erstellen eine temporäre Datei mit Frontmatter
        temp_file = self.output_dir / "_temp_marp.md"
        
        frontmatter = f"""---
marp: true
theme: {theme}
paginate: true
backgroundColor: #fff
---

"""
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(frontmatter + content)
        
        # Marp Kommando
        cmd = [
            "marp",
            str(temp_file),
            "--html",
            "--allow-local-files",
            "-o", str(output_file)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            temp_file.unlink()  # Temporäre Datei löschen
            print(f"✅ Marp Präsentation erstellt: {output_file}")
            return output_file
        except subprocess.CalledProcessError as e:
            print(f"❌ Fehler bei Marp Generierung: {e}")
            if temp_file.exists():
                temp_file.unlink()
            return None
    
    def generate_beamer_pdf(self) -> Path:
        """
        Generiert PDF-Präsentation mit Pandoc + Beamer
        
        Installation: 
        - Ubuntu: sudo apt-get install pandoc texlive-latex-extra
        - macOS: brew install pandoc basictex
        
        Returns:
            Path zur generierten PDF-Datei
        """
        print("📄 Generiere PDF mit Pandoc + Beamer...")
        
        if not self.check_dependency("pandoc"):
            print("❌ pandoc nicht gefunden!")
            print("Installation: sudo apt-get install pandoc texlive-latex-extra")
            return None
        
        output_file = self.output_dir / "presentation_beamer.pdf"
        
        # Pandoc Kommando
        cmd = [
            "pandoc",
            str(self.input_file),
            "-t", "beamer",
            "--slide-level=2",
            "-V", "theme:Madrid",
            "-V", "colortheme:dolphin",
            "-o", str(output_file)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ PDF-Präsentation erstellt: {output_file}")
            return output_file
        except subprocess.CalledProcessError as e:
            print(f"❌ Fehler bei PDF Generierung: {e}")
            return None
    
    def generate_standalone_html(self) -> Path:
        """
        Generiert einfache HTML-Präsentation mit Custom CSS
        Nutzt Python markdown für korrektes Rendering
        
        Returns:
            Path zur generierten HTML-Datei
        """
        print("🌐 Generiere standalone HTML...")
        
        output_file = self.output_dir / "presentation_standalone.html"
        
        # Lese Markdown
        with open(self.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Teile nach "---" für Slides
        slides = content.split('\n---\n')
        
        # Versuche markdown-Library zu nutzen
        try:
            import markdown
            from markdown.extensions.tables import TableExtension
            from markdown.extensions.fenced_code import FencedCodeExtension
            from markdown.extensions.codehilite import CodeHiliteExtension
            
            md = markdown.Markdown(extensions=[
                'tables',
                'fenced_code',
                'codehilite',
                'nl2br'
            ])
            
            slides_html = ""
            for i, slide in enumerate(slides):
                # Konvertiere Markdown zu HTML
                slide_html = md.convert(slide)
                md.reset()  # Reset für nächste Slide
                slides_html += f'<div class="slide">{slide_html}</div>\n'
        
        except ImportError:
            # Fallback auf einfache Konvertierung
            print("⚠️  markdown library nicht gefunden, nutze einfache Konvertierung")
            import html as html_module
            slides_html = ""
            for i, slide in enumerate(slides):
                slide_content = html_module.escape(slide)
                # Einfache Markdown-zu-HTML Konvertierung
                slide_content = slide_content.replace('\n\n', '</p><p>')
                slide_content = slide_content.replace('\n### ', '<h3>')
                slide_content = slide_content.replace('\n## ', '<h2>')
                slide_content = slide_content.replace('\n# ', '<h1>')
                slide_content = f'<div class="slide"><p>{slide_content}</p></div>\n'
                slides_html += slide_content
        
        # HTML Template - verwende String-Konkatenation statt .format()
        html_output = """<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Time Series Forecasting</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            overflow: hidden;
        }
        .slide {
            display: none;
            width: 100vw;
            height: 100vh;
            padding: 60px;
            background: white;
            overflow-y: auto;
        }
        .slide.active { display: block; }
        .slide h1 { 
            color: #667eea; 
            font-size: 3em; 
            margin-bottom: 20px;
            border-bottom: 4px solid #764ba2;
            padding-bottom: 10px;
        }
        .slide h2 { 
            color: #764ba2; 
            font-size: 2em; 
            margin: 30px 0 15px;
        }
        .slide h3 { 
            color: #667eea; 
            font-size: 1.5em; 
            margin: 20px 0 10px;
        }
        .slide table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .slide th, .slide td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .slide th {
            background: #667eea;
            color: white;
            font-weight: bold;
        }
        .slide tr:hover { background: #f5f5f5; }
        .slide code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        .slide pre {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 20px 0;
            font-size: 0.85em;
            line-height: 1.4;
        }
        .slide pre code {
            background: none;
            color: inherit;
            padding: 0;
        }
        .slide ul, .slide ol {
            margin-left: 40px;
            line-height: 1.8;
            margin-bottom: 15px;
        }
        .slide li {
            margin-bottom: 8px;
        }
        .slide strong { 
            color: #764ba2; 
            font-weight: 600;
        }
        .slide em {
            color: #667eea;
            font-style: italic;
        }
        .slide blockquote {
            border-left: 4px solid #667eea;
            padding-left: 20px;
            margin: 20px 0;
            color: #555;
            font-style: italic;
        }
        .slide p {
            margin-bottom: 15px;
            line-height: 1.6;
        }
        .slide img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            margin: 20px 0;
        }
        .controls {
            position: fixed;
            bottom: 30px;
            right: 30px;
            display: flex;
            gap: 15px;
            z-index: 1000;
        }
        .controls button {
            background: #667eea;
            color: white;
            border: none;
            padding: 15px 25px;
            font-size: 18px;
            border-radius: 50px;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            transition: all 0.3s ease;
        }
        .controls button:hover {
            background: #764ba2;
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(118, 75, 162, 0.6);
        }
        .slide-number {
            position: fixed;
            bottom: 30px;
            left: 30px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            font-size: 16px;
        }
    </style>
</head>
<body>
    <div id="slides">
""" + slides_html + """
    </div>
    
    <div class="controls">
        <button onclick="prevSlide()">◀ Zurück</button>
        <button onclick="nextSlide()">Weiter ▶</button>
    </div>
    
    <div class="slide-number">
        <span id="current">1</span> / <span id="total">""" + str(len(slides)) + """</span>
    </div>
    
    <script>
        let currentSlide = 0;
        const slides = document.querySelectorAll('.slide');
        const totalSlides = slides.length;
        
        document.getElementById('total').textContent = totalSlides;
        
        function showSlide(n) {
            slides[currentSlide].classList.remove('active');
            currentSlide = (n + totalSlides) % totalSlides;
            slides[currentSlide].classList.add('active');
            document.getElementById('current').textContent = currentSlide + 1;
        }
        
        function nextSlide() { showSlide(currentSlide + 1); }
        function prevSlide() { showSlide(currentSlide - 1); }
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowRight' || e.key === ' ') nextSlide();
            if (e.key === 'ArrowLeft') prevSlide();
        });
        
        // Show first slide
        showSlide(0);
    </script>
</body>
</html>"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_output)
        
        print(f"✅ Standalone HTML erstellt: {output_file}")
        return output_file
    
    def generate_all(self):
        """Generiert alle verfügbaren Formate"""
        print("🚀 Starte Präsentationsgenerierung...\n")
        
        results = {
            "reveal.js": self.generate_revealjs(),
            "Marp": self.generate_marp(),
            "PDF (Beamer)": self.generate_beamer_pdf(),
            "Standalone HTML": self.generate_standalone_html()
        }
        
        print("\n" + "="*60)
        print("📊 ZUSAMMENFASSUNG")
        print("="*60)
        
        for name, path in results.items():
            status = "✅" if path else "❌"
            print(f"{status} {name}: {path if path else 'Fehlgeschlagen'}")
        
        successful = [p for p in results.values() if p]
        if successful:
            print(f"\n🎉 {len(successful)}/{len(results)} Formate erfolgreich generiert!")
            print(f"\n📂 Output-Verzeichnis: {self.output_dir.absolute()}")
        else:
            print("\n⚠️ Keine Präsentationen konnten generiert werden.")
            print("Überprüfe die Installationsanweisungen oben.")


def main():
    parser = argparse.ArgumentParser(
        description="Generiert Slides-Präsentationen aus Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Alle Formate generieren (empfohlen)
  python generate_presentation.py
  
  # Nur reveal.js mit custom Theme
  python generate_presentation.py --format revealjs --theme sky
  
  # Nur PDF
  python generate_presentation.py --format pdf
  
  # Custom Input/Output
  python generate_presentation.py -i custom.md -o my_slides

Installationsanweisungen:
  reveal-md:  npm install -g reveal-md
  Marp:       npm install -g @marp-team/marp-cli
  Pandoc:     sudo apt-get install pandoc texlive-latex-extra
"""
    )
    
    parser.add_argument(
        "-i", "--input",
        default="../VORTRAG_ADVANCED_TIME_SERIES.md",
        help="Input Markdown-Datei (default: ../VORTRAG_ADVANCED_TIME_SERIES.md)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="presentation_output",
        help="Output-Verzeichnis (default: presentation_output)"
    )
    
    parser.add_argument(
        "-f", "--format",
        choices=["all", "revealjs", "marp", "pdf", "html"],
        default="all",
        help="Präsentationsformat (default: all)"
    )
    
    parser.add_argument(
        "-t", "--theme",
        default="black",
        help="Theme für reveal.js (black, white, league, etc.)"
    )
    
    args = parser.parse_args()
    
    # Resolve relative path
    script_dir = Path(__file__).parent
    input_path = script_dir / args.input
    
    try:
        generator = PresentationGenerator(input_path, args.output)
        
        if args.format == "all":
            generator.generate_all()
        elif args.format == "revealjs":
            generator.generate_revealjs(theme=args.theme)
        elif args.format == "marp":
            generator.generate_marp()
        elif args.format == "pdf":
            generator.generate_beamer_pdf()
        elif args.format == "html":
            generator.generate_standalone_html()
    
    except FileNotFoundError as e:
        print(f"❌ Fehler: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
