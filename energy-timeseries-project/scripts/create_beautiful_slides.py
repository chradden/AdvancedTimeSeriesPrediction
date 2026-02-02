#!/usr/bin/env python3
"""Erstellt eine schöne HTML-Präsentation mit richtigem Markdown-Rendering"""

from pathlib import Path
import markdown
from markdown.extensions.tables import TableExtension
from markdown.extensions.fenced_code import FencedCodeExtension

# Lese die Markdown-Datei
input_file = Path("../VORTRAG_ADVANCED_TIME_SERIES.md")
with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Teile in Slides
slides_raw = content.split('\n---\n')

# Markdown-Parser mit Extensions
md = markdown.Markdown(extensions=['tables', 'fenced_code', 'nl2br'])

# Konvertiere jede Slide und korrigiere Bildpfade
slides_html = []
for i, slide_md in enumerate(slides_raw):
    # Korrigiere Bildpfade: results/figures/ -> figures/
    # Da Bilder in presentation_output/figures/ kopiert werden
    import re
    slide_md_fixed = re.sub(r'!\[([^\]]*)\]\(results/figures/', r'![\1](figures/', slide_md)
    
    slide_html = md.convert(slide_md_fixed)
    md.reset()
    slides_html.append(slide_html)

# HTML Template
html_output = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Time Series Forecasting - Präsentation</title>
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

# Füge alle Slides hinzu
for slide_html in slides_html:
    html_output += f'        <div class="slide">{slide_html}</div>\n'

html_output += f"""    </div>
    
    <div class="controls">
        <button onclick="prevSlide()" title="Vorherige Folie (←)">◀ Zurück</button>
        <button onclick="nextSlide()" title="Nächste Folie (→)">Weiter ▶</button>
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

# Speichere Datei
output_file = Path("presentation_output/presentation_beautiful.html")
output_file.parent.mkdir(exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_output)

print(f"✅ Schöne Präsentation erstellt: {output_file}")
print(f"📊 {len(slides_html)} Folien generiert")
