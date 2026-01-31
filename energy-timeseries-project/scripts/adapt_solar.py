#!/usr/bin/env python
# Adapt consumption pipeline to solar

with open('scripts/run_solar_extended_pipeline.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Global replacements
replacements = {
    'consumption': 'solar',
    'Consumption': 'Solar',
    'CONSUMPTION': 'SOLAR',
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open('scripts/run_solar_extended_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Adapted solar script!")
