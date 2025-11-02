# resp_to_html.py
"""
Render a recommendation JSON (resp.json) into a simple, attractive HTML file (out.html).
This script reads resp.json and writes a styled out.html file.
"""

import json
import html
from pathlib import Path

IN = Path('resp.json')
OUT = Path('out.html')

def render():
    if not IN.exists():
        print('resp.json not found')
        return
    r = json.loads(IN.read_text(encoding='utf8'))
    items = r.get('results', [])
    s = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Recommendations</title>']
    s.append('<style>body{font-family:Inter,Arial; background:#f6fbff;padding:20px} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;max-width:1100px;margin:auto} .card{background:#fff;padding:16px;border-radius:10px;box-shadow:0 8px 20px rgba(2,6,23,0.06)} .title{font-weight:700} .score{color:#059669;font-weight:600}</style>')
    s.append('</head><body><h1 style="text-align:center;color:#0b74de">Medicine Recommendations</h1><div class="grid">')
    for i,it in enumerate(items, start=1):
        name = html.escape(str(it.get('name','')))
        desc = html.escape(str(it.get('desc','')))
        score = it.get('score', it.get('semantic_score', it.get('tfidf_score', 0)))
        s.append(f'<div class="card"><div class="title">{name} <span style="float:right;color:#6b7280;font-size:12px">#{i}</span></div><div class="score">Score: {float(score):.4f}</div><p style="margin-top:8px">{desc}</p></div>')
    s.append('</div><footer style="text-align:center;margin-top:20px;color:#6b7280">⚠️ For informational purposes only. Consult a medical professional.</footer></body></html>')
    OUT.write_text('\n'.join(s), encoding='utf8')
    print('Wrote', OUT)

if __name__=='__main__':
    render()
