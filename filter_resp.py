# filter_resp.py
"""Filter resp.json using a threshold and write resp_filtered.json"""
import json

THRESH = 0.01

with open('resp.json','r',encoding='utf8') as fh:
    r = json.load(fh)
filtered = [it for it in r.get('results',[]) if float(it.get('score',0)) > THRESH]
with open('resp_filtered.json','w',encoding='utf8') as fh:
    json.dump({"results": filtered}, fh, indent=2, ensure_ascii=False)
print('Wrote resp_filtered.json with', len(filtered), 'items')
