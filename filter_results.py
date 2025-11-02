import json
r = json.load(open("resp.json"))
filtered = [x for x in r.get("results", []) if x.get("score", 0) > 0.01]
print(json.dumps(filtered, indent=2))
