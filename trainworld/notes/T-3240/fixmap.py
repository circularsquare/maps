"""Airports to the most populous place whose centre is within 60 km (a city's airports often lie in a suburb's zone)."""
import json, math, numpy as np
TW = r"C:\Users\anita\projects\maps\trainworld"
rows = json.load(open(TW + r"\app\src\game\market-rows.json", encoding="utf-8"))
Z = list(rows); lon = np.array([rows[z]["lon"] for z in Z]); lat = np.array([rows[z]["lat"] for z in Z]); pop = np.array([rows[z]["pop"] for z in Z])
f = json.load(open(r"C:\Users\anita\projects\maps\flights\routes_fr24.json", encoding="utf-8"))
out = {}
for c, v in f["airports"].items():
    p, q = math.radians(v[1]), np.radians(lat)
    h = np.sin((q - p) / 2) ** 2 + math.cos(p) * np.cos(q) * np.sin(np.radians(lon - v[0]) / 2) ** 2
    d = 2 * 6371 * np.arcsin(np.sqrt(np.minimum(h, 1)))
    k = np.nonzero(d < 60)[0]
    out[c] = Z[k[np.argmax(pop[k])]] if len(k) else None
json.dump(out, open("airport_zone.json", "w"))
print(out.get("CDG"), out.get("ORY"), out.get("WAW"), out.get("BER"), out.get("KUL"), out.get("SIN"))
