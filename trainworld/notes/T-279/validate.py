"""T-279: the built router's drives (market_drives.py's cache) against OSRM's on the research sample.
python -I validate.py <work dir> [pairs.json]   (default artifacts/trainworld/t279/pairs.json)"""
import json, math, os, sys
import numpy as np

work = sys.argv[1]
pairs = json.load(open(sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts", "trainworld", "t279", "pairs.json")))
cache = {}
src = os.path.join(work, "sources")
for f in os.listdir(src):
    cache[f[:-5]] = json.load(open(os.path.join(src, f)))["res"]
rule = lambda d: 15 + 1.25 * d / 85 * 60
rows = {"rule": [], "research v8": [], "built": [], "built km": [], "research km": []}
named = []
ferries = 0
for p in pairs:
    a, b = p["a"], p["b"]
    v = cache.get(a, {}).get(b) or cache.get(b, {}).get(a)
    if v is None or not p.get("osrm_min"):
        continue
    m, L, F, nf = v
    if p.get("snap_m", 0) > 3000 or p["osrm_km"] < p["gc_km"]:
        continue
    o = p["osrm_min"]
    T = 24.0 + 1.252 * m
    if nf:
        ferries += 1
        T += nf * (30 + 50 * 1.5) + F * (60 / 35 + 0.75)  # perceived ferry minutes, a rough match
    rows["rule"].append(rule(p["gc_km"]) / o - 1)
    rows["research v8"].append((24.0 + 1.252 * p["v8_min"]) / o - 1)
    rows["built"].append(T / o - 1)
    rows["built km"].append(1.107 * L / p["osrm_km"] - 1)
    rows["research km"].append(1.107 * p["v8_km"] / p["osrm_km"] - 1)
    if p.get("name"):
        named.append((p["name"], round(o), round(rule(p["gc_km"])), round(24 + 1.252 * p["v8_min"]), round(T), nf))
for k, e in rows.items():
    e = np.abs(np.array(e)) * 100
    print(f"{k:12s} n {len(e)}  % off median {np.median(e):.1f}  90th {np.percentile(e, 90):.1f}")
print("pairs with a ferry", ferries)
print("named: OSRM / rule / research router / built (minutes), ferries")
for n in sorted(named):
    print(" ", *n)
