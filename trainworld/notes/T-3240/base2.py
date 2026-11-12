"""T-3254: the market's trips with no pull, rows only (as notes/T-3240/scen.py), saved once for the scenarios."""
import json
import numpy as np
TW = r"C:\Users\anita\projects\maps\trainworld"
HERE = r"C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\2ba1fd73-4c91-45ba-b838-fe8f908d4b21\scratchpad\pull2"
rows = json.load(open(TW + r"\app\src\game\market-rows.json", encoding="utf-8"))
rule = json.load(open(TW + r"\app\src\game\registry.json", encoding="utf-8"))["market"]["rule"]
OPEN = set("AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE NO IS CH LI GB".split())
Z = list(rows); R = [rows[z] for z in Z]; n = len(Z)
pix = {r["place"]: i for i, r in enumerate(R)}
lat = np.radians([r["lat"] for r in R]); lon = np.radians([r["lon"] for r in R])
cc = np.array([r["country"] for r in R])
inc = np.minimum(np.array([r["inc"] for r in R]), 1.)
g = np.array([r["g"] for r in R]); t = np.array([r["t"] for r in R])
a = np.array([r["a"] for r in R]); ta = np.array([r["ta"] for r in R])
pop = np.array([r["pop"] for r in R]); dep = np.array([r["dep"] for r in R])
openm = np.array([c in OPEN for c in cc])
d = np.zeros((n, n), dtype=np.float32)
for i in range(n):
    h = np.sin((lat - lat[i]) / 2) ** 2 + np.cos(lat[i]) * np.cos(lat) * np.sin((lon - lon[i]) / 2) ** 2
    d[i] = 2 * 6371.0088 * np.arcsin(np.sqrt(np.minimum(h, 1)))
on = lambda x, lo, hi: np.clip((x - lo) / (hi - lo), 0, 1)
T = np.zeros((n, n), dtype=np.float32)
for i in range(n):
    di = d[i].astype(np.float64)
    gr = on(di, *rule["onset_km"]) * np.maximum(di, rule["d0_km"]) ** -rule["A"] * (1 - on(di, *rule["tail_onset_km"]))
    tl = on(di, *rule["tail_onset_km"]) * np.maximum(di, rule["D1_km"]) ** -rule["G"]
    rel = np.minimum(inc / inc[i], 1) ** rule["eps_t"] if inc[i] > 0 else 1
    gen = rule["border"] * inc[i] ** rule["border_eps"]
    b = np.where(cc == cc[i], 1., np.where(openm & openm[i], max(0.5, gen), gen))
    T[i] = b * (g[i] * gr * a + t[i] * tl * ta * rel)
    T[i, i] = 0
for i in range(n):
    for q, v in R[i].get("near", []):
        j = pix.get(q)
        if j is not None:
            T[i, j] = v
            if not any(x[0] == R[i]["place"] for x in R[j].get("near", [])):
                T[j, i] = v
np.savez(HERE + r"\base.npz", T=T, d=d, cc=cc, pop=pop, dep=dep, Z=np.array(Z))
print("saved", n)
