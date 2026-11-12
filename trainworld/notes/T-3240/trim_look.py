"""T-3240 code stage: look at the places the pull reaches, with their own people and their hinterland's
(people of other places within R km of the centre), to find a trim rule for remote centres.
    python trim_look.py [R]
"""
import json, math, sys
import numpy as np
LD = r"C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\2ba1fd73-4c91-45ba-b838-fe8f908d4b21\scratchpad\longdist"
TW = r"C:\Users\anita\projects\maps\trainworld"
R = float(sys.argv[1]) if len(sys.argv) > 1 else 300.
rows = json.load(open(TW + r"\app\src\game\market-rows.json", encoding="utf-8"))
rank = json.load(open(LD + r"\place_rank.json", encoding="utf-8"))
Z = list(rows)
lat = np.radians([rows[z]["lat"] for z in Z]); lon = np.radians([rows[z]["lon"] for z in Z])
pop = np.array([rows[z]["pop"] for z in Z])
has = [z in rank for z in Z]
rk = np.array([rank[z][0] if z in rank else 99 for z in Z], float)
m = rk < 99
b, a = np.polyfit(np.log(pop[m]), rk[m], 1)
ex = np.where(m, a + b * np.log(pop) - rk, 0)
H = np.zeros(len(Z))
for i in range(len(Z)):
    h = np.sin((lat - lat[i]) / 2) ** 2 + np.cos(lat[i]) * np.cos(lat) * np.sin((lon - lon[i]) / 2) ** 2
    d = 2 * 6371 * np.arcsin(np.sqrt(np.minimum(h, 1)))
    H[i] = pop[(d < R)].sum()  # own included
np.save(LD + r"\..\pull\H%d.npy" % R, H)
o = np.argsort(-ex)
print(f"rank fit {a:.2f} {b:+.3f} ln(pop); R {R} km (own people included)")
for i in o[:90]:
    z = Z[i]
    print(f"{z} {rows[z]['country']} {rank[z][1][:18]:18s} rank {rk[i]:.0f} pop {pop[i]/1e6:6.2f}M within {R:.0f} km {H[i]/1e6:7.2f}M excess {ex[i]:.2f}")
