"""T-3240: writes market_pull.json, each place's importance pull (the design in notes/T-3240.md section 5).
pull = exp(GAMMA * clip(expected rank - rank, 0, CAP)), rank the best Natural Earth scalerank inside the
zone (place_rank.json), expected rank = A + B ln(people) fitted over every ranked place.
    python make_pull.py <out.json>
"""
import json, math, sys
import numpy as np
HERE = r"C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\2ba1fd73-4c91-45ba-b838-fe8f908d4b21\scratchpad\longdist"
TW = r"C:\Users\anita\projects\maps\trainworld"
GAMMA, CAP, PHASE, OFF = 0.8, 3.0, [50, 150], ["US", "JP", "GB", "KR"]
rows = json.load(open(TW + r"\app\src\game\market-rows.json", encoding="utf-8"))
rank = json.load(open(HERE + r"\place_rank.json", encoding="utf-8"))
z = [k for k in rows if k in rank]
b, a = np.polyfit([math.log(rows[k]["pop"]) for k in z], [rank[k][0] for k in z], 1)
out = {"task": "T-3240", "gamma": GAMMA, "cap": CAP, "phase_km": PHASE, "off": OFF,
       "rank_fit": [round(a, 3), round(b, 3)], "source": "Natural Earth 10m populated places (simple), scalerank",
       "pull": {}}
for k in sorted(z):
    e = a + b * math.log(rows[k]["pop"]) - rank[k][0]
    p = math.exp(GAMMA * min(max(e, 0), CAP))
    if p > 1.05:
        out["pull"][k] = [round(p, 2), rank[k][0], rank[k][1]]
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(len(out["pull"]), "places pulled; top:", sorted(out["pull"].items(), key=lambda x: -x[1][0])[:12])
