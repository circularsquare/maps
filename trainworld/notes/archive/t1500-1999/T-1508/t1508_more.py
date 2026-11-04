"""T-1508: tail share by band for a few countries, and each country's biggest far pairs (>= 700 km)."""
import json, sys, collections
d = json.load(open(sys.argv[1]))
P = d["places"]
BANDS = [100, 300, 700, 1500, 1e9]
agg = collections.defaultdict(lambda: [[0., 0.] for _ in BANDS[:-1]])
pair = collections.defaultdict(lambda: [0., 0., 0.])
for i, j, km, g, t in d["pairs"]:
    if P[i]["country"] != P[j]["country"]:
        continue
    b = max(k for k in range(4) if km >= BANDS[k])
    agg[P[i]["country"]][b][0] += g
    agg[P[i]["country"]][b][1] += t
    x = pair[(min(i, j), max(i, j))]
    x[0] += g; x[1] += t; x[2] = km
for c in ("CN", "US", "JP", "DE", "FR"):
    print(c, " | ".join(f"{BANDS[k]}+: ground {a[0]*425/1e6:.0f}M tail {a[1]*425/1e6:.0f}M" for k, a in enumerate(agg[c])))
for c in ("CN", "US", "JP"):
    top = sorted(((v[0] + v[1], k, v) for k, v in pair.items() if P[k[0]]["country"] == c and v[2] >= 700), reverse=True)[:15]
    print(c, "far pairs (both ways a weekday, ground/tail, km, people):")
    for tot, (a, b), v in top:
        print(f"  {P[a]['zone']}-{P[b]['zone']} {tot/1e3:.1f}k ({v[0]/1e3:.1f}/{v[1]/1e3:.1f}) {v[2]:.0f} km, {P[a]['pop']/1e6:.1f}M & {P[b]['pop']/1e6:.1f}M")
# Beijing's departures by band (directed from bei)
idx = {p["zone"]: k for k, p in enumerate(P)}
for z in ("bei", "sha", "nyc", "tyo"):
    i = idx[z]
    by = [[0., 0.] for _ in BANDS[:-1]]
    for a, j, km, g, t in d["pairs"]:
        if a == i:
            b = max(k for k in range(4) if km >= BANDS[k])
            by[b][0] += g; by[b][1] += t
    print(z, f"pop {P[i]['pop']/1e6:.1f}M dep {P[i]['dep']/1e3:.0f}k/day;", " | ".join(f"{BANDS[k]}+: g {x[0]/1e3:.1f}k t {x[1]/1e3:.1f}k" for k, x in enumerate(by)))
