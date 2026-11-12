"""T-3240: the pull's strength across borders against FR24's airport-pair passengers (maps/flights, read here only).
Each airport goes to the market place whose outline holds it; place pairs both of at least 1M people, 700 km
and more apart, with 100k passengers a year or more. y = ln(FR24 a year / (model both ways a weekday)), fitted on
x = excess_i + excess_j and ln km, separately for pairs within a country and across a border; the slope on x is
what the pull still misses (0: none). Countries in `off` left out (their residents see no pull).
    python fr24fit.py TAG [TAG ...]       (scen_<TAG>.npz from scen.py --save)
"""
import json, math, sys, os
import numpy as np
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
import market_pull
HERE = os.path.dirname(os.path.abspath(__file__))
TW = r"C:\Users\anita\projects\maps\trainworld"
rows = json.load(open(TW + r"\app\src\game\market-rows.json", encoding="utf-8"))
Z = list(rows); ix = {z: i for i, z in enumerate(Z)}
cache = HERE + r"\airport_zone.json"
f = json.load(open(r"C:\Users\anita\projects\maps\flights\routes_fr24.json", encoding="utf-8"))
if not os.path.exists(cache):
    geoms, of, tree = market_pull.outlines(Z)
    codes = list(f["airports"])
    at = market_pull.locate([(f["airports"][c][0], f["airports"][c][1]) for c in codes], geoms, of, tree)
    json.dump(dict(zip(codes, at)), open(cache, "w"))
az = json.load(open(cache))
cfg = market_pull.load()
off = set(cfg["off"]) | set(sys.argv[sys.argv.index("--skip") + 1].split(",") if "--skip" in sys.argv else [])
pp = {}
for a_, b_, n, *_ in f["routes"]:
    za, zb = az.get(a_), az.get(b_)
    if za and zb and za != zb:
        k = (min(za, zb), max(za, zb))
        pp[k] = pp.get(k, 0) + n
cc = {z: rows[z]["country"] for z in Z}
for tag in [a for a in sys.argv[1:] if not a.startswith("--") and "," not in a and a not in ("CN",)]:
    m = np.load(HERE + rf"\scen_{tag}.npz")
    Tn, d, ex = m["Tn"], m["d"], m["ex"]
    ex = np.array([cfg["places"].get(z, [0.])[0] for z in Z])  # the table's, for x
    out = {}
    for (za, zb), n in pp.items():
        i, j = ix[za], ix[zb]
        if rows[za]["pop"] < 1e6 or rows[zb]["pop"] < 1e6 or d[i, j] < 700 or n < 1e5 or cc[za] in off or cc[zb] in off:
            continue
        mdl = Tn[i, j] + Tn[j, i]
        if mdl <= 0:
            continue
        kind = "home" if cc[za] == cc[zb] else "abroad"
        out.setdefault(kind, []).append((math.log(n / mdl), ex[i] + ex[j], math.log(d[i, j]), za, zb, n, mdl))
    msg = [tag]
    for kind, L in sorted(out.items()):
        y = np.array([l[0] for l in L]); X = np.column_stack([np.ones(len(L)), [l[1] for l in L], [l[2] for l in L]])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        res = y - X @ coef
        se = math.sqrt(res.var() * np.linalg.inv(X.T @ X)[1, 1])
        msg.append(f"{kind}: {len(L)} pairs, slope on excess {coef[1]:+.3f} (se {se:.3f}), median ln ratio {np.median(y):.2f}")
    print("  ".join(msg))
    if "--list" in sys.argv:
        for kind, L in out.items():
            for l in sorted(L, key=lambda l: -l[5])[:25]:
                print(f"    {kind} {l[3]}-{l[4]} {l[5]/1e6:.2f}M/yr model {l[6]:.0f}/weekday ratio {math.exp(l[0]):.0f} excess {l[1]:.1f}")
