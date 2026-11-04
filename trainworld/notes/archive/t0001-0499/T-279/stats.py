import json, math, os, sys
import numpy as np
S = sys.argv[1]
sys.argv = [sys.argv[0], S]
exec(open(os.path.join(S, "fit.py")).read().split("MODELS = [")[0])  # classes, clean(), tab, old
m = Raster("v8").fit(tab)
rule = Rule()
T, K = m.pred(tab); Tr, Kr = rule.pred(tab)
Tt = X(tab, ["osrm_min"])[:, 0]; Kt = X(tab, ["osrm_km"])[:, 0]; g = X(tab, [d])[:, 0]
print("router/rule median", np.median(T / Tr), "router/osrm", np.median(T / Tt), "rule/osrm", np.median(Tr / Tt))
for lo, hi in ((20, 100), (100, 250), (250, 450), (450, 700)):
    s = (g >= lo) & (g < hi)
    sub = [tab[i] for i in np.nonzero(s)[0]]
    a = score(sub, Tr[s], Kr[s]); b = score(sub, T[s], K[s])
    print(f"{lo}-{hi} km n {s.sum()}: rule {a['med_pct']:.1f}% / {a['p90_pct']:.1f}%, odds x{a['med_odds']:.2f} / x{a['p90_odds']:.2f} | router {b['med_pct']:.1f}% / {b['p90_pct']:.1f}%, odds x{b['med_odds']:.2f} / x{b['p90_odds']:.2f}")
# share of pairs the rule gets wrong by over 20%, and the router
print("over 20% wrong: rule", np.mean(np.abs(Tr / Tt - 1) > 0.2), "router", np.mean(np.abs(T / Tt - 1) > 0.2))
print("countries", len({r['ca'] for r in tab}), "places", len({r['a'] for r in tab} | {r['b'] for r in tab}))
# England coarse graph, level-corrected
for name, res in json.load(open(os.path.join(S, "coarse_GB.json"))).items():
    r = np.array([v["min"] / v["osrm_min"] for v in res.values()])
    r = r / np.median(r)
    print("england", name, "after level: median abs %", round(float(np.median(np.abs(r - 1)) * 100), 1), "p90", round(float(np.percentile(np.abs(r - 1), 90) * 100), 1))
sub = [r for r in tab if r["ca"] == "GB" and r["cb"] == "GB"]
TT, KK = m.pred(sub); tt = X(sub, ["osrm_min"])[:, 0]
rr = TT / tt / np.median(TT / tt)
print("router on GB pairs n", len(sub), "after level: median abs %", round(float(np.median(np.abs(rr - 1)) * 100), 1), "p90", round(float(np.percentile(np.abs(rr - 1), 90) * 100), 1))
