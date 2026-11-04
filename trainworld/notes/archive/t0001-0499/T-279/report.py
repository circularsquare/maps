"""T-279 research: the chosen router against the rule on T-279's named pairs, by kind, and its
parameters; error by distance band on the held-out table pairs is in fit.py's numbers."""
import json, math, os, sys, statistics as st
S = sys.argv[1]
j = json.load(open(os.path.join(S, "fit_results.json")))
M = sys.argv[2] if len(sys.argv) > 2 else "raster router v8"
print(M, j["results"][M]["params"])
print("formula params", j["results"]["formula: ldet, slope, hwf"]["params"])
print("country", j["results"]["speed by country"]["params"])


def car(T, K):
    return T + K * 0.225


by = {}
print("pair | km | OSRM min | rule min | router min | rail odds, rule -> OSRM | rule -> router")
for r in j["t279"]:
    Tr, Kr = r["pred"]["rule (today)"]
    Tm, Km = r["pred"][M]
    co = car(r["osrm_min"], r["osrm_km"]); cr = car(Tr, Kr); cm = car(Tm, Km)
    b = 0.10 * math.sqrt(min(1, 60 / cr))
    o_true = math.exp(b * (co - cr)); o_mod = math.exp(b * (cm - cr))
    err_rule = abs(math.log(math.exp(0.10 * math.sqrt(min(1, 60 / co)) * (cr - co))))
    err_mod = abs(0.10 * math.sqrt(min(1, 60 / co)) * (cm - co))
    by.setdefault(r["kind"], []).append((o_true, o_mod, math.exp(err_rule), math.exp(err_mod), Tr / r["osrm_min"], Tm / r["osrm_min"]))
    print(f"{r['name']} | {r['gc_km']:.0f} | {r['osrm_min']:.0f} | {Tr:.0f} | {Tm:.0f} | x{o_true:.2f} | x{o_mod:.2f}")
print()
print("kind | n | rule/OSRM med | router/OSRM med | odds error rule med | router med | odds move OSRM med | router med")
for k, v in by.items():
    print(f"{k} | {len(v)} | {st.median(x[4] for x in v):.2f} | {st.median(x[5] for x in v):.2f} | x{st.median(x[2] for x in v):.2f} | x{st.median(x[3] for x in v):.2f} | x{st.median(x[0] for x in v):.2f} | x{st.median(x[1] for x in v):.2f}")
