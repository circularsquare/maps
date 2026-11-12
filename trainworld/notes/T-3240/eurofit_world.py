"""T-3240: residents' all overnight trips a head (Eurostat tour_dem_ttw 2019) against GDP per head
at PPP (countryIncome.json): the income exponent of long trips, crude.
    python eurofit.py
"""
import json, math, collections
import numpy as np
HERE = r"C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\2ba1fd73-4c91-45ba-b838-fe8f908d4b21\scratchpad\longdist"
G = r"C:\Users\anita\projects\maps\trainworld\app\src\game"
d = json.load(open(HERE + r"\dl\ttw.json"))
inc = json.load(open(G + r"\countryIncome.json"))["gdp"]
rows = json.load(open(G + r"\market-rows.json", encoding="utf-8"))
pop = collections.Counter()
for r in rows.values():
    pop[r["country"]] += r["pop"]
ids, size = d["id"], d["size"]
idx = {k: d["dimension"][k]["category"]["index"] for k in ids}
print("c_dest codes:", [k for k in idx["c_dest"] if len(k) > 2])


def val(**kw):
    i = 0
    for k, n in zip(ids, size):
        i = i * n + idx[k][kw[k]]
    return d["value"].get(str(i))


ISO = {"EL": "GR", "UK": "GB"}
pts = []
for g in idx["geo"]:
    if len(g) != 2:
        continue
    c = ISO.get(g, g)
    for dur in ("N_GE1", "N1-3"):
        pass
    dom = val(freq="A", c_dest="WORLD", purpose="TOTAL", duration="N_GE1", unit="NR", geo=g, time="2019")
    dom13 = val(freq="A", c_dest="WORLD", purpose="TOTAL", duration="N1-3", unit="NR", geo=g, time="2019")
    prof = val(freq="A", c_dest="WORLD", purpose="PROF", duration="N_GE1", unit="NR", geo=g, time="2019")
    if dom is None or c not in inc or pop[c] < 1e5:
        print("skip", g, dom, c in inc, pop[c]); continue
    pts.append((c, inc[c], dom / pop[c], (dom13 or 0) / pop[c], (prof or 0) / pop[c]))
pts.sort(key=lambda p: p[1])
for c, y, t, t13, pr in pts:
    print(f"{c} gdp {y:6.0f} all overnight trips a head {t:5.2f} (1-3 nights {t13:5.2f}, business {pr:4.2f})")
y = np.log([p[1] for p in pts]); t = np.log([p[2] for p in pts])
A = np.vstack([y, np.ones_like(y)]).T
b, res, *_ = np.linalg.lstsq(A, t, rcond=None)
r = t - A @ b
se = math.sqrt((r @ r) / (len(t) - 2) / ((y - y.mean()) @ (y - y.mean())))
print(f"fit: trips a head ~ GDP^{b[0]:.2f} (se {se:.2f}, n {len(t)})")
# without the small/odd ones (LU, IE: GDP inflated; MT, CY: islands)
keep = [i for i, p in enumerate(pts) if p[0] not in ("LU", "IE", "MT", "CY")]
b2 = np.linalg.lstsq(A[keep], t[keep], rcond=None)[0]
print(f"fit without LU IE MT CY: GDP^{b2[0]:.2f}")
# poorer than the US only matters (min(y/yUS,1)); all EU are under the US
