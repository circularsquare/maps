"""China regions whose largest group is not Han -> nonhan_regions.csv

    python asia1m/naming/nonhan.py

Ethnicity (2020 census, county level) from helper1m's countries/china/composition.json, carried
onto regions by people: each county's counts are split between regions by where the county's
people live (pop.npy). Nationality is not language: Hui, Manchu, Tujia and most She speak
Mandarin, so the `script` column says which groups have their own script in use.
"""
import os, sys, csv, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "regionnaming"))
import numpy as np, geopandas as gpd
from rasterio.features import rasterize
from geo import TRANSFORM, PROJ4, W, H
sys.stdout.reconfigure(encoding="utf-8")

H1 = os.path.join(HERE, "..", "..", "helper1m", "countries", "china")
comp = json.load(open(f"{H1}/composition.json", encoding="utf-8"))
groups = comp["groups"]; lv3 = comp["levels"]["3"]
g = gpd.read_file(f"{H1}/adm3.geojson").to_crs(PROJ4).reset_index(drop=True)
codes = list(g["code"])
cid = rasterize(((geom, i + 1) for i, geom in enumerate(g.geometry) if geom is not None),
                out_shape=(H, W), transform=TRANSFORM, fill=0, dtype=np.int32)
rid = np.load(os.path.join(HERE, "..", "regionnaming", "region_id.npy"))
pop = np.load(os.path.join(HERE, "pop.npy"))
regs = {int(r["rid"]): r for r in csv.DictReader(open(os.path.join(HERE, "..", "regionnaming", "regions_prelim.csv")))}
facts = {r["key"]: r for r in json.load(open(f"{HERE}/regions.json", encoding="utf-8"))}
by_color = {}
for k, f in facts.items(): by_color.setdefault(f["color"], []).append(f)
names = {r["key"]: r for r in csv.DictReader(open(f"{HERE}/all_names.csv", encoding="utf-8-sig"))}

m = (rid > 0) & (cid > 0)
r_, c_, p_ = rid[m].astype(np.int64), cid[m].astype(np.int64), pop[m].astype(np.float64)
M = len(codes) + 1
key = r_ * M + c_
u, inv = np.unique(key, return_inverse=True)
pp = np.bincount(inv, weights=p_)
county_pop = np.bincount(c_, weights=p_, minlength=M)

SCRIPT = {"tibetan": "Tibetan", "uyghur": "Uyghur (Arabic)", "kazak": "Kazakh (Arabic)", "kazakh": "Kazakh (Arabic)",
          "kirgiz": "Kirghiz (Arabic)", "kyrgyz": "Kirghiz (Arabic)", "mongol": "Mongolian (vertical)",
          "korean": "Korean (Hangul)", "yi": "Yi", "dai": "Tai Lue / Tai Nüa", "zhuang": "Zhuang (Latin)",
          "xibe": "Xibe", "uzbek": "Arabic", "tajik": "Arabic", "tatar": "Arabic"}
acc = {}
for kk, p in zip(u, pp):
    ri, ci = int(kk // M), int(kk % M)
    c = lv3.get(codes[ci - 1])
    if not c or county_pop[ci] <= 0: continue
    f = p / county_pop[ci]
    a = acc.setdefault(ri, np.zeros(len(groups)))
    for gi, n in zip(c["g"], c["k"]): a[gi] += n * f
rows = []
for ri, a in acc.items():
    tot = a.sum()
    if tot < 1000: continue
    order = np.argsort(-a); top = order[0]
    if groups[top]["key"] == "han": continue
    r = regs.get(ri)
    if not r: continue
    cands = by_color.get(r["color"], [])
    f = min(cands, key=lambda x: abs(x["lon"] - float(r["lon"])) + abs(x["lat"] - float(r["lat"]))) if cands else None
    kkey = f["key"] if f else r["color"]
    nm = names.get(kkey, {})
    han = a[[i for i, gg in enumerate(groups) if gg["key"] == "han"][0]] / tot
    gk = groups[top]["key"]
    rows.append(dict(key=kkey, name=nm.get("name", ""), zh=nm.get("zh", ""), native=nm.get("native", ""),
                     largest=groups[top]["en"], largest_share=round(a[top] / tot, 2), han_share=round(han, 2),
                     second=groups[order[1]]["en"], second_share=round(a[order[1]] / tot, 2),
                     script=SCRIPT.get(gk, "Mandarin speakers (Chinese)" if gk in ("hui", "manchu", "tujia", "she") else "check"),
                     lat=round(float(r["lat"]), 2), lon=round(float(r["lon"]), 2)))
# keep only regions that need a non-Chinese label: groups with their own script in use, in China
OWN = {"Tibetan", "Uyghur", "Kazakh", "Kirghiz", "Yi", "Mongol", "Zhuang", "Lisu"}
SCRIPT.update({"lisu": "Lisu (Fraser)", "zhuang": "Zhuang (Latin)"})
rows = [r for r in rows if r["largest"] in OWN and (facts.get(r["key"], {}).get("country") == "China")]
for r in rows:
    if r["largest"] == "Mongol":
        r["todo"] = "decide: vertical Mongolian script, or keep Chinese"
    elif r["native"]:
        r["todo"] = "use native label (spelling in native column), if not already on the map"
    else:
        r["todo"] = "native spelling still to fill (local_native.csv)"
rows.sort(key=lambda x: (x["largest"], -x["largest_share"]))
with open(f"{HERE}/nonhan_regions.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
from collections import Counter
print(len(rows), "regions with a non-Han largest group")
for k, n in Counter(r["largest"] for r in rows).most_common(): print(f"  {k}: {n}")
for r in rows: print(r["largest"], r["largest_share"], r["name"], r["zh"], r["native"] and "(native set)")
