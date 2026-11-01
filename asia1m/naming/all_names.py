"""Every region name on the map, in one list -> all_names.csv

    python asia1m/naming/all_names.py

Merges this tool's picks (local_names.csv: name, zh, native) with the names read off the map
(oldnames/oldnames.csv). A pick wins where both exist. Columns: key, country, name, zh, native,
lang, source (pick / map), confidence (picks are "picked"), notes. Sorted by country, then
north to south.
"""
import os, csv, json
HERE = os.path.dirname(os.path.abspath(__file__))
regions = {r["key"]: r for r in json.load(open(f"{HERE}/regions.json", encoding="utf-8"))}
rows = {}
for o in csv.DictReader(open(f"{HERE}/oldnames/oldnames.csv", encoding="utf-8-sig")):
    rows[o["key"]] = dict(key=o["key"], country=o.get("country", ""), name=o["name"], zh=o.get("zh", ""),
                          native="", lang="", source="map", confidence=o.get("confidence", ""), notes=o.get("notes", ""))
# hand corrections to what the reader got wrong (name_fixes.csv: key, name, zh, note)
fx = f"{HERE}/name_fixes.csv"
if os.path.exists(fx):
    for x in csv.DictReader(open(fx, encoding="utf-8")):
        country = rows.get(x["key"], {}).get("country") or regions.get(x["key"], {}).get("country") or ""
        rows[x["key"]] = dict(key=x["key"], country=country, name=x["name"], zh=x["zh"], native="", lang="",
                              source="fix", confidence="fixed", notes=x["note"])
for p in csv.DictReader(open(f"{HERE}/local_names.csv", encoding="utf-8-sig")):
    country = rows.get(p["key"], {}).get("country") or (regions.get(p["key"], {}).get("country") or "")
    rows[p["key"]] = dict(key=p["key"], country=country, name=p["name"], zh=p["zh"], native=p["native"],
                          lang=p["lang"], source="pick", confidence="picked", notes=f'{p["batch"]} {p["n"]}')
# native-script spellings kept by key (local_native.csv) apply to map-read names too, not only picks
nv = f"{HERE}/local_native.csv"
if os.path.exists(nv):
    for x in csv.DictReader(open(nv, encoding="utf-8")):
        if x["key"] in rows and x.get("native"):
            rows[x["key"]]["native"] = x["native"]; rows[x["key"]]["lang"] = x.get("lang", "")
out = sorted(rows.values(), key=lambda r: (r["country"] or "~", -(regions.get(r["key"], {}).get("lat") or 0)))
with open(f"{HERE}/all_names.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
# regions with no name in the list, with what they hold, so the gaps can be filled
miss = [dict(key=k, country=r.get("country") or "", lat=r["lat"], lon=r["lon"], people=r["people"],
             holds="; ".join(f'{u["name"]} {u.get("local") or ""} {round(u["people_share"]*100)}%' for u in r["units"][:3]))
        for k, r in regions.items() if k not in rows]
miss.sort(key=lambda m: (m["country"] or "~", -m["lat"]))
with open(f"{HERE}/missing_names.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["key", "country", "lat", "lon", "people", "holds"]); w.writeheader(); w.writerows(miss)
print(len(miss), "regions not in the list -> missing_names.csv")
from collections import Counter
print(len(out), "names;", dict(Counter(r["source"] for r in out)))
print("China without characters:", sum(1 for r in out if r["country"] == "China" and not r["zh"]))
