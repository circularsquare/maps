"""Second check of the level-2 join: each polygon's area against the area
Wikidata gives for its OKTMO code (P764 -> P2046).

A polygon paired with the wrong municipality shows up as an area off by a
large factor, whatever the populations say. Wikidata areas come from many
hands and some are for an older territory, so only the spread matters.
Polygons holding two Rosstat units are compared with the sum.

Caches helper1m/data/russia/raw/osm/wikidata_area.json; prints the share of
polygons within 10% and 25%, and the worst.
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd

HELPER = Path(__file__).resolve().parents[2]
DATA = HELPER / "data" / "russia"
CACHE = DATA / "raw" / "osm" / "wikidata_area.json"


def fetch(codes):
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [c for c in codes if c not in cache]
    for i in range(0, len(todo), 300):
        chunk = todo[i:i + 300]
        q = ("SELECT ?code ?area ?unit WHERE { VALUES ?code { " + " ".join(f'"{c}"' for c in chunk)
             + " } ?item wdt:P764 ?code ; p:P2046/psv:P2046 [ wikibase:quantityAmount ?area ;"
             " wikibase:quantityUnit ?unit ] . }")
        req = urllib.request.Request(
            "https://query.wikidata.org/sparql", data=urllib.parse.urlencode({"query": q}).encode(),
            headers={"User-Agent": "helper1m-research/1.0", "Accept": "application/sparql-results+json"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    res = json.load(r)
                break
            except Exception as e:
                print(f"  {e}; retrying", flush=True)
                time.sleep(20 * (attempt + 1))
        for c in chunk:
            cache.setdefault(c, None)
        for b in res["results"]["bindings"]:
            a = float(b["area"]["value"])
            unit = b["unit"]["value"].rsplit("/", 1)[1]
            km2 = a if unit == "Q712226" else a / 100 if unit == "Q35852" else None  # km2, hectare
            if km2 and cache.get(b["code"]["value"]) is None:
                cache[b["code"]["value"]] = km2
        print(f"  {min(i + 300, len(todo))}/{len(todo)}", flush=True)
        time.sleep(2)
    CACHE.write_text(json.dumps(cache, indent=0), encoding="utf-8")
    return cache


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    adm2 = gpd.read_file(DATA / "boundaries" / "adm2.gpkg")
    adm2["km2"] = adm2.to_crs("ESRI:54009").area / 1e6
    codes = sorted({c for code in adm2.code for c in code.split("+")})
    wd = fetch(codes)
    rows = []
    for _, r in adm2.iterrows():
        parts = r.code.split("+")
        if all(wd.get(c) for c in parts):
            rows.append((r.km2 / sum(wd[c] for c in parts), r.code, r["name"], r.km2,
                         sum(wd[c] for c in parts)))
    n = len(rows)
    w10 = sum(abs(x[0] - 1) <= 0.10 for x in rows) / n
    w25 = sum(abs(x[0] - 1) <= 0.25 for x in rows) / n
    print(f"{n} of {len(adm2)} polygons have a Wikidata area: within 10% {w10:.1%}, within 25% {w25:.1%}")
    rows.sort()
    print("smallest / largest polygon-to-Wikidata ratios:")
    for x in rows[:12] + rows[-12:]:
        print(f"  {x[1]} {x[2]}: polygon {x[3]:,.0f} km2, Wikidata {x[4]:,.0f} ({x[0]:.2f})")


if __name__ == "__main__":
    main()
