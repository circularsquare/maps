"""Türkiye — the twelve İBBS Düzey 1 regions, from Eurostat GISCO's NUTS layer.

Writes data/geo/tr/tr_ibbs1.gpkg: 12 polygons, `unit` (TR1..TRC) + `name`.

**THE JOIN IS THE IDENTITY FUNCTION, WHICH IS RARE HERE.** Türkiye is a candidate country in
the NUTS system, so GISCO ships TR1..TRC at level 1 with exactly the codes and names the
Diyanet's Table 4 prints in its own row labels — `TR1 İstanbul`, `TRB Ortadoğu Anadolu`. No
name matching, no crosswalk, and §12's second shape of failure (a confident wrong pairing)
cannot arise: `check()` asserts the two sets are equal and 12 = 12 with nothing left over on
either side. bg_geo.py's note that GISCO is the boundary answer for anything NUTS-coded holds
for one more country, and this is the first non-member state it holds for.

**TWELVE UNITS IS THE COARSEST GEOGRAPHY ON THIS MAP AFTER NOTHING.** Russia's 79 subjects at
1.8 million people each were the previous worst; these are 7.1 million each. That is the
source's own estimation level and not a choice (sources/tr.py), so the answer is not to find
a finer boundary file but to place the dots on a population surface so they at least sit
where Turks live — sources/tr_grid.py, and §8.2's emptiness case is acute at this scale.
Central Anatolia inside TR7 is mostly steppe, and an equal-share wash would put Sivas's dots
on it.

**THE 2024 NUTS EDITION IS THE RIGHT ONE EVEN THOUGH THE SURVEY IS 2013.** Türkiye's İBBS-1
grouping has not changed since it was defined in 2002 — the same twelve regions over the same
81 provinces — so there is no vintage question here of the kind that bites in countries whose
units get redrawn. What changed between editions is the coastline generalisation, nothing
else.

Usage:
    python sources/tr_geo.py --fetch     one ~13 MB GeoJSON from GISCO
    python sources/tr_geo.py             rebuild from data/raw/tr/
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "tr")
GEO = os.path.join(ROOT, "data", "geo", "tr")
OUT = os.path.join(GEO, "tr_ibbs1.gpkg")

NUTS_URL = ("https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/"
            "NUTS_RG_01M_2024_4326_LEVL_1.geojson")
NUTS_JSON = os.path.join(RAW, "NUTS_RG_01M_2024_4326_LEVL_1.geojson")

EXPECTED = {
    "TR1": "İstanbul",
    "TR2": "Batı Marmara",
    "TR3": "Ege",
    "TR4": "Doğu Marmara",
    "TR5": "Batı Anadolu",
    "TR6": "Akdeniz",
    "TR7": "Orta Anadolu",
    "TR8": "Batı Karadeniz",
    "TR9": "Doğu Karadeniz",
    "TRA": "Kuzeydoğu Anadolu",
    "TRB": "Ortadoğu Anadolu",
    "TRC": "Güneydoğu Anadolu",
}

# Türkiye spans roughly 26°E to 45°E. Anything much wider means the read went wrong --
# [[reference_antimeridian]]'s assertion, kept even though this country cannot cross 180°,
# because it also catches a file that quietly included the whole of Europe.
MAX_SPAN_DEG = 25.0


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(NUTS_JSON) and os.path.getsize(NUTS_JSON) > 1_000_000:
        print("already have", os.path.basename(NUTS_JSON))
        return
    print("GET", NUTS_URL)
    r = requests.get(NUTS_URL, timeout=900, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(NUTS_JSON, "wb") as fh:
        fh.write(r.content)
    print(f"  {len(r.content):,} bytes")


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv:
        fetch()
    if not os.path.exists(NUTS_JSON):
        raise SystemExit(f"missing {NUTS_JSON} -- run with --fetch first")

    gdf = gpd.read_file(NUTS_JSON)
    print(f"GISCO NUTS level 1: {len(gdf):,} features, crs={gdf.crs}")

    tr = gdf[gdf["NUTS_ID"].str.startswith("TR")].copy()
    if len(tr) != 12:
        raise SystemExit(f"{len(tr)} TR features at NUTS level 1, expected 12")

    namecol = "NAME_LATN" if "NAME_LATN" in tr.columns else "NUTS_NAME"
    tr["unit"] = tr["NUTS_ID"]
    tr["name"] = tr[namecol]

    got = dict(zip(tr["unit"], tr["name"]))
    if set(got) != set(EXPECTED):
        raise SystemExit(f"GISCO's TR codes are {sorted(got)}, expected {sorted(EXPECTED)}")
    for code, want in EXPECTED.items():
        if got[code] != want:
            raise SystemExit(f"{code} is named {got[code]!r} in GISCO, expected {want!r} -- "
                             "the Diyanet's row labels key on these, so fix rather than "
                             "rename")

    minx, miny, maxx, maxy = tr.total_bounds
    if (maxx - minx) > MAX_SPAN_DEG:
        raise SystemExit(f"the layer is {maxx - minx:.1f}° wide, which is not Türkiye")
    print(f"  bbox: [{minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f}]")

    tr = tr[["unit", "name", "geometry"]].sort_values("unit").reset_index(drop=True)
    os.makedirs(GEO, exist_ok=True)
    tr.to_file(OUT, driver="GPKG", layer="tr_ibbs1")
    print(f"  wrote {OUT}  {len(tr)} regions")
    for _, r in tr.iterrows():
        print(f"    {r['unit']}  {r['name']}")


if __name__ == "__main__":
    main()
