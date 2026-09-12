"""The Netherlands — the 403 gemeenten of 2014, which is the vintage the counts are on.

Writes data/geo/nl/nl_units.gpkg, one row per gemeente:

    unit     the CBS gemeentecode as `GM0003`, four digits, zero-padded
    name     the gemeente's name as CBS spells it
    prov     its province

THE BOUNDARY VINTAGE IS THE WHOLE REASON THIS FILE EXISTS. CBS's religion figures are a pool
of the 2010-2014 Enquete Beroepsbevolking published on the gemeentelijke indeling of 2014,
403 units. The Netherlands has merged hard since: 403 became 352 by 2021 and 342 by 2025, so
a current boundary file silently loses an eighth of the counting units and, worse, merges
several of the ones the country is interesting for. Molenwaard, Zederik, Graft-De Rijp,
Menterwolde and Ferwerderadiel are all gone by 2021 and all four are named in CBS's own
write-up of this table. Austria (sources/at_geo.py) solved the same problem by going to the
census's own vintage; this does the same.

PDOK PUBLISHES A WFS PER YEAR AND THAT IS THE CHEAP WAY IN.

    https://service.pdok.nl/cbs/gebiedsindelingen/<year>/wfs/v1_0

The year is a path component, so `2014` answers with the 2014 classification and nothing has
to be back-dated by hand. `gemeente_gegeneraliseerd` is the generalised outline, which is
what a dot map wants: dots are placed on Kontur hexes (sources/nl_grid.py), so the polygon
is used to decide which gemeente a hex centroid falls in and never to draw a coastline.

Usage:
    python sources/nl_geo.py --fetch     # one WFS GetFeature, about 5 MB
    python sources/nl_geo.py             # rebuild from data/raw/nl/
"""

import argparse
import json
import os
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "nl")
OUT_DIR = os.path.join(ROOT, "data", "geo", "nl")
OUT = os.path.join(OUT_DIR, "nl_units.gpkg")
RAW_JSON = os.path.join(RAW, "gemeente_2014.json")
PROV_JSON = os.path.join(RAW, "provincie_2014.json")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots"}
WFS = "https://service.pdok.nl/cbs/gebiedsindelingen/2014/wfs/v1_0"

N_UNITS = 403
N_PROV = 12


def _wfs(layer, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
        print(f"  already on disk: {dest} ({os.path.getsize(dest):,} bytes)")
        return
    url = (f"{WFS}?service=WFS&version=2.0.0&request=GetFeature"
           f"&typeName=gebiedsindelingen:{layer}&count=2000"
           f"&outputFormat=application/json")
    print("  GET", layer)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=900) as r:
        body = r.read()
    d = json.loads(body)
    n = len(d.get("features", []))
    if n == 0:
        sys.exit(f"!! WFS returned no features for {layer}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as fh:
        fh.write(body)
    print(f"  {n:,} features, {len(body):,} bytes")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    _wfs("gemeente_gegeneraliseerd", RAW_JSON)
    _wfs("provincie_gegeneraliseerd", PROV_JSON)


def main():
    import geopandas as gpd

    for p in (RAW_JSON, PROV_JSON):
        if not os.path.exists(p):
            sys.exit(f"missing {p} — run python sources/nl_geo.py --fetch")

    g = gpd.read_file(RAW_JSON)
    print(f"gemeenten: {len(g):,}, crs={g.crs}")
    if len(g) != N_UNITS:
        sys.exit(f"!! {len(g)} gemeenten, expected {N_UNITS} — the 2014 classification "
                 "moved, and data/normalized/nl.csv is keyed on it")
    g["unit"] = g["statcode"].astype(str).str.strip()
    g["name"] = g["statnaam"].astype(str).str.strip()
    if not g["unit"].str.fullmatch(r"GM\d{4}").all():
        sys.exit("!! statcode is not GMnnnn throughout")
    if g["unit"].duplicated().any():
        sys.exit("!! duplicate gemeentecode")

    # THE PROVINCE COMES FROM A SPATIAL JOIN AND NOT FROM A NAME. The maatwerk table prints a
    # province beside every gemeente and that column is checked against this one in
    # sources/nl.py, which is the point: two independent statements of the same fact, so a
    # row read off the wrong line of the spreadsheet fails loudly
    # ([[reference_name_join_wrong_neighbour]]).
    p = gpd.read_file(PROV_JSON)
    if len(p) != N_PROV:
        sys.exit(f"!! {len(p)} provinces, expected {N_PROV}")
    p = p.rename(columns={"statnaam": "prov"})[["prov", "geometry"]]
    reps = g.copy()
    reps["geometry"] = g.geometry.representative_point()
    j = gpd.sjoin(reps, p, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(reps.index)
    if j["prov"].isna().any():
        sys.exit(f"!! {int(j['prov'].isna().sum())} gemeenten sit in no province: "
                 f"{sorted(j.loc[j['prov'].isna(), 'name'])[:8]}")
    g["prov"] = j["prov"].to_numpy()
    print("  per province: " + ", ".join(
        f"{k} {v}" for k, v in g["prov"].value_counts().sort_index().items()))

    g = g.to_crs(4326)
    xmin, ymin, xmax, ymax = g.total_bounds
    print(f"  bbox {xmin:.3f} {ymin:.3f} {xmax:.3f} {ymax:.3f}")
    # The European Netherlands only; the Caribbean municipalities are a separate country on
    # this map and are not in this classification at all. Asserted so a file that quietly
    # includes Bonaire fails here rather than drawing a dot in the Caribbean.
    if not (3.0 < xmin < 3.6 and 50.5 < ymin < 51.0 and 6.8 < xmax < 7.5
            and 53.3 < ymax < 53.8):
        sys.exit("!! bounding box is not the European Netherlands")

    os.makedirs(OUT_DIR, exist_ok=True)
    g[["unit", "name", "prov", "geometry"]].to_file(OUT, driver="GPKG")
    print(f"wrote {OUT}  ({len(g):,} gemeenten)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    main()
