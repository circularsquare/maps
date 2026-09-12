"""Albania — the twelve qark polygons, from INSTAT's own ArcGIS Online organisation.

Writes data/geo/al/al_prefectures.gpkg.

**THE OFFICE PUBLISHES ITS OWN BOUNDARIES AND THEY CARRY THE CENSUS ON THEM**, which is what
makes this join provable rather than asserted ([[reference_gis_server_census]]). INSTAT owns
739 items on ArcGIS Online under `services7.arcgis.com/E9FE1JuiACmTPbPv`, one hosted view per
published indicator per geography per census, and five of them are used here: the 2023
resident population per qark, and the 2023 shares of Muslims, Bektashis, Catholics and
Orthodox. None of it is behind a key, and none of it is on the national open-data portal.

**THE JOIN IS PROVED TO THE PERSON, TWICE OVER.** `P_DISTRIB` on the polygon layer is the
qark's 2023 census population, and it matches the printed Total of that qark's worksheet in
INSTAT table 1.13 exactly, on all twelve; the twelve values are distinct, so a pair of qarqe
swapped between the codes and the polygons fails rather than passing quietly
([[reference_name_join_wrong_neighbour]]). Then the four religion layers, which are
percentages of that same population, reproduce `count / total` from the normalised file to
better than a millionth of a point, again on all twelve. The census's own numbers travel on
the census's own polygons, and both files have to agree about which qark is which for either
check to pass.

**THE SAME ORGANISATION HAS THE FOUR RELIGION SHARES AT 373 ADMINISTRATIVE UNITS, FOR 2011.**
That is thirty times finer than anything drawn here and it is not used; `sources/al.md` §3 is
the whole argument, and the layers are listed there by name so nobody has to find them again.

Usage:
    python sources/al_geo.py --fetch    five small ArcGIS queries, ~2 MB
    python sources/al_geo.py            rebuild from data/raw/al/
"""

import json
import os
import sys
import urllib.parse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "al")
GEO = os.path.join(ROOT, "data", "geo", "al")
NORM = os.path.join(ROOT, "data", "normalized", "al.csv")
OUT = os.path.join(GEO, "al_prefectures.gpkg")

ORG = "https://services7.arcgis.com/E9FE1JuiACmTPbPv/arcgis/rest/services"

# The population layer, which carries the geometry, and the four religion layers, which carry
# a percentage of it. `field` is the layer's own value column.
POP_LAYER = ("prefecture_p_distrib_2023_view", "P_DISTRIB")
RELIGION_LAYERS = {
    # layer name                          field       the source_category it is a share of
    "prefecture_p_muslim_2023_view": ("P_MUSLIM", "Muslim"),
    "prefecture_p_bekta_2023_view": ("P_BEKTA", "Muslim - Bektashism"),
    "prefecture_p_cathol_2023_view": ("P_CATHOL", "Christian - Catholicism"),
    "prefecture_p_orthod_2023_view": ("P_ORTHOD", "Christian - Orthodoxy"),
}

EXPECTED_QARQE = 12
# The percentages are stored as doubles and are exact quotients, so this is a floating-point
# tolerance and not a data one. Widening it is the wrong repair.
TOL = 1e-6


def _query(layer, fields, geometry):
    q = {
        "where": "1=1", "outFields": ",".join(fields),
        "returnGeometry": "true" if geometry else "false",
        "outSR": "4326", "f": "geojson" if geometry else "json",
    }
    return f"{ORG}/{layer}/FeatureServer/0/query?" + urllib.parse.urlencode(q)


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    jobs = [(POP_LAYER[0], ["CODE_PREFECTURE", "NAME_PREFECTURE", POP_LAYER[1]], True)]
    jobs += [(nm, ["CODE_PREFECTURE", f], False) for nm, (f, _) in RELIGION_LAYERS.items()]
    for layer, fields, geom in jobs:
        dest = os.path.join(RAW, layer + (".geojson" if geom else ".json"))
        if os.path.exists(dest) and os.path.getsize(dest) > 500:
            print("  have", os.path.basename(dest))
            continue
        url = _query(layer, fields, geom)
        r = requests.get(url, timeout=600, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        d = r.json()
        n = len(d.get("features", []))
        if n != EXPECTED_QARQE:
            raise SystemExit(f"{layer}: {n} features, expected {EXPECTED_QARQE}\n  {url}")
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(d, fh)
        print(f"  {os.path.basename(dest):<48} {n} features "
              f"{os.path.getsize(dest):>9,} bytes")


def _load(layer, geom):
    p = os.path.join(RAW, layer + (".geojson" if geom else ".json"))
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/al.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "qark"]
    census_total = df.groupby("geo_id")["count"].sum()
    census_cat = df.set_index(["geo_id", "source_category"])["count"]

    gj = _load(POP_LAYER[0], True)
    g = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")
    g["unit"] = g["CODE_PREFECTURE"].astype(str).str.zfill(2)
    g["name"] = g["NAME_PREFECTURE"].astype(str)
    if len(g) != EXPECTED_QARQE or g["unit"].nunique() != EXPECTED_QARQE:
        raise SystemExit(f"{len(g)} polygons, {g['unit'].nunique()} distinct codes")
    if not g.geometry.is_valid.all():
        g["geometry"] = g.geometry.buffer(0)

    # ---- check one: the polygon layer's population is the worksheet's printed Total ----
    print(f"{'unit':>5}  {'INSTAT polygon':<14}{'layer pop':>11}{'table 1.13':>12}   ")
    off = []
    for u, nm, pv in zip(g["unit"], g["name"], g[POP_LAYER[1]]):
        c = int(census_total.get(u, -1))
        print(f"{u:>5}  {nm[:14]:<14}{int(pv):>11,}{c:>12,}"
              f"{'' if int(pv) == c else '   <-- DISAGREES'}")
        if int(pv) != c:
            off.append(u)
    if off:
        raise SystemExit(f"the polygon layer and table 1.13 disagree on {off} -- the code "
                         "map in sources/al.py is pairing a qark with the wrong polygon")
    if census_total.nunique() != EXPECTED_QARQE:
        raise SystemExit("two qarqe have the same population, so this check cannot "
                         "discriminate between them -- do not report it as a proof")
    print(f"\n  all {EXPECTED_QARQE} populations agree to the person, and all "
          f"{EXPECTED_QARQE} are distinct,\n  so the pairing is tested and not asserted.")

    # ---- check two: the four religion layers reproduce count/total ----
    print("\n  the office's own religion shares against count/total from al.csv:")
    worst = 0.0
    for layer, (field, cat) in sorted(RELIGION_LAYERS.items()):
        d = _load(layer, False)
        vals = {str(f["attributes"]["CODE_PREFECTURE"]).zfill(2): float(f["attributes"][field])
                for f in d["features"]}
        if len(vals) != EXPECTED_QARQE:
            raise SystemExit(f"{layer}: {len(vals)} qarqe")
        dev = max(abs(vals[u] - 100.0 * census_cat[(u, cat)] / census_total[u])
                  for u in vals)
        worst = max(worst, dev)
        print(f"    {field:<10} {cat:<38} max deviation {dev:.3e} points")
        if dev > TOL:
            raise SystemExit(f"{layer} is {dev:.4g} points off count/total, above {TOL:g} -- "
                             "the layers and the table are not describing the same qarqe")
    print(f"  worst of the 48 comparisons: {worst:.3e} percentage points.")

    os.makedirs(GEO, exist_ok=True)
    out = g[["unit", "name", "geometry"]].copy()
    out["pop_2023"] = g[POP_LAYER[1]].astype(int)
    out = out.sort_values("unit").reset_index(drop=True)
    out.to_file(OUT, driver="GPKG", layer="prefectures")
    print(f"\nwrote {OUT} ({len(out)} polygons)")


if __name__ == "__main__":
    main()
