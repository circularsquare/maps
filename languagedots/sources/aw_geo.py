"""Aruba placement layer: religiondots' Kontur hexes for AW (read only), cut along CBS Aruba's
census zones -> data/geo/aw/aw_hexes.gpkg (unit = "AW<zone code>", as data/normalized/aw.csv).

    python sources/aw_geo.py [--fetch]

SOURCES.
  * CBS Aruba's zone polygons with their census populations, ArcGIS Online feature service
    `Population_Tables_Census_2010_2020` (owner R.vdBiezen, CBS Aruba's GIS account; 55 zones,
    fields Zone = the GAC2 code, region digit then zone digit, and TotPop_10 / TotPop_20).
  * religiondots/data/geo/aw/aw_hexes.gpkg: Kontur 2023 hexes for Aruba (279, read only).

WHY CUT THE HEXES. The zones are small (55 over 180 km^2, median about 2 km^2) against Kontur's
hexes (about 0.74 km^2), so assigning whole hexes by centroid would leave small Oranjestad and
San Nicolas zones without a hex. Each hex is intersected with the zones instead and its people
shared among its pieces by land area (the pieces of one hex sum to the hex's people, so a coastal
hex half in the sea keeps its people on land).

CHECKS. The zone codes with people in aw.csv equal the service's zones with TotPop_10 > 0, both
ways, and each zone's census total equals TotPop_10 within 3 (the table rounds cell by cell);
the pieces lose no Kontur people beyond FAR_MAX (hexes wholly offshore of every zone); Kontur per
zone against the 2010 census, normalised, with a log correlation against 500 shuffles. A zone
with census people and no Kontur people gets its own polygon at the census figure.
"""
import json
import math
import os
import random
import sys
import urllib.parse
import urllib.request
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "aw"
ZONES = RAW / "aw_zones_census_2010_2020.geojson"
NORM = HERE / "data" / "normalized" / "aw.csv"
RD_HEX = HERE.parent / "religiondots" / "data" / "geo" / "aw" / "aw_hexes.gpkg"
OUT = HERE / "data" / "geo" / "aw" / "aw_hexes.gpkg"
SVC = ("https://services7.arcgis.com/WPLwj4y7iq9wOYCj/arcgis/rest/services/"
       "Population_Tables_Census_2010_2020/FeatureServer/0/query?")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
CRS_M = 32619           # UTM 19N
N_ZONES = 55
TOTAL_2010 = 101_484
FAR_MAX = 100           # Kontur people allowed outside every zone


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    q = urllib.parse.urlencode({"where": "1=1", "outFields": "Zone,GAC2_NAAM,TotPop_10,TotPop_20",
                                "outSR": 4326, "f": "geojson"})
    req = urllib.request.Request(SVC + q, headers={"User-Agent": UA})
    data = urllib.request.urlopen(req, timeout=300).read()
    n = len(json.loads(data)["features"])
    if n != N_ZONES:
        raise SystemExit(f"aw: the zone service returned {n} features, expected {N_ZONES}")
    ZONES.write_bytes(data)
    print(f"  {ZONES.name}: {n} zones, {len(data):,} bytes")


def pear(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))


def main():
    import geopandas as gpd
    if "--fetch" in sys.argv or not ZONES.exists():
        fetch()
    z = gpd.read_file(ZONES)
    z["unit"] = "AW" + z["Zone"].astype(int).astype(str)
    if len(z) != N_ZONES or z["unit"].duplicated().any():
        raise SystemExit("aw: zone codes are not 55 distinct")
    if int(z["TotPop_10"].sum()) != TOTAL_2010:
        raise SystemExit(f"aw: TotPop_10 sums to {z['TotPop_10'].sum()}, expected {TOTAL_2010}")

    # join check: census table zones against the polygons' 2010 populations, both ways
    df = pd.read_csv(NORM, dtype={"geo_id": str})
    tab = df[df["geo_level"] == "zone"].groupby("geo_id")["count"].sum()
    svc = z.set_index("unit")["TotPop_10"]
    svc = svc[svc > 0]
    if set(tab.index) != set(svc.index):
        raise SystemExit(f"aw: zones differ: table only {sorted(set(tab.index) - set(svc.index))}, "
                         f"service only {sorted(set(svc.index) - set(tab.index))}")
    dev = (tab - svc.reindex(tab.index)).abs()
    if dev.max() > 3:
        raise SystemExit(f"aw: zone totals differ from TotPop_10: {dev[dev > 3].to_dict()}")
    print(f"  {len(tab)} populated zones; table against TotPop_10 within {int(dev.max())} each")

    h = gpd.read_file(RD_HEX)[["cellcode", "pop", "geometry"]].to_crs(CRS_M)
    zm = z[["unit", "geometry"]].to_crs(CRS_M)
    zm["geometry"] = zm.geometry.buffer(0)
    pieces = gpd.overlay(h, zm, how="intersection", keep_geom_type=True)
    pieces["area"] = pieces.geometry.area
    pieces = pieces[pieces["area"] > 1.0]
    share = pieces["area"] / pieces.groupby("cellcode")["area"].transform("sum")
    pieces["pop"] = pieces["pop"] * share
    lost = h.loc[~h["cellcode"].isin(pieces["cellcode"]), "pop"].sum()
    print(f"  Kontur: {len(h)} hexes, {h['pop'].sum():,.0f} people -> {len(pieces)} pieces in "
          f"{pieces['unit'].nunique()} zones; {lost:,.0f} people in hexes touching no zone")
    if lost > FAR_MAX:
        raise SystemExit(f"aw: {lost:.0f} Kontur people outside every zone")

    per = pieces.groupby("unit")["pop"].sum()
    rows = [(u, float(c), float(per.get(u, 0.0))) for u, c in tab.items()]
    ratio = sum(k for _, _, k in rows) / sum(c for _, c, _ in rows)
    normd = sorted((k / c / ratio, u) for u, c, k in rows)
    print(f"  Kontur / census 2010 nationally {ratio:.3f}; per zone normalised: p10 "
          f"{normd[len(normd) // 10][0]:.2f} median {normd[len(normd) // 2][0]:.2f} p90 "
          f"{normd[9 * len(normd) // 10][0]:.2f}")
    print("  lowest: " + ", ".join(f"{u} {r:.2f}" for r, u in normd[:4]))
    print("  highest: " + ", ".join(f"{u} {r:.2f}" for r, u in normd[-4:]))
    ok = [(c, k) for _, c, k in rows if k > 0]
    lc, lk = [math.log(c) for c, _ in ok], [math.log(k) for _, k in ok]
    r = pear(lc, lk)
    rng = random.Random(0)
    best = max(abs(pear(lc, rng.sample(lk, len(lk)))) for _ in range(500))
    print(f"  log correlation r = {r:.3f} against a best of {best:.3f} over 500 shuffles")
    if r <= best:
        raise SystemExit("aw: the zone join is not carrying information")

    empty = [u for u, c, k in rows if k <= 0]
    layer = pieces[["cellcode", "unit", "pop", "geometry"]].copy()
    if empty:
        add = zm[zm["unit"].isin(empty)].copy()
        add["pop"] = add["unit"].map(tab).astype(float)
        add["cellcode"] = add["unit"] + ":zone"
        layer = pd.concat([layer, add[["cellcode", "unit", "pop", "geometry"]]], ignore_index=True)
        print(f"  {len(empty)} populated zones with no Kontur people, given their own polygon: {empty}")
    layer = gpd.GeoDataFrame(layer, geometry="geometry", crs=CRS_M).to_crs(4326)
    layer = layer[layer["unit"].isin(tab.index)]
    if set(layer.loc[layer["pop"] > 0, "unit"]) != set(tab.index):
        raise SystemExit("aw: a populated zone has no populated piece")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    layer.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"  wrote {OUT} ({len(layer)} pieces, {layer['pop'].sum():,.0f} Kontur people)")


if __name__ == "__main__":
    main()
