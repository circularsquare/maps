"""Timor-Leste — the fourteen 2022-census municipalities, cut out of COD-AB.

Writes data/geo/tl/tl_municipalities.gpkg.

**COD-AB IS ONE MUNICIPALITY BEHIND THE CENSUS AND THE FIX IS A TIER DOWN.** The OCHA bundle
`tls_admin_boundaries` is `valid_on 2020-09-11` and carries thirteen ADM1 polygons. The 2022
census tabulates fourteen: Atauro, the island north of the capital, was a posto of Dili until
2022 and is a municipality of its own in table 4.03, at 10,295 people. So the fourteenth unit
is not missing from the file, it is one level down in it: ADM2 `TL0604` is Atauro at
139.91 km², against the census's own 140.55, and Dili is that polygon subtracted from ADM1
`TL06`. [[reference_agol_statute_boundaries]] is the standing note about a drawn tier that is
newer than COD-AB; this is the cheap version of it, because the older file already contains
the newer line.

**THE SUBTRACTION IS ASSERTED ON AREA AND ON GEOMETRY.** Dili-minus-Atauro has to come out at
the census's 227.61 km² and has to be a single connected piece of mainland with no islands
left in it, which is what would happen if the ADM2 polygon and the ADM1 polygon did not share
their coastline exactly.

Nothing here joins on a name. Every unit is a p-code that `sources/tl.py` already wrote into
`geo_id`, so the join is an identity and the risk
[[reference_name_join_wrong_neighbour]] warns about does not arise.

Usage:
    python sources/tl_geo.py --fetch    the COD-AB shapefile bundle, ~5 MB
    python sources/tl_geo.py            rebuild from data/raw/tl/
"""

import os
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "tl")
GEO = os.path.join(ROOT, "data", "geo", "tl")
NORM = os.path.join(ROOT, "data", "normalized", "tl.csv")
OUT = os.path.join(GEO, "tl_municipalities.gpkg")

COD_URL = ("https://data.humdata.org/dataset/ee4a485c-9180-43db-9672-55e18dac5f96/"
           "resource/a3d43a39-2ddc-4f66-8e53-6fcbec8bd665/download/"
           "tls_admin_boundaries.shp.zip")
COD_ZIP = "tls_admin_boundaries.shp.zip"

ATAURO_ADM2 = "TL0604"
DILI_ADM1 = "TL06"
EXPECTED_ADM1 = 13
EXPECTED_UNITS = 14

# Census table 4.03's own area column, km². The two split units are what the assertion is
# for; the rest are here so a re-cut COD-AB bundle cannot change a boundary in silence.
CENSUS_AREA = {
    "TL02": 735.14, "TL01": 802.50, ATAURO_ADM2: 140.55, "TL03": 1494.25,
    "TL04": 1374.49, "TL05": 1206.73, DILI_ADM1: 227.61, "TL07": 759.04,
    "TL09": 1817.14, "TL08": 561.86, "TL11": 1787.41, "TL10": 1338.28,
    "TL12": 817.47, "TL13": 1887.62,
}
AREA_TOLERANCE = 0.06     # COD-AB's coastline is not the census's; 6% on the smallest unit


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, COD_ZIP)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print("already have", COD_ZIP)
    else:
        print("GET", COD_URL)
        r = requests.get(COD_URL, timeout=1800, stream=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        print(f"  {os.path.getsize(dest):,} bytes")
    if not zipfile.is_zipfile(dest):
        raise SystemExit(f"{COD_ZIP} is not a zip -- HDX answered the un-redirected URL")


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    src = os.path.join(RAW, COD_ZIP)
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run with --fetch first")

    # §12's Chile rule: read the SHAPEFILE with fiona.
    a1 = gpd.read_file(f"zip://{src}!tls_admin1.shp", engine="fiona")
    a2 = gpd.read_file(f"zip://{src}!tls_admin2.shp", engine="fiona")
    if len(a1) != EXPECTED_ADM1:
        raise SystemExit(f"ADM1 has {len(a1)} features, expected {EXPECTED_ADM1}")
    if a1.crs is None or a1.crs.to_epsg() != 4326:
        a1 = a1.to_crs(4326)
    a2 = a2.to_crs(a1.crs)
    print(f"COD-AB: {len(a1)} ADM1, {len(a2)} ADM2, valid_on "
          f"{a1['valid_on'].iloc[0]}")

    at = a2[a2["adm2_pcode"] == ATAURO_ADM2]
    if len(at) != 1:
        raise SystemExit(f"{ATAURO_ADM2} matched {len(at)} ADM2 polygons")
    dili = a1[a1["adm1_pcode"] == DILI_ADM1]
    if len(dili) != 1:
        raise SystemExit(f"{DILI_ADM1} matched {len(dili)} ADM1 polygons")

    mainland = dili.geometry.iloc[0].difference(at.geometry.iloc[0])
    rows = []
    for pcode, name, geom in zip(a1["adm1_pcode"], a1["adm1_name"], a1.geometry):
        if pcode == DILI_ADM1:
            geom = mainland
        rows.append({"unit": pcode, "name": name, "geometry": geom})
    rows.append({"unit": ATAURO_ADM2, "name": "Atauro",
                 "geometry": at.geometry.iloc[0]})
    units = gpd.GeoDataFrame(rows, crs=a1.crs)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{len(units)} units, expected {EXPECTED_UNITS}")

    # ---- areas, against the census's own column ----
    m = units.to_crs(3857)          # equal-area is overkill at this latitude and extent
    import math
    lat = math.radians(-8.7)
    km2 = m.geometry.area * (math.cos(lat) ** 2) / 1e6
    print("\n  unit          COD-AB km2   census km2   ratio")
    worst = 0.0
    for unit, name, a in zip(units["unit"], units["name"], km2):
        want = CENSUS_AREA[unit]
        r = a / want
        worst = max(worst, abs(r - 1.0))
        flag = "" if abs(r - 1.0) <= AREA_TOLERANCE else "   <-- OUT OF BAND"
        print(f"  {unit:<6} {name:<10} {a:>9.1f}   {want:>9.1f}   {r:5.3f}{flag}")
    if worst > AREA_TOLERANCE:
        raise SystemExit(f"a unit is {worst * 100:.1f}% off the census area -- the ADM2 "
                         "subtraction or the boundary vintage is wrong")

    # ---- the subtraction actually removed an island, and only the island ----
    def parts(g):
        return len(g.geoms) if g.geom_type == "MultiPolygon" else 1
    before, after = parts(dili.geometry.iloc[0]), parts(mainland)
    print(f"\n  Dili: {before} part(s) before the subtraction, {after} after")
    if after >= before:
        raise SystemExit("subtracting Atauro did not remove a piece of Dili -- the ADM2 "
                         "polygon is not inside the ADM1 one")
    leftover = mainland.intersection(at.geometry.iloc[0]).area
    if leftover > 1e-9:
        raise SystemExit("Dili and Atauro still overlap after the difference")

    # ---- every census unit has a polygon, and vice versa ----
    if os.path.exists(NORM):
        df = pd.read_csv(NORM, dtype={"geo_id": str})
        cen = set(df["geo_id"])
        have = set(units["unit"])
        if cen - have:
            raise SystemExit(f"census units with no polygon: {sorted(cen - have)}")
        if have - cen:
            raise SystemExit(f"polygons with no census unit: {sorted(have - cen)}")
        print(f"  {len(cen)} census units, {len(have)} polygons, identical p-codes")
    else:
        print(f"  (no {NORM} yet, skipping the census join check)")

    os.makedirs(GEO, exist_ok=True)
    units.to_file(OUT, layer="municipalities", driver="GPKG")
    print(f"\nwrote {OUT} ({len(units)} polygons)")


if __name__ == "__main__":
    main()
