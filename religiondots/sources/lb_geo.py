"""Lebanon: the 26 cazas, from COD-AB `cod-ab-lbn` (OCHA, HDX release of 2026-01-26).

Writes data/geo/lb/lb_units.gpkg (`unit` = pcode, `name`, `pop` = people drawn, geometry) and
data/geo/lb/lb_lookup.csv (`geo_id` -> `unit`), from data/raw/lb/lbn_admin_boundaries.geojson.zip.

The join is on the pcode, which `sources/lb.py` writes from the same names OCHA's population package
uses; both the names and the pcodes are asserted here against the file, so a renamed or renumbered
caza stops the build. The witness that neither key decides is in `sources/lb_grid.py`: Kontur's
people per caza against OCHA's residents per caza.

SHEBAA FARMS ARE CLIPPED. Natural Earth's `ne_10m_admin_0_disputed_areas` has `Shebaa Farms`,
"Admin. By Israel; Claimed by Lebanon"; this map gives disputed land to its administrator (spec
§14.18), so any part of a caza inside it is removed and the area printed (`SHEBAA_KM2`).

Usage:
    python sources/lb_geo.py        (the zip is fetched by sources/lb.py --fetch's sibling below)
    python sources/lb_geo.py --fetch
"""

import json
import os
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "6")

import pandas as pd

from lb import PCODE

RAW = os.path.join(ROOT, "data", "raw", "lb")
GEO = os.path.join(ROOT, "data", "geo", "lb")
ZIP = os.path.join(RAW, "lbn_admin_boundaries.geojson.zip")
ZIP_URL = ("https://data.humdata.org/dataset/569beba7-bad7-4951-a19d-468a035461cd/resource/"
           "81ca0135-b5c9-46e0-a546-f0bdd6d7fd42/download/lbn_admin_boundaries.geojson.zip")
DISPUTED = os.path.join(ROOT, "data", "geo", "ne_10m_admin_0_disputed_areas.geojson")
NORMALIZED = os.path.join(ROOT, "data", "normalized", "lb.csv")
SHEBAA = "Shebaa Farms"
SHEBAA_KM2 = (0.0, 60.0)     # the farms are about 25 km2; COD may hold some, none or all of them
EQ = "EPSG:6933"


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(ZIP) and os.path.getsize(ZIP) > 1_000_000:
        return
    print("GET", ZIP_URL)
    r = requests.get(ZIP_URL, timeout=600)
    r.raise_for_status()
    with open(ZIP + ".part", "wb") as fh:
        fh.write(r.content)
    os.replace(ZIP + ".part", ZIP)


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv or not os.path.exists(ZIP):
        fetch()
    with zipfile.ZipFile(ZIP) as z:
        g = gpd.GeoDataFrame.from_features(json.loads(z.read("lbn_admin2.geojson")), crs=4326)
    if len(g) != 26:
        raise SystemExit(f"COD-AB LBN adm2 has {len(g)} features, expected 26")
    got = dict(zip(g["adm2_name"], g["adm2_pcode"]))
    if got != PCODE:
        raise SystemExit(f"COD-AB names/pcodes differ from sources/lb.py's PCODE: "
                         f"{sorted(set(got.items()) ^ set(PCODE.items()))}")

    ne = json.load(open(DISPUTED, encoding="utf-8"))
    sheb = [f for f in ne["features"] if f["properties"].get("BRK_NAME") == SHEBAA]
    if len(sheb) != 1 or "Admin. By Israel" not in sheb[0]["properties"].get("NOTE_BRK", ""):
        raise SystemExit(f"{DISPUTED} no longer has one '{SHEBAA}' administered by Israel")
    sh = gpd.GeoDataFrame.from_features(sheb, crs=4326).geometry.iloc[0]
    cut = {}
    for i, row in g.iterrows():
        if row.geometry.intersects(sh):
            before = gpd.GeoSeries([row.geometry], crs=4326).to_crs(EQ).area.iloc[0]
            g.at[i, "geometry"] = row.geometry.difference(sh)
            after = gpd.GeoSeries([g.at[i, "geometry"]], crs=4326).to_crs(EQ).area.iloc[0]
            cut[row["adm2_name"]] = (before - after) / 1e6
    print(f"  Shebaa Farms clipped: " + (", ".join(f"{k} {v:.1f} km2" for k, v in cut.items())
                                          or "no caza touches it"))
    total = sum(cut.values())
    if not SHEBAA_KM2[0] <= total <= SHEBAA_KM2[1]:
        raise SystemExit(f"the Shebaa clip took {total:.1f} km2, outside {SHEBAA_KM2}")

    if not os.path.exists(NORMALIZED):
        raise SystemExit(f"missing {NORMALIZED}; run sources/lb.py first")
    rows = pd.read_csv(NORMALIZED, keep_default_na=False, na_values=[""])
    rows = rows[~rows["source_category"].str.startswith("Migrants")]
    pop = rows.groupby("geo_id")["count"].sum()
    if set(pop.index) != set(PCODE.values()):
        raise SystemExit(f"normalized rows cover {sorted(pop.index)}")

    units = gpd.GeoDataFrame({"unit": g["adm2_pcode"], "name": g["adm2_name"]},
                             geometry=g.geometry, crs=4326)
    units["pop"] = units["unit"].map(pop).astype(int)
    a = units.to_crs(EQ).area / 1e6
    print("  caza               km2      drawn")
    for i in units.sort_values("unit").index:
        print(f"    {units.at[i, 'unit']} {units.at[i, 'name']:<17} {a[i]:>7,.0f}  "
              f"{units.at[i, 'pop']:>9,}")
    os.makedirs(GEO, exist_ok=True)
    units.to_file(os.path.join(GEO, "lb_units.gpkg"), layer="units", driver="GPKG")
    pd.DataFrame({"geo_id": list(PCODE.values()), "unit": list(PCODE.values())}).to_csv(
        os.path.join(GEO, "lb_lookup.csv"), index=False, encoding="utf-8")
    print(f"\nwrote {GEO}/lb_units.gpkg (26 cazas, {units['pop'].sum():,} people, "
          f"{a.sum():,.0f} km2) and lb_lookup.csv")


if __name__ == "__main__":
    main()
