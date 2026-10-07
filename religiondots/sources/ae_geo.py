"""United Arab Emirates: the seven emirates, from OCHA's COD-AB.

Writes data/geo/ae/ae_units.gpkg (`unit`, `pcode`, `pop`, geometry) and data/geo/ae/ae_lookup.csv
(`geo_id` -> `unit`), from data/raw/ae/:

  * COD-AB `cod-ab-are` (HDX, CC BY-IGO; valid 2023-05-15 to 2024-12-19), `are_admin1.geojson`:
    seven emirates, pcodes AE01-AE07, with their exclaves (Sharjah's east-coast towns, Ajman's
    Masfout and Manama, Dubai's Hatta) as parts of one feature each;
  * geoBoundaries gbOpen ARE ADM1 (OSM, Wambacher, 2017), a witness only.

The join is on pcode: `sources/ae.py` keys its rows by the same pcodes, and the names are checked
against COD's English names. Witness that neither key decides: each emirate's COD polygon against
geoBoundaries' polygon of the same name (IoU, at least `IOU_MIN` and `MARGIN` times the next best),
and both areas printed. `sources/ae_grid.py` counts Kontur's people on both layers.

Usage:
    python sources/ae_geo.py --fetch    both files (3.2 MB and 2.3 MB)
    python sources/ae_geo.py            rebuild from data/raw/ae/
"""

import io
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

from ae import EMIRATES

RAW = os.path.join(ROOT, "data", "raw", "ae")
GEO = os.path.join(ROOT, "data", "geo", "ae")
COD = os.path.join(RAW, "are_admin_boundaries.geojson.zip")
COD_URL = ("https://data.humdata.org/dataset/23d41c1f-41ef-4957-a47e-b8c08c984d83/resource/"
           "29e34577-9c64-4792-bcd0-1dbe1121e7f5/download/are_admin_boundaries.geojson.zip")
GB = os.path.join(RAW, "geoBoundaries-ARE-ADM1.geojson")
GB_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/ARE/ADM1/"
          "geoBoundaries-ARE-ADM1.geojson")
NORMALIZED = [os.path.join(ROOT, "data", "normalized", "ae.csv"),
              os.path.join(ROOT, "data", "normalized", "ae_foreign.csv")]
GB_NAMES = {"Abu Dhabi": "AE01", "Dubai": "AE02", "Sharjah": "AE03", "Ajman": "AE04",
            "Umm al-Quwain": "AE05", "Ras al-Khaimah": "AE06", "Fujairah": "AE07"}
IOU_MIN = 0.70
MARGIN = 5.0
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for url, dst in ((COD_URL, COD), (GB_URL, GB)):
        if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
            continue
        print("GET", url)
        r = requests.get(url, timeout=600, headers={"User-Agent": UA})
        r.raise_for_status()
        with open(dst + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(dst + ".part", dst)


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv or not (os.path.exists(COD) and os.path.exists(GB)):
        fetch()
    with zipfile.ZipFile(COD) as z:
        cod = gpd.read_file(io.BytesIO(z.read("are_admin1.geojson")))
    if len(cod) != 7 or set(cod["adm1_pcode"]) != set(EMIRATES):
        raise SystemExit(f"COD-AB ADM1 has {len(cod)} units, pcodes {sorted(cod['adm1_pcode'])}")
    bad = [(p, n) for p, n in zip(cod["adm1_pcode"], cod["adm1_name"]) if EMIRATES[p] != n]
    if bad:
        raise SystemExit(f"COD names differ from sources/ae.py's: {bad}")

    gb = gpd.read_file(GB)
    if set(gb["shapeName"]) != set(GB_NAMES):
        raise SystemExit(f"geoBoundaries names {sorted(gb['shapeName'])}")
    gb["pcode"] = gb["shapeName"].map(GB_NAMES)
    eq = "EPSG:6933"
    c_m = cod.set_index("adm1_pcode").geometry.to_crs(eq)
    g_m = gb.set_index("pcode").geometry.to_crs(eq)
    # The two layers trace the northern emirates' inland lines and coasts differently, so a right
    # pairing reads IoU 0.76-0.99 (Umm Al Quwain lowest). What the witness must rule out is a
    # permuted pcode: each COD emirate's own geoBoundaries polygon must be the one it overlaps most,
    # by at least `MARGIN` times the next.
    print("  emirate            COD km2   gB km2    IoU   next best IoU   COD parts")
    bad = []
    for p in EMIRATES:
        a = c_m[p]
        ious = {q: a.intersection(g_m[q]).area / a.union(g_m[q]).area for q in EMIRATES}
        own = ious.pop(p)
        nxt = max(ious.values())
        parts = len(a.geoms) if hasattr(a, "geoms") else 1
        print(f"  {EMIRATES[p]:<16} {a.area / 1e6:>9,.0f} {g_m[p].area / 1e6:>8,.0f}  {own:.3f}"
              f"   {nxt:.3f}        {parts:>5}")
        if own < IOU_MIN or own < MARGIN * nxt:
            bad.append(p)
    if bad:
        raise SystemExit(f"COD and geoBoundaries do not pair by name for {bad}")

    missing = [p for p in NORMALIZED if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing {missing}; run sources/ae.py first")
    rows = pd.concat([pd.read_csv(p, usecols=["geo_id", "count"]) for p in NORMALIZED])
    pop = rows.groupby("geo_id")["count"].sum()
    if set(pop.index) != set(EMIRATES):
        raise SystemExit(f"normalized rows cover {sorted(pop.index)}")

    units = cod[["adm1_pcode", "adm1_name", "geometry"]].rename(
        columns={"adm1_pcode": "pcode", "adm1_name": "unit"})
    units["pop"] = units["pcode"].map(pop).astype(int)
    os.makedirs(GEO, exist_ok=True)
    units.to_file(os.path.join(GEO, "ae_units.gpkg"), layer="units", driver="GPKG")
    pd.DataFrame({"geo_id": list(EMIRATES), "unit": list(EMIRATES.values())}).to_csv(
        os.path.join(GEO, "ae_lookup.csv"), index=False, encoding="utf-8")
    print(f"\nwrote {GEO}/ae_units.gpkg (7 emirates, {units['pop'].sum():,} people) and ae_lookup.csv")


if __name__ == "__main__":
    main()
