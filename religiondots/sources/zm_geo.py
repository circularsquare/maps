"""Zambia — boundaries for the 156 parliamentary constituencies.

Writes data/geo/zm/zm_constituencies.gpkg and data/geo/zm/zm_lookup.csv.

  * **boundaries** — OCHA COD-AB Zambia (`cod-ab-zmb`, version 02), the **shapefile** bundle
    rather than the geodatabase on §12's Chile rule, read with `engine="fiona"`. Admin 3 is
    the constituency: 156 features, `ZM101001001`.., nested in 116 districts (admin 2) and 10
    provinces (admin 1). HDX's own notes give 10 / 116 / 156 / 1,853.
  * **no population is read here.** The religion tables' own constituency totals are the
    counting universe, and `sources/zm.py` joins the census table to this lookup by name and
    asserts the join; putting a second population into this file would invite the two to be
    confused (de facto 18,340,343 against the census's de jure 19,693,423, see zm.py).

## WHY CONSTITUENCIES AND NOT DISTRICTS

ZamStats' 2022 religion volume (Series B, April 2026) prints Table B.4 and B.5 at province,
district AND constituency, and the constituency is the finest thing it prints. COD-AB carries
the same 156 at admin 3, so the finer of the two tiers costs nothing. Both tiers only split
Christianity from everything else; the denominations are province-level (sources/zm.py).

## THE VINTAGE

COD-AB says `valid_on 2023-06-29`, boundaries created 2020-11-26. The census reports use "the
current administrative boundaries as revised in December 2021" (Summary Report Part 2,
foreword). The count agrees (156 constituencies, 116 districts) and zm.py asserts every census
constituency lands on exactly one polygon inside the district of the same name, which is the
check a boundary revision would fail.

Usage:
    python sources/zm_geo.py --fetch    one ~104 MB zip from HDX (all five admin levels)
    python sources/zm_geo.py            rebuild from data/raw/zm/
"""

import os
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "zm")
SHP_DIR = os.path.join(RAW, "shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "zm")
OUT = os.path.join(OUT_DIR, "zm_constituencies.gpkg")
LOOKUP = os.path.join(OUT_DIR, "zm_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

ZIP_NAME = "zmb_admin_boundaries.shp.zip"
ZIP_URL = ("https://data.humdata.org/dataset/01c71cfd-32ff-4a32-a1dd-5909702c12db/resource/"
           "7e9c310d-b9d1-4f82-8ad5-09c1ce744482/download/zmb_admin_boundaries.shp.zip")

EXPECTED = {"admin1": 10, "admin2": 116, "admin3": 156}


def fetch():
    os.makedirs(RAW, exist_ok=True)
    dst = os.path.join(RAW, ZIP_NAME)
    if os.path.exists(dst) and os.path.getsize(dst) > 50_000_000:
        print(f"  have {ZIP_NAME} ({os.path.getsize(dst):,} bytes)")
    else:
        req = urllib.request.Request(ZIP_URL, headers=UA)
        with urllib.request.urlopen(req, timeout=1800) as r:
            data = r.read()
        # §5a: a 200 is not a download.
        if data[:2] != b"PK":
            raise SystemExit(f"{ZIP_NAME} is not a zip -- starts {data[:16]!r}")
        with open(dst + ".part", "wb") as f:
            f.write(data)
        os.replace(dst + ".part", dst)
        print(f"  got  {ZIP_NAME} ({os.path.getsize(dst):,} bytes)")
    with zipfile.ZipFile(dst) as z:
        z.extractall(SHP_DIR)


def main():
    if "--fetch" in sys.argv:
        fetch()

    layers = {}
    for lvl, n in EXPECTED.items():
        shp = os.path.join(SHP_DIR, f"zmb_{lvl}.shp")
        if not os.path.exists(shp):
            raise SystemExit(f"missing {shp} -- run with --fetch first")
        # engine="fiona": pyogrio is the engine that has silently returned zero features here.
        g = gpd.read_file(shp, engine="fiona")
        if len(g) != n:
            raise SystemExit(f"{shp} has {len(g)} features, expected {n}")
        layers[lvl] = g
    g = layers["admin3"]
    print(f"COD-AB Zambia admin3: {len(g)} constituencies, crs={g.crs}, "
          f"valid_on {sorted(set(g['valid_on']))}, version {sorted(set(g['version']))}")

    # The nesting, asserted: every constituency's district is an admin-2 feature and every
    # district's province an admin-1 feature, by pcode AND by name.
    a2 = dict(zip(layers["admin2"]["adm2_pcode"], layers["admin2"]["adm2_name"]))
    a1 = dict(zip(layers["admin1"]["adm1_pcode"], layers["admin1"]["adm1_name"]))
    bad = [(p, n) for p, n in zip(g["adm2_pcode"], g["adm2_name"]) if a2.get(p) != n]
    bad += [(p, n) for p, n in zip(g["adm1_pcode"], g["adm1_name"]) if a1.get(p) != n]
    if bad:
        raise SystemExit(f"admin3 parent codes/names disagree with admin1/admin2: {bad[:5]}")
    if set(g["adm2_pcode"]) != set(a2):
        raise SystemExit("some district holds no constituency")
    if g["adm3_pcode"].duplicated().any():
        raise SystemExit("duplicate admin3 pcodes")
    dup = g.duplicated(subset=["adm2_name", "adm3_name"])
    if dup.any():
        raise SystemExit(f"two constituencies share a name inside one district: "
                         f"{g.loc[dup, ['adm2_name', 'adm3_name']].values.tolist()}")

    g = g.rename(columns={"adm3_pcode": "unit", "adm3_name": "name",
                          "adm2_name": "district", "adm1_name": "province"})
    g["geo_id"] = g["unit"]

    print(f"  area {g['area_sqkm'].min():,.0f} to {g['area_sqkm'].max():,.0f} km2 "
          f"({g.loc[g['area_sqkm'].idxmin(), 'name']} / {g.loc[g['area_sqkm'].idxmax(), 'name']})")

    os.makedirs(OUT_DIR, exist_ok=True)
    keep = g[["geo_id", "unit", "name", "district", "province", "adm2_pcode", "adm1_pcode",
              "area_sqkm", "geometry"]]
    keep.to_file(OUT, layer="constituencies", driver="GPKG")
    pd.DataFrame(keep.drop(columns="geometry")).to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"\nwrote {OUT}")
    print(f"wrote {LOOKUP}")


if __name__ == "__main__":
    main()
