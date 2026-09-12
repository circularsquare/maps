"""Grenada — boundaries for the 7 drawn units.

Writes data/geo/gd/gd_parishes.gpkg and data/geo/gd/gd_lookup.csv.

OCHA COD-AB Grenada, `grd_admin1.shp`. **COD GIVES 8 POLYGONS AND THE MAP DRAWS 7**, and the
two tiers disagree at both ends, in opposite directions:

  * **COD splits Carriacou from Petite Martinique** (GD01, GD08). The census does not — Table
    23 publishes one figure for `CARRIACOU AND PETITE MARTINIǪUE` — so the two polygons are
    **dissolved into one** here. Splitting one published figure between two islands would be
    inventing a magnitude (§14.4), and Petite Martinique is 2.4 km off Carriacou's east coast
    with a few hundred people on it.
  * **The census splits the Town of St. George from the rest of the parish** and COD does
    not. Nothing does: OpenStreetMap has the six parishes at `admin_level=6` and, for the
    capital, only a `place=town` **node**. So `sources/gd.py` folds the town back in. The
    fold is on the counting side, this file's join is 7-on-7, and `gd.csv` keeps the census's
    own 8-unit tier so nothing is lost from the record.

**AFTER BOTH, THE TIERS AGREE EXACTLY.** Six parishes and one dependency, which is also what
OSM independently has.

**THE ONLY NAME DIFFERENCE IS `St.` AGAINST THE CENSUS'S CLOSED-UP `ST.`**, plus the
dependency's own name. `fold()` drops everything that is not alphanumeric rather than
carrying an alias table (§12), and the dependency is matched as the union of the two COD
names rather than by string.

**THE JOIN IS BY NAME AND THE INDEPENDENT CHECK IS THE P-CODE.** CSO publishes no code, so
`sources/gd.py` carries COD's `adm1_pcode` against each parish by hand and this file asserts
the pairing from the boundary side — every total in `gd.py` reconciles whichever polygon a
parish is paired with, so nothing else would catch a transposition (§9n's `TMA` lesson).

Usage:
    python sources/gd_geo.py --fetch    one ~1.4 MB shapefile zip from HDX
    python sources/gd_geo.py            rebuild from data/raw/gd/
"""

import os
import sys
import unicodedata
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "gd")
OUT_DIR = os.path.join(ROOT, "data", "geo", "gd")
OUT = os.path.join(OUT_DIR, "gd_parishes.gpkg")
LOOKUP = os.path.join(OUT_DIR, "gd_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "gd.csv")

ZIP_URL = ("https://data.humdata.org/dataset/ffdc3580-45ee-4630-9abf-4dd8a8b002d3/"
           "resource/75a40c8b-06c5-4b35-8215-106fa370c985/download/"
           "grd_admin_boundaries.shp.zip")
ZIP_NAME = "grd_admin_boundaries.shp.zip"
SHP = "grd_admin1.shp"
COD_POLYGONS = 8
EXPECTED = 7

UTM = 32620                     # UTM 20N covers Grenada

# COD keeps these apart and the census does not. Dissolved into the first, which is the
# pcode sources/gd.py carries for the joint unit.
DEPENDENCY = ("GD01", "GD08")
DEPENDENCY_NAME = "Carriacou and Petite Martinique"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, ZIP_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 300_000:
        print("already have", dest)
        return
    print("GET", ZIP_URL)
    r = requests.get(ZIP_URL, timeout=900, stream=True, headers={"User-Agent": UA})
    r.raise_for_status()
    with open(dest + ".part", "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    os.replace(dest + ".part", dest)
    with open(dest, "rb") as fh:
        magic = fh.read(2)
    if magic != b"PK":
        raise SystemExit(f"{dest} is not a zip -- starts {magic!r}")
    print(f"  {os.path.getsize(dest):,} bytes")


def fold(s):
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s.lower() if c.isalnum())


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    zp = os.path.join(RAW, ZIP_NAME)
    if not os.path.exists(zp):
        raise SystemExit(f"missing {zp} -- run with --fetch first")
    with zipfile.ZipFile(zp) as z:
        if not any(n.endswith(SHP) for n in z.namelist()):
            raise SystemExit(f"{zp} has no {SHP} -- it holds {z.namelist()[:8]}")

    g = gpd.read_file(f"zip://{zp}!{SHP}")
    # §12 (Chile): a read that succeeds is not a read that returned data.
    if len(g) != COD_POLYGONS:
        raise SystemExit(f"{SHP} returned {len(g)} features, expected {COD_POLYGONS}")
    for c in ("adm1_name", "adm1_pcode"):
        if c not in g.columns:
            raise SystemExit(f"{SHP} has no {c!r} -- columns are {list(g.columns)}")
    g["pcode"] = g["adm1_pcode"].astype(str).str.strip()
    print(f"COD-AB ADM1: {len(g)} polygons, crs={g.crs}")
    for _, r in g.sort_values("pcode").iterrows():
        print(f"    {r['pcode']}  {r['adm1_name']}")

    # ---- the dissolve, and it must find exactly the two it is told to find ----
    have = set(g["pcode"])
    if not set(DEPENDENCY) <= have:
        raise SystemExit(f"COD no longer has both of {DEPENDENCY} -- it has "
                         f"{sorted(have)}. The dependency dissolve must be rechecked "
                         "against the new release before anything is drawn.")
    keep, merge = DEPENDENCY[0], list(DEPENDENCY)
    g["unit"] = g["pcode"].map(lambda c: keep if c in merge else c)
    dissolved = g.dissolve(by="unit", as_index=False)
    if len(dissolved) != EXPECTED:
        raise SystemExit(f"dissolve gave {len(dissolved)} units, expected {EXPECTED}")
    print(f"\n  dissolved {merge} -> {keep}: {COD_POLYGONS} polygons become "
          f"{len(dissolved)} units,\n  because the census publishes ONE figure for "
          f"{DEPENDENCY_NAME}")

    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/gd.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[])
    cen = (df[df["geo_level"] == "parish"]
           .drop_duplicates("geo_id")[["geo_id", "geo_name"]])
    census = dict(zip(cen["geo_id"], cen["geo_name"]))
    if len(census) != EXPECTED:
        raise SystemExit(f"{len(census)} census parishes, expected {EXPECTED}")

    # ---- the join. The dependency is matched as the UNION of COD's two names rather
    #      than by string, because the census's name is neither of them.
    cod_name = dict(zip(g["pcode"], g["adm1_name"]))
    poly = {}
    for u in dissolved["unit"]:
        if u == keep:
            poly[fold(DEPENDENCY_NAME)] = (DEPENDENCY_NAME, u)
        else:
            k = fold(cod_name[u])
            if k in poly:
                raise SystemExit(f"COD name {cod_name[u]!r} folds onto an existing one")
            poly[k] = (cod_name[u], u)

    pairs, missing = {}, []
    for code, nm in census.items():
        k = fold(nm)
        if k in poly:
            pairs[code] = poly[k]
        else:
            missing.append((code, nm))
    used = {v[1] for v in pairs.values()}
    spare = [(nm, u) for nm, u in poly.values() if u not in used]

    print("\n  the join, both ways (§12):")
    print(f"    census parishes         {len(census):>4}")
    print(f"    dissolved polygons      {len(poly):>4}")
    print(f"    matched                 {len(pairs):>4}")
    print(f"    census with no polygon  {len(missing):>4}  {missing}")
    print(f"    polygons with no census {len(spare):>4}  {spare}")
    if missing or spare:
        raise SystemExit("join FAILED")

    bad = [(c, census[c], p[1]) for c, p in pairs.items() if p[1] != c]
    print(f"\n    independent check — gd.py's name->pcode matches COD's adm1_pcode on "
          f"{len(pairs) - len(bad)}/{len(pairs)}")
    for c, nm, pc in bad:
        print(f"      {nm!r}: gd.py says {c}, COD says {pc}")
    if bad:
        raise SystemExit("gd.py's PARISHES pcodes disagree with COD -- every parish's "
                         "dots would be placed in the wrong parish")

    out = dissolved[["unit", "geometry"]].copy()
    # Take the NAME from the statistical source, not the boundary file (§12, Chile).
    out["name"] = out["unit"].map({v[1]: census[k] for k, v in pairs.items()})
    out["pcode"] = out["unit"]

    areas = out.to_crs(UTM).area / 1e6
    pop = {r["geo_id"]: int(r["count"]) for _, r in
           df[(df["geo_level"] == "parish") &
              (df["source_category"] == "TOTAL")].iterrows()}
    ordered = out.sort_values("unit")
    print(f"\n  {len(out)} drawn units:")
    for (_, row), a in zip(ordered.iterrows(), areas[ordered.index]):
        p = pop[row["unit"]]
        print(f"    {row['unit']}  {row['name']:<32} {a:6.1f} km2   {p:>7,} people   "
              f"{p / a:6.0f}/km2")

    os.makedirs(OUT_DIR, exist_ok=True)
    out[["unit", "name", "pcode", "geometry"]].to_file(
        OUT, layer="parishes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    pd.DataFrame({"geo_id": sorted(pairs), "unit": sorted(pairs)}).to_csv(
        LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(pairs)} rows)")


if __name__ == "__main__":
    main()
