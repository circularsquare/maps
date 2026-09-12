"""Guatemala — boundaries for the 22 departamentos.

Writes data/geo/gt/gt_departamentos.gpkg and data/geo/gt/gt_lookup.csv.

OCHA COD-AB Guatemala (`cod-ab-gtm`), the **shapefile** bundle rather than the geodatabase on
§12's Chile rule: GDAL's OpenFileGDB driver has been seen to open a .gdb, list its layers,
report the right CRS and return ZERO features while raising nothing. Read with
`engine="fiona"`, because pyogrio is geopandas' default when installed and is the engine that
has silently returned zero. The feature count is asserted either way.

**THIS IS THE EASIEST JOIN IN THE PROJECT AND THE FILE STILL CHECKS IT TWICE.** Guatemala's
22 departments have carried the same official numbering since the nineteenth century, and
all three sides agree on it independently:

    LAPOP `prov`    201..222   -> department 01..22   (sources/gt.py)
    COD-AB          GT01..GT22
    COD-PS          GT01..GT22

So the code join and the name join are two separate signals over the same 22 rows, and this
file requires BOTH to succeed. sources/ni_geo.py is the reason: Nicaragua's code join looked
perfect, was available, matched 145 of 153, and **silently sent Waspám's Moravians inland**.
A permutation preserves every total, so no arithmetic check can find one. Two independent
keys can.

**AND THE THIRD WITNESS IS THE POPULATION.** COD-PS's department populations are joined here
and checked against the polygons' own areas only in the loose sense that Petén — 35,903 km²,
a fifth of the country — must come out large and thinly populated. The real population check
lives in `sources/gt.py`, which has LAPOP's own department sample to compare against.

Usage:
    python sources/gt_geo.py --fetch    one ~2.9 MB zip from HDX, plus a 7 KB CSV
    python sources/gt_geo.py            rebuild from data/raw/gt/
"""

import os
import sys
import unicodedata
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "gt")
SHP_DIR = os.path.join(RAW, "shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "gt")
OUT = os.path.join(OUT_DIR, "gt_departamentos.gpkg")
LOOKUP = os.path.join(OUT_DIR, "gt_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

# cod-ab-gtm and cod-ps-gtm on HDX. The resource ids are stable; the dataset ids are the
# uuids in the middle of each path.
DOWNLOADS = {
    "gtm_admin_boundaries.shp.zip":
        "https://data.humdata.org/dataset/0b20f310-7d22-479c-b7e2-e1bb9737fa72/resource/"
        "56c73009-60a8-4987-88b2-bc493f8b544c/download/gtm_admin_boundaries.shp.zip",
    "gtm_admpop_adm1_2024.csv":
        "https://data.humdata.org/dataset/452f4fe9-6baf-46ef-a50e-7a6b1862d9fb/resource/"
        "a983f193-2cb3-4f7c-af0b-6b889b7201bc/download/gtm_admpop_adm1_2024.csv",
}

# The 22 departments in Guatemala's own official order, which is what the pcode encodes and
# what LAPOP's `prov` code encodes. Written out so a change on either side is a failure here
# rather than a quiet re-pairing downstream.
DEPARTMENTS = {
    "01": "Guatemala",       "02": "El Progreso",    "03": "Sacatepéquez",
    "04": "Chimaltenango",   "05": "Escuintla",      "06": "Santa Rosa",
    "07": "Sololá",          "08": "Totonicapán",    "09": "Quetzaltenango",
    "10": "Suchitepéquez",   "11": "Retalhuleu",     "12": "San Marcos",
    "13": "Huehuetenango",   "14": "Quiché",         "15": "Baja Verapaz",
    "16": "Alta Verapaz",    "17": "Petén",          "18": "Izabal",
    "19": "Zacapa",          "20": "Chiquimula",     "21": "Jalapa",
    "22": "Jutiapa",
}


def fold(s):
    """Accent- and case-insensitive key for a department name."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalnum())


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for name, url in DOWNLOADS.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst):
            print(f"  have {name} ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=300) as r, open(dst, "wb") as f:
            f.write(r.read())
        print(f"  got  {name} ({os.path.getsize(dst):,} bytes)")


def main():
    if "--fetch" in sys.argv:
        fetch()

    zpath = os.path.join(RAW, "gtm_admin_boundaries.shp.zip")
    if not os.path.exists(zpath):
        raise SystemExit(f"{zpath} missing — run with --fetch")
    os.makedirs(SHP_DIR, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(SHP_DIR)

    shp = os.path.join(SHP_DIR, "gtm_admin1.shp")
    g = gpd.read_file(shp, engine="fiona")
    if len(g) != 22:
        raise SystemExit(f"{len(g)} ADM1 features, expected 22 — COD has re-cut Guatemala")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"unexpected CRS {g.crs}")
    print(f"read {shp}: {len(g)} departments, {g.crs}")

    g["pcode"] = g["adm1_pcode"].astype(str).str.strip()
    g["geo_id"] = g["pcode"].str.replace("^GT", "", regex=True).str.zfill(2)

    # ---- witness 1: the pcode's own two digits are the official department number ----
    bad_code = sorted(set(g["geo_id"]) ^ set(DEPARTMENTS))
    if bad_code:
        raise SystemExit(f"COD pcodes do not cover 01..22 exactly: {bad_code}")
    print("  witness 1 — COD's pcodes are exactly Guatemala's 01..22")

    # ---- witness 2: and the NAME on each of those codes is the expected one ----
    mism = [(c, DEPARTMENTS[c], n) for c, n in zip(g["geo_id"], g["adm1_name"])
            if fold(n) != fold(DEPARTMENTS[c])]
    if mism:
        for c, want, got in mism:
            print(f"    {c}: expected {want!r}, COD says {got!r}")
        raise SystemExit("a COD pcode carries a different department's name — the numbering "
                         "has changed and every downstream join is now a permutation. STOP.")
    print(f"  witness 2 — the name on each pcode matches on all {len(g)}, independently of "
          "the code")

    # ---- populations, from COD-PS, joined on the same pcode ----
    ppath = os.path.join(RAW, "gtm_admpop_adm1_2024.csv")
    pop = pd.read_csv(ppath, encoding="utf-8-sig")
    pop["pcode"] = pop["ADM1_PCODE"].astype(str).str.strip()
    if len(pop) != 22 or set(pop["pcode"]) != set(g["pcode"]):
        raise SystemExit("COD-PS ADM1 does not cover the same 22 pcodes as COD-AB")
    g = g.merge(pop[["pcode", "T_TL"]], on="pcode", how="left")
    if g["T_TL"].isna().any():
        raise SystemExit("a department came out of the population join with no total")
    g["pop"] = g["T_TL"].astype("int64")
    print(f"  witness 3 — COD-PS 2024 joins on the same key: {g['pop'].sum():,} people")

    # Petén is 35,903 km² and the emptiest department; Guatemala is the fullest. If the
    # population join is permuted this is what changes shape.
    g["density"] = g["pop"] / g["area_sqkm"]
    lo = g.loc[g["density"].idxmin(), "adm1_name"]
    hi = g.loc[g["density"].idxmax(), "adm1_name"]
    print(f"    sparsest {lo!r}, densest {hi!r}")
    if fold(lo) != fold("Petén") or fold(hi) != fold("Guatemala"):
        raise SystemExit("Petén is not the sparsest department or Guatemala is not the "
                         "densest — the population join is permuted")

    # Take the NAME from DEPARTMENTS rather than from the boundary file (§12, Chile): the
    # statistical source names the unit, the polygon only draws it.
    g["unit"] = g["pcode"]
    g["name"] = g["geo_id"].map(DEPARTMENTS)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = g[["unit", "name", "pcode", "geo_id", "pop", "geometry"]]
    out.to_file(OUT, layer="departamentos", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    lut = pd.DataFrame({"geo_id": sorted(DEPARTMENTS),
                        "unit": [f"GT{c}" for c in sorted(DEPARTMENTS)],
                        "name": [DEPARTMENTS[c] for c in sorted(DEPARTMENTS)]})
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows)")


if __name__ == "__main__":
    main()
