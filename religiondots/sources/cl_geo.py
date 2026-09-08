"""Chile — boundaries for the 346 census comunas.

Writes data/geo/cl/cl_comunas.gpkg and data/geo/cl/cl_lookup.csv.

Source: OCHA COD-AB `cod-ab-chl` v01, `chl_admin3` — **345 comuna polygons**, valid_on
2021-10-08, reviewed 2025-02-17. 205 MB of shapefile, nearly all of it the fjord coastline
at full resolution.

**THE 92 MB GEODATABASE IS THE SMALLER FILE AND IT CANNOT BE READ HERE — and it fails in
the worst possible way.** `chl_admin_boundaries.gdb.zip` is less than half the size and was
tried first. GDAL 3.8.5's OpenFileGDB driver opens it, lists all six layers with the right
geometry types, reports `crs=EPSG:4326`, and returns **zero features** from every one of
them. No exception, no warning, no empty-file error: a successful read of nothing.

That is §5a's "HTTP 200 is not a download" wearing new clothes, and it generalises past
this country — see spec §12. **A read that succeeds is not a read that returned data.**
Assert the feature count, not the absence of an exception; had this script trusted the
call, Chile would have failed later with an empty join and the cause would have looked
like the join rather than the driver.

**The join is by CUT and it is the easy case — which is exactly why it gets checked.**
COD's `adm3_pcode` is `CL` + the five-digit Código Único Territorial, and INE's census
table carries the CUT directly. 345 match, both ways, with no spares. After Sri Lanka
(sources/lk_geo.md) a clean code join is not something to accept on its own evidence, so
two independent checks run below: names, and a population ratio the codes do not determine.

Unlike Sri Lanka's, this one survives them. The CUT is a stable national standard
maintained by SUBDERE, both files are within four years of each other, and Chile has not
renumbered.

**ONE COMUNA HAS NO POLYGON: Antártica (CUT 12202), 60 people aged 15 or over.** COD's
admin3 stops at the continental and island territory and does not carry the Antarctic
claim, which is the right call for a boundary set and would in any case put a dot near the
South Pole. It is dropped, named here and named in the build log, and at 1:1,000 it draws
no dot either way.

Usage:
    python sources/cl_geo.py --fetch    205 MB shapefile bundle from HDX
    python sources/cl_geo.py
"""

import csv
import os
import re
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "cl")
SHP = os.path.join(RAW, "chl_admin_boundaries.shp.zip")
OUT_DIR = os.path.join(ROOT, "data", "geo", "cl")
OUT = os.path.join(OUT_DIR, "cl_comunas.gpkg")
LOOKUP = os.path.join(OUT_DIR, "cl_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "cl.csv")
POP = os.path.join(RAW, "D1_Poblacion-por-sexo-y-edad.xlsx")

URL = ("https://data.humdata.org/dataset/70ca85b0-336a-407e-b9e5-f32a7e4840d0/resource/"
       "91f5ea86-b409-4185-a520-b8586c2c42d0/download/chl_admin_boundaries.shp.zip")

LAYER = "chl_admin3.shp"
EXPECTED_POLYGONS = 345
EXPECTED_COMUNAS = 346
ANTARCTICA = "12202"

# The three comunas where COD's name and INE's disagree. Two are spelling. The third is a
# real error in COD and is the reason this check exists rather than being a formality.
KNOWN_NAME_DIFFS = {
    "06204": "spelling — Marchihue / Marchigüe, the same comuna.",
    "16207": "spelling — Trehuaco / Treguaco, the same comuna.",
    "01401": (
        "COD IS WRONG, NOT THE JOIN. CL01401 is named 'Tocopilla' in COD, but its province "
        "is Tamarugal, its region Tarapacá and its area 13,738 km2 — which is Pozo Almonte "
        "(13,766 km2), 400 km away from the real Tocopilla. COD carries the name twice and "
        "has no polygon called Pozo Almonte at all; the genuine Tocopilla is CL02301, in "
        "Antofagasta, 4,101 km2, present and correct. So the GEOMETRY here is Pozo Almonte's "
        "and only the label is wrong. This script therefore writes INE's names, not COD's."),
}


def fetch():
    import requests
    import urllib3
    import zipfile
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(SHP) and os.path.getsize(SHP) > 150_000_000:
        print("already have", SHP)
        return
    print("GET", URL)
    with requests.get(URL, timeout=1800, verify=False, stream=True,
                      headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        with open(SHP, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    if not zipfile.is_zipfile(SHP):
        raise SystemExit(f"{SHP} is not a zip")
    with zipfile.ZipFile(SHP) as z:
        if LAYER not in z.namelist():
            raise SystemExit(f"no {LAYER} in the bundle: {z.namelist()[:8]}")
    print(f"  {os.path.getsize(SHP):,} bytes -> {SHP}")


def fold(s):
    """Compare Spanish place names without accents or punctuation. Used only to REPORT
    agreement on an already-made code join, never to make one."""
    s = unicodedata.normalize("NFKD", str(s)).lower()
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", s)


def _read_layer():
    """Read the comuna layer and ASSERT IT HAS FEATURES.

    The zero-feature check is the point of this function, not a formality: the geodatabase
    edition of this same dataset returns an empty frame with a valid CRS on this machine
    (see the module docstring), so `read_file` succeeding proves nothing.
    """
    import geopandas as gpd
    gdf = gpd.read_file(f"zip://{SHP}!{LAYER}")
    if len(gdf) == 0:
        raise SystemExit(
            f"{LAYER} read cleanly and returned ZERO features. That is the geodatabase "
            "failure mode in the docstring, now happening to the shapefile: check the GDAL "
            "driver rather than the file.")
    return gdf


def main():
    import pandas as pd

    if not os.path.exists(SHP):
        raise SystemExit(f"missing {SHP} -- run with --fetch first")
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/cl.py first")

    # ---- census side ----
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    uni = df[df["note"].str.contains("universe total")].copy()
    if len(uni) != EXPECTED_COMUNAS:
        raise SystemExit(f"{len(uni)} census comunas, expected {EXPECTED_COMUNAS}")
    cen_name = dict(zip(uni["geo_id"], uni["geo_name"]))
    cen_pop = dict(zip(uni["geo_id"], uni["count"]))
    print(f"census comunas: {len(uni)}")

    # ---- geo side ----
    print(f"reading {LAYER} from the bundle (205 MB shapefile, ~1 min)…")
    cod = _read_layer()
    print(f"COD admin3 polygons: {len(cod):,}  crs={cod.crs}")
    if len(cod) != EXPECTED_POLYGONS:
        raise SystemExit(f"expected {EXPECTED_POLYGONS} polygons")
    cod["adm3_pcode"] = cod["adm3_pcode"].astype(str).str.strip()
    if not cod["adm3_pcode"].str.fullmatch(r"CL\d{5}").all():
        raise SystemExit("adm3_pcode is not CL+5 digits -- COD has changed the format")
    cod["cut"] = cod["adm3_pcode"].str[2:]
    if cod["cut"].duplicated().any():
        raise SystemExit("duplicate CUT in COD")

    # ---- the join, both ways (§12) ----
    c, g = set(cen_pop), set(cod["cut"])
    matched, only_cen, only_cod = c & g, c - g, g - c
    print(f"\n  the join, both ways:")
    print(f"    matched                  {len(matched):>5}")
    print(f"    census with no polygon   {len(only_cen):>5}  "
          f"{[(x, cen_name[x], cen_pop[x]) for x in sorted(only_cen)]}")
    print(f"    polygons with no census  {len(only_cod):>5}  {sorted(only_cod)}")
    if only_cod or only_cen != {ANTARCTICA}:
        raise SystemExit("join FAILED -- the only expected miss is Antártica (12202)")
    print(f"    the one miss is Antártica, {cen_pop[ANTARCTICA]} people aged 15+, which COD "
          "does not carry\n      and which draws no dot at 1:1,000 in any case.")

    # ---- check 1: names. Independent of the code, and it earned its place. ----
    cod_name = dict(zip(cod["cut"], cod["adm3_name"]))
    diffs = {k for k in matched if fold(cen_name[k]) != fold(cod_name.get(k, ""))}
    agree = len(matched) - len(diffs)
    print(f"\n  check 1 — comuna names agree after folding on {agree} of {len(matched)} "
          f"({100.0 * agree / len(matched):.1f}%)")
    for k in sorted(diffs):
        print(f"      {k}  census {cen_name[k]!r}  vs COD {cod_name.get(k)!r}"
              f"   {KNOWN_NAME_DIFFS.get(k, '*** NEW, INVESTIGATE ***')}")
    new = diffs - set(KNOWN_NAME_DIFFS)
    if new:
        raise SystemExit(f"unexplained name disagreement on {sorted(new)} -- do not assume "
                         "it is a spelling variant; check the province, region and area the "
                         "way 01401 was checked")
    if set(KNOWN_NAME_DIFFS) - diffs:
        print(f"      note: {sorted(set(KNOWN_NAME_DIFFS) - diffs)} now agree — COD has been "
              "corrected upstream, and this table can lose them")

    # ---- check 2: a quantity the codes do not determine ----
    # The census's 15+ count per comuna against that comuna's TOTAL population from D1.
    # A correct join makes the ratio systematic — everyone is somewhere near 0.8. A
    # scrambled one pairs a retirement comuna with a young one and scatters it. This is
    # the check Sri Lanka's pcode join would have failed.
    if os.path.exists(POP):
        d1 = pd.read_excel(POP, sheet_name="2", header=3)
        d1 = d1[d1["Código comuna"].notna()]
        d1["cut"] = d1["Código comuna"].astype(int).astype(str).str.zfill(5)
        tot = dict(zip(d1["cut"], d1["Población censada"]))
        ratios = sorted((cen_pop[k] / tot[k], k) for k in matched
                        if tot.get(k) and tot[k] > 0)
        med = ratios[len(ratios) // 2][0]
        print(f"\n  check 2 — census 15+ / total population, on {len(ratios)} comunas:")
        print(f"    min {ratios[0][0]:.3f} ({cen_name[ratios[0][1]]})   median {med:.3f}   "
              f"max {ratios[-1][0]:.3f} ({cen_name[ratios[-1][1]]})")
        if not (0.72 <= med <= 0.88) or ratios[0][0] < 0.55 or ratios[-1][0] > 1.00:
            raise SystemExit("the 15+ share is not systematic -- the join is pairing the "
                             "wrong comunas")
        print("    OK  systematic, not scattered — every comuna's 15+ share sits in a narrow "
              "band,\n        which a scrambled join cannot produce.")
    else:
        print(f"\n  check 2 SKIPPED — {POP} missing; run sources/cl.py --fetch")

    # ---- write ----
    # NAMES COME FROM THE CENSUS, NOT FROM COD. INE is authoritative for Chilean comuna
    # names, and COD mislabels CL01401 (see KNOWN_NAME_DIFFS), so taking COD's would put
    # "Tocopilla" on Pozo Almonte's polygon in every tooltip.
    out = cod[cod["cut"].isin(matched)][["cut", "area_sqkm", "geometry"]].copy()
    out["name"] = out["cut"].map(cen_name)
    out = out.rename(columns={"cut": "comuna"})[["comuna", "name", "area_sqkm", "geometry"]]
    if out["name"].isna().any():
        raise SystemExit("a polygon has no census name")
    if out.geometry.isna().any() or out.geometry.is_empty.any():
        raise SystemExit("empty geometries in the output")
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_file(OUT, layer="comunas", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    with open(LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "adm3_pcode", "drawn"])
        for k in sorted(c):
            w.writerow([k, f"CL{k}" if k in matched else "", int(k in matched)])
    print(f"wrote {LOOKUP} ({len(c)} rows, {len(matched)} drawn)")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    main()
