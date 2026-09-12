"""El Salvador — boundaries for the 14 departamentos.

Writes data/geo/sv/sv_departamentos.gpkg and data/geo/sv/sv_lookup.csv.

OCHA COD-AB El Salvador (`cod-ab-slv`), the **shapefile** bundle rather than the geodatabase
on §12's Chile rule, read with `engine="fiona"` because pyogrio is the engine that has
silently returned zero features from a .gdb. The feature count is asserted either way.

## THE CODE JOIN IS AVAILABLE, LOOKS EXACTLY LIKE GUATEMALA'S, AND IS A PERMUTATION

`sources/gt_geo.py` joins on the pcode, because LAPOP's `prov` for Guatemala is 200 plus the
official department number and COD's pcode is `GT` plus the same number. **El Salvador looks
identical and is not.** LAPOP's `prov` is 300 plus the official west-to-east department
number; **COD's `SV` pcodes are ALPHABETICAL BY NAME.**

    LAPOP 302 Santa Ana      ->  COD SV02 is Cabañas
    LAPOP 303 Sonsonate      ->  COD SV03 is Chalatenango
    LAPOP 306 San Salvador   ->  COD SV06 is La Paz
    LAPOP 309 Cabañas        ->  COD SV09 is San Miguel

**Two of the fourteen coincide** — Ahuachapán at 01 and La Libertad at 05, which is exactly
enough to make a spot check pass. The other twelve are wrong, and **a permutation preserves
every total**, so no reconciliation, no national figure and no row count would ever show it.
San Salvador's 1.7 million people would have been drawn in La Paz.

So: **the join is on NAME and the code is demoted to evidence.** Names are unique on both
sides, there are no duplicates, and all fourteen fold to the same string with no aliases
needed. `check_code_join()` below asserts that the code join still mispairs, so that if OCHA
ever re-cuts these pcodes to the official order this file stops and someone decides,
rather than the map quietly improving or quietly breaking.

That is `sources/ni_geo.py`'s finding met a second time in a different country, and it is the
reason that file says what it says.

Usage:
    python sources/sv_geo.py --fetch    one ~3.9 MB zip from HDX, plus a 5 KB CSV
    python sources/sv_geo.py            rebuild from data/raw/sv/
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
RAW = os.path.join(ROOT, "data", "raw", "sv")
SHP_DIR = os.path.join(RAW, "shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "sv")
OUT = os.path.join(OUT_DIR, "sv_departamentos.gpkg")
LOOKUP = os.path.join(OUT_DIR, "sv_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

DOWNLOADS = {
    "slv_admin_boundaries.shp.zip":
        "https://data.humdata.org/dataset/08bfc7f6-3acc-45df-a69a-b853cf1cff05/resource/"
        "ef8b30ca-c268-409f-bf72-f7c0c23e54ea/download/slv_admin_boundaries.shp.zip",
    "slv_admpop_adm1_2024.csv":
        "https://data.humdata.org/dataset/89127169-6002-456f-adb1-7151d987467f/resource/"
        "de772d53-4ae2-4b3f-8686-c8fd622e7472/download/slv_admpop_adm1_2024.csv",
}

# LAPOP's `prov` numbering for El Salvador: 300 + the official west-to-east department
# number. This is NOT COD's ordering — see the module docstring. Written out so that a
# change on either side fails here rather than re-pairing something downstream.
LAPOP_DEPARTMENTS = {
    301: "Ahuachapán",   302: "Santa Ana",   303: "Sonsonate",    304: "Chalatenango",
    305: "La Libertad",  306: "San Salvador", 307: "Cuscatlán",   308: "La Paz",
    309: "Cabañas",      310: "San Vicente", 311: "Usulután",     312: "San Miguel",
    313: "Morazán",      314: "La Unión",
}

# How many of the fourteen the naive `prov - 300` -> `SVnn` join happens to get right.
# Two: Ahuachapán and La Libertad. If this ever changes, the pcodes have been re-cut.
CODE_JOIN_CORRECT = 2


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


def check_code_join(by_name):
    """Report what a `prov - 300` -> `SVnn` join would have done, and refuse to let it pass.

    Kept as an assertion rather than a comment because the failure it guards is invisible:
    every total reconciles under a permutation.
    """
    right, wrong = [], []
    for code, name in sorted(LAPOP_DEPARTMENTS.items()):
        naive = f"SV{code - 300:02d}"
        actual = by_name[fold(name)]
        (right if naive == actual else wrong).append((code, name, naive, actual))
    print(f"\n  witness 2 — the code join is NOT used. `prov - 300` -> SVnn would pair "
          f"{len(right)} of {len(LAPOP_DEPARTMENTS)} correctly and MISPAIR {len(wrong)}:")
    for code, name, naive, actual in wrong:
        other = next(n for n, p in by_name.items() if p == naive)
        print(f"      LAPOP {code} {name:<14} -> {naive}, which COD says is {other!r} "
              f"(the real one is {actual})")
    if len(right) != CODE_JOIN_CORRECT:
        raise SystemExit(
            f"the code join now gets {len(right)} of 14 right, not {CODE_JOIN_CORRECT}. "
            "OCHA has re-cut El Salvador's pcodes — STOP and decide deliberately whether "
            "this file should switch to them. Do not delete this assertion.")


def main():
    if "--fetch" in sys.argv:
        fetch()

    zpath = os.path.join(RAW, "slv_admin_boundaries.shp.zip")
    if not os.path.exists(zpath):
        raise SystemExit(f"{zpath} missing — run with --fetch")
    os.makedirs(SHP_DIR, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(SHP_DIR)

    # `slv_admin1.shp`, not `slv_admin1_em.shp`: the `_em` layers are the edge-matched
    # variants cut against neighbouring countries' boundaries, and they carry the same 14
    # features with a different coastline. The plain one is what every other country here uses.
    shp = os.path.join(SHP_DIR, "slv_admin1.shp")
    g = gpd.read_file(shp, engine="fiona")
    if len(g) != 14:
        raise SystemExit(f"{len(g)} ADM1 features, expected 14 — COD has re-cut El Salvador")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"unexpected CRS {g.crs}")
    print(f"read {shp}: {len(g)} departments, {g.crs}")

    g["pcode"] = g["adm1_pcode"].astype(str).str.strip()
    by_name = dict(zip(g["adm1_name"].map(fold), g["pcode"]))
    if len(by_name) != 14:
        raise SystemExit("COD's department names are not unique — the name join is unsafe")

    # ---- witness 1: every LAPOP department name is a COD department name ----
    missing = [n for n in LAPOP_DEPARTMENTS.values() if fold(n) not in by_name]
    spare = [n for n in g["adm1_name"]
             if fold(n) not in {fold(v) for v in LAPOP_DEPARTMENTS.values()}]
    if missing or spare:
        print(f"    LAPOP names with no polygon: {missing}")
        print(f"    polygons with no LAPOP name: {spare}")
        raise SystemExit("the name join FAILED")
    print(f"  witness 1 — all {len(LAPOP_DEPARTMENTS)} LAPOP names match a COD name exactly, "
          "no aliases needed")

    check_code_join(by_name)

    # ---- populations, joined on the pcode, which is COD's own key on both sides ----
    ppath = os.path.join(RAW, "slv_admpop_adm1_2024.csv")
    pop = pd.read_csv(ppath, encoding="utf-8-sig")
    pop["pcode"] = pop["ADM1_PCODE"].astype(str).str.strip()
    if len(pop) != 14 or set(pop["pcode"]) != set(g["pcode"]):
        raise SystemExit("COD-PS ADM1 does not cover the same 14 pcodes as COD-AB")
    g = g.merge(pop[["pcode", "T_TL"]], on="pcode", how="left")
    if g["T_TL"].isna().any():
        raise SystemExit("a department came out of the population join with no total")
    g["pop"] = g["T_TL"].astype("int64")
    print(f"  witness 3 — COD-PS 2024 joins on the pcode: {g['pop'].sum():,} people")

    # San Salvador is the capital department and by far the densest; Chalatenango, the
    # northern mountain border with Honduras, is the sparsest. A permuted population join
    # is what this catches.
    g["density"] = g["pop"] / g["area_sqkm"]
    lo = g.loc[g["density"].idxmin(), "adm1_name"]
    hi = g.loc[g["density"].idxmax(), "adm1_name"]
    print(f"    sparsest {lo!r}, densest {hi!r}")
    if fold(hi) != fold("San Salvador"):
        raise SystemExit("San Salvador is not the densest department — the population join "
                         "is permuted")

    g["unit"] = g["pcode"]
    # geo_id IS the pcode, deliberately: LAPOP's own numbering never leaves sources/sv.py,
    # so there is only one department numbering anywhere in data/ and nobody downstream can
    # pick the wrong one.
    g["geo_id"] = g["pcode"]
    g["name"] = g["adm1_name"]

    os.makedirs(OUT_DIR, exist_ok=True)
    out = g[["unit", "name", "pcode", "geo_id", "pop", "geometry"]]
    out.to_file(OUT, layer="departamentos", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    lut = pd.DataFrame({
        "geo_id": sorted(g["pcode"]),
        "unit": sorted(g["pcode"]),
        "name": [g.loc[g["pcode"] == p, "name"].iloc[0] for p in sorted(g["pcode"])],
        "lapop_prov": [next(c for c, n in LAPOP_DEPARTMENTS.items()
                            if by_name[fold(n)] == p) for p in sorted(g["pcode"])],
    })
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows, with the LAPOP prov code alongside)")


if __name__ == "__main__":
    main()
