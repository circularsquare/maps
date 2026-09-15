"""Colombia — boundaries for the 33 departamentos, and the population that goes on them.

Writes data/geo/co/co_departamentos.gpkg, data/geo/co/co_lookup.csv and
data/geo/co/co_pop_2025.csv.

OCHA COD-AB Colombia (`cod-ab-col`, the 2020-04-16 MGN shapefile bundle) for the polygons and
COD-PS 2025 for the people. COD-PS here is **DANE's own post-2018-census projection** (the
explanatory note on HDX: source DANE, baseline 2018 census, reference year 2025), so unlike
Ecuador there is no office count it could be replaced with; Colombia's last census is 2018.

## THE JOIN: CODE AND NAME AGREE, AND BOTH ARE REQUIRED

LAPOP's `prov` for Colombia is **800 plus the DANE department code**, and COD's pcode is `CO`
plus the same code, so `prov - 800` is the pcode. DANE's codes are not contiguous (05, 08, 11,
13 ... 97), which is itself evidence that these are DANE's and not a survey's own ordering:
Honduras's merge (sources.md §11ap) numbered its departments 1..N in a private order, and no
private order produces 885 and 897. The name join is checked beside it on every run
(`sources/ec_geo.py`'s construction), all 26 folding to the same string with no alias.

**Seven departments have no LAPOP code at all**, not merely no respondents: the merge's
`prov_es` label set has 26 entries in the 800s. They are asserted here by pcode, carried with
their population and an empty `lapop_prov`, and `sources/co.py` decides what to draw there.

Usage:
    python sources/co_geo.py --fetch    a 117 MB zip and two CSVs from HDX
    python sources/co_geo.py            rebuild from data/raw/co/
"""

import os
import re
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
RAW = os.path.join(ROOT, "data", "raw", "co")
SHP_DIR = os.path.join(RAW, "shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "co")
OUT = os.path.join(OUT_DIR, "co_departamentos.gpkg")
LOOKUP = os.path.join(OUT_DIR, "co_lookup.csv")
POP_OUT = os.path.join(OUT_DIR, "co_pop_2025.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

DOWNLOADS = {
    "col-administrative-divisions-shapefiles.zip":
        "https://data.humdata.org/dataset/50ea7fee-f9af-45a7-8a52-abb9c790a0b6/resource/"
        "32fba556-0109-4d1c-84cb-c8abddf7775b/download/col-administrative-divisions-shapefiles.zip",
    "col_admpop_adm1_2025.csv":
        "https://data.humdata.org/dataset/8520e386-9263-48c9-b1bf-b2349e019fbb/resource/"
        "d1ef58d9-677d-4e8f-ad54-b439a014be1e/download/copy-of-col_admpop_adm1_2025.csv",
    # municipality names by DANE code: sources/co.py's witness for LAPOP's `municipio` labels
    "col_admpop_adm2_2025.csv":
        "https://data.humdata.org/dataset/8520e386-9263-48c9-b1bf-b2349e019fbb/resource/"
        "56f0ef9b-b1df-4d7f-b2e1-6cb27aaecdb4/download/copy-of-col_admpop_adm2_2025.csv",
}
SHP = "col_admbnda_adm1_mgn_20200416.shp"

# LAPOP's `prov_es` value labels in the 800s, verbatim from the Grand Merge. 800 + DANE code.
LAPOP_DEPARTMENTS = {
    805: "Antioquia",       808: "Atlántico",       811: "Bogotá, D.C.",
    813: "Bolívar",         815: "Boyacá",          817: "Caldas",
    818: "Caquetá",         819: "Cauca",           820: "Cesar",
    823: "Córdoba",         825: "Cundinamarca",    841: "Huila",
    844: "La Guajira",      847: "Magdalena",       850: "Meta",
    852: "Nariño",          854: "Norte de Santander", 863: "Quindio",
    866: "Risaralda",       868: "Santander",       870: "Sucre",
    873: "Tolima",          876: "Valle del Cauca", 885: "Casanare",
    886: "Putumayo",        897: "Vaupes",
}

# The departments LAPOP has no code for. Asserted by pcode, so a change in either file stops.
NEVER_SAMPLED = {
    "CO27": "Chocó", "CO81": "Arauca", "CO88": "San Andrés, Providencia y Santa Catalina",
    "CO91": "Amazonas", "CO94": "Guainía", "CO95": "Guaviare", "CO99": "Vichada",
}

EXPECTED = 33
CODPS_TOTAL = 53_216_592


def fold(s):
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
        with urllib.request.urlopen(req, timeout=1800) as r, open(dst + ".part", "wb") as f:
            f.write(r.read())
        os.replace(dst + ".part", dst)
        print(f"  got  {name} ({os.path.getsize(dst):,} bytes)")


def main():
    if "--fetch" in sys.argv:
        fetch()

    zpath = os.path.join(RAW, "col-administrative-divisions-shapefiles.zip")
    if not os.path.exists(zpath):
        raise SystemExit(f"{zpath} missing — run with --fetch")
    os.makedirs(SHP_DIR, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        members = [n for n in z.namelist() if n.startswith(SHP[:-4])]
        if not members:
            raise SystemExit("no adm1 members in the COD-AB zip — it has been re-cut")
        for n in members:
            z.extract(n, SHP_DIR)

    # NOT `engine="fiona"`, against §12's Chile rule, because fiona cannot read this file at
    # all: one of COD's date fields holds year 0 and fiona raises `year 0 is out of range`.
    # The Chile rule is about pyogrio returning zero features from a .gdb; this is a
    # shapefile, and the feature count is asserted on the next line, which is the actual guard.
    g = gpd.read_file(os.path.join(SHP_DIR, SHP), engine="pyogrio")
    if len(g) != EXPECTED:
        raise SystemExit(f"{len(g)} ADM1 features, expected {EXPECTED} — COD has re-cut Colombia")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"unexpected CRS {g.crs}")
    print(f"read {SHP}: {len(g)} departments, {g.crs}")

    g["pcode"] = g["ADM1_PCODE"].astype(str).str.strip()
    g["name"] = g["ADM1_ES"].astype(str).str.strip()
    by_name = dict(zip(g["name"].map(fold), g["pcode"]))
    if len(by_name) != EXPECTED:
        raise SystemExit("COD's department names are not unique — the name join is unsafe")

    # ---- witness 1: every LAPOP label is a COD name ----
    missing = [n for n in LAPOP_DEPARTMENTS.values() if fold(n) not in by_name]
    if missing:
        raise SystemExit(f"LAPOP labels with no polygon: {missing} — the name join FAILED")
    print(f"  witness 1 — all {len(LAPOP_DEPARTMENTS)} LAPOP labels match a COD name, no alias")

    # ---- witness 2: the code join agrees with the name join on every one ----
    wrong = [(c, n, f"CO{c - 800:02d}", by_name[fold(n)]) for c, n in LAPOP_DEPARTMENTS.items()
             if f"CO{c - 800:02d}" != by_name[fold(n)]]
    if wrong:
        for w in wrong:
            print(f"      LAPOP {w[0]} {w[1]} -> {w[2]}, but the name says {w[3]}")
        raise SystemExit("Colombia's code and name joins have stopped agreeing. A permutation "
                         "preserves every total, so nothing downstream would catch it. STOP.")
    print(f"  witness 2 — `prov - 800` -> COnn pairs all {len(LAPOP_DEPARTMENTS)} the way the "
          "names do")

    spare = sorted(set(g["pcode"]) - {f"CO{c - 800:02d}" for c in LAPOP_DEPARTMENTS})
    if spare != sorted(NEVER_SAMPLED):
        raise SystemExit(f"polygons with no LAPOP code: {spare}, expected {sorted(NEVER_SAMPLED)}")
    print(f"    the {len(spare)} departments LAPOP has no code for: "
          f"{', '.join(g.set_index('pcode').loc[spare, 'name'])}")

    # ---- witness 3: COD-PS 2025 on the pcode, and its names agree with COD-AB's ----
    pop = pd.read_csv(os.path.join(RAW, "col_admpop_adm1_2025.csv"), encoding="utf-8-sig")
    pop["pcode"] = pop["ADM1_PCODE"].astype(str).str.strip()
    if len(pop) != EXPECTED or set(pop["pcode"]) != set(g["pcode"]):
        raise SystemExit("COD-PS ADM1 does not cover the same 33 pcodes as COD-AB")
    if int(pop["T_TL"].sum()) != CODPS_TOTAL:
        raise SystemExit(f"COD-PS sums to {int(pop['T_TL'].sum()):,}, not {CODPS_TOTAL:,}")
    ab_name = dict(zip(g["pcode"], g["name"].map(fold)))
    differ = [(p, n) for p, n in zip(pop["pcode"], pop["ADM1_ES"]) if fold(n) != ab_name[p]]
    print(f"  witness 3 — COD-PS 2025 joins on the pcode, {CODPS_TOTAL:,} people; names differ "
          f"from COD-AB's on {len(differ)}: {differ}")
    g = g.merge(pop[["pcode", "T_TL"]], on="pcode", how="left")
    g["pop"] = g["T_TL"].astype("int64")

    g["area_km2"] = g.to_crs(6933).geometry.area / 1e6
    g["density"] = g["pop"] / g["area_km2"]
    hi = g.loc[g["density"].idxmax(), "name"]
    big = g.loc[g["pop"].idxmax(), "name"]
    print(f"    densest {hi!r}, largest {big!r}, sparsest "
          f"{g.loc[g['density'].idxmin(), 'name']!r}")
    if fold(hi) != fold("Bogotá, D.C.") or fold(big) != fold("Bogotá, D.C."):
        raise SystemExit("Bogotá is not the densest and largest unit — the population join is "
                         "permuted")

    g["unit"] = g["pcode"]
    g["geo_id"] = g["pcode"]
    os.makedirs(OUT_DIR, exist_ok=True)
    g[["unit", "name", "pcode", "geo_id", "pop", "geometry"]].to_file(
        OUT, layer="departamentos", driver="GPKG")
    print(f"\nwrote {OUT} ({len(g)} polygons)")

    prov_by_pcode = {f"CO{c - 800:02d}": c for c in LAPOP_DEPARTMENTS}
    lut = pd.DataFrame({"geo_id": sorted(g["pcode"])})
    lut["unit"] = lut["geo_id"]
    lut["name"] = lut["geo_id"].map(dict(zip(g["pcode"], g["name"])))
    lut["lapop_prov"] = lut["geo_id"].map(prov_by_pcode).astype("Int64")
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows, {int(lut['lapop_prov'].notna().sum())} with a "
          "LAPOP prov code)")

    bands = [c for c in pop.columns if re.fullmatch(r"T_(\d+_\d+|\d+Plus)", c)]
    off = (pop[bands].sum(axis=1) - pop["T_TL"]).abs().max()
    if off > len(bands):
        raise SystemExit(f"COD-PS age bands do not sum to T_TL (worst {off})")
    po = pop.rename(columns={"pcode": "geo_id", "ADM1_ES": "name", "T_TL": "pop"})
    po[["geo_id", "name", "pop"] + bands].sort_values("geo_id").to_csv(
        POP_OUT, index=False, encoding="utf-8")
    print(f"wrote {POP_OUT} ({len(po)} departments, {len(bands)} age bands)")


if __name__ == "__main__":
    main()
