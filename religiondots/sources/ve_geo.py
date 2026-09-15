"""Venezuela — boundaries for the 25 federal entities, with the 2011 census count.

Writes, under data/geo/ve/:
    ve_estados.gpkg, ve_lookup.csv, ve_pop_2011.csv     the 23 states, Distrito Capital and
                                                        the Dependencias Federales
    ve_municipios.csv     COD-AB's 336 municipality names, the witness for LAPOP's `municipio`

OCHA COD-AB Venezuela v01 (`cod-ab-ven`, INE via OCHA Venezuela, valid from 2021-02-23) for the
polygons. It holds the 25 federal entities and nothing east of the Essequibo line Venezuela
claims: the eastern bound is asserted, so a later edition that adds the Guayana Esequiba
drawing fails here (spec §14.18, de facto administration).

## THE PEOPLE ARE THE 2011 CENSUS

The XIV Censo Nacional de Población y Vivienda (reference date 30 October 2011) is the last
count, and COD-PS for Venezuela is that count (`cod-ps-ven`, reference year 2011, "UNFPA
investigated replacement or projection of this dataset in 2021. No better alternative is
currently available."). **Every figure is asserted against INE's own table**, Cuadro 2.2 of
*Resultados Total Nacional* (May 2014, `ine.gob.ve/wp-content/uploads/2024/09/Censo-Nacional-2011.pdf`,
p. 13), transcribed below and checked by its printed total, 27,227,930. COD-PS leaves out the
Dependencias Federales (2,155 people), which the table gives.

No newer base was taken, deliberately. There has been no count since, and a projection from this
census cannot see the emigration of the late 2010s, so it would draw people who left, unevenly by
state and by an amount nobody has measured per state. The pooled LAPOP rounds are 2010-2016/17,
the census is 2011, and the map says so.

## THE JOINS

COD-AB and COD-PS share INE's entity codes (`VE01` Distrito Capital to `VE25` Dependencias
Federales) and are joined on them, with every name checked beside the code. Two names differ and
are the same entity: COD-PS's `Distrito Federal` is the Distrito Capital's name before the 1999
constitution, and `Vargas` became `La Guaira` in 2019. LAPOP's state labels are joined to these
by name in `sources/ve.py`, wave by wave, and checked there against the municipality names.

Usage:
    python sources/ve_geo.py --fetch    COD-AB (19 MB) and COD-PS from HDX
    python sources/ve_geo.py            rebuild from data/raw/ve/
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
RAW = os.path.join(ROOT, "data", "raw", "ve")
SHP_DIR = os.path.join(RAW, "shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "ve")
UNITS_OUT = os.path.join(OUT_DIR, "ve_estados.gpkg")
LOOKUP = os.path.join(OUT_DIR, "ve_lookup.csv")
POP_OUT = os.path.join(OUT_DIR, "ve_pop_2011.csv")
MUNIS_OUT = os.path.join(OUT_DIR, "ve_municipios.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}
DOWNLOADS = {
    "ven_admin_boundaries.shp.zip":
        "https://data.humdata.org/dataset/5b141d29-534f-4f01-a0bc-41e2f375d925/resource/"
        "94150d22-778b-44db-9996-adad1a2d927d/download/ven_admin_boundaries.shp.zip",
    "ven_admpop_adm1_2011_v2.csv":
        "https://data.humdata.org/dataset/5c74e336-3162-44da-8c52-d4818c31c37b/resource/"
        "4b1de050-b5a5-48d9-aa54-5bc8ca96ef4b/download/ven_admpop_adm1_2011_v2.csv",
}

# INE, Censo 2011, Resultados Total Nacional, Cuadro 2.2 (p. 13), 2011 column, by entity name.
INE_2011 = {
    "Distrito Capital": 1_943_901, "Amazonas": 146_480, "Anzoátegui": 1_469_747,
    "Apure": 459_025, "Aragua": 1_630_308, "Barinas": 816_264, "Bolívar": 1_413_115,
    "Carabobo": 2_245_744, "Cojedes": 323_165, "Delta Amacuro": 165_525, "Falcón": 902_847,
    "Guárico": 747_739, "Lara": 1_774_867, "Mérida": 828_592, "Miranda": 2_675_165,
    "Monagas": 905_443, "Nueva Esparta": 491_610, "Portuguesa": 876_496, "Sucre": 896_291,
    "Táchira": 1_168_908, "Trujillo": 686_367, "Yaracuy": 600_852, "Zulia": 3_704_404,
    "Vargas": 352_920, "Dependencias Federales": 2_155,
}
INE_TOTAL = 27_227_930

# Old name -> COD-AB's current name for the same INE code
RENAMED = {"distritofederal": "distritocapital", "vargas": "laguaira"}
EAST_BOUND = -59.7    # Venezuela's own eastern border; the Guayana Esequiba lies east of it


def fold(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    f = "".join(ch for ch in s.lower() if ch.isalnum())
    return RENAMED.get(f, f)


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


def layer(level, geometry=True):
    return gpd.read_file(os.path.join(SHP_DIR, f"ven_admin{level}.shp"), engine="pyogrio",
                         ignore_geometry=not geometry)


def main():
    if "--fetch" in sys.argv:
        fetch()
    zpath = os.path.join(RAW, "ven_admin_boundaries.shp.zip")
    for p in (zpath, os.path.join(RAW, "ven_admpop_adm1_2011_v2.csv")):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing — run with --fetch")
    os.makedirs(SHP_DIR, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        members = [n for n in z.namelist() if n.startswith(("ven_admin1.", "ven_admin2."))]
        if len(members) != 10:
            raise SystemExit(f"{len(members)} admin1-2 members in the COD-AB zip — it has been re-cut")
        for n in members:
            z.extract(n, SHP_DIR)

    # ---------------------------------------------------------------- the census table itself
    if sum(INE_2011.values()) != INE_TOTAL:
        raise SystemExit(f"the Cuadro 2.2 transcription sums to {sum(INE_2011.values()):,}, "
                         f"not the printed {INE_TOTAL:,}")
    ine = {fold(k): v for k, v in INE_2011.items()}

    # ---------------------------------------------------------------- entities
    g = layer(1)
    if len(g) != 25:
        raise SystemExit(f"{len(g)} ADM1 features, expected 25")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"unexpected CRS {g.crs}")
    g["pcode"] = g["adm1_pcode"].astype(str).str.strip()
    g["name"] = g["adm1_name"].astype(str).str.strip()
    if sorted(g["pcode"]) != [f"VE{i:02d}" for i in range(1, 26)]:
        raise SystemExit(f"ADM1 pcodes {sorted(g['pcode'])}")
    if g.geometry.isna().any() or g.geometry.is_empty.any():
        raise SystemExit("an entity with empty geometry")
    east = float(g.total_bounds[2])
    if east > EAST_BOUND:
        raise SystemExit(f"the entities reach {east:.3f} E; the Guayana Esequiba is in the file, "
                         "and it is administered by Guyana (spec §14.18)")
    if set(g["name"].map(fold)) != set(ine):
        raise SystemExit(f"COD-AB names and Cuadro 2.2 differ: "
                         f"{sorted(set(g['name'].map(fold)) ^ set(ine))}")
    g["pop"] = g["name"].map(lambda n: ine[fold(n)])

    cp = pd.read_csv(os.path.join(RAW, "ven_admpop_adm1_2011_v2.csv"), encoding="utf-8-sig")
    if len(cp) != 24:
        raise SystemExit(f"COD-PS adm1 has {len(cp)} rows, expected 24")
    ab = dict(zip(g["pcode"], g["name"]))
    bad = [(p, n, ab.get(p)) for p, n in zip(cp["ADM1_PCODE"], cp["ADM1_ES"])
           if fold(n) != fold(ab.get(p, ""))]
    if bad:
        raise SystemExit(f"COD-PS and COD-AB names disagree at the same code: {bad}")
    off = [(n, int(v), ine[fold(n)]) for n, v in zip(cp["ADM1_ES"], cp["T_TL"])
           if int(v) != ine[fold(n)]]
    if off:
        raise SystemExit(f"COD-PS 2011 is not INE's census count for: {off}")
    missing = sorted(set(g["pcode"]) - set(cp["ADM1_PCODE"]))
    print(f"read ADM1: 25 entities, eastern bound {east:.3f}. COD-PS 2011 joins on INE's code and "
          f"equals Cuadro 2.2 for all {len(cp)} of its rows; it has no row for "
          f"{', '.join(ab[p] for p in missing)} ({sum(ine[fold(ab[p])] for p in missing):,} "
          f"people in the table). {INE_TOTAL:,} people.")

    g["area_km2"] = g.to_crs(6933).geometry.area / 1e6
    g["density"] = g["pop"] / g["area_km2"]
    big, dense = g.loc[g["pop"].idxmax(), "name"], g.loc[g["density"].idxmax(), "name"]
    print(f"    largest {big}, densest {dense}, sparsest {g.loc[g['density'].idxmin(), 'name']}; "
          f"area {g['area_km2'].sum():,.0f} km2 against INE's 916,445 (lakes included)")
    if (fold(big), fold(dense)) != ("zulia", "distritocapital"):
        raise SystemExit("Zulia is not the largest or the Distrito Capital not the densest — the "
                         "population join is permuted")
    g["unit"] = g["pcode"]
    g["geo_id"] = g["pcode"]
    os.makedirs(OUT_DIR, exist_ok=True)
    g[["unit", "name", "pcode", "geo_id", "pop", "geometry"]].to_file(
        UNITS_OUT, layer="estados", driver="GPKG")
    lut = g[["geo_id", "unit", "name"]].sort_values("geo_id")
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    g[["geo_id", "name", "pop"]].sort_values("geo_id").to_csv(POP_OUT, index=False,
                                                              encoding="utf-8")
    print(f"wrote {UNITS_OUT}, {LOOKUP}, {POP_OUT}")

    # ---------------------------------------------------------------- municipalities
    m2 = layer(2, geometry=False)
    if len(m2) != 336:
        raise SystemExit(f"{len(m2)} ADM2 features, expected 336")
    if (m2["adm2_pcode"].str[:4] != m2["adm1_pcode"]).any():
        raise SystemExit("ADM2 pcodes do not start with their ADM1 pcode")
    m2[["adm2_name", "adm2_pcode", "adm1_name", "adm1_pcode"]].to_csv(
        MUNIS_OUT, index=False, encoding="utf-8")
    print(f"wrote {MUNIS_OUT} (336 municipalities)")


if __name__ == "__main__":
    main()
