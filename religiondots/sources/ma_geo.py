"""Morocco: 73 units (69 provinces and prefectures, 4 Western Sahara units) with RGPH 2024 people.

Writes data/geo/ma/ma_units.gpkg, data/geo/ma/ma_lookup.csv and data/geo/ma/ma_pieces.csv.
`sources/ma.md` §5 is the record in prose.

Two sources:

  * **people**: HCP, *Population légale du Royaume du Maroc répartie par régions, provinces et
    préfectures et communes*, RGPH 2024 (`hcp.ma/file/242341/`, Excel). Every row gives
    Moroccans, foreigners, population and households, for the nation, 12 regions, 75 provinces
    and prefectures and every commune, and again for the urban and the rural part of each region
    and province. 36,828,330 people, 36,680,178 Moroccans and 148,152 foreigners.
  * **polygons**: COD-AB `cod-ab-mar` v01 ADM2 (69 features, HCP lineage) and COD-AB `cod-ab-esh`
    v01 ADM1 (4 features, GADM lineage). The Morocco file stops at 27°40'N and the Western Sahara
    file starts there.

## WHY 73 UNITS AND NOT 75

`cod-ab-mar` has 69 provinces because it leaves out the six HCP provinces with land south of
27°40'N, and **one of them, Tarfaya, is mostly north of it**: GeoNames' Tarfaya (27.94°N) and
Akhfennir (28.09°N) fall inside neither file. `cod-ab-esh` has Morocco's four pre-2009 southern
provinces, not the six of 2024. So:

  * Laâyoune and Tarfaya are one unit: ESH `Laayoune`, plus the Kontur hexes of the strip between
    COD's Tan-Tan and 27°40'N (assigned in `sources/ma_grid.py`). Tarfaya's own communes Daoura and
    El Hagounia are inside ESH `Laayoune` by GeoNames.
  * Oued Ed-Dahab and Aousserd are one unit, ESH `Oued el Dahab`; ESH has no Aousserd line.
  * The Assa-Zag commune of Al Mahbass (19,139 people in 170 households, the berm) is inside ESH
    `Es Semara` by GeoNames, so it is moved there from COD's Assa Zag, commune row and all.

## WESTERN SAHARA

Drawn inside Morocco on spec §14.18's rule, de facto administration: Natural Earth's disputed-areas
layer records the part west of the berm as `Admin. by Morocco; Claimed by Western Sahara` (B19) and
the part east of it as `Self admin.; Claimed by Morocco` (B28). The ESH units are cut to B19 here.
Anita has not ruled on Western Sahara itself: `ask/031`.

Ceuta (B60) and Melilla (B61) are Spanish and are cut out of COD's Fnideq and Nador polygons, which
overlap them by 7.1 and 0.9 km2.

Usage:
    python sources/ma_geo.py --fetch    COD-AB Morocco and Western Sahara zips; the HCP workbook
    python sources/ma_geo.py            rebuild from data/raw/ma/
"""

import io
import json
import os
import re
import ssl
import sys
import unicodedata
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ma")
OUT_DIR = os.path.join(ROOT, "data", "geo", "ma")
OUT = os.path.join(OUT_DIR, "ma_units.gpkg")
LOOKUP = os.path.join(OUT_DIR, "ma_lookup.csv")
PIECES = os.path.join(OUT_DIR, "ma_pieces.csv")
DISPUTED = os.path.join(ROOT, "data", "geo", "ne_10m_admin_0_disputed_areas.geojson")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
FILES = {
    "mar_admin_boundaries.geojson.zip": (
        "https://data.humdata.org/dataset/7fb8f891-35f2-4bd5-8c75-5e7797ff1731/resource/"
        "4df64f29-003c-401e-bdb0-40c5f96a568b/download/mar_admin_boundaries.geojson.zip", b"PK"),
    "esh_admin_boundaries.geojson.zip": (
        "https://data.humdata.org/dataset/ab6e70bd-3cab-4a68-9a02-04ecd3df6b46/resource/"
        "8f17d035-5cae-4b7b-9a49-4acdcb594570/download/esh_admin_boundaries.geojson.zip", b"PK"),
    "hcp_population_legale_2024.xlsx": ("https://www.hcp.ma/file/242341/", b"PK"),
}
WORKBOOK = os.path.join(RAW, "hcp_population_legale_2024.xlsx")

NATIONAL = (36_680_178, 148_152, 36_828_330)      # Moroccans, foreigners, population
N_REGIONS, N_PROVINCES = 12, 75

# COD-AB Morocco names that do not fold to HCP's spelling of the same province.
ALIASES = {"fquihbensaleh": "fquihbensalah", "elkelaatessraghna": "elkelaadessraghna",
           "rhamna": "rehamna", "mohammedia": "mohammadia", "tangierassilah": "tangerassilah",
           "taroudant": "taroudannt"}

# ESH unit -> the HCP pieces it holds: a province code, or ("commune", code).
ESH_UNITS = {
    "Laayoune": ("Laâyoune and Tarfaya", [11321, 11537]),
    "Boujdour": ("Boujdour", [11121]),
    "Es Semara": ("Es-Semara", [11221, ("commune", 100710501)]),
    "Oued el Dahab": ("Oued Ed-Dahab and Aousserd", [12391, 12066]),
}
MAHBASS = 100710501                     # Commune d'Al Mahbass, province of Assa-Zag (10071)
ASSA_ZAG = 10071
ASSA_ZAG_URBAN = (100710101, 100710103)  # Assa and Zag, the province's two municipalities


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"^(prefecture d arrondissements de|prefecture of|prefecture de|prefecture d|"
               r"province de|province d|province)\b", "", s.replace("'", " ").strip())
    s = re.sub(r"\bprovince$|\bprefecture$", "", s.strip())
    return re.sub(r"[^a-z]", "", s)


def fetch():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE      # hcp.ma's chain does not verify from here (2026-09-15)
    os.makedirs(RAW, exist_ok=True)
    for name, (url, magic) in FILES.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 50_000:
            print(f"  have {name} ({os.path.getsize(dst):,} bytes)")
            continue
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600,
                                    context=ctx) as r:
            data = r.read()
        if not data.startswith(magic):
            raise SystemExit(f"{url} did not return the expected file; starts {data[:24]!r}")
        with open(dst + ".part", "wb") as fh:
            fh.write(data)
        os.replace(dst + ".part", dst)
        print(f"  got  {name} ({len(data):,} bytes)")


def read_workbook():
    """Regions and provinces as {(milieu, code): row}, and every commune row as {code: row}."""
    import openpyxl
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wb = openpyxl.load_workbook(WORKBOOK, read_only=True, data_only=True)
    rows = [list(r) + [None] * (7 - len(r)) for r in wb.worksheets[0].iter_rows(values_only=True)]
    admin, communes, notes = {}, {}, []
    for r in rows:
        name, mor, fo, pop, _hh, _ar, code = r[:7]
        if not isinstance(name, str):
            continue
        name = name.strip()
        if not isinstance(pop, (int, float)):
            if "*" in name:
                notes.append(name)
            continue
        if mor + fo != pop:
            raise SystemExit(f"workbook row {name!r}: {mor:,} + {fo:,} != {pop:,}")
        milieu = ("urban" if "(milieu urbain)" in name else
                  "rural" if "(milieu rural)" in name else "total")
        level = name.split(" ")[0]
        if name.startswith("Ensemble du territoire national"):
            key = (milieu, 0)
        elif level in ("Région", "Province", "Préfecture") and not name.startswith(
                "Préfecture d'arrondissements"):
            key = (milieu, int(code))
        elif level in ("Commune", "Arrondissement") and code:
            communes.setdefault(int(code), (name, int(mor), int(fo), int(pop)))
            continue
        else:
            continue
        val = (name.replace(" (milieu urbain)", "").replace(" (milieu rural)", ""),
               int(mor), int(fo), int(pop))
        if key in admin and admin[key][1:] != val[1:]:
            raise SystemExit(f"workbook prints {name!r} twice with different figures")
        admin[key] = val
    if notes:
        print("  workbook footnote rows: " + " | ".join(notes[:4]))

    nat = admin[("total", 0)][1:]
    if nat != NATIONAL:
        raise SystemExit(f"the national row reads {nat}, not {NATIONAL}")
    regions = sorted(c for (m, c) in admin if m == "total" and 1 <= c <= N_REGIONS)
    # A province is a code the workbook prints with its urban and its rural part. The commune
    # section repeats some prefecture headers under codes of their own; those are not provinces.
    totals = sorted(c for (m, c) in admin if m == "total" and c > 100)
    provinces = [c for c in totals if ("urban", c) in admin and ("rural", c) in admin]
    extra = [f"{c} {admin[('total', c)][0]} ({admin[('total', c)][3]:,})"
             for c in totals if c not in provinces]
    if extra:
        print(f"  total rows with no urban and rural part, not provinces: {extra}")
    if len(regions) != N_REGIONS or len(provinces) != N_PROVINCES:
        raise SystemExit(f"{len(regions)} regions and {len(provinces)} provinces, expected "
                         f"{N_REGIONS} and {N_PROVINCES}")
    for m in ("total", "urban", "rural"):
        for i in (1, 2):
            s = sum(admin[(m, c)][i] for c in provinces)
            t = sum(admin[(m, c)][i] for c in regions)
            if s != t or (m == "total" and s != NATIONAL[i - 1]):
                raise SystemExit(f"{m} column {i}: provinces {s:,}, regions {t:,}")
    for c in provinces:
        for i in (1, 2):
            if admin[("urban", c)][i] + admin[("rural", c)][i] != admin[("total", c)][i]:
                raise SystemExit(f"{admin[('total', c)][0]}: urban + rural != total")
    region_of = {}
    for c in provinces:
        reg = [r for r in regions if str(c).startswith(str(r)) and len(str(c)) - len(str(r)) == 3]
        if len(reg) != 1:
            raise SystemExit(f"province code {c} does not name one region: {reg}")
        region_of[c] = reg[0]
    for r in regions:
        if sum(admin[("total", c)][3] for c in provinces if region_of[c] == r) != admin[("total", r)][3]:
            raise SystemExit(f"region {r}'s provinces do not add to it")
    print(f"  HCP RGPH 2024: {len(regions)} regions, {len(provinces)} provinces, "
          f"{NATIONAL[2]:,} people; provinces add to regions and urban + rural to totals, "
          "for Moroccans and foreigners separately")
    return admin, communes, provinces, region_of


def pieces(admin, communes, provinces, region_of):
    """One row per piece of people a unit can hold: 74 provinces and Al Mahbass apart."""
    name, mor, fo, pop = communes[MAHBASS]
    if not name.endswith("Al Mahbass") or not str(MAHBASS).startswith(str(ASSA_ZAG)):
        raise SystemExit(f"commune {MAHBASS} is {name!r}, not Assa-Zag's Al Mahbass")
    urban = [communes[c] for c in ASSA_ZAG_URBAN]
    if (sum(u[1] for u in urban) != admin[("urban", ASSA_ZAG)][1]
            or sum(u[2] for u in urban) != admin[("urban", ASSA_ZAG)][2]):
        raise SystemExit("Assa and Zag are not Assa-Zag's whole urban population, so Al Mahbass "
                         "cannot be taken as rural")
    out = []
    for c in provinces:
        t, u, r = admin[("total", c)], admin[("urban", c)], admin[("rural", c)]
        row = dict(piece=c, piece_name=t[0], region=region_of[c], mor_urban=u[1], mor_rural=r[1],
                   for_urban=u[2], for_rural=r[2], pop=t[3])
        if c == ASSA_ZAG:
            row.update(mor_rural=r[1] - mor, for_rural=r[2] - fo, pop=t[3] - pop)
        out.append(row)
    out.append(dict(piece=MAHBASS, piece_name=name, region=region_of[ASSA_ZAG], mor_urban=0,
                    mor_rural=mor, for_urban=0, for_rural=fo, pop=pop))
    p = pd.DataFrame(out)
    if int(p["pop"].sum()) != NATIONAL[2]:
        raise SystemExit("the pieces do not add to the nation")
    return p


def main():
    if "--fetch" in sys.argv:
        fetch()
    for n in FILES:
        if not os.path.exists(os.path.join(RAW, n)):
            raise SystemExit(f"{n} missing; run with --fetch")

    admin, communes, provinces, region_of = read_workbook()
    pc = pieces(admin, communes, provinces, region_of)

    with zipfile.ZipFile(os.path.join(RAW, "mar_admin_boundaries.geojson.zip")) as zf:
        mar = gpd.read_file(io.BytesIO(zf.read("mar_admin2.geojson")))
    with zipfile.ZipFile(os.path.join(RAW, "esh_admin_boundaries.geojson.zip")) as zf:
        esh = gpd.read_file(io.BytesIO(zf.read("esh_admin1.geojson")))
    if len(mar) != 69 or len(esh) != 4:
        raise SystemExit(f"COD-AB: {len(mar)} Morocco ADM2 and {len(esh)} Western Sahara ADM1 "
                         "features, expected 69 and 4")

    # ---- the join: COD Morocco's 69 provinces to HCP's by folded name, a bijection ----
    esh_codes = {p if isinstance(p, int) else None for _n, ps in ESH_UNITS.values() for p in ps}
    hcp_north = {fold(admin[("total", c)][0]): c for c in provinces if c not in esh_codes}
    cod = {ALIASES.get(fold(n), fold(n)): pcode for n, pcode in zip(mar["adm2_name"], mar["adm2_pcode"])}
    if len(hcp_north) != 69 or len(cod) != 69 or set(cod) != set(hcp_north):
        raise SystemExit(f"COD Morocco and HCP do not pair 1:1. COD only: "
                         f"{sorted(set(cod) - set(hcp_north))}; HCP only: "
                         f"{sorted(set(hcp_north) - set(cod))}")
    unit_of = {hcp_north[k]: pcode for k, pcode in cod.items()}
    print("  COD-AB Morocco's 69 provinces pair 1:1 with HCP's 69 north of 27°40'N by name "
          f"({len(ALIASES)} spellings aliased)")

    # ---- Western Sahara: cut to Natural Earth's B19, Ceuta and Melilla out of Morocco ----
    with open(DISPUTED, encoding="utf-8") as fh:
        feats = json.load(fh)["features"]
    brk = {f["properties"]["BRK_A3"]: shape(f["geometry"]) for f in feats
           if f["properties"]["BRK_A3"] in ("B19", "B28", "B60", "B61")}
    if set(brk) != {"B19", "B28", "B60", "B61"}:
        raise SystemExit(f"Natural Earth disputed areas: found {sorted(brk)}")
    if "Admin. by Morocco" not in next(f["properties"]["NOTE_BRK"] for f in feats
                                       if f["properties"]["BRK_A3"] == "B19"):
        raise SystemExit("Natural Earth's B19 no longer says it is administered by Morocco")
    spanish = gpd.GeoSeries([brk["B60"], brk["B61"]], crs=4326).union_all()
    mar["geometry"] = mar.geometry.difference(spanish)

    rows = []
    for _i, r in mar.iterrows():
        code = next(c for c, pcode in unit_of.items() if pcode == r["adm2_pcode"])
        members = [MAHBASS if False else code]
        rows.append(dict(geo_id=r["adm2_pcode"], name=re.sub(r"^(Province|Préfecture) (de |d')", "",
                                                             admin[("total", code)][0]),
                         members=members, west_sahara=False, geometry=r.geometry))
    b19 = brk["B19"]
    for _i, r in esh.iterrows():
        nm, members = ESH_UNITS[r["adm1_name"]]
        kept = r.geometry.intersection(b19)
        rows.append(dict(geo_id=r["adm1_pcode"], name=nm,
                         members=[m if isinstance(m, int) else m[1] for m in members],
                         west_sahara=True, geometry=kept))
    g = gpd.GeoDataFrame(rows, crs=4326)

    # every piece in exactly one unit
    member_of = {}
    for _i, r in g.iterrows():
        for m in r["members"]:
            if m in member_of:
                raise SystemExit(f"piece {m} is in two units")
            member_of[m] = r["geo_id"]
    if set(member_of) != set(pc["piece"]):
        raise SystemExit(f"pieces with no unit: {sorted(set(pc['piece']) - set(member_of))}; "
                         f"units naming unknown pieces: {sorted(set(member_of) - set(pc['piece']))}")
    pc["geo_id"] = pc["piece"].map(member_of)

    lk = pc.groupby("geo_id")[["mor_urban", "mor_rural", "for_urban", "for_rural", "pop"]].sum()
    g = g.set_index("geo_id").join(lk).reset_index()
    g["area_km2"] = g.to_crs("EPSG:6933").geometry.area / 1e6
    if int(g["pop"].sum()) != NATIONAL[2] or len(g) != 73:
        raise SystemExit("the 73 units do not hold the nation")
    ws = g[g["west_sahara"]]
    print(f"  Western Sahara, cut to B19: {len(ws)} units, {int(ws['pop'].sum()):,} people, "
          f"{ws['area_km2'].sum():,.0f} km2 ({brk['B28'].area and 'B28 left out'})")

    os.makedirs(OUT_DIR, exist_ok=True)
    g["unit"] = g["geo_id"]
    g[["unit", "geo_id", "name", "pop", "geometry"]].to_file(OUT, layer="units", driver="GPKG")
    cols = ["geo_id", "unit", "name", "west_sahara", "pop", "mor_urban", "mor_rural",
            "for_urban", "for_rural", "area_km2"]
    g[cols].sort_values("geo_id").to_csv(LOOKUP, index=False, encoding="utf-8")
    pc.to_csv(PIECES, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} (73 polygons), {LOOKUP} and {PIECES} ({len(pc)} pieces)")


if __name__ == "__main__":
    main()
