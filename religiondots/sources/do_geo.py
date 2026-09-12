"""Dominican Republic — boundaries and populations for the 32 provinces.

Writes data/geo/do/do_provincias.gpkg and data/geo/do/do_lookup.csv.

OCHA COD-AB Dominican Republic (`cod-ab-dom`), the **shapefile** bundle rather than the
geodatabase on §12's Chile rule, read with `engine="fiona"`.

## THE PROVINCES ARE COD'S ADM2, NOT ITS ADM1

Every other Latin American country on this map draws its first-order units off COD's ADM1.
**The Dominican Republic's ADM1 is the ten planning regions** — `DO01 Región Cibao
Nordeste`, `DO02 Región Cibao Noroeste`, and so on — and the 32 provinces are ADM2. An agent
copying `sources/sv_geo.py` and reading `dom_admin1.shp` gets ten polygons and no error, and
ten units for 10.8 million people is a country drawn three times coarser than its own survey
measured it. The layer is asserted at 32 features below.

## THE POPULATION IS THE 2022 CENSUS, AND COD-PS AGREES NATIONALLY WHILE BEING WRONG

`cod-ps-dom` publishes a 2023 projection at ADM2, which HDX's own methodology field describes
as a projection from a census that does not exist ("2015"); the Dominican Republic counted in
2010 and again in November 2022. Nationally it is excellent, **0.56% under ONE's census**,
which is better than the 3.4% error that made `sources/ec_geo.py` reject Ecuador's. Per
province it is not:

    San José de Ocoa   census  69,082   COD-PS  52,687   -23.7%
    La Altagracia      census 446,060   COD-PS 375,872   -15.7%
    Hato Mayor         census 100,133   COD-PS  85,738   -14.4%
    ...
    Distrito Nacional  census 1,029,110 COD-PS 1,062,476  +3.2%
    Santo Domingo      census 2,769,588 COD-PS 3,054,470 +10.3%

**A 34-point spread under a national agreement of half a point.** The projection carried
forward the drift into the capital that the 2010 census was seeing, and the 2022 count did not
find it: drawing on COD-PS would put 285,000 people in Santo Domingo province who are not
there and take a quarter of San José de Ocoa away. So the base is **ONE's own X Censo
Nacional de Población y Vivienda 2022, 10,771,504 people**, and COD-PS is read only to print
the comparison above. The generalisation is in sources.md §9cf: a national-level agreement
between COD-PS and a census says nothing about the subnational split, and it is exactly the
countries where COD-PS looks safe that nobody checks.

## THE JOIN IS ON THE NAME, AND THE CODE JOIN IS NOT AVAILABLE AT ALL

ENHOGAR numbers the provinces 1 to 32 in ONE's official order (Distrito Nacional first, then
the other thirty-one alphabetically), which is the same order the 2022 census microdata uses.
**COD's ADM2 pcodes are ordered by region and then by province inside it**, so `DO0101` is
Duarte and `DO0801` is the Distrito Nacional. Pairing the sorted pcodes with 1..32 gets
**three of the thirty-two right**, all by coincidence, which is the safe kind of wrong:
unlike El Salvador's two of fourteen (§9bl) and Uruguay's nine of nineteen (§9ce), the three
that coincide are scattered and no spot check would survive. It is asserted anyway, because a
re-cut that made it *mostly* right is the dangerous direction.

One alias is needed and only one: ENHOGAR writes **Bahoruco** where COD-AB and the census
both write **Baoruco**.

Usage:
    python sources/do_geo.py --fetch    a ~46 MB zip from HDX, a 33 KB xlsx, a 4 KB csv
    python sources/do_geo.py            rebuild from data/raw/do/
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
RAW = os.path.join(ROOT, "data", "raw", "do")
SHP_DIR = os.path.join(RAW, "shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "do")
OUT = os.path.join(OUT_DIR, "do_provincias.gpkg")
LOOKUP = os.path.join(OUT_DIR, "do_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

N_PROVINCES = 32

# ONE's own census total, printed on row 7 of the workbook and asserted against the sum of
# the 32 province rows below.
CENSUS_TOTAL = 10_771_504

DOWNLOADS = {
    # OCHA COD-AB, https://data.humdata.org/dataset/cod-ab-dom
    "dom_admin_boundaries.shp.zip":
        "https://data.humdata.org/dataset/1b26ea94-4d28-4167-9209-4984ab087973/resource/"
        "fbef96f2-a759-4aec-a33f-3e98e00e0107/download/dom_admin_boundaries.shp.zip",
    # OCHA COD-PS 2023, read for the comparison only.
    "dom_admpop_adm2_2023.csv":
        "https://data.humdata.org/dataset/23eb98d2-e167-4b89-a026-f767cc81487e/resource/"
        "c93ab474-c5ad-4a76-9532-5eaa9f5f5348/download/dom_admpop_adm2_2023.csv",
    # ONE, "Población por sexo según provincia, municipio y distrito municipal", X CNPV 2022.
    # www.one.gob.do sits behind a Cloudflare challenge that answers 403 to curl AND to
    # WebFetch, so this goes through the Wayback Machine's byte-identical copy. The canonical
    # URL is the `if_` target at the end of the path.
    "xcnpv_pob_provincia.xlsx":
        "https://web.archive.org/web/20231225210316id_/https://www.one.gob.do/media/1kijehxj/"
        "poblaci%C3%B3n-por-sexo-seg%C3%BAn-provincia-municipio-y-distrito-municipal-xcnpv.xlsx",
}

# ENHOGAR-MICS6 2019's own value labels for `HH7A`, read out of the .sav's label set. This is
# ONE's official province numbering: Distrito Nacional first, then the thirty-one provinces
# alphabetically. It is the same order the 2022 census microdata's `PROVINCIA` uses, and it
# is NOT COD's pcode order. Written out so a relabelled release fails here.
ENHOGAR_PROVINCES = {
    1: "Distrito Nacional",        2: "Azua",               3: "Bahoruco",
    4: "Barahona",                 5: "Dajabón",            6: "Duarte",
    7: "Elías Piña",               8: "El Seibo",           9: "Espaillat",
    10: "Independencia",          11: "La Altagracia",     12: "La Romana",
    13: "La Vega",                14: "María Trinidad Sánchez", 15: "Monte Cristi",
    16: "Pedernales",             17: "Peravia",           18: "Puerto Plata",
    19: "Hermanas Mirabal",       20: "Samaná",            21: "San Cristóbal",
    22: "San Juan",               23: "San Pedro de Macorís", 24: "Sánchez Ramírez",
    25: "Santiago",               26: "Santiago Rodríguez", 27: "Valverde",
    28: "Monseñor Nouel",         29: "Monte Plata",       30: "Hato Mayor",
    31: "San José de Ocoa",       32: "Santo Domingo",
}

# The one spelling that differs between ENHOGAR and both COD-AB and the census.
ALIAS = {"bahoruco": "baoruco"}

# How many of the thirty-two a positional `sorted(pcode)[i] -> i + 1` join gets right. Three,
# by coincidence, and it should stay at three: a re-cut that made it mostly right is the
# failure mode §9bl and §9ce are about, because a spot check would then start passing.
CODE_JOIN_CORRECT = 3


def fold(s):
    """Accent- and case-insensitive key for a province name."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalnum())


def key(name):
    return ALIAS.get(fold(name), fold(name))


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for name, url in DOWNLOADS.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 1024:
            print(f"  have {name} ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=900) as r, open(dst + ".part", "wb") as f:
            f.write(r.read())
        os.replace(dst + ".part", dst)
        print(f"  got  {name} ({os.path.getsize(dst):,} bytes)")


def read_census():
    """The 32 province totals out of ONE's `Cuadro 4`.

    The workbook nests province, municipio and distrito municipal in ONE column and marks the
    level with leading spaces, so the provinces are the rows that do not start with one.
    **One row of 590 breaks that**: `Presidente Don Antonio Guzmán Fernández (DM)`, a
    municipal district in Duarte, is written flush left like a province. It is excluded on the
    `(DM)` suffix rather than on its name, and the province count and the total both assert.
    """
    import openpyxl

    path = os.path.join(RAW, "xcnpv_pob_provincia.xlsx")
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing — run with --fetch")
    ws = openpyxl.load_workbook(path, data_only=True)["Cuadro 4"]
    rows, total = {}, None
    for row in ws.iter_rows(min_row=7, values_only=True):
        name, val = row[1], row[2]
        if not isinstance(name, str) or val is None:
            continue
        if name.startswith(" ") or name.strip().endswith("(DM)"):
            continue
        nm = name.strip()
        if nm.lower().startswith("total"):
            total = int(val)
            continue
        rows[key(nm)] = (nm, int(val))
    if len(rows) != N_PROVINCES:
        raise SystemExit(f"the census workbook yielded {len(rows)} provinces, not "
                         f"{N_PROVINCES} — the indentation convention has changed")
    if total != CENSUS_TOTAL or sum(v for _, v in rows.values()) != CENSUS_TOTAL:
        raise SystemExit(f"census total {total:,} / rows {sum(v for _, v in rows.values()):,} "
                         f"against the expected {CENSUS_TOTAL:,}")
    print(f"  ONE X CNPV 2022: {N_PROVINCES} provinces summing to {CENSUS_TOTAL:,}, "
          "and the printed total agrees")
    return rows


def check_sav_labels():
    """Assert ENHOGAR_PROVINCES against the .sav's own label set, when it is on disk.

    The public `.sav` reaches us truncated at exactly 1,048,576 bytes — the Wayback Machine's
    capture cap, not a damaged file — and SPSS puts the whole label dictionary in the header,
    so `metadataonly=True` reads it in full while the cases are unreachable. That is why the
    build takes its DATA from the CSV and its LABELS from here.
    """
    sav = os.path.join(RAW, "mics6_2019_hogares.sav")
    if not os.path.exists(sav):
        print("  (mics6_2019_hogares.sav not on disk — province labels not cross-checked; "
              "run sources/do.py --fetch)")
        return
    import pyreadstat

    _, meta = pyreadstat.read_sav(sav, metadataonly=True)
    labs = meta.variable_value_labels.get("HH7A", {})
    got = {int(k): v for k, v in labs.items()}
    if got != ENHOGAR_PROVINCES:
        diff = {k: (got.get(k), ENHOGAR_PROVINCES.get(k))
                for k in set(got) | set(ENHOGAR_PROVINCES)
                if got.get(k) != ENHOGAR_PROVINCES.get(k)}
        raise SystemExit(f"HH7A's value labels have changed: {diff}")
    print(f"  witness 0 — all {len(got)} HH7A labels match the .sav's own label set")


def check_code_join(by_name):
    """Report what a positional pcode->1..32 join would have done, and keep it at zero."""
    order = sorted(by_name.values())
    right = [n for i, n in ENHOGAR_PROVINCES.items()
             if by_name[key(n)] == order[i - 1]]
    print(f"\n  witness 3 — the code join is NOT used. Pairing sorted pcodes with ENHOGAR's "
          f"1..{N_PROVINCES} gets {len(right)} right by coincidence ({', '.join(right)}); "
          "three examples of what it would do to the rest:")
    for i in (1, 8, 32):
        nm = ENHOGAR_PROVINCES[i]
        cod_says = next(n for n, p in by_name.items() if p == order[i - 1])
        print(f"      ENHOGAR {i:2d} {nm:<22} -> {order[i - 1]}, which COD says is "
              f"{cod_says!r} (the real one is {by_name[key(nm)]})")
    if len(right) != CODE_JOIN_CORRECT:
        raise SystemExit(
            f"the positional code join now gets {len(right)} of {N_PROVINCES} right, not "
            f"{CODE_JOIN_CORRECT}. Somebody has re-cut the pcodes into ONE's order — STOP "
            "and decide deliberately. Do not delete this assertion.")


def main():
    if "--fetch" in sys.argv:
        fetch()

    zpath = os.path.join(RAW, "dom_admin_boundaries.shp.zip")
    if not os.path.exists(zpath):
        raise SystemExit(f"{zpath} missing — run with --fetch")
    os.makedirs(SHP_DIR, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(SHP_DIR)

    # dom_admin2.shp: the PROVINCES. dom_admin1.shp is the ten planning regions — see the
    # module docstring. The `_em` variants are the edge-matched cuts and are not used here.
    shp = os.path.join(SHP_DIR, "dom_admin2.shp")
    g = gpd.read_file(shp, engine="fiona")
    if len(g) != N_PROVINCES:
        raise SystemExit(f"{len(g)} ADM2 features, expected {N_PROVINCES} — either COD has "
                         "re-cut the Dominican Republic or this is the wrong layer")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"unexpected CRS {g.crs}")
    print(f"read {shp}: {len(g)} provinces, {g.crs}")

    g["pcode"] = g["adm2_pcode"].astype(str).str.strip()
    # COD prefixes every province but the Distrito Nacional with the word "Provincia".
    g["short"] = g["adm2_name"].str.replace(r"^Provincia\s+", "", regex=True).str.strip()
    by_name = dict(zip(g["short"].map(key), g["pcode"]))
    if len(by_name) != N_PROVINCES:
        raise SystemExit("COD's province names are not unique — the name join is unsafe")

    check_sav_labels()

    # ---- witness 1: every ENHOGAR province name is a COD province name ----
    missing = [n for n in ENHOGAR_PROVINCES.values() if key(n) not in by_name]
    spare = [n for n in g["short"]
             if key(n) not in {key(v) for v in ENHOGAR_PROVINCES.values()}]
    if missing or spare:
        print(f"    ENHOGAR names with no polygon: {missing}")
        print(f"    polygons with no ENHOGAR name: {spare}")
        raise SystemExit("the name join FAILED")
    print(f"  witness 1 — all {N_PROVINCES} ENHOGAR names match a COD name, one alias "
          f"({', '.join(f'{a} -> {b}' for a, b in ALIAS.items())})")

    # ---- witness 2: the census names are the same set again, joined the same way ----
    census = read_census()
    if set(census) != set(by_name):
        raise SystemExit("the census province names and COD's do not fold to the same set")
    g["pop"] = g["short"].map(lambda n: census[key(n)][1]).astype("int64")
    if int(g["pop"].sum()) != CENSUS_TOTAL:
        raise SystemExit("the population join lost or gained people")
    print(f"  witness 2 — the census joins on the same names: {int(g['pop'].sum()):,} people")

    check_code_join(by_name)

    # ---- COD-PS, read for the comparison and not used ----
    ps_path = os.path.join(RAW, "dom_admpop_adm2_2023.csv")
    if os.path.exists(ps_path):
        ps = pd.read_csv(ps_path, encoding="utf-8-sig")
        ps = dict(zip(ps["ADM2_PCODE"].astype(str).str.strip(), ps["T_TL"].astype(int)))
        diff = [(n, p, int(c), ps[p], (ps[p] - c) / c * 100)
                for n, p, c in zip(g["short"], g["pcode"], g["pop"])]
        diff.sort(key=lambda t: t[4])
        nat = (sum(ps.values()) - CENSUS_TOTAL) / CENSUS_TOTAL * 100
        print(f"\n  COD-PS 2023 against the census, NOT USED as a base "
              f"(national {nat:+.2f}%, per province {diff[0][4]:+.1f}% to {diff[-1][4]:+.1f}%):")
        for n, p, c, v, d in diff[:3] + diff[-3:]:
            print(f"      {n:<24} census {c:>9,}   COD-PS {v:>9,}   {d:+6.1f}%")

    # Santo Domingo province rings the capital and is the densest; Pedernales, the empty
    # south-western corner against the Haitian border, is the sparsest. A permuted population
    # join is what this catches.
    g["density"] = g["pop"] / g["area_sqkm"]
    lo = g.loc[g["density"].idxmin(), "short"]
    hi = g.loc[g["density"].idxmax(), "short"]
    print(f"\n  sparsest {lo!r}, densest {hi!r}")
    if key(hi) != key("Distrito Nacional"):
        raise SystemExit("the Distrito Nacional is not the densest province — the population "
                         "join is permuted")

    g["unit"] = g["pcode"]
    # geo_id IS the pcode, deliberately: ENHOGAR's 1..32 never leaves sources/do.py, so there
    # is only one province numbering anywhere under data/.
    g["geo_id"] = g["pcode"]
    g["name"] = g["short"]

    os.makedirs(OUT_DIR, exist_ok=True)
    out = g[["unit", "name", "pcode", "geo_id", "pop", "geometry"]]
    out.to_file(OUT, layer="provincias", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    order = sorted(g["pcode"])
    lut = pd.DataFrame({
        "geo_id": order,
        "unit": order,
        "name": [g.loc[g["pcode"] == p, "name"].iloc[0] for p in order],
        "region": [g.loc[g["pcode"] == p, "adm1_name"].iloc[0] for p in order],
        "pop_2022": [int(g.loc[g["pcode"] == p, "pop"].iloc[0]) for p in order],
        "enhogar_hh7a": [next(i for i, n in ENHOGAR_PROVINCES.items()
                              if by_name[key(n)] == p) for p in order],
    })
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows, with ENHOGAR's HH7A code alongside)")


if __name__ == "__main__":
    main()
