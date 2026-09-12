"""Uruguay — boundaries for the 19 departamentos, and the population that goes on them.

Writes data/geo/uy/uy_departamentos.gpkg, data/geo/uy/uy_lookup.csv and
data/geo/uy/uy_pop_2023.csv.

## THE CODE JOIN IS AVAILABLE, GETS NINE OF NINETEEN RIGHT, AND IS A PERMUTATION

This is `sources/sv_geo.py`'s trap in a country where it is harder to see, because here the
naive join is *mostly* right.

**INE numbers Uruguay's departments with Montevideo first and the other eighteen
alphabetically after it** — 1 Montevideo, 2 Artigas, 3 Canelones, … 19 Treinta y Tres. That
numbering is INE's own `dpto` variable in the ENHA microdata and it is also LAPOP's, whose
`prov` for Uruguay is 1400 plus the same number. **COD-AB's `UY01`..`UY19` are alphabetical
with no Montevideo exception**: UY01 is Artigas and Montevideo is UY10.

So `UY{dpto:02d}` shifts the first ten departments by one place and leaves the last nine
alone:

    INE  1 Montevideo    -> UY01, which COD says is Artigas
    INE  2 Artigas       -> UY02, which COD says is Canelones
    INE  9 Lavalleja     -> UY09, which COD says is Maldonado
    INE 10 Maldonado     -> UY10, which COD says is Montevideo
    INE 11 Paysandú      -> UY11, which is Paysandú           (correct, and so are 12-19)

**Nine of nineteen coincide, and they are the last nine**, which is exactly the
shape that survives a spot check: look at Salto, Soriano and Treinta y Tres and the join
appears fine. The ten that move include Montevideo, 1.29 million people and 37% of the
country, which would have been drawn in Artigas.

A permutation preserves every total, so no reconciliation, no national figure and no row
count would show it (`sources/ni_geo.py`). **The join is therefore on NAME**, and
`check_code_join()` asserts that the code join still mispairs exactly ten, so that an OCHA
re-cut to INE's order stops the build rather than silently changing which department is which.

## THE POPULATION IS INE'S OWN ESTIMATE FOR 2023, THE CENSUS YEAR

Uruguay counted in 2023, so §9bn's rule applies and COD-PS is not the neutral choice here:
its Uruguay file is a projection off the 2011 census. INE's *Estimaciones y proyecciones,
revisión 2025* is the post-census series — `A.1.2 Departamentos.xlsx`, one sheet per
department, both sexes and five-year age bands, 1996 to 2023 — and it is read here for two
years rather than one:

  * **2023**, the magnitude the map is drawn on;
  * **2006**, the ENHA's own year, which is what `sources/uy.py` checks the survey's
    departmental weighting against. Comparing a 2006 survey's unit shares with a 2023
    population would fail on Uruguay's real internal migration rather than on a bad join.

**THE DRAWN UNIVERSE IS AGES 7 AND OVER AND THE BANDS ARE FIVE YEARS WIDE**, so `pop7` is
`total - (0-4) - 0.6 * (5-9)`. Nothing INE publishes for Uruguay is by single year of age
(the 2025 revision's `(100ymas)` files are five-year bands too, the `100 y más` being the
open top band). The interpolation moves at most a few thousand people nationally: the 5-9
band is 6.3% of the country, and being wrong about its internal split by two points in five
is 0.05% of Uruguay. Stated here rather than hidden because it is arithmetic on a published
band and not a published number.

Usage:
    python sources/uy_geo.py --fetch    one ~1 MB zip from HDX, plus a 0.6 MB xlsx from INE
    python sources/uy_geo.py            rebuild from data/raw/uy/
"""

import os
import re
import ssl
import sys
import unicodedata
import urllib.parse
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "uy")
SHP_DIR = os.path.join(RAW, "shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "uy")
OUT = os.path.join(OUT_DIR, "uy_departamentos.gpkg")
LOOKUP = os.path.join(OUT_DIR, "uy_lookup.csv")
POP_OUT = os.path.join(OUT_DIR, "uy_pop_2023.csv")

POP_XLSX = "A.1.2 Departamentos.xlsx"

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"}

DOWNLOADS = {
    # cod-ab-ury on HDX. ADM0..ADM2 plus the boundary lines; 1 MB in total.
    "ury_adm_2020_shp.zip":
        "https://data.humdata.org/dataset/d31c3690-b6d9-461c-8ac6-553dd9c69b81/resource/"
        "75c7c51c-8f05-4b49-8b47-2f94112f986c/download/ury_adm_2020_shp.zip",
    # INE's own post-census estimates, revisión 2025. Linked from
    # gub.uy/instituto-nacional-estadistica/estimaciones2025 and nowhere in the census pages.
    POP_XLSX:
        "https://www5.ine.gub.uy/documents/Demograf%C3%ADayEESS/SERIES%20Y%20OTROS/"
        "Estimaciones%20y%20proyecciones/Revisi%C3%B3n%202025/A.1.2%20Departamentos.xlsx",
}

# www5.ine.gub.uy serves an incomplete certificate chain, the same fault as Ghana's
# StatsBank and Ecuador's censoecuador.gob.ec. Nothing is trusted on the strength of the
# transport: the workbook's own national total is asserted below.
LAX_TLS = {"www5.ine.gub.uy"}

# INE's department numbering, verbatim from the `dpto` value labels in the ENHA 2006 person
# file. Montevideo first, then the other eighteen alphabetically. LAPOP's `prov` for Uruguay
# is 1400 plus these, and its own `prov_es` labels agree name for name, with 1407 (Flores)
# carrying an EMPTY label rather than a wrong one.
INE_DEPARTMENTS = {
    1: "Montevideo",     2: "Artigas",       3: "Canelones",   4: "Cerro Largo",
    5: "Colonia",        6: "Durazno",       7: "Flores",      8: "Florida",
    9: "Lavalleja",     10: "Maldonado",    11: "Paysandú",   12: "Río Negro",
    13: "Rivera",       14: "Rocha",        15: "Salto",      16: "San José",
    17: "Soriano",      18: "Tacuarembó",   19: "Treinta y Tres",
}

# LAPOP's `prov_es` labels for the 1400s, verbatim. Unaccented, and 1407 is blank because
# LAPOP never filled Flores in. Used as an independent witness on the same numbering.
LAPOP_PROVINCES = {
    1401: "Montevideo",  1402: "Artigas",    1403: "Canelones", 1404: "Cerro Largo",
    1405: "Colonia",     1406: "Durazno",    1407: "",          1408: "Florida",
    1409: "Lavalleja",   1410: "Maldonado",  1411: "Paysandu",  1412: "Rio Negro",
    1413: "Rivera",      1414: "Rocha",      1415: "Salto",     1416: "San Jose",
    1417: "Soriano",     1418: "Tacuarembo", 1419: "Treinta y Tres",
}

# How many of the nineteen a naive `UY{dpto:02d}` join happens to get right: the nine from
# Paysandú (11) onwards, plus nothing else. If this ever changes, the pcodes have been re-cut
# and somebody has to decide which numbering is now which.
CODE_JOIN_CORRECT = 9

EXPECTED_DEPARTMENTS = 19

# INE's estimated population at 30 June 2023, from the workbook's own Uruguay sheet. Asserted
# so a re-issued revision fails here instead of quietly moving every dot.
POP_2023 = 3_496_400

# The drawn universe: ENHA 2006 asked the religion question of people OVER 6 (the code-0
# not-applicable group is exactly ages 0 to 6, checked in sources/uy.py). Five-year bands
# cannot cut at 7, so three fifths of the 5-9 band is taken as 7-9.
BAND_5_9_OVER_6 = 0.6


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
        host = urllib.parse.urlsplit(url).hostname
        ctx = None
        if host in LAX_TLS:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=900, context=ctx) as r:
            body = r.read()
        with open(dst + ".part", "wb") as f:
            f.write(body)
        os.replace(dst + ".part", dst)
        # §5a: a 200 is not a download. Both of these are zip containers (.xlsx is one).
        with open(dst, "rb") as f:
            magic = f.read(4)
        if magic[:2] != b"PK":
            raise SystemExit(f"{dst} is not a zip container — starts {magic!r}")
        print(f"  got  {name} ({os.path.getsize(dst):,} bytes)")


def read_pop(years=(2006, 2023)):
    """INE A.1.2: one sheet per department, `Ambos sexos` plus five-year bands, 1996-2023.

    Returns a frame indexed by department name with, for each requested year, `pop_<year>`
    and `pop7_<year>`, plus `T_<lo>_<hi>` / `T_90Plus` columns for the LAST year requested,
    shaped so `lapop._age_bands` could read them.

    The three sexes are stacked in one column, so only the FIRST block is read: `Ambos sexos`
    and the bands under it, stopping at `Hombres`. Reading past that would double the country.
    """
    import openpyxl

    path = os.path.join(RAW, POP_XLSX)
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing — run with --fetch")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)

    sheets = [s for s in wb.sheetnames if s != "Uruguay"]
    if len(sheets) != EXPECTED_DEPARTMENTS:
        raise SystemExit(f"{len(sheets)} department sheets, expected {EXPECTED_DEPARTMENTS}: "
                         f"{sheets}")

    def read_sheet(name):
        ws = wb[name]
        year_col, block, out = None, False, {}
        for row in ws.iter_rows(values_only=True):
            label = "" if row[0] is None else str(row[0]).strip()
            if year_col is None:
                # THE HEADER IS NOT ALWAYS A NUMBER. Eighteen sheets write 2023 as an
                # integer and Paysandú writes it as the string `2023*`, a footnote marker
                # that nothing else in the workbook explains. Parsed rather than typed, so
                # that sheet does not silently lose the year the whole map is drawn on.
                yrs = {}
                for i, v in enumerate(row):
                    m = re.fullmatch(r"\s*(\d{4})\D*", str(v)) if v is not None else None
                    if m and 1900 < int(m.group(1)) < 2100:
                        yrs[int(m.group(1))] = i
                if len(yrs) > 20:
                    year_col = yrs
                continue
            if label in ("Hombres", "Mujeres"):
                break
            if label == "Ambos sexos":
                block = True
                out["total"] = row
                continue
            if not block or not label:
                continue
            m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", label)
            if m:
                out[f"T_{int(m.group(1))}_{int(m.group(2))}"] = row
            elif label.lower().startswith("90"):
                out["T_90Plus"] = row
        if year_col is None or "total" not in out:
            raise SystemExit(f"sheet {name!r} has no year header row or no `Ambos sexos`")
        for y in years:
            if y not in year_col:
                raise SystemExit(f"sheet {name!r} has no column for {y}")
        return year_col, out

    rows = {}
    bands_last = {}
    for name in sheets + ["Uruguay"]:
        year_col, blk = read_sheet(name)
        rec = {}
        for y in years:
            i = year_col[y]
            total = float(blk["total"][i])
            b04 = float(blk["T_0_4"][i])
            b59 = float(blk["T_5_9"][i])
            band_sum = sum(float(v[i]) for k, v in blk.items() if k != "total")
            if abs(band_sum - total) > 1.0:
                raise SystemExit(f"{name} {y}: bands sum to {band_sum:,.0f} against a total "
                                 f"of {total:,.0f} — the sheet is not one block")
            rec[f"pop_{y}"] = total
            rec[f"pop7_{y}"] = total - b04 - BAND_5_9_OVER_6 * b59
        rows[name] = rec
        if name != "Uruguay":
            i = year_col[years[-1]]
            bands_last[name] = {k: float(v[i]) for k, v in blk.items() if k != "total"}
    wb.close()

    pop = pd.DataFrame(rows).T
    nat = pop.loc["Uruguay"]
    pop = pop.drop(index="Uruguay")
    pop = pop.join(pd.DataFrame(bands_last).T)

    for y in years:
        off = pop[f"pop_{y}"].sum() - nat[f"pop_{y}"]
        if abs(off) > 1.0:
            raise SystemExit(f"the {len(pop)} departments sum to {pop[f'pop_{y}'].sum():,.0f} "
                             f"in {y}, against the Uruguay sheet's {nat[f'pop_{y}']:,.0f}")
    if round(nat[f"pop_{years[-1]}"]) != POP_2023:
        raise SystemExit(f"INE's {years[-1]} national total is "
                         f"{nat[f'pop_{years[-1]}']:,.0f}, not the {POP_2023:,} this file "
                         "was written against — the revision has been re-issued")
    print(f"  INE A.1.2: {len(pop)} departments summing EXACTLY to {POP_2023:,} in "
          f"{years[-1]}, and to {nat[f'pop_{years[0]}']:,.0f} in {years[0]}")
    over6 = pop[f"pop7_{years[-1]}"].sum()
    print(f"    aged 7 and over in {years[-1]}: {over6:,.0f}, "
          f"{over6 / POP_2023:.2%} of the country")
    return pop


def check_code_join(by_name):
    """Report what a `UY{dpto:02d}` join would have done, and refuse to let it pass.

    Kept as an assertion rather than a comment because the failure it guards is invisible:
    a permutation preserves every total, and here it preserves nine of the pairings too.
    """
    right, wrong = [], []
    for num, name in sorted(INE_DEPARTMENTS.items()):
        naive = f"UY{num:02d}"
        actual = by_name[fold(name)]
        (right if naive == actual else wrong).append((num, name, naive, actual))
    print(f"\n  witness 2 — the code join is NOT used. `UY{{dpto:02d}}` would pair "
          f"{len(right)} of {EXPECTED_DEPARTMENTS} correctly and MISPAIR {len(wrong)}:")
    for num, name, naive, actual in wrong:
        other = next(n for n, p in by_name.items() if p == naive)
        print(f"      INE {num:>2} {name:<15} -> {naive}, which COD says is {other!r} "
              f"(the name says {actual})")
    if len(right) != CODE_JOIN_CORRECT:
        raise SystemExit(
            f"the code join now gets {len(right)} of {EXPECTED_DEPARTMENTS} right, not "
            f"{CODE_JOIN_CORRECT}. Either OCHA has re-cut the pcodes or INE has renumbered. "
            "STOP and decide which numbering is which before anything is drawn.")


def main():
    if "--fetch" in sys.argv:
        fetch()

    zpath = os.path.join(RAW, "ury_adm_2020_shp.zip")
    if not os.path.exists(zpath):
        raise SystemExit(f"{zpath} missing — run with --fetch")
    os.makedirs(SHP_DIR, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        members = [n for n in z.namelist() if "adm1" in n and not n.endswith(".xml")]
        if not members:
            raise SystemExit("no adm1 members in the COD-AB zip — it has been re-cut")
        for n in members:
            z.extract(n, SHP_DIR)
    print(f"extracted {len(members)} adm1 members")

    shp = os.path.join(SHP_DIR, "ury_admbnda_adm1_2020.shp")
    # `engine="fiona"` on §12's Chile rule: pyogrio is geopandas' default when installed and
    # is the engine that has silently returned zero features. The count is asserted either way.
    g = gpd.read_file(shp, engine="fiona")
    if len(g) != EXPECTED_DEPARTMENTS:
        raise SystemExit(f"{len(g)} ADM1 features, expected {EXPECTED_DEPARTMENTS} — COD has "
                         "re-cut Uruguay")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"unexpected CRS {g.crs}")
    print(f"read {shp}: {len(g)} departments, {g.crs}")

    g["pcode"] = g["ADM1_PCODE"].astype(str).str.strip()
    g["name"] = g["ADM1_ES"].astype(str).str.strip()
    by_name = dict(zip(g["name"].map(fold), g["pcode"]))
    if len(by_name) != EXPECTED_DEPARTMENTS:
        raise SystemExit("COD's department names are not unique — the name join is unsafe")

    # ---- witness 1: every INE department name is a COD name, with no aliases ----
    missing = [n for n in INE_DEPARTMENTS.values() if fold(n) not in by_name]
    if missing:
        raise SystemExit(f"INE names with no polygon: {missing} — the name join FAILED")
    spare = sorted(n for n in g["name"]
                   if fold(n) not in {fold(v) for v in INE_DEPARTMENTS.values()})
    if spare:
        raise SystemExit(f"polygons with no INE department: {spare}")
    print(f"  witness 1 — all {EXPECTED_DEPARTMENTS} INE names match a COD name exactly, "
          "no aliases needed")

    check_code_join(by_name)

    # ---- witness 3: LAPOP's own labels, on the same numbering ----
    bad = []
    for code, label in sorted(LAPOP_PROVINCES.items()):
        if not label:
            continue
        if fold(label) != fold(INE_DEPARTMENTS[code - 1400]):
            bad.append((code, label, INE_DEPARTMENTS[code - 1400]))
    blank = sorted(c for c, v in LAPOP_PROVINCES.items() if not v)
    if bad:
        for code, label, ine in bad:
            print(f"      LAPOP {code} {label!r} against INE {ine!r}")
        raise SystemExit("LAPOP's province labels no longer agree with INE's numbering")
    print(f"  witness 3 — LAPOP's `prov_es` labels agree with INE's numbering on "
          f"{len(LAPOP_PROVINCES) - len(blank)} of {EXPECTED_DEPARTMENTS}; "
          f"{blank} carries an EMPTY label, which is Flores")

    # ---- witness 4: INE's own population workbook, joined on the name ----
    pop = read_pop()
    pop_missing = [n for n in pop.index if fold(n) not in by_name]
    if pop_missing:
        raise SystemExit(f"population sheets with no polygon: {pop_missing}")
    pop["pcode"] = [by_name[fold(n)] for n in pop.index]
    if set(pop["pcode"]) != set(g["pcode"]):
        raise SystemExit("the population workbook and COD-AB do not cover the same pcodes")
    g = g.merge(pop.reset_index(names="ine_name")[["pcode", "pop_2023"]], on="pcode",
                how="left")
    if g["pop_2023"].isna().any():
        raise SystemExit("a department came out of the population join with no total")
    g["pop"] = g["pop_2023"].round().astype("int64")
    print(f"  witness 4 — INE A.1.2 joins on the name, all {len(g)} departments: "
          f"{g['pop'].sum():,} people")

    # Montevideo is the smallest department by area and by far the largest by population, and
    # Durazno the sparsest; a permuted population join is what this catches.
    g["area_km2"] = g.to_crs(3857).geometry.area / 1e6
    g["density"] = g["pop"] / g["area_km2"]
    hi = g.loc[g["density"].idxmax(), "name"]
    biggest = g.loc[g["pop"].idxmax(), "name"]
    print(f"    densest {hi!r}, sparsest {g.loc[g['density'].idxmin(), 'name']!r}; "
          f"largest population {biggest!r}, smallest {g.loc[g['pop'].idxmin(), 'name']!r}")
    if fold(hi) != fold("Montevideo") or fold(biggest) != fold("Montevideo"):
        raise SystemExit("Montevideo is not both the densest and the largest department — "
                         "the population join is permuted")

    g["unit"] = g["pcode"]
    # geo_id IS the pcode. INE's own department numbering never leaves sources/uy.py, so
    # there is one department key in data/ and nobody downstream can pick the wrong one.
    g["geo_id"] = g["pcode"]

    os.makedirs(OUT_DIR, exist_ok=True)
    out = g[["unit", "name", "pcode", "geo_id", "pop", "geometry"]]
    out.to_file(OUT, layer="departamentos", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    num_by_pcode = {by_name[fold(n)]: k for k, n in INE_DEPARTMENTS.items()}
    order = sorted(g["pcode"])
    lut = pd.DataFrame({
        "geo_id": order,
        "unit": order,
        "name": [g.loc[g["pcode"] == p, "name"].iloc[0] for p in order],
        "ine_dpto": [num_by_pcode[p] for p in order],
        "lapop_prov": [1400 + num_by_pcode[p] for p in order],
    })
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows)")

    pop = pop.reset_index(names="ine_name")
    pop["geo_id"] = pop["pcode"]
    pop["ine_dpto"] = pop["geo_id"].map(num_by_pcode)
    band_cols = sorted((c for c in pop.columns if c.startswith("T_")),
                       key=lambda c: int(re.findall(r"\d+", c)[0]))
    cols = ["geo_id", "ine_name", "ine_dpto", "pop_2023", "pop7_2023",
            "pop_2006", "pop7_2006"] + band_cols
    pop.sort_values("geo_id")[cols].to_csv(POP_OUT, index=False, encoding="utf-8")
    print(f"wrote {POP_OUT} ({len(pop)} departments, {len(band_cols)} age bands, "
          f"{pop['pop_2023'].sum():,.0f} people in 2023 of whom "
          f"{pop['pop7_2023'].sum():,.0f} are 7 or over)")


if __name__ == "__main__":
    main()
