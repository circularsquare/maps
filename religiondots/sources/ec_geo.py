"""Ecuador — boundaries for the 24 provincias, and the population that goes on them.

Writes data/geo/ec/ec_provincias.gpkg, data/geo/ec/ec_lookup.csv and
data/geo/ec/ec_pop_2022.csv.

## THE POPULATION IS A CENSUS COUNT, NOT A PROJECTION, AND THAT IS NEW IN THIS SET

Guatemala and El Salvador are drawn on OCHA COD-PS projections because neither country has
counted recently. **Ecuador has.** The VIII Censo de Población y VII de Vivienda enumerated
**16,938,986 people in November 2022**, and INEC publishes it by province, by canton, by
parroquia and by five-year age band. So this file reads INEC and not COD-PS, and the reason
is not a preference for the local source — it is that COD-PS is wrong here in a way that
would have moved dots:

    COD-PS 2020 (projected off the 2010 census)   17,510,643
    INEC 2022 census (counted)                    16,938,986      -3.4%

**And the error is not a level shift, it is uneven.** The 2010-based projection missed by
different amounts in different provinces, so using it would have distorted the shares as well
as the total:

    Loja          COD-PS 521,154   census 485,421    -6.9%
    Galápagos              33,042           28,583   -13.5%
    Bolívar               209,933          199,078    -5.2%
    Pichincha           3,228,233        3,089,473    -4.3%
    Guayas              4,387,434        4,391,923    +0.1%
    Manabí              1,562,079        1,592,840    +2.0%

Pichincha down 4.3% while Manabí is up 2.0% is a 6-point swing between the second and third
provinces of the country. **A survey column on the wrong population is a wrong count**, and
§14.4 rule 1's promise — every person drawn is a person somebody counted in that unit — is
only true if the somebody counted them.

## AND IT FIXES A UNIT THAT HAS NO POLYGON

COD-PS 2020 carries **25 rows, not 24**: there is an `EC90` *Zona no delimitada* holding
41,907 people. COD-AB 2024 has no such polygon, because Ecuador resolved its inter-provincial
disputed zones by referendum in 2015-16 and assigned them. So on COD-PS those 41,907 people
would have had to be dropped or invented a home. **The 2022 census has no such row**: 24
provinces, summing exactly to the national total, matching COD-AB's 24 polygons one to one.

## THE JOIN IS GUATEMALA'S, NOT EL SALVADOR'S, AND BOTH KEYS ARE REQUIRED

Ecuador's provinces carry INEC's official DPA numbering, and all three sides use it:

    LAPOP `prov`   901..924  ->  province 01..24   (and it LABELS them, see below)
    COD-AB         EC01..EC24
    INEC census    the same names in the same order

So the code join and the name join are two independent signals over the same 24 rows, and
this file requires BOTH — `sources/gt_geo.py`'s construction, for `sources/ni_geo.py`'s
reason. El Salvador is the warning: there the two keys disagree on twelve of fourteen rows
while every total still reconciles, so an arithmetic check can never find the fault.

**LAPOP labels its own province codes and that is a third witness**, better than either.
`prov_es` in the Grand Merge names 901 Azuay through 924 Santa Elena, and those labels are
checked here one at a time against COD's. **One alias is needed**: LAPOP abbreviates 923 to
`S.D. De los Tsáchilas` where COD and INEC write `Santo Domingo de los Tsáchilas`.

**LAPOP HAS NO CODE 920.** Galápagos is not merely unsampled — it is absent from the value
label set, so the survey never offered it. The province is carried here with its census
population and `lapop_prov` empty; `sources/ec.py` decides what to draw there.

## THE DOWNLOAD IS 736 MB FOR 24 POLYGONS

`cod-ab-ecu` ships one zip containing ADM0 through **ADM4**, and ADM4 is the census sector
layer. Guatemala's equivalent is 2.9 MB and El Salvador's 3.9 MB. Only the seven `adm1`
members are extracted; the rest is never unpacked. The zip is kept so `--fetch` is not a
three-quarter-gigabyte round trip every time, and it is the one file in this country worth
deleting if the tree needs room.

Usage:
    python sources/ec_geo.py --fetch    a 736 MB zip from HDX, plus a 1.5 MB xlsx from INEC
    python sources/ec_geo.py            rebuild from data/raw/ec/
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
RAW = os.path.join(ROOT, "data", "raw", "ec")
SHP_DIR = os.path.join(RAW, "shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "ec")
OUT = os.path.join(OUT_DIR, "ec_provincias.gpkg")
LOOKUP = os.path.join(OUT_DIR, "ec_lookup.csv")
POP_OUT = os.path.join(OUT_DIR, "ec_pop_2022.csv")

CENSUS_XLSX = "01_2022_CPV_Estructura_poblacional.xlsx"

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"}

DOWNLOADS = {
    # cod-ab-ecu on HDX. 736 MB, because it carries ADM0..ADM4 and ADM4 is census sectors.
    "ecu_adm_2024.zip":
        "https://data.humdata.org/dataset/ab3c7592-3b0c-41cd-999a-2919a6b243f2/resource/"
        "d00145f6-141c-4bf2-a881-c32341ddec75/download/ecu_adm_2024.zip",
    # INEC's own 2022 census tabulado. Sheet '1' is population by province and area of
    # residence; sheet '2' is the same by five-year age band. Found through the census
    # site's WordPress media library, which lists files no page links to.
    CENSUS_XLSX:
        "https://www.censoecuador.gob.ec/wp-content/uploads/2024/04/"
        "01_2022_CPV_Estructura_poblacional.xlsx",
}

# censoecuador.gob.ec serves an incomplete certificate chain. Nothing is trusted on the
# strength of the transport: the workbook is INEC's own and its national total is asserted
# against the published 16,938,986 below.
LAX_TLS = {"www.censoecuador.gob.ec"}

# LAPOP's `prov_es` value labels for the 900s, verbatim from the Grand Merge. 900 + INEC's
# official province number. Written out so a change on either side fails here rather than
# re-pairing something downstream. NOTE THE GAP AT 920: LAPOP has no Galápagos code at all.
LAPOP_PROVINCES = {
    901: "Azuay",            902: "Bolívar",          903: "Cañar",
    904: "Carchi",           905: "Cotopaxi",         906: "Chimborazo",
    907: "El Oro",           908: "Esmeraldas",       909: "Guayas",
    910: "Imbabura",         911: "Loja",             912: "Los Ríos",
    913: "Manabí",           914: "Morona Santiago",  915: "Napo",
    916: "Pastaza",          917: "Pichincha",        918: "Tungurahua",
    919: "Zamora Chinchipe", 921: "Sucumbíos",        922: "Orellana",
    923: "S.D. De los Tsáchilas",                     924: "Santa Elena",
}

# The one place LAPOP's label is not COD's name. LAPOP abbreviates; COD and INEC do not.
ALIASES = {"sddelostsachilas": "santodomingodelostsachilas"}

# The province LAPOP does not offer. Not a sampling accident — there is no code 920 in the
# value label set, so no Ecuadorian respondent could ever have been placed there.
UNSAMPLED = "EC20"

EXPECTED_PROVINCES = 24
CENSUS_TOTAL = 16_938_986          # INEC's published national count, 2022


def fold(s):
    """Accent- and case-insensitive key for a province name."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalnum())


def key(name):
    """`fold`, then LAPOP's one abbreviation folded onto COD's spelling."""
    k = fold(name)
    return ALIASES.get(k, k)


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
        with urllib.request.urlopen(req, timeout=1800, context=ctx) as r:
            body = r.read()
        with open(dst + ".part", "wb") as f:
            f.write(body)
        os.replace(dst + ".part", dst)
        # §5a: a 200 is not a download. Both of these are zip containers.
        with open(dst, "rb") as f:
            magic = f.read(4)
        if magic[:2] != b"PK":
            raise SystemExit(f"{dst} is not a zip container — starts {magic!r}")
        print(f"  got  {name} ({os.path.getsize(dst):,} bytes)")


def read_census():
    """INEC's 2022 census: province totals from sheet 1, age bands from sheet 2.

    Returns a frame indexed by province NAME with `pop` and `T_<lo>_<hi>` / `T_85Plus`
    columns, shaped so `lapop._age_bands` can read the bands off the columns present.
    """
    import openpyxl

    path = os.path.join(RAW, CENSUS_XLSX)
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing — run with --fetch")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)

    # ---- sheet 1: `<province> | Total <province> | <n> | <men> | <women>` ----
    totals = {}
    national = None
    for row in wb["1"].iter_rows(values_only=True):
        c = ["" if x is None else str(x).strip() for x in row]
        if len(c) < 4 or not c[1] or not c[2]:
            continue
        if c[1] == "Total Nacional" and c[2] == "Total Nacional":
            national = int(float(c[3]))
        elif c[2] == f"Total {c[1]}":
            totals[c[1]] = int(float(c[3]))
    if national != CENSUS_TOTAL:
        raise SystemExit(f"sheet 1 national total is {national:,}, not the published "
                         f"{CENSUS_TOTAL:,} — INEC has revised the tabulado")
    if len(totals) != EXPECTED_PROVINCES:
        raise SystemExit(f"{len(totals)} provinces in sheet 1, expected {EXPECTED_PROVINCES}")
    if sum(totals.values()) != CENSUS_TOTAL:
        raise SystemExit(f"the {len(totals)} provinces sum to {sum(totals.values()):,}, "
                         f"not {CENSUS_TOTAL:,}")
    print(f"  INEC sheet 1: {len(totals)} provinces summing EXACTLY to {CENSUS_TOTAL:,}")

    # ---- sheet 2: the same provinces by five-year band ----
    bands = {}
    for row in wb["2"].iter_rows(values_only=True):
        c = ["" if x is None else str(x).strip() for x in row]
        if len(c) < 4 or not c[1] or not c[2] or c[1] == "Total Nacional":
            continue
        m = re.fullmatch(r"De (\d+)-(\d+)", c[2])
        col = None
        if m:
            col = f"T_{int(m.group(1))}_{int(m.group(2))}"
        elif c[2] == "85 o más":
            col = "T_85Plus"
        if col:
            bands.setdefault(c[1], {})[col] = int(float(c[3]))
    wb.close()

    if set(bands) != set(totals):
        raise SystemExit(f"sheet 2 covers {sorted(set(bands) ^ set(totals))} differently "
                         "from sheet 1")
    pop = pd.DataFrame(bands).T
    pop["pop"] = pd.Series(totals)
    # The bands must be the province, not a subset of it — this is the check that the two
    # sheets are the same tabulation rather than two different universes.
    band_cols = [c for c in pop.columns if c.startswith("T_")]
    off = (pop[band_cols].sum(axis=1) - pop["pop"]).abs()
    if off.max() != 0:
        raise SystemExit(f"sheet 2's bands do not sum to sheet 1's totals; worst is "
                         f"{off.idxmax()} by {int(off.max()):,}")
    print(f"  INEC sheet 2: {len(band_cols)} age bands, summing to sheet 1 on all "
          f"{len(pop)} provinces with no residual")
    return pop


def check_code_join(by_name):
    """Ecuador's code join AGREES with its name join, and this requires it to.

    The opposite assertion to `sources/sv_geo.py`, which requires El Salvador's code join to
    keep mispairing. Both files exist for the same reason: whichever way the two keys relate,
    a silent change in that relationship must stop the build rather than redraw the country.
    """
    wrong = []
    for code, name in sorted(LAPOP_PROVINCES.items()):
        naive = f"EC{code - 900:02d}"
        actual = by_name[key(name)]
        if naive != actual:
            wrong.append((code, name, naive, actual))
    print(f"\n  witness 2 — the code join AGREES: `prov - 900` -> ECnn pairs "
          f"{len(LAPOP_PROVINCES) - len(wrong)} of {len(LAPOP_PROVINCES)} the same way the "
          "names do")
    if wrong:
        for code, name, naive, actual in wrong:
            print(f"      LAPOP {code} {name} -> {naive}, but the name says {actual}")
        raise SystemExit(
            "Ecuador's two keys have stopped agreeing. That is either an OCHA re-cut or a "
            "LAPOP re-code, and it is exactly the failure sources/ni_geo.py describes: a "
            "permutation preserves every total, so nothing downstream would catch it. STOP "
            "and decide which key is right.")


def main():
    if "--fetch" in sys.argv:
        fetch()

    zpath = os.path.join(RAW, "ecu_adm_2024.zip")
    if not os.path.exists(zpath):
        raise SystemExit(f"{zpath} missing — run with --fetch")
    os.makedirs(SHP_DIR, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        members = [n for n in z.namelist() if "adm1" in n and not n.endswith(".xml")]
        if not members:
            raise SystemExit("no adm1 members in the COD-AB zip — it has been re-cut")
        for n in members:
            z.extract(n, SHP_DIR)
    print(f"extracted {len(members)} adm1 members, leaving ADM0 and ADM2-4 packed")

    shp = os.path.join(SHP_DIR, "ecu_adm_adm1_2024.shp")
    # `engine="fiona"` on §12's Chile rule: pyogrio is geopandas' default when installed and
    # is the engine that has silently returned zero features. The count is asserted either way.
    g = gpd.read_file(shp, engine="fiona")
    if len(g) != EXPECTED_PROVINCES:
        raise SystemExit(f"{len(g)} ADM1 features, expected {EXPECTED_PROVINCES} — COD has "
                         "re-cut Ecuador")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"unexpected CRS {g.crs}")
    print(f"read {shp}: {len(g)} provinces, {g.crs}")

    g["pcode"] = g["ADM1_PCODE"].astype(str).str.strip()
    g["name"] = g["ADM1_ES"].astype(str).str.strip()
    by_name = dict(zip(g["name"].map(key), g["pcode"]))
    if len(by_name) != EXPECTED_PROVINCES:
        raise SystemExit("COD's province names are not unique — the name join is unsafe")

    # ---- witness 1: every LAPOP label is a COD name ----
    missing = [n for n in LAPOP_PROVINCES.values() if key(n) not in by_name]
    if missing:
        raise SystemExit(f"LAPOP labels with no polygon: {missing} — the name join FAILED")
    spare = sorted(n for n in g["name"] if key(n) not in
                   {key(v) for v in LAPOP_PROVINCES.values()})
    print(f"  witness 1 — all {len(LAPOP_PROVINCES)} LAPOP labels match a COD name "
          f"(one alias: {list(ALIASES)[0]!r})")
    if spare != ["Galápagos"]:
        raise SystemExit(f"polygons with no LAPOP label: {spare}, expected only Galápagos. "
                         "LAPOP's province card has changed.")
    print(f"    the one province LAPOP never offers is {spare[0]} ({UNSAMPLED}) — there is "
          "no code 920 in `prov_es`")

    check_code_join(by_name)

    # ---- witness 3: INEC's own census, joined on the name ----
    pop = read_census()
    pop_missing = [n for n in pop.index if key(n) not in by_name]
    if pop_missing:
        raise SystemExit(f"census provinces with no polygon: {pop_missing}")
    pop["pcode"] = [by_name[key(n)] for n in pop.index]
    if set(pop["pcode"]) != set(g["pcode"]):
        raise SystemExit("the census and COD-AB do not cover the same 24 pcodes")
    g = g.merge(pop.reset_index(names="census_name")[["pcode", "pop"]], on="pcode",
                how="left")
    if g["pop"].isna().any():
        raise SystemExit("a province came out of the population join with no total")
    g["pop"] = g["pop"].astype("int64")
    print(f"  witness 3 — INEC 2022 joins on the name, all {len(g)} provinces: "
          f"{g['pop'].sum():,} people")

    # Pichincha (Quito) is the densest province and Pastaza — Amazonian, 29,000 km² — the
    # sparsest. A permuted population join is what this catches, and it is worth having
    # because Ecuador's provinces vary in population by a factor of 154.
    g["area_km2"] = g.to_crs(3857).geometry.area / 1e6
    g["density"] = g["pop"] / g["area_km2"]
    lo = g.loc[g["density"].idxmin(), "name"]
    hi = g.loc[g["density"].idxmax(), "name"]
    print(f"    sparsest {lo!r}, densest {hi!r}; largest population "
          f"{g.loc[g['pop'].idxmax(), 'name']!r}, smallest "
          f"{g.loc[g['pop'].idxmin(), 'name']!r}")
    if fold(hi) != fold("Pichincha"):
        raise SystemExit(f"{hi} is not the densest province — the population join is permuted")
    if fold(g.loc[g["pop"].idxmax(), "name"]) != fold("Guayas"):
        raise SystemExit("Guayas is not the largest province — the population join is permuted")

    g["unit"] = g["pcode"]
    # geo_id IS the pcode, deliberately: LAPOP's own numbering never leaves sources/ec.py,
    # so there is one province numbering in data/ and nobody downstream can pick the wrong one.
    g["geo_id"] = g["pcode"]

    os.makedirs(OUT_DIR, exist_ok=True)
    out = g[["unit", "name", "pcode", "geo_id", "pop", "geometry"]]
    out.to_file(OUT, layer="provincias", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    prov_by_pcode = {by_name[key(n)]: c for c, n in LAPOP_PROVINCES.items()}
    lut = pd.DataFrame({
        "geo_id": sorted(g["pcode"]),
        "unit": sorted(g["pcode"]),
        "name": [g.loc[g["pcode"] == p, "name"].iloc[0] for p in sorted(g["pcode"])],
        "lapop_prov": [prov_by_pcode.get(p, "") for p in sorted(g["pcode"])],
    })
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    n_sampled = int((lut["lapop_prov"] != "").sum())
    print(f"wrote {LOOKUP} ({len(lut)} rows, {n_sampled} with a LAPOP prov code)")

    pop = pop.reset_index(names="census_name")
    pop["geo_id"] = pop["pcode"]
    cols = ["geo_id", "census_name", "pop"] + sorted(
        (c for c in pop.columns if c.startswith("T_")),
        key=lambda c: int(re.findall(r"\d+", c)[0]))
    pop[cols].sort_values("geo_id").to_csv(POP_OUT, index=False, encoding="utf-8")
    print(f"wrote {POP_OUT} ({len(pop)} provinces, {len(cols) - 3} age bands, "
          f"{int(pop['pop'].sum()):,} people)")


if __name__ == "__main__":
    main()
