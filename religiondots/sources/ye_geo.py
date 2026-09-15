"""Yemen — boundaries and populations for the 22 governorates.

Writes data/geo/ye/ye_governorates.gpkg and data/geo/ye/ye_lookup.csv.

Two sources, both OCHA-hosted and both the Central Statistical Organization's underneath:

  * **boundaries** — **COD-AB for Yemen** on HDX, `yem_admin1.shp` inside
    `yem_admin_boundaries.shp.zip`: 22 governorates with `YE11`-`YE32` p-codes, provided by the
    CSO and approved by the Humanitarian Country Team in October 2019, reviewed December 2024.
  * **populations** — **the Population Task Force's 2025 estimate** (CSO, UNFPA, IOM, OCHA),
    `yem_population_projection_2025_final.xlsx`, by district with p-codes: the CSO's own 2025
    projection from the 2004 census, adjusted district by district for displacement from the
    DTM, CCCM and UNHCR lists. **34,879,018 people.** The methodology note prints the
    governorate totals as its Results table, and `pdf_totals` reads that table back and asserts
    the spreadsheet's district sums equal it.

## THERE HAS BEEN NO CENSUS SINCE 2004, AND THIS IS A PROJECTION OF IT

The note says so in its own words: *"the reliance on projections from the last official census
(2004) introduces a level of uncertainty"*. Nothing better exists. It is the figure every
humanitarian plan for Yemen uses, it is two decades of projection plus a displacement
correction, and the CSO's unadjusted projection sits beside it in the same file, so
`ye_lookup.csv` carries both and this prints the ratio per governorate. Ma'rib is the one that
moves: it is the largest IDP destination in the country.

## SOCOTRA IS A GOVERNORATE THE SURVEY NEVER SAMPLED

Socotra was split from Hadramawt in December 2013. Both files here have it (75,725 people); the
Arab Barometer's governorate list has 21 units in both Yemeni waves and not Socotra. It is kept
as a polygon with its population so the build can say exactly who is not drawn.

Usage:
    python sources/ye_geo.py --fetch    one 5.3 MB shapefile zip, one spreadsheet, one PDF
    python sources/ye_geo.py            rebuild from data/raw/ye/
"""

import os
import re
import ssl
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ye")
OUT_DIR = os.path.join(ROOT, "data", "geo", "ye")
OUT = os.path.join(OUT_DIR, "ye_governorates.gpkg")
LOOKUP = os.path.join(OUT_DIR, "ye_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

HDX_AB = ("https://data.humdata.org/dataset/6b2656e2-b915-4671-bfed-468d5edcd80a/resource/"
          "eb58b807-bea0-450f-a654-0fa49054b4e0/download/yem_admin_boundaries.shp.zip")
HDX_PS = "https://data.humdata.org/dataset/1ffe81f1-b980-430f-b53e-dd79e936f291/resource/"
POP_URL = HDX_PS + ("5a3cb3f9-5148-4f5d-b269-f50c5a81a2d3/download/"
                    "yem_population_projection_2025_final.xlsx")
NOTE_URL = HDX_PS + ("fc3e440b-1e14-408e-b69f-32931a8ee65e/download/"
                     "2025-population-estimates-methodology-english_hp.pdf")

COD_ZIP = os.path.join(RAW, "yem_admin_boundaries.shp.zip")
POP_XLSX = os.path.join(RAW, "yem_population_projection_2025_final.xlsx")
NOTE_PDF = os.path.join(RAW, "yem_population_methodology_2025_en.pdf")
POP_SHEET = "Population 2025 SAAD"

EXPECTED_GOVERNORATES = 22
PTF_TOTAL = 34_879_018             # the note's Grand Total and the Summary sheet's
SOCOTRA = "YE32"

# Column positions in the district sheet, which has three header rows (English, a merged
# label row, HXL tags). Read by position and asserted by the HXL tag on row 3, so a
# re-issue that moves a column stops here.
COLS = {0: "#adm1 +name", 1: "#adm1 +code", 2: "#adm2 +name", 3: "#adm2 +code",
        4: "2025_CSO Estimated Population", 5: "2025_#population+idps",
        6: "2025_#population +total"}
UNDER15 = [7, 8, 9, 24, 25, 26]    # F 0-4, 5-9, 10-14 and M 0-4, 5-9, 10-14
AGE_BINS = list(range(7, 41))

# Density witnesses. Al Maharah and Hadramawt are the empty east, and the next tier is Al Jawf,
# Shabwah and Socotra, within 10% of each other, so the assertion names the set and not the
# order. Sana'a City is the capital's own governorate (Amanat al-Asimah) at about 500 km2, and
# Aden the port city on 800.
SPARSEST_TWO = {"YE28", "YE19"}
NEXT_TIER = {"YE16", "YE21", "YE32"}
DENSEST_TWO = ["YE13", "YE24"]

LAEA = "+proj=laea +lat_0=15.5 +lon_0=48 +datum=WGS84 +units=m +no_defs"


def fold(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


def _ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _get(url, dst, magic, minsize):
    if os.path.exists(dst) and os.path.getsize(dst) > minsize:
        print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
        return
    print("  GET", url)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=900,
                                context=_ctx()) as r:
        data = r.read()
    if not data.startswith(magic):          # §5a: a 200 is not a download
        raise SystemExit(f"{url} did not return the expected file, starts {data[:24]!r}")
    with open(dst + ".part", "wb") as f:
        f.write(data)
    os.replace(dst + ".part", dst)
    print(f"  got  {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    _get(HDX_AB, COD_ZIP, b"PK", 1_000_000)
    _get(POP_URL, POP_XLSX, b"PK", 100_000)
    _get(NOTE_URL, NOTE_PDF, b"%PDF", 50_000)


def districts():
    """The Task Force's district table: one row per district, p-coded, with the age bins."""
    raw = pd.read_excel(POP_XLSX, sheet_name=POP_SHEET, header=None)
    tags = raw.iloc[2]
    for i, tag in COLS.items():
        if str(tags[i]).strip() != tag:
            raise SystemExit(f"column {i} of {POP_SHEET!r} is tagged {tags[i]!r}, not {tag!r}; "
                             "the sheet has been re-laid out")
    d = raw.iloc[3:].copy()
    ok = d[1].astype(str).str.fullmatch(r"YE\d\d") & d[3].astype(str).str.fullmatch(r"YE\d{4}")
    # The sheet ends in one unlabelled row holding its own column totals. It is read as that,
    # asserted, and dropped; any other row without codes stops.
    total_row = ~ok & d[[0, 1, 2, 3]].isna().all(axis=1) & d[6].notna()
    if int(total_row.sum()) != 1 or int(pd.to_numeric(d.loc[total_row, 6]).iloc[0]) != PTF_TOTAL:
        raise SystemExit(f"expected one unlabelled total row reading {PTF_TOTAL:,} at the foot of "
                         f"{POP_SHEET!r}, found {int(total_row.sum())}")
    d, ok = d[~total_row], ok[~total_row]
    stray = d[~ok & d.notna().any(axis=1)]
    if len(stray):
        raise SystemExit(f"{len(stray)} rows in {POP_SHEET!r} are not district rows:\n"
                         f"{stray.iloc[:, :7].head()}")
    d = d[ok]
    bad = d[d[3].str[:4] != d[1]]
    if len(bad):
        raise SystemExit(f"districts whose p-code is not inside their governorate's: "
                         f"{bad[[1, 3]].values.tolist()}")
    out = pd.DataFrame({
        "gov": d[1].astype(str), "gov_name": d[0].astype(str).str.strip(),
        "district": d[3].astype(str), "cso": pd.to_numeric(d[4]).astype(float),
        "idps": pd.to_numeric(d[5]).astype(float), "pop": pd.to_numeric(d[6]).astype("int64"),
        "under15": d[UNDER15].apply(pd.to_numeric).sum(axis=1).astype("int64"),
        "bins": d[AGE_BINS].apply(pd.to_numeric).sum(axis=1).astype("int64"),
    })
    if out["district"].duplicated().any():
        raise SystemExit(f"duplicated district p-codes: "
                         f"{sorted(out.loc[out['district'].duplicated(), 'district'])}")
    drift = (out["bins"] - out["pop"]).abs()
    print(f"  {len(out)} districts, {out['gov'].nunique()} governorates, "
          f"{int(out['pop'].sum()):,} people; the 34 sex-by-age bins miss each district's total "
          f"by at most {int(drift.max())}")
    if int(out["pop"].sum()) != PTF_TOTAL:
        raise SystemExit(f"the districts sum to {int(out['pop'].sum()):,}, not the Task Force's "
                         f"{PTF_TOTAL:,}")
    if drift.max() > 100:
        raise SystemExit("an age-bin row does not add up to its district total, so the columns "
                         "are not the ones this reads")
    return out


def pdf_totals():
    """The methodology note's Results table, read back off page 6: governorate -> total."""
    import fitz

    doc = fitz.open(NOTE_PDF)
    if doc.page_count == 0:
        raise SystemExit(f"{NOTE_PDF} opened with zero pages; the download is damaged")
    lines = [l.strip() for p in doc for l in p.get_text().splitlines() if l.strip()]
    num = re.compile(r"^\d{1,3}(,\d{3})*$")
    out = {}
    for i in range(len(lines) - 4):
        if all(num.match(x) for x in lines[i + 1:i + 5]) and not num.match(lines[i]):
            vals = [int(x.replace(",", "")) for x in lines[i + 1:i + 5]]
            if vals[0] + vals[1] + vals[2] == vals[3]:
                out[lines[i]] = vals[3]
    if out.pop("Grand Total", None) != PTF_TOTAL:
        raise SystemExit("the note's Results table has no Grand Total of "
                         f"{PTF_TOTAL:,}; its layout has changed")
    return out


def main():
    if "--fetch" in sys.argv:
        fetch()
    for p in (COD_ZIP, POP_XLSX, NOTE_PDF):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing; run with --fetch")

    g = gpd.read_file(f"zip://{COD_ZIP}!yem_admin1.shp")
    if len(g) != EXPECTED_GOVERNORATES:
        raise SystemExit(f"{len(g)} ADM1 features, expected {EXPECTED_GOVERNORATES}")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"unexpected CRS {g.crs}")
    print(f"read yem_admin1.shp: {len(g)} governorates, valid_on "
          f"{sorted(set(g['valid_on'].astype(str)))}")

    dist = districts()
    gov = dist.groupby("gov").agg(pop=("pop", "sum"), cso=("cso", "sum"), idps=("idps", "sum"),
                                  under15=("under15", "sum"), districts=("district", "size"),
                                  name=("gov_name", "first"))

    # ---- witness 1: the p-codes on both sides are the same 22 ----
    off = sorted(set(g["adm1_pcode"].astype(str)) ^ set(gov.index))
    if off:
        raise SystemExit(f"COD-AB and the Task Force table disagree on governorate p-codes: {off}")
    print("  witness 1: COD-AB's 22 p-codes are the Task Force table's 22")

    # ---- witness 2: the note's printed Results table against the spreadsheet ----
    printed = pdf_totals()
    by_name = {fold(n): v for n, v in printed.items()}
    cod_name = dict(zip(g["adm1_pcode"], g["adm1_name"]))
    miss = [cod_name[k] for k in gov.index if fold(cod_name[k]) not in by_name]
    if miss or len(printed) != EXPECTED_GOVERNORATES:
        raise SystemExit(f"the note's Results table has {len(printed)} governorates and does not "
                         f"name {miss} the way COD-AB does")
    wrong = {cod_name[k]: (int(gov.loc[k, "pop"]), by_name[fold(cod_name[k])])
             for k in gov.index if int(gov.loc[k, "pop"]) != by_name[fold(cod_name[k])]}
    if wrong:
        raise SystemExit(f"district sums against the note's printed totals: {wrong}")
    print("  witness 2: every governorate's district sum is the total the methodology note "
          "prints for it, to the person")

    # ---- witness 3: the district polygons sit inside the governorate their code names ----
    d2 = gpd.read_file(f"zip://{COD_ZIP}!yem_admin2.shp")
    pts = gpd.GeoDataFrame({"code": d2["adm2_pcode"].astype(str).str[:4]},
                           geometry=d2.to_crs(LAEA).representative_point(),
                           crs=LAEA).to_crs(g.crs)
    hit = gpd.sjoin(pts, g[["adm1_pcode", "geometry"]], how="left", predicate="within")
    hit = hit[~hit.index.duplicated()]
    agree = int((hit["code"] == hit["adm1_pcode"]).sum())
    print(f"  witness 3: {agree} of {len(d2)} COD district polygons fall inside the governorate "
          "polygon their own p-code names")
    if agree < len(d2) - 2:
        raise SystemExit("more than two districts sit outside their coded governorate; the "
                         "admin1 geometry and codes are not the pairing they look like")
    only_pop = sorted(set(dist["district"]) - set(d2["adm2_pcode"]))
    only_cod = sorted(set(d2["adm2_pcode"]) - set(dist["district"]))
    print(f"    districts in the population table and not COD-AB: {only_pop or 'none'}; in "
          f"COD-AB and not the table: {only_cod or 'none'} (printed; the build is governorates)")

    g["geo_id"] = g["adm1_pcode"].astype(str)
    g["unit"] = g["geo_id"]
    g["name"] = g["adm1_name"].astype(str)
    g["name_ar"] = g["adm1_name1"].astype(str)
    for col in ("pop", "under15", "districts"):
        g[col] = g["geo_id"].map(gov[col]).astype("int64")
    g["cso"] = g["geo_id"].map(gov["cso"]).round().astype("int64")
    g["idps"] = g["geo_id"].map(gov["idps"]).round().astype("int64")
    g["area_km2"] = g.to_crs(LAEA).geometry.area / 1e6

    # ---- witness 4: the shape of the population on the ground ----
    g["density"] = g["pop"] / g["area_km2"]
    order = g.sort_values("density")
    print("  witness 4: sparsest "
          + ", ".join(f"{n} {d:.1f}/km2" for n, d in zip(order["name"][:5], order["density"][:5]))
          + f"; densest {order['name'].iloc[-1]} {order['density'].iloc[-1]:,.0f}/km2 and "
          f"{order['name'].iloc[-2]} {order['density'].iloc[-2]:,.0f}/km2")
    if set(order["geo_id"][:2]) != SPARSEST_TWO or not set(order["geo_id"][2:4]) <= NEXT_TIER:
        raise SystemExit("the sparsest governorates are not the empty east; the population join "
                         "is permuted")
    if list(order["geo_id"][::-1][:2]) != DENSEST_TWO:
        raise SystemExit("the two densest governorates are not Sana'a City and Aden")

    g["cso_ratio"] = g["pop"] / g["cso"]
    print("\n  Task Force 2025 against the CSO's own unadjusted 2025 projection, per governorate:")
    for _i, r in g.sort_values("cso_ratio").iterrows():
        print(f"    {r['name']:<14}{r['pop']:>11,}  CSO {r['cso']:>11,}  {r['cso_ratio']:.3f}x  "
              f"IDPs {r['idps']:>9,}")
    u15 = int(g["under15"].sum())
    print(f"\n  under 15: {u15:,} of {PTF_TOTAL:,}, {u15 / PTF_TOTAL:.1%} (the survey interviews "
          "people 18 and over)")
    print(f"  Socotra ({SOCOTRA}): {int(g.loc[g['geo_id'] == SOCOTRA, 'pop'].iloc[0]):,} people, "
          f"{int(g.loc[g['geo_id'] == SOCOTRA, 'pop'].iloc[0]) / PTF_TOTAL:.4%} of the total")

    os.makedirs(OUT_DIR, exist_ok=True)
    g[["unit", "name", "name_ar", "geo_id", "pop", "geometry"]].to_file(
        OUT, layer="governorates", driver="GPKG")
    print(f"\nwrote {OUT} ({len(g)} polygons)")
    lut = (g[["geo_id", "unit", "name", "name_ar", "pop", "cso", "idps", "under15", "districts",
              "area_km2"]].sort_values("geo_id").reset_index(drop=True))
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows, {int(lut['pop'].sum()):,} people)")


if __name__ == "__main__":
    main()
