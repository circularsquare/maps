"""Tanzania — boundaries and 2022 census populations for the 31 regions, drawn as 30 units.

Writes data/geo/tz/tz_regions.gpkg, data/geo/tz/tz_lookup.csv and data/geo/tz/tz_districts.csv.

  * **boundaries**: OCHA COD-AB Tanzania (`cod-ab-tza`), the 2018-10-19 shapefiles, sourced from
    the National Bureau of Statistics and OCHA ROSA, read with `engine="fiona"`. ADM1 is the 31
    regions, pcodes TZ01-TZ26 on the mainland and TZ51-TZ55 in Zanzibar, and it already has
    Songwe (TZ26, created January 2016). ADM2 is read for its district NAMES only, into
    `tz_districts.csv`, which `sources/tz.py` uses to place round 4's respondents in the regions
    created in 2012, after round 4 was in the field.
  * **populations**: National Bureau of Statistics, *2022 Population and Housing Census: Initial
    Results* (October 2022), Table 1 "Population Distribution by Sex and Region", report page 8
    (PDF page 17). 61,741,120 people. Transcribed below and re-read from the PDF on every run, so
    a digit slip is a failure here rather than a drift downstream.

## THE OFFICE'S 2022 COUNT, NOT COD-PS

COD-PS Tanzania (`cod-ps-tza`) is a 2020 projection off the 2012 census, and HDX's own page says
it does not reflect the creation of Songwe. The 2022 census counted every region two years
later, so the office's count is used, on Ecuador's reasoning (§9bn).

## MBEYA AND SONGWE ARE ONE UNIT

Songwe was cut from the western half of Mbeya in January 2016. The Afrobarometer labels it on its
own only in rounds 8 and 9. In rounds 4, 6 and 7 its districts are filed under `Mbeya` (round 6
lists Mbozi and Momba there, round 4 lists Mbozi), so the survey's pooled Mbeya describes the two
regions together and cannot be split back into them. They are dissolved into one unit, `TZ12`,
"Mbeya and Songwe", carrying the sum of the two census counts.

## THE JOIN

COD-AB and the census table spell one region differently (`Dar-es-salaam` against `Dar es
Salaam`). The join is on a folded name and is asserted to be a bijection over all 31, in both
directions ([[reference_name_join_wrong_neighbour]]).

Usage:
    python sources/tz_geo.py --fetch    two zips from HDX (~9 MB) and the NBS report (~6 MB)
    python sources/tz_geo.py            rebuild from data/raw/tz/
"""

import os
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "tz")
OUT_DIR = os.path.join(ROOT, "data", "geo", "tz")
OUT = os.path.join(OUT_DIR, "tz_regions.gpkg")
LOOKUP = os.path.join(OUT_DIR, "tz_lookup.csv")
DISTRICTS = os.path.join(OUT_DIR, "tz_districts.csv")

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"

_HDX = ("https://data.humdata.org/dataset/451bdd28-d06d-46ea-91c0-2e081f884395/resource/")
DOWNLOADS = {
    "tza_admbnda_adm1_20181019.zip":
        _HDX + "55e9e1d3-6585-4e38-ae35-726875c1e8ed/download/tza_admbnda_adm1_20181019.zip",
    "tza_admbnda_adm2_20181019.zip":
        _HDX + "50fec0b7-ba83-43fb-bb1c-941e2dbc7a95/download/tza_admbnda_adm2_20181019.zip",
}
PDF_NAME = "2022_PHC_Initial_Results_English.pdf"
PDF_URL = ("https://www.nbs.go.tz/uploads/statistics/documents/"
           "en-1720088450-2022%20PHC%20Initial%20Results%20-%20English.pdf")
PDF_PAGE = 17                   # 1-based; the report's own page 8
PDF_TITLE = "Table 1: Population Distribution by Sex and Region"

N_REGIONS = 31
N_UNITS = 30

TOTAL_2022 = 61_741_120
MAINLAND_2022 = 59_851_347
ZANZIBAR_2022 = 1_889_773

# Table 1, "Both Sexes", in the table's own order and spelling.
CENSUS_2022 = {
    # Tanzania Mainland
    "Dodoma": 3_085_625, "Arusha": 2_356_255, "Kilimanjaro": 1_861_934, "Tanga": 2_615_597,
    "Morogoro": 3_197_104, "Pwani": 2_024_947, "Dar es Salaam": 5_383_728, "Lindi": 1_194_028,
    "Mtwara": 1_634_947, "Ruvuma": 1_848_794, "Iringa": 1_192_728, "Mbeya": 2_343_754,
    "Singida": 2_008_058, "Tabora": 3_391_679, "Rukwa": 1_540_519, "Kigoma": 2_470_967,
    "Shinyanga": 2_241_299, "Kagera": 2_989_299, "Mwanza": 3_699_872, "Mara": 2_372_015,
    "Manyara": 1_892_502, "Njombe": 889_946, "Katavi": 1_152_958, "Simiyu": 2_140_497,
    "Geita": 2_977_608, "Songwe": 1_344_687,
    # Tanzania Zanzibar
    "Kaskazini Unguja": 257_290, "Kusini Unguja": 195_873, "Mjini Magharibi": 893_169,
    "Kaskazini Pemba": 272_091, "Kusini Pemba": 271_350,
}
ZANZIBAR = ["Kaskazini Unguja", "Kusini Unguja", "Mjini Magharibi", "Kaskazini Pemba",
            "Kusini Pemba"]

# Songwe (TZ26) is drawn inside Mbeya's unit; see the docstring.
MERGE = {"TZ26": "TZ12"}
MERGED_NAME = {"TZ12": "Mbeya and Songwe"}


def fold(s):
    return " ".join(str(s).replace("-", " ").split()).casefold()


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for name, url in DOWNLOADS.items():
        dst = os.path.join(RAW, name)
        if not (os.path.exists(dst) and os.path.getsize(dst) > 10_000):
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=900) as r:
                data = r.read()
            if data[:2] != b"PK":                       # §5a: a 200 is not a download
                raise SystemExit(f"{name} is not a zip, starts {data[:16]!r}")
            with open(dst + ".part", "wb") as f:
                f.write(data)
            os.replace(dst + ".part", dst)
            print(f"  got  {name} ({os.path.getsize(dst):,} bytes)")
        else:
            print(f"  have {name}")
        with zipfile.ZipFile(dst) as z:
            z.extractall(os.path.join(RAW, name.split("_")[2]))

    pdf = os.path.join(RAW, PDF_NAME)
    if not (os.path.exists(pdf) and os.path.getsize(pdf) > 1_000_000):
        import requests

        # nbs.go.tz answers a complete browser header set; §9cp's note on header completeness.
        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"),
            "Accept": "application/pdf,text/html,*/*;q=0.8", "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br", "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Site": "none",
        }
        r = requests.get(PDF_URL, headers=headers, timeout=600)
        r.raise_for_status()
        if r.content[:4] != b"%PDF" or b"%%EOF" not in r.content[-2048:]:
            raise SystemExit("the NBS report is not a complete PDF (no %PDF header or no %%EOF "
                             "trailer); [[reference_pdf_truncated_at_source]]")
        with open(pdf + ".part", "wb") as f:
            f.write(r.content)
        os.replace(pdf + ".part", pdf)
        print(f"  got  {PDF_NAME} ({os.path.getsize(pdf):,} bytes)")
    else:
        print(f"  have {PDF_NAME}")


def check_pdf():
    """Re-read Table 1 from the report and assert every transcribed figure against it."""
    import fitz

    pdf = os.path.join(RAW, PDF_NAME)
    doc = fitz.open(pdf)
    if doc.page_count < PDF_PAGE:
        raise SystemExit(f"{PDF_NAME} has {doc.page_count} pages; a truncated download reads as "
                         "a short document ([[reference_pdf_truncated_at_source]])")
    lines = [ln.strip() for ln in doc[PDF_PAGE - 1].get_text().splitlines() if ln.strip()]
    if PDF_TITLE not in lines:
        raise SystemExit(f"PDF page {PDF_PAGE} is not {PDF_TITLE!r}; the report has been re-laid")
    want = dict(CENSUS_2022)
    want.update({"Tanzania": TOTAL_2022, "Tanzania Mainland": MAINLAND_2022,
                 "Tanzania Zanzibar": ZANZIBAR_2022})
    for name, n in want.items():
        at = [i for i, ln in enumerate(lines) if ln == name]
        if len(at) != 1:
            raise SystemExit(f"{name!r} appears {len(at)} times on PDF page {PDF_PAGE}")
        got = int(lines[at[0] + 1].replace(",", ""))
        if got != n:
            raise SystemExit(f"{name}: transcribed {n:,}, the report prints {got:,}")
    print(f"  NBS 2022 Table 1: all {len(want)} figures re-read from PDF page {PDF_PAGE} and equal")


def main():
    if "--fetch" in sys.argv:
        fetch()

    check_pdf()
    main_sum = sum(v for k, v in CENSUS_2022.items() if k not in ZANZIBAR)
    zan_sum = sum(CENSUS_2022[k] for k in ZANZIBAR)
    if (main_sum, zan_sum) != (MAINLAND_2022, ZANZIBAR_2022):
        raise SystemExit(f"the transcription sums to {main_sum:,} mainland and {zan_sum:,} "
                         "Zanzibar, not the table's own subtotals")

    shp = os.path.join(RAW, "adm1", "tza_admbnda_adm1_20181019.shp")
    if not os.path.exists(shp):
        raise SystemExit(f"missing {shp}; run with --fetch first")
    # engine="fiona": pyogrio is the engine that has silently returned zero features here.
    g = gpd.read_file(shp, engine="fiona")
    if len(g) != N_REGIONS:
        raise SystemExit(f"{shp} has {len(g)} features, expected {N_REGIONS}")
    print(f"COD-AB Tanzania admin1: {len(g)} regions, crs={g.crs}")

    by_fold = {fold(k): k for k in CENSUS_2022}
    if len(by_fold) != N_REGIONS:
        raise SystemExit("two census names fold to the same key")
    g["census_name"] = g["ADM1_EN"].map(lambda s: by_fold.get(fold(s)))
    miss = sorted(g.loc[g["census_name"].isna(), "ADM1_EN"])
    if miss:
        raise SystemExit(f"COD-AB regions with no census row: {miss}")
    if g["census_name"].nunique() != N_REGIONS:
        raise SystemExit("two COD-AB regions joined the same census row")
    differ = g.loc[g["ADM1_EN"] != g["census_name"], ["ADM1_PCODE", "ADM1_EN", "census_name"]]
    print(f"  COD-AB x NBS 2022: {N_REGIONS} of {N_REGIONS} joined; spelled differently: "
          + ", ".join(f"{r.ADM1_EN!r} = {r.census_name!r}" for r in differ.itertuples()))
    if set(g.loc[g["ADM1_PCODE"].str.startswith("TZ5"), "census_name"]) != set(ZANZIBAR):
        raise SystemExit("the TZ5x pcodes are not the five Zanzibar regions")

    g["pop"] = g["census_name"].map(CENSUS_2022).astype("int64")
    g["unit"] = g["ADM1_PCODE"].map(lambda c: MERGE.get(c, c))
    g["name"] = g.apply(lambda r: MERGED_NAME.get(r["unit"], r["census_name"]), axis=1)
    g["zanzibar"] = g["census_name"].isin(ZANZIBAR)

    u = g[["unit", "name", "pop", "zanzibar", "geometry"]].dissolve(
        by="unit", aggfunc={"name": "first", "pop": "sum", "zanzibar": "first"}).reset_index()
    if len(u) != N_UNITS:
        raise SystemExit(f"{len(u)} units after the Mbeya/Songwe dissolve, expected {N_UNITS}")
    if int(u["pop"].sum()) != TOTAL_2022:
        raise SystemExit("the units do not sum to the census total")
    u["geo_id"] = u["unit"]
    u["area_sqkm"] = u.to_crs("EPSG:6933").area / 1e6

    print(f"\n  {len(u)} units, {int(u['pop'].sum()):,} people (NBS 2022 census):")
    for _i, r in u.sort_values("pop", ascending=False).iterrows():
        print(f"    {r['unit']}  {r['name']:<22}{int(r['pop']):>11,}  {r['area_sqkm']:>9,.0f} km2"
              + ("  Zanzibar" if r["zanzibar"] else ""))

    # ---- district names, for sources/tz.py's round-4 decode ----
    shp2 = os.path.join(RAW, "adm2", "tza_admbnda_adm2_20181019.shp")
    d = gpd.read_file(shp2, engine="fiona", ignore_geometry=True)
    for c in ("ADM2_EN", "ADM1_PCODE"):
        if c not in d.columns:
            raise SystemExit(f"{shp2} has no {c}: {list(d.columns)}")
    d = d.rename(columns={"ADM2_EN": "district", "ADM1_PCODE": "region_pcode"})
    d["unit"] = d["region_pcode"].map(lambda c: MERGE.get(c, c))
    bad = sorted(set(d["unit"]) - set(u["unit"]))
    if bad:
        raise SystemExit(f"districts whose region is not a unit: {bad}")
    print(f"\n  COD-AB admin2: {len(d)} districts over {d['region_pcode'].nunique()} regions")

    os.makedirs(OUT_DIR, exist_ok=True)
    keep = u[["geo_id", "unit", "name", "pop", "zanzibar", "area_sqkm", "geometry"]]
    keep.to_file(OUT, layer="regions", driver="GPKG")
    pd.DataFrame(keep.drop(columns="geometry")).to_csv(LOOKUP, index=False, encoding="utf-8")
    d[["district", "region_pcode", "unit"]].sort_values(["unit", "district"]).to_csv(
        DISTRICTS, index=False, encoding="utf-8")
    print(f"\nwrote {OUT}\nwrote {LOOKUP}\nwrote {DISTRICTS}")


if __name__ == "__main__":
    main()
