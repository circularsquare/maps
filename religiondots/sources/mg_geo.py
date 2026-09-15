"""Madagascar — boundaries and 2018 census populations for the 22 regions.

Writes data/geo/mg/mg_regions.gpkg, data/geo/mg/mg_lookup.csv and data/geo/mg/mg_districts.csv.

  * **boundaries**: OCHA COD-AB Madagascar (`cod-ab-mdg`), the edition of 2026-08-13 (BNGRC
    source, reviewed March 2024), `mdg_admin1` and `mdg_admin2`, read with `engine="fiona"`. The
    older 2018-10-31 shapefile zip now returns 404 on HDX. `mgd_op_adm1_old_names_pcodes` gives
    each region's pre-2007 province, which `sources/mg.py` checks round 9's location column
    against.
  * **populations**: INSTAT, *Résultats globaux du RGPH-3 de 2018 de Madagascar, Tome 1*
    (December 2020), **Tableau 6** "Répartition de la population résidente de Madagascar par
    milieu de résidence et taux d'urbanisation selon la région", report page 17 (PDF page 49).
    25,674,196 people. Transcribed below and re-read from the PDF on every run. UNFPA
    Madagascar's copy is fetched, because `instat.mg` is a Cloudflare challenge (§11aq); a
    Wayback capture of INSTAT's own upload (`INSTAT_RGPH3-Definitif-ResultatsGlogaux-Tome1_17-2021.pdf`)
    exists as a fallback.

## COD-PS IS NOT THE CENSUS

COD-PS Madagascar (`cod-ps-mdg`) is labelled 2018, but its HDX caveat says it is the 2009 BNGRC
population grown forward, "rounding of decimal places from the population growth estimate
calculations for 2010 to 2018". The census counted every region that year, so the office's count
is the base, on Ecuador's reasoning (§9bn).

## THE FILE HAS 24 REGIONS; THE CENSUS AND THE SURVEY HAVE 22

COD-AB carries Vatovavy (MG26) and Fitovinany (MG27) apart, and Ambatosoa (MG34, the districts of
Mananara-Avaratra and Maroantsetra). The 2018 census prints Vatovavy Fitovinany as one region and
has no Ambatosoa, and every Afrobarometer round but the last does the same. So MG27 is dissolved
into MG26 and MG34 into Analanjirofo (MG32).

The witness is the census itself: its table of urban communes by region (PDF page 46) lists
Maroantsetra and Mananara-Avaratra under Analanjirofo, and Manakara Atsimo, Mananjary, Ifanadiana,
Ikongo and Vohipeno under Vatovavy Fitovinany. `check_dissolve()` reads that page and asserts it,
and asserts that COD-AB files the same districts under the regions being dissolved.

## THE JOIN

COD-AB and the census spell two regions differently (`Amoron I Mania` against `Amoron'i Mania`,
and the dissolved `Vatovavy` against `Vatovavy Fitovinany`). The join is on letters only and is
asserted to be a bijection over all 22, in both directions ([[reference_name_join_wrong_neighbour]]).

Usage:
    python sources/mg_geo.py --fetch    one zip from HDX (~130 MB) and the census report (~8 MB)
    python sources/mg_geo.py            rebuild from data/raw/mg/
"""

import os
import re
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "mg")
COD = os.path.join(RAW, "cod")
OUT_DIR = os.path.join(ROOT, "data", "geo", "mg")
OUT = os.path.join(OUT_DIR, "mg_regions.gpkg")
LOOKUP = os.path.join(OUT_DIR, "mg_lookup.csv")
DISTRICTS = os.path.join(OUT_DIR, "mg_districts.csv")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")

ZIP_NAME = "mdg_admin_boundaries.shp.zip"
ZIP_URL = ("https://data.humdata.org/dataset/26fa506b-0727-4d9d-a590-d2abee21ee22/resource/"
           "d5ede998-21e3-437b-9157-dc16593b44eb/download/mdg_admin_boundaries.shp.zip")
ZIP_MEMBERS = ("mdg_admin1", "mdg_admin2", "mgd_op_adm1_old_names_pcodes")

PDF_NAME = "resultat_globaux_rgph3_tome_01.pdf"
PDF_URL = "https://madagascar.unfpa.org/sites/default/files/pub-pdf/resultat_globaux_rgph3_tome_01.pdf"
PDF_PAGE = 49                   # 1-based; the report's own page 17
PDF_TITLE = "Tableau 6."
PDF_COMMUNE_PAGE = 46           # urban communes by region, continued

EXPECTED_PCODES = {
    "MG11", "MG12", "MG13", "MG14", "MG21", "MG22", "MG24", "MG25", "MG26", "MG27",
    "MG31", "MG32", "MG33", "MG34", "MG41", "MG42", "MG43", "MG44",
    "MG51", "MG52", "MG53", "MG54", "MG71", "MG72",
}
N_UNITS = 22
MERGE = {"MG27": "MG26", "MG34": "MG32"}
MERGED_NAME = {"MG26": "Vatovavy Fitovinany"}

TOTAL_2018 = 25_674_196
URBAN_2018 = 4_942_902

# Tableau 6, "Ensemble", and "Urbain", in the table's own order. The apostrophe in Amoron'i Mania
# is typographic in the report; the join and the PDF check both compare letters only.
CENSUS_2018 = {
    "Analamanga": (3_623_925, 1_371_135), "Vakinankaratra": (2_079_659, 312_981),
    "Itasy": (898_549, 151_431), "Bongolava": (670_993, 44_461),
    "Haute Matsiatra": (1_444_587, 246_613), "Amoron'i Mania": (837_116, 107_719),
    "Vatovavy Fitovinany": (1_440_657, 136_575), "Ihorombe": (417_312, 39_556),
    "Atsimo Atsinanana": (1_030_404, 73_213), "Atsinanana": (1_478_472, 407_358),
    "Analanjirofo": (1_150_089, 181_983), "Alaotra Mangoro": (1_249_931, 175_261),
    "Boeny": (929_312, 333_096), "Sofia": (1_507_591, 182_041),
    "Betsiboka": (393_278, 50_899), "Melaky": (308_944, 33_624),
    "Atsimo Andrefana": (1_797_894, 254_993), "Androy": (900_235, 86_317),
    "Anosy": (809_051, 130_600), "Menabe": (692_463, 112_218),
    "Diana": (889_962, 302_238), "Sava": (1_123_772, 208_590),
}

# The census's urban-commune table, by region, for the two dissolves (see the docstring).
COMMUNE_WITNESS = {
    ("Analanjirofo", "Alaotra Mangoro"): ["Maroantsetra", "Mananara-Avaratra"],
    ("Vatovavy Fitovinany", "Ihorombe"): ["Manakara Atsimo", "Mananjary", "Ifanadiana",
                                          "Ikongo", "Vohipeno"],
}
# COD-AB districts under the dissolved regions, which the census files under the target unit.
DISTRICT_WITNESS = {
    "MG34": {"Maroantsetra", "Mananara-Avaratra"},
    "MG26": {"Mananjary", "Ifanadiana"},
    "MG27": {"Manakara Atsimo", "Ikongo", "Vohipeno"},
}


def fold(s):
    return re.sub(r"[^a-z]", "", str(s).casefold())


def fetch():
    os.makedirs(COD, exist_ok=True)
    dst = os.path.join(RAW, ZIP_NAME)
    if not (os.path.exists(dst) and os.path.getsize(dst) > 10_000_000):
        req = urllib.request.Request(ZIP_URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=1800) as r:
            data = r.read()
        if data[:2] != b"PK":                           # §5a: a 200 is not a download
            raise SystemExit(f"{ZIP_NAME} is not a zip, starts {data[:16]!r}")
        with open(dst + ".part", "wb") as f:
            f.write(data)
        os.replace(dst + ".part", dst)
        print(f"  got  {ZIP_NAME} ({os.path.getsize(dst):,} bytes)")
    else:
        print(f"  have {ZIP_NAME} ({os.path.getsize(dst):,} bytes)")
    with zipfile.ZipFile(dst) as z:
        members = [n for n in z.namelist() if n.split(".")[0] in ZIP_MEMBERS]
        if len(members) != 5 * len(ZIP_MEMBERS):
            raise SystemExit(f"{ZIP_NAME} has {members}, expected five files for each of "
                             f"{ZIP_MEMBERS}")
        for n in members:
            z.extract(n, COD)

    pdf = os.path.join(RAW, PDF_NAME)
    if not (os.path.exists(pdf) and os.path.getsize(pdf) > 1_000_000):
        import requests

        r = requests.get(PDF_URL, headers={"User-Agent": UA, "Accept": "application/pdf,*/*"},
                         timeout=600)
        r.raise_for_status()
        if r.content[:4] != b"%PDF" or b"%%EOF" not in r.content[-2048:]:
            raise SystemExit("the RGPH-3 Tome 1 is not a complete PDF (no %PDF header or no "
                             "%%EOF trailer); [[reference_pdf_truncated_at_source]]")
        with open(pdf + ".part", "wb") as f:
            f.write(r.content)
        os.replace(pdf + ".part", pdf)
        print(f"  got  {PDF_NAME} ({os.path.getsize(pdf):,} bytes)")
    else:
        print(f"  have {PDF_NAME}")


def _lines(doc, page):
    return [ln.strip() for ln in doc[page - 1].get_text().splitlines() if ln.strip()]


def _int(s):
    return int(re.sub(r"\s", "", s))


def check_pdf():
    """Re-read Tableau 6 from the report and assert every transcribed figure against it."""
    import fitz

    doc = fitz.open(os.path.join(RAW, PDF_NAME))
    if doc.page_count != 192:
        raise SystemExit(f"{PDF_NAME} has {doc.page_count} pages, not 192; a truncated download "
                         "reads as a short document ([[reference_pdf_truncated_at_source]])")
    lines = _lines(doc, PDF_PAGE)
    if PDF_TITLE not in lines:
        raise SystemExit(f"PDF page {PDF_PAGE} has no {PDF_TITLE!r}; the report has been re-laid")
    start = lines.index(PDF_TITLE)
    body = lines[start:]
    want = dict(CENSUS_2018)
    want["MADAGASCAR"] = (TOTAL_2018, URBAN_2018)
    for name, (total, urban) in want.items():
        at = [i for i, ln in enumerate(body) if fold(ln) == fold(name)]
        if len(at) != 1:
            raise SystemExit(f"{name!r} appears {len(at)} times in Tableau 6")
        u, r, t = (_int(body[at[0] + k]) for k in (1, 2, 3))
        if (t, u) != (total, urban):
            raise SystemExit(f"{name}: transcribed {total:,} ({urban:,} urban), the report prints "
                             f"{t:,} ({u:,} urban)")
        if u + r != t:
            raise SystemExit(f"{name}: urban {u:,} + rural {r:,} is not the printed {t:,}")
    print(f"  RGPH-3 Tableau 6: all {len(want)} rows re-read from PDF page {PDF_PAGE}; urban plus "
          "rural closes on every one")
    return doc


def check_dissolve(doc, d):
    """The census's urban-commune table puts the dissolved regions' towns in the target unit."""
    lines = _lines(doc, PDF_COMMUNE_PAGE)
    folded = [fold(x) for x in lines]
    for (region, next_region), towns in COMMUNE_WITNESS.items():
        a, b = folded.index(fold(region)), folded.index(fold(next_region))
        block = set(folded[a:b])
        missing = [t for t in towns if fold(t) not in block]
        if missing:
            raise SystemExit(f"the census's urban-commune table does not list {missing} under "
                             f"{region}; the dissolve needs re-reading")
        print(f"  census urban communes under {region}: {', '.join(towns)} (PDF page "
              f"{PDF_COMMUNE_PAGE})")
    for pcode, names in DISTRICT_WITNESS.items():
        have = {fold(n) for n in d.loc[d["adm1_pcode"] == pcode, "adm2_name"]}
        missing = sorted(n for n in names if fold(n) not in have)
        if missing:
            raise SystemExit(f"COD-AB does not file {missing} under {pcode}")
    print("  COD-AB files the same districts under MG26, MG27 and MG34")


def main():
    if "--fetch" in sys.argv:
        fetch()

    doc = check_pdf()
    if sum(t for t, _u in CENSUS_2018.values()) != TOTAL_2018:
        raise SystemExit("the transcription does not sum to the table's own total")

    shp = os.path.join(COD, "mdg_admin1.shp")
    if not os.path.exists(shp):
        raise SystemExit(f"missing {shp}; run with --fetch first")
    # engine="fiona": pyogrio is the engine that has silently returned zero features here.
    g = gpd.read_file(shp, engine="fiona")
    if set(g["adm1_pcode"]) != EXPECTED_PCODES or len(g) != len(EXPECTED_PCODES):
        raise SystemExit(f"{shp}: {len(g)} features, pcodes {sorted(g['adm1_pcode'])}; expected "
                         f"{sorted(EXPECTED_PCODES)}")
    print(f"COD-AB Madagascar admin1: {len(g)} regions, crs={g.crs}")

    d = gpd.read_file(os.path.join(COD, "mdg_admin2.shp"), engine="fiona", ignore_geometry=True)
    for c in ("adm2_name", "adm2_pcode", "adm1_pcode"):
        if c not in d.columns:
            raise SystemExit(f"mdg_admin2 has no {c}: {list(d.columns)}")
    check_dissolve(doc, d)

    prov = gpd.read_file(os.path.join(COD, "mgd_op_adm1_old_names_pcodes.shp"), engine="fiona",
                         ignore_geometry=True)
    province = dict(zip(prov["adm1_pcode"], prov["old_provin"]))
    for child, parent in MERGE.items():
        if province[child] != province[parent]:
            raise SystemExit(f"{child} and {parent} are in different old provinces")

    g["unit"] = g["adm1_pcode"].map(lambda c: MERGE.get(c, c))
    keep_name = dict(zip(g["adm1_pcode"], g["adm1_name"]))
    u = g[["unit", "geometry"]].dissolve(by="unit").reset_index()
    if len(u) != N_UNITS:
        raise SystemExit(f"{len(u)} units after the dissolve, expected {N_UNITS}")
    u["cod_name"] = u["unit"].map(lambda c: MERGED_NAME.get(c, keep_name[c]))

    by_fold = {fold(k): k for k in CENSUS_2018}
    if len(by_fold) != N_UNITS:
        raise SystemExit("two census names fold to the same key")
    u["name"] = u["cod_name"].map(lambda s: by_fold.get(fold(s)))
    miss = sorted(u.loc[u["name"].isna(), "cod_name"])
    if miss:
        raise SystemExit(f"COD-AB units with no census row: {miss}")
    if u["name"].nunique() != N_UNITS:
        raise SystemExit("two COD-AB units joined the same census row")
    differ = u.loc[u["cod_name"] != u["name"], ["unit", "cod_name", "name"]]
    print(f"  COD-AB x RGPH-3: {N_UNITS} of {N_UNITS} joined; spelled differently: "
          + ", ".join(f"{r.cod_name!r} = {r.name!r}" for r in differ.itertuples()))

    u["pop"] = u["name"].map(lambda n: CENSUS_2018[n][0]).astype("int64")
    u["province"] = u["unit"].map(province)
    if int(u["pop"].sum()) != TOTAL_2018:
        raise SystemExit("the units do not sum to the census total")
    u["geo_id"] = u["unit"]
    u["area_sqkm"] = u.to_crs("EPSG:6933").area / 1e6

    print(f"\n  {len(u)} units, {int(u['pop'].sum()):,} people (RGPH-3 2018):")
    for _i, r in u.sort_values("pop", ascending=False).iterrows():
        print(f"    {r['unit']}  {r['name']:<22}{int(r['pop']):>11,}  {r['area_sqkm']:>9,.0f} km2"
              f"  {r['province']}")

    d = d.rename(columns={"adm2_name": "district", "adm2_pcode": "district_pcode"})
    d["unit"] = d["adm1_pcode"].map(lambda c: MERGE.get(c, c))
    bad = sorted(set(d["unit"]) - set(u["unit"]))
    if bad:
        raise SystemExit(f"districts whose region is not a unit: {bad}")
    print(f"\n  COD-AB admin2: {len(d)} districts over {d['adm1_pcode'].nunique()} regions")

    os.makedirs(OUT_DIR, exist_ok=True)
    keep = u[["geo_id", "unit", "name", "pop", "province", "area_sqkm", "geometry"]]
    keep.to_file(OUT, layer="regions", driver="GPKG")
    pd.DataFrame(keep.drop(columns="geometry")).to_csv(LOOKUP, index=False, encoding="utf-8")
    d[["district", "district_pcode", "unit"]].sort_values(["unit", "district"]).to_csv(
        DISTRICTS, index=False, encoding="utf-8")
    print(f"\nwrote {OUT}\nwrote {LOOKUP}\nwrote {DISTRICTS}")


if __name__ == "__main__":
    main()
