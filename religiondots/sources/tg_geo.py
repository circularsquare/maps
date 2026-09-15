"""Togo — 6 units (the 5 regions, with Lomé apart), the 2022 census populations, and the grid.

Writes:
    data/geo/tg/tg_units.gpkg         the 6 units (`units`)
    data/geo/tg/tg_hexes.gpkg         Kontur 400 m hexes with `unit` and `pop` (`place`)
    data/geo/tg/tg_lookup.csv         unit -> name, 2022 census population, Kontur population
    data/geo/tg/tg_prefectures.csv    COD-AB's 40 prefectures and their region, for `sources/tg.py`'s
                                      check of the survey's prefecture column against its region

Usage:
    python sources/tg_geo.py --fetch    COD-AB shapefile zip (~2 MB), Kontur (~3.5 MB), Livret 01 (~2 MB)
    python sources/tg_geo.py            rebuild from data/raw/tg/

## WHY 6 UNITS AND NOT 5

Every Afrobarometer round that reached Togo (5 to 9) samples the commune of Lomé as a stratum of
its own (`Lomé commune`, `Lome Commune`, `Lome`, `LOME`, `LOME Commune`), and the location column
of rounds 6, 7 and 9 puts those respondents in Arrondissements I to V and nobody else there. The
rest of Golfe prefecture is sampled inside `Maritime` (round 9: 160 of Maritime's 368). So the
survey measures the city apart from its region, and drawing Maritime as one unit would pour Lomé's
mix over the farms of Zio and Yoto.

## THE LOMÉ UNIT IS GOLFE 1 TO 5, ON COD-AB'S COMMUNE PLUS TWO CANTONS

The commune of Lomé is not a unit of the 2022 census. Since the 2017 communal law (loi 2017-08)
Grand Lomé is 13 communes, and Livret 01's Tableau 4 counts them. Golfe 1 to 5 are the old city's
cantons (Bè, Amoutivé and Aflao-Gakli, whose quartiers Tableau 10 lists: Bè, Tokoin, Nyékonakpoè,
Hédzranawoé, Totsi, Agbalépédogan); Golfe 6 is the Baguida canton and Golfe 7 Aflao-Sagbado, both
outside the old commune. So the Lomé unit's people are Golfe 1-5, **866,307**.

COD-AB v02 draws the old commune (ADM2 `Lome Commune`, 104.6 km2) and leaves two canton pieces in
Golfe under the same names, `Aflao Gakli` (11.0 km2) and `Amoutive` (23.6 km2), both on the
commune's northern edge. Golfe 4 and Golfe 5 take their whole cantons, so the unit's polygon is the
commune plus those two pieces. Kontur is the witness (it counts nothing here): against Golfe 1-5 it
reads **1.12x** on the three pieces and **0.90x** on the commune polygon alone, where the national
ratio is 1.13x. Asserted (`LOME_KONTUR_TOL`).

## BOUNDARIES: OCHA COD-AB TOGO (`cod-ab-tgo`), VERSION 02

5 regions, 40 prefectures, 373 cantons; boundaries created 12 January 2020, reviewed 30 October
2025. The units are dissolved from the cantons, so Lomé and Maritime share edges exactly, and each
dissolved region is asserted against COD's own region polygon by area.

## POPULATION: THE 2022 CENSUS (RGPH-5), NOT COD-PS

INSEED, *Distribution spatiale de la population résidente par sexe* (Livret 01, September 2023),
Tableau 2 (regions) and Tableau 4 (the communes of Grand Lomé), re-read from the PDF on every run.
COD-PS Togo is a 2021 projection from the 2010 census, and the religion margin in `sources/tg.py`
is the 2022 census, so the population has to be the same count (`sources/lr.md` §3, Liberia's
reason). COD-PS's projection puts Lomé Commune at 1,100,387 in 2021; the 2022 count of Golfe 1-5 is
866,307, so the projection's error sits exactly on the unit this build separates.
"""

import os
import re
import sys
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "tg")
SHP_DIR = os.path.join(RAW, "shp")
GEO = os.path.join(ROOT, "data", "geo", "tg")
OUT_UNITS = os.path.join(GEO, "tg_units.gpkg")
OUT_HEXES = os.path.join(GEO, "tg_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "tg_lookup.csv")
OUT_PREFS = os.path.join(GEO, "tg_prefectures.csv")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")

LIVRET01 = "rgph5_livret01_distribution_spatiale.pdf"
KONTUR_GPKG = "kontur_population_TG_20231101.gpkg"
DOWNLOADS = {
    "tgo_admin_boundaries.shp.zip": (
        "https://data.humdata.org/dataset/e6439952-e487-4682-8251-e9688aac60e1/resource/"
        "fcde37fa-154e-4ba4-a703-5a229a8e877e/download/tgo_admin_boundaries.shp.zip",
        b"PK", 1_000_000),
    KONTUR_GPKG + ".gz": (
        "https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
        "kontur_population_TG_20231101.gpkg.gz", b"\x1f\x8b", 1_000_000),
    # INSEED's Download Monitor id 6616, "RGPH-5 Distribution spatiale de la population résidente
    # par sexe (Livret 01)".
    LIVRET01: ("https://inseed.tg/download/6616/", b"%PDF", 1_000_000),
}

N_REGIONS = 5
N_PREFECTURES = 40
N_CANTONS = 373
N_UNITS = 6
CENSUS_2022 = 8_095_498

LOME = "TG0305"
LOME_ADM2 = "TG0305"
# COD-AB's two Golfe canton pieces that belong to Golfe 4 (Amoutivé) and Golfe 5 (Aflao-Gakli).
LOME_CANTONS = {"TG030301": "Aflao Gakli", "TG030302": "Amoutive"}
UNIT_NAME = {"TG0305": "Lomé (Golfe 1 to 5)", "TG03": "Maritime", "TG04": "Plateaux",
             "TG01": "Centrale", "TG02": "Kara", "TG05": "Savanes"}

# Livret 01, Tableau 2, PDF page 23: region, (men, women, both).
T2_PAGE = 23
TABLEAU_2 = {
    "MARITIME (GRAND LOME INCLUS)": (1_703_380, 1_831_611, 3_534_991),
    "PLATEAUX": (806_154, 829_792, 1_635_946),
    "CENTRALE": (397_336, 398_193, 795_529),
    "KARA": (488_225, 497_287, 985_512),
    "SAVANES": (549_415, 594_105, 1_143_520),
    "TOGO": (3_944_510, 4_150_988, 8_095_498),
}
T2_PCODE = {"MARITIME (GRAND LOME INCLUS)": "TG03", "PLATEAUX": "TG04", "CENTRALE": "TG01",
            "KARA": "TG02", "SAVANES": "TG05"}
# Livret 01, Tableau 4, PDF page 26: the communes of Golfe that make the Lomé unit.
T4_PAGE = 26
TABLEAU_4 = {
    "GOLFE 1": (171_322, 180_228, 351_550),
    "GOLFE 2": (65_558, 70_595, 136_153),
    "GOLFE 3": (26_480, 26_289, 52_769),
    "GOLFE 4": (76_501, 79_341, 155_842),
    "GOLFE 5": (80_221, 89_772, 169_993),
}

# A dissolved region against COD's own region polygon, in km2 (equal-area).
AREA_TOL = 0.005
# Kontur 2023-11 against the 2022 census. First build (2026-09-15): 1.128 nationally, every unit
# between 1.10 (Savanes) and 1.15 (Plateaux). The band sits well outside that.
KONTUR_UNIT_BAND = (0.95, 1.35)
# The Lomé unit's Kontur ratio must sit within this of the national one; the commune polygon alone
# reads 0.90 against 1.14 and would fail, which is the reason for the two canton pieces.
LOME_KONTUR_TOL = 0.05


def fold(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.casefold())


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for name, (url, magic, min_size) in DOWNLOADS.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst) and os.path.getsize(dst) > min_size:
            print(f"  have {name} ({os.path.getsize(dst):,} bytes)")
            continue
        r = requests.get(url, headers={"User-Agent": UA}, timeout=900)
        r.raise_for_status()
        if not r.content.startswith(magic):                    # §5a: a 200 is not a download
            raise SystemExit(f"{name} starts {r.content[:16]!r}, not {magic!r}")
        if magic == b"%PDF" and b"%%EOF" not in r.content[-2048:]:
            raise SystemExit(f"{name} has no %%EOF trailer ([[reference_pdf_truncated_at_source]])")
        with open(dst + ".part", "wb") as f:
            f.write(r.content)
        os.replace(dst + ".part", dst)
        print(f"  got  {name} ({len(r.content):,} bytes)")


def unpack():
    import gzip
    import shutil

    z = os.path.join(RAW, "tgo_admin_boundaries.shp.zip")
    if not os.path.exists(z):
        raise SystemExit(f"missing {z}; run with --fetch first")
    if not os.path.exists(os.path.join(SHP_DIR, "tgo_admin3.shp")):
        with zipfile.ZipFile(z) as zz:
            zz.extractall(SHP_DIR)
    gz = os.path.join(RAW, KONTUR_GPKG + ".gz")
    gpkg = os.path.join(RAW, KONTUR_GPKG)
    if not os.path.exists(gpkg):
        if not os.path.exists(gz):
            raise SystemExit(f"missing {gz}; run with --fetch first")
        with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")
    return gpkg


def _num(n):
    """8095498 -> '8 095 498', the way Livret 01 prints it."""
    return f"{n:,}".replace(",", " ")


def read_census():
    """The 6 units' 2022 populations, every figure re-read from Livret 01."""
    import fitz

    pdf = os.path.join(RAW, LIVRET01)
    if not os.path.exists(pdf):
        raise SystemExit(f"missing {pdf}; run with --fetch")
    doc = fitz.open(pdf)
    if doc.page_count != 102:
        raise SystemExit(f"Livret 01 has {doc.page_count} pages, expected 102; a truncated or "
                         "different file")
    for page, title, table in ((T2_PAGE, "Tableau 2", TABLEAU_2), (T4_PAGE, "Tableau 4", TABLEAU_4)):
        text = " ".join(doc[page - 1].get_text().split())
        if title not in text:
            raise SystemExit(f"PDF page {page} is not {title}")
        for name, (m, f, t) in table.items():
            if m + f != t:
                raise SystemExit(f"{title} {name}: {m:,} + {f:,} is not {t:,}")
            if f"{name} {_num(m)} {_num(f)} {_num(t)}" not in text:
                raise SystemExit(f"{title} {name}: transcribed {m:,} {f:,} {t:,}, not on PDF page {page}")
    regions = {T2_PCODE[k]: v[2] for k, v in TABLEAU_2.items() if k in T2_PCODE}
    if sum(regions.values()) != TABLEAU_2["TOGO"][2] or TABLEAU_2["TOGO"][2] != CENSUS_2022:
        raise SystemExit("Tableau 2's regions do not sum to the census total")
    lome = sum(v[2] for v in TABLEAU_4.values())
    pop = dict(regions)
    pop[LOME] = lome
    pop["TG03"] -= lome
    print(f"Livret 01 re-read: Tableau 2 (5 regions, {CENSUS_2022:,}) and Tableau 4 (Golfe 1-5, "
          f"{lome:,}) equal their transcriptions")
    return pop


def main():
    import geopandas as gpd
    import pandas as pd

    import geo_checks

    if "--fetch" in sys.argv:
        fetch()
    gpkg = unpack()
    pop = read_census()

    a1 = gpd.read_file(os.path.join(SHP_DIR, "tgo_admin1.shp"), engine="fiona")
    a2 = gpd.read_file(os.path.join(SHP_DIR, "tgo_admin2.shp"), engine="fiona")
    a3 = gpd.read_file(os.path.join(SHP_DIR, "tgo_admin3.shp"), engine="fiona")
    if (len(a1), len(a2), len(a3)) != (N_REGIONS, N_PREFECTURES, N_CANTONS):
        raise SystemExit(f"COD-AB has {len(a1)}/{len(a2)}/{len(a3)} features, expected "
                         f"{N_REGIONS}/{N_PREFECTURES}/{N_CANTONS}")
    print(f"COD-AB Togo: {len(a1)} regions, {len(a2)} prefectures, {len(a3)} cantons, version "
          f"{a1['version'].iloc[0]}, valid_on {a1['valid_on'].iloc[0]}, crs={a1.crs}")
    names = {r.adm1_pcode: fold(r.adm1_name) for r in a1.itertuples()}
    want = {"TG01": "centrale", "TG02": "kara", "TG03": "maritime", "TG04": "plateaux", "TG05": "savanes"}
    if names != want:
        raise SystemExit(f"COD-AB region pcodes are not {want}: {names}")
    row = a2[a2["adm2_pcode"] == LOME_ADM2]
    if len(row) != 1 or fold(row["adm2_name"].iloc[0]) != "lomecommune":
        raise SystemExit(f"COD-AB prefecture {LOME_ADM2} is not Lome Commune")
    for pc, nm in LOME_CANTONS.items():
        r = a3[a3["adm3_pcode"] == pc]
        if len(r) != 1 or fold(r["adm3_name"].iloc[0]) != fold(nm) or fold(r["adm2_name"].iloc[0]) != "golfe":
            raise SystemExit(f"COD-AB canton {pc} is not {nm} in Golfe")

    in_lome = (a3["adm2_pcode"] == LOME_ADM2) | a3["adm3_pcode"].isin(LOME_CANTONS)
    a3["unit"] = a3["adm1_pcode"].where(~in_lome, LOME)
    units = a3[["unit", "geometry"]].dissolve(by="unit").reset_index()
    if len(units) != N_UNITS:
        raise SystemExit(f"{len(units)} units after the dissolve, expected {N_UNITS}")
    eq = "EPSG:6933"
    units["area_sqkm"] = units.to_crs(eq).area / 1e6
    reg_area = a1.set_index("adm1_pcode").to_crs(eq).area / 1e6
    back = units.assign(region=units["unit"].map(lambda u: "TG03" if u == LOME else u)) \
        .groupby("region")["area_sqkm"].sum()
    bad = {pc: (round(back[pc], 1), round(reg_area[pc], 1)) for pc in reg_area.index
           if abs(back[pc] / reg_area[pc] - 1) > AREA_TOL}
    if bad:
        raise SystemExit(f"dissolved cantons do not rebuild COD's regions by area: {bad}")
    print(f"  the 373 cantons dissolve into {N_UNITS} units and rebuild every region's area within "
          f"{AREA_TOL:.1%}")

    units["geo_id"] = units["unit"]
    units["name"] = units["unit"].map(UNIT_NAME)
    units["pop"] = units["unit"].map(pop).astype("int64")
    if units["name"].isna().any() or int(units["pop"].sum()) != CENSUS_2022:
        raise SystemExit("a unit has no name, or the units do not sum to the 2022 census")

    # ---- Kontur ----
    hexes = geo_checks.read_layer(gpkg, "Kontur TG")
    popcol = next(c for c in hexes.columns if c.lower() == "population")
    pts = gpd.GeoDataFrame({"pop": hexes[popcol].to_numpy(dtype=float)},
                           geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs(units.crs)
    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, "pop"].sum())
    print(f"\nKontur hexes: {len(hexes):,}, population {pts['pop'].sum():,.0f}; "
          f"{int(outside.sum()):,} centroids outside every unit ({lost:,.0f} people, "
          f"{100 * lost / pts['pop'].sum():.3f}%) dropped")
    keep = ~outside
    out = gpd.GeoDataFrame({"unit": joined.loc[keep, "unit"].to_numpy(),
                            "pop": pts.loc[keep, "pop"].to_numpy()},
                           geometry=hexes.to_crs(units.crs).geometry[keep.to_numpy()].to_numpy(),
                           crs=units.crs)
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"units with no populated hex: {missing}")
    tot = float(out["pop"].sum())
    national = tot / CENSUS_2022
    print(f"  Kontur {tot:,.0f} against the 2022 census {CENSUS_2022:,}: ratio {national:.3f}")
    units["kontur_pop"] = units["unit"].map(per["sum"]).round().astype("int64")
    units["hexes"] = units["unit"].map(per["size"]).astype("int64")
    print("\n  per unit, the 2022 census against Kontur (the witness; Kontur counts nothing here):")
    for r in units.sort_values("pop", ascending=False).itertuples():
        print(f"    {r.unit:<7} {r.name:<22}{r.pop:>11,}  Kontur {r.kontur_pop:>11,}  "
              f"{r.kontur_pop / r.pop:5.2f}x  {r.hexes:>7,} hexes  {r.area_sqkm:>9,.0f} km2")
    geo_checks.ratio_band(dict(zip(units["unit"], units["pop"])),
                          dict(zip(units["unit"], units["kontur_pop"])),
                          *KONTUR_UNIT_BAND, what="unit")

    # the Lomé definition: commune plus the two canton pieces, against the commune polygon alone
    commune = a3[a3["adm2_pcode"] == LOME_ADM2][["geometry"]]
    jc = gpd.sjoin(pts, commune.to_crs(pts.crs), how="inner", predicate="within")
    commune_only = float(pts.loc[jc.index.unique(), "pop"].sum())
    lome_ratio = float(per.loc[LOME, "sum"]) / pop[LOME]
    print(f"\n  Lomé against Golfe 1-5 ({pop[LOME]:,}): Kontur {lome_ratio:.3f}x on the commune plus "
          f"Aflao Gakli and Amoutivé, {commune_only / pop[LOME]:.3f}x on the commune polygon alone; "
          f"national {national:.3f}x")
    if abs(lome_ratio / national - 1) > LOME_KONTUR_TOL:
        raise SystemExit("the Lomé unit's Kontur ratio has left the national one; re-read the "
                         "docstring's argument for the two canton pieces")

    # ---- write ----
    os.makedirs(GEO, exist_ok=True)
    keep_cols = ["geo_id", "unit", "name", "pop", "kontur_pop", "hexes", "area_sqkm", "geometry"]
    units[keep_cols].to_file(OUT_UNITS, layer="units", driver="GPKG")
    pd.DataFrame(units[keep_cols].drop(columns="geometry")).to_csv(OUT_LOOKUP, index=False,
                                                                  encoding="utf-8")
    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    a2[["adm2_name", "adm2_pcode", "adm1_name", "adm1_pcode"]].sort_values("adm2_pcode") \
        .to_csv(OUT_PREFS, index=False, encoding="utf-8")
    print(f"\nwrote {OUT_UNITS}\nwrote {OUT_LOOKUP}\nwrote {OUT_HEXES} ({len(out):,} hexes)\n"
          f"wrote {OUT_PREFS}")


if __name__ == "__main__":
    main()
