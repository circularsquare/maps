"""Namibia — the 14 regions, their 2023 census populations, and the Kontur grid.

Writes:
    data/geo/na/na_regions.gpkg      the 14 regions (`units`)
    data/geo/na/na_hexes.gpkg        Kontur 400 m hexes with `unit` and `pop` (`place`)
    data/geo/na/na_lookup.csv        unit -> name, 2023 census population, Kontur population

Usage:
    python sources/na_geo.py --fetch    COD-AB shapefile zip (~3 MB), Kontur (~6 MB), census report (~20 MB)
    python sources/na_geo.py            rebuild from data/raw/na/

## BOUNDARIES: OCHA COD-AB NAMIBIA (`cod-ab-nam`), VERSION 01

Namibia Statistics Agency's 14 regions (Kavango split into East and West in 2013, Caprivi renamed
Zambezi), valid on 2020-01-09. Joined to the census by name; there is no shared code.

## POPULATION: THE 2023 CENSUS ITSELF, NOT COD-PS

COD-PS Namibia (`cod-ps-nam`, 2023) is the US Census Bureau's projection from the 2011 census (the
workbook's own metadata), 2,777,232 people, and its adm1 sheet has its total columns shifted
(Erongo's `T_TL` reads 15,022). The census was counted in September-November 2023 and its *Main
Report* (NSA, 28 October 2024) prints every region's population in Table 2.2 (PDF page 35): 3,022,401
in all. That table is the row margin. It counts everyone present, 145,395 non-Namibians (4.8%,
Table 3.1) included.

## AREAS: THE CENSUS'S DENSITIES AGREE WITH COD-AB EXCEPT ON THE KAVANGO LINE

The report's region profiles (PDF pages 14-27) print people per km2 to one decimal. Population over
density, with the rounding taken as an interval, contains COD-AB's area for 12 of the 14 regions.
Kavango East and West do not: the census implies about 24,000 and 24,650 km2, COD-AB has 25,367 and
23,091, while the pair's sum agrees (48,655 against 48,458). So COD-AB draws the line between the two
Kavango regions about 1,400 km2 west of where the census's areas put it. It moves nobody's religion:
both regions take the one Kavango mix the survey measures (`sources/na.py`). It moves dots between
the two only through their census totals, and the Kontur witness below shows how many people that is.
"""

import gzip
import os
import re
import shutil
import sys
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "na")
SHP_DIR = os.path.join(RAW, "shp")
GEO = os.path.join(ROOT, "data", "geo", "na")
OUT_UNITS = os.path.join(GEO, "na_regions.gpkg")
OUT_HEXES = os.path.join(GEO, "na_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "na_lookup.csv")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")

CENSUS_PDF = "phc2023_main_report.pdf"
DOWNLOADS = {
    "nam_admin_boundaries.shp.zip": (
        "https://data.humdata.org/dataset/50fda5c8-bc93-48e7-b542-f123b2038350/resource/"
        "f3f8c77c-27fe-4d65-a7cd-e31ae380db47/download/nam_admin_boundaries.shp.zip",
        b"PK", 1_000_000),
    "kontur_population_NA_20231101.gpkg.gz": (
        "https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
        "kontur_population_NA_20231101.gpkg.gz", b"\x1f\x8b", 1_000_000),
    CENSUS_PDF: (
        "https://nsa.org.na/wp-content/uploads/2024/10/"
        "2023-Population-and-Housing-Census-Main-Report-28-Oct-2024.pdf", b"%PDF", 10_000_000),
}
KONTUR_GPKG = "kontur_population_NA_20231101.gpkg"

N_REGIONS = 14
CENSUS_2023 = 3_022_401
# Table 2.2 (PDF page 35), both sexes. Transcribed, then re-read from the PDF in `read_census`.
CENSUS_T22_PAGE = 35
CENSUS_T22 = {
    "//Kharas": 109_893, "Erongo": 240_206, "Hardap": 106_680, "Kavango East": 218_421,
    "Kavango West": 123_266, "Khomas": 494_605, "Kunene": 120_762, "Ohangwena": 337_729,
    "Omaheke": 102_881, "Omusati": 316_671, "Oshana": 230_801, "Oshikoto": 257_302,
    "Otjozondjupa": 220_811, "Zambezi": 142_373,
}
# The census's spelling -> COD-AB's `adm1_name`; only //Kharas differs.
COD_NAME = {"//Kharas": "Karas"}
# Each region's profile page (PDF page) and the 2023 people per km2 printed on it.
PROFILE_PAGE = {
    "//Kharas": 14, "Erongo": 15, "Hardap": 16, "Kavango East": 17, "Kavango West": 18,
    "Khomas": 19, "Kunene": 20, "Ohangwena": 21, "Omaheke": 22, "Omusati": 23, "Oshana": 24,
    "Oshikoto": 25, "Otjozondjupa": 26, "Zambezi": 27,
}
# The two regions whose COD-AB area falls outside the census density's rounding interval; their
# sum is asserted instead (see the docstring).
AREA_DISAGREES = {"Kavango East", "Kavango West"}
AREA_SLACK = 0.02
PAIR_SUM_TOL = 0.01
NE_COUNTRIES = os.path.join(ROOT, "data", "geo", "ne_10m_admin_0_countries.geojson")
METRIC = "EPSG:32733"
SNAP_KM = 5
# Measured 2026-10-03: 19,900 people snapped (Kavango West 5,399, Kavango East 4,307, Ohangwena 2,894).
SNAP_MAX_PEOPLE = 25_000
# Kontur 2023-11 against the 2023 census per region, over the national ratio (0.868). Measured on
# the first build (2026-10-03): Kavango West 0.83 (the region COD-AB draws short, see the docstring)
# to Khomas 1.13; the band sits a little outside that.
KONTUR_TOL = 0.25
KONTUR_UNIT_BAND = (0.75, 1.25)


def fold(s):
    return re.sub(r"[^a-z]", "", str(s).casefold())


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
    z = os.path.join(RAW, "nam_admin_boundaries.shp.zip")
    if not os.path.exists(z):
        raise SystemExit(f"missing {z}; run with --fetch first")
    if not os.path.exists(os.path.join(SHP_DIR, "nam_admin1.shp")):
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


def read_census():
    """Table 2.2's populations and each profile's density, re-read from the PDF."""
    import fitz

    path = os.path.join(RAW, CENSUS_PDF)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run with --fetch first")
    doc = fitz.open(path)
    if doc.page_count != 122:
        raise SystemExit(f"the census report has {doc.page_count} pages, expected 122")
    t = " ".join(doc[CENSUS_T22_PAGE - 1].get_text().split())
    if "Table 2.2: Population and percent distribution by sex and area" not in t:
        raise SystemExit(f"PDF page {CENSUS_T22_PAGE} is not Table 2.2")
    # The table prints thousands with spaces and runs the columns together in the text layer, so
    # each transcribed figure is looked for as printed, followed by the male column's first digit.
    def printed(v):
        return f"{v:,}".replace(",", " ")
    if f"Namibia {printed(CENSUS_2023)} " not in t:
        raise SystemExit(f"Table 2.2's national total is not {CENSUS_2023:,}")
    for name, v in CENSUS_T22.items():
        if not re.search(re.escape(f"{name} {printed(v)}") + r" \d", t):
            near = re.search(re.escape(name) + r" ([\d ]{0,20})", t)
            raise SystemExit(f"Table 2.2 {name}: transcribed {v:,}, the PDF row reads "
                             f"{near.group(1) if near else 'nothing'!r}")
    if sum(CENSUS_T22.values()) != CENSUS_2023:
        raise SystemExit("Table 2.2's regions do not sum to the national total")
    dens = {}
    for name, page in PROFILE_PAGE.items():
        pt = " ".join(doc[page - 1].get_text().split())
        head = name.replace("//", "")
        if not re.search(re.escape(head) + r"\s*–\s*Census Indicators", pt):
            raise SystemExit(f"PDF page {page} is not {name}'s profile")
        m = re.search(r"People per sq\. ?km\.? ([\d.]+) ([\d.]+)", pt)
        if not m:
            raise SystemExit(f"no density on {name}'s profile (PDF page {page})")
        dens[name] = m.group(2)                 # the 2023 column, kept as printed
    print(f"  census 2023 Table 2.2 re-read from the PDF: 14 regions, {CENSUS_2023:,} people")
    return dens


def area_check(units, dens):
    """COD-AB's area against population over the printed density, rounding as an interval."""
    print("\n  COD-AB area against the census's population / density (2023, rounding as an interval):")
    out = []
    for r in units.itertuples():
        d = dens[r.name]
        step = 0.5 * 10 ** -(len(d.split(".")[1]) if "." in d else 0)
        lo, hi = r.pop / (float(d) + step), r.pop / (float(d) - step)
        inside = lo * (1 - AREA_SLACK) <= r.area_sqkm <= hi * (1 + AREA_SLACK)
        out.append((r.name, inside))
        print(f"    {r.name:<14}{r.area_sqkm:>10,.0f} km2   census {lo:>9,.0f}-{hi:>9,.0f}  "
              f"(density {d})  {'ok' if inside else 'OUTSIDE'}")
    bad = {n for n, ok in out if not ok}
    if bad != AREA_DISAGREES:
        raise SystemExit(f"regions outside the census area interval are {sorted(bad)}, expected "
                         f"{sorted(AREA_DISAGREES)}; read the table above")
    sub = units[units["name"].isin(AREA_DISAGREES)]
    cod = float(sub["area_sqkm"].sum())
    cen = float(sum(r.pop / float(dens[r.name]) for r in sub.itertuples()))
    print(f"    Kavango East + West: COD-AB {cod:,.0f} km2, census {cen:,.0f}; the pair agrees, "
          "the line between them does not")
    if abs(cod / cen - 1) > PAIR_SUM_TOL:
        raise SystemExit("the two Kavango regions no longer agree with the census even together")


def border_and_coast(pts, outside, units, geo_id):
    """Hexes whose centroid is outside every region: a neighbour's town or the sea.

    Kontur's NA extract runs over the border rivers. Measured 2026-10-03: all 628 such centroids lie
    within 3.5 km of a region, most within 0.5 km, and the big ones are across the river from
    Katima Mulilo (Sesheke, Zambia), Nkurenkuru (Calai, Angola) and Oshikango (Santa Clara, Angola).
    The playbook's rule (`playbooks/geography.md`, Bhutan): across a land border drop the hex, on a
    coast snap it. A centroid inside a Natural Earth country other than Namibia is dropped (20,406
    people). The rest are in the Atlantic off Walvis Bay and Swakopmund, or inside Natural Earth's
    Namibia where COD-AB's line runs a few hundred metres short of the river (Oshikango's edge, the
    Kavango riverbank); they go to the nearest region within `SNAP_KM` (19,900 people). Asserted: nothing beyond `SNAP_KM`, and what is snapped stays small.
    """
    import geopandas as gpd

    ne = gpd.read_file(NE_COUNTRIES)
    others = ne[ne["ADM0_A3"] != "NAM"][["ADM0_A3", "geometry"]].to_crs(units.crs)
    sub = pts[outside]
    hit = gpd.sjoin(sub, others, how="left", predicate="within")
    hit = hit[~hit.index.duplicated(keep="first")]
    foreign = hit["ADM0_A3"].notna()
    m = sub.to_crs(METRIC)
    nn = gpd.sjoin_nearest(m, units[["geo_id", "geometry"]].to_crs(METRIC), distance_col="d")
    nn = nn[~nn.index.duplicated(keep="first")].reindex(sub.index)
    if (nn["d"] > SNAP_KM * 1000).any():
        far = nn[nn["d"] > SNAP_KM * 1000]
        raise SystemExit(f"{len(far)} outside hexes are more than {SNAP_KM} km from any region "
                         f"({far['pop'].sum():,.0f} people); read them before snapping")
    snap = ~foreign
    by_country = hit.loc[foreign].groupby("ADM0_A3")["pop"].sum().round()
    print(f"    in a neighbour (dropped): {int(foreign.sum())} hexes, "
          f"{float(sub.loc[foreign, 'pop'].sum()):,.0f} people "
          f"({', '.join(f'{k} {v:,.0f}' for k, v in by_country.items())})")
    by_unit = nn.loc[snap].groupby("geo_id")["pop"].sum().round()
    print(f"    in no country (snapped to the nearest region within {SNAP_KM} km): "
          f"{int(snap.sum())} hexes, {float(sub.loc[snap, 'pop'].sum()):,.0f} people "
          f"({', '.join(f'{k} {v:,.0f}' for k, v in by_unit.items())})")
    if float(sub.loc[snap, "pop"].sum()) > SNAP_MAX_PEOPLE:
        raise SystemExit("more people snapped than the first build saw; read the hexes")
    out = geo_id.copy()
    out.loc[nn.index[snap.to_numpy()]] = nn.loc[snap, "geo_id"]
    return out


def main():
    import geopandas as gpd
    import pandas as pd

    import geo_checks

    if "--fetch" in sys.argv:
        fetch()
    gpkg = unpack()
    dens = read_census()

    a1 = gpd.read_file(os.path.join(SHP_DIR, "nam_admin1.shp"), engine="fiona")
    if len(a1) != N_REGIONS:
        raise SystemExit(f"COD-AB has {len(a1)} regions, expected {N_REGIONS}")
    print(f"COD-AB Namibia: {len(a1)} regions, version {a1['version'].iloc[0]}, valid_on "
          f"{a1['valid_on'].iloc[0]}, crs={a1.crs}")
    cod = {fold(n): pc for n, pc in zip(a1["adm1_name"], a1["adm1_pcode"])}
    pcode = {}
    for name in CENSUS_T22:
        k = fold(COD_NAME.get(name, name))
        if k not in cod:
            raise SystemExit(f"census region {name!r} has no COD-AB region")
        pcode[name] = cod[k]
    if len(set(pcode.values())) != N_REGIONS:
        raise SystemExit("two census regions joined one COD-AB region")

    units = a1[["adm1_pcode", "geometry"]].rename(columns={"adm1_pcode": "geo_id"}).copy()
    name_of = {pc: n for n, pc in pcode.items()}
    units["name"] = units["geo_id"].map(name_of)
    units["pop"] = units["name"].map(CENSUS_T22).astype("int64")
    units["area_sqkm"] = units.to_crs("EPSG:6933").area / 1e6
    area_check(units, dens)

    # ---- Kontur ----
    hexes = geo_checks.read_layer(gpkg, "Kontur NA")
    popcol = next(c for c in hexes.columns if c.lower() == "population")
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({"pop": hexes[popcol].to_numpy(dtype=float)}, geometry=cent,
                           crs=hexes.crs).to_crs(units.crs)
    joined = gpd.sjoin(pts, units[["geo_id", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    outside = joined["geo_id"].isna()
    print(f"\nKontur hexes: {len(hexes):,}, population {pts['pop'].sum():,.0f}; "
          f"{int(outside.sum()):,} centroids outside every region "
          f"({float(pts.loc[outside, 'pop'].sum()):,.0f} people)")
    joined["geo_id"] = border_and_coast(pts, outside, units, joined["geo_id"])
    outside = joined["geo_id"].isna()
    keep = ~outside
    out = gpd.GeoDataFrame({"unit": joined.loc[keep, "geo_id"].to_numpy(),
                            "pop": pts.loc[keep, "pop"].to_numpy()},
                           geometry=hexes.to_crs(units.crs).geometry[keep.to_numpy()].to_numpy(),
                           crs=units.crs)
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["geo_id"]) - set(per.index))
    if missing:
        raise SystemExit(f"regions with no populated hex: {missing}")
    tot = float(out["pop"].sum())
    nat = tot / CENSUS_2023
    print(f"  Kontur {tot:,.0f} against the 2023 census {CENSUS_2023:,}: ratio {nat:.3f}")
    if abs(nat - 1) > KONTUR_TOL:
        raise SystemExit("Kontur and the census disagree nationally by more than the tolerance")
    units["kontur_pop"] = units["geo_id"].map(per["sum"]).round().astype("int64")
    units["hexes"] = units["geo_id"].map(per["size"]).astype("int64")
    print("\n  per region, census 2023 against Kontur (the witness; Kontur counts nothing here),"
          " and that ratio over the national one:")
    for r in units.sort_values("pop", ascending=False).itertuples():
        print(f"    {r.geo_id:<6}{r.name:<14}{r.pop:>10,}  Kontur {r.kontur_pop:>10,}  "
              f"{r.kontur_pop / r.pop:5.2f}x ({r.kontur_pop / r.pop / nat:5.2f})  "
              f"{r.hexes:>7,} hexes  {r.area_sqkm:>9,.0f} km2")
    geo_checks.ratio_band(dict(zip(units["geo_id"], units["pop"])),
                          dict(zip(units["geo_id"], units["kontur_pop"] / nat)),
                          *KONTUR_UNIT_BAND, what="region")

    # ---- write ----
    os.makedirs(GEO, exist_ok=True)
    keep_cols = ["geo_id", "name", "pop", "kontur_pop", "hexes", "area_sqkm", "geometry"]
    units[keep_cols].to_file(OUT_UNITS, layer="regions", driver="GPKG")
    pd.DataFrame(units[keep_cols].drop(columns="geometry")).to_csv(OUT_LOOKUP, index=False,
                                                                  encoding="utf-8")
    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT_UNITS}\nwrote {OUT_LOOKUP}\nwrote {OUT_HEXES} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
