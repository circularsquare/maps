"""Lesotho — the 10 districts, their 2016 census populations, and the Kontur grid.

Writes:
    data/geo/ls/ls_districts.gpkg    the 10 districts (`units`)
    data/geo/ls/ls_hexes.gpkg        Kontur 400 m hexes with `unit` and `pop` (`place`)
    data/geo/ls/ls_lookup.csv        unit -> name, 2016 census population, Kontur population

Usage:
    python sources/ls_geo.py --fetch    COD-AB shapefile zip (~4 MB), Kontur (~3 MB), census key findings (~1 MB)
    python sources/ls_geo.py            rebuild from data/raw/ls/

## BOUNDARIES: OCHA COD-AB LESOTHO (`cod-ab-lso`), FAO AND THE MINISTRY OF LOCAL GOVERNMENT 2016

Ten districts, unchanged since independence; the pcodes are letters (LSA Maseru ... LSK). Joined to
the census by name; the census spells Butha-Buthe `Botha-Bothe`.

## POPULATION: THE 2016 CENSUS ITSELF, NOT COD-PS

COD-PS Lesotho (`cod-ps-lso`) is a 2022 projection from the 2016 census. The 2016 Population and
Housing Census *Summary Key Findings* (Bureau of Statistics) prints every district's count in Table
2.1.2 (PDF page 2): 2,007,201 in all, of whom 6,564 non-citizens in the urban centres (page 13). That
table is the row margin; the survey pool runs 2008-2022, centred on 2015. The 2026 census was
enumerated in April 2026 and on 2026-10-03 has published nothing (bos.gov.ls/census.htm: "data
processing"). REOPEN on its district table.

## KONTUR AND THE SOUTH AFRICAN BORDER

Lesotho is an enclave. Kontur's LS extract runs over the border, so a hex whose centroid is outside
every district is either a South African town across the Caledon (Ficksburg, Ladybrand, Wepener) or
inside Natural Earth's Lesotho where COD-AB's line stops short. The first is dropped and the second
snapped to the nearest district (`playbooks/geography.md`, Bhutan and Namibia). Measured
2026-10-03: 222 centroids outside every district, 94 in South Africa (4,362 people, dropped) and
128 inside Natural Earth's Lesotho (7,285, snapped, all within 5 km).

## KONTUR HAS LOST MOST OF BEREA, EVENLY

Kontur 2023-11 holds 2,337,275 people against the census's 2,007,201 (1.164x), and per district
0.99 to 1.24 of that ratio for eight districts. Berea reads 0.257 (78,707 against 262,616) and
Mokhotlong 1.389. Berea's deficit is flat, Cuba's Granma case (`playbooks/geography.md`): its median
hex holds 20 people against Leribe's 78 next door, its 90th percentile 92 against 441, and
Teyateyaneng (24,001 urban in 2016) holds 4,677 within 3 km and 19,167 within 10 km, about the
town's share of the district once scaled. Neither neighbour is inflated by Berea's loss (Maseru 1.08,
Leribe 0.99). The scatter gives each district its census count and uses Kontur only to place dots
inside it, so a flat error in one district moves nobody; the two ratios are pinned in `OUT_OF_BAND`
and the other eight banded.
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
RAW = os.path.join(ROOT, "data", "raw", "ls")
SHP_DIR = os.path.join(RAW, "shp")
GEO = os.path.join(ROOT, "data", "geo", "ls")
OUT_UNITS = os.path.join(GEO, "ls_districts.gpkg")
OUT_HEXES = os.path.join(GEO, "ls_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "ls_lookup.csv")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")

CENSUS_PDF = "2016_Summary_Key_Findings.pdf"
DOWNLOADS = {
    "lso_adm_fao_mlgca_2019.zip": (
        "https://data.humdata.org/dataset/55b1367e-667a-447b-952d-5bb139835628/resource/"
        "f922a67a-9840-4174-bd1e-e4b10cc88591/download/lso_adm_fao_mlgca_2019.zip",
        b"PK", 1_000_000),
    "kontur_population_LS_20231101.gpkg.gz": (
        "https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
        "kontur_population_LS_20231101.gpkg.gz", b"\x1f\x8b", 300_000),
    CENSUS_PDF: ("http://www.bos.gov.ls/2016_Summary_Key_Findings.pdf", b"%PDF", 500_000),
}
KONTUR_GPKG = "kontur_population_LS_20231101.gpkg"
ADM1_SHP = "lso_admbnda_adm1_FAO_MLGCA_2019.shp"

N_DISTRICTS = 10
CENSUS_2016 = 2_007_201
# Table 2.1.2 (PDF page 2), the 2016 column. Transcribed, then re-read from the PDF in `read_census`.
CENSUS_PAGE = 2
CENSUS_T212 = {
    "Botha-Bothe": 118_242, "Leribe": 337_521, "Berea": 262_616, "Maseru": 519_186,
    "Mafeteng": 178_222, "Mohale's Hoek": 165_590, "Quthing": 115_469, "Qacha's Nek": 74_566,
    "Mokhotlong": 100_442, "Thaba-Tseka": 135_347,
}
# The census's spelling -> COD-AB's `ADM1_EN`, where they differ.
COD_NAME = {"Botha-Bothe": "Butha-Buthe"}
NE_COUNTRIES = os.path.join(ROOT, "data", "geo", "ne_10m_admin_0_countries.geojson")
METRIC = "EPSG:32735"
SNAP_KM = 5
SNAP_MAX_PEOPLE = 25_000          # set before the first read; see the printed figure
KONTUR_TOL = 0.30
KONTUR_UNIT_BAND = (0.75, 1.25)
# Kontur over the census share, over the national ratio (1.164), measured 2026-10-03; see the
# docstring. Each district's dots are its census count, so these move nobody between districts.
OUT_OF_BAND = {"LSD": (0.20, 0.32),      # Berea 0.257: Kontur has lost three quarters of it, flatly
               "LSJ": (1.30, 1.45)}      # Mokhotlong 1.389


def fold(s):
    return re.sub(r"[^a-z]", "", str(s).casefold().replace("’", "'"))


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
    z = os.path.join(RAW, "lso_adm_fao_mlgca_2019.zip")
    if not os.path.exists(z):
        raise SystemExit(f"missing {z}; run with --fetch first")
    if not os.path.exists(os.path.join(SHP_DIR, ADM1_SHP)):
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
    """Table 2.1.2's 2016 column, re-read from the PDF."""
    import fitz

    path = os.path.join(RAW, CENSUS_PDF)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run with --fetch first")
    t = " ".join(fitz.open(path)[CENSUS_PAGE - 1].get_text().split())
    if "Table 2.1.2: Population and Percentage Distribution of Population by Place of Residence" not in t:
        raise SystemExit(f"PDF page {CENSUS_PAGE} is not Table 2.1.2")
    # Each row prints 1996, 2006, 2016 and two percentage changes; the 2016 figure is the third.
    for name, v in CENSUS_T212.items():
        m = re.search(re.escape(name) + r" [\d,]+ [\d,]+ ([\d,]+) -?[\d.]+ -?[\d.]+", t)
        got = int(m.group(1).replace(",", "")) if m else None
        if got != v:
            raise SystemExit(f"Table 2.1.2 {name}: transcribed {v:,}, the PDF has {got}")
    m = re.search(r"Lesotho [\d,]+ [\d,]+ ([\d,]+) ", t)
    if not m or int(m.group(1).replace(",", "")) != CENSUS_2016:
        raise SystemExit(f"Table 2.1.2's national total is not {CENSUS_2016:,}")
    if sum(CENSUS_T212.values()) != CENSUS_2016:
        raise SystemExit("Table 2.1.2's districts do not sum to the national total")
    print(f"  census 2016 Table 2.1.2 re-read from the PDF: 10 districts, {CENSUS_2016:,} people")


def border(pts, outside, units, geo_id):
    """Hexes whose centroid is outside every district: South Africa across the line, or Lesotho
    land COD-AB's line leaves out. Dropped if inside a Natural Earth country other than Lesotho,
    else snapped to the nearest district within `SNAP_KM`."""
    import geopandas as gpd

    ne = gpd.read_file(NE_COUNTRIES)
    others = ne[ne["ADM0_A3"] != "LSO"][["ADM0_A3", "geometry"]].to_crs(units.crs)
    sub = pts[outside]
    hit = gpd.sjoin(sub, others, how="left", predicate="within")
    hit = hit[~hit.index.duplicated(keep="first")]
    foreign = hit["ADM0_A3"].notna()
    m = sub.to_crs(METRIC)
    nn = gpd.sjoin_nearest(m, units[["geo_id", "geometry"]].to_crs(METRIC), distance_col="d")
    nn = nn[~nn.index.duplicated(keep="first")].reindex(sub.index)
    snap = ~foreign
    far = snap & (nn["d"] > SNAP_KM * 1000)
    if far.any():
        raise SystemExit(f"{int(far.sum())} outside hexes in no country are more than {SNAP_KM} km "
                         f"from any district ({sub.loc[far, 'pop'].sum():,.0f} people); read them")
    by_country = hit.loc[foreign].groupby("ADM0_A3")["pop"].sum().round()
    print(f"    in a neighbour (dropped): {int(foreign.sum())} hexes, "
          f"{float(sub.loc[foreign, 'pop'].sum()):,.0f} people "
          f"({', '.join(f'{k} {v:,.0f}' for k, v in by_country.items())})")
    by_unit = nn.loc[snap].groupby("geo_id")["pop"].sum().round()
    print(f"    in no country (snapped to the nearest district within {SNAP_KM} km): "
          f"{int(snap.sum())} hexes, {float(sub.loc[snap, 'pop'].sum()):,.0f} people "
          f"({', '.join(f'{k} {v:,.0f}' for k, v in by_unit.items())})")
    if float(sub.loc[snap, "pop"].sum()) > SNAP_MAX_PEOPLE:
        raise SystemExit("more people snapped than SNAP_MAX_PEOPLE; read the hexes")
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
    read_census()

    a1 = gpd.read_file(os.path.join(SHP_DIR, ADM1_SHP), engine="fiona")
    if len(a1) != N_DISTRICTS:
        raise SystemExit(f"COD-AB has {len(a1)} districts, expected {N_DISTRICTS}")
    print(f"COD-AB Lesotho: {len(a1)} districts, columns {list(a1.columns)}, crs={a1.crs}")
    namecol = next(c for c in a1.columns if c.upper() == "ADM1_EN")
    pcol = next(c for c in a1.columns if c.upper() == "ADM1_PCODE")
    cod = {fold(n): pc for n, pc in zip(a1[namecol], a1[pcol])}
    pcode = {}
    for name in CENSUS_T212:
        k = fold(COD_NAME.get(name, name))
        if k not in cod:
            raise SystemExit(f"census district {name!r} has no COD-AB district (have {sorted(cod)})")
        pcode[name] = cod[k]
    if len(set(pcode.values())) != N_DISTRICTS:
        raise SystemExit("two census districts joined one COD-AB district")

    units = a1[[pcol, "geometry"]].rename(columns={pcol: "geo_id"}).copy()
    name_of = {pc: COD_NAME.get(n, n) for n, pc in pcode.items()}
    census_of = {pc: CENSUS_T212[n] for n, pc in pcode.items()}
    units["name"] = units["geo_id"].map(name_of)
    units["pop"] = units["geo_id"].map(census_of).astype("int64")
    units["area_sqkm"] = units.to_crs("EPSG:6933").area / 1e6
    print(f"  total area {units['area_sqkm'].sum():,.0f} km2 (Lesotho's official 30,355)")

    # ---- Kontur ----
    hexes = geo_checks.read_layer(gpkg, "Kontur LS")
    popcol = next(c for c in hexes.columns if c.lower() == "population")
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({"pop": hexes[popcol].to_numpy(dtype=float)}, geometry=cent,
                           crs=hexes.crs).to_crs(units.crs)
    joined = gpd.sjoin(pts, units[["geo_id", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    outside = joined["geo_id"].isna()
    print(f"\nKontur hexes: {len(hexes):,}, population {pts['pop'].sum():,.0f}; "
          f"{int(outside.sum()):,} centroids outside every district "
          f"({float(pts.loc[outside, 'pop'].sum()):,.0f} people)")
    joined["geo_id"] = border(pts, outside, units, joined["geo_id"])
    keep = joined["geo_id"].notna()
    out = gpd.GeoDataFrame({"unit": joined.loc[keep, "geo_id"].to_numpy(),
                            "pop": pts.loc[keep, "pop"].to_numpy()},
                           geometry=hexes.to_crs(units.crs).geometry[keep.to_numpy()].to_numpy(),
                           crs=units.crs)
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["geo_id"]) - set(per.index))
    if missing:
        raise SystemExit(f"districts with no populated hex: {missing}")
    tot = float(out["pop"].sum())
    nat = tot / CENSUS_2016
    print(f"  Kontur {tot:,.0f} against the 2016 census {CENSUS_2016:,}: ratio {nat:.3f}")
    if abs(nat - 1) > KONTUR_TOL:
        raise SystemExit("Kontur and the census disagree nationally by more than the tolerance")
    units["kontur_pop"] = units["geo_id"].map(per["sum"]).round().astype("int64")
    units["hexes"] = units["geo_id"].map(per["size"]).astype("int64")
    print("\n  per district, census 2016 against Kontur (the witness), and that ratio over the national one:")
    for r in units.sort_values("pop", ascending=False).itertuples():
        print(f"    {r.geo_id:<6}{r.name:<14}{r.pop:>10,}  Kontur {r.kontur_pop:>10,}  "
              f"{r.kontur_pop / r.pop:5.2f}x ({r.kontur_pop / r.pop / nat:5.2f})  "
              f"{r.hexes:>7,} hexes  {r.area_sqkm:>9,.0f} km2")
    rel = units.set_index("geo_id").eval("kontur_pop / pop") / nat
    for g, (lo, hi) in OUT_OF_BAND.items():
        if not lo <= rel[g] <= hi:
            raise SystemExit(f"{g}'s Kontur/census ratio is now {rel[g]:.3f}, outside its pinned "
                             f"[{lo}, {hi}]; read the docstring's Berea section again")
    rest = units[~units["geo_id"].isin(OUT_OF_BAND)]
    geo_checks.ratio_band(dict(zip(rest["geo_id"], rest["pop"])),
                          dict(zip(rest["geo_id"], rest["kontur_pop"] / nat)),
                          *KONTUR_UNIT_BAND, what="district")

    # ---- write ----
    os.makedirs(GEO, exist_ok=True)
    keep_cols = ["geo_id", "name", "pop", "kontur_pop", "hexes", "area_sqkm", "geometry"]
    units[keep_cols].to_file(OUT_UNITS, layer="districts", driver="GPKG")
    pd.DataFrame(units[keep_cols].drop(columns="geometry")).to_csv(OUT_LOOKUP, index=False,
                                                                  encoding="utf-8")
    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT_UNITS}\nwrote {OUT_LOOKUP}\nwrote {OUT_HEXES} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
