"""Djibouti — the six régions and the placement grid.

Writes:
    data/geo/dj/dj_regions.gpkg    the 6 counted units (`regions`)
    data/geo/dj/dj_hexes.gpkg      Kontur 400 m hexes with `unit` and `pop` (`hexes`)
    data/geo/dj/dj_lookup.csv      unit -> census population, area, Kontur population

Usage:
    python sources/dj_geo.py --fetch    COD-AB shapefile zip (0.3 MB), Kontur DJ, GeoNames DJ
    python sources/dj_geo.py            rebuild from data/raw/dj/

## THE LAYER

OCHA COD-AB Djibouti (`cod-ab-dji`, `dji_adm_gadm_2022_shp.zip`, dated 2022-12-06) is GADM's
boundaries vetted by ITOS; its own notes say the date the boundaries were established is unknown
and that it has not been reviewed in the last twelve months. ADM1 is the six régions under the
census's names (Arta has its own polygon, so it postdates the 2003 creation of Arta). Joined by
name, folded; the six names are asserted both ways.

## THE WITNESSES THE NAME DOES NOT DECIDE

1. Area. The final report prints only three densities: 46 per km2 nationally, 4,437 for
   Djibouti-Ville and 9 for Tadjourah (pp.35-36). Population over density gives the office's
   area for those three, banded with the printed density's rounding.
2. People. Kontur's 2023 grid per région against the census's de jure count by région (final
   report Tableau 12), over the national ratio.
3. Towns. Every GeoNames first-order seat (PPLA, PPLC) holds Kontur people within 5 km of at
   least a third of its town's census count (final report Tableau 14's `-Ville` units, ordinary
   households), so a seat Kontur has lost (Béchar, Tan-Tan) is caught.

## HEXES OUTSIDE THE RÉGIONS

Djibouti borders Ethiopia and Somalia, both drawn here, and Eritrea, which is being built now. A
hex whose centroid is outside COD's régions is (a) dropped when a drawn neighbour's place layer
holds the same hex or its centroid lies in a neighbour's Natural Earth polygon, (b) snapped to
the nearest région within SNAP_KM when it is in Natural Earth's Djibouti or the sea (the coast
and the Gulf of Tadjourah), and (c) otherwise dropped. People per rule are pinned.
"""

import csv
import gzip
import os
import re
import shutil
import sys
import unicodedata
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from dj import DE_JURE, DE_JURE_REGION, LABELS, REGIONS, T42   # noqa: E402

RAW = os.path.join(ROOT, "data", "raw", "dj")
SHP_DIR = os.path.join(RAW, "shp")
GEO = os.path.join(ROOT, "data", "geo", "dj")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur")
NORM = os.path.join(ROOT, "data", "normalized", "dj.csv")
NE = os.path.join(ROOT, "data", "geo", "ne_10m_admin_0_countries.geojson")

OUT_UNITS = os.path.join(GEO, "dj_regions.gpkg")
OUT_HEXES = os.path.join(GEO, "dj_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "dj_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}
COD_ZIP = os.path.join(RAW, "dji_adm_gadm_2022_shp.zip")
KONTUR_GZ = os.path.join(KONTUR, "kontur_population_DJ_20231101.gpkg.gz")
KONTUR_GPKG = KONTUR_GZ[:-3]
GEONAMES = os.path.join(RAW, "geonames_DJ.zip")
DOWNLOADS = {
    COD_ZIP: ("https://data.humdata.org/dataset/ceee47a3-cd1d-4eae-ab93-87d6f2ca00b9/resource/"
              "fd8a241a-bd65-4dc2-bb64-1e79e2e27673/download/dji_adm_gadm_2022_shp.zip",
              b"PK", 200_000),
    KONTUR_GZ: ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
                "kontur_population_DJ_20231101.gpkg.gz", b"\x1f\x8b", 100_000),
    GEONAMES: ("https://download.geonames.org/export/dump/DJ.zip", b"PK", 10_000),
}
GEONAMES_COLS = ["geonameid", "name", "asciiname", "alternatenames", "lat", "lon", "fclass",
                 "fcode", "cc", "cc2", "admin1", "admin2", "admin3", "admin4", "population",
                 "elevation", "dem", "timezone", "modified"]

UNITS = 6
# folded COD ADM1 name -> census unit. COD spells two of them its own way: `Djiboutii` (sic) and
# `Tadjoura`; both pinned here as read 2026-10-03, so a renamed file stops the join.
COD_NAMES = {"alisabieh": "Ali-Sabieh", "arta": "Arta", "dikhil": "Dikhil",
             "djiboutii": "Djibouti-Ville", "obock": "Obock", "tadjoura": "Tadjourah"}
# the final report's printed densities (pp.35-36): unit -> persons per km2, and its rounding
DENSITY = {"Djibouti-Ville": (4437, 0.5), "Tadjourah": (9, 0.5)}
DENSITY_NATIONAL = (46, 0.5)
AREA_BAND = (0.85, 1.15)          # COD area over the office's, over the national ratio, beyond rounding
# Djibouti-Ville: COD (GADM) draws 195.6 km2 against the 172.9 its printed density implies
# (1.17 over the national ratio). It moves nobody between régions that matters: the densest
# 173 km2 of COD's polygon hold CORE_MIN or more of its Kontur people (asserted below).
AREA_PINNED = {"Djibouti-Ville": (1.10, 1.25)}
CORE_MIN = 0.97
METRIC = "EPSG:32638"             # UTM 38N
SNAP_KM = 2.0                     # every snapped hex is coastal; the furthest is 1.71 km out
MATCH_DEG = 1e-4                  # the same hex in a neighbour's layer: centroids within ~10 m
NEIGHBOUR_LAYERS = ("et", "so")   # drawn neighbours; Eritrea is not drawn yet (2026-10-03)
NE_DJ = "DJI"
KONTUR_NATIONAL = (0.6, 1.6)      # Kontur Nov 2023 over the May 2024 de jure count
KONTUR_UNIT = (0.5, 2.0)          # per région over the national ratio
# Tadjourah reads 2.47: Kontur's région shares are 2009's, not 2024's (CENSUS_2009 below), and
# Tadjourah fell from 86,704 people in 2009 to 60,645 in 2024 while the capital grew 1.6x.
# Placement is inside each région only, so this moves no count; the seat test shows Kontur's
# split between each chief town and its countryside is close to Tableau 14's.
KONTUR_UNIT_PINNED = {"Tadjourah": (2.2, 2.7)}
# The 2009 census by région (DISED, Annuaire statistique 2012, Tableau 2.1.2, total resident
# population; ministere-finances.dj/Annuaire2012Ver02102012-Correct3.pdf, read 2026-10-03).
CENSUS_2009 = {"Djibouti-Ville": 475_322, "Ali-Sabieh": 86_949, "Dikhil": 88_948,
               "Tadjourah": 86_704, "Obock": 37_856, "Arta": 42_380}
SEAT_KM = 5.0
SEAT_MIN = 1 / 3                  # Kontur within 5 km over the town's census count
# Tableau 14, ordinary and nomadic households of each région's chief town ("-Ville" unit)
TOWNS = {"Djibouti-Ville": 728_010, "Ali-Sabieh": 43_002, "Dikhil": 25_776, "Tadjourah": 17_534,
         "Obock": 18_626, "Arta": 9_751}
# people per rule for the hexes outside COD's régions, measured 2026-10-03. `SOL` is Natural
# Earth's Somaliland, which `so` draws inside Somalia; none of the et_hexes cells is among them.
OUTSIDE_PINNED = {"drop: in Natural Earth ERI": 7, "drop: in Natural Earth ETH": 42,
                  "drop: in Natural Earth SOL": 84, "drop: in so_hexes": 118, "snap": 45727}


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.casefold())


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    os.makedirs(KONTUR, exist_ok=True)
    for dst, (url, magic, min_size) in DOWNLOADS.items():
        if os.path.exists(dst) and os.path.getsize(dst) > min_size:
            print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
            continue
        r = requests.get(url, headers=UA, timeout=900)
        r.raise_for_status()
        if not r.content.startswith(magic) or len(r.content) < min_size:   # a 200 is not a file
            raise SystemExit(f"{os.path.basename(dst)}: {len(r.content):,} bytes starting "
                             f"{r.content[:16]!r}, expected {magic!r} and > {min_size:,}")
        with open(dst + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(dst + ".part", dst)
        print(f"  got  {os.path.basename(dst)} ({len(r.content):,} bytes)")


def unpack():
    if not os.path.isdir(SHP_DIR):
        if not os.path.exists(COD_ZIP):
            raise SystemExit(f"missing {COD_ZIP}; run with --fetch first")
        with zipfile.ZipFile(COD_ZIP) as zz:
            zz.extractall(SHP_DIR)
    if not os.path.exists(KONTUR_GPKG):
        if not os.path.exists(KONTUR_GZ):
            raise SystemExit(f"missing {KONTUR_GZ}; run with --fetch first")
        with gzip.open(KONTUR_GZ, "rb") as src, open(KONTUR_GPKG + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(KONTUR_GPKG + ".part", KONTUR_GPKG)


def _shp(level):
    hits = [os.path.join(dp, f) for dp, _d, fs in os.walk(SHP_DIR) for f in fs
            if f.lower().endswith(".shp") and re.search(rf"adm{level}(?!\d)", f.lower())]
    if len(hits) != 1:
        raise SystemExit(f"expected one ADM{level} shapefile under {SHP_DIR}, found {hits}")
    return hits[0]


def neighbour_held(pts):
    """cc or 'none' per point: whether a drawn neighbour's place layer has the same hex."""
    import geopandas as gpd
    from scipy.spatial import cKDTree

    p = pts.to_crs(4326)
    xy = np.column_stack([p.geometry.x, p.geometry.y])
    held = pd.Series("none", index=pts.index)
    for cc in NEIGHBOUR_LAYERS:
        path = os.path.join(ROOT, "data", "geo", cc, f"{cc}_hexes.gpkg")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}: the border rule needs {cc}'s place layer")
        g = gpd.read_file(path, engine="pyogrio")
        c = g.geometry.to_crs(3857).centroid.to_crs(4326)
        dist, _ = cKDTree(np.column_stack([c.x, c.y])).query(xy)
        hit = (dist < MATCH_DEG) & (held.to_numpy() == "none")
        held[hit] = cc
        print(f"  {cc}_hexes.gpkg: {len(g):,} cells, {int(hit.sum()):,} of the outside hexes in it")
    return held


def main():
    import geopandas as gpd

    import geo_checks

    if "--fetch" in sys.argv:
        fetch()
    unpack()
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM}; run sources/dj.py first")
    census = pd.read_csv(NORM, keep_default_na=False, na_values=[""]).groupby("geo_id")["count"].sum()
    want = {LABELS[lb]: r[8] for lb, r in T42.items()}
    if census.to_dict() != want:
        raise SystemExit(f"dj.csv régions {census.to_dict()} are not Tableau 42's {want}")
    eq = "EPSG:6933"

    # ---- 1. COD-AB ADM1 by folded name
    a0 = gpd.read_file(_shp(0), engine="pyogrio")
    a1 = gpd.read_file(_shp(1), engine="pyogrio")
    a2 = gpd.read_file(_shp(2), engine="pyogrio")
    print(f"COD-AB Djibouti: ADM1 {len(a1)} features, ADM2 {len(a2)}; columns {list(a1.columns)}")
    namecol = next(c for c in a1.columns if c.lower() in ("adm1_en", "adm1_fr", "adm1_name"))
    pcol = next(c for c in a1.columns if c.lower() == "adm1_pcode")
    names = {fold(n): n for n in a1[namecol]}
    if len(a1) != UNITS or set(names) != set(COD_NAMES):
        raise SystemExit(f"COD ADM1 names {sorted(a1[namecol])} are not the six régions {sorted(COD_NAMES)}")
    a1["unit"] = a1[namecol].map(lambda n: COD_NAMES[fold(n)])
    units = a1[["unit", pcol, "geometry"]].rename(columns={pcol: "adm1_pcode"}).copy()
    units = units.dissolve(by="unit", as_index=False) if units["unit"].duplicated().any() else units
    units["area_km2"] = (units.to_crs(eq).area / 1e6).to_numpy()
    nat = float(a0.to_crs(eq).area.sum() / 1e6)
    if abs(units["area_km2"].sum() / nat - 1) > 0.002:
        raise SystemExit(f"COD's régions sum to {units['area_km2'].sum():,.0f} km2 against its national {nat:,.0f}")
    p2 = next(c for c in a2.columns if c.lower() == "adm1_pcode")
    n2 = a2.merge(units[["adm1_pcode", "unit"]], left_on=p2, right_on="adm1_pcode").groupby("unit").size()
    print(f"  ADM2 per région: {n2.to_dict()}")
    n2col = next(c for c in a2.columns if c.lower() == "adm2_en")
    for u, g in a2.merge(units[["adm1_pcode", "unit"]], left_on=p2, right_on="adm1_pcode").groupby("unit"):
        print(f"    {u}: {sorted(g[n2col])}")
    by = units.set_index("unit")

    # ---- 2. area against the printed densities
    fails = []
    lo_n = DE_JURE / (DENSITY_NATIONAL[0] + DENSITY_NATIONAL[1])
    hi_n = DE_JURE / (DENSITY_NATIONAL[0] - DENSITY_NATIONAL[1])
    print(f"\n  COD national {nat:,.0f} km2; the census's {lo_n:,.0f}-{hi_n:,.0f} (46/km2, rounded)")
    nat_ratio = nat / (DE_JURE / DENSITY_NATIONAL[0])
    if not AREA_BAND[0] * lo_n <= nat <= AREA_BAND[1] * hi_n:
        fails.append(f"COD's national area {nat:,.0f} outside the census's band")
    for u, (d, r) in DENSITY.items():
        lo, hi = DE_JURE_REGION[u] / (d + r), DE_JURE_REGION[u] / (d - r)
        a = by.loc[u, "area_km2"]
        print(f"  {u:<15} COD {a:>9,.1f} km2; census {lo:,.1f}-{hi:,.1f} ({d}/km2); "
              f"x{a / ((lo + hi) / 2):.3f}, /nat {a / ((lo + hi) / 2) / nat_ratio:.3f}")
        blo, bhi = AREA_PINNED.get(u, AREA_BAND)
        if not blo * lo <= a / nat_ratio <= bhi * hi:
            fails.append(f"{u}: COD area {a:,.1f} km2 outside the census's {lo:,.1f}-{hi:,.1f}")
    for u in REGIONS:
        print(f"  {u:<15} COD {by.loc[u, 'area_km2']:>9,.1f} km2, de jure density "
              f"{DE_JURE_REGION[u] / by.loc[u, 'area_km2']:,.1f}/km2")

    # ---- 3. Kontur: join, the outside hexes, per-région witness
    hexes = geo_checks.read_layer(KONTUR_GPKG, "Kontur DJ")
    popcol = next(c for c in hexes.columns if c.lower() == "population")
    pts = gpd.GeoDataFrame({"pop": hexes[popcol].to_numpy(dtype=float)},
                           geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs(units.crs)
    ktot = float(pts["pop"].sum())
    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    unit_of = joined["unit"].copy()
    out = unit_of.isna()
    print(f"\nKontur DJ 2023-11: {len(hexes):,} hexes, {ktot:,.0f} people; {int(out.sum()):,} hexes "
          f"({pts.loc[out, 'pop'].sum():,.0f} people) outside COD's régions")
    rule = pd.Series("", index=pts.index)
    if out.any():
        world = gpd.read_file(NE, engine="pyogrio")
        iso = next(c for c in ("ADM0_A3", "ISO_A3", "SOV_A3") if c in world.columns)
        world = world[[iso, "geometry"]].to_crs(units.crs)
        op = pts.loc[out]
        ne_of = gpd.sjoin(op, world, how="left", predicate="within")
        ne_of = ne_of[~ne_of.index.duplicated(keep="first")][iso].reindex(op.index)
        held = neighbour_held(op)
        near = gpd.sjoin_nearest(op.to_crs(METRIC), units[["unit", "geometry"]].to_crs(METRIC),
                                 how="left", distance_col="dist_m")
        near = near[~near.index.duplicated(keep="first")].reindex(op.index)
        for i in op.index:
            if held[i] != "none":
                rule[i] = f"drop: in {held[i]}_hexes"
            elif isinstance(ne_of[i], str) and ne_of[i] != NE_DJ:
                rule[i] = f"drop: in Natural Earth {ne_of[i]}"
            elif near.loc[i, "dist_m"] <= SNAP_KM * 1000:
                rule[i] = "snap"
                unit_of[i] = near.loc[i, "unit"]
            else:
                rule[i] = "drop: far"
        per_rule = pts.loc[out].groupby(rule[out])["pop"].sum().round().astype(int).to_dict()
        print("  outside hexes by rule:", per_rule)
        sn = rule.eq("snap")
        if sn.any():
            d = near.loc[sn[out].to_numpy(), "dist_m"]
            g = pd.DataFrame({"unit": unit_of[sn], "pop": pts.loc[sn, "pop"], "km": d.to_numpy() / 1000})
            print("  snapped, by région:", g.groupby("unit")["pop"].sum().round().astype(int).to_dict(),
                  f"; distance max {g['km'].max():.2f} km, people-weighted mean "
                  f"{(g['km'] * g['pop']).sum() / g['pop'].sum():.2f} km; in sea (no NE country) "
                  f"{pts.loc[sn & ~rule.index.isin(ne_of.dropna().index), 'pop'].sum():,.0f}")
        if OUTSIDE_PINNED is not None and per_rule != OUTSIDE_PINNED:
            fails.append(f"outside-hex rules moved: {per_rule} against pinned {OUTSIDE_PINNED}")
    keep = unit_of.notna()
    per = pts.loc[keep].groupby(unit_of[keep])["pop"].agg(["size", "sum"])
    kn = float(per["sum"].sum()) / DE_JURE
    print(f"\n  Kontur kept {per['sum'].sum():,.0f} against the de jure {DE_JURE:,} (x{kn:.3f})")
    if not KONTUR_NATIONAL[0] <= kn <= KONTUR_NATIONAL[1]:
        fails.append(f"Kontur over the census {kn:.3f} outside {KONTUR_NATIONAL}")
    print(f"  {'région':<15}{'de jure':>9}{'table':>9}{'Kontur':>10}{'x':>7}{'/nat':>7}{'hexes':>7}")
    for u in REGIONS:
        k = float(per["sum"].get(u, 0.0))
        print(f"  {u:<15}{DE_JURE_REGION[u]:>9,}{want[u]:>9,}{k:>10,.0f}{k / DE_JURE_REGION[u]:>7.3f}"
              f"{k / DE_JURE_REGION[u] / kn:>7.3f}{int(per['size'].get(u, 0)):>7,}")
    free = [u for u in REGIONS if u not in KONTUR_UNIT_PINNED]
    try:
        geo_checks.ratio_band({u: DE_JURE_REGION[u] for u in free},
                              {u: per["sum"].get(u, 0.0) / kn for u in free},
                              *KONTUR_UNIT, what="région (Kontur 2023 over the 2024 de jure count, "
                                                 "over the national ratio)")
    except SystemExit as e:
        fails.append(str(e))
    for u, (lo, hi) in KONTUR_UNIT_PINNED.items():
        r = per["sum"].get(u, 0.0) / DE_JURE_REGION[u] / kn
        if not lo <= r <= hi:
            fails.append(f"{u}: Kontur {r:.3f} over the national ratio, outside its pinned ({lo}, {hi})")
    ks = per["sum"] / per["sum"].sum()
    l1_09 = 0.5 * sum(abs(ks[u] - CENSUS_2009[u] / sum(CENSUS_2009.values())) for u in REGIONS)
    l1_24 = 0.5 * sum(abs(ks[u] - DE_JURE_REGION[u] / DE_JURE) for u in REGIONS)
    print(f"  Kontur's région shares against the census's: half-L1 {l1_09:.3f} to 2009, "
          f"{l1_24:.3f} to 2024")
    if not l1_09 < l1_24 / 2:
        fails.append(f"Kontur is not nearer 2009's région shares ({l1_09:.3f}) than 2024's "
                     f"({l1_24:.3f}); the vintage reading of Tadjourah's pin does not hold")
    # Djibouti-Ville: the people in COD's polygon beyond a city of the office's area
    dv = pts.loc[keep & unit_of.eq("Djibouti-Ville"), "pop"].to_numpy()
    hex_km2 = float(hexes.loc[(keep & unit_of.eq("Djibouti-Ville")).to_numpy()].to_crs(eq).area.median() / 1e6)
    core_km2 = DE_JURE_REGION["Djibouti-Ville"] / DENSITY["Djibouti-Ville"][0]
    core = np.sort(dv)[::-1][:int(round(core_km2 / hex_km2))].sum() / dv.sum()
    print(f"  Djibouti-Ville: the densest {core_km2:,.0f} km2 hold {100 * core:.1f}% of its "
          f"{dv.sum():,.0f} Kontur people (hex {hex_km2:.3f} km2)")
    if core < CORE_MIN:
        fails.append(f"Djibouti-Ville: only {core:.3f} of Kontur's people in the densest {core_km2:,.0f} km2")

    # ---- 4. seats
    with zipfile.ZipFile(GEONAMES) as zf:
        gn = pd.read_csv(zf.open("DJ.txt"), sep="\t", header=None, names=GEONAMES_COLS,
                         quoting=3, dtype=str, keep_default_na=False)
    seats = gn[(gn["fclass"] == "P") & gn["fcode"].isin(["PPLA", "PPLC"])]
    gs = gpd.GeoDataFrame(seats[["asciiname", "fcode"]].copy(),
                          geometry=gpd.points_from_xy(seats["lon"].astype(float),
                                                      seats["lat"].astype(float)), crs=4326)
    gs = gpd.sjoin(gs, units[["unit", "geometry"]].to_crs(4326), how="left", predicate="within")
    kept = pts.loc[keep].to_crs(METRIC)
    print(f"\n  GeoNames seats ({len(gs)}):")
    seen = set()
    for _i, s in gs.iterrows():
        d = kept.distance(gpd.GeoSeries([s.geometry], crs=4326).to_crs(METRIC).iloc[0])
        k = float(kept.loc[d <= SEAT_KM * 1000, "pop"].sum())
        town = TOWNS.get(s["unit"])
        print(f"    {s['asciiname']:<14}{s['fcode']:<6}{str(s['unit']):<15} Kontur within "
              f"{SEAT_KM:g} km {k:>9,.0f}; town (Tableau 14) {town:,}; x{k / town:.2f}")
        seen.add(s["unit"])
        if k < SEAT_MIN * town * kn:
            fails.append(f"seat {s['asciiname']}: {k:,.0f} Kontur people within {SEAT_KM:g} km "
                         f"against {town:,} counted")
    if seen != set(REGIONS):
        fails.append(f"régions with no GeoNames seat: {sorted(set(REGIONS) - seen)}")
    if fails:
        raise SystemExit("région layer checks failed:\n  " + "\n  ".join(fails))

    # ---- 5. write
    outg = gpd.GeoDataFrame({"unit": unit_of[keep].to_numpy(), "pop": pts.loc[keep, "pop"].to_numpy()},
                            geometry=hexes.to_crs(units.crs).geometry[keep.to_numpy()].to_numpy(),
                            crs=units.crs)
    units = units.copy()
    units["census_pop"] = units["unit"].map(want).astype("int64")
    units["kontur_pop"] = units["unit"].map(per["sum"]).round().astype("int64")
    units["hexes"] = units["unit"].map(per["size"]).astype("int64")
    print(f"\n  drawn: COD-AB ADM1, {len(outg):,} hexes, {outg['pop'].sum():,.0f} people")
    os.makedirs(GEO, exist_ok=True)
    cols = ["unit", "adm1_pcode", "census_pop", "kontur_pop", "hexes", "area_km2", "geometry"]
    for path, gdf, layer in ((OUT_UNITS, units[cols], "regions"), (OUT_HEXES, outg, "hexes")):
        if os.path.exists(path + ".part"):
            os.remove(path + ".part")
        gdf.to_file(path + ".part", layer=layer, driver="GPKG")
        os.replace(path + ".part", path)
    with open(OUT_LOOKUP + ".part", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "adm1_pcode", "table_2024", "de_jure_2024", "area_cod_km2",
                    "kontur_2023", "hexes"])
        for u in REGIONS:
            w.writerow([u, by.loc[u, "adm1_pcode"], want[u], DE_JURE_REGION[u],
                        round(float(by.loc[u, "area_km2"]), 1), round(float(per["sum"][u])),
                        int(per["size"][u])])
    os.replace(OUT_LOOKUP + ".part", OUT_LOOKUP)
    print(f"\nwrote {OUT_UNITS}\nwrote {OUT_HEXES}\nwrote {OUT_LOOKUP}")


if __name__ == "__main__":
    main()
