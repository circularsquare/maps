"""Burkina Faso — the 45 provinces of the 2006 census, and the placement grid.

Writes:
    data/geo/bf/bf_provinces.gpkg   the 45 counted units (`units`)
    data/geo/bf/bf_hexes.gpkg       Kontur 400 m hexes with `unit` and `pop` (`place`)
    data/geo/bf/bf_lookup.csv       unit -> census population, Kontur population, areas

Usage:
    python sources/bf_geo.py --fetch    geoBoundaries ADM2 + ADM1 (~3 MB) + Kontur (~5 MB)
    python sources/bf_geo.py            rebuild from data/raw/bf/

THE BOUNDARIES ARE geoBoundaries `gbOpen/BFA/ADM2` (commit 9469f09, World Bank source,
`boundaryYearRepresented` 2017, CC BY 4.0): 45 provinces, the tier the census prints. COD-AB
Burkina Faso is no use here: its current version (v03, valid 2025-08-01) is the July 2025
reform's 17 régions and 47 provinces, and HDX keeps no older version.

THE JOIN IS BY NAME, folded, with one alias (`Komandjoari` in the census, `Komonjdjari` in
geoBoundaries), asserted to be a bijection. `boundaryYearRepresented` is a claim, not a check
(playbooks/geography.md), so two witnesses that the name key does not determine:

1. **Area.** Tableau A 3.1 bis of the same census volume prints each province's area. Every
   polygon must be within AREA_TOL of it; a province joined to the wrong polygon fails unless
   the two are the same size.
2. **Région.** geoBoundaries ADM1 is the 13 régions, drawn separately. Each province polygon's
   representative point must fall inside the région that `sources/bf.py::REGION_OF` gives it,
   and REGION_OF is itself proven by the census (A5.6 summed by région equals A5.5).

KONTUR IS NEEDED because the provinces are very unequal: Kadiogo is 2,869 km2 holding
1,727,390 people (Ouagadougou), Kompienga 6,967 km2 holding 75,867, and the northern and eastern
provinces have their people in villages along seasonal rivers. Kontur 2023-11 is used only as a
WITHIN-province weight.
"""

import csv
import gzip
import os
import re
import shutil
import sys
import unicodedata

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
RAW = os.path.join(ROOT, "data", "raw", "bf")
GEO = os.path.join(ROOT, "data", "geo", "bf")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur")
NORM = os.path.join(ROOT, "data", "normalized", "bf.csv")

GB = "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/BFA/{0}/geoBoundaries-BFA-{0}.geojson"
GB_ADM2 = os.path.join(RAW, "geoBoundaries-BFA-ADM2.geojson")
GB_ADM1 = os.path.join(RAW, "geoBoundaries-BFA-ADM1.geojson")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_BF_20231101.gpkg.gz")
GZ_NAME = "kontur_population_BF_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_BF_20231101.gpkg"

OUT_UNITS = os.path.join(GEO, "bf_provinces.gpkg")
OUT_HEXES = os.path.join(GEO, "bf_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "bf_lookup.csv")

UNITS = 45
REGIONS = 13
ALIASES = {"Komandjoari": ["Komonjdjari", "Komondjari", "Komandjari"]}
AREA_TOL = 0.04

# Census December 2006 against Kontur 2023-11: the 2019 census counted 18,171,751, 1.30x 2006
# in 12.9 years (2.0% a year), and four more years at that rate is about 1.40x. The band is wide
# around it, since displacement since 2019 moves people between provinces, not the total.
KONTUR_RATIO_MIN = 1.10
KONTUR_RATIO_MAX = 2.20

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _get(url, dst, first, min_size):
    import requests

    if os.path.exists(dst) and os.path.getsize(dst) > min_size:
        print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
        return
    print("GET", url)
    r = requests.get(url, timeout=1800, stream=True, headers=UA)
    r.raise_for_status()
    with open(dst + ".part", "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    with open(dst + ".part", "rb") as fh:
        head = fh.read(64).lstrip()
    # A 200 is not a download: GitHub and S3 both serve error pages happily.
    if not head.startswith(first):
        raise SystemExit(f"{dst}: starts {head[:16]!r}, expected {first!r}")
    os.replace(dst + ".part", dst)
    print(f"  got {os.path.getsize(dst):,} bytes")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(KONTUR, exist_ok=True)
    _get(GB.format("ADM2"), GB_ADM2, b"{", 500_000)
    _get(GB.format("ADM1"), GB_ADM1, b"{", 200_000)
    gz = os.path.join(KONTUR, GZ_NAME)
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    _get(GZ_URL, gz, b"\x1f\x8b", 1_000_000)
    if not (os.path.exists(gpkg) and os.path.getsize(gpkg) > 1_000_000):
        with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")
    print(f"  Kontur {os.path.getsize(gpkg):,} bytes")


def main():
    import geopandas as gpd
    import pandas as pd

    import bf

    if "--fetch" in sys.argv:
        fetch()
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    for p in (GB_ADM2, GB_ADM1, gpkg, NORM):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} — run sources/bf.py --fetch and "
                             "sources/bf_geo.py --fetch")
    os.makedirs(GEO, exist_ok=True)

    # ---- 1. the counted units, from the normalised file
    df = pd.read_csv(NORM, keep_default_na=False, na_values=[""])
    census = df[df["source_category"] == "Total"].groupby("geo_id")["count"].sum()
    if len(census) != UNITS:
        raise SystemExit(f"{len(census)} units in bf.csv, expected {UNITS}")
    # census name (bf.csv, CANON applied) -> A5.6 spelling, for bf.py's dicts
    printed = {bf.CANON.get(p, p): p for p in bf.A56}
    if set(printed) != set(census.index):
        raise SystemExit(f"bf.csv units differ from bf.A56: {sorted(set(census.index) ^ set(printed))}")

    g = gpd.read_file(GB_ADM2, engine="fiona")
    print(f"geoBoundaries BFA ADM2: {len(g)} features, crs={g.crs}")
    if len(g) != UNITS:
        raise SystemExit(f"{GB_ADM2} has {len(g)} features, expected {UNITS}")
    g["key"] = g["shapeName"].map(norm)
    if g["key"].duplicated().any():
        raise SystemExit(f"repeated folded names: {sorted(g.loc[g['key'].duplicated(), 'shapeName'])}")

    lut = {}
    for name in census.index:
        keys = {norm(name)} | {norm(a) for a in ALIASES.get(printed[name], [])}
        hits = g.index[g["key"].isin(keys)].tolist()
        if len(hits) != 1:
            raise SystemExit(f"census province {name!r} matched {len(hits)} polygons "
                             f"(tried {sorted(keys)}): {sorted(g['shapeName'])}")
        lut[name] = hits[0]
    if len(set(lut.values())) != UNITS:
        raise SystemExit("two census provinces matched the same polygon")
    print(f"  name join: {UNITS}/{UNITS} provinces matched one polygon each, 0 polygons unused")
    for name, i in sorted(lut.items()):
        if norm(g.loc[i, "shapeName"]) != norm(name):
            print(f"    census {name!r} <-> geoBoundaries {g.loc[i, 'shapeName']!r}")

    units = g.loc[list(lut.values())].copy()
    units["unit"] = list(lut.keys())
    units["census_pop"] = units["unit"].map(census).astype(int)
    units["region"] = units["unit"].map(lambda u: bf.REGION_OF[printed[u]])
    units["area_census"] = units["unit"].map(
        lambda u: bf.A31[bf.A31_NAME.get(printed[u], printed[u])][3])
    units["area_km2"] = units.to_crs(6933).geometry.area / 1e6
    units["area_ratio"] = units["area_km2"] / units["area_census"]

    # ---- witness 1: area against Tableau A 3.1 bis
    worst = units.loc[(units["area_ratio"] - 1).abs().idxmax()]
    off = units[(units["area_ratio"] - 1).abs() > AREA_TOL]
    print(f"\n  witness 1, area: polygon / census area {units['area_ratio'].min():.3f} to "
          f"{units['area_ratio'].max():.3f} (worst {worst['unit']}, {worst['area_km2']:,.0f} "
          f"against {worst['area_census']:,} km2)")
    if len(off):
        raise SystemExit("provinces whose polygon area is off the census area by more than "
                         f"{AREA_TOL:.0%}: " + ", ".join(
                             f"{r.unit} {r.area_ratio:.3f}" for r in off.itertuples()))
    near = [(a, b) for a in units.itertuples() for b in units.itertuples()
            if a.unit < b.unit and abs(a.area_census / b.area_census - 1) <= AREA_TOL]
    print(f"    {len(near)} pairs of provinces are within {AREA_TOL:.0%} of each other's area, "
          "so a swap inside a pair would pass witness 1; witness 2 covers the pairs in "
          f"different régions ({sum(1 for a, b in near if a.region != b.region)} of them)")

    # ---- witness 2: each province inside its census région
    r1 = gpd.read_file(GB_ADM1, engine="fiona")
    if len(r1) != REGIONS:
        raise SystemExit(f"{GB_ADM1} has {len(r1)} features, expected {REGIONS}")
    r1["key"] = r1["shapeName"].map(norm)
    want = {norm(r) for r in bf.A55}
    if set(r1["key"]) != want:
        raise SystemExit(f"ADM1 régions do not fold to the census's: {sorted(set(r1['key']) ^ want)}")
    pts = units[["unit", "region"]].copy()
    pts = gpd.GeoDataFrame(pts, geometry=units.to_crs(32630).geometry.representative_point()
                           .to_crs(units.crs), crs=units.crs)
    inside = gpd.sjoin(pts, r1[["key", "geometry"]], how="left", predicate="within")
    inside = inside[~inside.index.duplicated(keep="first")]
    wrong = inside[inside["key"] != inside["region"].map(norm)]
    if len(wrong):
        raise SystemExit("provinces outside their census région: " + ", ".join(
            f"{r.unit} ({r.region}) in {r.key}" for r in wrong.itertuples()))
    print(f"  witness 2, région: all {UNITS} provinces' representative points fall inside the "
          "geoBoundaries ADM1 région the census puts them in")

    units = units[["unit", "region", "census_pop", "area_census", "area_km2", "geometry"]]
    units.to_file(OUT_UNITS, layer="provinces", driver="GPKG")
    print(f"wrote {OUT_UNITS} ({len(units)} provinces)")

    # ---- 2. Kontur, joined on hex CENTROIDS
    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"\nKontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)
    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    # Burkina Faso is landlocked, so a centroid in no province is on the national border, where
    # geoBoundaries' line and Kontur's country cut differ. Snap to the nearest province within
    # SNAP_M; anything farther is dropped, and more than 2% dropped stops the build.
    SNAP_M = 2_000
    outside = joined["unit"].isna()
    stray = float(pts.loc[outside, popcol].sum())
    print(f"  hexes whose centroid is in no province: {int(outside.sum()):,} "
          f"({stray:,.0f} people, {100.0 * stray / float(pts[popcol].sum()):.3f}%)")
    if outside.any():
        utm = units[["unit", "geometry"]].to_crs(32630)
        near = gpd.sjoin_nearest(pts.loc[outside].to_crs(32630), utm, how="left",
                                 max_distance=SNAP_M, distance_col="dist")
        near = near[~near.index.duplicated(keep="first")]
        got = near["unit"].notna()
        joined.loc[near.index[got], "unit"] = near.loc[got, "unit"].to_numpy()
        print(f"  snapped {int(got.sum()):,} of them within {SNAP_M:,} m "
              f"({float(near.loc[got, popcol].sum()):,.0f} people)")
    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    share_lost = lost / float(pts[popcol].sum())
    print(f"  still in no province after the snap: {int(outside.sum()):,} hexes "
          f"({lost:,.0f} people, {100.0 * share_lost:.3f}%), dropped")
    if share_lost > 0.02:
        raise SystemExit(f"{100 * share_lost:.2f}% of Kontur falls outside every province even "
                         f"after a {SNAP_M:,} m snap; check the border")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"provinces with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"provinces whose hexes sum to zero population: {zero}")
    print(f"  every one of the {UNITS} provinces has hexes: "
          f"{per['size'].min():,}-{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    census_total = int(units["census_pop"].sum())
    ratio = tot / census_total
    print(f"\n  Kontur 2023 {tot:,.0f} vs the 2006 census {census_total:,}: ratio {ratio:.3f}")
    if not KONTUR_RATIO_MIN <= ratio <= KONTUR_RATIO_MAX:
        raise SystemExit(f"ratio {ratio:.3f} outside [{KONTUR_RATIO_MIN}, "
                         f"{KONTUR_RATIO_MAX}], check the download")
    print("     used only as a WITHIN-province weight, so the level does not matter and the "
          "shape does.")

    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT_HEXES} ({len(out):,} hexes)")

    per = per.rename(columns={"size": "hexes", "sum": "kontur_pop"})
    lk = units.drop(columns="geometry").merge(per, left_on="unit", right_index=True, how="left")
    lk["kontur_over_census"] = lk["kontur_pop"] / lk["census_pop"]
    with open(OUT_LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "region", "census_pop_2006", "kontur_pop_2023", "hexes",
                    "kontur_over_census", "area_census_km2", "area_polygon_km2"])
        for r in lk.sort_values("unit").itertuples(index=False):
            w.writerow([r.unit, r.region, int(r.census_pop), round(r.kontur_pop, 1),
                        int(r.hexes), round(r.kontur_over_census, 3), int(r.area_census),
                        round(r.area_km2, 1)])
    print(f"wrote {OUT_LOOKUP}")
    print("\n  per-province Kontur/census ratio, lowest and highest five:")
    s = lk.sort_values("kontur_over_census")
    for r in list(s.head(5).itertuples(index=False)) + list(s.tail(5).itertuples(index=False)):
        print(f"    {r.unit:<14} {r.region:<18} {r.kontur_over_census:5.2f}x  {int(r.hexes):>6,} hexes")


if __name__ == "__main__":
    main()
