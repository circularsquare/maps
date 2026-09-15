"""Taiwan: county polygons, and the Kontur 400 m grid that places dots inside them.

Writes data/geo/tw/tw_units.gpkg (22 counties, `unit` = ISO 3166-2), data/geo/tw/tw_hexes.gpkg
(the placement layer, `unit` and `pop`) and data/geo/tw/tw_lookup.csv.

**THE GOVERNMENT FILE IS WALLED FROM HERE.** The National Land Surveying and Mapping Center's
county boundaries (data.gov.tw dataset 7442, `COUNTY_MOI_1140318`) are served from `www.tgos.tw`,
which answered 403 to a scripted GET on 2026-09-14. Not retried; the URL is in sources/tw.md.

**So the polygons are geoBoundaries' TWN ADM1** (OpenStreetMap via Wambacher, 2017, ODbL),
22 features keyed by ISO 3166-2. 2017 is after the last change to the tier (Taoyuan, December
2014). The pairing of code and polygon is not taken on trust: each polygon's area is set against
the Ministry of the Interior's land area for that county (ODRP048, end of 2025), and each
county's Kontur population against the register, with a shuffle of the labels as the null.

Usage:
    python sources/tw_geo.py --fetch    geoBoundaries ADM1 and the Kontur TW extract
    python sources/tw_geo.py
"""

import gzip
import os
import shutil
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import geo_checks  # noqa: E402
from tw import UNITS, moi_counties  # noqa: E402

GEO = os.path.join(ROOT, "data", "geo", "tw")
GB_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/TWN/ADM1/"
          "geoBoundaries-TWN-ADM1.geojson")
GB = os.path.join(GEO, "geoBoundaries-TWN-ADM1.geojson")
KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
              "kontur_population_TW_20231101.gpkg.gz")
GZ = os.path.join(GEO, "kontur_population_TW_20231101.gpkg.gz")
GPKG = GZ[:-3]
UNITS_OUT = os.path.join(GEO, "tw_units.gpkg")
HEX_OUT = os.path.join(GEO, "tw_hexes.gpkg")
LOOKUP = os.path.join(GEO, "tw_lookup.csv")

METRIC = "EPSG:3826"            # TWD97 / TM2 zone 121
SNAP_M = 1000.0
# Kontur's TW extract reaches Taiping Island in the Spratlys (about 114.4 E) and Dongsha (116.7 E),
# which Kaohsiung administers, as well as Lanyu (121.6 E): about 7.7 degrees. A torn polygon spans
# hundreds. Those island hexes fall in no county polygon and are dropped below.
MAX_SPAN_DEG = 8.5
WEST_EAST = (114.0, 122.2)

# geoBoundaries' shapeName for each code, where it is not the English name's first word.
GB_NAME = {"TW-LIE": "Matsu Islands", "TW-CYI": "Chiayi", "TW-HSZ": "Hsinchu", "TW-PEN": "Penghu",
           "TW-KIN": "Kinmen", "TW-KEE": "Keelung", "TW-TXG": "Taichung", "TW-TAO": "Taoyuan",
           "TW-NWT": "New Taipei", "TW-TNN": "Tainan", "TW-TPE": "Taipei", "TW-KHH": "Kaohsiung"}


def fetch():
    import requests
    os.makedirs(GEO, exist_ok=True)
    for url, path in [(GB_URL, GB), (KONTUR_URL, GZ)]:
        if os.path.exists(path):
            continue
        r = requests.get(url, timeout=600, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(path + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(path + ".part", path)
        print(f"  {os.path.basename(path)}: {len(r.content):,} bytes")


def unpack():
    if os.path.exists(GPKG) and os.path.getsize(GPKG) > 50_000:
        return
    with gzip.open(GZ, "rb") as src, open(GPKG + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(GPKG + ".part", GPKG)
    with open(GPKG, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{GPKG} is not a GeoPackage")


def shuffle_null(a, b, observed, draws=20000, seed=0):
    """How many random pairings of `b` against `a` reach the observed Spearman."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(draws):
        if spearmanr(a, rng.permutation(b)).statistic >= observed:
            hits += 1
    return hits


def main():
    import geopandas as gpd
    from scipy.stats import spearmanr

    if "--fetch" in sys.argv:
        fetch()
    for p in (GB, GZ):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing; run with --fetch")
    unpack()

    moi = moi_counties()
    gb = geo_checks.read_layer(GB, "geoBoundaries TWN ADM1")
    if len(gb) != 22:
        raise SystemExit(f"geoBoundaries TWN ADM1 has {len(gb)} features, expected 22")
    if sorted(gb["shapeISO"]) != sorted(UNITS):
        raise SystemExit(f"shapeISO codes differ from UNITS: {sorted(set(gb['shapeISO']) ^ set(UNITS))}")
    for _, r in gb.iterrows():
        want = GB_NAME.get(r["shapeISO"], UNITS[r["shapeISO"]][1].split(" ")[0])
        if r["shapeName"].split(" County")[0] != want:
            raise SystemExit(f"{r['shapeISO']}: geoBoundaries calls it {r['shapeName']!r}, "
                             f"expected {want!r}")
    gb = gb.rename(columns={"shapeISO": "unit"})
    gb["name_en"] = gb["unit"].map(lambda u: UNITS[u][1])
    gb["name_zh"] = gb["unit"].map(lambda u: UNITS[u][0])
    gb = gb[["unit", "name_en", "name_zh", "geometry"]].to_crs("EPSG:4326")

    # ---- witness 1: polygon area against the MOI's land area ----------------------------------
    area = gb.to_crs(METRIC).set_index("unit").area / 1e6
    ratio = (area / moi["area"]).reindex(list(UNITS))
    r_area = spearmanr(moi["area"].reindex(list(UNITS)), area.reindex(list(UNITS))).statistic
    hits = shuffle_null(moi["area"].reindex(list(UNITS)).to_numpy(),
                        area.reindex(list(UNITS)).to_numpy(), r_area)
    print("witness 1, polygon area against MOI land area (km2):")
    for u in UNITS:
        print(f"    {u}  {UNITS[u][1]:<18}{moi.loc[u, 'area']:>9,.1f}{area[u]:>10,.1f}"
              f"{ratio[u]:>7.2f}")
    print(f"  Spearman {r_area:+.3f}; {hits} of 20,000 shuffled pairings reach it; "
          f"ratio {ratio.min():.2f} to {ratio.max():.2f}")

    # ---- the grid ---------------------------------------------------------------------------
    hexes = geo_checks.read_layer(GPKG, "Kontur TW 2023-11-01")
    popcol = next(c for c in hexes.columns if c.lower() == "population")
    kontur_total = float(hexes[popcol].sum())
    pts = gpd.GeoDataFrame({"pop": hexes[popcol].to_numpy(float)},
                           geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs("EPSG:4326")
    west, _, east, _ = pts.total_bounds
    span = east - west
    if span > MAX_SPAN_DEG or west < WEST_EAST[0] or east > WEST_EAST[1]:
        raise SystemExit(f"the grid runs {west:.2f} to {east:.2f} E ({span:.1f} degrees); "
                         "something is torn or the extract is not Taiwan's")
    print(f"\nKontur: {len(hexes):,} hexes, {kontur_total:,.0f} people "
          f"({kontur_total / moi['pop'].sum():.3f}x the register); {west:.2f} to {east:.2f} E")
    joined = gpd.sjoin(pts, gb[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    outside = joined["unit"].isna()
    print(f"  centroids outside every county: {int(outside.sum()):,} "
          f"({pts.loc[outside, 'pop'].sum():,.0f} people)")
    if outside.any():
        near = gpd.sjoin_nearest(pts.loc[outside].to_crs(METRIC), gb[["unit", "geometry"]].to_crs(METRIC),
                                 how="left", max_distance=SNAP_M, distance_col="_d")
        near = near[~near.index.duplicated(keep="first")]
        joined.loc[near.index, "unit"] = near["unit"]
        print(f"  snapped within {SNAP_M:,.0f} m: {int(near['unit'].notna().sum()):,} "
              f"({pts.loc[near.index[near['unit'].notna()], 'pop'].sum():,.0f} people)")
    keep = joined["unit"].notna().to_numpy()
    lost = float(pts.loc[~keep, "pop"].sum())
    print(f"  left unplaced and dropped: {int((~keep).sum()):,} hexes ({lost:,.0f} people, "
          f"{lost / kontur_total:.3%}); these are weights, not counts")
    out = gpd.GeoDataFrame({"unit": joined.loc[keep, "unit"].to_numpy(),
                            "pop": pts.loc[keep, "pop"].to_numpy()},
                           geometry=hexes.geometry[keep].to_numpy(), crs=hexes.crs).to_crs("EPSG:4326")
    missing = sorted(set(UNITS) - set(out["unit"]))
    if missing:
        raise SystemExit(f"counties with no hex: {missing}")

    # ---- witness 2: Kontur people per county against the register ---------------------------
    grid = out.groupby("unit")["pop"].sum().reindex(list(UNITS))
    kr = (grid / moi["pop"]).reindex(list(UNITS))
    r_pop = spearmanr(moi["pop"].reindex(list(UNITS)), grid).statistic
    hits2 = shuffle_null(moi["pop"].reindex(list(UNITS)).to_numpy(), grid.to_numpy(), r_pop)
    print("\nwitness 2, Kontur people against the MOI register, end of 2025:")
    for u in UNITS:
        print(f"    {u}  {UNITS[u][1]:<18}{moi.loc[u, 'pop']:>11,}{grid[u]:>12,.0f}{kr[u]:>7.2f}")
    print(f"  Spearman {r_pop:+.3f}; {hits2} of 20,000 shuffled pairings reach it; "
          f"ratio {kr.min():.2f} to {kr.max():.2f}")
    # The band covers drawn counties only. Kinmen reads 0.40 because its register is well known
    # to hold people who live on Taiwan proper; Kinmen is not drawn (sources/tw.py NOT_DRAWN).
    # Keelung's 1.71 is Kontur's and moves no dot between counties: counts come from the register.
    from tw import NOT_DRAWN
    drawn = [u for u in UNITS if u not in NOT_DRAWN]
    print(f"  drawn counties only: ratio {kr[drawn].min():.2f} to {kr[drawn].max():.2f}")
    if hits or hits2 or r_pop < 0.95 or kr[drawn].min() < 0.5 or kr[drawn].max() > 2.0:
        raise SystemExit("a county is paired with the wrong polygon, or the grid is badly off; "
                         "read both witnesses above")

    os.makedirs(GEO, exist_ok=True)
    gb.to_file(UNITS_OUT, driver="GPKG", layer="units")
    out.to_file(HEX_OUT, driver="GPKG", layer="hexes")
    look = pd.DataFrame({"geo_id": list(UNITS),
                         "name_en": [UNITS[u][1] for u in UNITS],
                         "name_zh": [UNITS[u][0] for u in UNITS],
                         "pop_moi_2025": moi["pop"].reindex(list(UNITS)).to_numpy(),
                         "area_moi_km2": moi["area"].reindex(list(UNITS)).round(2).to_numpy(),
                         "area_polygon_km2": area.reindex(list(UNITS)).round(2).to_numpy(),
                         "kontur_pop": grid.round().astype(int).to_numpy()})
    look.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"\nwrote {UNITS_OUT}, {HEX_OUT} ({len(out):,} hexes), {LOOKUP}")
    cells = out.groupby("unit").size()
    print(f"  hexes per county: median {cells.median():,.0f}, fewest {cells.min():,} "
          f"({UNITS[cells.idxmin()][1]})")


if __name__ == "__main__":
    main()
