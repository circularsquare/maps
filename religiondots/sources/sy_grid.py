"""Syria: the placement layer, Kontur 400 m population hexagons keyed to governorate.

Writes data/geo/sy/sy_hexes.gpkg. Copied in shape from `sources/sd_grid.py`; `sources/sy.md` §5 is
the record.

Homs, Deir-ez-Zor, Rural Damascus, Ar-Raqqa and Al-Hasakeh hold the Syrian desert between them.
Spread flat, their dots would sit on the Badia; with Kontur an empty hex takes no dots.

THE JOIN IS ON HEX CENTROIDS. Hexes whose centroid falls inside the Golan polygon `sy_geo.py`
clipped (Natural Earth's "Golan Heights") are dropped first and never snapped: Majdal Shams is
about a kilometre from the ceasefire line and is drawn with Israel. Other hexes outside every
governorate are snapped to the nearest within `SNAP_KM` when they hold people and dropped beyond it.

THREE CHECKS AGAINST THE CBS 2011 ESTIMATE:

  * **the national ratio**, Kontur over CBS, inside `EXPECTED_RATIO` +/- `TOLERANCE`;
  * **the rank witness for the join**: Kontur people per governorate against CBS, Spearman,
    against `N_PERM` shuffles;
  * **no town lost**: every GeoNames seat of `SEAT_MIN_POP` or more, Kontur within 5 and 10 km,
    a hole counted only in a governorate under `LOW_RATIO` of its share (Morocco's gate).

Usage:
    python sources/sy_grid.py --fetch    Kontur SY (about 5 MB gzipped) and GeoNames SY.zip
    python sources/sy_grid.py            rebuild from data/raw/sy/
"""

import gzip
import json
import os
import shutil
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "sy")
GEO = os.path.join(ROOT, "data", "geo", "sy")
UNITS = os.path.join(GEO, "sy_governorates.gpkg")
OUT = os.path.join(GEO, "sy_hexes.gpkg")
DISPUTED = os.path.join(ROOT, "data", "geo", "ne_10m_admin_0_disputed_areas.geojson")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_SY_20231101.gpkg.gz")
GZ_NAME = "kontur_population_SY_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_SY_20231101.gpkg"
GEONAMES_URL = "https://download.geonames.org/export/dump/SY.zip"
GEONAMES = os.path.join(RAW, "geonames_SY.zip")

EXPECTED_UNITS = 14
EXPECTED_RATIO = 1.0
TOLERANCE = 0.25
N_PERM = 20_000
SNAP_KM = 2.0
SNAP_BANDS_KM = (0.5, 1.0, 2.0, 5.0, 10.0)
GOLAN = "Golan Heights"

SEAT_MIN_POP = 50_000
HOLE_KM = 5.0
WIDE_KM = 10.0
HOLE_RATIO = 0.10
LOW_RATIO = 0.5
KONTUR_HOLES = set()

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
GEONAMES_COLS = ["geonameid", "name", "asciiname", "alternatenames", "lat", "lon", "fclass",
                 "fcode", "cc", "cc2", "admin1", "admin2", "admin3", "admin4", "population",
                 "elevation", "dem", "timezone", "modified"]
METRIC = "EPSG:32637"


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if not os.path.exists(GEONAMES):
        req = urllib.request.Request(GEONAMES_URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        if data[:2] != b"PK":
            raise SystemExit(f"{GEONAMES_URL} is not a zip")
        with open(GEONAMES, "wb") as fh:
            fh.write(data)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 1_000_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz):
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=1800, stream=True, headers={"User-Agent": UA})
        r.raise_for_status()
        with open(gz + ".part", "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        os.replace(gz + ".part", gz)
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")


def km(lat0, lon0, lat, lon):
    p = np.radians
    a = (np.sin(p(lat - lat0) / 2) ** 2
         + np.cos(p(lat0)) * np.cos(p(lat)) * np.sin(p(lon - lon0) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def seat_check(out, units, names, rel):
    import geopandas as gpd

    with zipfile.ZipFile(GEONAMES) as zf:
        t = pd.read_csv(zf.open("SY.txt"), sep="\t", header=None, names=GEONAMES_COLS,
                        quoting=3, dtype=str, keep_default_na=False)
    s = t[t["fcode"].isin(["PPLA", "PPLA2", "PPLC"])].copy()
    s["lat"], s["lon"] = s["lat"].astype(float), s["lon"].astype(float)
    s["population"] = pd.to_numeric(s["population"], errors="coerce").fillna(0).astype(int)
    pts = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s["lon"], s["lat"]), crs=4326)
    j = gpd.sjoin(pts, units[["unit", "geometry"]], how="inner", predicate="within")
    print(f"\n  GeoNames: {len(s)} PPLA/PPLA2/PPLC places, {j['unit'].nunique()} of {len(units)} "
          "governorates hold one")
    rows = []
    for _i, r in j[j["population"] >= SEAT_MIN_POP].iterrows():
        h = out[out["unit"] == r["unit"]]
        d = km(r["lat"], r["lon"], h["lat"].to_numpy(), h["lon"].to_numpy())
        rows.append((r["unit"], r["name"], int(r["population"]),
                     float(h.loc[d <= HOLE_KM, "pop"].sum()), float(h.loc[d <= WIDE_KM, "pop"].sum())))
    c = pd.DataFrame(rows, columns=["unit", "seat", "geonames", "kontur", "kontur_wide"])
    c["ratio"] = c["kontur"] / c["geonames"]
    c["ratio_wide"] = c["kontur_wide"] / c["geonames"]
    c["gov"] = c["unit"].map(rel)
    c = c.sort_values("ratio")
    print(f"  Kontur within {HOLE_KM:g} and {WIDE_KM:g} km of each seat of {SEAT_MIN_POP:,}+, "
          f"all {len(c)}, with the governorate's own Kontur/CBS ratio:")
    for _i, r in c.iterrows():
        print(f"      {names[r['unit']]:<15} {r['seat']:<18} GeoNames {r['geonames']:>9,}   "
              f"Kontur {r['kontur']:>9,.0f} ({r['ratio']:.2f})  {r['kontur_wide']:>9,.0f} "
              f"({r['ratio_wide']:.2f})   gov {r['gov']:.2f}")
    holes = set(c.loc[(c["ratio"] < HOLE_RATIO) & (c["gov"] < LOW_RATIO), "unit"])
    if holes != KONTUR_HOLES:
        raise SystemExit(f"seats with under {HOLE_RATIO:.0%} of their people in Kontur, in a "
                         f"governorate under {LOW_RATIO} of its share: {sorted(holes)}, not "
                         f"{sorted(KONTUR_HOLES)}")


def main():
    import geopandas as gpd
    from shapely.geometry import shape
    from geo_checks import read_layer

    gpkg = os.path.join(RAW, GPKG_NAME)
    if "--fetch" in sys.argv or not os.path.exists(gpkg) or not os.path.exists(GEONAMES):
        fetch()
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS}; run sources/sy_geo.py first")

    hexes = read_layer(gpkg, "Kontur SY")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, population {hexes[popcol].sum():,.0f}")

    units = gpd.read_file(UNITS)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS} has {len(units)} governorates, expected {EXPECTED_UNITS}")

    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()}, geometry=cent,
                           crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)

    ne = json.load(open(DISPUTED, encoding="utf-8"))
    gol = [shape(f["geometry"]) for f in ne["features"] if f["properties"].get("BRK_NAME") == GOLAN]
    if len(gol) != 1:
        raise SystemExit(f"{DISPUTED} has {len(gol)} '{GOLAN}' features")
    gol = gpd.GeoSeries(gol, crs=4326).to_crs(units.crs).iloc[0]
    in_golan = pts.within(gol).to_numpy()
    print(f"  hexes in the Golan (drawn with Israel), dropped before any snap: {int(in_golan.sum()):,} "
          f"({pts.loc[in_golan, popcol].sum():,.0f} people)")

    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    joined.loc[in_golan, "unit"] = None

    outside = joined["unit"].isna() & ~in_golan
    if outside.any():
        near = gpd.sjoin_nearest(pts.loc[outside].to_crs(METRIC),
                                 units[["unit", "geometry"]].to_crs(METRIC),
                                 how="left", distance_col="d")
        near = near[~near.index.duplicated(keep="first")]
        print(f"\n  other hexes whose centroid is outside every governorate: {int(outside.sum()):,} "
              f"({pts.loc[outside, popcol].sum():,.0f} people); people by distance:")
        for b in SNAP_BANDS_KM:
            m = near["d"] <= b * 1000
            print(f"      within {b:>4g} km: {int(m.sum()):>6,} hexes, "
                  f"{pts.loc[near.index[m], popcol].sum():>10,.0f} people")
        snapped = near["d"] <= SNAP_KM * 1000
        joined.loc[near.index[snapped], "unit"] = near.loc[snapped, "unit"]
        print(f"  snapped within {SNAP_KM:g} km: {int(snapped.sum()):,} hexes")
    dropped = joined["unit"].isna()
    print(f"  dropped (the Golan, and beyond {SNAP_KM:g} km of every governorate): "
          f"{int(dropped.sum()):,} hexes, {pts.loc[dropped, popcol].sum():,.0f} people "
          f"({100.0 * pts.loc[dropped, popcol].sum() / pts[popcol].sum():.3f}%)")

    keep = ~dropped
    out = gpd.GeoDataFrame({"unit": joined.loc[keep, "unit"].to_numpy(),
                            "pop": pts.loc[keep, popcol].to_numpy(dtype=float),
                            "lat": pts.loc[keep].geometry.y.to_numpy(),
                            "lon": pts.loc[keep].geometry.x.to_numpy()},
                           geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing or (per["sum"] <= 0).any():
        raise SystemExit(f"governorates with no populated hex: {missing}")
    est = dict(zip(units["unit"], units["pop"]))
    names = dict(zip(units["unit"], units["name"]))
    tot = float(out["pop"].sum())
    ratio = tot / sum(est.values())
    print(f"\n  Kontur {tot:,.0f} vs CBS end-2011 {sum(est.values()):,}: ratio {ratio:.3f} "
          f"(expected about {EXPECTED_RATIO})")
    if abs(ratio - EXPECTED_RATIO) > TOLERANCE:
        raise SystemExit("Kontur and the CBS estimate disagree beyond the band")

    u = sorted(est)
    a = np.array([per.loc[x, "sum"] for x in u])
    b = np.array([est[x] for x in u], dtype=float)
    rho = stats.spearmanr(a, b).statistic
    rng = np.random.default_rng(0)
    perm = np.array([stats.spearmanr(a, rng.permutation(b)).statistic for _ in range(N_PERM)])
    beaten = int((perm >= rho).sum())
    print(f"  join witness: Spearman(Kontur, CBS) over {len(u)} governorates = {rho:+.3f}; "
          f"{beaten} of {N_PERM:,} shuffles reach it (best {perm.max():+.3f})")
    if beaten:
        raise SystemExit("the rank witness fails; the governorate join may be permuted")

    print("\n  per-governorate Kontur 2023 / CBS 2011, over the national ratio:")
    rel = {x: (per.loc[x, "sum"] / est[x]) / ratio for x in u}
    for x in sorted(rel, key=rel.get):
        print(f"      {names[x]:<15} {int(per.loc[x, 'size']):>7,} hexes   {rel[x]:5.2f}")

    seat_check(out, units, names, rel)

    os.makedirs(GEO, exist_ok=True)
    out[["unit", "pop", "geometry"]].to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} cells)")


if __name__ == "__main__":
    main()
