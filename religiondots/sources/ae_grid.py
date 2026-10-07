"""United Arab Emirates: the placement layer, Kontur 400 m population hexagons inside the seven
emirates.

Writes data/geo/ae/ae_hexes.gpkg (`unit`, `pop` as Kontur has it, geometry).

The units are COD-AB's emirates (`sources/ae_geo.py`), so `scatter.py` allocates each emirate's
dots from `sources/ae.py`'s 2024 counts before any weight is read; Kontur only decides where inside
an emirate. It is NOT calibrated: nothing finer than the emirate is used. Kontur's density cap is
handled at scatter time by `kontur_cap.apply` against `kontur_cap.csv`.

THE JOIN IS ON HEX CENTROIDS. A hex whose centroid is outside every emirate is dropped if it falls
in Oman (Musandam, Madha, the Al Buraimi side of Al Ain: `playbooks/geography.md`, across a land
border a hex outside the units is the neighbour's town) and otherwise snapped to the nearest emirate
within `SNAP_KM` (coast, reclaimed islands), dropped beyond.

## CHECKS

  * **the national ratio**, Kontur (November 2023) over FCSC's 2024 total, inside
    `EXPECTED_RATIO` +/- `TOLERANCE`;
  * **per emirate**, Kontur over the drawn count over the national ratio, printed, on COD and on
    geoBoundaries' polygons both (the boundary witness the playbook asks for);
  * **the rank witness**: Spearman over seven emirates, against every shuffle (printed);
  * **no town lost**: every GeoNames seat (PPLC, PPLA, PPLA2) of `SEAT_MIN_POP` or more against
    Kontur's people within `HOLE_KM`.

Usage:
    python sources/ae_grid.py --fetch    Kontur AE (gzipped gpkg) and GeoNames AE
    python sources/ae_grid.py            rebuild from data/raw/ae/
"""

import gzip
import itertools
import os
import shutil
import sys
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
RAW = os.path.join(ROOT, "data", "raw", "ae")
GEO = os.path.join(ROOT, "data", "geo", "ae")
UNITS_GPKG = os.path.join(GEO, "ae_units.gpkg")
OMAN_UNITS = os.path.join(ROOT, "data", "geo", "om", "om_units.gpkg")
OUT = os.path.join(GEO, "ae_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_AE_20231101.gpkg.gz")
GZ_NAME = "kontur_population_AE_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_AE_20231101.gpkg"
GEONAMES_URL = "https://download.geonames.org/export/dump/AE.zip"
GEONAMES = os.path.join(RAW, "geonames_AE.zip")

EXPECTED_UNITS = 7
EXPECTED_RATIO = 1.0
TOLERANCE = 0.3
UNIT_BAND = (0.5, 2.0)
SNAP_KM = 2.0
SNAP_BANDS_KM = (0.5, 1.0, 2.0, 5.0, 10.0)
SEAT_MIN_POP = 20_000
HOLE_KM = 5.0
HOLE_RATIO = 0.10
LOW_RATIO = 0.5
KONTUR_HOLES = set()

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
GEONAMES_COLS = ["geonameid", "name", "asciiname", "alternatenames", "lat", "lon", "fclass",
                 "fcode", "cc", "cc2", "admin1", "admin2", "admin3", "admin4", "population",
                 "elevation", "dem", "timezone", "modified"]
METRIC = "EPSG:32640"


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if not (os.path.exists(gpkg) and os.path.getsize(gpkg) > 1_000_000):
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
    if not os.path.exists(GEONAMES):
        print("GET", GEONAMES_URL)
        r = requests.get(GEONAMES_URL, timeout=600, headers={"User-Agent": UA})
        r.raise_for_status()
        with open(GEONAMES + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(GEONAMES + ".part", GEONAMES)


def km(lat0, lon0, lat, lon):
    p = np.radians
    a = (np.sin(p(lat - lat0) / 2) ** 2
         + np.cos(p(lat0)) * np.cos(p(lat)) * np.sin(p(lon - lon0) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def seat_check(out, units, rel):
    """GeoNames' figure for each emirate's seat is the whole city or emirate (Dubai 3,790,000,
    Abu Dhabi 1,807,000), so the ratio within 5 km is low everywhere; as Oman, a seat counts as a
    hole only when its emirate also reads under `LOW_RATIO` of its drawn count."""
    import geopandas as gpd

    with zipfile.ZipFile(GEONAMES) as zf:
        t = pd.read_csv(zf.open("AE.txt"), sep="\t", header=None, names=GEONAMES_COLS,
                        quoting=3, dtype=str, keep_default_na=False)
    s = t[t["fcode"].isin(["PPLA", "PPLC", "PPLA2"])].copy()
    s["lat"], s["lon"] = s["lat"].astype(float), s["lon"].astype(float)
    s["population"] = pd.to_numeric(s["population"], errors="coerce").fillna(0).astype(int)
    pts = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s["lon"], s["lat"]), crs=4326)
    j = gpd.sjoin(pts, units[["unit", "geometry"]], how="inner", predicate="within")
    holes = set()
    print(f"\n  GeoNames seats of {SEAT_MIN_POP:,}+: Kontur people within {HOLE_KM:g} km")
    for _i, r in j[j["population"] >= SEAT_MIN_POP].sort_values("population").iterrows():
        h = out[out["unit"] == r["unit"]]
        d = km(r["lat"], r["lon"], h["lat"].to_numpy(), h["lon"].to_numpy())
        k = float(h.loc[d <= HOLE_KM, "pop"].sum())
        ratio = k / r["population"]
        print(f"      {r['unit']:<16} {r['name']:<24} GeoNames {r['population']:>9,}   Kontur "
              f"{k:>10,.0f} ({ratio:.2f})   emirate {rel[r['unit']]:.2f}")
        if ratio < HOLE_RATIO and rel[r["unit"]] < LOW_RATIO:
            holes.add(r["name"])
    if holes != KONTUR_HOLES:
        raise SystemExit(f"seats with under {HOLE_RATIO:.0%} of their people in Kontur: {sorted(holes)}")


def per_unit(pts, layer, popcol):
    import geopandas as gpd

    j = gpd.sjoin(pts, layer[["unit", "geometry"]], how="inner", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    return j.groupby("unit")[popcol].sum()


def main():
    import geopandas as gpd
    from geo_checks import read_layer
    from ae_geo import GB, GB_NAMES
    from ae import EMIRATES

    gpkg = os.path.join(RAW, GPKG_NAME)
    if "--fetch" in sys.argv or not os.path.exists(gpkg) or not os.path.exists(GEONAMES):
        fetch()
    if not os.path.exists(UNITS_GPKG):
        raise SystemExit(f"missing {UNITS_GPKG}; run sources/ae_geo.py first")

    hexes = read_layer(gpkg, "Kontur AE")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, population {hexes[popcol].sum():,.0f}")

    units = gpd.read_file(UNITS_GPKG)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS_GPKG} has {len(units)} emirates, expected {EXPECTED_UNITS}")

    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()}, geometry=cent,
                           crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)
    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    print(f"\n  hexes whose centroid is outside every emirate: {int(outside.sum()):,} "
          f"({pts.loc[outside, popcol].sum():,.0f} people)")
    oman = gpd.read_file(OMAN_UNITS).to_crs(units.crs) if os.path.exists(OMAN_UNITS) else None
    in_oman = pd.Series(False, index=pts.index)
    if oman is None:
        raise SystemExit(f"missing {OMAN_UNITS}; run sources/om_geo.py")
    if outside.any():
        oj = gpd.sjoin(pts.loc[outside], oman[["geometry"]], how="inner", predicate="within")
        in_oman.loc[oj.index.unique()] = True
        print(f"      inside Oman's wilayat, dropped: {int(in_oman.sum()):,} hexes, "
              f"{pts.loc[in_oman, popcol].sum():,.0f} people")
        rest = outside & ~in_oman
        near = gpd.sjoin_nearest(pts.loc[rest].to_crs(METRIC),
                                 units[["unit", "geometry"]].to_crs(METRIC),
                                 how="left", distance_col="d")
        near = near[~near.index.duplicated(keep="first")]
        for b in SNAP_BANDS_KM:
            m = near["d"] <= b * 1000
            print(f"      not in Oman, within {b:>4g} km of an emirate: {int(m.sum()):>6,} hexes, "
                  f"{pts.loc[near.index[m], popcol].sum():>10,.0f} people")
        snapped = near["d"] <= SNAP_KM * 1000
        joined.loc[near.index[snapped], "unit"] = near.loc[snapped, "unit"]
        print(f"  snapped within {SNAP_KM:g} km: {int(snapped.sum()):,} hexes")
    dropped = joined["unit"].isna()
    print(f"  dropped: {int(dropped.sum()):,} hexes, {pts.loc[dropped, popcol].sum():,.0f} people "
          f"({100.0 * pts.loc[dropped, popcol].sum() / pts[popcol].sum():.3f}%)")

    keep = ~dropped
    out = gpd.GeoDataFrame({"unit": joined.loc[keep, "unit"].to_numpy(),
                            "pop": pts.loc[keep, popcol].to_numpy(dtype=float),
                            "lat": pts.loc[keep].geometry.y.to_numpy(),
                            "lon": pts.loc[keep].geometry.x.to_numpy()},
                           geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["pop"].sum()
    drawn = dict(zip(units["unit"], units["pop"].astype(int)))
    missing = sorted(set(drawn) - set(per.index[per > 0]))
    if missing:
        raise SystemExit(f"emirates with no populated hex: {missing}")
    tot = float(per.sum())
    ratio = tot / sum(drawn.values())
    print(f"\n  Kontur {tot:,.0f} vs the drawn 2024 total {sum(drawn.values()):,}: ratio {ratio:.3f}")
    if abs(ratio - EXPECTED_RATIO) > TOLERANCE:
        raise SystemExit("Kontur and the drawn total disagree beyond the band")

    gb = gpd.read_file(GB).to_crs(units.crs)
    gb["unit"] = gb["shapeName"].map(GB_NAMES).map(EMIRATES)
    on_gb = per_unit(pts, gb, popcol)
    gb_tot = float(on_gb.sum())
    print("  per emirate: Kontur / drawn over the national ratio, on COD (as drawn) and on "
          "geoBoundaries")
    u = sorted(drawn)
    rel = {x: (per[x] / drawn[x]) / ratio for x in u}
    for x in sorted(u, key=lambda x: -drawn[x]):
        rel_gb = (on_gb.get(x, 0) / gb_tot) / (drawn[x] / sum(drawn.values()))
        flag = "" if UNIT_BAND[0] <= rel[x] <= UNIT_BAND[1] else "   <- outside UNIT_BAND"
        print(f"      {x:<16} drawn {drawn[x]:>10,}  Kontur {per[x]:>11,.0f}  {rel[x]:5.2f}   "
              f"gB {on_gb.get(x, 0):>11,.0f}  {rel_gb:5.2f}{flag}")
    a = np.array([per[x] for x in u])
    b = np.array([drawn[x] for x in u], dtype=float)
    rho = stats.spearmanr(a, b).statistic
    allp = [stats.spearmanr(a, b[list(p)]).statistic for p in itertools.permutations(range(len(u)))]
    reach = sum(1 for v in allp if v >= rho - 1e-12)
    print(f"  rank witness: Spearman over {len(u)} emirates {rho:+.3f}; {reach} of {len(allp):,} "
          f"orderings reach it")
    if reach > len(allp) * 0.01:
        raise SystemExit("the rank witness fails; the emirate join may be permuted")
    moved = 0.5 * sum(abs(per[x] / tot - drawn[x] / sum(drawn.values())) for x in u)
    print(f"  share of Kontur's people in a different emirate from the drawn counts: {moved:.1%}")

    seat_check(out, units, rel)

    os.makedirs(GEO, exist_ok=True)
    out[["unit", "pop", "geometry"]].to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} cells)")


if __name__ == "__main__":
    main()
