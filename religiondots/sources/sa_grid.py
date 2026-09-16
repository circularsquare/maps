"""Saudi Arabia: the placement layer, Kontur 400 m population hexagons inside the 13 regions.

Writes data/geo/sa/sa_hexes.gpkg (`unit`, `pop` as Kontur has it, geometry).

Saudi Arabia is 1.9 million km2 and most of it is empty; Riyadh, Jeddah, Makkah, Madinah and
Dammam hold 49.2% of the 2022 census (report, page 16). Spread flat, a region's dots would sit in
the Empty Quarter; with Kontur an empty hex takes none.

Nothing finer than the region is published with Saudi and non-Saudi counts (the Royal Commission for
Riyadh City's portal splits Ar Riyadh region into three parts and nothing else), so Kontur is NOT
calibrated: it decides where people are inside a region, and `scatter.py` allocates each region's
dots from the census counts before any weight is read. Kontur's density cap is handled at scatter
time by `kontur_cap.apply` against `kontur_cap.csv`.

THE JOIN IS ON HEX CENTROIDS to COD-AB's 13 regions (`sources/sa_geo.py`). Hexes whose centroid is
outside every region (coast, reclaimed land, borders) are snapped to the nearest region within
`SNAP_KM` and dropped beyond it.

## CHECKS

  * **the national ratio**, Kontur 2023 over the census of May 2022, inside `EXPECTED_RATIO` +/-
    `TOLERANCE`;
  * **per region**, Kontur over the census over the national ratio, inside `REGION_BAND`;
  * **the rank witness for the join**: Spearman of Kontur people per region against the census,
    against `N_PERM` shuffles;
  * **no town lost**: every GeoNames seat (PPLA, PPLC) of `SEAT_MIN_POP` or more against Kontur's
    people within `HOLE_KM`, gated on its region's ratio; `KONTUR_HOLES` names the holes.

Usage:
    python sources/sa_grid.py --fetch    one gzipped gpkg from Kontur (17 MB); GeoNames SA.zip
    python sources/sa_grid.py            rebuild from data/raw/sa/
"""

import gzip
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
RAW = os.path.join(ROOT, "data", "raw", "sa")
GEO = os.path.join(ROOT, "data", "geo", "sa")
REGIONS_GPKG = os.path.join(GEO, "sa_regions.gpkg")
OUT = os.path.join(GEO, "sa_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_SA_20231101.gpkg.gz")
GZ_NAME = "kontur_population_SA_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_SA_20231101.gpkg"
GEONAMES_URL = "https://download.geonames.org/export/dump/SA.zip"
GEONAMES = os.path.join(RAW, "geonames_SA.zip")

EXPECTED_UNITS = 13
EXPECTED_RATIO = 1.0
TOLERANCE = 0.3
REGION_BAND = (0.6, 1.6)
N_PERM = 20_000
SNAP_KM = 5.0
SNAP_BANDS_KM = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0)

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
METRIC = "EPSG:32638"


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if not os.path.exists(GEONAMES):
        req = urllib.request.Request(GEONAMES_URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        if data[:2] != b"PK":
            raise SystemExit(f"{GEONAMES_URL} is not a zip")
        with open(GEONAMES + ".part", "wb") as fh:
            fh.write(data)
        os.replace(GEONAMES + ".part", GEONAMES)
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


def seat_check(out, units, rel):
    import geopandas as gpd

    with zipfile.ZipFile(GEONAMES) as zf:
        t = pd.read_csv(zf.open("SA.txt"), sep="\t", header=None, names=GEONAMES_COLS,
                        quoting=3, dtype=str, keep_default_na=False)
    s = t[t["fcode"].isin(["PPLA", "PPLC", "PPLA2"])].copy()
    s["lat"], s["lon"] = s["lat"].astype(float), s["lon"].astype(float)
    s["population"] = pd.to_numeric(s["population"], errors="coerce").fillna(0).astype(int)
    pts = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s["lon"], s["lat"]), crs=4326)
    j = gpd.sjoin(pts, units[["unit", "geometry"]], how="inner", predicate="within")
    rows = []
    for _i, r in j[j["population"] >= SEAT_MIN_POP].iterrows():
        h = out[out["unit"] == r["unit"]]
        d = km(r["lat"], r["lon"], h["lat"].to_numpy(), h["lon"].to_numpy())
        rows.append((r["unit"], r["name"], int(r["population"]),
                     float(h.loc[d <= HOLE_KM, "pop"].sum()),
                     float(h.loc[d <= WIDE_KM, "pop"].sum())))
    c = pd.DataFrame(rows, columns=["unit", "seat", "geonames", "kontur", "kontur_wide"])
    c["ratio"] = c["kontur"] / c["geonames"]
    c["ratio_wide"] = c["kontur_wide"] / c["geonames"]
    c["u_ratio"] = c["unit"].map(rel)
    c = c.sort_values("ratio")
    print(f"\n  GeoNames: Kontur within {HOLE_KM:g} and {WIDE_KM:g} km of each of the {len(c)} seats "
          f"of {SEAT_MIN_POP:,}+, with the region's Kontur/census ratio:")
    for _i, r in c.iterrows():
        print(f"      {r['unit']}  {r['seat']:<22} GeoNames {r['geonames']:>9,}   Kontur "
              f"{r['kontur']:>10,.0f} ({r['ratio']:.2f})  {r['kontur_wide']:>10,.0f} "
              f"({r['ratio_wide']:.2f})   region {r['u_ratio']:.2f}")
    holes = set(c.loc[(c["ratio"] < HOLE_RATIO) & (c["u_ratio"] < LOW_RATIO), "seat"])
    if holes != KONTUR_HOLES:
        raise SystemExit(f"seats with under {HOLE_RATIO:.0%} of their people in Kontur, in a region "
                         f"under {LOW_RATIO} of its count: {sorted(holes)}, not {sorted(KONTUR_HOLES)}")


def main():
    import geopandas as gpd
    from geo_checks import read_layer

    gpkg = os.path.join(RAW, GPKG_NAME)
    if "--fetch" in sys.argv or not os.path.exists(gpkg) or not os.path.exists(GEONAMES):
        fetch()
    if not os.path.exists(REGIONS_GPKG):
        raise SystemExit(f"missing {REGIONS_GPKG}; run sources/sa_geo.py first")

    hexes = read_layer(gpkg, "Kontur SA")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, population {hexes[popcol].sum():,.0f}")

    units = gpd.read_file(REGIONS_GPKG)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{REGIONS_GPKG} has {len(units)} regions, expected {EXPECTED_UNITS}")

    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()}, geometry=cent,
                           crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)
    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    if outside.any():
        near = gpd.sjoin_nearest(pts.loc[outside].to_crs(METRIC),
                                 units[["unit", "geometry"]].to_crs(METRIC),
                                 how="left", distance_col="d")
        near = near[~near.index.duplicated(keep="first")]
        print(f"\n  hexes whose centroid is outside every region: {int(outside.sum()):,} "
              f"({pts.loc[outside, popcol].sum():,.0f} people); people by distance:")
        for b in SNAP_BANDS_KM:
            m = near["d"] <= b * 1000
            print(f"      within {b:>4g} km: {int(m.sum()):>6,} hexes, "
                  f"{pts.loc[near.index[m], popcol].sum():>10,.0f} people")
        snapped = near["d"] <= SNAP_KM * 1000
        joined.loc[near.index[snapped], "unit"] = near.loc[snapped, "unit"]
        print(f"  snapped within {SNAP_KM:g} km: {int(snapped.sum()):,} hexes")
    dropped = joined["unit"].isna()
    print(f"  dropped (beyond {SNAP_KM:g} km of every region): {int(dropped.sum()):,} hexes, "
          f"{pts.loc[dropped, popcol].sum():,.0f} people "
          f"({100.0 * pts.loc[dropped, popcol].sum() / pts[popcol].sum():.3f}%)")

    keep = ~dropped
    out = gpd.GeoDataFrame({"unit": joined.loc[keep, "unit"].to_numpy(),
                            "pop": pts.loc[keep, popcol].to_numpy(dtype=float),
                            "lat": pts.loc[keep].geometry.y.to_numpy(),
                            "lon": pts.loc[keep].geometry.x.to_numpy()},
                           geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    census = dict(zip(units["unit"], units["pop"].astype(int)))
    names = dict(zip(units["unit"], units["name"]))
    missing = sorted(set(census) - set(per.index[per["sum"] > 0]))
    if missing:
        raise SystemExit(f"regions with no populated hex: {missing}")
    tot = float(out["pop"].sum())
    ratio = tot / sum(census.values())
    print(f"\n  Kontur {tot:,.0f} vs census 2022 {sum(census.values()):,}: ratio {ratio:.3f}")
    if abs(ratio - EXPECTED_RATIO) > TOLERANCE:
        raise SystemExit("Kontur and the census disagree beyond the band")

    u = sorted(census)
    a = np.array([per.loc[x, "sum"] for x in u])
    b = np.array([census[x] for x in u], dtype=float)
    rho = stats.spearmanr(a, b).statistic
    rng = np.random.default_rng(0)
    perm = np.array([stats.spearmanr(a, rng.permutation(b)).statistic for _ in range(N_PERM)])
    beaten = int((perm >= rho).sum())
    print(f"  join witness: Spearman(Kontur, census) over {len(u)} regions = {rho:+.3f}; "
          f"{beaten} of {N_PERM:,} shuffles reach it (best {perm.max():+.3f})")
    if beaten:
        raise SystemExit("the rank witness fails; the region join may be permuted")

    rel = {x: (per.loc[x, "sum"] / census[x]) / ratio for x in u}
    print("  per-region Kontur / census, over the national ratio:")
    for x in sorted(rel, key=rel.get):
        print(f"      {x} {names[x]:<18} {int(per.loc[x, 'size']):>7,} hexes  census "
              f"{census[x]:>10,}  {rel[x]:5.2f}")
    moved = 0.5 * sum(abs(per.loc[x, "sum"] / tot - census[x] / sum(census.values())) for x in u)
    print(f"  share of Kontur's people in a different region from the census: {moved:.1%}")
    out_band = {names[x]: round(r, 2) for x, r in rel.items()
                if not REGION_BAND[0] <= r <= REGION_BAND[1]}
    if out_band:
        raise SystemExit(f"regions outside {REGION_BAND}: {out_band}")

    seat_check(out, units, rel)

    os.makedirs(GEO, exist_ok=True)
    out[["unit", "pop", "geometry"]].to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} cells)")


if __name__ == "__main__":
    main()
