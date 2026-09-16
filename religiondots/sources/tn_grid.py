"""Tunisia: the placement layer, Kontur 400 m population hexagons keyed to governorate.

Writes data/geo/tn/tn_hexes.gpkg.

Tunisia's south is Saharan: Tataouine and Kébili are 60,400 km2 between them, 37% of the country,
and held 2.9% of its people in 2024. Spread flat, their dots would sit on the erg; with Kontur an
empty hex takes no dots.

THE JOIN IS ON HEX CENTROIDS, so a hex on a governorate line belongs to one side and nobody is
counted twice. Hexes whose centroid is outside every governorate (the `TN` extract overruns into
Algeria and Libya, and the coast) are snapped to the nearest governorate within `SNAP_KM` when
they hold people (the coast and the Djerba causeway), and dropped beyond it.

THREE CHECKS AGAINST THE 2024 CENSUS, which Kontur 2023 is within a year of:

  * **the national ratio**, Kontur over the census, inside `EXPECTED_RATIO` +/- `TOLERANCE`;
  * **the rank witness for the join** in `sources/tn_geo.py`: Kontur people per governorate
    against the census count, Spearman, against every one of `N_PERM` shuffles of the census
    column. A permuted join would pair Tunis with Tataouine and fail it;
  * **every governorate seat** GeoNames lists with 50,000 or more people (PPLA, PPLC) holds at
    least `HOLE_RATIO` of that population within 5 km in Kontur (Algeria's Béchar and Morocco's
    Smara were holes). `KONTUR_HOLES` names the ones that fail, and none was expected.

Usage:
    python sources/tn_grid.py --fetch    one gzipped gpkg from Kontur (5.6 MB); GeoNames TN.zip
    python sources/tn_grid.py            rebuild from data/raw/tn/
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

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "tn")
GEO = os.path.join(ROOT, "data", "geo", "tn")
GOVS = os.path.join(GEO, "tn_governorates.gpkg")
OUT = os.path.join(GEO, "tn_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_TN_20231101.gpkg.gz")
GZ_NAME = "kontur_population_TN_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_TN_20231101.gpkg"
GEONAMES_URL = "https://download.geonames.org/export/dump/TN.zip"
GEONAMES = os.path.join(RAW, "geonames_TN.zip")

EXPECTED_UNITS = 24
# Kontur 2023 over the 6 November 2024 census. Written before the first run: about 1.0.
EXPECTED_RATIO = 1.0
TOLERANCE = 0.25
N_PERM = 20_000
SNAP_KM = 1.0

SEAT_MIN_POP = 50_000
HOLE_KM = 5.0
HOLE_RATIO = 0.10
KONTUR_HOLES = set()      # governorate seats Kontur has lost; asserted

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")

GEONAMES_COLS = ["geonameid", "name", "asciiname", "alternatenames", "lat", "lon", "fclass",
                 "fcode", "cc", "cc2", "admin1", "admin2", "admin3", "admin4", "population",
                 "elevation", "dem", "timezone", "modified"]


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
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        magic = fh.read(16)
    if magic[:4] != b"SQLi":
        raise SystemExit(f"{gpkg} is not a GeoPackage; starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def km(lat0, lon0, lat, lon):
    p = np.radians
    a = (np.sin(p(lat - lat0) / 2) ** 2
         + np.cos(p(lat0)) * np.cos(p(lat)) * np.sin(p(lon - lon0) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def seat_check(out, govs, names):
    """Kontur people within HOLE_KM of every GeoNames seat of SEAT_MIN_POP or more."""
    import geopandas as gpd

    with zipfile.ZipFile(GEONAMES) as zf:
        t = pd.read_csv(zf.open("TN.txt"), sep="\t", header=None, names=GEONAMES_COLS,
                        quoting=3, dtype=str, keep_default_na=False)
    s = t[t["fcode"].isin(["PPLA", "PPLC"])].copy()
    s["lat"], s["lon"] = s["lat"].astype(float), s["lon"].astype(float)
    s["population"] = pd.to_numeric(s["population"], errors="coerce").fillna(0).astype(int)
    pts = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s["lon"], s["lat"]), crs=4326)
    j = gpd.sjoin(pts, govs[["unit", "geometry"]], how="inner", predicate="within")
    print(f"\n  GeoNames: {len(s)} PPLA/PPLC places, {j['unit'].nunique()} of {len(govs)} "
          "governorates hold one")
    rows = []
    for _i, r in j[j["population"] >= SEAT_MIN_POP].iterrows():
        h = out[out["unit"] == r["unit"]]
        near = km(r["lat"], r["lon"], h["lat"].to_numpy(), h["lon"].to_numpy()) <= HOLE_KM
        rows.append((r["unit"], r["name"], int(r["population"]), float(h.loc[near, "pop"].sum())))
    c = pd.DataFrame(rows, columns=["unit", "seat", "geonames", "kontur"])
    c["ratio"] = c["kontur"] / c["geonames"]
    c = c.sort_values("ratio")
    print(f"  Kontur within {HOLE_KM:g} km of each seat of {SEAT_MIN_POP:,}+, lowest five of "
          f"{len(c)}:")
    for _i, r in c.head(5).iterrows():
        print(f"      {names[r['unit']]:<12} {r['seat']:<22} GeoNames {r['geonames']:>9,}   "
              f"Kontur {r['kontur']:>9,.0f}   {r['ratio']:.2f}")
    holes = set(c.loc[c["ratio"] < HOLE_RATIO, "unit"])
    if holes != KONTUR_HOLES:
        raise SystemExit(f"seats with under {HOLE_RATIO:.0%} of their people in Kontur: "
                         f"{sorted(holes)}, not {sorted(KONTUR_HOLES)}")


def main():
    import geopandas as gpd
    from geo_checks import read_layer

    gpkg = os.path.join(RAW, GPKG_NAME)
    if "--fetch" in sys.argv or not os.path.exists(gpkg) or not os.path.exists(GEONAMES):
        fetch()
    if not os.path.exists(GOVS):
        raise SystemExit(f"missing {GOVS}; run sources/tn_geo.py first")

    hexes = read_layer(gpkg, "Kontur TN")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, population {hexes[popcol].sum():,.0f}")

    govs = gpd.read_file(GOVS)
    if len(govs) != EXPECTED_UNITS:
        raise SystemExit(f"{GOVS} has {len(govs)} governorates, expected {EXPECTED_UNITS}")

    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()}, geometry=cent,
                           crs=hexes.crs).to_crs(govs.crs)
    hexes = hexes.to_crs(govs.crs)
    joined = gpd.sjoin(pts, govs[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    metric = "EPSG:32632"
    if outside.any():
        near = gpd.sjoin_nearest(pts.loc[outside].to_crs(metric),
                                 govs[["unit", "geometry"]].to_crs(metric),
                                 how="left", max_distance=SNAP_KM * 1000, distance_col="d")
        near = near[~near.index.duplicated(keep="first")]
        snapped = near["unit"].notna()
        joined.loc[near.index[snapped], "unit"] = near.loc[snapped, "unit"]
        print(f"\n  hexes whose centroid is outside every governorate: {int(outside.sum()):,} "
              f"({pts.loc[outside, popcol].sum():,.0f} people); snapped within {SNAP_KM:g} km: "
              f"{int(snapped.sum()):,} ({pts.loc[near.index[snapped], popcol].sum():,.0f} people)")
    dropped = joined["unit"].isna()
    print(f"  dropped (beyond {SNAP_KM:g} km of Tunisia's governorates): {int(dropped.sum()):,} "
          f"hexes, {pts.loc[dropped, popcol].sum():,.0f} people "
          f"({100.0 * pts.loc[dropped, popcol].sum() / pts[popcol].sum():.3f}%)")

    keep = ~dropped
    out = gpd.GeoDataFrame({"unit": joined.loc[keep, "unit"].to_numpy(),
                            "pop": pts.loc[keep, popcol].to_numpy(dtype=float),
                            "lat": pts.loc[keep].geometry.y.to_numpy(),
                            "lon": pts.loc[keep].geometry.x.to_numpy()},
                           geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=govs.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(govs["unit"]) - set(per.index))
    if missing or (per["sum"] <= 0).any():
        raise SystemExit(f"governorates with no populated hex: {missing}")
    census = dict(zip(govs["unit"], govs["pop"]))
    names = dict(zip(govs["unit"], govs["name"]))
    tot = float(out["pop"].sum())
    ratio = tot / sum(census.values())
    print(f"\n  Kontur {tot:,.0f} vs RGPH 2024 {sum(census.values()):,}: ratio {ratio:.3f} "
          f"(expected about {EXPECTED_RATIO})")
    if abs(ratio - EXPECTED_RATIO) > TOLERANCE:
        raise SystemExit("Kontur and the 2024 census disagree beyond the band")

    # ---- the join witness: Kontur's rank of the governorates against the census ----
    u = sorted(census)
    a = np.array([per.loc[x, "sum"] for x in u])
    b = np.array([census[x] for x in u], dtype=float)
    rho = stats.spearmanr(a, b).statistic
    rng = np.random.default_rng(0)
    perm = np.array([stats.spearmanr(a, rng.permutation(b)).statistic for _ in range(N_PERM)])
    beaten = int((perm >= rho).sum())
    print(f"  join witness: Spearman(Kontur, census) over 24 governorates = {rho:+.3f}; "
          f"{beaten} of {N_PERM:,} shuffles reach it (best {perm.max():+.3f})")
    if beaten:
        raise SystemExit("the rank witness fails; the governorate join may be permuted")

    print("\n  per-governorate Kontur 2023 / RGPH 2024, over the national ratio:")
    rel = {x: (per.loc[x, "sum"] / census[x]) / ratio for x in u}
    for x in sorted(rel, key=rel.get):
        print(f"      {names[x]:<12} {int(per.loc[x, 'size']):>6,} hexes   {rel[x]:5.2f}")

    seat_check(out, govs, names)

    os.makedirs(GEO, exist_ok=True)
    out[["unit", "pop", "geometry"]].to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} cells)")


if __name__ == "__main__":
    main()
