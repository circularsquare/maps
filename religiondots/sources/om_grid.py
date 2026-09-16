"""Oman: the placement layer, Kontur 400 m population hexagons inside the 61 wilayat.

Writes data/geo/om/om_hexes.gpkg (`unit`, `pop` as Kontur has it, geometry).

Oman is 310,000 km2 and most of it is desert; the register counts 1.5 million people in Muscat
governorate alone. Spread flat, a wilaya's dots would sit in the sand; with Kontur an empty hex
takes none.

The units are the register's own wilayat (`sources/om_geo.py`), so `scatter.py` allocates each
wilaya's dots from the end-2024 register counts before any weight is read, and Kontur only decides
where inside a wilaya. It is NOT calibrated: nothing finer than the wilaya is published. Kontur's
density cap is handled at scatter time by `kontur_cap.apply` against `kontur_cap.csv`.

THE JOIN IS ON HEX CENTROIDS to the 61 wilayat. Hexes whose centroid is outside every wilaya (coast,
reclaimed land, borders) are snapped to the nearest wilaya within `SNAP_KM` and dropped beyond it.

## CHECKS

  * **the national ratio**, Kontur (November 2023) over the register (December 2024), inside
    `EXPECTED_RATIO` +/- `TOLERANCE`;
  * **the rank witness for the name join**: Spearman of Kontur people per wilaya against the
    register, against `N_PERM` shuffles;
  * **per wilaya**, Kontur over the register over the national ratio, printed, with the units
    outside `UNIT_BAND` listed (not asserted: the dots follow the register's counts whatever
    Kontur says, and a new town the grid missed only moves dots inside its wilaya);
  * **no town lost**: every GeoNames seat (PPLC, PPLA, PPLA2) of `SEAT_MIN_POP` or more against
    Kontur's people within `HOLE_KM`, gated on its wilaya's ratio; `KONTUR_HOLES` names the holes;
  * **every wilaya has a populated hex**.

Usage:
    python sources/om_grid.py --fetch    one gzipped gpkg from Kontur (2.5 MB)
    python sources/om_grid.py            rebuild from data/raw/om/
"""

import gzip
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
RAW = os.path.join(ROOT, "data", "raw", "om")
GEO = os.path.join(ROOT, "data", "geo", "om")
UNITS_GPKG = os.path.join(GEO, "om_units.gpkg")
OUT = os.path.join(GEO, "om_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_OM_20231101.gpkg.gz")
GZ_NAME = "kontur_population_OM_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_OM_20231101.gpkg"
GEONAMES = os.path.join(RAW, "geonames_OM.zip")

EXPECTED_UNITS = 61
EXPECTED_RATIO = 1.0
TOLERANCE = 0.3
UNIT_BAND = (0.5, 2.0)
N_PERM = 20_000
SNAP_KM = 5.0
SNAP_BANDS_KM = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0)

SEAT_MIN_POP = 20_000
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
METRIC = "EPSG:32640"


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
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
        t = pd.read_csv(zf.open("OM.txt"), sep="\t", header=None, names=GEONAMES_COLS,
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
          f"of {SEAT_MIN_POP:,}+, with the wilaya's Kontur/register ratio:")
    for _i, r in c.iterrows():
        print(f"      {r['unit']:<32} {r['seat']:<22} GeoNames {r['geonames']:>9,}   Kontur "
              f"{r['kontur']:>10,.0f} ({r['ratio']:.2f})  {r['kontur_wide']:>10,.0f} "
              f"({r['ratio_wide']:.2f})   wilaya {r['u_ratio']:.2f}")
    holes = set(c.loc[(c["ratio"] < HOLE_RATIO) & (c["u_ratio"] < LOW_RATIO), "seat"])
    if holes != KONTUR_HOLES:
        raise SystemExit(f"seats with under {HOLE_RATIO:.0%} of their people in Kontur, in a wilaya "
                         f"under {LOW_RATIO} of its count: {sorted(holes)}, not {sorted(KONTUR_HOLES)}")


def main():
    import geopandas as gpd
    from geo_checks import read_layer

    gpkg = os.path.join(RAW, GPKG_NAME)
    if "--fetch" in sys.argv or not os.path.exists(gpkg):
        fetch()
    if not os.path.exists(UNITS_GPKG):
        raise SystemExit(f"missing {UNITS_GPKG}; run sources/om_geo.py first")

    hexes = read_layer(gpkg, "Kontur OM")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, population {hexes[popcol].sum():,.0f}")

    units = gpd.read_file(UNITS_GPKG)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS_GPKG} has {len(units)} wilayat, expected {EXPECTED_UNITS}")

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
        print(f"\n  hexes whose centroid is outside every wilaya: {int(outside.sum()):,} "
              f"({pts.loc[outside, popcol].sum():,.0f} people); people by distance:")
        for b in SNAP_BANDS_KM:
            m = near["d"] <= b * 1000
            print(f"      within {b:>4g} km: {int(m.sum()):>6,} hexes, "
                  f"{pts.loc[near.index[m], popcol].sum():>10,.0f} people")
        snapped = near["d"] <= SNAP_KM * 1000
        joined.loc[near.index[snapped], "unit"] = near.loc[snapped, "unit"]
        print(f"  snapped within {SNAP_KM:g} km: {int(snapped.sum()):,} hexes")
    dropped = joined["unit"].isna()
    print(f"  dropped (beyond {SNAP_KM:g} km of every wilaya): {int(dropped.sum()):,} hexes, "
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
    gov = dict(zip(units["unit"], units["governorate"]))
    missing = sorted(set(census) - set(per.index[per["sum"] > 0]))
    if missing:
        raise SystemExit(f"wilayat with no populated hex: {missing}")
    tot = float(out["pop"].sum())
    ratio = tot / sum(census.values())
    print(f"\n  Kontur {tot:,.0f} vs register end-2024 {sum(census.values()):,}: ratio {ratio:.3f}")
    if abs(ratio - EXPECTED_RATIO) > TOLERANCE:
        raise SystemExit("Kontur and the register disagree beyond the band")

    u = sorted(census)
    a = np.array([per.loc[x, "sum"] for x in u])
    b = np.array([census[x] for x in u], dtype=float)
    rho = stats.spearmanr(a, b).statistic
    rng = np.random.default_rng(0)
    perm = np.array([stats.spearmanr(a, rng.permutation(b)).statistic for _ in range(N_PERM)])
    beaten = int((perm >= rho).sum())
    print(f"  join witness: Spearman(Kontur, register) over {len(u)} wilayat = {rho:+.3f}; "
          f"{beaten} of {N_PERM:,} shuffles reach it (best {perm.max():+.3f})")
    if beaten:
        raise SystemExit("the rank witness fails; the wilaya join may be permuted")

    rel = {x: (per.loc[x, "sum"] / census[x]) / ratio for x in u}
    print("  per-wilaya Kontur / register, over the national ratio:")
    for x in sorted(rel, key=rel.get):
        flag = "" if UNIT_BAND[0] <= rel[x] <= UNIT_BAND[1] else "   <- outside UNIT_BAND"
        print(f"      {x:<32} {gov[x]:<20} {int(per.loc[x, 'size']):>6,} hexes  register "
              f"{census[x]:>9,}  {rel[x]:5.2f}{flag}")
    moved = 0.5 * sum(abs(per.loc[x, "sum"] / tot - census[x] / sum(census.values())) for x in u)
    print(f"  share of Kontur's people in a different wilaya from the register: {moved:.1%}")
    gsum = pd.Series({x: per.loc[x, "sum"] for x in u}).groupby(pd.Series(gov)).sum()
    gcen = pd.Series(census).groupby(pd.Series(gov)).sum()
    print("  per governorate, Kontur / register over the national ratio: "
          + ", ".join(f"{k} {v:.2f}" for k, v in ((gsum / gcen) / ratio).sort_values().items()))

    seat_check(out, units, rel)

    os.makedirs(GEO, exist_ok=True)
    out[["unit", "pop", "geometry"]].to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} cells)")


if __name__ == "__main__":
    main()
