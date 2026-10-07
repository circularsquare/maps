"""Lebanon: the placement layer, Kontur 400 m population hexagons inside the 26 cazas.

Writes data/geo/lb/lb_hexes.gpkg (`unit`, `pop` as Kontur has it, geometry).

`scatter.py` allocates each caza's dots from `sources/lb.py`'s counts before any weight is read, so
Kontur only decides where inside a caza. It is NOT calibrated: the Lebanese dots per caza follow the
register (where families are registered) and Kontur follows where people live, and calibrating one to
the other would mean nothing. Kontur's density cap is handled at scatter time by `kontur_cap.apply`.

THE JOIN IS ON HEX CENTROIDS. A hex whose centroid is outside every caza is classed (geography.md,
Bhutan and Namibia): inside a Natural Earth neighbour (Syria, Israel) or Shebaa Farms, it is the
neighbour's and dropped; otherwise it is coast and is snapped to the nearest caza within `SNAP_KM`.

## CHECKS

  * the national ratio, Kontur (November 2023) over OCHA's 2026 residents of every nationality
    (Lebanese, Syrians, Palestinians, migrants), inside `EXPECTED_RATIO` +/- `TOLERANCE`;
  * per caza, Kontur over OCHA's residents over the national ratio, PRINTED against `UNIT_BAND`
    beside the register's own share, which is expected to disagree (the ruling's whole caveat).
    Kontur fails the band in six cazas (Baabda 2.12, Akkar 0.23), so the band is not the witness;
    the stop is the cazas' rank against OCHA's, over `RANK_MIN` and the 99th percentile of 2,000
    shuffles, which a wrong join or a wrong boundary file would fail;
  * the dropped people, under `DROP_MAX` of Kontur's total.

Usage:
    python sources/lb_grid.py --fetch    Kontur LB (gzipped gpkg)
    python sources/lb_grid.py            rebuild from data/raw/lb/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "6")

import pandas as pd

RAW = os.path.join(ROOT, "data", "raw", "lb")
GEO = os.path.join(ROOT, "data", "geo", "lb")
UNITS_GPKG = os.path.join(GEO, "lb_units.gpkg")
OUT = os.path.join(GEO, "lb_hexes.gpkg")
NE = os.path.join(ROOT, "data", "geo", "ne_10m_admin_0_countries.geojson")
DISPUTED = os.path.join(ROOT, "data", "geo", "ne_10m_admin_0_disputed_areas.geojson")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_LB_20231101.gpkg.gz")
GZ_NAME = "kontur_population_LB_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_LB_20231101.gpkg"

EXPECTED_UNITS = 26
EXPECTED_RATIO = 1.0
TOLERANCE = 0.3
UNIT_BAND = (0.5, 2.0)       # printed only; see the rank witness in main()
RANK_MIN = 0.6
SNAP_KM = 2.0
SNAP_BANDS_KM = (0.5, 1.0, 2.0, 5.0)
DROP_MAX = 0.02
NEIGHBOURS = ("SYR", "ISR")
METRIC = "EPSG:32636"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 100_000:
        return
    if not os.path.exists(gz):
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=600, headers={"User-Agent": UA})
        r.raise_for_status()
        with open(gz + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(gz + ".part", gz)
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")


def main():
    import geopandas as gpd
    from geo_checks import read_layer
    from lb import read_ocha, PCODE

    gpkg = os.path.join(RAW, GPKG_NAME)
    if "--fetch" in sys.argv or not os.path.exists(gpkg):
        fetch()
    if not os.path.exists(UNITS_GPKG):
        raise SystemExit(f"missing {UNITS_GPKG}; run sources/lb_geo.py first")

    hexes = read_layer(gpkg, "Kontur LB")
    popcol = "population"
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, population {hexes[popcol].sum():,.0f}")
    units = gpd.read_file(UNITS_GPKG)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS_GPKG} has {len(units)} cazas, expected {EXPECTED_UNITS}")

    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()}, geometry=hexes.geometry.centroid,
                           crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)
    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    outside = joined["unit"].isna()
    print(f"\n  hexes whose centroid is outside every caza: {int(outside.sum()):,} "
          f"({pts.loc[outside, popcol].sum():,.0f} people)")
    if outside.any():
        ne = gpd.read_file(NE)
        nb = ne[ne["ADM0_A3"].isin(NEIGHBOURS)][["ADM0_A3", "geometry"]].to_crs(units.crs)
        dis = gpd.read_file(DISPUTED)
        sheb = dis[dis["BRK_NAME"] == "Shebaa Farms"][["geometry"]].to_crs(units.crs)
        sheb["ADM0_A3"] = "ISR (Shebaa Farms)"
        across = gpd.sjoin(pts.loc[outside], pd.concat([nb, sheb]), how="left",
                           predicate="within")
        across = across[~across.index.duplicated(keep="first")]
        nbr = across["ADM0_A3"]
        for k, v in nbr.fillna("coast or none").value_counts().items():
            print(f"      {k:<22} {v:>5,} hexes, "
                  f"{pts.loc[nbr.index[nbr.fillna('coast or none') == k], popcol].sum():>9,.0f} "
                  f"people")
        cand = nbr.index[nbr.isna()]
        near = gpd.sjoin_nearest(pts.loc[cand].to_crs(METRIC),
                                 units[["unit", "geometry"]].to_crs(METRIC),
                                 how="left", distance_col="d")
        near = near[~near.index.duplicated(keep="first")]
        for b in SNAP_BANDS_KM:
            m = near["d"] <= b * 1000
            print(f"      not in a neighbour, within {b:>4g} km of a caza: {int(m.sum()):>5,} "
                  f"hexes, {pts.loc[near.index[m], popcol].sum():>9,.0f} people")
        snapped = near["d"] <= SNAP_KM * 1000
        joined.loc[near.index[snapped], "unit"] = near.loc[snapped, "unit"]
        print(f"  snapped within {SNAP_KM:g} km: {int(snapped.sum()):,} hexes")
    dropped = joined["unit"].isna()
    share = pts.loc[dropped, popcol].sum() / pts[popcol].sum()
    print(f"  dropped: {int(dropped.sum()):,} hexes, {pts.loc[dropped, popcol].sum():,.0f} people "
          f"({share:.3%})")
    if share > DROP_MAX:
        raise SystemExit(f"more than {DROP_MAX:.0%} of Kontur's people dropped")

    keep = ~dropped
    out = gpd.GeoDataFrame({"unit": joined.loc[keep, "unit"].to_numpy(),
                            "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
                           geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)
    per = out.groupby("unit")["pop"].sum()
    missing = sorted(set(units["unit"]) - set(per.index[per > 0]))
    if missing:
        raise SystemExit(f"cazas with no populated hex: {missing}")

    ocha = read_ocha()
    res = (ocha[["lebanese", "syrian", "palestinian", "migrant"]].sum(axis=1)
           .rename(index=PCODE))
    drawn = dict(zip(units["unit"], units["pop"].astype(int)))
    tot = float(per.sum())
    ratio = tot / float(res.sum())
    print(f"\n  Kontur {tot:,.0f} vs OCHA's 2026 residents {res.sum():,.0f}: ratio {ratio:.3f}")
    if abs(ratio - EXPECTED_RATIO) > TOLERANCE:
        raise SystemExit("Kontur and OCHA's total disagree beyond the band")
    name = dict(zip(units["unit"], units["name"]))
    print("  per caza: Kontur / OCHA residents over the national ratio; then drawn / Kontur, "
          "which follows the register and is expected to differ")
    bad = []
    dtot = sum(drawn.values())
    for u in sorted(res.index, key=lambda x: -res[x]):
        rel = (per[u] / res[u]) / ratio
        print(f"      {u} {name[u]:<17} OCHA {res[u]:>9,.0f}  Kontur {per[u]:>10,.0f}  {rel:5.2f}"
              f"   drawn/Kontur {(drawn[u] / dtot) / (per[u] / tot):5.2f}")
        if not UNIT_BAND[0] <= rel <= UNIT_BAND[1]:
            bad.append(u)
    moved = 0.5 * sum(abs(per[u] / tot - res[u] / res.sum()) for u in res.index)
    print(f"  outside UNIT_BAND {UNIT_BAND}: {[name[u] for u in bad]}")
    print(f"  share of Kontur's people in a different caza from OCHA: {moved:.1%}")
    # The band is not a stop here (measured 2026-10-03, sources/lb.md §10.5): Kontur puts 1.2 million
    # people in Baabda against OCHA's 568,296 and 112,226 in Akkar against 484,765, while its
    # footprint covers every caza (Akkar: 912 populated hexes over 789 km2). Kontur is not
    # calibrated and only places dots inside a caza, so a caza total it gets wrong moves nobody
    # between cazas. The join witness is the rank of the cazas instead, against shuffles.
    import numpy as np
    from scipy.stats import spearmanr

    k = per.reindex(res.index).to_numpy()
    o = res.to_numpy()
    rho = spearmanr(k, o).correlation
    rng = np.random.default_rng(49)
    null = np.array([spearmanr(rng.permutation(k), o).correlation for _ in range(2000)])
    bar = float(np.quantile(null, 0.99))
    print(f"  rank witness: Spearman {rho:+.3f} over {len(o)} cazas, shuffled 99th percentile "
          f"{bar:+.3f}")
    if rho <= max(bar, RANK_MIN):
        raise SystemExit("Kontur's cazas do not rank like OCHA's: check the join")

    os.makedirs(GEO, exist_ok=True)
    out[["unit", "pop", "geometry"]].to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} cells)")


if __name__ == "__main__":
    main()
