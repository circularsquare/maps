"""Kuwait: the placement layer, Kontur 400 m hexagons cut to the 143 drawn units, used only for
WHERE people live inside a unit and not for how many.

Writes data/geo/kw/kw_hexes.gpkg (`unit`, `pop`, geometry).

## KONTUR'S COUNTS ARE WRONG IN KUWAIT, SO ONLY ITS FOOTPRINT IS USED

Against the 2021 census per unit, Kontur (November 2023) is far off in both directions: Sabah
Al-Salem holds 2,785 of the census's 88,904, Mishrief 1,251 of 45,877, Al-Mahbula 13,886 of
142,145, while Wafra Farms holds 257,607 of 11,961, Al-Abdalli 103,751 of 10,169 and the Jahra
desert 150,757 of 2,752. Per unit of 1,000 or more people its share over the census share runs
from p10 0.12 to p90 15.6, so the bar set before reading (p10 at least 0.4, p90 at most 2.5)
fails, and inside a unit its counts would pile a suburb's dots onto a few edge hexes.

What Kontur does get right is where anybody lives at all. So, as DR Congo's empty-territory fix
(`sources/cd_geo.py`, `HOLE_FACTOR`), only the footprint is kept: inside each unit, the pieces of
hexes Kontur puts at `FOOTPRINT_MIN` people per km2 or more, each weighted by its area. In a
residential area that is nearly the whole unit (median footprint share 1.0), so placement is
uniform; in the two desert units (Ahmadi 1,440 km2, Jahra 9,682 km2) and the farms it keeps dots
on the 6% to 60% that is settled. A unit with no footprint piece takes all its pieces, and one with
none its own polygon. `scatter.py` allocates each unit's dots from the 2021 census first.

Each hex is intersected with the units, as Malta (`sources/mt_geo.py`), because the units are
small (median 4.4 km2, six hexes) and a centroid join is too coarse at that grain.

## CHECKS

  * **the rank witness for the join**: Spearman of Kontur people per unit against the census,
    against `N_PERM` shuffles (weak here, because Kontur is; `sources/kw_geo.py` has the stronger
    witness, OSM's own population tags);
  * Kontur per unit against the census, printed, with the bar that fails;
  * every unit with people has a placement piece; the bbox is Kuwait.

Usage:
    python sources/kw_grid.py --fetch    one gzipped gpkg from Kontur (about 1 MB)
    python sources/kw_grid.py            rebuild from data/raw/kw/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "kw")
GEO = os.path.join(ROOT, "data", "geo", "kw")
UNITS_GPKG = os.path.join(GEO, "kw_units.gpkg")
OUT = os.path.join(GEO, "kw_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_KW_20231101.gpkg.gz")
GZ_NAME = "kontur_population_KW_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_KW_20231101.gpkg"

EXPECTED_UNITS = 143
EXPECTED_RATIO = 1.0
TOLERANCE = 0.3
FOOTPRINT_MIN = 50.0       # Kontur people per km2 of the whole hex
N_PERM = 20_000
SMALL_KM2 = 5.0
KEEP_BAR = (0.4, 2.5)
BAR_MIN_POP = 1_000
METRIC = "EPSG:32638"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 100_000:
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


def main():
    import geopandas as gpd
    from geo_checks import read_layer

    gpkg = os.path.join(RAW, GPKG_NAME)
    if "--fetch" in sys.argv or not os.path.exists(gpkg):
        fetch()
    if not os.path.exists(UNITS_GPKG):
        raise SystemExit(f"missing {UNITS_GPKG}; run sources/kw_geo.py first")

    hexes = read_layer(gpkg, "Kontur KW")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    hexes = hexes[hexes[popcol] > 0].to_crs(METRIC).reset_index(drop=True)
    hexes["hid"] = np.arange(len(hexes))
    hexes["dens"] = hexes[popcol] / (hexes.area / 1e6)
    k_all = float(hexes[popcol].sum())
    print(f"Kontur KW: {len(hexes):,} populated hexes, {k_all:,.0f} people")

    units = gpd.read_file(UNITS_GPKG)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS_GPKG} has {len(units)} units, expected {EXPECTED_UNITS}")
    u = units.to_crs(METRIC)
    ukm2 = u.set_index("unit").area / 1e6
    census = dict(zip(units["unit"], units["pop"].astype(int)))
    print(f"  units: median {ukm2.median():.2f} km2 ({ukm2.median() / 0.74:.1f} hexes, spec §8.2e); "
          f"{int((ukm2 < SMALL_KM2).sum())} under {SMALL_KM2:g} km2, {int((ukm2 < 0.74).sum())} "
          f"smaller than one hex")

    cent = gpd.GeoDataFrame({"hid": hexes["hid"]}, geometry=hexes.centroid, crs=METRIC)
    cj = gpd.sjoin(cent, u[["unit", "geometry"]], how="inner", predicate="within")
    per = cj.groupby("unit").size().reindex(u["unit"]).fillna(0)
    print(f"  centroid join (not used): median {per.median():.0f} hexes per unit, "
          f"{int((per == 0).sum())} units with none")

    pieces = gpd.overlay(hexes[["hid", popcol, "dens", "geometry"]], u[["unit", "geometry"]],
                         how="intersection", keep_geom_type=True)
    inside_km2 = pieces.area.groupby(pieces["hid"]).transform("sum") / 1e6
    pieces["pop"] = pieces[popcol] * (pieces.area / 1e6) / inside_km2
    pieces = pieces[pieces["pop"] > 0].reset_index(drop=True)
    kept = float(pieces["pop"].sum())
    print(f"  cut layer: {len(pieces):,} pieces; {kept:,.0f} of Kontur's {k_all:,.0f} people are in "
          f"hexes touching a unit; {k_all - kept:,.0f} ({100 * (k_all - kept) / k_all:.2f}%) in hexes "
          f"wholly at sea or in areas no census row claims")

    lost = hexes[~hexes["hid"].isin(pieces["hid"])]
    if len(lost):
        lost_c = gpd.GeoDataFrame({"pop": lost[popcol].to_numpy()}, geometry=lost.centroid.to_numpy(),
                                  crs=METRIC)
        near = gpd.sjoin_nearest(lost_c, u[["unit", "geometry"]], how="left", distance_col="d")
        near = near[~near.index.duplicated()]
        top = near.groupby("unit")["pop"].sum().sort_values(ascending=False).head(6)
        print("    nearest unit to those hexes, by people: "
              + ", ".join(f"{k} {v:,.0f}" for k, v in top.items()))

    total = sum(census.values())
    ratio = kept / total
    print(f"\n  Kontur {kept:,.0f} vs census 2021 {total:,}: ratio {ratio:.3f}")
    if abs(ratio - EXPECTED_RATIO) > TOLERANCE:
        raise SystemExit("Kontur and the census disagree beyond the band")

    kpu = pieces.groupby("unit")["pop"].sum().reindex(u["unit"]).fillna(0)
    names = list(u["unit"])
    a = np.array([kpu[x] for x in names])
    b = np.array([census[x] for x in names], dtype=float)
    rho = stats.spearmanr(a, b).statistic
    rng = np.random.default_rng(0)
    perm = np.array([stats.spearmanr(a, rng.permutation(b)).statistic for _ in range(N_PERM)])
    beaten = int((perm >= rho).sum())
    print(f"  join witness: Spearman(Kontur, census) over {len(names)} units = {rho:+.3f}; "
          f"{beaten} of {N_PERM:,} shuffles reach it (best {perm.max():+.3f})")
    if beaten:
        raise SystemExit("the rank witness fails; the area join may be permuted")

    rel = pd.Series({x: (kpu[x] / census[x]) / ratio for x in names if census[x] > 0})
    big = rel[[x for x in rel.index if census[x] >= BAR_MIN_POP]]
    q = big.quantile([0.1, 0.5, 0.9])
    print(f"  per unit of {BAR_MIN_POP:,}+ people ({len(big)}), Kontur share / census share: p10 "
          f"{q[0.1]:.2f}, median {q[0.5]:.2f}, p90 {q[0.9]:.2f}; bar {KEEP_BAR}")
    for label, s in (("lowest", big.nsmallest(8)), ("highest", big.nlargest(8))):
        print(f"     {label}: " + ", ".join(f"{k} {v:.2f} ({census[k]:,})" for k, v in s.items()))
    if q[0.1] >= KEEP_BAR[0] and q[0.9] <= KEEP_BAR[1]:
        raise SystemExit("Kontur now passes the bar; its counts could be used, so read the docstring")
    print("  Kontur's counts fail the bar, as recorded; only its footprint is used")
    moved = 0.5 * sum(abs(kpu[x] / kept - census[x] / total) for x in names)
    print(f"  share of Kontur's people in a different unit from the census: {moved:.1%}")
    npieces = pieces.groupby("unit").size().reindex(u["unit"]).fillna(0)
    print(f"  pieces per unit: median {npieces.median():.0f}, min {npieces.min():.0f}, "
          f"{int((npieces < 3).sum())} with fewer than three")

    # ---- the layer built: the footprint, weighted by area ----
    pieces["km2"] = pieces.area / 1e6
    foot = pieces[pieces["dens"] >= FOOTPRINT_MIN].copy()
    no_foot = sorted(set(pieces["unit"]) - set(foot["unit"]))
    if no_foot:
        foot = pd.concat([foot, pieces[pieces["unit"].isin(no_foot)]], ignore_index=True)
        print(f"  units with no piece at {FOOTPRINT_MIN:g}/km2, all their pieces kept: {no_foot}")
    fk = foot.groupby("unit")["km2"].transform("sum")
    foot["pop"] = foot["unit"].map(census) * foot["km2"] / fk
    share = (foot.groupby("unit")["km2"].sum() / ukm2.reindex(foot["unit"].unique())).sort_values()
    print(f"  footprint at {FOOTPRINT_MIN:g} people/km2: median {share.median():.2f} of a unit's area; "
          f"smallest " + ", ".join(f"{k} {v:.2f}" for k, v in share.head(6).items()))
    out = foot[["unit", "pop", "geometry"]]
    empty = sorted(x for x in names if census[x] > 0 and x not in set(foot["unit"]))
    if empty:
        fill = u[u["unit"].isin(empty)][["unit", "geometry"]].copy()
        fill["pop"] = fill["unit"].map(census).astype(float)
        out = pd.concat([out, fill[["unit", "pop", "geometry"]]], ignore_index=True)
        print(f"  {len(empty)} units with no Kontur piece get their own polygon: {empty}")
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=METRIC).to_crs("EPSG:4326")
    w, s, e, n = out.total_bounds
    if not (46.5 < w < e < 48.6 and 28.4 < s < n < 30.2):
        raise SystemExit(f"bbox {w:.3f} {s:.3f} {e:.3f} {n:.3f} is not Kuwait")
    missing = sorted(set(x for x in names if census[x] > 0) - set(out["unit"]))
    if missing:
        raise SystemExit(f"units with no placement: {missing}")
    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT + ".part.gpkg", driver="GPKG", layer="hexes")
    os.replace(OUT + ".part.gpkg", OUT)
    print(f"\nwrote {OUT} ({len(out):,} pieces)")


if __name__ == "__main__":
    main()
