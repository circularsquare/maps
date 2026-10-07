"""Puerto Rico placement layer: Kontur 400 m hexes cut to 2024 tracts -> data/geo/pr/pr_tracts.gpkg.

    python sources/pr_geo.py

UNITS. The PRCS 2020-2024 tables are on 2020 tracts, which cb_2024_us_tract_500k carries
(religiondots/data/geo/, read in place, STATEFP 72): one polygon per tract, `unit` = GEOID. The
cartographic-boundary file is already cut back to the shoreline.

WHY CUT HEXES. 921 tracts with people over 8,900 km2, but the median tract is small (San Juan,
Bayamon, Carolina), under the ~0.74 km2 r8 hex floor several times over, so a centroid join would
leave many tracts with no hex (religiondots playbooks/geography.md, "Below the floor, cut the
hexes to the units"). Each Kontur hex is intersected with the tracts and its people shared over
its LAND pieces by area (every bit of Puerto Rico's land is in some tract, so dividing by the sum
of the pieces is right). The mountain tracts of the Cordillera Central are large and mostly
empty; this is what puts their dots on the roads and barrios rather than evenly over the forest.

Kontur is read in place from religiondots/data/raw/pr/ (the November 2023 extract religiondots'
own Puerto Rico layer uses). The output is named `_tracts`, not `_hexes`, because its features are
hex pieces and religiondots' cap check would read their densities on the wrong areas; the cap is
checked here on the raw hexes instead (no hex within 0.5% of Kontur's 46,200/km2 cap; asserted).

CHECKS (asserted unless said)
  1. every tract with people in data/normalized/pr.csv has a polygon and at least one piece
  2. Kontur people in hexes touching no tract (sea, Mona) bounded at 0.5% of Kontur's total
  3. per tract, Kontur over the PRCS's people aged 5+, normalised by the national ratio (printed:
     p10, median, p90 and the extremes), and the log correlation against 500 shuffles (asserted
     above the best shuffle)
  4. raw Kontur below the density cap
"""
import math
import random
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RD = HERE.parent / "religiondots"
TRACTS = RD / "data" / "geo" / "cb_2024_us_tract_500k.zip"
KONTUR = RD / "data" / "raw" / "pr" / "kontur_population_PR_20231101.gpkg"
NORM = HERE / "data" / "normalized"
OUT = HERE / "data" / "geo" / "pr" / "pr_tracts.gpkg"
CRS_M = 32161                  # NAD83 / Puerto Rico & Virgin Is., metres
CAP = 46_200.0
OUTSIDE_MAX = 0.005


def main():
    g = gpd.read_file(f"zip://{TRACTS}", where="STATEFP = '72'")
    if g["GEOID"].duplicated().any():
        raise SystemExit("duplicate tract GEOIDs")
    g = g[["GEOID", "geometry"]].rename(columns={"GEOID": "unit"}).to_crs(CRS_M)
    d = pd.read_csv(NORM / "pr.csv", dtype={"geo_id": str})
    census = d.groupby("geo_id")["count"].sum()
    lost = sorted(set(census.index) - set(g["unit"]))
    if lost:
        raise SystemExit(f"check 1: {len(lost)} tracts with people and no polygon: {lost[:6]}")
    print(f"{len(g)} tracts in cb_2024 for Puerto Rico; all {len(census)} with people aged 5+ have "
          f"a polygon; {len(g) - len(census)} have none in the tables (water, forest reserves)")
    area = g.set_index("unit").area / 1e6
    print(f"  tract area km2: median {area.median():.2f}, p10 {area.quantile(.1):.2f}, "
          f"p90 {area.quantile(.9):.1f}; median {area.median() / 0.74:.1f} hexes' worth")

    k = gpd.read_file(KONTUR)
    if len(k) == 0:
        raise SystemExit("Kontur PR extract has ZERO features")
    k = k.rename(columns={"population": "kpop"})[["kpop", "geometry"]]
    k["hex"] = np.arange(len(k))
    km = k.to_crs(CRS_M)
    dens = (km["kpop"] / (km.area / 1e6)).max()
    if dens >= 0.995 * CAP:                                               # check 4
        raise SystemExit(f"check 4: a raw Kontur hex at {dens:,.0f}/km2, at the cap; register it")
    print(f"  Kontur PR: {len(k):,} hexes, {k['kpop'].sum():,.0f} people; densest hex "
          f"{dens:,.0f}/km2 (cap {CAP:,.0f}: none at it)")

    pieces = gpd.overlay(km, g, how="intersection", keep_geom_type=True)
    pieces["a"] = pieces.area
    pieces = pieces[pieces["a"] > 0].copy()
    land = pieces.groupby("hex")["a"].transform("sum")
    pieces["pop"] = pieces["kpop"] * pieces["a"] / land
    inside = set(pieces["hex"])
    out_pop = k.loc[~k["hex"].isin(inside), "kpop"].sum()
    print(f"  {len(pieces):,} pieces from {len(inside):,} hexes; "
          f"{(~k['hex'].isin(inside)).sum():,} hexes touch no tract ({out_pop:,.0f} people)")
    if out_pop > OUTSIDE_MAX * k["kpop"].sum():                           # check 2
        raise SystemExit(f"check 2: {out_pop:,.0f} Kontur people outside every tract")
    assert abs(pieces["pop"].sum() + out_pop - k["kpop"].sum()) < 1, "people lost in the cut"

    per_n = pieces.groupby("unit").size()
    no_piece = sorted(set(census.index) - set(per_n.index))
    if no_piece:                                                          # check 1
        raise SystemExit(f"check 1: {len(no_piece)} tracts with people and no piece: {no_piece[:6]}")
    per = pieces.groupby("unit")["pop"].sum()
    zero = [u for u in census.index if per.get(u, 0) <= 0]
    print(f"  pieces per tract with people: median {int(per_n[census.index].median())}, "
          f"min {int(per_n[census.index].min())}; {len(zero)} tracts where Kontur has nobody "
          f"(placed on equal shares of their pieces)")

    # check 3
    rows = [(u, float(census[u]), float(per.get(u, 0.0))) for u in census.index]
    ratio = sum(r[2] for r in rows) / sum(r[1] for r in rows)
    norm = sorted((kk / c / ratio, u) for u, c, kk in rows)
    n = len(norm)
    print(f"  check 3: Kontur / PRCS 5+ nationally {ratio:.3f}; per tract normalised p10 "
          f"{norm[n // 10][0]:.2f}, median {norm[n // 2][0]:.2f}, p90 {norm[9 * n // 10][0]:.2f}")
    print("    lowest: " + ", ".join(f"{u} {r:.2f}" for r, u in norm[:5]))
    print("    highest: " + ", ".join(f"{u} {r:.2f}" for r, u in norm[-5:]))
    print(f"    {sum(1 for r, _ in norm if not (1 / 3 <= r <= 3))} of {n} tracts outside a factor of 3")
    pos = [(c, kk) for _, c, kk in rows if kk > 0 and c > 0]
    lc = [math.log(c) for c, _ in pos]
    lk = [math.log(kk) for _, kk in pos]

    def pear(a, b):
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    r = pear(lc, lk)
    rng = random.Random(0)
    best = max(abs(pear(lc, rng.sample(lk, len(lk)))) for _ in range(500))
    print(f"    log correlation r = {r:.3f} against a best of {best:.3f} over 500 shuffles")
    if r <= best:
        raise SystemExit("check 3: the join is not carrying information")

    keep = pieces[pieces["unit"].isin(census.index)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({"unit": keep["unit"].to_numpy(), "pop": keep["pop"].to_numpy()},
                     geometry=keep.geometry.to_numpy(), crs=CRS_M).to_crs(4326).to_file(
        OUT, layer="pieces", driver="GPKG")
    print(f"wrote {OUT} ({len(keep):,} pieces over {keep['unit'].nunique()} tracts)")


if __name__ == "__main__":
    main()
