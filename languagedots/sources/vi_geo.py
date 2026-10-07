"""US Virgin Islands placement layer: Kontur 400 m hexes cut to 2020 block groups
-> data/geo/vi/vi_bg.gpkg.

    python sources/vi_geo.py

UNITS. PBG5 is on 2020 block groups. TIGER/Line 2020 block groups for state 78
(https://www2.census.gov/geo/tiger/TIGER2020/BG/tl_2020_78_bg.zip, in data/raw/vi/) carry the
GEOIDs but run out to sea; they are clipped to the land of the cartographic-boundary 2020 tracts
(religiondots/data/geo/cb_2020_us_tract_500k.zip, STATEFP 78, read in place), which are cut back
to the shoreline. `unit` = the 12-digit GEOID, as data/normalized/vi.csv.

WHY CUT HEXES. The median block group is a few km2 and the Charlotte Amalie and Christiansted
ones far less, near the ~0.74 km2 hex floor, so a centroid join would leave block groups empty
(religiondots playbooks/geography.md, "Below the floor, cut the hexes to the units"). Each Kontur
hex is intersected with the block groups and its people shared over its land pieces by area. As
pr_geo.py, the output is named `_bg`, not `_hexes` (pieces, not hexes), and the density cap is
checked here on the raw hexes.

CHECKS (asserted unless said)
  1. the block groups join both ways: every block group with people in vi.csv has a polygon and
     at least one piece; every TIGER block group without one in vi.csv is a water or empty one
  2. Kontur people in hexes touching no block group bounded at 1% of Kontur's total
  3. per block group, Kontur over the census's people aged 5+ in households, normalised by the
     national ratio (printed), and the log correlation against 500 shuffles (asserted above)
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
sys.path.insert(0, str(HERE / "sources"))
from _grid import kontur_path  # noqa: E402

RD = HERE.parent / "religiondots"
TRACTS = RD / "data" / "geo" / "cb_2020_us_tract_500k.zip"
BG = HERE / "data" / "raw" / "vi" / "tl_2020_78_bg.zip"
NORM = HERE / "data" / "normalized"
OUT = HERE / "data" / "geo" / "vi" / "vi_bg.gpkg"
CRS_M = 32161                  # NAD83 / Puerto Rico & Virgin Is., metres
CAP = 46_200.0
OUTSIDE_MAX = 0.01


def main():
    b = gpd.read_file(f"zip://{BG}")
    if len(b) != 92 or b["GEOID"].duplicated().any():
        raise SystemExit(f"expected 92 unique block groups, got {len(b)}")
    b = b[["GEOID", "ALAND", "geometry"]].rename(columns={"GEOID": "unit"}).to_crs(CRS_M)
    t = gpd.read_file(f"zip://{TRACTS}", where="STATEFP = '78'").to_crs(CRS_M)
    land = t.geometry.union_all() if hasattr(t.geometry, "union_all") else t.unary_union
    b["geometry"] = b.geometry.intersection(land)
    b = b[~b.geometry.is_empty]
    d = pd.read_csv(NORM / "vi.csv", dtype={"geo_id": str})
    census = d.groupby("geo_id")["count"].sum()
    lost = sorted(set(census.index) - set(b["unit"]))
    if lost:
        raise SystemExit(f"check 1: {len(lost)} block groups with people and no land polygon: {lost}")
    extra = b[~b["unit"].isin(census.index)]
    print(f"{len(b)} block groups with land; all {len(census)} with people have a polygon; "
          f"without people: {sorted(extra['unit'])} (ALAND {extra['ALAND'].astype(float).sum() / 1e6:.1f} km2)")
    area = b.set_index("unit").area / 1e6
    print(f"  block group area km2: median {area.median():.2f}, p10 {area.quantile(.1):.2f}, "
          f"p90 {area.quantile(.9):.1f}")

    k = gpd.read_file(kontur_path("vi"))
    if len(k) == 0:
        raise SystemExit("Kontur VI extract has ZERO features")
    k = k.rename(columns={"population": "kpop"})[["kpop", "geometry"]]
    k["hex"] = np.arange(len(k))
    km = k.to_crs(CRS_M)
    dens = (km["kpop"] / (km.area / 1e6)).max()
    if dens >= 0.995 * CAP:                                               # check 4
        raise SystemExit(f"check 4: a raw Kontur hex at {dens:,.0f}/km2, at the cap; register it")
    print(f"  Kontur VI: {len(k):,} hexes, {k['kpop'].sum():,.0f} people; densest hex "
          f"{dens:,.0f}/km2 (cap {CAP:,.0f}: none at it)")

    pieces = gpd.overlay(km, b[["unit", "geometry"]], how="intersection", keep_geom_type=True)
    pieces["a"] = pieces.area
    pieces = pieces[pieces["a"] > 0].copy()
    tot = pieces.groupby("hex")["a"].transform("sum")
    pieces["pop"] = pieces["kpop"] * pieces["a"] / tot
    inside = set(pieces["hex"])
    out_pop = k.loc[~k["hex"].isin(inside), "kpop"].sum()
    print(f"  {len(pieces):,} pieces from {len(inside):,} hexes; "
          f"{(~k['hex'].isin(inside)).sum():,} hexes touch no block group ({out_pop:,.0f} people)")
    if out_pop > OUTSIDE_MAX * k["kpop"].sum():                           # check 2
        raise SystemExit(f"check 2: {out_pop:,.0f} Kontur people outside every block group")

    per_n = pieces.groupby("unit").size()
    no_piece = sorted(set(census.index) - set(per_n.index))
    if no_piece:                                                          # check 1
        raise SystemExit(f"check 1: {len(no_piece)} block groups with people and no piece: {no_piece}")
    per = pieces.groupby("unit")["pop"].sum()
    zero = [u for u in census.index if per.get(u, 0) <= 0]
    print(f"  pieces per block group with people: median {int(per_n[census.index].median())}, "
          f"min {int(per_n[census.index].min())}; {len(zero)} where Kontur has nobody")

    rows = [(u, float(census[u]), float(per.get(u, 0.0))) for u in census.index]   # check 3
    ratio = sum(r[2] for r in rows) / sum(r[1] for r in rows)
    norm = sorted((kk / c / ratio, u) for u, c, kk in rows)
    n = len(norm)
    print(f"  check 3: Kontur / census 5+ in households {ratio:.3f}; per block group normalised p10 "
          f"{norm[n // 10][0]:.2f}, median {norm[n // 2][0]:.2f}, p90 {norm[9 * n // 10][0]:.2f}")
    print("    lowest: " + ", ".join(f"{u} {r:.2f}" for r, u in norm[:5]))
    print("    highest: " + ", ".join(f"{u} {r:.2f}" for r, u in norm[-5:]))
    print(f"    {sum(1 for r, _ in norm if not (1 / 3 <= r <= 3))} of {n} outside a factor of 3")
    pos = [(c, kk) for _, c, kk in rows if kk > 0 and c > 0]
    lc = [math.log(c) for c, _ in pos]
    lk = [math.log(kk) for _, kk in pos]

    def pear(a, b_):
        ma, mb = sum(a) / len(a), sum(b_) / len(b_)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b_))
        return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b_))
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
    print(f"wrote {OUT} ({len(keep):,} pieces over {keep['unit'].nunique()} block groups)")


if __name__ == "__main__":
    main()
