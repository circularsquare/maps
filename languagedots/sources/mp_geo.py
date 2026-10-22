"""Northern Mariana Islands placement layer: Kontur 400 m hexes cut to the 2020 tracts
-> data/geo/mp/mp_tracts.gpkg.

    python sources/mp_geo.py

UNITS. PCT24 is on 2020 tracts (sources/mp_census.py); cb_2020_us_tract_500k carries the CNMI's 23
(religiondots/data/geo/, read in place, STATEFP 69): one polygon per tract, already cut back to
the shoreline, `unit` = GEOID. 21 of them hold people in the table: Saipan 17, Tinian 2, Rota 1
and the Northern Islands 1 (7 people).

WHY CUT HEXES. Saipan's tracts are 1 to 24 km2 and several are smaller than one r8 hex's worth,
so a centroid join would leave some with no hex (religiondots playbooks/geography.md, "Below the
floor, cut the hexes to the units"). Each Kontur hex is cut to the tracts and its people shared
over its land pieces by area, as Guam's (sources/gu_geo.py).

CHECKS (asserted unless said)
  1. every tract with people in mp.csv has a polygon and a piece
  2. Kontur people in hexes touching no tract bounded at 1% of Kontur's total
  3. per tract, Kontur over the census's total population (P1), normalised by the national
     ratio (printed), and the log correlation against 500 shuffles (asserted above the best)
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
NORM = HERE / "data" / "normalized"
OUT = HERE / "data" / "geo" / "mp" / "mp_tracts.gpkg"
CRS_M = 32655                  # WGS 84 / UTM 55N, metres
CAP = 46_200.0
OUTSIDE_MAX = 0.01
N_TRACTS = 23


def main():
    allg = gpd.read_file(f"zip://{TRACTS}", where="STATEFP = '69'")[["GEOID", "geometry"]].to_crs(CRS_M)
    if len(allg) != N_TRACTS or allg["GEOID"].duplicated().any():
        raise SystemExit(f"expected {N_TRACTS} distinct CNMI tracts, got {len(allg)}")
    d = pd.read_csv(NORM / "mp.csv", dtype={"geo_id": str})
    census = d.groupby("geo_id")["count"].sum()
    p1 = pd.read_csv(NORM / "mp_p1.csv", dtype={"geo_id": str}).set_index("geo_id")["p1"]
    need = set(census.index)
    lost = sorted(need - set(allg["GEOID"]))
    if lost:                                                                      # check 1
        raise SystemExit(f"check 1: {len(lost)} tracts with people and no polygon: {lost}")
    extra = sorted(set(allg["GEOID"]) - need)
    print(f"{len(allg)} tracts in cb_2020 for the CNMI; all {len(need)} tracts with people have a polygon; "
          f"{len(extra)} others (nobody in the table: " + ", ".join(f"{t} P1 {p1.get(t, 0):,}" for t in extra) + ")")
    area = allg[allg["GEOID"].isin(need)].set_index("GEOID").area / 1e6
    print(f"  tract area km2 (the Northern Islands' includes sea): median {area.median():.2f}, "
          f"p10 {area.quantile(.1):.2f}, p90 {area.quantile(.9):.1f}")

    k = gpd.read_file(kontur_path("mp"))
    if len(k) == 0:
        raise SystemExit("Kontur MP extract has ZERO features")
    k = k.rename(columns={"population": "kpop"})[["kpop", "geometry"]]
    k["hex"] = np.arange(len(k))
    km = k.to_crs(CRS_M)
    dens = (km["kpop"] / (km.area / 1e6)).max()
    if dens >= 0.995 * CAP:                                                       # check 4
        raise SystemExit(f"check 4: a raw Kontur hex at {dens:,.0f}/km2, at the cap; register it")
    print(f"  Kontur MP: {len(k):,} hexes, {k['kpop'].sum():,.0f} people (census 2020: "
          f"{int(p1.sum()):,}); densest hex {dens:,.0f}/km2 (cap {CAP:,.0f}: none at it)")

    pieces = gpd.overlay(km, allg, how="intersection", keep_geom_type=True)
    pieces["a"] = pieces.area
    pieces = pieces[pieces["a"] > 0].copy()
    land = pieces.groupby("hex")["a"].transform("sum")
    pieces["pop"] = pieces["kpop"] * pieces["a"] / land
    inside = set(pieces["hex"])
    out_pop = k.loc[~k["hex"].isin(inside), "kpop"].sum()
    print(f"  {len(pieces):,} pieces from {len(inside):,} hexes; "
          f"{(~k['hex'].isin(inside)).sum():,} hexes touch no tract ({out_pop:,.0f} people)")
    if out_pop > OUTSIDE_MAX * k["kpop"].sum():                                   # check 2
        raise SystemExit(f"check 2: {out_pop:,.0f} Kontur people outside every tract")
    pieces = pieces.rename(columns={"GEOID": "tract"})
    kper = pieces.groupby("tract")["pop"].sum()

    # check 3: Kontur against total population per tract (Kontur counts everyone)
    rows = [(t, float(p1[t]), float(kper.get(t, 0.0))) for t in sorted(need) if p1[t] > 0]
    ratio = sum(r[2] for r in rows) / sum(r[1] for r in rows)
    norm = sorted((kk / c / ratio, t) for t, c, kk in rows)
    n = len(norm)
    print(f"  check 3: Kontur / census total population {ratio:.3f}; per tract normalised p10 "
          f"{norm[n // 10][0]:.2f}, median {norm[n // 2][0]:.2f}, p90 {norm[9 * n // 10][0]:.2f}")
    print("    lowest: " + ", ".join(f"{t} {r:.2f}" for r, t in norm[:5]))
    print("    highest: " + ", ".join(f"{t} {r:.2f}" for r, t in norm[-5:]))
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

    keep = pieces[pieces["tract"].isin(need)].copy()
    no_piece = sorted(need - set(keep["tract"]))
    if no_piece:                                                                  # check 1
        raise SystemExit(f"check 1: tracts with no piece: {no_piece}")
    zero = [t for t in need if kper.get(t, 0) <= 0]
    for t in zero:                     # Kontur has nobody: equal by area
        m = keep["tract"] == t
        keep.loc[m, "pop"] = keep.loc[m, "a"] / keep.loc[m, "a"].sum()
    keep["unit"] = keep["tract"]
    per_n = keep.groupby("unit").size()
    print(f"  pieces per unit: median {int(per_n.median())}, min {int(per_n.min())}; "
          f"{len(zero)} tracts where Kontur has nobody (equal by area): {zero}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({"unit": keep["unit"].to_numpy(), "tract": keep["tract"].to_numpy(),
                      "pop": keep["pop"].to_numpy()},
                     geometry=keep.geometry.to_numpy(), crs=CRS_M).to_crs(4326).to_file(
        OUT, layer="pieces", driver="GPKG")
    print(f"wrote {OUT} ({len(keep):,} pieces over {keep['unit'].nunique()} units)")


if __name__ == "__main__":
    main()
