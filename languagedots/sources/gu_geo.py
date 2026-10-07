"""Guam placement layer: Kontur 400 m hexes cut to the 2020 tracts -> data/geo/gu/gu_tracts.gpkg.

    python sources/gu_geo.py

UNITS. PCT25 is on 2020 tracts; cb_2020_us_tract_500k carries Guam's 56 (religiondots/data/geo/,
read in place, STATEFP 66): one polygon per tract, `unit` = GEOID, already cut back to the
shoreline. The six tracts whose PCT25 cells the Bureau suppresses (sources/gu_census.py) are one
unit in the counts, `66010SUPPR`; their pieces carry that unit.

WHY CUT HEXES. Guam's median tract is a few km2, a handful of r8 hexes, and the Tamuning and
Hagåtña tracts are smaller than that, so a centroid join would leave some with no hex (religiondots
playbooks/geography.md, "Below the floor, cut the hexes to the units"). Each Kontur hex is cut to
the tracts and its people shared over its land pieces by area.

THE SUPPRESSED TRACTS. Within `66010SUPPR` each tract's pieces are scaled so the tract holds its
own published total population (P1, data/normalized/gu_suppressed_p1.csv), shared over its pieces
by Kontur (equally by area where Kontur has nobody). So Umatac's 647 people and Tamuning's
9,000-odd get their own share of the pooled dots, not Kontur's view of it. Elsewhere the weight is
Kontur's people inside the tract; the tract's count is the census's either way.

Kontur counts everyone, the military bases included, which the language table leaves out (it
covers people in households outside military housing). Inside a tract that mixes base housing
and civilian homes the dots lean towards the base a little. Andersen (9501) and the naval base
tracts hold under 100 people in the table each.

CHECKS (asserted unless said)
  1. every tract with people in gu.csv, and every suppressed tract, has a polygon and a piece
  2. Kontur people in hexes touching no tract bounded at 1% of Kontur's total
  3. per tract, Kontur over the census's total population (P1), normalised by the national
     ratio (printed), and the log correlation against 500 shuffles (asserted above the best)
  4. raw Kontur below the density cap
  5. the suppressed unit's weights sum to the six tracts' P1, each tract to its own
"""
import math
import random
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
from _grid import kontur_path  # noqa: E402
from gu_census import GEO_COLS, RAW, SUPPRESSED_TRACTS, SUPPRESSED_UNIT, read_table  # noqa: E402

RD = HERE.parent / "religiondots"
TRACTS = RD / "data" / "geo" / "cb_2020_us_tract_500k.zip"
NORM = HERE / "data" / "normalized"
OUT = HERE / "data" / "geo" / "gu" / "gu_tracts.gpkg"
CRS_M = 32655                  # WGS 84 / UTM 55N, metres
CAP = 46_200.0
OUTSIDE_MAX = 0.01
N_TRACTS = 56


def tract_p1():
    z = zipfile.ZipFile(RAW / "gu2020.dhc.zip")
    with z.open("gugeo2020.dhc") as fh:
        geo = pd.read_csv(fh, sep="|", header=None, dtype=str, encoding="latin-1")
    geo = geo[list(GEO_COLS)].rename(columns=GEO_COLS)
    geo = geo[geo["GEOCOMP"] == "00"]
    p1 = read_table(z, geo, "P1")
    p1 = p1[p1["SUMLEV"] == "140"]
    return pd.Series(p1[1].to_numpy(), index=p1["GEOID"].str[9:])


def main():
    g = gpd.read_file(f"zip://{TRACTS}", where="STATEFP = '66'")
    if len(g) != N_TRACTS or g["GEOID"].duplicated().any():
        raise SystemExit(f"expected {N_TRACTS} distinct Guam tracts, got {len(g)}")
    g = g[["GEOID", "geometry"]].rename(columns={"GEOID": "tract"}).to_crs(CRS_M)
    d = pd.read_csv(NORM / "gu.csv", dtype={"geo_id": str})
    census = d.groupby("geo_id")["count"].sum()
    published = [u for u in census.index if u != SUPPRESSED_UNIT]
    p1 = tract_p1()
    need = set(published) | set(SUPPRESSED_TRACTS)
    lost = sorted(need - set(g["tract"]))
    if lost:
        raise SystemExit(f"check 1: {len(lost)} tracts with people and no polygon: {lost}")
    extra = sorted(set(g["tract"]) - need)
    print(f"{len(g)} tracts in cb_2020 for Guam; all {len(published)} published tracts with people and the "
          f"{len(SUPPRESSED_TRACTS)} suppressed ones have a polygon; {len(extra)} others (nobody in the "
          f"table: " + ", ".join(f"{t} P1 {p1.get(t, 0):,}" for t in extra) + ")")
    g = g[g["tract"].isin(need)].copy()
    g["unit"] = np.where(g["tract"].isin(SUPPRESSED_TRACTS), SUPPRESSED_UNIT, g["tract"])
    area = g.set_index("tract").area / 1e6
    print(f"  tract area km2: median {area.median():.2f}, p10 {area.quantile(.1):.2f}, "
          f"p90 {area.quantile(.9):.1f}; median {area.median() / 0.74:.1f} hexes' worth")

    k = gpd.read_file(kontur_path("gu"))
    if len(k) == 0:
        raise SystemExit("Kontur GU extract has ZERO features")
    k = k.rename(columns={"population": "kpop"})[["kpop", "geometry"]]
    k["hex"] = np.arange(len(k))
    km = k.to_crs(CRS_M)
    dens = (km["kpop"] / (km.area / 1e6)).max()
    if dens >= 0.995 * CAP:                                               # check 4
        raise SystemExit(f"check 4: a raw Kontur hex at {dens:,.0f}/km2, at the cap; register it")
    print(f"  Kontur GU: {len(k):,} hexes, {k['kpop'].sum():,.0f} people (census 2020: "
          f"{p1.sum():,}); densest hex {dens:,.0f}/km2 (cap {CAP:,.0f}: none at it)")

    allg = gpd.read_file(f"zip://{TRACTS}", where="STATEFP = '66'")[["GEOID", "geometry"]].to_crs(CRS_M)
    pieces = gpd.overlay(km, allg, how="intersection", keep_geom_type=True)
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
    if no_piece:                                                          # check 1
        raise SystemExit(f"check 1: tracts with no piece: {no_piece}")
    zero = [t for t in need if kper.get(t, 0) <= 0]
    for t in zero:                     # Kontur has nobody: equal by area
        m = keep["tract"] == t
        keep.loc[m, "pop"] = keep.loc[m, "a"] / keep.loc[m, "a"].sum()
    # the suppressed tracts: each scaled to its own P1
    for t in SUPPRESSED_TRACTS:
        m = keep["tract"] == t
        keep.loc[m, "pop"] = keep.loc[m, "pop"] * p1[t] / keep.loc[m, "pop"].sum()
    keep["unit"] = np.where(keep["tract"].isin(SUPPRESSED_TRACTS), SUPPRESSED_UNIT, keep["tract"])
    s = keep[keep["unit"] == SUPPRESSED_UNIT].groupby("tract")["pop"].sum()     # check 5
    exp = p1[SUPPRESSED_TRACTS]
    if (s.reindex(exp.index) - exp).abs().max() > 1e-6:
        raise SystemExit(f"check 5: suppressed tract weights {s.to_dict()} != P1 {exp.to_dict()}")
    print(f"  check 5 ok: {SUPPRESSED_UNIT}'s pieces hold each suppressed tract's P1 "
          f"({int(exp.sum()):,} in all); Kontur had " + ", ".join(
              f"{t[-6:]} {kper.get(t, 0) / p1[t]:.2f}x" for t in SUPPRESSED_TRACTS) + " of P1")
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
