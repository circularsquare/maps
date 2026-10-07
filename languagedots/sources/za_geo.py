"""South Africa placement layer: Kontur hexes cut to Census 2011's 84,907 small areas.

    python sources/za_geo.py

-> data/geo/za/za_hexes.gpkg (unit = SAL code, pop = Kontur people in the piece)

WHY CUT AND NOT A CENTROID JOIN. A small area holds about 600 people, and in the townships and
cities it is far smaller than a Kontur r8 hex (0.74 km²): a centroid join would leave thousands
of small areas with no hex and pile their neighbours' hexes on others. Following the playbook's
Malta rule ("below the floor, cut the hexes to the units"), every hex is intersected with the
small areas it touches and its people shared over the pieces by area.

SHARED OVER THE WHOLE HEX, NOT OVER ITS PIECES. The playbook divides by the pieces' sum where
every piece of land is in some unit. Here a border hex's remainder is Lesotho, Eswatini or
another neighbour, which the small areas rightly leave out, and dividing by the whole hex is
the playbook's own rule for that case. It also keeps every piece's density at or below its
hex's, so religiondots' Kontur cap check still sees raw Kontur. On the coast it gives the sea a
share, which only tilts a coastal small area's own dots slightly inland; each small area's dot
count comes from the census, never from these weights.

THE SMALL AREAS DO NOT TILE THE COUNTRY, AND THAT IS RIGHT FOR A 2011 MAP. They cover 1,149,551
km² of 1,220,813: Stats SA dissolved them from the 2011 enumeration areas (SAL_APRI.XML,
"Dissolve EA_SA_2011_080413_MunicChange ... SAL_CODE1") and left the 9,097 EAs its frame calls
Vacant (109,071 km², some inside small areas) out. So 4.45M of Kontur's 60.5M fall outside every
small area: in Intsika Yethu (EC135) 38% of Kontur's people sit on 2,107 km² of Vacant EAs
against a 1,916 km² gap (measured with the EA attribute table, 2026-10-04). Kontur is 2023 and
models built-up land; the census put nobody there in 2011, so no dot goes there.

SMALL AREAS WITH NO KONTUR PEOPLE get their own polygon at pop 0, which the scatter places on
equal shares (one polygon, so uniformly over the small area).

CHECKS: the 84,907 units, every one placed; Kontur people falling outside every small area
(border and coast); Kontur against the census per small area, per main place and per 2011
municipality, each with a shuffled-join control on the log correlation.
"""
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "za"
KONTUR = HERE.parent / "religiondots" / "data" / "raw" / "za" / "kontur_population_ZA_20231101.gpkg"
NORM = HERE / "data" / "normalized" / "za.csv"
OUT = HERE / "data" / "geo" / "za" / "za_hexes.gpkg"
EXPECTED_SAL = 84_907
SLIVER_M2 = 500.0       # pieces smaller than this are dropped, unless a small area's only piece


def pear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])


def witness(name, census, kontur, n_shuffle=200):
    """Log correlation of Kontur against the census per unit, and a shuffled-join control."""
    df = pd.DataFrame({"c": census, "k": kontur}).dropna()
    df = df[(df["c"] > 0) & (df["k"] > 0)]
    lc, lk = np.log(df["c"].to_numpy()), np.log(df["k"].to_numpy())
    r = pear(lc, lk)
    rng = random.Random(0)
    lk_l = list(lk)
    best = max(abs(pear(lc, rng.sample(lk_l, len(lk_l)))) for _ in range(n_shuffle))
    ratio = (df["k"] / df["c"]) / (df["k"].sum() / df["c"].sum())
    print(f"  {name}: {len(df):,} units; log r = {r:.3f} against a best of {best:.3f} over "
          f"{n_shuffle} shuffles; Kontur/census normalised p10 {ratio.quantile(0.1):.2f} "
          f"median {ratio.median():.2f} p90 {ratio.quantile(0.9):.2f}")
    if r <= best:
        raise SystemExit(f"!! {name}: THE JOIN IS NOT CARRYING INFORMATION")
    return r


def main():
    sal = gpd.read_file(RAW / "SAL_APRI.SHP")
    if len(sal) != EXPECTED_SAL:
        raise SystemExit(f"SAL_APRI: {len(sal)} features, expected {EXPECTED_SAL}")
    sal["unit"] = sal["SAL_CODE"].astype("int64").astype(str)
    if sal["unit"].duplicated().any():
        raise SystemExit("duplicate SAL codes")
    bad = ~sal.geometry.is_valid
    if bad.any():
        print(f"  {int(bad.sum())} invalid small-area polygons repaired with make_valid")
        sal.loc[bad, "geometry"] = shapely.make_valid(sal.geometry[bad].values)
    sal = sal.set_crs(4148, allow_override=True)[["unit", "MP_CODE", "MN_MDB_C", "geometry"]]
    sal_m = sal.to_crs(3857)

    hexes = gpd.read_file(KONTUR)
    if len(hexes) == 0:
        raise SystemExit("Kontur ZA extract has ZERO features")
    hexes = hexes.rename(columns={"population": "pop"})[["pop", "geometry"]]
    print(f"Kontur ZA: {len(hexes):,} hexes, {hexes['pop'].sum():,.0f} people")

    print("cutting hexes to small areas…")
    hi, si = sal_m.sindex.query(hexes.geometry.values, predicate="intersects")
    print(f"  {len(hi):,} hex x small-area pairs")
    hg = hexes.geometry.values[hi]
    sg = sal_m.geometry.values[si]
    # a hex wholly inside one small area needs no cut
    inside = shapely.contains_properly(sg, hg)
    pieces = np.empty(len(hi), dtype=object)
    pieces[inside] = hg[inside]
    rest = np.nonzero(~inside)[0]
    step = 50_000
    for s in range(0, len(rest), step):
        j = rest[s:s + step]
        pieces[j] = shapely.intersection(hg[j], sg[j])
        print(f"    cut {min(s + step, len(rest)):,} of {len(rest):,}", flush=True)
    pieces = shapely.make_valid(pieces)
    # keep polygonal parts only (a shared edge intersects as a line)
    for k in np.nonzero(shapely.get_type_id(pieces) == 7)[0]:
        parts = shapely.get_parts(pieces[k])
        keep = parts[np.isin(shapely.get_type_id(parts), (3, 6))]
        pieces[k] = shapely.union_all(keep) if len(keep) else shapely.Polygon()
    area = np.nan_to_num(shapely.area(pieces))
    hex_area = shapely.area(hg)
    pop = hexes["pop"].to_numpy(float)[hi] * area / hex_area
    df = pd.DataFrame({"unit": sal["unit"].to_numpy()[si], "hex": hi, "area": area, "pop": pop})
    df["geometry"] = pieces

    covered = df.groupby("hex")["area"].sum()
    frac = (covered / pd.Series(shapely.area(hexes.geometry.values))[covered.index]).clip(upper=1)
    out_people = hexes["pop"].sum() - float((hexes["pop"].iloc[covered.index].to_numpy() * frac.to_numpy()).sum())
    untouched = hexes["pop"].sum() - hexes["pop"].iloc[covered.index].sum()
    print(f"  Kontur people outside every small area: {out_people:,.0f} "
          f"({out_people / hexes['pop'].sum():.2%}), of them {untouched:,.0f} in hexes touching none "
          "(mostly the 2011 frame's Vacant EAs, which no small area holds; see the docstring)")
    lost = pd.Series(hexes["pop"].iloc[covered.index].to_numpy() * (1 - frac.to_numpy()),
                     index=covered.index).sort_values(ascending=False)
    print(f"  covered share of touched hexes: p1 {frac.quantile(0.01):.3f}, p10 {frac.quantile(0.1):.3f}, "
          f"median {frac.median():.3f}; the most people lost:")
    cen = hexes.geometry.iloc[lost.index[:8]].centroid.to_crs(4326)
    for h, pt in zip(lost.index[:8], cen):
        print(f"      hex {h} at {pt.x:.4f},{pt.y:.4f}: {hexes['pop'].iat[h]:,.0f} people, "
              f"{frac[h]:.2f} covered")

    poly = df["area"] > 0
    n_slivers = int((poly & (df["area"] < SLIVER_M2)).sum())
    big = df[df["area"] >= SLIVER_M2]
    only = df[poly & ~df["unit"].isin(big["unit"])]
    only = only.loc[only.groupby("unit")["area"].idxmax()]
    df = pd.concat([big, only], ignore_index=True)
    print(f"  {len(df):,} pieces kept; {n_slivers:,} slivers under {SLIVER_M2:.0f} m² dropped "
          f"({len(only):,} kept as a small area's only piece)")

    missing = sorted(set(sal["unit"]) - set(df["unit"]))
    census = pd.read_csv(NORM, dtype={"geo_id": str})
    census = census[~census["source_category"].isin(["Not applicable", "Unspecified"])]
    cpop = census.groupby("geo_id")["count"].sum()
    print(f"  {len(missing):,} small areas with no Kontur hex at all ({cpop.reindex(missing).sum():,.0f} "
          "census people with a language): given their own polygon at pop 0")
    own = sal_m[sal_m["unit"].isin(missing)][["unit", "geometry"]].copy()
    own["pop"] = 0.0
    own["hex"] = -1
    layer = gpd.GeoDataFrame(pd.concat([df[["unit", "hex", "pop", "geometry"]], own], ignore_index=True),
                             geometry="geometry", crs=3857)

    per = layer.groupby("unit")["pop"].sum()
    zero = per[per <= 0]
    print(f"  {len(zero):,} small areas with zero Kontur weight (equal shares), holding "
          f"{cpop.reindex(zero.index).sum():,.0f} census people ({cpop.reindex(zero.index).sum() / cpop.sum():.2%})")
    cells = layer.groupby("unit").size()
    print(f"  pieces per small area: median {cells.median():.0f}, p90 {cells.quantile(0.9):.0f}, max {cells.max()}")

    if set(layer["unit"]) != set(sal["unit"]):
        raise SystemExit("the layer does not hold every small area")
    if set(cpop.index) - set(layer["unit"]):
        raise SystemExit("census small areas missing from the layer")

    print("Kontur against the census:")
    witness("small areas", cpop, per.reindex(cpop.index))
    mp = sal.set_index("unit")["MP_CODE"]
    witness("main places", cpop.groupby(mp.reindex(cpop.index)).sum(),
            per.groupby(mp.reindex(per.index)).sum())
    mn = sal.set_index("unit")["MN_MDB_C"]
    witness("2011 municipalities", cpop.groupby(mn.reindex(cpop.index)).sum(),
            per.groupby(mn.reindex(per.index)).sum())

    layer = layer.to_crs(4326)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name(OUT.stem + f".{os.getpid()}.tmp.gpkg")
    layer[["unit", "pop", "geometry"]].to_file(tmp, layer="hexes", driver="GPKG")
    os.replace(tmp, OUT)
    print(f"wrote {OUT} ({len(layer):,} pieces, {layer['unit'].nunique():,} small areas, "
          f"{layer['pop'].sum():,.0f} Kontur people)")


if __name__ == "__main__":
    main()
