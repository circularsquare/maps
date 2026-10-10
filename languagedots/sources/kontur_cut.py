"""Density weighting INSIDE each placement polygon, for countries whose census units are already
small enough to be their own placement layer (us, uk, ca, au, ie, il, vc; Anita, 2026-10-08).

    python sources/kontur_cut.py us

-> data/geo/<cc>/<cc>_konturcut.gpkg: the country's current placement polygons (countries/<cc>.py
`place`), each cut by the Kontur r8 hexes it touches, `unit` as before and `pop` a weight.

WHY. With place_weight=None a dot lands uniformly over its polygon's land. A rural US tract or
Canadian dissemination area can be hundreds of km² with everyone in one town, so its dots fell
anywhere in the empty part (the "1,000 Vietnamese on a Minnesota island" complaint).

THE WEIGHTS STAY WHAT THEY WERE BETWEEN POLYGONS; KONTUR ONLY MOVES DOTS INSIDE ONE. Every
original polygon keeps a total weight of 1, as place_weight=None gave it (equal shares, which
for Australia's SA1s, built to about 400 people, religiondots measured to be a population
weighting already), and its pieces share that 1 by Kontur people. So a polygon's share of its
unit's dots is unchanged; only where in the polygon they go. Kontur is 2023 modelled built-up
population, finer than these units but no truer about their totals.

A POLYGON WITH NO KONTUR PEOPLE is kept whole at weight 1: uniform, as before.

NOT A `_hexes` LAYER, ON PURPOSE. kontur_cap (religiondots) checks `*_hexes.gpkg` layers for
blocks at Kontur's 46,200/km² cap and stops on one its registry does not name. Here a false
block can only pull a polygon's own dots a few hundred metres within that polygon, never between
polygons, and `pop` is a share, not people, so the check does not apply and is skipped by name.

Pieces under SLIVER_M2 are dropped unless they are a polygon's only piece; the shares are taken
after the drop, so every polygon's pieces still sum to 1.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "sources"))
from kontur_fetch import kontur_path  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SLIVER_M2 = 500.0
CHUNK = 20_000      # polygons per batch
KONTUR_CC = {"uk": "GB"}


def out_path(cc):
    return HERE / "data" / "geo" / cc / f"{cc}_konturcut.gpkg"


def main(cc, src=None):
    from countries import COUNTRIES
    cfg = COUNTRIES[cc]
    src = Path(src or cfg["place"])
    if src.name.endswith("_konturcut.gpkg"):
        raise SystemExit(f"{cc}: `place` is already the cut layer; pass the original with --src")
    t0 = time.time()
    poly = gpd.read_file(src)
    poly["unit"] = cfg["place_unit"](poly).astype(str)
    poly = poly[~(poly.geometry.isna() | poly.geometry.is_empty)].reset_index(drop=True)
    bad = ~poly.geometry.is_valid
    if bad.any():
        poly.loc[bad, "geometry"] = shapely.make_valid(poly.geometry[bad].values)
    poly_m = poly.to_crs(3857)
    print(f"{cc}: {len(poly):,} polygons, {poly['unit'].nunique():,} units ({src.name})")

    kp = kontur_path(KONTUR_CC.get(cc, cc.upper()))
    hexes = gpd.read_file(kp).rename(columns={"population": "pop"})[["pop", "geometry"]]
    if hexes.crs is None or hexes.crs.to_epsg() != 3857:
        hexes = hexes.to_crs(3857)
    hexes = hexes[hexes["pop"] > 0].reset_index(drop=True)
    print(f"  Kontur {kp.name}: {len(hexes):,} hexes, {hexes['pop'].sum():,.0f} people")
    hg_all, hpop = hexes.geometry.values, hexes["pop"].to_numpy(float)
    hex_area = shapely.area(hg_all)
    tree = shapely.STRtree(hg_all)

    frames = []
    for s in range(0, len(poly_m), CHUNK):
        pg = poly_m.geometry.values[s:s + CHUNK]
        si, hi = tree.query(pg, predicate="intersects")
        g_p, g_h = pg[si], hg_all[hi]
        inside = shapely.contains_properly(g_p, g_h)
        pieces = np.empty(len(si), dtype=object)
        pieces[inside] = g_h[inside]
        rest = np.nonzero(~inside)[0]
        pieces[rest] = shapely.intersection(g_h[rest], g_p[rest])
        pieces = shapely.make_valid(pieces)
        for k in np.nonzero(shapely.get_type_id(pieces) == 7)[0]:
            parts = shapely.get_parts(pieces[k])
            keep = parts[np.isin(shapely.get_type_id(parts), (3, 6))]
            pieces[k] = shapely.union_all(keep) if len(keep) else shapely.Polygon()
        area = np.nan_to_num(shapely.area(pieces))
        frames.append(pd.DataFrame({"poly": si + s, "area": area,
                                    "k": hpop[hi] * area / hex_area[hi], "geometry": pieces}))
        print(f"    {min(s + CHUNK, len(poly_m)):,} of {len(poly_m):,} polygons, "
              f"{time.time() - t0:.0f}s", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["area"] > 0) & (df["k"] > 0)]

    big = df[df["area"] >= SLIVER_M2]
    only = df[~df["poly"].isin(big["poly"])]
    only = only.loc[only.groupby("poly")["area"].idxmax()] if len(only) else only
    df = pd.concat([big, only], ignore_index=True)
    df["pop"] = df["k"] / df.groupby("poly")["k"].transform("sum")

    whole = np.setdiff1d(np.arange(len(poly_m)), df["poly"].unique())
    own = pd.DataFrame({"poly": whole, "pop": 1.0, "geometry": poly_m.geometry.values[whole]})
    layer = pd.concat([df[["poly", "pop", "geometry"]], own], ignore_index=True)
    layer["unit"] = poly["unit"].to_numpy()[layer["poly"].to_numpy()]
    layer = gpd.GeoDataFrame(layer, geometry="geometry", crs=3857)

    per_poly = layer.groupby("poly")["pop"].sum()
    if len(per_poly) != len(poly_m) or not np.allclose(per_poly.to_numpy(), 1.0):
        raise SystemExit("a polygon's pieces do not sum to 1")
    if set(layer["unit"]) != set(poly["unit"]):
        raise SystemExit("the layer lost a unit")
    inside_people = float(df["k"].sum())
    n_pieces = layer.groupby("poly").size()
    print(f"  {len(layer):,} pieces; pieces per polygon median {n_pieces.median():.0f}, "
          f"p90 {n_pieces.quantile(0.9):.0f}, max {n_pieces.max():,}")
    print(f"  {len(whole):,} polygons with no Kontur people kept whole (uniform, as before)")
    print(f"  Kontur people inside the polygons: {inside_people:,.0f} of {hpop.sum():,.0f} "
          f"({inside_people / hpop.sum():.1%}; the rest is outside every unit or in dropped slivers)")

    out = out_path(cc)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + f".{os.getpid()}.tmp.gpkg")
    layer.to_crs(4326)[["unit", "pop", "geometry"]].to_file(tmp, layer="pieces", driver="GPKG")
    os.replace(tmp, out)
    print(f"wrote {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cc")
    ap.add_argument("--src", help="the original placement layer, once countries/<cc>.py points at the cut")
    a = ap.parse_args()
    main(a.cc, a.src)
