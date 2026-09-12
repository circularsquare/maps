"""Switzerland — the 2,197 communes, and a population grid to place dots inside them.

Writes:
    data/geo/ch/ch_communes.gpkg     2,197 polygons keyed as ch.py keys them  (`units`)
    data/geo/ch/ch_grid_400m.gpkg    Kontur H3 r8 hexes with `unit` and `pop`  (`place`)

Usage:
    python sources/ch_geo.py --fetch   # one download: Kontur ~3 MB
    python sources/ch_geo.py           # build both layers

**THE BOUNDARIES COST NOTHING AND THE JOIN IS A NUMBER RATHER THAN A NAME.** GISCO LAU 2021 —
on disk since Poland (§9e) — carries Switzerland's **2,242 features** with `LAU_ID` as `CH`
plus the BFS commune number, which is the same number the census publishes. So there is no
name matching here at all, and none of §8.1's usual failure modes: `sources/ch.py` has already
moved the 2000 counts onto these codes through BFS's own correspondence API.

**45 OF THE 2,242 FEATURES ARE NOT COMMUNES**, and finding that out by the register rather
than by a code range is the point. BFS numbers the **lake surfaces** in the 9xxx block —
Lac Léman, the Bodensee, the Zürichsee are all apportioned to no municipality — and the
Ticino and Graubünden **comunanze**, common land held jointly by several communes, take
numbers of their own. 44 of the 45 are empty; `5399` holds 802 people. They are dropped
because BFS's commune register does not list them, so a renumbering fails loudly here instead
of quietly deleting a real commune. What is left is **2,197**. One of those — a Graubünden comunanza the 2000 census did count —
is absent from `ch_rescale.py`'s output because BFS's register does not list it either, so
2,196 communes actually draw.

**WHY A POPULATION GRID AND NOT §8.2's EQUAL SHARE, AND SWITZERLAND IS THE STRONGEST CASE ON
THIS MAP FOR IT.** The median commune is small, but the distribution is not the point: Swiss
communes contain the Alps. Bagnes is 282 km² of which the inhabited part is a valley floor;
Zermatt is 243 km² and its people are in one village at the end of it; Davos, Glarus Süd,
Scuol and Val Müstair are the same shape. Scattering a commune's dots uniformly paints
glacier and rock face, and it does it precisely in the cantons — Valais, Graubünden, Uri —
whose religious composition is most distinctive. Portugal (§9v) left this undone at a 16.5 km²
median and said so; Switzerland cannot.

The two approximations are Kosovo's and are bounded by the hex size: a hex belongs to the
commune containing its centre and is then clipped to it. Kontur's populations are a model and
are used ONLY as relative weights inside a unit — every commune's dot count comes from
`ch_rescale.py`.
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEO = os.path.join(ROOT, "data", "geo", "ch")
RAW = os.path.join(ROOT, "data", "raw", "ch")
NORM = os.path.join(ROOT, "data", "normalized", "ch_commune_rescaled.csv")
LAU = os.path.join(ROOT, "data", "geo", "lau2021", "shp4326",
                   "LAU_RG_01M_2021_4326.shp")
LEVELS = os.path.join(RAW, "levels.csv")

KONTUR_DIR = os.path.join(ROOT, "data", "geo", "kontur")
KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.eu-central-1.amazonaws.com/"
              "kontur_datasets/kontur_population_CH_20231101.gpkg.gz")
KONTUR_GZ = os.path.join(KONTUR_DIR, "kontur_population_CH_20231101.gpkg.gz")
KONTUR = os.path.join(KONTUR_DIR, "kontur_population_CH_20231101.gpkg")

UNITS_OUT = os.path.join(GEO, "ch_communes.gpkg")
GRID_OUT = os.path.join(GEO, "ch_grid_400m.gpkg")

EXPECTED_UNITS = 2_197
LV95 = 2056           # CH1903+ / LV95, the Swiss national projection


def fetch():
    import gzip
    import shutil

    import requests

    os.makedirs(KONTUR_DIR, exist_ok=True)
    if os.path.exists(KONTUR) and os.path.getsize(KONTUR) > 1_000_000:
        print("already have", KONTUR)
        return
    if not os.path.exists(KONTUR_GZ) or os.path.getsize(KONTUR_GZ) < 100_000:
        print("GET", KONTUR_URL)
        r = requests.get(KONTUR_URL, timeout=600, headers={"User-Agent": "religiondots/1.0"})
        r.raise_for_status()
        if r.content[:2] != b"\x1f\x8b":
            raise SystemExit(f"not gzip: first bytes {r.content[:40]!r}")
        with open(KONTUR_GZ, "wb") as fh:
            fh.write(r.content)
        print(f"  {os.path.getsize(KONTUR_GZ):,} bytes")
    with gzip.open(KONTUR_GZ, "rb") as src, open(KONTUR, "wb") as dst:
        shutil.copyfileobj(src, dst)
    print(f"  decompressed to {KONTUR}  {os.path.getsize(KONTUR):,} bytes")


def build_units():
    import geopandas as gpd
    import pandas as pd

    for path in (LAU, LEVELS, NORM):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} -- run sources/ch.py --fetch and "
                             "python ch_rescale.py first")

    g = gpd.read_file(LAU, columns=["CNTR_CODE", "LAU_ID", "LAU_NAME", "POP_2021"])
    g = g[g["CNTR_CODE"] == "CH"].copy().to_crs(4326)
    g["unit"] = g["LAU_ID"].str.replace("CH", "", regex=False).str.zfill(4)
    print(f"  GISCO LAU 2021, Switzerland: {len(g)} features")

    lv = pd.read_csv(LEVELS)
    register = {f"{int(b):04d}" for b in lv["BfsCode"]}
    orphan = g[~g["unit"].isin(register)]
    print(f"  {len(orphan)} are not communes in BFS's register — lake surfaces (9xxx) and "
          f"the\n    Ticino/Graubünden comunanze — holding "
          f"{orphan['POP_2021'].sum():,.0f} people; dropped")
    g = g[g["unit"].isin(register)].copy()

    if len(g) != EXPECTED_UNITS:
        raise SystemExit(f"expected {EXPECTED_UNITS} communes, got {len(g)}")

    counts = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    want = set(counts["geo_id"])
    missing = want - set(g["unit"])
    if missing:
        raise SystemExit(f"{len(missing)} counted communes have no polygon: "
                         f"{sorted(missing)[:10]}")
    empty = sorted(set(g["unit"]) - want)
    if empty:
        print(f"  {len(empty)} polygons carry no counts and will draw nothing: {empty[:6]}")
    print(f"  OK  all {len(want):,} counted communes join to exactly one polygon, on the "
          "BFS number")

    out = g.rename(columns={"LAU_NAME": "geo_name"})[["unit", "geo_name", "geometry"]].copy()
    out["area_km2"] = g.to_crs(LV95).area.values / 1e6
    print(f"    area: median {out['area_km2'].median():.1f} km², "
          f"max {out['area_km2'].max():.0f} km² "
          f"({out.loc[out['area_km2'].idxmax(), 'geo_name']}), "
          f"total {out['area_km2'].sum():,.0f} km²")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(UNITS_OUT, layer="communes", driver="GPKG")
    print(f"\n  wrote {UNITS_OUT}  {len(out):,} communes")
    return out


def fiona_layers(path):
    import pyogrio
    return list(pyogrio.list_layers(path)[:, 0])


def build_grid(units):
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import shapely

    if not os.path.exists(KONTUR):
        raise SystemExit(f"missing {KONTUR} -- run sources/ch_geo.py --fetch first")

    layers = fiona_layers(KONTUR)
    layer = "population" if "population" in layers else layers[0]
    print(f"\n  reading Kontur r8 hexes from layer {layer!r}…")
    hexes = gpd.read_file(KONTUR, layer=layer).to_crs(4326)
    print(f"    {len(hexes):,} hexes, {hexes['population'].sum():,.0f} people")

    centres = gpd.GeoDataFrame(geometry=hexes.geometry.representative_point(), crs=4326)
    hit = gpd.sjoin(centres, units[["unit", "geometry"]], how="left", predicate="within")
    hit = hit[~hit.index.duplicated(keep="first")]
    hexes["unit"] = hit["unit"].to_numpy()

    outside = hexes["unit"].isna()
    print(f"    {outside.sum():,} hexes fall outside the {len(units):,} communes "
          f"({hexes.loc[outside, 'population'].sum():,.0f} people) — the lakes this build "
          "drops,\n      plus the border strip Kontur rounds outwards")
    hexes = hexes[~outside].copy()

    # ---- the independent check (§9p) --------------------------------------------------
    # POP_2021 is in the boundary file itself, so Switzerland gets this check for free and
    # against a real published population rather than against its own counts.
    lau = gpd.read_file(LAU, columns=["CNTR_CODE", "LAU_ID", "POP_2021"])
    lau = lau[lau["CNTR_CODE"] == "CH"].copy()
    lau["unit"] = lau["LAU_ID"].str.replace("CH", "", regex=False).str.zfill(4)
    pop = dict(zip(lau["unit"], lau["POP_2021"]))
    kon = hexes.groupby("unit")["population"].sum().to_dict()
    name = dict(zip(units["unit"], units["geo_name"]))

    ratios = sorted((kon.get(u, 0.0) / pop[u], u) for u in units["unit"]
                    if pop.get(u, 0) > 200)
    med = ratios[len(ratios) // 2][0]
    print("\n  independent check — Kontur 2023 population / GISCO POP_2021, over the "
          f"{len(ratios):,} communes above 200 people:")
    print(f"    median {med:.2f}x, min {ratios[0][0]:.2f} ({name[ratios[0][1]]}), "
          f"max {ratios[-1][0]:.2f} ({name[ratios[-1][1]]})")
    bad = [(r, u) for r, u in ratios if r < 0.4 or r > 2.5]
    print(f"    {len(bad)} outside 0.4-2.5x "
          f"({100.0 * len(bad) / len(ratios):.1f}%)")
    if len(bad) > 0.02 * len(ratios):
        for r, u in bad[:12]:
            print(f"       {name[u]:26s} {r:.2f}")
        raise SystemExit("too many communes off the band -- the join is suspect")
    print("    OK  a scrambled join would not keep 98% of 2,196 units inside that band")

    print(f"\n  clipping {len(hexes):,} hexes to their commune…")
    poly = units.set_index("unit")["geometry"]
    geom = hexes.geometry.to_numpy()
    who = hexes["unit"].to_numpy()
    out = np.empty(len(hexes), dtype=object)
    n_clipped = 0
    for unit, parent in poly.items():
        idx = np.flatnonzero(who == unit)
        if idx.size == 0:
            continue
        shapely.prepare(parent)
        inside = shapely.contains_properly(parent, geom[idx])
        out[idx[inside]] = geom[idx[inside]]
        edge = idx[~inside]
        if edge.size:
            out[edge] = shapely.intersection(parent, geom[edge])
            n_clipped += edge.size
    print(f"    {n_clipped:,} boundary hexes clipped, {len(hexes) - n_clipped:,} left whole")

    hexes["geometry"] = gpd.GeoSeries(out, crs=4326, index=hexes.index)
    empty = hexes.geometry.is_empty | hexes.geometry.isna()
    if empty.any():
        print(f"    dropped {empty.sum():,} hexes whose clip came out empty")
        hexes = hexes[~empty].copy()

    hexes = hexes.rename(columns={"population": "pop"})[["unit", "pop", "geometry"]]

    missing = sorted(set(units["unit"]) - set(hexes["unit"]))
    if missing:
        print(f"    {len(missing)} communes hold no hex centre and carry their own polygon "
              f"as one cell: {[name[u] for u in missing][:8]}")
        fill = units[units["unit"].isin(missing)][["unit", "geometry"]].copy()
        fill["pop"] = 1.0
        hexes = gpd.GeoDataFrame(pd.concat([hexes, fill[["unit", "pop", "geometry"]]],
                                           ignore_index=True), crs=4326)
    print(f"    all {units['unit'].nunique():,} communes are covered")

    hexes.to_file(GRID_OUT, layer="grid400m", driver="GPKG")
    print(f"  wrote {GRID_OUT}  {len(hexes):,} hexes")
    return hexes


def main():
    if "--fetch" in sys.argv:
        fetch()
    units = build_units()
    build_grid(units)


if __name__ == "__main__":
    main()
