"""Jamaica — the 14 parishes, and a grid to place dots inside them.

Writes:
    data/geo/jm/jm_parishes.gpkg     14 polygons keyed as jm.py keys them  (`units`)
    data/geo/jm/jm_grid_400m.gpkg    Kontur H3 r8 hexes with `unit` and `pop` (`place`)

Usage:
    python sources/jm_geo.py --fetch   # one download: Kontur 0.9 MB (the gdb is jm.py's)
    python sources/jm_geo.py           # build both layers

**THE JOIN IS FREE AND THAT IS THE WHOLE POINT OF THE USCB SERIES.** `GEO_MATCH` keys the
counts layer to the boundary layer by construction — `JAM_GEO1_01` … `JAM_GEO1_14` on both
sides — so there is no name matching, no code bridge and no ambiguity. §11j measured this
across the series and Jamaica behaves like the rest: **14 table keys, 14 geo keys, 14
matched, 0 unmatched**. Compare Bosnia the same week, where geoBoundaries needed four
documented repairs before a name join could be attempted at all (`sources/ba_geo.md` §2).

**AND THE GEOG VINTAGE IS THE RIGHT ONE, WHICH HAS TO BE CHECKED RATHER THAN ASSUMED.** The
geodatabase ships **two** boundary generations — `JM_GEOG1_*` cut for the 2011 census and
`JM_GEOG2_ADM2_2012` for the 2012 survey — and §11j's finding on the Central African Republic
was that *taking the newer layers because they are newer silently breaks the join*. The
religion table is `..._GEOG1_2011census_...`, so `GEOG1` is the correct pair and `GEOG2` is
not a Jamaica boundary set at all in the sense that matters here. Asserted in `build_units`.

**WHY A POPULATION GRID FOR A COUNTRY THIS SMALL.** 14 parishes over 10,991 km² averages
785 km², and Jamaica's people are not evenly spread across it — the Kingston/Saint Andrew
conurbation is roughly a quarter of the country on about 2% of its area, and the Cockpit
Country and the Blue Mountains are close to empty. Uniform scatter would put dots on
limestone karst and rainforest, and at 1:1,000 Saint Andrew's 569 dots would spread across
a parish that is mostly mountain. This is §8.2's argument at the small end.

**The ADM2 "Special Areas" are NOT used, and the reason is worth recording.** The gdb carries
`JM_GEOG1_ADM2_2011` — STATIN's special areas, rebuilt by USCB from 248 original shapefiles —
and they are a real, census-vintage, finer geography. They are not used because **no religion
count exists at that tier** (`sources/jm.py`), so they could only serve as a placement
weight — and for that job a population surface beats an administrative subdivision, because
the subdivision is uniform inside itself and Kontur is not. Named here so the next reader
does not rediscover the layer and assume it was missed.

**The independent check is §9p's and Jamaica passes it comfortably**, which is not
guaranteed at 14 units: with so few, a scrambled join has few places to hide and the ratio
spread is the tell either way.
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEO = os.path.join(ROOT, "data", "geo", "jm")
RAW = os.path.join(ROOT, "data", "raw", "jm")
NORM = os.path.join(ROOT, "data", "normalized", "jm.csv")

GDB = os.path.join(RAW, "Jamaica.gdb")
LAYER_ADM1 = "JM_GEOG1_ADM1_2011_uscb_202302"

KONTUR_DIR = os.path.join(ROOT, "data", "geo", "kontur")
KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.eu-central-1.amazonaws.com/"
              "kontur_datasets/kontur_population_JM_20231101.gpkg.gz")
KONTUR_GZ = os.path.join(KONTUR_DIR, "kontur_population_JM_20231101.gpkg.gz")
KONTUR = os.path.join(KONTUR_DIR, "kontur_population_JM_20231101.gpkg")

UNITS_OUT = os.path.join(GEO, "jm_parishes.gpkg")
GRID_OUT = os.path.join(GEO, "jm_grid_400m.gpkg")

EXPECTED_UNITS = 14
UTM = 32617                      # UTM 17N, metres — Jamaica sits inside it


def fetch():
    import gzip
    import shutil

    import requests

    if not os.path.isdir(GDB):
        raise SystemExit(f"missing {GDB} -- run `python sources/jm.py --fetch` first; the "
                         "boundaries ship inside the same geodatabase as the counts")
    ua = {"User-Agent": "religiondots/1.0"}
    os.makedirs(KONTUR_DIR, exist_ok=True)
    if os.path.exists(KONTUR) and os.path.getsize(KONTUR) > 300_000:
        print("already have", KONTUR)
        return
    if not os.path.exists(KONTUR_GZ) or os.path.getsize(KONTUR_GZ) < 100_000:
        print("GET", KONTUR_URL)
        r = requests.get(KONTUR_URL, timeout=600, headers=ua)
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
    import pyogrio

    if not os.path.isdir(GDB):
        raise SystemExit(f"missing {GDB} -- run `python sources/jm.py --fetch` first")
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/jm.py first")

    layers = [n for n, _ in pyogrio.list_layers(GDB)]
    if LAYER_ADM1 not in layers:
        raise SystemExit(f"{LAYER_ADM1} not in the geodatabase; layers are {layers}")
    # §11j's CAR trap: the file offers two boundary generations and only one matches the
    # religion table's vintage. Fail loudly if the naming stops making that checkable.
    if "GEOG1" not in LAYER_ADM1 or "2011" not in LAYER_ADM1:
        raise SystemExit(f"{LAYER_ADM1} is not the GEOG1/2011 pair the religion table uses")

    g = gpd.read_file(GDB, layer=LAYER_ADM1)
    if g.crs is None:
        g = g.set_crs(4326)
    g = g.to_crs(4326)
    print(f"  {LAYER_ADM1}: {len(g)} polygons")
    if len(g) != EXPECTED_UNITS:
        raise SystemExit(f"expected {EXPECTED_UNITS} polygons, got {len(g)}")

    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    par = (df[df["geo_level"] == "parish"][["geo_id", "geo_name"]]
           .drop_duplicates("geo_id").copy())
    if len(par) != EXPECTED_UNITS:
        raise SystemExit(f"{len(par)} census parishes, expected {EXPECTED_UNITS}")

    for label, s in (("geodatabase", g["GEO_MATCH"]), ("census", par["geo_id"])):
        if s.duplicated().any():
            raise SystemExit(f"{label} GEO_MATCH is not unique")

    only_c = set(par["geo_id"]) - set(g["GEO_MATCH"])
    only_g = set(g["GEO_MATCH"]) - set(par["geo_id"])
    print("\n  the GEO_MATCH join, both ways (§12 — a count match is not a join):")
    print(f"    matched                      {len(set(par['geo_id']) & set(g['GEO_MATCH'])):>3}")
    if only_c or only_g:
        print("    census keys with no polygon:", sorted(only_c))
        print("    polygons with no census row:", sorted(only_g))
        raise SystemExit("the key join is not 1:1 -- which should be impossible in this series")
    print(f"    OK  all {len(par)} parishes join on GEO_MATCH, by construction")

    out = g.merge(par, left_on="GEO_MATCH", right_on="geo_id", how="inner")
    out = out.rename(columns={"geo_id": "unit", "AREA_NAME": "geo_name_gdb"})
    out = out[["unit", "geo_name", "geo_name_gdb", "geometry"]]
    out["area_km2"] = out.to_crs(UTM).area.values / 1e6
    print(f"    area: median {out['area_km2'].median():.0f} km², "
          f"total {out['area_km2'].sum():,.0f} km² (Jamaica is 10,991)")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(UNITS_OUT, layer="parishes", driver="GPKG")
    print(f"\n  wrote {UNITS_OUT}  {len(out)} parishes")
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
        raise SystemExit(f"missing {KONTUR} -- run sources/jm_geo.py --fetch first")

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
    print(f"    {outside.sum():,} hexes fall outside the {len(units)} parishes "
          f"({hexes.loc[outside, 'population'].sum():,.0f} people) — the coastal strip "
          "Kontur rounds outwards")
    hexes = hexes[~outside].copy()

    # ---- independent check (§9p), and the band is measured rather than inherited (§9u) ---
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    tot = (df[df["geo_level"] == "parish"].groupby("geo_id")["count"].sum().to_dict())
    kon = hexes.groupby("unit")["population"].sum().to_dict()
    name = dict(zip(units["unit"], units["geo_name"]))

    ratios = sorted((kon.get(u, 0.0) / tot[u], name[u]) for u in units["unit"] if tot.get(u))
    med = ratios[len(ratios) // 2][0]
    print("\n  independent check — Kontur 2023 population / census 2011 count:")
    print(f"    median {med:.2f}x, min {ratios[0][0]:.2f} ({ratios[0][1]}), "
          f"max {ratios[-1][0]:.2f} ({ratios[-1][1]})")
    lo, hi = med / 2.0, med * 2.0
    bad = [(r, u) for r, u in ratios if not (lo <= r <= hi)]
    print(f"    band {lo:.2f}-{hi:.2f}x (median/2 to median*2): "
          f"{len(ratios) - len(bad)}/{len(ratios)} inside")
    for r, u in bad:
        print(f"       outside: {u:22s} {r:.2f}x")
    # Tighter than Bosnia's median/3 because the vintages are closer (2011 vs 2023, and
    # Jamaica's population barely moved) and because 14 units give a scrambled join nowhere
    # to hide — with this few, more than one outlier is a real signal rather than noise.
    if len(bad) > 1:
        raise SystemExit(f"{len(bad)} of {len(ratios)} parishes outside the band -- with "
                         "only 14 units that is not noise; the join is suspect")

    print(f"\n  clipping {len(hexes):,} hexes to their parish…")
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
    hexes["geometry"] = out
    hexes = hexes[~hexes.geometry.is_empty & hexes.geometry.notna()].copy()
    print(f"    {n_clipped:,} edge hexes clipped to their parish boundary")

    grid = hexes[["unit", "population", "geometry"]].rename(columns={"population": "pop"})
    grid = gpd.GeoDataFrame(grid, geometry="geometry", crs=4326)
    os.makedirs(GEO, exist_ok=True)
    grid.to_file(GRID_OUT, layer="grid", driver="GPKG")
    print(f"  wrote {GRID_OUT}  {len(grid):,} hexes, "
          f"{grid['pop'].sum():,.0f} modelled people")

    empty = set(units["unit"]) - set(grid.loc[grid["pop"] > 0, "unit"])
    if empty:
        print(f"    !! {len(empty)} parishes have no populated hex: "
              f"{sorted(name[u] for u in empty)}")


def main():
    if "--fetch" in sys.argv:
        fetch()
    units = build_units()
    build_grid(units)


if __name__ == "__main__":
    main()
