"""Liechtenstein — the 11 communes, and a population grid to place dots inside them.

Writes:
    data/geo/li/li_communes.gpkg      11 polygons keyed as li.py keys them  (`units`)
    data/geo/li/li_grid_400m.gpkg     Kontur H3 r8 hexes with `unit` and `pop`  (`place`)

Usage:
    python sources/li_geo.py --fetch   # one download: Kontur, 17 KB
    python sources/li_geo.py           # build both layers

**THE BOUNDARIES WERE ALREADY ON DISK.** GISCO LAU 2021 carries Liechtenstein's **11 communes**
— §9e's file, downloaded for Poland and reused ever since — and the census names them
identically, so the join is eleven exact string matches with no folding, no stemming and no
aliases. Liechtenstein is not in the EU-27 correspondence *workbook*, but it is in the
boundary *shapefile*, which is the same distinction Switzerland turns on (§9ad).

**THE COMMUNES ARE NOT CONTIGUOUS AND THAT IS WHY THE GRID IS NOT OPTIONAL.** Liechtenstein
divides its high alpine pasture among the valley communes as **exclaves**: Vaduz is **six
separate polygons**, Schaan four, Balzers and Planken three, Eschen, Gamprin and Triesenberg
two. Nobody lives in the detached pieces — they are summer grazing above 1,500 m — so §8.2's
equal share would scatter a third of Vaduz's dots onto a mountainside with no houses on it.
This is the same argument as Switzerland's (§9ad) at a twentieth of the scale and it is
sharper here, because the empty part is not merely the thin end of a commune but a disjoint
piece of one.

**THE COUNTRY IS SMALL ENOUGH THAT THE CHECK HAS TO BE READ DIFFERENTLY.** §9p's test — every
unit's (other population / census count) inside a tight band — is run, but on 11 units of a few
thousand people a single hex assignment moves a ratio visibly, and the counts are 2015 against
a 2023 surface in a country that grew about 4%. The band is therefore wider than Kosovo's and
its job is to catch a scrambled join, which would be obvious, rather than to certify each unit.
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEO = os.path.join(ROOT, "data", "geo", "li")
NORM = os.path.join(ROOT, "data", "normalized", "li.csv")
LAU = os.path.join(ROOT, "data", "geo", "lau2021", "shp4326",
                   "LAU_RG_01M_2021_4326.shp")

KONTUR_DIR = os.path.join(ROOT, "data", "geo", "kontur")
KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.eu-central-1.amazonaws.com/"
              "kontur_datasets/kontur_population_LI_20231101.gpkg.gz")
KONTUR_GZ = os.path.join(KONTUR_DIR, "kontur_population_LI_20231101.gpkg.gz")
KONTUR = os.path.join(KONTUR_DIR, "kontur_population_LI_20231101.gpkg")

UNITS_OUT = os.path.join(GEO, "li_communes.gpkg")
GRID_OUT = os.path.join(GEO, "li_grid_400m.gpkg")

EXPECTED_UNITS = 11
LV95 = 2056           # CH1903+ / LV95 covers Liechtenstein too


def fetch():
    import gzip
    import shutil

    import requests

    os.makedirs(KONTUR_DIR, exist_ok=True)
    if os.path.exists(KONTUR) and os.path.getsize(KONTUR) > 20_000:
        print("already have", KONTUR)
        return
    if not os.path.exists(KONTUR_GZ) or os.path.getsize(KONTUR_GZ) < 5_000:
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

    for path in (LAU, NORM):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} -- run sources/li.py --fetch first "
                             "(and sources/pl_geo.py --fetch for the GISCO LAU file)")

    g = gpd.read_file(LAU, columns=["CNTR_CODE", "LAU_ID", "LAU_NAME"])
    g = g[g["CNTR_CODE"] == "LI"].copy().to_crs(4326)
    print(f"  GISCO LAU 2021, Liechtenstein: {len(g)} communes")
    if len(g) != EXPECTED_UNITS:
        raise SystemExit(f"expected {EXPECTED_UNITS} communes, got {len(g)}")

    df = pd.read_csv(NORM, low_memory=False)
    muni = (df[df["geo_level"] == "municipality"][["geo_id", "geo_name"]]
            .drop_duplicates("geo_id").copy())
    if len(muni) != EXPECTED_UNITS:
        raise SystemExit(f"{len(muni)} census communes, expected {EXPECTED_UNITS}")

    only_c = set(muni["geo_name"]) - set(g["LAU_NAME"])
    only_g = set(g["LAU_NAME"]) - set(muni["geo_name"])
    print("\n  the name join, both ways (§12 — a count match is not a join):")
    if only_c or only_g:
        print("    census names with no polygon:", sorted(only_c))
        print("    polygons with no census row:", sorted(only_g))
        raise SystemExit("the name join is not 1:1")
    print(f"    OK  all {len(muni)} communes join exactly, with no folding or aliases")

    out = g.merge(muni, left_on="LAU_NAME", right_on="geo_name", how="inner")
    out = out.rename(columns={"geo_id": "unit"})[["unit", "geo_name", "geometry"]]
    out["area_km2"] = out.to_crs(LV95).area.values / 1e6
    out["parts"] = out.geometry.apply(
        lambda x: len(x.geoms) if x.geom_type == "MultiPolygon" else 1)
    print(f"    area: total {out['area_km2'].sum():.0f} km², "
          f"median {out['area_km2'].median():.1f} km²")
    frag = out[out["parts"] > 1].sort_values("parts", ascending=False)
    print(f"    {len(frag)} of {len(out)} communes are made of more than one polygon — the "
          "alpine exclaves:")
    print("      " + ", ".join(f"{r.geo_name} {r.parts}" for r in frag.itertuples()))

    os.makedirs(GEO, exist_ok=True)
    out[["unit", "geo_name", "area_km2", "geometry"]].to_file(
        UNITS_OUT, layer="communes", driver="GPKG")
    print(f"\n  wrote {UNITS_OUT}  {len(out)} communes")
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
        raise SystemExit(f"missing {KONTUR} -- run sources/li_geo.py --fetch first")

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
    print(f"    {outside.sum()} hexes fall outside the {len(units)} communes "
          f"({hexes.loc[outside, 'population'].sum():,.0f} people) — the border strip "
          "Kontur rounds outwards")
    hexes = hexes[~outside].copy()

    # ---- the independent check (§9p), read for a scrambled join rather than per unit ---
    df = pd.read_csv(NORM, low_memory=False)
    tot = df[(df["geo_level"] == "municipality")
             & (df["source_category"] == "Religion - total")]
    tot = dict(zip(tot["geo_id"], tot["count"]))
    kon = hexes.groupby("unit")["population"].sum().to_dict()

    ratios = sorted((kon.get(u, 0.0) / tot[u], u) for u in units["unit"] if tot.get(u))
    med = ratios[len(ratios) // 2][0]
    print("\n  independent check — Kontur 2023 population / census 2015 count:")
    print(f"    median {med:.2f}x, min {ratios[0][0]:.2f} ({ratios[0][1]}), "
          f"max {ratios[-1][0]:.2f} ({ratios[-1][1]})")
    bad = [(r, u) for r, u in ratios if r < 0.5 or r > 2.0]
    if bad:
        print(f"    !! {len(bad)} of {len(ratios)} outside 0.5-2.0x:")
        for r, u in bad:
            print(f"       {u:16s} {r:.2f}")
        raise SystemExit("the population check failed -- the join is suspect")
    print(f"    every one of the {len(ratios)} inside 0.5-2.0x. On 11 units this is a test "
          "for a\n      scrambled join, not a certificate for each commune (see the "
          "docstring).")

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
    print(f"    {n_clipped} boundary hexes clipped, {len(hexes) - n_clipped} left whole")

    hexes["geometry"] = gpd.GeoSeries(out, crs=4326, index=hexes.index)
    empty = hexes.geometry.is_empty | hexes.geometry.isna()
    if empty.any():
        print(f"    dropped {empty.sum()} hexes whose clip came out empty")
        hexes = hexes[~empty].copy()

    hexes = hexes.rename(columns={"population": "pop"})[["unit", "pop", "geometry"]]

    missing = sorted(set(units["unit"]) - set(hexes["unit"]))
    if missing:
        print(f"    {len(missing)} communes hold no hex centre and carry their own polygon "
              f"as one cell: {missing}")
        fill = units[units["unit"].isin(missing)][["unit", "geometry"]].copy()
        fill["pop"] = 1.0
        hexes = gpd.GeoDataFrame(pd.concat([hexes, fill[["unit", "pop", "geometry"]]],
                                           ignore_index=True), crs=4326)
    print(f"    all {units['unit'].nunique()} communes are covered")

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
