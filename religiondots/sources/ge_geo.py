"""Georgia — the 11 enumerated regions, and a population grid to place dots inside them.

Writes:
    data/geo/ge/ge_regions.gpkg      11 polygons keyed as ge.py keys them  (`units`)
    data/geo/ge/ge_grid_400m.gpkg    Kontur H3 r8 hexes with `unit` and `pop` (`place`)

Usage:
    python sources/ge_geo.py --fetch   # geoBoundaries ADM1 + ADM2, and the Kontur extract
    python sources/ge_geo.py           # build both layers

**GEORGIA IS NOT IN GISCO LAU** — that file covers the EU27 and the candidates, and Georgia
is neither — so this is geoBoundaries again, at **ADM1**, which geoBoundaries calls
*Autonomous Republic, Region, City*. Twelve polygons for eleven counted regions, and the
spare one is **Abkhazia**, which the 2014 census did not enumerate. It is dropped here rather
than carried with no data: a polygon with no counts draws nothing either way, and keeping it
would put an empty unit in the layer that every downstream check then has to explain.

**THE NAME JOIN IS TWO ALIASES AND A DASH.** Geostat writes `C. Tbilisi` and `Autonomous
Republic of Adjara` where geoBoundaries writes `Tbilisi` and `Adjara`; and geoBoundaries
writes `Samtskhe–Javakheti` with an EN DASH where Geostat uses a hyphen. The dash costs
nothing because the fold is letters-only, which is why folds should be letters-only.

**SOUTH OSSETIA IS SUBTRACTED, AND IT IS SMALLER THAN IT LOOKS.** The census covers territory
under Georgian government control, so the Tskhinvali region is outside it — but unlike
Abkhazia it has no ADM1 polygon of its own, and its municipalities sit inside Shida Kartli
(**Java**) and Mtskheta-Mtianeti (**Akhalgori**). Those two ADM2 polygons are removed from
the placement layer, so no dot lands on ground the census did not count. The cost is small
because Kontur's Georgian extract barely covers South Ossetia in the first place — Java holds
2,114 modelled people and Akhalgori 3,979, against 3.74 million nationally — but a
*correction that turns out to be small is still worth making*, and the alternative is a
map that quietly claims Tskhinvali was counted.

**THE CITY/RING PAIR, WHICH §9q SAID WOULD RECUR.** Kontur/census comes out at 0.85 for
Tbilisi and **1.70 for Mtskheta-Mtianeti, the region wrapped around it** — geoBoundaries'
Tbilisi polygon is 249 km² against the city's ~500 km², so outer Tbilisi is inside its ring
region here. That is Lithuania's Vilnius check collecting in a fourth post-Soviet country,
and it is a *placement* fact rather than a count fact: every region's dot count still comes
from the census. Reported and not asserted; the band is asserted on the other ten.

WHY A POPULATION GRID AND NOT §8.2's EQUAL SHARE. Eleven regions over 60,000 enumerated km²
is 5,500 km² each, and Georgia is the Greater and Lesser Caucasus with people in the valleys.
Uniform scatter would put dots on ice.
"""

import os
import re
import sys
import unicodedata

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEO = os.path.join(ROOT, "data", "geo", "ge")
RAW = os.path.join(ROOT, "data", "raw", "ge")
NORM = os.path.join(ROOT, "data", "normalized", "ge.csv")

GB = "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/GEO/"
ADM1 = os.path.join(RAW, "geoBoundaries-GEO-ADM1.geojson")
ADM2 = os.path.join(RAW, "geoBoundaries-GEO-ADM2.geojson")

KONTUR_DIR = os.path.join(ROOT, "data", "geo", "kontur")
KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.eu-central-1.amazonaws.com/"
              "kontur_datasets/kontur_population_GE_20231101.gpkg.gz")
KONTUR_GZ = os.path.join(KONTUR_DIR, "kontur_population_GE_20231101.gpkg.gz")
KONTUR = os.path.join(KONTUR_DIR, "kontur_population_GE_20231101.gpkg")

UNITS_OUT = os.path.join(GEO, "ge_regions.gpkg")
GRID_OUT = os.path.join(GEO, "ge_grid_400m.gpkg")

EXPECTED_GB = 12
EXPECTED_UNITS = 11
NOT_ENUMERATED_ADM1 = "abkhazia"
# ADM2 units of the Tskhinvali region, which have no ADM1 of their own.
NOT_ENUMERATED_ADM2 = ("Java", "Akhalgori")

ALIAS = {"ctbilisi": "tbilisi", "autonomousrepublicofadjara": "adjara"}
# The ring half of the §9q city/ring pair, reported rather than asserted. Only the RING is
# here: `C. Tbilisi` itself comes out at 0.85x, inside the band, so it is asserted with
# everything else — the city loses population to its ring and stays plausible, while the
# ring gains a whole city's suburbs and does not.
CITY_RING = ("Mtskheta-Mtianeti",)


def fetch():
    import gzip
    import shutil

    import requests

    ua = {"User-Agent": "religiondots/1.0"}
    os.makedirs(RAW, exist_ok=True)
    for path, name in ((ADM1, "ADM1"), (ADM2, "ADM2")):
        if os.path.exists(path) and os.path.getsize(path) > 100_000:
            print("already have", path)
            continue
        url = f"{GB}{name}/geoBoundaries-GEO-{name}.geojson"
        print("GET", url)
        r = requests.get(url, timeout=600, headers=ua)
        r.raise_for_status()
        if b'"FeatureCollection"' not in r.content[:400]:
            raise SystemExit(f"not GeoJSON: first bytes {r.content[:120]!r}")
        with open(path, "wb") as fh:
            fh.write(r.content)
        print(f"  {os.path.getsize(path):,} bytes")

    os.makedirs(KONTUR_DIR, exist_ok=True)
    if os.path.exists(KONTUR) and os.path.getsize(KONTUR) > 1_000_000:
        print("already have", KONTUR)
        return
    if not os.path.exists(KONTUR_GZ) or os.path.getsize(KONTUR_GZ) < 500_000:
        print("GET", KONTUR_URL)
        r = requests.get(KONTUR_URL, timeout=900, headers=ua)
        r.raise_for_status()
        if r.content[:2] != b"\x1f\x8b":
            raise SystemExit(f"not gzip: first bytes {r.content[:40]!r}")
        with open(KONTUR_GZ, "wb") as fh:
            fh.write(r.content)
        print(f"  {os.path.getsize(KONTUR_GZ):,} bytes")
    with gzip.open(KONTUR_GZ, "rb") as src, open(KONTUR, "wb") as dst:
        shutil.copyfileobj(src, dst)
    print(f"  decompressed to {KONTUR}  {os.path.getsize(KONTUR):,} bytes")


def norm(s):
    """Letters-only, diacritic-free. Letters-only is what neutralises the EN DASH."""
    s = unicodedata.normalize("NFKD", str(s).strip())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^A-Za-z]+", "", s).lower()
    return ALIAS.get(s, s)


def build_units():
    import geopandas as gpd
    import pandas as pd

    for p in (ADM1, ADM2):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run sources/ge_geo.py --fetch first")
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/ge.py first")

    g = gpd.read_file(ADM1)
    g = (g.set_crs(4326) if g.crs is None else g.to_crs(4326))
    print(f"  geoBoundaries GEO ADM1: {len(g)} polygons")
    if len(g) != EXPECTED_GB:
        raise SystemExit(f"expected {EXPECTED_GB} polygons, got {len(g)}")
    g["key"] = [norm(n) for n in g["shapeName"]]

    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    reg = (df[df["geo_level"] == "region"][["geo_id", "geo_name"]]
           .drop_duplicates("geo_id").copy())
    reg["key"] = [norm(n) for n in reg["geo_name"]]
    if len(reg) != EXPECTED_UNITS:
        raise SystemExit(f"{len(reg)} census regions, expected {EXPECTED_UNITS}")

    for label, s in (("geoBoundaries", g), ("census", reg)):
        dup = s["key"][s["key"].duplicated(keep=False)]
        if len(dup):
            raise SystemExit(f"{label} keys are not unique: {sorted(set(dup))}")

    only_c = set(reg["key"]) - set(g["key"])
    only_g = set(g["key"]) - set(reg["key"])
    print("\n  the name join, both ways (§12 — a count match is not a join):")
    print(f"    matched                      {len(set(reg['key']) & set(g['key'])):>3}")
    print(f"    census regions with no polygon {len(only_c):>3}")
    print(f"    polygons with no census region {len(only_g):>3}  {sorted(only_g)}")
    if only_c:
        raise SystemExit(f"census regions with no polygon: {sorted(only_c)}")
    if only_g != {NOT_ENUMERATED_ADM1}:
        raise SystemExit(f"expected only {NOT_ENUMERATED_ADM1!r} spare, got {sorted(only_g)}")
    print(f"    OK  the only spare polygon is Abkhazia, which the census did not enumerate")

    out = g.merge(reg, on="key", how="inner").rename(
        columns={"geo_id": "unit", "shapeName": "geo_name_gb"})
    out = out[["unit", "geo_name", "geo_name_gb", "geometry"]]
    out["area_km2"] = out.to_crs(32638).area.values / 1e6      # UTM 38N
    print(f"    enumerated area {out['area_km2'].sum():,.0f} km², "
          f"median region {out['area_km2'].median():,.0f} km²")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(UNITS_OUT, layer="regions", driver="GPKG")
    print(f"\n  wrote {UNITS_OUT}  {len(out)} regions")
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
        raise SystemExit(f"missing {KONTUR} -- run sources/ge_geo.py --fetch first")

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
    print(f"    {outside.sum():,} hexes fall outside the {len(units)} enumerated regions "
          f"({hexes.loc[outside, 'population'].sum():,.0f} people) — Abkhazia, and the "
          "border strip Kontur rounds outwards")
    hexes = hexes[~outside].copy()

    # ---- subtract the Tskhinvali region, which has no ADM1 of its own -----------------
    a2 = gpd.read_file(ADM2)
    a2 = (a2.set_crs(4326) if a2.crs is None else a2.to_crs(4326))
    so = a2[a2["shapeName"].isin(NOT_ENUMERATED_ADM2)]
    if len(so) != len(NOT_ENUMERATED_ADM2):
        raise SystemExit(f"expected {NOT_ENUMERATED_ADM2} in ADM2, found "
                         f"{sorted(so['shapeName'])}")
    centres = gpd.GeoDataFrame(geometry=hexes.geometry.representative_point(),
                               crs=4326, index=hexes.index)
    inso = gpd.sjoin(centres, so[["shapeName", "geometry"]], how="left", predicate="within")
    inso = inso[~inso.index.duplicated(keep="first")]
    drop = inso["shapeName"].notna().to_numpy()
    print(f"    dropping {drop.sum():,} hexes ({hexes.loc[drop, 'population'].sum():,.0f} "
          f"people) in {', '.join(NOT_ENUMERATED_ADM2)} — the Tskhinvali region, which the "
          "census did not enumerate")
    hexes = hexes[~drop].copy()

    # ---- independent check: Kontur against the census, per region (§9p) ---------------
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    tot = df[(df["geo_level"] == "region") & (df["source_category"] == "Total")]
    tot = dict(zip(tot["geo_id"], tot["count"]))
    kon = hexes.groupby("unit")["population"].sum().to_dict()
    name = dict(zip(units["unit"], units["geo_name"]))

    rows = [(u, tot[u], kon.get(u, 0.0)) for u in units["unit"] if u in tot]
    ring = [r for r in rows if name[r[0]] in CITY_RING]
    rest = [r for r in rows if name[r[0]] not in CITY_RING]
    ratios = sorted((k / c, name[u]) for u, c, k in rest if c)
    print(f"\n  independent check — Kontur 2023 population / census 2014 count:")
    print(f"    the {len(ratios)} regions outside the Tbilisi pair: "
          f"median {ratios[len(ratios) // 2][0]:.2f}x, "
          f"min {ratios[0][0]:.2f} ({ratios[0][1]}), "
          f"max {ratios[-1][0]:.2f} ({ratios[-1][1]})")
    bad = [(r, u) for r, u in ratios if r < 0.7 or r > 1.4]
    if bad:
        print(f"    !! {len(bad)} outside 0.7-1.4x — a correct join keeps every unit in a "
              "tight band, a scrambled one does not:")
        for r, u in bad[:12]:
            print(f"       {u:34s} {r:.2f}")
        raise SystemExit("the population check failed -- the join is suspect")
    print("    every one inside 0.7-1.4x, which a scrambled join would not be")
    print("\n    THE CITY/RING PAIR (§9q), reported and NOT asserted — geoBoundaries' "
          "Tbilisi is\n    249 km² against the city's ~500, so outer Tbilisi sits in its "
          "ring region here:")
    for u, c, k in ring:
        print(f"       {name[u]:34s} census {c:>9,}   Kontur {k:>9,.0f}   "
              f"{k / c if c else float('nan'):>5.2f}x")

    print(f"\n  clipping {len(hexes):,} hexes to their region…")
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
    print(f"    {n_clipped:,} boundary hexes clipped, "
          f"{len(hexes) - n_clipped:,} left whole")

    hexes["geometry"] = gpd.GeoSeries(out, crs=4326, index=hexes.index)
    empty = hexes.geometry.is_empty | hexes.geometry.isna()
    if empty.any():
        print(f"    dropped {empty.sum():,} hexes whose clip came out empty")
        hexes = hexes[~empty].copy()

    hexes = hexes.rename(columns={"population": "pop"})[["unit", "pop", "geometry"]]

    missing = sorted(set(units["unit"]) - set(hexes["unit"]))
    if missing:
        raise SystemExit(f"regions with no hex at all: {[name[u] for u in missing]}")
    print(f"    all {units['unit'].nunique()} regions are covered")

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
