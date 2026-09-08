"""Armenia — the eleven marzes, and a population grid to place dots inside them.

Writes:
    data/geo/am/am_marzes.gpkg      11 polygons keyed as am.py keys them  (`units`)
    data/geo/am/am_grid_400m.gpkg   Kontur H3 r8 hexes with `unit` and `pop` (`place`)

Usage:
    python sources/am_geo.py --fetch   # geoBoundaries ADM1 and the Kontur extract
    python sources/am_geo.py           # build both layers

**ARMENIA IS NOT IN GISCO LAU** — that file is the EU27 and the candidates — so this is
geoBoundaries at **ADM1**, which for Armenia is the ten marzes plus the city of Yerevan. The
census units are the same eleven and there is no spare polygon and no missing one, which is
the easy case Georgia next door was not (Abkhazia and South Ossetia).

**THE JOIN IS ELEVEN NAMES AND IT IS ASSERTED BOTH WAYS**, not eyeballed: a count match is
not a join ([[reference_name_join_wrong_neighbour]]). Armstat writes the marz name in the
genitive, which is how Armenian names a province, so the romanisations disagree on the ending:
`Aragatsotni`/`Aragatsotn`, `Lorri`/`Lori`, `Syuniki`/`Syunik`. am.py already emits the
nominative English form, so the fold only has to survive case and punctuation; the aliases
below are the ones that still do not close, and each is named rather than hidden behind a
fuzzy match.

**AND THE JOIN IS PROVEN ON POPULATION, NOT ON NAMES ALONE.** Kontur's modelled 2023
population is compared to Armstat's own 2022 census total per marz. Two same-sized marzes
swapped by a name join would be invisible to a count check and obvious here.

**THE CITY/RING PAIR APPEARS AGAIN, AND HERE IT STAYS INSIDE THE BAND.** §9q's Tbilisi
problem is a capital whose polygon is smaller than its own built-up area, and Armenia has the
same shape: **Yerevan comes out at 0.79x and Kotayk, the marz wrapped around its northern
edge, at 1.34x** — the two extremes of the eleven, and the towns between them (Abovyan, Nor
Hachn, Charentsavan) are Kotayk's on paper and Yerevan's in practice. Both are inside 0.7-1.4
so both are asserted rather than excused, and this is a PLACEMENT fact and not a count fact:
every marz's dot count still comes from the census. The other nine sit between 0.86 and 1.10.

WHY A POPULATION GRID AND NOT §8.2's EQUAL SHARE. Eleven units over 29,700 km2 is 2,700 km2
each, and Armenia is a plateau cut by gorges with the people in the valleys and around Sevan.
A uniform scatter would put dots on rock.
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
GEO = os.path.join(ROOT, "data", "geo", "am")
RAW = os.path.join(ROOT, "data", "raw", "am")
NORM = os.path.join(ROOT, "data", "normalized", "am.csv")

GB = "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/ARM/"
ADM1 = os.path.join(RAW, "geoBoundaries-ARM-ADM1.geojson")

KONTUR_DIR = os.path.join(ROOT, "data", "geo", "kontur")
KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.eu-central-1.amazonaws.com/"
              "kontur_datasets/kontur_population_AM_20231101.gpkg.gz")
KONTUR_GZ = os.path.join(KONTUR_DIR, "kontur_population_AM_20231101.gpkg.gz")
KONTUR = os.path.join(KONTUR_DIR, "kontur_population_AM_20231101.gpkg")

UNITS_OUT = os.path.join(GEO, "am_marzes.gpkg")
GRID_OUT = os.path.join(GEO, "am_grid_400m.gpkg")

EXPECTED_UNITS = 11
UTM = 32638                      # UTM 38N, 42-48E: Armenia sits inside it

# geoBoundaries' romanisation against am.py's. Letters-only folding takes most of it; these
# are the endings it does not. Genitive `-i` is how Armenian forms a province name.
ALIAS = {
    "aragatsotni": "aragatsotn",
    "ararati": "ararat",
    "armaviri": "armavir",
    "gegharkuniki": "gegharkunik",
    "lorri": "lori",
    "kotayki": "kotayk",
    "shiraki": "shirak",
    "syuniki": "syunik",
    "tavushi": "tavush",
    "vayotsdzori": "vayotsdzor",
    "yerevani": "yerevan",
    "erevan": "yerevan",
}

# Reported rather than asserted where a unit is a city or the ring around one; filled in
# from the first run rather than guessed. Empty means every unit is inside the band.
CITY_RING = ()
BAND = (0.7, 1.4)


def fetch():
    import gzip
    import shutil

    import requests

    ua = {"User-Agent": "religiondots/1.0"}
    os.makedirs(RAW, exist_ok=True)
    if not (os.path.exists(ADM1) and os.path.getsize(ADM1) > 100_000):
        url = f"{GB}ADM1/geoBoundaries-ARM-ADM1.geojson"
        print("GET", url)
        r = requests.get(url, timeout=600, headers=ua)
        r.raise_for_status()
        if b'"FeatureCollection"' not in r.content[:400]:
            raise SystemExit(f"not GeoJSON: first bytes {r.content[:120]!r}")
        with open(ADM1, "wb") as fh:
            fh.write(r.content)
        print(f"  {os.path.getsize(ADM1):,} bytes")
    else:
        print("already have", ADM1)

    os.makedirs(KONTUR_DIR, exist_ok=True)
    if os.path.exists(KONTUR) and os.path.getsize(KONTUR) > 1_000_000:
        print("already have", KONTUR)
        return
    if not os.path.exists(KONTUR_GZ) or os.path.getsize(KONTUR_GZ) < 200_000:
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
    """Letters-only, diacritic-free, then the alias table."""
    s = unicodedata.normalize("NFKD", str(s).strip())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^A-Za-z]+", "", s).lower()
    return ALIAS.get(s, s)


def build_units():
    import geopandas as gpd
    import pandas as pd

    if not os.path.exists(ADM1):
        raise SystemExit(f"missing {ADM1} -- run sources/am_geo.py --fetch first")
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/am.py first")

    g = gpd.read_file(ADM1)
    g = (g.set_crs(4326) if g.crs is None else g.to_crs(4326))
    print(f"  geoBoundaries ARM ADM1: {len(g)} polygons")
    g["key"] = [norm(n) for n in g["shapeName"]]

    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    reg = (df[df["geo_level"] == "marz"][["geo_id", "geo_name"]]
           .drop_duplicates("geo_id").copy())
    reg["key"] = [norm(n) for n in reg["geo_name"]]
    if len(reg) != EXPECTED_UNITS:
        raise SystemExit(f"{len(reg)} census marzes, expected {EXPECTED_UNITS}")

    for label, s in (("geoBoundaries", g), ("census", reg)):
        dup = s["key"][s["key"].duplicated(keep=False)]
        if len(dup):
            raise SystemExit(f"{label} keys are not unique: {sorted(set(dup))}")

    only_c = set(reg["key"]) - set(g["key"])
    only_g = set(g["key"]) - set(reg["key"])
    print("\n  the name join, both ways (§12 — a count match is not a join):")
    print(f"    matched                       {len(set(reg['key']) & set(g['key'])):>3}")
    print(f"    census marzes with no polygon {len(only_c):>3}  {sorted(only_c)}")
    print(f"    polygons with no census marz  {len(only_g):>3}  {sorted(only_g)}")
    if only_c or only_g:
        raise SystemExit("the eleven marzes must join both ways with nothing spare")

    out = g.merge(reg, on="key", how="inner").rename(
        columns={"geo_id": "unit", "shapeName": "geo_name_gb"})
    out = out[["unit", "geo_name", "geo_name_gb", "geometry"]]
    out["area_km2"] = out.to_crs(UTM).area.values / 1e6
    print(f"    total area {out['area_km2'].sum():,.0f} km2 (Armenia is 29,743), "
          f"median marz {out['area_km2'].median():,.0f} km2")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(UNITS_OUT, layer="marzes", driver="GPKG")
    print(f"\n  wrote {UNITS_OUT}  {len(out)} marzes")
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
        raise SystemExit(f"missing {KONTUR} -- run sources/am_geo.py --fetch first")

    layers = fiona_layers(KONTUR)
    layer = "population" if "population" in layers else layers[0]
    print(f"\n  reading Kontur r8 hexes from layer {layer!r}...")
    hexes = gpd.read_file(KONTUR, layer=layer).to_crs(4326)
    print(f"    {len(hexes):,} hexes, {hexes['population'].sum():,.0f} people")

    centres = gpd.GeoDataFrame(geometry=hexes.geometry.representative_point(), crs=4326)
    hit = gpd.sjoin(centres, units[["unit", "geometry"]], how="left", predicate="within")
    hit = hit[~hit.index.duplicated(keep="first")]
    hexes["unit"] = hit["unit"].to_numpy()

    outside = hexes["unit"].isna()
    print(f"    {outside.sum():,} hexes fall outside the {len(units)} marzes "
          f"({hexes.loc[outside, 'population'].sum():,.0f} people) — the border strip "
          "Kontur rounds outwards")
    hexes = hexes[~outside].copy()

    # ---- independent check: Kontur against the census, per marz (§9p) ----------------
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    tot = df[(df["geo_level"] == "marz") & (df["source_category"] == "Total")]
    tot = dict(zip(tot["geo_id"], tot["count"].astype(int)))
    if len(tot) != EXPECTED_UNITS:
        raise SystemExit(f"{len(tot)} marz totals in {NORM}, expected {EXPECTED_UNITS}")
    kon = hexes.groupby("unit")["population"].sum().to_dict()
    name = dict(zip(units["unit"], units["geo_name"]))

    rows = [(u, tot[u], kon.get(u, 0.0)) for u in units["unit"] if u in tot]
    ring = [r for r in rows if name[r[0]] in CITY_RING]
    rest = [r for r in rows if name[r[0]] not in CITY_RING]
    ratios = sorted((k / c, name[u]) for u, c, k in rest if c)
    print("\n  independent check — Kontur 2023 population / census 2022 count:")
    for r, u in ratios:
        print(f"    {u:<14} {r:5.2f}x")
    print(f"    {len(ratios)} marzes: median {ratios[len(ratios) // 2][0]:.2f}x, "
          f"min {ratios[0][0]:.2f} ({ratios[0][1]}), "
          f"max {ratios[-1][0]:.2f} ({ratios[-1][1]})")
    bad = [(r, u) for r, u in ratios if not BAND[0] <= r <= BAND[1]]
    if bad:
        print(f"    !! {len(bad)} outside {BAND[0]}-{BAND[1]}x — a correct join keeps every "
              "unit in a tight band, a scrambled one does not:")
        for r, u in bad[:12]:
            print(f"       {u:<14} {r:.2f}")
        raise SystemExit("the population check failed -- the join is suspect")
    print(f"    every one inside {BAND[0]}-{BAND[1]}x, which a scrambled join would not be")
    for u, c, k in ring:
        print(f"    reported, not asserted: {name[u]:<14} census {c:>9,}  "
              f"Kontur {k:>9,.0f}  {k / c if c else float('nan'):>5.2f}x")

    print(f"\n  clipping {len(hexes):,} hexes to their marz...")
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
        raise SystemExit(f"marzes with no hex at all: {[name[u] for u in missing]}")
    print(f"    all {units['unit'].nunique()} marzes are covered")

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
