"""Montenegro — the 23 municipalities, and a population grid to place dots inside them.

Writes:
    data/geo/me/me_municipalities.gpkg   23 polygons keyed as me.py keys them  (`units`)
    data/geo/me/me_grid_400m.gpkg        Kontur H3 r8 hexes with `unit` and `pop` (`place`)

Usage:
    python sources/me_geo.py --fetch   # two downloads: geoBoundaries ~2 MB, Kontur ~0.5 MB
    python sources/me_geo.py           # build both layers

**MONTENEGRO IS THE SECOND HOLE IN THE GISCO LAU FILE, AND UNLIKE KOSOVO'S IT IS NOT ABOUT
STATUS.** §9e's rule is that European boundaries are free in `LAU_RG_01M_2021_4326.shp`, and
that file does cover the candidates — Albania has its 61 bashki in it, Serbia 169, North
Macedonia 80, Liechtenstein its 11. **Montenegro has zero features, and so does Bosnia.**
Checked directly rather than inferred: the two are simply absent. So geoBoundaries MNE ADM1
supplies the polygons, as it did for Kosovo (§9w).

**THE JOIN IS BY NAME AND IT IS EASY — THE VINTAGE IS THE HARD PART.** geoBoundaries writes
`Andrijevica Municipality` where MONSTAT writes `Andrijevica`, and stripping that one English
suffix matches 23 of 23 with no aliases and no stemming. What does not match is the count:
**the census has 25 municipalities and geoBoundaries has 23.** Montenegro has been splitting
new municipalities out of old ones for a decade — Petnjica off Berane in 2013, Gusinje off
Plav in 2014, **Tuzi off Podgorica in 2018 and Zeta off Podgorica in 2022** — and this
geoBoundaries cut predates the last two. It has Petnjica and Gusinje; it has no Tuzi and no
Zeta.

`sources/me.py` therefore folds Tuzi and Zeta back into Podgorica, which is where the polygon
still puts them, and the drawn geography is 23. **The cost is a real one and worth naming:
Tuzi is Montenegro's Albanian municipality** — heavily Catholic and Muslim in an Orthodox
country — and merging it into Podgorica averages exactly the contrast a religion map exists
to show. This is a boundary limit and not a data one: MONSTAT publishes both units, and
either a newer geoBoundaries release or an OSM `admin_level=6` extract would restore them.
**That is the single best upgrade available to this country.**

**THE INDEPENDENT CHECK IS §9p's AND MONTENEGRO PASSES IT ORDINARILY**, which is worth
stating because Kosovo did not: every unit's Kontur-population / census-count ratio has to
sit in a tight band, and a scrambled name join scatters it. Podgorica is the unit to watch,
because it is the one whose census figure was assembled by merging three rows — if the merge
were wrong it would show up here as an outlier and nowhere else.

**WHY A POPULATION GRID AND NOT §8.2's EQUAL SHARE.** 23 municipalities over 13,800 km² is
600 km² each, and Montenegro's people are on the coast, in the Zeta plain and along the
northern river valleys, not on Durmitor or the Komovi. Uniform scatter would paint mountain —
the same argument as Kosovo, more so, because the relief is worse.

The two approximations are Kosovo's and are bounded by the hex size: a hex belongs to the
municipality containing its centre and is then clipped to it. Kontur's populations are a
model and are used ONLY as relative weights inside a unit — every municipality's dot count
comes from the census.
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
GEO = os.path.join(ROOT, "data", "geo", "me")
RAW = os.path.join(ROOT, "data", "raw", "me")
NORM = os.path.join(ROOT, "data", "normalized", "me.csv")

ADM1 = os.path.join(RAW, "geoBoundaries-MNE-ADM1.geojson")
ADM1_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/"
            "MNE/ADM1/geoBoundaries-MNE-ADM1.geojson")

KONTUR_DIR = os.path.join(ROOT, "data", "geo", "kontur")
KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.eu-central-1.amazonaws.com/"
              "kontur_datasets/kontur_population_ME_20231101.gpkg.gz")
KONTUR_GZ = os.path.join(KONTUR_DIR, "kontur_population_ME_20231101.gpkg.gz")
KONTUR = os.path.join(KONTUR_DIR, "kontur_population_ME_20231101.gpkg")

UNITS_OUT = os.path.join(GEO, "me_municipalities.gpkg")
GRID_OUT = os.path.join(GEO, "me_grid_400m.gpkg")

EXPECTED_UNITS = 23
SUFFIX = " municipality"

# Split out of Podgorica after this boundary cut; me.py merges their counts back in.
NO_POLYGON = ("Tuzi", "Zeta")

# UTM 34N covers all of Montenegro.
UTM = 32634


def fetch():
    import gzip
    import shutil

    import requests

    ua = {"User-Agent": "religiondots/1.0"}
    os.makedirs(RAW, exist_ok=True)
    if not os.path.exists(ADM1) or os.path.getsize(ADM1) < 50_000:
        print("GET", ADM1_URL)
        r = requests.get(ADM1_URL, timeout=600, headers=ua)
        r.raise_for_status()
        # §5a: HTTP 200 is not a download.
        if b'"FeatureCollection"' not in r.content[:400]:
            raise SystemExit(f"not GeoJSON: first bytes {r.content[:120]!r}")
        with open(ADM1, "wb") as fh:
            fh.write(r.content)
        print(f"  {os.path.getsize(ADM1):,} bytes")
    else:
        print("already have", ADM1)

    os.makedirs(KONTUR_DIR, exist_ok=True)
    if os.path.exists(KONTUR) and os.path.getsize(KONTUR) > 300_000:
        print("already have", KONTUR)
        return
    if not os.path.exists(KONTUR_GZ) or os.path.getsize(KONTUR_GZ) < 50_000:
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


def norm(s):
    """Diacritic-free, letters-only, geoBoundaries' English ` Municipality` suffix removed.

    Montenegrin uses Š, Ž, Ć, Č, Đ and MONSTAT writes them; geoBoundaries writes some of
    them and transliterates others, so folding diacritics is what makes the join 23/23.
    """
    s = str(s).strip()
    if s.lower().endswith(SUFFIX):
        s = s[: -len(SUFFIX)]
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # Đ/đ carry no combining form and have to be spelled out.
    s = s.replace("Đ", "Dj").replace("đ", "dj")
    return re.sub(r"[^A-Za-z]+", "", s).lower()


def build_units():
    import geopandas as gpd
    import pandas as pd

    if not os.path.exists(ADM1):
        raise SystemExit(f"missing {ADM1} -- run sources/me_geo.py --fetch first")
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/me.py first")

    g = gpd.read_file(ADM1)
    if g.crs is None:
        g = g.set_crs(4326)
    g = g.to_crs(4326)
    print(f"  geoBoundaries MNE ADM1: {len(g)} polygons")
    if len(g) != EXPECTED_UNITS:
        raise SystemExit(
            f"expected {EXPECTED_UNITS} polygons, got {len(g)}. If this release has grown "
            f"to 25 it now carries {' and '.join(NO_POLYGON)} — which is the upgrade this "
            "module's docstring asks for, and sources/me.py's merge should be dropped "
            "rather than this number bumped.")
    g["key"] = [norm(n) for n in g["shapeName"]]

    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    muni = (df[df["geo_level"] == "municipality"][["geo_id", "geo_name"]]
            .drop_duplicates("geo_id").copy())
    muni["key"] = [norm(n) for n in muni["geo_name"]]
    if len(muni) != EXPECTED_UNITS:
        raise SystemExit(f"{len(muni)} census municipalities, expected {EXPECTED_UNITS}")

    # An ambiguous join fails quietly rather than loudly, so check both sides are unique
    # before merging (§9w).
    for label, s in (("geoBoundaries", g), ("census", muni)):
        dup = s["key"][s["key"].duplicated(keep=False)]
        if len(dup):
            raise SystemExit(f"{label} keys are not unique: {sorted(set(dup))}")

    only_c = set(muni["key"]) - set(g["key"])
    only_g = set(g["key"]) - set(muni["key"])
    print("\n  the name join, both ways (§12 — a count match is not a join):")
    print(f"    matched                      {len(set(muni['key']) & set(g['key'])):>3}")
    if only_c or only_g:
        print("    census keys with no polygon:", sorted(only_c))
        print("    polygons with no census row:", sorted(only_g))
        raise SystemExit("the name join is not 1:1 -- fix it before going further")
    print(f"    OK  all {len(muni)} census units join to exactly one polygon, on the "
          "` Municipality` suffix alone")

    out = g.merge(muni, on="key", how="inner").rename(
        columns={"geo_id": "unit", "shapeName": "geo_name_gb"})
    out = out[["unit", "geo_name", "geo_name_gb", "geometry"]]
    out["area_km2"] = out.to_crs(UTM).area.values / 1e6
    print(f"    area: median {out['area_km2'].median():.0f} km², "
          f"total {out['area_km2'].sum():,.0f} km²")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(UNITS_OUT, layer="municipalities", driver="GPKG")
    print(f"\n  wrote {UNITS_OUT}  {len(out)} municipalities")
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
        raise SystemExit(f"missing {KONTUR} -- run sources/me_geo.py --fetch first")

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
    print(f"    {outside.sum():,} hexes fall outside the {len(units)} municipalities "
          f"({hexes.loc[outside, 'population'].sum():,.0f} people) — the border strip "
          "Kontur rounds outwards")
    hexes = hexes[~outside].copy()

    # ---- the independent check (§9p) --------------------------------------------------
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    tot = df[(df["geo_level"] == "municipality") & (df["source_category"] == "Ukupno")]
    tot = dict(zip(tot["geo_id"], tot["count"]))
    kon = hexes.groupby("unit")["population"].sum().to_dict()
    name = dict(zip(units["unit"], units["geo_name"]))

    ratios = sorted((kon.get(u, 0.0) / tot[u], name[u]) for u in units["unit"] if tot.get(u))
    med = ratios[len(ratios) // 2][0]
    print("\n  independent check — Kontur 2023 population / census 2023 count:")
    print(f"    median {med:.2f}x, min {ratios[0][0]:.2f} ({ratios[0][1]}), "
          f"max {ratios[-1][0]:.2f} ({ratios[-1][1]})")
    # PODGORICA IS THE ONE TO WATCH: its census figure is three rows added together (itself,
    # Tuzi, Zeta) and this is the only place a bad merge would show. Reported by name so it
    # cannot hide inside an aggregate pass.
    pod = next((r for r, u in ratios if u == "Podgorica"), None)
    if pod is not None:
        print(f"    Podgorica {pod:.2f}x — the unit whose count is Podgorica + Tuzi + Zeta "
              "added together,\n      so a wrong merge would show here and nowhere else")
    bad = [(r, u) for r, u in ratios if r < 0.5 or r > 2.0]
    if bad:
        print(f"    !! {len(bad)} units outside 0.5-2.0x — a correct join keeps every unit "
              "in a tight band:")
        for r, u in bad[:12]:
            print(f"       {u:26s} {r:.2f}")
        raise SystemExit("the population check failed -- the join is suspect")
    print(f"    every one of the {len(ratios)} inside 0.5-2.0x, which a scrambled join "
          "would not be")

    print(f"\n  clipping {len(hexes):,} hexes to their municipality…")
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
        print(f"    {len(missing)} municipalities hold no hex centre and carry their own "
              f"polygon as one cell: {[name[u] for u in missing]}")
        fill = units[units["unit"].isin(missing)][["unit", "geometry"]].copy()
        fill["pop"] = 1.0
        hexes = gpd.GeoDataFrame(pd.concat([hexes, fill[["unit", "pop", "geometry"]]],
                                           ignore_index=True), crs=4326)
    print(f"    all {units['unit'].nunique()} municipalities are covered")

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
