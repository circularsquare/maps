"""Lithuania — the 60 municipalities, and a population grid to place dots inside them.

Writes:
    data/geo/lt/lt_municipalities.gpkg   60 polygons keyed by savivaldybė code   (`units`)
    data/geo/lt/lt_grid_400m.gpkg        Kontur H3 r8 hexes, `unit` and `pop`    (`place`)

Usage:
    python sources/lt_geo.py --fetch   # one Kontur country extract
    python sources/lt_geo.py           # build both layers

**THE EASIEST JOIN IN THE PROJECT.** GISCO LAU 2021 — already on disk since North Macedonia
— carries exactly 60 Lithuanian polygons, and its `LAU_ID` **is** the savivaldybė code that
Statistics Lithuania keys its cube on: `11` Alytaus miesto, `13` Vilniaus miesto, `41`
Vilniaus rajono. No names, no diacritics, no transliteration, no aliases, no collisions.
Serbia the day before needed a computed collision-resolver and a `Belgrade`/`Beograd` alias
for the same job (`rs_geo.md` §3); Lithuania needs a dictionary lookup.

Worth stating why, because it is the thing to look for first: **the source publishes codes
and GISCO publishes the same codes.** Romania, Ghana and Serbia all publish names only, and
every trap in their `_geo.md` files follows from that one fact.

**PLACEMENT IS A POPULATION GRID, AND LITHUANIA NEEDS IT MORE THAN SERBIA DID.** Lithuanian
savivaldybės average **1,088 km²** — more than twice a Serbian opština — and the country is
forest, farmland and a scatter of small towns. §8.2's equal share over a fine layer has
nothing to work with: these are historical districts, not units built to a population
target, and the rajono savivaldybės are doughnuts drawn around a city that is its own
separate unit. Kontur's r8 hexes (~0.74 km²) divide the average one about 1,470 ways.

Two approximations, both Russia's and Serbia's: a hex belongs to the municipality containing
its CENTRE, and is then CLIPPED to it. And one thing that is not an approximation: the hex
populations are relative weights INSIDE a unit only. Every municipality's dot count comes
from the census and never from this file.
"""

import os
import sys

# Cap BLAS before numpy is imported below it (see sources/ru_geo.py).
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEO = os.path.join(ROOT, "data", "geo", "lt")
LAU = os.path.join(ROOT, "data", "geo", "lau2021", "shp4326",
                   "LAU_RG_01M_2021_4326.shp")
KONTUR_DIR = os.path.join(ROOT, "data", "geo", "kontur")
KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.eu-central-1.amazonaws.com/"
              "kontur_datasets/kontur_population_LT_20231101.gpkg.gz")
KONTUR_GZ = os.path.join(KONTUR_DIR, "kontur_population_LT_20231101.gpkg.gz")
KONTUR = os.path.join(KONTUR_DIR, "kontur_population_LT_20231101.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "lt.csv")

UNITS_OUT = os.path.join(GEO, "lt_municipalities.gpkg")
GRID_OUT = os.path.join(GEO, "lt_grid_400m.gpkg")

EXPECTED_UNITS = 60


def fetch():
    import gzip
    import shutil

    import requests

    os.makedirs(KONTUR_DIR, exist_ok=True)
    if os.path.exists(KONTUR) and os.path.getsize(KONTUR) > 1_000_000:
        print("already have", KONTUR)
        return
    if not os.path.exists(KONTUR_GZ) or os.path.getsize(KONTUR_GZ) < 500_000:
        print("GET", KONTUR_URL)
        r = requests.get(KONTUR_URL, timeout=600,
                         headers={"User-Agent": "religiondots/1.0"})
        r.raise_for_status()
        if r.content[:2] != b"\x1f\x8b":            # §5a: 200 is not a download
            raise SystemExit(f"not gzip: first bytes {r.content[:40]!r}")
        with open(KONTUR_GZ, "wb") as fh:
            fh.write(r.content)
        print(f"  {os.path.getsize(KONTUR_GZ):,} bytes")
    with gzip.open(KONTUR_GZ, "rb") as src, open(KONTUR, "wb") as dst:
        shutil.copyfileobj(src, dst)
    print(f"  decompressed to {KONTUR}  {os.path.getsize(KONTUR):,} bytes")


def layers(path):
    import pyogrio
    return list(pyogrio.list_layers(path)[:, 0])


def build_units():
    import geopandas as gpd
    import pandas as pd

    if not os.path.exists(LAU):
        raise SystemExit(f"missing {LAU} -- unzip data/geo/lau2021/"
                         "LAU_RG_01M_2021_4326.shp.zip into shp4326/")
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/lt.py first")

    g = gpd.read_file(LAU)
    lt = g[g["CNTR_CODE"] == "LT"].copy()
    print(f"  GISCO LT polygons: {len(lt)}")
    if len(lt) != EXPECTED_UNITS:
        raise SystemExit(f"expected {EXPECTED_UNITS} GISCO polygons, got {len(lt)}")
    lt["unit"] = lt["LAU_ID"].astype(str).str.strip().str.zfill(2)

    df = pd.read_csv(NORM, dtype={"geo_id": str})
    muni = df[df["geo_level"] == "municipality"][["geo_id", "geo_name"]].drop_duplicates()
    muni["geo_id"] = muni["geo_id"].str.zfill(2)
    print(f"  census municipality rows: {len(muni)}")

    only_c = sorted(set(muni["geo_id"]) - set(lt["unit"]))
    only_g = sorted(set(lt["unit"]) - set(muni["geo_id"]))
    if only_c or only_g:
        print("  census codes with no polygon:", only_c)
        print("  polygons with no census row:", only_g)
        raise SystemExit("the code join is not 1:1")
    print(f"  OK  all {len(muni)} census codes join to exactly one polygon, by CODE")

    out = lt.merge(muni.rename(columns={"geo_id": "unit"}), on="unit", how="inner")
    out = out.rename(columns={"LAU_NAME": "gisco_name", "POP_2021": "pop_2021",
                              "AREA_KM2": "area_km2"})
    out = out[["unit", "gisco_name", "geo_name", "pop_2021", "area_km2", "geometry"]]
    out = out.to_crs(4326)

    # ---- independent check: GISCO's own population against the census (§9i) ----
    tot = df[(df["geo_level"] == "municipality") &
             (df["source_category"] == "Iš viso pagal religiją")]
    tot = {str(k).zfill(2): v for k, v in zip(tot["geo_id"], tot["count"])}
    rows = [(u, tot[u], p) for u, p in zip(out["unit"], out["pop_2021"])
            if p and p > 0 and u in tot]
    ratios = sorted((p / c, u, n) for u, c, p in rows
                    for n in [dict(zip(out["unit"], out["geo_name"]))[u]])
    print(f"\n  independent check — GISCO POP_2021 / census 2021 total, {len(rows)} units:")
    print(f"    national {sum(p for _, _, p in rows):,.0f} / "
          f"{sum(c for _, c, p in rows):,} = "
          f"{sum(p for _, _, p in rows) / sum(c for _, c, p in rows):.3f}x")
    print(f"    per unit: median {ratios[len(ratios) // 2][0]:.3f}, "
          f"min {ratios[0][0]:.3f} ({ratios[0][2]}), "
          f"max {ratios[-1][0]:.3f} ({ratios[-1][2]})")
    bad = [(r, n) for r, u, n in ratios if r < 0.85 or r > 1.2]
    if bad:
        print(f"    !! {len(bad)} units outside 0.85-1.2x:")
        for r, n in bad[:12]:
            print(f"       {n:34s} {r:.3f}")
        raise SystemExit("the population check failed -- the join is suspect")
    print("    every unit inside 0.85-1.2x, which a scrambled join would not be")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(UNITS_OUT, layer="municipalities", driver="GPKG")
    print(f"\n  wrote {UNITS_OUT}  {len(out)} municipalities")
    return out


def build_grid(units):
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import shapely

    if not os.path.exists(KONTUR):
        raise SystemExit(f"missing {KONTUR} -- run sources/lt_geo.py --fetch first")

    names = layers(KONTUR)
    layer = "population" if "population" in names else names[0]
    print(f"  reading Kontur r8 hexes from layer {layer!r}…")
    hexes = gpd.read_file(KONTUR, layer=layer)
    print(f"    {len(hexes):,} hexes, {hexes['population'].sum():,.0f} people")
    hexes = hexes.to_crs(4326)

    centres = gpd.GeoDataFrame(geometry=hexes.geometry.representative_point(), crs=4326)
    hit = gpd.sjoin(centres, units[["unit", "geometry"]], how="left", predicate="within")
    hit = hit[~hit.index.duplicated(keep="first")]
    hexes["unit"] = hit["unit"].to_numpy()

    outside = hexes["unit"].isna()
    print(f"    {outside.sum():,} hexes fall outside the 60 municipalities "
          f"({hexes.loc[outside, 'population'].sum():,.0f} people) — the border strip "
          "Kontur rounds outwards, and the Curonian Lagoon")
    hexes = hexes[~outside].copy()

    print(f"  clipping {len(hexes):,} hexes to their municipality…")
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

    # A unit smaller than one cell holds no hex centre and would vanish from the layer
    # (rs_geo.py §5, de_grid.py's 34 Gemeinden). None of Lithuania's 60 is that small.
    missing = sorted(set(units["unit"]) - set(hexes["unit"]))
    if missing:
        print(f"    {len(missing)} municipalities hold no hex centre and carry their own "
              f"polygon as one cell: {missing}")
        fill = units[units["unit"].isin(missing)][["unit", "geometry"]].copy()
        fill["pop"] = 1.0
        hexes = gpd.GeoDataFrame(pd.concat([hexes, fill[["unit", "pop", "geometry"]]],
                                           ignore_index=True), crs=4326)
    print(f"    all {units['unit'].nunique()} municipalities are covered")

    hexes.to_file(GRID_OUT, layer="grid400m", driver="GPKG")
    print(f"  wrote {GRID_OUT}  {len(hexes):,} hexes")
    return hexes


def report(units, hexes):
    import pandas as pd

    df = pd.read_csv(NORM, dtype={"geo_id": str})
    tot = df[(df["geo_level"] == "municipality") &
             (df["source_category"] == "Iš viso pagal religiją")]
    census = pd.Series({str(k).zfill(2): v for k, v in zip(tot["geo_id"], tot["count"])})
    modelled = hexes.groupby("unit")["pop"].sum()
    name = dict(zip(units["unit"], units["geo_name"]))
    both = pd.DataFrame({"census": census, "kontur": modelled}).dropna()
    both["ratio"] = both["kontur"] / both["census"]
    print(f"\n  Kontur surface vs the 2021 census, {len(both)} municipalities")
    print(f"    totals: census {both['census'].sum():,.0f}  "
          f"kontur {both['kontur'].sum():,.0f}  "
          f"({both['kontur'].sum() / both['census'].sum():.3f}x)")
    print(f"    per-unit ratio: median {both['ratio'].median():.3f}, "
          f"min {both['ratio'].min():.3f} ({name.get(both['ratio'].idxmin())}), "
          f"max {both['ratio'].max():.3f} ({name.get(both['ratio'].idxmax())})")
    n_hex = hexes.groupby("unit").size()
    print(f"    hexes per municipality: median {n_hex.median():.0f}, "
          f"min {n_hex.min()} ({name.get(n_hex.idxmin())}), "
          f"max {n_hex.max()} ({name.get(n_hex.idxmax())})")
    off = both[(both["ratio"] < 0.6) | (both["ratio"] > 1.5)].sort_values("ratio")
    print(f"    {len(off)} municipalities outside 0.6-1.5x — where placement inside the "
          "unit is weakest:")
    for unit, r in off.iterrows():
        print(f"       {name.get(unit, unit):26s} census {r['census']:>8,.0f}  "
              f"kontur {r['kontur']:>9,.0f}  {r['ratio']:.2f}x")

    # ---- and the check that says those outliers are Kontur and not a bad join ----
    #
    # Six Lithuanian cities are their own municipality sitting as an ENCLAVE inside the
    # rajono municipality named after them. Every outlier above is half of such a pair —
    # the city low, its ring high — which is what a modelled population surface does to a
    # dense city: GHSL and building footprints spread apartment-block population outward
    # across the boundary. A WRONG JOIN would look different: it would put one unit's
    # people somewhere unrelated, and the pair would not close.
    #
    # The pairs are derived from the names rather than listed, on rs_geo.py's principle.
    stem = {}
    for u, n in name.items():
        s = n.replace(" miesto savivaldybė", "").replace(" rajono savivaldybė", "")
        s = s.replace(" m. sav.", "").replace(" r. sav.", "").strip()
        if s != n.strip():
            stem.setdefault(s, []).append(u)
    pairs = {s: us for s, us in stem.items() if len(us) == 2}
    print(f"\n    {len(pairs)} city/ring pairs, checked TOGETHER — a displaced surface "
          "closes, a bad join does not:")
    for s, us in sorted(pairs.items()):
        have = [u for u in us if u in both.index]
        if len(have) != 2:
            continue
        c = both.loc[have, "census"].sum()
        k = both.loc[have, "kontur"].sum()
        flag = "  <-- does NOT close" if not 0.75 <= k / c <= 1.3 else ""
        print(f"       {s:16s} census {c:>9,.0f}  kontur {k:>9,.0f}  "
              f"{k / c:.2f}x{flag}")


def main():
    if "--fetch" in sys.argv:
        fetch()
    units = build_units()
    hexes = build_grid(units)
    report(units, hexes)


if __name__ == "__main__":
    main()
