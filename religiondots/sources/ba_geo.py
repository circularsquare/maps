"""Bosnia and Herzegovina — the 142 municipalities, and a grid to place dots inside them.

Writes:
    data/geo/ba/ba_municipalities.gpkg   142 polygons keyed as ba.py keys them  (`units`)
    data/geo/ba/ba_grid_400m.gpkg        Kontur H3 r8 hexes with `unit` and `pop` (`place`)

Usage:
    python sources/ba_geo.py --fetch   # two downloads: geoBoundaries 8 MB, Kontur 2.9 MB
    python sources/ba_geo.py           # build both layers

**BiH IS THE OTHER HOLE IN GISCO, AND FOR THE SAME REASON KOSOVO IS.** GISCO LAU 2021
covers the EU27 plus the candidates — AL, BG, CH, IS, LI, MK, NO, RS are all in it — and
Bosnia is not, because it had no candidate status when the file was cut. So §9e's
"boundaries are free in Europe" rule has a second exception. geoBoundaries BIH ADM3
supplies the 142 and Kontur supplies the grid, exactly as for Kosovo.

**AND THE geoBoundaries FILE IS DIRTY IN FOUR SPECIFIC WAYS, ALL OF THEM REPAIRED HERE
AND EACH REPAIR ASSERTED GEOMETRICALLY.** Both sides report 142 units, which is the
coincidence that makes this dangerous: §9s's missing-county trap looks exactly like this
from the outside, and a count match is not a join.

1.  **One polygon is named `Republika Srpska`** — an entity name in a municipality file.
    The census has `VIŠEGRAD` with no partner, and the polygon's centroid is
    **43.791 N, 19.301 E** against Višegrad's 43.782/19.293, with an area of 468 km²
    against Višegrad's 448. It is Višegrad, mislabelled. Renamed, with the centroid
    asserted inside a small box — so a future release that fixes the name, or that puts a
    genuinely different polygon here, fails loudly instead of drawing Višegrad's 10,668
    people somewhere else.
2.  **`Novi Grad` appears twice**, and the two are 130 km apart: Novi Grad in the RS
    (ex-Bosanski Novi, 45.02 N, 484 km²) and Novi Grad Sarajevo (43.86 N, 41 km²). A plain
    name join picks whichever pandas sees first and is wrong about **118,553 people** —
    Novi Grad Sarajevo is the second largest municipality in the country. Split by
    latitude, with both the gap and the areas asserted.
3.  **`Kupres` and `Kupres (BiH)` are the two Kupres municipalities and the labels are
    actively misleading** — `(BiH)` reads as "the state" and actually marks the
    *Federation* one. Kupres RS is the small eastern remnant at 57 km²; Kupres FBiH is
    572 km². Split by area, which is a ten-fold difference and cannot be got backwards.
4.  **`Kupra na Uni` is a typo for `Krupa na Uni`** — confirmed by centroid (44.90 N,
    16.30 E), not by the spelling being close.

**THE REPAIRS RUN FIRST AND THE JOIN IS A PLAIN NAME JOIN AFTERWARDS.** Renaming the
polygons and then joining is worth more than a clever matcher: every correction is one
named line with a test beside it, and the join itself stays something that either works
completely or fails completely.

**WHY A POPULATION GRID AND NOT §8.2's EQUAL SHARE.** 142 municipalities over 51,200 km² is
360 km² each, and Bosnia is mountains with people in the valleys — the Sava plain, the
Bosna and Vrbas corridors, the Sarajevo basin. Uniform scatter would paint the Dinarides.

**THE INDEPENDENT CHECK HAS TO ALLOW FOR TEN YEARS OF EMIGRATION**, and the band is
measured here rather than copied. §9u's warning against inheriting Kenya's band applies
with force: the counts are 2013 and Kontur's surface is 2023, and BiH lost a large share of
its population in between, so the ratio sits below 1 across the country instead of near it.
What the check is actually testing is that the ratio is *consistent* — a scrambled join
scatters it — so the band is set from the observed spread and the outliers are named.
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
GEO = os.path.join(ROOT, "data", "geo", "ba")
RAW = os.path.join(ROOT, "data", "raw", "ba")
NORM = os.path.join(ROOT, "data", "normalized", "ba.csv")

ADM3 = os.path.join(RAW, "geoBoundaries-BIH-ADM3.geojson")
ADM3_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/f549eab/releaseData/gbOpen/"
            "BIH/ADM3/geoBoundaries-BIH-ADM3.geojson")

KONTUR_DIR = os.path.join(ROOT, "data", "geo", "kontur")
KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.eu-central-1.amazonaws.com/"
              "kontur_datasets/kontur_population_BA_20231101.gpkg.gz")
KONTUR_GZ = os.path.join(KONTUR_DIR, "kontur_population_BA_20231101.gpkg.gz")
KONTUR = os.path.join(KONTUR_DIR, "kontur_population_BA_20231101.gpkg")

UNITS_OUT = os.path.join(GEO, "ba_municipalities.gpkg")
GRID_OUT = os.path.join(GEO, "ba_grid_400m.gpkg")

EXPECTED_UNITS = 142
UTM = 32633                      # UTM 33N, metres — BiH sits inside it

# Census name -> repaired geoBoundaries `shapeName`, for the pairs that are alternative
# names or different conventions rather than the same string. Keys are census strings
# verbatim; §2.4 says the key must match what sources/ba.py writes.
ALIAS = {
    "BRČKO": "Brcko District",
    "CENTAR SARAJEVO": "Centar",
    "DOBOJ-ISTOK": "Doboj East",             # geoBoundaries translates `Istok` to English
    "FOČA - F BiH": "Foča-Ustikolina",       # the Federation half, seat at Ustikolina
    "FOČA - RS": "Foča",
    "GRAD MOSTAR": "Mostar",                 # `Grad` = City; one unit since 2004
    "KRUPA NA UNI": "Kupra na Uni",          # geoBoundaries typo, see the docstring
    "KUPRES - F BiH": "Kupres (BiH)",
    "KUPRES - RS": "Kupres",
    "NOVI GRAD SARAJEVO": "Novi Grad Sarajevo",
    "NOVO GORAŽDE": "Ustiprača",             # Novo Goražde's seat; the RS half of Goražde
    "PALE - F BiH": "Pale-Prača",            # the Federation half, seat at Prača
    "PALE - RS": "Pale",
    "PROZOR": "Prozor-Rama",
    "STARI GRAD SARAJEVO": "Stari Grad",
    "TRNOVO - F BiH": "Trnovo (BiH)",        # `(BiH)` marks the Federation one here too
    # VIŠEGRAD needs no entry: `repair()` renames the mislabelled `Republika Srpska`
    # polygon to `Višegrad` before any matching happens, so it joins on its own name.
    # NOVI GRAD (the RS one) likewise keeps its name; only the Sarajevo one is renamed.
}


def fetch():
    import gzip
    import shutil

    import requests

    ua = {"User-Agent": "religiondots/1.0"}
    os.makedirs(RAW, exist_ok=True)
    if not os.path.exists(ADM3) or os.path.getsize(ADM3) < 100_000:
        print("GET", ADM3_URL)
        r = requests.get(ADM3_URL, timeout=600, headers=ua)
        r.raise_for_status()
        # §5a: HTTP 200 is not a download.
        if b'"FeatureCollection"' not in r.content[:400]:
            raise SystemExit(f"not GeoJSON: first bytes {r.content[:120]!r}")
        with open(ADM3, "wb") as fh:
            fh.write(r.content)
        print(f"  {os.path.getsize(ADM3):,} bytes")
    else:
        print("already have", ADM3)

    os.makedirs(KONTUR_DIR, exist_ok=True)
    if os.path.exists(KONTUR) and os.path.getsize(KONTUR) > 500_000:
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


def norm(s):
    """Diacritic-free, letters-only, upper. `Đ` does not decompose under NFKD."""
    s = str(s).strip().replace("Đ", "DJ").replace("đ", "dj")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^A-Za-z]+", "", s).upper()


def repair(g):
    """Fix the four defects in geoBoundaries BIH ADM3, asserting each one geometrically.

    Runs before any name matching, so that what follows is a plain join. Every branch
    raises rather than falling through: a release that has fixed one of these should stop
    the build and be looked at, not be silently accommodated.
    """
    import numpy as np

    names = g["shapeName"].tolist()
    cen = g.geometry.representative_point()
    lat = cen.y.to_numpy()
    lon = cen.x.to_numpy()
    km2 = (g.to_crs(UTM).area / 1e6).to_numpy()

    # ---- 1. `Republika Srpska` is Višegrad -------------------------------------------
    idx = [i for i, n in enumerate(names) if n == "Republika Srpska"]
    if len(idx) != 1:
        raise SystemExit(f"expected exactly one 'Republika Srpska' polygon, got {len(idx)}")
    i = idx[0]
    if not (43.6 < lat[i] < 44.0 and 19.1 < lon[i] < 19.5 and 350 < km2[i] < 600):
        raise SystemExit(
            f"the 'Republika Srpska' polygon is not where Višegrad is: "
            f"{lat[i]:.3f}N {lon[i]:.3f}E {km2[i]:,.0f} km2. It was Višegrad in the "
            f"f549eab release; do not rename it blind.")
    names[i] = "Višegrad"
    print(f"    repaired: 'Republika Srpska' -> Višegrad  "
          f"({lat[i]:.3f}N {lon[i]:.3f}E, {km2[i]:,.0f} km2)")

    # ---- 2. the two `Novi Grad` -------------------------------------------------------
    idx = [i for i, n in enumerate(names) if n == "Novi Grad"]
    if len(idx) != 2:
        raise SystemExit(f"expected exactly two 'Novi Grad' polygons, got {len(idx)}")
    north, south = (idx if lat[idx[0]] > lat[idx[1]] else idx[::-1])
    if not (lat[north] > 44.5 and lat[south] < 44.2):
        raise SystemExit(f"the two 'Novi Grad' polygons do not separate by latitude: "
                         f"{lat[north]:.3f} and {lat[south]:.3f}")
    if not (km2[north] > 300 and km2[south] < 100):
        raise SystemExit(f"'Novi Grad' areas are not the expected 484/41 km2: "
                         f"{km2[north]:,.0f} and {km2[south]:,.0f}")
    names[south] = "Novi Grad Sarajevo"
    print(f"    repaired: the southern 'Novi Grad' -> Novi Grad Sarajevo  "
          f"({lat[south]:.3f}N, {km2[south]:,.0f} km2); the RS one keeps the name "
          f"({lat[north]:.3f}N, {km2[north]:,.0f} km2)")

    # ---- 3. Kupres: check the areas rather than trusting `(BiH)` ----------------------
    a = [i for i, n in enumerate(names) if n == "Kupres"]
    b = [i for i, n in enumerate(names) if n == "Kupres (BiH)"]
    if len(a) != 1 or len(b) != 1:
        raise SystemExit(f"expected one 'Kupres' and one 'Kupres (BiH)', got "
                         f"{len(a)} and {len(b)}")
    if not (km2[a[0]] < 150 < km2[b[0]]):
        raise SystemExit(
            f"Kupres areas are not the expected 57 (RS) and 572 (FBiH) km2: "
            f"'Kupres'={km2[a[0]]:,.0f}, 'Kupres (BiH)'={km2[b[0]]:,.0f}. ALIAS maps the "
            f"census's KUPRES - RS to the small one; if that has flipped, fix ALIAS.")
    print(f"    confirmed: 'Kupres' is the RS remnant at {km2[a[0]]:,.0f} km2 and "
          f"'Kupres (BiH)' the Federation one at {km2[b[0]]:,.0f} km2")

    # ---- 4. `Kupra na Uni` is Krupa na Uni -------------------------------------------
    idx = [i for i, n in enumerate(names) if n == "Kupra na Uni"]
    if len(idx) != 1:
        raise SystemExit(f"expected one 'Kupra na Uni', got {len(idx)}")
    i = idx[0]
    if not (44.7 < lat[i] < 45.1 and 16.1 < lon[i] < 16.5):
        raise SystemExit(f"'Kupra na Uni' is not where Krupa na Uni is: "
                         f"{lat[i]:.3f}N {lon[i]:.3f}E")
    print(f"    confirmed: 'Kupra na Uni' is at {lat[i]:.3f}N {lon[i]:.3f}E, which is "
          f"Krupa na Uni; ALIAS carries the typo verbatim")

    g = g.copy()
    g["shapeName"] = names
    return g


def build_units():
    import geopandas as gpd
    import pandas as pd

    if not os.path.exists(ADM3):
        raise SystemExit(f"missing {ADM3} -- run sources/ba_geo.py --fetch first")
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/ba.py first")

    g = gpd.read_file(ADM3)
    if g.crs is None:
        g = g.set_crs(4326)
    g = g.to_crs(4326)
    print(f"  geoBoundaries BIH ADM3: {len(g)} polygons")
    if len(g) != EXPECTED_UNITS:
        raise SystemExit(f"expected {EXPECTED_UNITS} polygons, got {len(g)}")

    print("\n  repairing the file (see the module docstring):")
    g = repair(g)

    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    muni = (df[df["geo_level"] == "municipality"][["geo_id", "geo_name"]]
            .drop_duplicates("geo_id").copy())
    if len(muni) != EXPECTED_UNITS:
        raise SystemExit(f"{len(muni)} census municipalities, expected {EXPECTED_UNITS}")

    # census name -> the geoBoundaries name it should meet, then fold both
    muni["target"] = [ALIAS.get(n, n) for n in muni["geo_name"]]
    muni["key"] = [norm(n) for n in muni["target"]]
    g["key"] = [norm(n) for n in g["shapeName"]]

    for label, s in (("geoBoundaries", g), ("census", muni)):
        dup = s["key"][s["key"].duplicated(keep=False)]
        if len(dup):
            raise SystemExit(f"{label} keys are not unique after repair: "
                             f"{sorted(set(dup))} -- add a repair or an ALIAS entry rather "
                             "than loosening `norm`")

    only_c = set(muni["key"]) - set(g["key"])
    only_g = set(g["key"]) - set(muni["key"])
    print("\n  the name join, both ways (§12 — a count match is not a join):")
    print(f"    matched                      {len(set(muni['key']) & set(g['key'])):>3}")
    if only_c or only_g:
        print("    census keys with no polygon:", sorted(only_c))
        print("    polygons with no census row:", sorted(only_g))
        raise SystemExit("the name join is not 1:1 -- fix it before going further")
    print(f"    OK  all {len(muni)} census units join to exactly one polygon, "
          f"{len(ALIAS)} of them through ALIAS")

    out = g.merge(muni, on="key", how="inner").rename(
        columns={"geo_id": "unit", "shapeName": "geo_name_gb"})
    out = out[["unit", "geo_name", "geo_name_gb", "geometry"]]
    out["area_km2"] = out.to_crs(UTM).area.values / 1e6
    print(f"    area: median {out['area_km2'].median():.0f} km², "
          f"total {out['area_km2'].sum():,.0f} km² (BiH is 51,197)")

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
        raise SystemExit(f"missing {KONTUR} -- run sources/ba_geo.py --fetch first")

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

    # ---- independent check: does the ratio hold TOGETHER, not near 1 (§9u) ------------
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    tot = df[(df["geo_level"] == "municipality") & (df["source_category"] == "Ukupno")]
    tot = dict(zip(tot["geo_id"], tot["count"]))
    kon = hexes.groupby("unit")["population"].sum().to_dict()
    name = dict(zip(units["unit"], units["geo_name"]))

    ratios = sorted((kon.get(u, 0.0) / tot[u], name[u]) for u in units["unit"]
                    if tot.get(u))
    med = ratios[len(ratios) // 2][0]
    print("\n  independent check — Kontur 2023 population / census 2013 count:")
    print(f"    median {med:.2f}x, min {ratios[0][0]:.2f} ({ratios[0][1]}), "
          f"max {ratios[-1][0]:.2f} ({ratios[-1][1]})")
    # THE BAND IS MEASURED, NOT INHERITED (§9u). It is centred on the observed median
    # rather than on 1.0, because the counts are ten years older than the surface and BiH
    # emigrated heavily in between — a ratio near 1 is not what a correct join looks like
    # here. What is being tested is CONSISTENCY: a scrambled join throws units to 5x and
    # 0.1x, which this band catches while tolerating a decade of real population change.
    lo, hi = med / 3.0, med * 3.0
    bad = [(r, u) for r, u in ratios if not (lo <= r <= hi)]
    print(f"    band {lo:.2f}-{hi:.2f}x (median/3 to median*3): "
          f"{len(ratios) - len(bad)}/{len(ratios)} inside")
    for r, u in bad:
        print(f"       outside: {u:26s} {r:.2f}x")
    if len(bad) > 6:
        raise SystemExit(f"{len(bad)} units outside the band -- a correct join keeps "
                         "almost every unit together; this looks scrambled")

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
    hexes["geometry"] = out
    hexes = hexes[~hexes.geometry.is_empty & hexes.geometry.notna()].copy()
    print(f"    {n_clipped:,} edge hexes clipped to their municipality boundary")

    grid = hexes[["unit", "population", "geometry"]].rename(
        columns={"population": "pop"})
    grid = gpd.GeoDataFrame(grid, geometry="geometry", crs=4326)
    os.makedirs(GEO, exist_ok=True)
    grid.to_file(GRID_OUT, layer="grid", driver="GPKG")
    print(f"  wrote {GRID_OUT}  {len(grid):,} hexes, "
          f"{grid['pop'].sum():,.0f} modelled people")

    empty = set(units["unit"]) - set(grid.loc[grid["pop"] > 0, "unit"])
    if empty:
        print(f"    !! {len(empty)} municipalities have no populated hex: "
              f"{sorted(name[u] for u in empty)}")


def main():
    if "--fetch" in sys.argv:
        fetch()
    units = build_units()
    build_grid(units)


if __name__ == "__main__":
    main()
