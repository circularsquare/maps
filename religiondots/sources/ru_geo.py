"""Russia — the 79 federal subjects, and a 3km population grid to place dots inside them.

Writes:
    data/geo/ru/ru_subjects.gpkg    79 ADM1 polygons keyed by ISO 3166-2   (the `units`)
    data/geo/ru/ru_grid_3km.gpkg    Kontur H3 r6 hexes, carrying `unit` and `pop`  (`place`)

Usage:
    python sources/ru_geo.py --fetch    # download geoBoundaries ADM1 (59MB)
    python sources/ru_geo.py            # build both layers

WHY RUSSIA CANNOT USE THE §8.2 TRICK, AND IS THE CLEAREST CASE OF IT ON THE MAP.

spec §8.2 places dots by splitting a unit's dots equally over a finer layer, on the grounds
that statistical agencies build those layers to a population target so an equal share is
already a population weighting. Russia has no such layer and could not benefit from one
anyway, because the counts are at FEDERAL SUBJECT — a mean of 1.8 million people over a mean
of 200,000 km². Sakha alone is 3.08 million km² with a million people living almost entirely
along four rivers. Spread evenly, its dots would cover an area the size of India.

So the weight has to be a real population surface, and §8.2d's rule applies: where a fine
layer carries a population, use it and stop reasoning about proxies. Kontur's H3 grid is on
HDX under CC BY, is built from GHSL, HRSL and building footprints, and is already named in
sources.md §5 as the project's population layer of choice.

WHY r6 (3km) AND NOT r8 (400m), which is the same argument de_grid.py makes for Germany.
The dot value is the binding constraint, not the grid. Russia draws about 135,000 dots at
1:1,000 and r6 gives it 107,240 populated hexes inside the drawn subjects — the same order,
so the grid is not what limits the picture. r8 is a 2.4GB download for roughly 13× more
cells than anything can be shown at, and — the part that actually settles it — every dot
inside a subject is drawn from ONE distribution, because that is the only thing Arena
measures. A finer grid would place the same undifferentiated mixture more precisely and
tell the reader nothing more.

Kontur only ships hexes that hold people, so the 107,240 cover about 3.9 million km² of a
16.4 million km² country. That is the single most useful thing this layer does for Russia:
everywhere else is empty, and the dots know it.

TWO APPROXIMATIONS, both stated, both the same ones de_grid.py names:

  * A hex is assigned to the subject containing its CENTRE, so a boundary hex's people may
    belong to either side. Bounded by the hex size, and it moves weight within Russia, never
    a count.
  * Each hex is then CLIPPED to its subject, so a dot cannot land outside its own unit or
    across the coast. Without it the Arctic and Pacific hexes hang over open water.

AND ONE THING THAT IS NOT AN APPROXIMATION BUT LOOKS LIKE ONE. The hex populations are
Kontur's model, not Rosstat's count, and they are used ONLY as relative weights inside a
subject — the number of dots in each subject comes from the census population in
sources/ru.py and never from this file. The agreement between the two is REPORTED at the end
of this script rather than asserted, on §9i's principle: they measure different things (a
2023 modelled surface against a 2021 enumeration) and demanding equality would either fail
on every honest difference or be loosened until it detected nothing. What a correct join
looks like is every subject's ratio sitting in a tight band; what a scrambled one looks like
is that band spanning orders of magnitude.
"""

import argparse
import os
import sys
import urllib.request

# Cap BLAS before numpy is imported anywhere below it: on this machine it otherwise sizes
# itself to every core for work that is not the bottleneck.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEO = os.path.join(ROOT, "data", "geo", "ru")
KONTUR_DIR = os.path.join(ROOT, "data", "geo", "kontur")

ADM1_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/"
            "RUS/ADM1/geoBoundaries-RUS-ADM1.geojson")
ADM1 = os.path.join(GEO, "geoBoundaries-RUS-ADM1.geojson")
KONTUR = os.path.join(KONTUR_DIR, "kontur_population_20231101_r6.gpkg")

SUBJECTS_OUT = os.path.join(GEO, "ru_subjects.gpkg")
GRID_OUT = os.path.join(GEO, "ru_grid_3km.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "ru.csv")
FILLED = os.path.join(ROOT, "data", "normalized", "ru_filled.csv")

# Russia in EPSG:3857, the CRS Kontur is delivered in — and it takes TWO boxes, because
# Chukotka crosses 180°. That was not true until 2026-09-05: Arena did not cover Chukotka,
# so one box sufficed and the docstring said the antimeridian needed no handling. Filling
# the four missing subjects made it wrong, which is the ordinary way a bbox goes stale.
# Both are padded generously; the sjoin does the real filtering.
RU_BBOX_3857 = [
    (2_000_000.0, 4_800_000.0, 20_037_508.0, 17_500_000.0),      # 18°E .. 180°
    (-20_037_508.0, 6_000_000.0, -18_400_000.0, 13_000_000.0),   # 180° .. 165°W, Chukotka
]


def fetch():
    os.makedirs(GEO, exist_ok=True)
    req = urllib.request.Request(ADM1_URL, headers={"User-Agent": "religiondots/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r:
        body = r.read()
    if len(body) < 1_000_000 or not body.lstrip()[:1] == b"{":
        raise SystemExit(f"ADM1 download is not GeoJSON: {len(body)} bytes")
    open(ADM1, "wb").write(body)
    print(f"  geoBoundaries-RUS-ADM1.geojson  {len(body):,} bytes")


def build_subjects():
    """All 83 federal subjects, keyed by ISO 3166-2, from geoBoundaries' own shapeISO.

    It was the 79 Arena covers until 2026-09-05. `ru_fill.py` now estimates the other four
    from census ethnicity, so every subject needs a polygon and the COUNTS decide what is
    drawn — which is the right split anyway: a geography layer that silently omits a region
    makes a missing count look like missing land.
    """
    import geopandas as gpd

    sys.path.insert(0, HERE)
    from ru import WP_TO_ISO

    if not os.path.exists(ADM1):
        raise SystemExit(f"missing {ADM1} -- run sources/ru_geo.py --fetch first")

    g = gpd.read_file(ADM1)
    print(f"  geoBoundaries ADM1: {len(g)} features")
    # 83, and no Crimea or Sevastopol -- geoBoundaries follows the pre-2014 composition,
    # which is also Arena's, so the two agree about what Russia is without any editing.
    if len(g) != 83:
        print(f"  !! expected 83 ADM1 units, got {len(g)} -- the release changed")

    g = g.rename(columns={"shapeISO": "iso", "shapeName": "name"})
    dup = g["iso"][g["iso"].duplicated()].tolist()
    if dup:
        raise SystemExit(f"duplicate shapeISO in the boundary file: {dup}")

    want = set(WP_TO_ISO.values())
    have = set(g["iso"])
    missing = sorted(want - have)
    if missing:
        raise SystemExit(f"federal subjects with no polygon: {missing}")
    dropped = sorted(have - want)
    print(f"  keeping {len(want)} subjects" + (f"; dropping {dropped}" if dropped else ""))

    g = g[g["iso"].isin(want)][["iso", "name", "geometry"]].copy()
    if g.crs is None:
        g = g.set_crs(4326)
    g = g.to_crs(4326)
    os.makedirs(GEO, exist_ok=True)
    g.to_file(SUBJECTS_OUT, layer="subjects", driver="GPKG")
    print(f"  wrote {SUBJECTS_OUT}  {len(g)} subjects")
    return g


def build_grid(subjects):
    """Kontur H3 r6 hexes, assigned to a subject and clipped to it."""
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import shapely

    if not os.path.exists(KONTUR):
        raise SystemExit(
            f"missing {KONTUR}\n"
            "  Download and decompress it once:\n"
            "    https://geodata-eu-central-1-kontur-public.s3.eu-central-1.amazonaws.com/"
            "kontur_datasets/kontur_population_20231101_r6.gpkg.gz")

    print("  reading Kontur hexes in Russia's bboxes…")
    parts = []
    for box in RU_BBOX_3857:
        part = gpd.read_file(KONTUR, layer="population", bbox=box)
        print(f"    {len(part):,} hexes in {tuple(round(v / 1e6, 1) for v in box)}")
        parts.append(part)
    hexes = pd.concat(parts, ignore_index=True)
    hexes = gpd.GeoDataFrame(hexes, geometry="geom" if "geom" in hexes else "geometry",
                             crs=parts[0].crs)
    print(f"    {len(hexes):,} hexes total, {hexes['population'].sum():,.0f} people")
    hexes = hexes.to_crs(4326)

    # ---- assign each hex to the subject containing its CENTRE
    centres = gpd.GeoDataFrame(geometry=hexes.geometry.representative_point(), crs=4326)
    hit = gpd.sjoin(centres, subjects[["iso", "geometry"]], how="left", predicate="within")
    hit = hit[~hit.index.duplicated(keep="first")]     # a centre on a shared edge
    hexes["unit"] = hit["iso"].to_numpy()

    outside = hexes["unit"].isna()
    print(f"    {outside.sum():,} hexes are outside Russia's 83 subjects "
          f"({hexes.loc[outside, 'population'].sum():,.0f} people) — the bboxes reach "
          f"well into Europe and Asia and the sjoin is what actually selects Russia")
    hexes = hexes[~outside].copy()

    # ---- clip boundary hexes to their subject. Interior hexes are left alone: prepared
    # containment is cheap and the intersection against a 227,000-vertex polygon is not,
    # so testing first turns the expensive operation into a small minority of the work.
    print(f"  clipping {len(hexes):,} hexes to their subject…")
    poly = subjects.set_index("iso")["geometry"]
    geom = hexes.geometry.to_numpy()
    units = hexes["unit"].to_numpy()
    out = np.empty(len(hexes), dtype=object)
    n_clipped = 0
    for iso, parent in poly.items():
        idx = np.flatnonzero(units == iso)
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
        print(f"    dropped {empty.sum():,} hexes whose clip came out empty "
              f"({hexes.loc[empty, 'population'].sum():,.0f} people)")
        hexes = hexes[~empty].copy()

    hexes = hexes.rename(columns={"population": "pop"})[["unit", "pop", "geometry"]]
    hexes.to_file(GRID_OUT, layer="grid3km", driver="GPKG")
    print(f"  wrote {GRID_OUT}  {len(hexes):,} hexes")
    return hexes


def report(hexes):
    """Kontur's surface against the census population, as a RELATIONSHIP (spec §9i)."""
    import pandas as pd

    # Prefer the filled file so the check covers all 83; ru.csv alone is only Arena's 79.
    path = FILLED if os.path.exists(FILLED) else NORM
    if not os.path.exists(path):
        print("  (no ru.csv yet — skipping the population check)")
        return
    print(f"  checking against {os.path.basename(path)}")
    df = pd.read_csv(path)
    df = df[df["geo_level"] == "subject"]
    census = df.groupby("geo_id")["count"].sum()
    modelled = hexes.groupby("unit")["pop"].sum()

    both = pd.DataFrame({"census": census, "kontur": modelled}).dropna()
    both["ratio"] = both["kontur"] / both["census"]
    print(f"\n  Kontur surface vs 2021 census, {len(both)} subjects")
    print(f"    totals: census {both['census'].sum():,.0f}  "
          f"kontur {both['kontur'].sum():,.0f}  "
          f"({both['kontur'].sum() / both['census'].sum():.3f}x)")
    print(f"    per-subject ratio: median {both['ratio'].median():.3f}, "
          f"min {both['ratio'].min():.3f} ({both['ratio'].idxmin()}), "
          f"max {both['ratio'].max():.3f} ({both['ratio'].idxmax()})")
    bad = both[(both["ratio"] < 0.5) | (both["ratio"] > 2.0)]
    if len(bad):
        print(f"    !! {len(bad)} subjects outside a factor of two — a correct join keeps "
              f"every subject inside one:")
        for iso, r in bad.iterrows():
            print(f"       {iso:8s} census {r['census']:>11,.0f}  kontur {r['kontur']:>11,.0f}")
    else:
        print("    every subject inside a factor of two, which a scrambled join would not be")

    empty = set(census.index) - set(modelled.index)
    if empty:
        print(f"    !! subjects with counts and NO hexes: {sorted(empty)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fetch", action="store_true", help="download geoBoundaries ADM1 first")
    ap.add_argument("--subjects-only", action="store_true",
                    help="skip the Kontur grid (it is the slow half)")
    args = ap.parse_args()

    os.makedirs(GEO, exist_ok=True)
    if args.fetch:
        fetch()

    subjects = build_subjects()
    if args.subjects_only:
        return
    hexes = build_grid(subjects)
    report(hexes)


if __name__ == "__main__":
    main()
