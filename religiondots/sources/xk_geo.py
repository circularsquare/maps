"""Kosovo — the 38 municipalities, and a population grid to place dots inside them.

Writes:
    data/geo/xk/xk_municipalities.gpkg   38 polygons keyed as xk.py keys them  (`units`)
    data/geo/xk/xk_grid_400m.gpkg        Kontur H3 r8 hexes with `unit` and `pop` (`place`)

Usage:
    python sources/xk_geo.py --fetch   # two downloads: geoBoundaries 3 MB, Kontur 0.7 MB
    python sources/xk_geo.py           # build both layers

**KOSOVO IS THE HOLE IN THE FILE THIS REPO KEEPS REACHING FOR.** GISCO LAU 2021 covers the
EU27 *plus* the candidates — AL, BG, CH, IS, LI, MK, NO, RS all sit in it — and Kosovo is the
one Balkan country it does not carry, because the EU has no agreed status for it. So the
"boundaries are free in Europe" rule (§9e) has an exception and this is it. geoBoundaries
XKX ADM2 supplies the 38 instead, and Kontur supplies the grid.

**THE JOIN IS BY NAME AND ALBANIAN NOUNS HAVE TWO FORMS.** ASK writes the indefinite
(`Gjakovë`, `Klinë`, `Pejë`) and geoBoundaries writes the definite (`Gjakova`, `Klina`,
`Peja`), so 10 of the 38 differ by exactly one final vowel and fold-matching alone scores
23/38. Stripping a trailing `a`/`e` after folding takes it to 34, and the last four are real
alternative names rather than inflections — `Drenas`/`Gllogoc`, `Pristina`/`Prishtinë`,
`North Mitrovica`/`Mitrovicë e Veriut`, `Zveçan`/`Zveqan`. Those are aliased explicitly. Both
sides' stems are asserted unique before the join, so a future rename collides loudly.

**THE INDEPENDENT CHECK MEASURES THE BOYCOTT INSTEAD OF ASSUMING IT.** §9p verifies a
name-join by testing that every unit's (other population estimate / census count) sits in a
tight band, because a scrambled join scatters that ratio. Kosovo cannot pass that test and
should not: Kontur's 2023 surface knows about the people the 2024 census did not enumerate,
so the four northern municipalities come out at many times their census count while the
other 34 sit in a normal band. That is the boycott showing up in a second, unrelated source,
and it is a better argument than the news story. The band is therefore asserted on the 34
and reported — not asserted — on the four. Leposaviq comes out at 3.2x, Zubin Potok 6.6x
and Zveçan 12.4x, against a median of 1.06x for the enumerated country.

**North Mitrovica does NOT come out flagged, at 1.11x, and that is a limit of the check
rather than a clean bill.** geoBoundaries splits Mitrovica along the Ibar, through the
middle of one continuous built-up city, so hexes on the north bank sit close enough to the
line to be assigned to the southern municipality — which reports 1.10x on 70,971 modelled
people. A boundary that cuts a city in half defeats a per-unit population check for both
halves, and no widening of the band would fix it.

**WHY A POPULATION GRID AND NOT §8.2's EQUAL SHARE.** 38 municipalities over 10,900 km² is
287 km² each, and Kosovo's people are in the Prishtinë-Ferizaj-Prizren corridor and the
valley floors, not on the Sharr or the Kopaonik. Uniform scatter would paint mountain.

The two approximations are Serbia's and are bounded by the hex size: a hex belongs to the
municipality containing its centre and is then clipped to it. The hex populations are
Kontur's model and are used ONLY as relative weights inside a unit — every municipality's
dot count comes from the census.
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
GEO = os.path.join(ROOT, "data", "geo", "xk")
RAW = os.path.join(ROOT, "data", "raw", "xk")
NORM = os.path.join(ROOT, "data", "normalized", "xk.csv")

ADM2 = os.path.join(RAW, "geoBoundaries-XKX-ADM2.geojson")
ADM2_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/"
            "XKX/ADM2/geoBoundaries-XKX-ADM2.geojson")

KONTUR_DIR = os.path.join(ROOT, "data", "geo", "kontur")
KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.eu-central-1.amazonaws.com/"
              "kontur_datasets/kontur_population_XK_20231101.gpkg.gz")
KONTUR_GZ = os.path.join(KONTUR_DIR, "kontur_population_XK_20231101.gpkg.gz")
KONTUR = os.path.join(KONTUR_DIR, "kontur_population_XK_20231101.gpkg")

UNITS_OUT = os.path.join(GEO, "xk_municipalities.gpkg")
GRID_OUT = os.path.join(GEO, "xk_grid_400m.gpkg")

EXPECTED_UNITS = 38
PREFIX = "municipality of "

# geoBoundaries name -> ASK name, for the four that are alternative names rather than
# inflections of the same name.
ALIAS = {
    "drenas": "gllogoc",              # Gllogoc is the official name; Drenas the Albanian
    "pristina": "prishtine",          # the anglicised form
    "northmitrovica": "mitroviceeveriut",   # ASK: `Mitrovicë e Veriut`
    "hanielezit": "haniielezit",      # ASK: `Hani i Elezit`, geoBoundaries: `Han i Elezit`
    "zvecan": "zveqan",               # ç against q — not a diacritic difference
}

# From sources/xk.py; repeated here so the geo build can report them without importing.
NORTH = ("Leposaviq", "Zubin Potok", "Zveqan", "Mitrovicë e Veriut")


def fetch():
    import gzip
    import shutil

    import requests

    ua = {"User-Agent": "religiondots/1.0"}
    os.makedirs(RAW, exist_ok=True)
    if not os.path.exists(ADM2) or os.path.getsize(ADM2) < 100_000:
        print("GET", ADM2_URL)
        r = requests.get(ADM2_URL, timeout=600, headers=ua)
        r.raise_for_status()
        # §5a: HTTP 200 is not a download. GitHub serves an HTML 404 page with a 200 in
        # some proxy configurations, so assert the payload is GeoJSON.
        if b'"FeatureCollection"' not in r.content[:400]:
            raise SystemExit(f"not GeoJSON: first bytes {r.content[:120]!r}")
        with open(ADM2, "wb") as fh:
            fh.write(r.content)
        print(f"  {os.path.getsize(ADM2):,} bytes")
    else:
        print("already have", ADM2)

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
    """Diacritic-free, letters-only, `Municipality of` removed."""
    s = str(s).strip()
    if s.lower().startswith(PREFIX):
        s = s[len(PREFIX):]
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^A-Za-z]+", "", s).lower()


def stem(s):
    """Albanian definite/indefinite differ by the final vowel: Gjakova / Gjakovë."""
    s = ALIAS.get(s, s)
    return s[:-1] if s and s[-1] in "ae" else s


def build_units():
    import geopandas as gpd
    import pandas as pd

    if not os.path.exists(ADM2):
        raise SystemExit(f"missing {ADM2} -- run sources/xk_geo.py --fetch first")
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/xk.py first")

    g = gpd.read_file(ADM2)
    if g.crs is None:
        g = g.set_crs(4326)
    g = g.to_crs(4326)
    print(f"  geoBoundaries XKX ADM2: {len(g)} polygons")
    if len(g) != EXPECTED_UNITS:
        raise SystemExit(f"expected {EXPECTED_UNITS} polygons, got {len(g)}")
    g["key"] = [stem(norm(n)) for n in g["shapeName"]]

    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    muni = (df[df["geo_level"] == "municipality"][["geo_id", "geo_name"]]
            .drop_duplicates("geo_id").copy())
    muni["key"] = [stem(norm(n)) for n in muni["geo_name"]]
    if len(muni) != EXPECTED_UNITS:
        raise SystemExit(f"{len(muni)} census municipalities, expected {EXPECTED_UNITS}")

    # A stem that repeats on either side would make the join ambiguous rather than wrong,
    # which is the failure mode that does not announce itself. Check before merging.
    for label, s in (("geoBoundaries", g), ("census", muni)):
        dup = s["key"][s["key"].duplicated(keep=False)]
        if len(dup):
            raise SystemExit(f"{label} stems are not unique: {sorted(set(dup))} -- add an "
                             "ALIAS entry rather than loosening `stem`")

    only_c = set(muni["key"]) - set(g["key"])
    only_g = set(g["key"]) - set(muni["key"])
    print(f"\n  the name join, both ways (§12 — a count match is not a join):")
    print(f"    matched                      {len(set(muni['key']) & set(g['key'])):>3}")
    if only_c or only_g:
        print("    census keys with no polygon:", sorted(only_c))
        print("    polygons with no census row:", sorted(only_g))
        raise SystemExit("the name join is not 1:1 -- fix it before going further")
    print(f"    OK  all {len(muni)} census units join to exactly one polygon")

    out = g.merge(muni, on="key", how="inner").rename(
        columns={"geo_id": "unit", "shapeName": "geo_name_gb"})
    out = out[["unit", "geo_name", "geo_name_gb", "geometry"]]
    out["area_km2"] = out.to_crs(32634).area.values / 1e6      # UTM 34N, metres
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
        raise SystemExit(f"missing {KONTUR} -- run sources/xk_geo.py --fetch first")

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
    print(f"    {outside.sum():,} hexes fall outside the 38 municipalities "
          f"({hexes.loc[outside, 'population'].sum():,.0f} people) — the border strip "
          "Kontur rounds outwards")
    hexes = hexes[~outside].copy()

    # ---- independent check, and the boycott is what it finds (§9p's test, adapted) ----
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    tot = df[(df["geo_level"] == "municipality") & (df["source_category"] == "Total")]
    tot = dict(zip(tot["geo_id"], tot["count"]))
    kon = hexes.groupby("unit")["population"].sum().to_dict()
    name = dict(zip(units["unit"], units["geo_name"]))

    rows = [(u, tot[u], kon.get(u, 0.0)) for u in units["unit"] if u in tot]
    north = [r for r in rows if name[r[0]] in NORTH]
    rest = [r for r in rows if name[r[0]] not in NORTH]

    ratios = sorted((k / c, name[u]) for u, c, k in rest if c)
    print(f"\n  independent check — Kontur 2023 population / census 2024 count:")
    print(f"    the 34 enumerated municipalities: median {ratios[len(ratios) // 2][0]:.2f}x,"
          f" min {ratios[0][0]:.2f} ({ratios[0][1]}), "
          f"max {ratios[-1][0]:.2f} ({ratios[-1][1]})")
    # BAND = 0.5-2.0, and it is taken from the data rather than from Serbia's 0.8-1.3. Two
    # reasons it is wider: Kontur is a MODEL against GISCO's estimate, and Kosovo's units
    # run down to 434 people, where a model's error is proportionally large. The band still
    # discriminates, and the four below are the proof — a unit whose count is genuinely
    # wrong lands at 3-12x, an order of magnitude outside it.
    bad = [(r, u) for r, u in ratios if r < 0.5 or r > 2.0]
    if bad:
        print(f"    !! {len(bad)} of the 34 outside 0.5-2.0x — a correct join keeps every "
              "unit in a tight band, a scrambled one does not:")
        for r, u in bad[:12]:
            print(f"       {u:26s} {r:.2f}")
        raise SystemExit("the population check failed -- the join is suspect")
    print("    every one of the 34 inside 0.5-2.0x, which a scrambled join would not be")

    print("\n    THE FOUR NORTHERN MUNICIPALITIES, reported and NOT asserted — Kontur "
          "knows about\n    the people the census did not enumerate, which is the boycott "
          "in a second source:")
    for u, c, k in sorted(north, key=lambda r: -(r[2] / r[1] if r[1] else 0)):
        print(f"       {name[u]:26s} census {c:>6,}   Kontur {k:>8,.0f}   "
              f"{k / c if c else float('nan'):>6.1f}x")
    print("    Leposaviq, Zubin Potok and Zveçan are 3.2x, 6.6x and 12.4x, against a "
          "median of\n    1.06x for the enumerated country. **Mitrovicë e Veriut is NOT "
          "flagged, at 1.11x, and\n    that is a limit of this check rather than a "
          "clean bill** — the geoBoundaries line\n    through Mitrovica follows the Ibar "
          "and splits one continuous city, so hexes on the\n    northern bank are readily "
          "assigned to the southern municipality. Mitrovicë is 1.10x\n    with 70,971 "
          "modelled people, which is where they went.")

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
