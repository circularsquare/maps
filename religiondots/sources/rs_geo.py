"""Serbia — the 168 municipalities, and a population grid to place dots inside them.

Writes:
    data/geo/rs/rs_municipalities.gpkg   168 polygons keyed as rs.py keys them  (`units`)
    data/geo/rs/rs_grid_400m.gpkg        Kontur H3 r8 hexes with `unit` and `pop` (`place`)

Usage:
    python sources/rs_geo.py --fetch   # one 4 MB download, the Kontur country extract
    python sources/rs_geo.py           # build both layers

THE BOUNDARIES COST NOTHING. Eurostat's GISCO LAU 2021 file — fetched for North Macedonia
in §9i — carries **169 Serbian polygons**, and the reason is worth stating because it keeps
paying: GISCO's LAU set is not the EU27. It covers candidate countries too, so RS, BG, AL
and CH are all sitting in a file this repo already has.

**THE PLACEMENT LAYER IS KONTUR'S COUNTRY EXTRACT, NOT THE GLOBAL FILE.** Russia uses the
global r6 grid (~36 km² hexes) because the global r8 is a 2.4 GB download and Russia does
not need it; `sources/ke_grid.py` uses Kenya's own extract. Serbia needs the fine one and
the reason is measurable: **at r6 the whole country is 1,991 hexes, and Vračar, Stari grad,
Medijana and Sremski Karlovci are each smaller than a single hex**, so the coarse grid
holds no centre at all in four of the densest municipalities in the country. The extract
`kontur_population_RS_20231101.gpkg.gz` is **4.2 MB**, is H3 **r8** (~0.74 km², about 460 m
across), and gives Serbia 59,823 hexes inside the drawn units — a median of 312 per
municipality. The pattern generalises: a country needing a population surface should take
its own extract rather than the global grid, and the URL is the same one with the ISO code
in it.

**169 GISCO POLYGONS AGAINST 168 CENSUS UNITS, AND THE ODD ONE OUT IS PETROVARADIN.**
GISCO splits Novi Sad into Novi Sad and Petrovaradin — the city municipality across the
Danube — while the census publishes Novi Sad whole. So Petrovaradin's polygon is dissolved
into Novi Sad's, and the join is then exactly 1:1. The direction matters: dissolving a
GISCO polygon into its neighbour loses nothing, whereas splitting a census figure between
the two would be inventing a magnitude (§14.4).

**THE JOIN IS BY NAME, BECAUSE THE CENSUS PUBLISHES NO CODES.** GISCO's `LAU_ID` is the
Serbian municipality code (`70017`) and the workbook carries nothing to join it to, so this
is Romania's and Ghana's situation. Three things make it safe rather than lucky:

  * the two sides disagree only about DIACRITICS and about `đ` vs `dj`, which normalising
    away is lossless here — no two Serbian municipalities differ only by an accent;
  * **`Palilula` is a municipality of Belgrade and also one of Niš**, so bare names are not
    unique. Names that repeat on either side are keyed by their city as well, which is
    generated rather than hand-listed, so a future collision is caught by the same code;
  * **GISCO writes the city as `Belgrade` where the census writes `Beograd`.** One alias,
    below, and it is the only translated place name in the file.

And an independent check that the join is the right one, not merely a complete one: GISCO's
own `POP_2021` against the census total, per unit. They are DIFFERENT QUANTITIES — a 2021
estimate against a 2022 enumeration — so §9i applies and the test is that every unit's
ratio sits in a tight band, not that the two are equal.

WHY A POPULATION GRID AND NOT §8.2's EQUAL SHARE. Serbian opštine average 461 km², and the
population inside one is not spread across it: a Serbian municipality is typically one town
and a scatter of villages in a valley, with the rest forest and mountain. §8.2's usual trick
does not apply because these are historical units, not units engineered to a population
target — Germany's case (§3.9a) at a coarser grain.

Two approximations, both Russia's and both bounded by the hex size: a hex belongs to the
municipality containing its CENTRE, and it is then CLIPPED to that municipality so a dot
cannot land across a border. And one thing that is not an approximation: the hex
populations are Kontur's model and are used ONLY as relative weights inside a unit. Every
municipality's dot count comes from the census and never from this file.
"""

import os
import re
import sys
import unicodedata

# Cap BLAS before numpy is imported below it (see sources/ru_geo.py).
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEO = os.path.join(ROOT, "data", "geo", "rs")
LAU = os.path.join(ROOT, "data", "geo", "lau2021", "shp4326",
                   "LAU_RG_01M_2021_4326.shp")
KONTUR_DIR = os.path.join(ROOT, "data", "geo", "kontur")
KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.eu-central-1.amazonaws.com/"
              "kontur_datasets/kontur_population_RS_20231101.gpkg.gz")
KONTUR_GZ = os.path.join(KONTUR_DIR, "kontur_population_RS_20231101.gpkg.gz")
KONTUR = os.path.join(KONTUR_DIR, "kontur_population_RS_20231101.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "rs.csv")

UNITS_OUT = os.path.join(GEO, "rs_municipalities.gpkg")
GRID_OUT = os.path.join(GEO, "rs_grid_400m.gpkg")

EXPECTED_GISCO = 169
EXPECTED_UNITS = 168

# GISCO's only translated place name in Serbia.
CITY_ALIAS = {"belgrade": "beograd"}
# GISCO splits Novi Sad; the census does not. Dissolve the child into the parent.
MERGE_INTO = {"Petrovaradin": "Novi Sad"}


def fetch():
    """The Kontur country extract. 4.2 MB gzipped, ~25 MB on disk."""
    import gzip
    import shutil

    import requests

    os.makedirs(KONTUR_DIR, exist_ok=True)
    if os.path.exists(KONTUR) and os.path.getsize(KONTUR) > 1_000_000:
        print("already have", KONTUR)
        return
    if not os.path.exists(KONTUR_GZ) or os.path.getsize(KONTUR_GZ) < 1_000_000:
        print("GET", KONTUR_URL)
        r = requests.get(KONTUR_URL, timeout=600,
                         headers={"User-Agent": "religiondots/1.0"})
        r.raise_for_status()
        # §5a: HTTP 200 is not a download. A gzip member starts 1f 8b.
        if r.content[:2] != b"\x1f\x8b":
            raise SystemExit(f"not gzip: first bytes {r.content[:40]!r}")
        with open(KONTUR_GZ, "wb") as fh:
            fh.write(r.content)
        print(f"  {os.path.getsize(KONTUR_GZ):,} bytes")
    with gzip.open(KONTUR_GZ, "rb") as src, open(KONTUR, "wb") as dst:
        shutil.copyfileobj(src, dst)
    print(f"  decompressed to {KONTUR}  {os.path.getsize(KONTUR):,} bytes")


def norm(s):
    """Diacritic-free, letters-only key. `đ` first, because NFKD does not decompose it."""
    if not isinstance(s, str):
        return ""
    s = s.replace("đ", "dj").replace("Đ", "Dj")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^A-Za-z]+", "", s).lower()


def _split(name):
    """`City - Name` -> (city, name); anything else -> (None, name)."""
    if " - " in name:
        city, bare = name.split(" - ", 1)
        return city.strip(), bare.strip()
    return None, name.strip()


def _keys(pairs):
    """(city, bare) pairs -> join keys, qualified by city only where bare names repeat."""
    bares = [norm(b) for _, b in pairs]
    dup = {b for b in bares if bares.count(b) > 1}
    out = []
    for (city, bare), nb in zip(pairs, bares):
        if nb in dup:
            c = norm(city)
            c = CITY_ALIAS.get(c, c)
            out.append(f"{c}|{nb}")
        else:
            out.append(nb)
    return out, dup


def build_units():
    import geopandas as gpd
    import pandas as pd

    if not os.path.exists(LAU):
        raise SystemExit(f"missing {LAU} -- unzip data/geo/lau2021/LAU_RG_01M_2021_4326"
                         ".shp.zip into shp4326/ (it is there for North Macedonia)")
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/rs.py first")

    g = gpd.read_file(LAU)
    rs = g[g["CNTR_CODE"] == "RS"].copy()
    print(f"  GISCO RS polygons: {len(rs)}")
    if len(rs) != EXPECTED_GISCO:
        raise SystemExit(f"expected {EXPECTED_GISCO} GISCO polygons, got {len(rs)}")

    # ---- dissolve Petrovaradin into Novi Sad ---------------------------------------
    rs["LAU_NAME"] = rs["LAU_NAME"].astype(str).str.strip()
    for child, parent in MERGE_INTO.items():
        ci = rs.index[rs["LAU_NAME"] == child]
        pi = rs.index[rs["LAU_NAME"] == parent]
        if len(ci) != 1 or len(pi) != 1:
            raise SystemExit(f"cannot merge {child!r} into {parent!r}: "
                             f"{len(ci)} and {len(pi)} matches")
        ci, pi = ci[0], pi[0]
        merged = rs.loc[[ci, pi]].union_all()
        rs.loc[pi, "geometry"] = merged
        rs.loc[pi, "POP_2021"] = (rs.loc[pi, "POP_2021"] or 0) + (rs.loc[ci, "POP_2021"] or 0)
        rs.loc[pi, "AREA_KM2"] = (rs.loc[pi, "AREA_KM2"] or 0) + (rs.loc[ci, "AREA_KM2"] or 0)
        rs = rs.drop(index=ci)
        print(f"  dissolved {child} into {parent} "
              f"(GISCO splits it, the census does not) -> {len(rs)} polygons")

    # ---- keys ----------------------------------------------------------------------
    gis_pairs = [_split(n) for n in rs["LAU_NAME"]]
    gis_keys, gis_dup = _keys(gis_pairs)
    rs["key"] = gis_keys

    df = pd.read_csv(NORM, dtype={"geo_id": str})
    muni = df[df["geo_level"] == "municipality"][["geo_id", "geo_name"]].drop_duplicates()
    cen_pairs = [_split(x) for x in muni["geo_id"]]
    cen_keys, cen_dup = _keys(cen_pairs)
    muni = muni.assign(key=cen_keys)
    print(f"  census municipality rows: {len(muni)}")
    print(f"  bare names that repeat and are therefore city-qualified: "
          f"{sorted(cen_dup | gis_dup)}")

    if len(muni) != EXPECTED_UNITS:
        raise SystemExit(f"expected {EXPECTED_UNITS} census units, got {len(muni)}")
    for label, s in (("GISCO", rs["key"]), ("census", muni["key"])):
        d = s[s.duplicated()].tolist()
        if d:
            raise SystemExit(f"duplicate join keys on the {label} side: {d}")

    only_c = set(muni["key"]) - set(rs["key"])
    only_g = set(rs["key"]) - set(muni["key"])
    if only_c or only_g:
        print("  census keys with no polygon:", sorted(only_c))
        print("  polygons with no census row:", sorted(only_g))
        raise SystemExit("the name join is not 1:1 -- fix it before going further")
    print(f"  OK  all {len(muni)} census units join to exactly one polygon")

    out = rs.merge(muni, on="key", how="inner")
    out = out.rename(columns={"geo_id": "unit", "LAU_NAME": "gisco_name",
                              "LAU_ID": "lau_id", "POP_2021": "pop_2021",
                              "AREA_KM2": "area_km2"})
    out = out[["unit", "gisco_name", "geo_name", "lau_id", "pop_2021", "area_km2",
               "geometry"]]
    out = out.set_crs(4326, allow_override=True) if out.crs is None else out.to_crs(4326)

    # ---- independent check: GISCO's own population against the census (§9i) ---------
    tot = df[(df["geo_level"] == "municipality") & (df["source_category"] == "Total")]
    tot = dict(zip(tot["geo_id"], tot["count"]))
    rows = [(u, tot[u], p) for u, p in zip(out["unit"], out["pop_2021"])
            if p and p > 0 and u in tot]
    ratios = sorted((p / c, u) for u, c, p in rows)
    print(f"\n  independent check — GISCO POP_2021 / census 2022 total, "
          f"{len(rows)} of {len(out)} units GISCO populates:")
    print(f"    national {sum(p for _, _, p in rows):,.0f} / "
          f"{sum(c for _, c, p in rows):,} = "
          f"{sum(p for _, _, p in rows) / sum(c for _, c, p in rows):.3f}x")
    print(f"    per unit: median {ratios[len(ratios) // 2][0]:.3f}, "
          f"min {ratios[0][0]:.3f} ({ratios[0][1]}), "
          f"max {ratios[-1][0]:.3f} ({ratios[-1][1]})")
    bad = [(r, u) for r, u in ratios if r < 0.8 or r > 1.3]
    if bad:
        print(f"    !! {len(bad)} units outside 0.8-1.3x — a correct join keeps every "
              "unit in a tight band, a scrambled one does not:")
        for r, u in bad[:12]:
            print(f"       {u:32s} {r:.3f}")
        raise SystemExit("the population check failed -- the join is suspect")
    print("    every unit inside 0.8-1.3x, which a scrambled join would not be")

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
        raise SystemExit(f"missing {KONTUR} -- run sources/rs_geo.py --fetch first")

    layers = [l for l in fiona_layers(KONTUR)]
    layer = "population" if "population" in layers else layers[0]
    print(f"  reading Kontur r8 hexes from layer {layer!r}…")
    hexes = gpd.read_file(KONTUR, layer=layer)
    print(f"    {len(hexes):,} hexes, {hexes['population'].sum():,.0f} people")
    hexes = hexes.to_crs(4326)

    centres = gpd.GeoDataFrame(geometry=hexes.geometry.representative_point(), crs=4326)
    hit = gpd.sjoin(centres, units[["unit", "geometry"]], how="left", predicate="within")
    hit = hit[~hit.index.duplicated(keep="first")]
    hexes["unit"] = hit["unit"].to_numpy()

    outside = hexes["unit"].isna()
    print(f"    {outside.sum():,} hexes fall outside the 168 drawn municipalities "
          f"({hexes.loc[outside, 'population'].sum():,.0f} people) — Kosovo, which the "
          "census does not enumerate, and the border strip Kontur rounds outwards")
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

    # A municipality smaller than one hex holds no hex CENTRE and would otherwise vanish
    # from the placement layer entirely — its dots would have nowhere to go. de_grid.py
    # has the same case for 34 German Gemeinden and the same answer: the unit carries its
    # own polygon as a single cell, which is §8.2's equal share over one shape, i.e. the
    # thing the grid replaces everywhere else. At r8 this is rare; at r6 it was four of
    # Belgrade's and Niš's inner municipalities, which is why r6 is not used.
    missing = sorted(set(units["unit"]) - set(hexes["unit"]))
    if missing:
        print(f"    {len(missing)} municipalities hold no hex centre and carry their own "
              f"polygon as one cell: {missing}")
        fill = units[units["unit"].isin(missing)][["unit", "geometry"]].copy()
        fill["pop"] = 1.0
        hexes = gpd.GeoDataFrame(pd.concat([hexes, fill[["unit", "pop", "geometry"]]],
                                           ignore_index=True), crs=4326)
    print(f"    all {units['unit'].nunique()} municipalities are covered")

    hexes.to_file(GRID_OUT, layer="grid3km", driver="GPKG")
    print(f"  wrote {GRID_OUT}  {len(hexes):,} hexes")
    return hexes


def report(units, hexes):
    import pandas as pd

    df = pd.read_csv(NORM, dtype={"geo_id": str})
    tot = df[(df["geo_level"] == "municipality") & (df["source_category"] == "Total")]
    census = pd.Series(dict(zip(tot["geo_id"], tot["count"])))
    modelled = hexes.groupby("unit")["pop"].sum()
    both = pd.DataFrame({"census": census, "kontur": modelled}).dropna()
    both["ratio"] = both["kontur"] / both["census"]
    print(f"\n  Kontur surface vs the 2022 census, {len(both)} municipalities")
    print(f"    totals: census {both['census'].sum():,.0f}  "
          f"kontur {both['kontur'].sum():,.0f}  "
          f"({both['kontur'].sum() / both['census'].sum():.3f}x)")
    print(f"    per-unit ratio: median {both['ratio'].median():.3f}, "
          f"min {both['ratio'].min():.3f} ({both['ratio'].idxmin()}), "
          f"max {both['ratio'].max():.3f} ({both['ratio'].idxmax()})")
    n_hex = hexes.groupby("unit").size()
    print(f"    hexes per municipality: median {n_hex.median():.0f}, "
          f"min {n_hex.min()} ({n_hex.idxmin()}), max {n_hex.max()} ({n_hex.idxmax()})")

    # The ratio does not affect a single count — the weights are relative WITHIN a unit —
    # but a unit Kontur models badly is a unit whose dots are placed on a worse surface,
    # so name them rather than leaving the min/max above to stand for them.
    off = both[(both["ratio"] < 0.6) | (both["ratio"] > 1.5)].sort_values("ratio")
    print(f"    {len(off)} municipalities outside 0.6-1.5x — where Kontur's surface is "
          "least like the census, and so where placement inside the unit is weakest:")
    for unit, r in off.iterrows():
        print(f"       {unit:26s} census {r['census']:>8,.0f}  kontur {r['kontur']:>9,.0f}"
              f"  {r['ratio']:.2f}x  {n_hex.get(unit, 0):>5} hexes")


def main():
    if "--fetch" in sys.argv:
        fetch()
    units = build_units()
    hexes = build_grid(units)
    report(units, hexes)


if __name__ == "__main__":
    main()
