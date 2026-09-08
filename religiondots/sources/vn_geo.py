"""Vietnam — boundaries and placement grid for the 63 provinces of the 2009 census.

Writes:
    data/geo/vn/vn_provinces.gpkg     the 63 provinces (`units`)
    data/geo/vn/vn_grid_400m.gpkg     Kontur H3 r8 hexes with `unit` and `pop` (`place`)
    data/geo/vn/vn_lookup.csv         code -> name, area, census population, Kontur population

Usage:
    python sources/vn_geo.py --fetch     two downloads, ~17 MB
    python sources/vn_geo.py             rebuild from data/raw/vn/ and data/geo/kontur/

**THE VINTAGE IS THE WHOLE GAME HERE, AND IT EXPIRED FIFTEEN MONTHS AGO.** On 1 July 2025
Vietnam merged its 63 provinces into **34**. spec §8.1 says boundaries must be the vintage the
data was published on, and for once the wrong file is not subtly wrong but a different country:
a 2025-vintage ADM1 has no Hà Nam, no Bạc Liêu, no Ninh Thuận, and its An Giang is An Giang
plus Kiên Giang. The geoBoundaries release below is **pinned to commit 9469f09**, which is the
same commit five other countries here use and which predates the merger. The 63-feature
assertion is what notices if that ever stops being true.

THE JOIN IS TWO PUBLISHED STANDARDS BRIDGED BY HAND, AND THE BRIDGE IS CHECKED THREE WAYS.
The census identifies a province by GSO's administrative code (`89. AN GIANG`); geoBoundaries
identifies it by `shapeISO`, which is ISO 3166-2:VN. The two are different code spaces that
both look numeric and **do not agree on a single province** — GSO's 02 is Hà Giang and ISO's
VN-02 is Lào Cai — so a numeric join would produce 63 confident, silent, wrong assignments.
`CODE_TO_ISO` below is the bridge, and nothing about it is taken on trust:

  1. **62 of the 63 are re-derived by folded name on every run**, from the census's own
     province names against geoBoundaries' `shapeName`, and must agree with the table.
  2. **The 63rd is Ho Chi Minh City**, which the census calls `Tp Hồ Chí Minh` and the
     boundary file romanises as `Ho Chi Minh`. It is the only unmatched census province and
     `VN-SG` is the only unused ISO code, so the residual pairing is forced rather than
     chosen — and that is asserted too.
  3. **Kontur population against the census population, per province.** The bridge decides
     which polygon is which province; it does not decide how many people a modelled surface
     puts inside one. A scrambled bridge pairs Ho Chi Minh City's 7.2M with Bắc Kạn's 294k
     and the ratios scatter over orders of magnitude. §9i's North Macedonia check.

THE BOUNDARY FILE HAS 64 FEATURES AND 63 ISO CODES, AND THE DUPLICATE IS THE FIX RATHER THAN
THE BUG. geoBoundaries carries **Côn Đảo** — the offshore islands, which are a *district* of
Bà Rịa–Vũng Tàu — as its own polygon, and correctly gives it the parent's `VN-43`. So
dissolving on `shapeISO` reassembles the province, and a feature-count check alone would have
called this an off-by-one and gone looking for a 64th province that does not exist.

AND THE CENSUS SPELLS ITS OWN PROVINCES TWO WAYS. Vietnamese admits two tone-mark placements
on `oa`/`oe`/`uy`, and GSO uses both: Biểu 1 writes `Hoà Bình`, `Thanh Hoá` and `Khánh Hoà`
where geoBoundaries writes `Hòa Bình`, `Thanh Hóa` and `Khánh Hóa`. Diacritic folding removes
the difference; an exact-string join fails on exactly three provinces and looks like a vintage
problem.

PLACEMENT. Vietnam is 63 provinces for 86M people — 1.36M each, coarser than Kenya's county —
and the population is in two deltas with a long thin middle. An equal share per polygon would
put as many dots in the Central Highlands forest as in the Red River Delta. Kontur's 400 m H3
grid is the same answer as Kenya's, Ethiopia's and Guyana's, for the same reason.
"""

import csv
import os
import re
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "vn")
OUT_DIR = os.path.join(ROOT, "data", "geo", "vn")
KONTUR_DIR = os.path.join(ROOT, "data", "geo", "kontur")

UNITS_OUT = os.path.join(OUT_DIR, "vn_provinces.gpkg")
GRID_OUT = os.path.join(OUT_DIR, "vn_grid_400m.gpkg")
LOOKUP = os.path.join(OUT_DIR, "vn_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "vn.csv")

# Pinned to a pre-merger commit -- see the module docstring.
ADM1_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/"
            "VNM/ADM1/geoBoundaries-VNM-ADM1.geojson")
ADM1 = os.path.join(RAW, "geoBoundaries-VNM-ADM1.geojson")

KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.eu-central-1.amazonaws.com/"
              "kontur_datasets/kontur_population_VN_20231101.gpkg.gz")
KONTUR_GZ = os.path.join(KONTUR_DIR, "kontur_population_VN_20231101.gpkg.gz")
KONTUR = os.path.join(KONTUR_DIR, "kontur_population_VN_20231101.gpkg")

PROVINCES = 63
FEATURES = 64          # the 63 provinces plus Côn Đảo, which shares VN-43

# GSO administrative code -> ISO 3166-2:VN. Derived by folded name, then frozen, then
# re-derived and compared on every run (check 1 in the docstring). The two spaces overlap
# numerically and agree nowhere, so this table is load-bearing.
CODE_TO_ISO = {
    "01": "VN-HN", "02": "VN-03", "04": "VN-04", "06": "VN-53", "08": "VN-07",
    "10": "VN-02", "11": "VN-71", "12": "VN-01", "14": "VN-05", "15": "VN-06",
    "17": "VN-14", "19": "VN-69", "20": "VN-09", "22": "VN-13", "24": "VN-54",
    "25": "VN-68", "26": "VN-70", "27": "VN-56", "30": "VN-61", "31": "VN-HP",
    "33": "VN-66", "34": "VN-20", "35": "VN-63", "36": "VN-67", "37": "VN-18",
    "38": "VN-21", "40": "VN-22", "42": "VN-23", "44": "VN-24", "45": "VN-25",
    "46": "VN-26", "48": "VN-DN", "49": "VN-27", "51": "VN-29", "52": "VN-31",
    "54": "VN-32", "56": "VN-34", "58": "VN-36", "60": "VN-40", "62": "VN-28",
    "64": "VN-30", "66": "VN-33", "67": "VN-72", "68": "VN-35", "70": "VN-58",
    "72": "VN-37", "74": "VN-57", "75": "VN-39", "77": "VN-43", "79": "VN-SG",
    "80": "VN-41", "82": "VN-46", "83": "VN-50", "84": "VN-51", "86": "VN-49",
    "87": "VN-45", "89": "VN-44", "91": "VN-47", "92": "VN-CT", "93": "VN-73",
    "94": "VN-52", "95": "VN-55", "96": "VN-59",
}

# The one province whose name does not fold to the boundary file's. Asserted, not assumed.
ROMANISED = {"79": "VN-SG"}

# A Kontur/census ratio outside this band on any province means the polygons are not the
# provinces the counts think they are. Kontur is a 2023 surface against a 2009 census and
# Vietnam grew 12% over that period with the growth heavily concentrated in Ho Chi Minh City,
# Bình Dương and Hà Nội, so the band is wide on the upper side and any pair-swap still breaks
# it -- the smallest province is 294k and the largest 7.2M.
RATIO_LO, RATIO_HI = 0.60, 2.5


def fold(s):
    """Vietnamese name -> comparison key: diacritics stripped, đ folded, letters only."""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"[^a-z]+", " ", s.lower()).strip()


def fetch():
    import gzip
    import shutil

    import requests

    os.makedirs(RAW, exist_ok=True)
    os.makedirs(KONTUR_DIR, exist_ok=True)

    if not os.path.exists(ADM1) or os.path.getsize(ADM1) < 300_000:
        print("GET", ADM1_URL)
        r = requests.get(ADM1_URL, timeout=300, headers={"User-Agent": "religiondots/1.0"})
        r.raise_for_status()
        # §5a. A GeoJSON FeatureCollection, not an HTML error page.
        if b'"FeatureCollection"' not in r.content[:400]:
            raise SystemExit(f"not GeoJSON -- starts {r.content[:80]!r}")
        with open(ADM1, "wb") as fh:
            fh.write(r.content)
        print(f"  {os.path.getsize(ADM1):,} bytes")

    if os.path.exists(KONTUR) and os.path.getsize(KONTUR) > 10_000_000:
        print("already have", KONTUR)
        return
    if not os.path.exists(KONTUR_GZ) or os.path.getsize(KONTUR_GZ) < 1_000_000:
        print("GET", KONTUR_URL)
        r = requests.get(KONTUR_URL, timeout=900, headers={"User-Agent": "religiondots/1.0"})
        r.raise_for_status()
        if r.content[:2] != b"\x1f\x8b":
            raise SystemExit(f"not gzip: first bytes {r.content[:40]!r}")
        with open(KONTUR_GZ, "wb") as fh:
            fh.write(r.content)
        print(f"  {os.path.getsize(KONTUR_GZ):,} bytes")
    with gzip.open(KONTUR_GZ, "rb") as src, open(KONTUR, "wb") as dst:
        shutil.copyfileobj(src, dst)
    print(f"  decompressed to {KONTUR}  {os.path.getsize(KONTUR):,} bytes")


def census():
    """code -> (name, population), from the normalised file. §12's Chile rule: the names
    come from the statistical source, not from the boundary file."""
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run: python sources/vn.py")
    name, pop = {}, {}
    with open(NORM, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["geo_level"] != "province":
                continue
            name[r["geo_id"]] = r["geo_name"]
            if r["source_category"] == "Dân số - Population":
                pop[r["geo_id"]] = int(r["count"])
    if len(name) != PROVINCES or len(pop) != PROVINCES:
        raise SystemExit(f"{len(name)} provinces and {len(pop)} population rows in {NORM}, "
                         f"expected {PROVINCES} of each")
    return name, pop


def build_units(name):
    import geopandas as gpd

    if not os.path.exists(ADM1):
        raise SystemExit(f"missing {ADM1} -- run: python sources/vn_geo.py --fetch")

    gdf = gpd.read_file(ADM1)
    if len(gdf) != FEATURES:
        raise SystemExit(
            f"{len(gdf)} features in {ADM1}, expected {FEATURES} (63 provinces plus Côn "
            "Đảo). If this is 34, the pinned commit has moved past Vietnam's 2025 provincial "
            "merger and the file is a different country from the one the census counted "
            "(spec §8.1).")
    if "shapeISO" not in gdf.columns:
        raise SystemExit("no shapeISO column -- geoBoundaries has changed its schema and the "
                         "ISO join is gone; re-derive the bridge before trusting anything")

    iso = sorted(set(gdf["shapeISO"]))
    if len(iso) != PROVINCES:
        raise SystemExit(f"{len(iso)} distinct shapeISO values, expected {PROVINCES}")
    if sorted(CODE_TO_ISO.values()) != iso:
        raise SystemExit("the bridge's ISO codes are not the file's:\n"
                         f"  only in bridge: {sorted(set(CODE_TO_ISO.values()) - set(iso))}\n"
                         f"  only in file  : {sorted(set(iso) - set(CODE_TO_ISO.values()))}")

    # ---- check 1: re-derive the bridge by folded name -------------------------------
    by_fold = {}
    for _, r in gdf.iterrows():
        by_fold.setdefault(fold(r["shapeName"]), set()).add(r["shapeISO"])
    derived, unmatched = {}, []
    for code, nm in name.items():
        hits = by_fold.get(fold(nm), set())
        if len(hits) == 1:
            derived[code] = next(iter(hits))
        else:
            unmatched.append(code)
    wrong = {c: (derived[c], CODE_TO_ISO[c]) for c in derived if derived[c] != CODE_TO_ISO[c]}
    if wrong:
        raise SystemExit("the name-derived bridge disagrees with CODE_TO_ISO:\n  " +
                         "\n  ".join(f"{c} {name[c]}: name says {a}, table says {b}"
                                     for c, (a, b) in wrong.items()))
    if sorted(unmatched) != sorted(ROMANISED):
        raise SystemExit(f"provinces unmatched by name: {sorted(unmatched)} "
                         f"({[name[c] for c in sorted(unmatched)]}); expected exactly "
                         f"{sorted(ROMANISED)}")
    # ---- check 2: the residual pairing is forced, not chosen -------------------------
    left = set(iso) - {CODE_TO_ISO[c] for c in derived}
    if left != set(ROMANISED.values()):
        raise SystemExit(f"after the name join {sorted(left)} is unclaimed; expected "
                         f"{sorted(ROMANISED.values())}")
    print(f"  bridge verified: {len(derived)}/{PROVINCES} by folded name, "
          f"{len(ROMANISED)} forced by elimination "
          f"({', '.join(f'{name[c]} = {i}' for c, i in ROMANISED.items())})")

    gb_name = {r["shapeISO"]: " ".join(str(r["shapeName"]).split())
               for _, r in gdf.iterrows() if fold(r["shapeName"]) != "con dao"}
    spell = [(nm, gb_name[CODE_TO_ISO[code]]) for code, nm in sorted(name.items())
             if gb_name.get(CODE_TO_ISO[code], nm) != nm]
    if spell:
        print(f"    {len(spell)} province name(s) spelled differently in the two files; the "
              "census spelling is used (§12's Chile rule):")
        for nm, other in spell:
            print(f"      geoBoundaries {other!r} -> census {nm!r}")

    # ---- dissolve Côn Đảo back into its parent --------------------------------------
    iso_to_code = {v: k for k, v in CODE_TO_ISO.items()}
    gdf = gdf.to_crs(4326)
    gdf["unit"] = [iso_to_code[c] for c in gdf["shapeISO"]]
    merged = gdf.dissolve(by="unit", as_index=False)[["unit", "geometry"]]
    if len(merged) != PROVINCES:
        raise SystemExit(f"dissolve gave {len(merged)} provinces, expected {PROVINCES}")
    print(f"  dissolved {FEATURES} features to {len(merged)} provinces "
          "(Côn Đảo rejoins Bà Rịa–Vũng Tàu, which shares its ISO code)")

    merged["name"] = merged["unit"].map(name)
    merged["iso"] = merged["unit"].map(CODE_TO_ISO)
    merged["area_km2"] = merged.to_crs(6933).area / 1e6
    merged = merged.sort_values("unit").reset_index(drop=True)
    return merged[["unit", "name", "iso", "area_km2", "geometry"]]


def build_grid(units):
    import fiona
    import geopandas as gpd
    import numpy as np
    import shapely

    if not os.path.exists(KONTUR):
        raise SystemExit(f"missing {KONTUR} -- run: python sources/vn_geo.py --fetch")

    layers = list(fiona.listlayers(KONTUR))
    layer = "population" if "population" in layers else layers[0]
    print(f"  reading Kontur r8 hexes from layer {layer!r}…")
    hexes = gpd.read_file(KONTUR, layer=layer)
    if len(hexes) == 0:
        raise SystemExit("Kontur read returned ZERO features -- §12's pyogrio/fiona trap; "
                         "retry with engine='fiona'")
    print(f"    {len(hexes):,} hexes, {hexes['population'].sum():,.0f} people")
    hexes = hexes.to_crs(4326)

    # Join on hex CENTRES so no hex is split between two provinces.
    centres = gpd.GeoDataFrame(geometry=hexes.geometry.representative_point(), crs=4326)
    hit = gpd.sjoin(centres, units[["unit", "geometry"]], how="left", predicate="within")
    hit = hit[~hit.index.duplicated(keep="first")]
    hexes["unit"] = hit["unit"].to_numpy()

    outside = hexes["unit"].isna()
    print(f"    {outside.sum():,} hexes ({100.0 * outside.mean():.2f}%) fall outside every "
          f"province, {hexes.loc[outside, 'population'].sum():,.0f} people — the coastal "
          "strip Kontur rounds outwards, and the islands geoBoundaries does not carry")
    hexes = hexes[~outside].copy()

    print(f"  clipping {len(hexes):,} hexes to their province…")
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
        raise SystemExit(f"provinces with no hex centre: {missing}. At r8 over provinces "
                         "this large that is impossible; the join is wrong, not the grid.")
    return hexes


def check(units, hexes, pop):
    import pandas as pd

    ok = True
    k = hexes.groupby("unit")["pop"].sum()
    rows = [(u["unit"], u["name"], u["area_km2"], pop[u["unit"]],
             float(k.get(u["unit"], 0.0)),
             float(k.get(u["unit"], 0.0)) / pop[u["unit"]])
            for _, u in units.iterrows()]
    df = pd.DataFrame(rows, columns=["unit", "name", "area_km2", "census", "kontur", "ratio"])
    df = df.sort_values("ratio")

    print(f"\n  {'code':>4}  {'province':<20} {'area km2':>9} {'census':>10} {'kontur':>10} "
          f"{'ratio':>6}")
    show = pd.concat([df.head(5), df.tail(5)])
    for _, r in show.iterrows():
        flag = "" if RATIO_LO <= r["ratio"] <= RATIO_HI else "   <-- OUT OF BAND"
        print(f"  {r['unit']:>4}  {r['name']:<20} {r['area_km2']:>9,.0f} {r['census']:>10,} "
              f"{r['kontur']:>10,.0f} {r['ratio']:>6.2f}{flag}")
    print(f"        … {len(df) - 10} provinces between them …")

    bad = df[(df["ratio"] < RATIO_LO) | (df["ratio"] > RATIO_HI)]
    ok &= bad.empty
    print(f"\n  {'OK ' if bad.empty else 'BAD'} every province's Kontur/census ratio is "
          f"inside [{RATIO_LO}, {RATIO_HI}] ({len(bad)} outside)")
    for _, r in bad.iterrows():
        print(f"        {r['unit']} {r['name']}: {r['ratio']:.2f}x")

    tot = df["kontur"].sum() / df["census"].sum()
    print(f"      national ratio {tot:.3f} — Kontur models {df['kontur'].sum():,.0f} against "
          f"a 2009 census {df['census'].sum():,}; Vietnam grew 12% by 2019 and Kontur is a "
          "2023 surface, so a ratio above 1 is expected")

    if not ok:
        raise SystemExit("boundary check FAILED")
    return df


def main():
    if "--fetch" in sys.argv:
        fetch()
    os.makedirs(OUT_DIR, exist_ok=True)

    name, pop = census()
    units = build_units(name)
    hexes = build_grid(units)
    df = check(units, hexes, pop)

    units.to_file(UNITS_OUT, driver="GPKG", layer="provinces")
    print(f"\nwrote {UNITS_OUT} ({len(units)} provinces)")
    hexes.to_file(GRID_OUT, driver="GPKG", layer="grid")
    print(f"wrote {GRID_OUT} ({len(hexes):,} hexes)")
    df.to_csv(LOOKUP, index=False)
    print(f"wrote {LOOKUP}")


if __name__ == "__main__":
    main()
