"""Brunei — the four districts of the 2021 census, and the placement grid.

Writes:
    data/geo/bn/bn_districts.gpkg     the 4 counted units (`units`)
    data/geo/bn/bn_hexes.gpkg         Kontur 400 m hexes with `unit` and `pop` (`place`)
    data/geo/bn/bn_lookup.csv         unit -> census population, Kontur population, area
    data/geo/bn/bn_mukim_lookup.csv   the witness below, per mukim

Usage:
    python sources/bn_geo.py --fetch    geoBoundaries ADM1 + ADM2 (~0.15 MB), Kontur (~0.16 MB)
    python sources/bn_geo.py            rebuild from data/raw/bn/

## THE BOUNDARIES ARE geoBoundaries gbOpen BRN ADM1

Commit 9469f09: four districts, `boundaryYear` 2011, traced from a Wikimedia Commons map (user
Tachymetre), public domain. There is no HDX COD-AB for Brunei (the scout, sources.md
§scout-2026-09-14-asia-oceania). The districts have not changed since 1938, so the vintage is
not the risk; the tracing is. The join is by folded name (`Brunei-Muara` against the census's
`Brunei Muara`), asserted a bijection.

## A WITNESS NEITHER KEY DETERMINES: THE MUKIMS

geoBoundaries ADM2 is 38 mukims traced separately (`boundaryYear` 2006, Wikimedia Commons user
Rarelibra). The census's Table C1 lists 39 mukims under their districts: Gadong was split into
Gadong A and Gadong B after 2006, and the census spells `Bokok` what geoBoundaries spells
`Bunkok`. With those two folds the 38 join one to one, and then the district holding most of each mukim's
area (of the part inside any district) must be the one Table C1 lists it under. Neither the
district name join nor the mukim name join can make that true by construction.

The mukim layer is a membership witness only, never a placement layer. It is poorly traced at
the coast: it leaves out a 42.5 km2 strip of the Kuala Belait and Seria shore holding 29,972
Kontur people, and draws the Brunei River inside mukims the district tracing leaves out. An
area test between the two layers failed for that reason and was dropped (sources/bn.md §5).

## KONTUR IS A WEIGHT INSIDE A DISTRICT

Belait is 2,700 km2 with 63,919 of its 65,531 people in three coastal mukims (Kuala Belait,
Seria, Liang) and 16 people in Kuala Balai; Temburong's people are along the Temburong river.
Hexes join on their centroid; a centroid just offshore is snapped to the nearest district within
SNAP_M. Kontur per mukim is printed against Table C1 as a check on the grid, not used.
"""

import csv
import gzip
import os
import re
import shutil
import sys
import unicodedata

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

RAW = os.path.join(ROOT, "data", "raw", "bn")
GEO = os.path.join(ROOT, "data", "geo", "bn")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur")
NORM = os.path.join(ROOT, "data", "normalized", "bn.csv")

OUT_UNITS = os.path.join(GEO, "bn_districts.gpkg")
OUT_HEXES = os.path.join(GEO, "bn_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "bn_lookup.csv")
OUT_MUKIMS = os.path.join(GEO, "bn_mukim_lookup.csv")

GB = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/BRN/{0}/"
      "geoBoundaries-BRN-{0}.geojson")
GB_ADM1 = os.path.join(RAW, "geoBoundaries-BRN-ADM1.geojson")
GB_ADM2 = os.path.join(RAW, "geoBoundaries-BRN-ADM2.geojson")
KONTUR_GPKG = os.path.join(KONTUR, "kontur_population_BN_20231101.gpkg")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}
DOWNLOADS = {
    GB_ADM1: (GB.format("ADM1"), b"{", 20_000),
    GB_ADM2: (GB.format("ADM2"), b"{", 20_000),
    KONTUR_GPKG + ".gz": ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/"
                          "kontur_datasets/kontur_population_BN_20231101.gpkg.gz",
                          b"\x1f\x8b", 50_000),
}

UNITS = 4
GB_MUKIMS = 38
# census Table C1 name -> geoBoundaries ADM2 name, where they differ
MUKIM_ALIAS = {"Gadong A": "Gadong", "Gadong B": "Gadong", "Bokok": "Bunkok"}
MUKIM_SHARE_MIN = 0.60   # of a mukim's traced area, inside the district C1 names
SNAP_M = 500
UTM = "EPSG:32650"
EQ = "EPSG:6933"

# Census 2021 against Kontur November 2023. DEPS's mid-year estimates put Brunei within a few
# per cent of the census over those two years, so the band is about the grid, not growth.
KONTUR_RATIO = (0.80, 1.30)
KONTUR_UNIT_BAND = (0.60, 1.70)


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.casefold())


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    os.makedirs(KONTUR, exist_ok=True)
    for dst, (url, magic, min_size) in DOWNLOADS.items():
        if os.path.exists(dst) and os.path.getsize(dst) > min_size:
            print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
            continue
        r = requests.get(url, headers=UA, timeout=900)
        r.raise_for_status()
        if not r.content.lstrip().startswith(magic):             # a 200 is not a download
            raise SystemExit(f"{os.path.basename(dst)} starts {r.content[:16]!r}, not {magic!r}")
        with open(dst + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(dst + ".part", dst)
        print(f"  got  {os.path.basename(dst)} ({len(r.content):,} bytes)")


def unpack():
    gz = KONTUR_GPKG + ".gz"
    if not os.path.exists(KONTUR_GPKG):
        if not os.path.exists(gz):
            raise SystemExit(f"missing {gz}; run with --fetch first")
        with gzip.open(gz, "rb") as src, open(KONTUR_GPKG + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(KONTUR_GPKG + ".part", KONTUR_GPKG)
    with open(KONTUR_GPKG, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{KONTUR_GPKG} is not a GeoPackage")


def _join_on_centroids(pts, polys, key):
    """Series of `key` per point: within, then nearest within SNAP_M for the rest."""
    import geopandas as gpd

    j = gpd.sjoin(pts, polys[[key, "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)
    got = j[key].copy()
    out = got.isna()
    snapped = 0
    if out.any():
        near = gpd.sjoin_nearest(pts.loc[out].to_crs(UTM), polys[[key, "geometry"]].to_crs(UTM),
                                 how="left", max_distance=SNAP_M, distance_col="dist_m")
        near = near[~near.index.duplicated(keep="first")]
        s = near[key].dropna()
        got.loc[s.index] = s
        snapped = len(s)
    return got, int(out.sum()), snapped


def main():
    import geopandas as gpd
    import pandas as pd

    import bn
    import geo_checks

    if "--fetch" in sys.argv:
        fetch()
    unpack()
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM}; run sources/bn.py first")
    census = pd.read_csv(NORM, keep_default_na=False, na_values=[""]).groupby(
        "geo_id")["count"].sum()
    if len(census) != UNITS:
        raise SystemExit(f"{len(census)} districts in bn.csv, expected {UNITS}")

    # ---- 1. districts
    d1 = geo_checks.read_layer(GB_ADM1, "geoBoundaries BRN ADM1")
    if len(d1) != UNITS:
        raise SystemExit(f"geoBoundaries BRN ADM1 has {len(d1)} features, expected {UNITS}")
    d1["key"] = d1["shapeName"].map(fold)
    lut = {}
    for u in census.index:
        hits = d1.index[d1["key"] == fold(u)].tolist()
        if len(hits) != 1:
            raise SystemExit(f"district {u!r} matched {len(hits)} polygons: {sorted(d1['shapeName'])}")
        lut[u] = hits[0]
    if len(set(lut.values())) != UNITS:
        raise SystemExit("two census districts matched one polygon")
    units = d1.loc[list(lut.values()), ["geometry"]].copy()
    units["unit"] = list(lut.keys())
    units = units.reset_index(drop=True)
    units["census_pop"] = units["unit"].map(census).astype("int64")
    units["area_km2"] = units.to_crs(EQ).area / 1e6
    print(f"geoBoundaries BRN ADM1: {UNITS}/{UNITS} districts joined by name, crs={d1.crs}, "
          f"{units['area_km2'].sum():,.0f} km2")
    for r in units.itertuples():
        print(f"    {r.unit:<14} {r.area_km2:>7,.0f} km2  {r.census_pop:>8,} people  "
              f"(geoBoundaries `{d1.loc[lut[r.unit], 'shapeName']}`)")

    # ---- 2. the mukim witness
    c1 = bn.read_c1(bn.workbook())
    m_pop, m_district = {}, {}
    for d, muk in c1.items():
        for name, p in muk.items():
            g = MUKIM_ALIAS.get(name, name)
            if g in m_district and m_district[g] != d:
                raise SystemExit(f"mukim {g!r} is under both {m_district[g]} and {d}")
            m_district[g] = d
            m_pop[g] = m_pop.get(g, 0) + p
    m2 = geo_checks.read_layer(GB_ADM2, "geoBoundaries BRN ADM2")
    if len(m2) != GB_MUKIMS or len(m_pop) != GB_MUKIMS:
        raise SystemExit(f"{len(m2)} geoBoundaries mukims and {len(m_pop)} census mukims after "
                         f"folding, expected {GB_MUKIMS} each")
    m2["key"] = m2["shapeName"].map(fold)
    if m2["key"].duplicated().any():
        raise SystemExit(f"repeated mukim names: {sorted(m2.loc[m2['key'].duplicated(), 'shapeName'])}")
    by_key = dict(zip(m2["key"], m2.index))
    missing = sorted(g for g in m_pop if fold(g) not in by_key)
    unused = sorted(set(m2["key"]) - {fold(g) for g in m_pop})
    if missing or unused:
        raise SystemExit(f"mukim join: census without polygon {missing}; polygons unused {unused}")
    m2["mukim"] = m2["key"].map({fold(g): g for g in m_pop})
    m2["district"] = m2["mukim"].map(m_district)
    m2["census_pop"] = m2["mukim"].map(m_pop).astype("int64")
    print(f"\n  mukim join: {GB_MUKIMS}/{GB_MUKIMS} (census C1's 39 with Gadong A and B merged, "
          "Bokok as Bunkok)")

    # By area, not by a representative point: the mukim tracing draws the Brunei River inside
    # Kota Batu, Sungai Kebun and Burong Pingai Ayer, where the district tracing leaves it out,
    # so those three mukims' points landed in no district at all (first run, 2026-09-15), and
    # 58% of Sungai Kebun's traced area is in no district (second run). So the share is of the
    # part of a mukim that lies in some district: this tests which district a mukim is in, and
    # witness 2 below tests how closely the two tracings agree.
    me = m2.to_crs(EQ)
    ue = units.set_index("unit").to_crs(EQ)
    wrong, shares = [], []
    for r in me.itertuples():
        over = {u: r.geometry.intersection(g).area for u, g in ue.geometry.items()}
        covered = sum(over.values())
        best = max(over, key=over.get)
        share = over[best] / covered if covered else 0.0
        shares.append((share, r.mukim, best, 1 - covered / r.geometry.area))
        if best != r.district or share < MUKIM_SHARE_MIN:
            wrong.append(f"{r.mukim} (C1 {r.district}): {100 * share:.1f}% in {best}")
    if wrong:
        raise SystemExit("mukims not mostly inside the district Table C1 puts them in: "
                         + "; ".join(wrong))
    low = sorted(shares)[:4]
    print(f"  witness 1: all {GB_MUKIMS} mukims lie mostly inside the district Table C1 lists "
          "them under; the four lowest shares of the area inside any district: "
          + ", ".join(f"{m} {100 * s:.1f}%" for s, m, _, _ in low))
    outside = sorted(((o, m) for _, m, _, o in shares if o > 0.05), reverse=True)
    print("    traced mukim area in no district (over 5%): "
          + (", ".join(f"{m} {100 * o:.0f}%" for o, m in outside) or "none"))
    # NO AREA TEST AGAINST THE MUKIM TRACING. The first build compared each district with the
    # union of its mukims by IoU and stopped on Brunei Muara at 0.831, but area was the wrong
    # measure: sources/bn.md §5 has the measurement. The mukim tracing is poor at the coast. It
    # leaves out one 42.5 km2 strip of the Kuala Belait and Seria shore holding 29,972 Kontur
    # people, so on the mukim layer Belait reads 0.71x the census and 25,130 people fall in no
    # district, while the district outlines below read 0.96x to 1.01x for the three large
    # districts. The mukim layer is a MEMBERSHIP witness (witness 1 above) and never a
    # placement layer; the people check is the per-district Kontur band in step 3.

    # ---- 3. Kontur
    hexes = geo_checks.read_layer(KONTUR_GPKG, "Kontur BN")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    pts = gpd.GeoDataFrame({"pop": hexes[popcol].to_numpy(dtype=float)},
                           geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs(units.crs)
    unit_of, n_out, n_snap = _join_on_centroids(pts, units, "unit")
    lost_mask = unit_of.isna()
    lost = float(pts.loc[lost_mask, "pop"].sum())
    total_k = float(pts["pop"].sum())
    print(f"\nKontur hexes: {len(hexes):,}, population {total_k:,.0f}; {n_out:,} centroids in no "
          f"district, {n_snap:,} snapped within {SNAP_M} m, {int(lost_mask.sum()):,} dropped "
          f"({lost:,.0f} people, {100 * lost / total_k:.3f}%)")
    if lost / total_k > 0.02:
        raise SystemExit("more than 2% of Kontur falls outside every district; check the border")

    keep = ~lost_mask
    out = gpd.GeoDataFrame({"unit": unit_of[keep].to_numpy(),
                            "pop": pts.loc[keep, "pop"].to_numpy()},
                           geometry=hexes.to_crs(units.crs).geometry[keep.to_numpy()].to_numpy(),
                           crs=units.crs)
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"districts with no populated hex: {missing}")
    tot = float(out["pop"].sum())
    ratio = tot / float(units["census_pop"].sum())
    print(f"  Kontur 2023 {tot:,.0f} against the 2021 census {int(units['census_pop'].sum()):,}: "
          f"ratio {ratio:.3f}")
    if not KONTUR_RATIO[0] <= ratio <= KONTUR_RATIO[1]:
        raise SystemExit(f"ratio outside {KONTUR_RATIO}; check the download")
    units["kontur_pop"] = units["unit"].map(per["sum"]).round().astype("int64")
    units["hexes"] = units["unit"].map(per["size"]).astype("int64")
    print("\n  per district, census 2021 against Kontur 2023:")
    for r in units.sort_values("census_pop", ascending=False).itertuples():
        print(f"    {r.unit:<14}{r.census_pop:>9,}  Kontur {r.kontur_pop:>9,}  "
              f"{r.kontur_pop / r.census_pop:5.2f}x  {r.hexes:>6,} hexes  {r.area_km2:>7,.0f} km2")
    geo_checks.ratio_band(dict(zip(units["unit"], units["census_pop"])),
                          dict(zip(units["unit"], units["kontur_pop"])),
                          *KONTUR_UNIT_BAND, what="district")

    # per mukim, printed only
    mk_of, _, _ = _join_on_centroids(pts, m2.rename(columns={"mukim": "mk"}), "mk")
    mk = pd.DataFrame({"mk": mk_of, "pop": pts["pop"]}).groupby("mk")["pop"].agg(["size", "sum"])
    m2["kontur_pop"] = m2["mukim"].map(mk["sum"]).fillna(0).round().astype("int64")
    m2["hexes"] = m2["mukim"].map(mk["size"]).fillna(0).astype("int64")
    m2["area_km2"] = me.area / 1e6
    print("\n  per mukim (geoBoundaries 2006 lines), census C1 against Kontur 2023; printed, not used:")
    for r in m2.sort_values(["district", "census_pop"], ascending=[True, False]).itertuples():
        kr = r.kontur_pop / r.census_pop if r.census_pop else float("nan")
        flag = "  <--" if r.census_pop >= 1_000 and not 0.5 <= kr <= 2.0 else ""
        print(f"    {r.district:<13}{r.mukim:<20}{r.census_pop:>8,}  Kontur {r.kontur_pop:>8,}  "
              f"{kr:6.2f}x  {r.hexes:>5,} hexes  {r.area_km2:>7,.1f} km2{flag}")

    # ---- write
    os.makedirs(GEO, exist_ok=True)
    cols = ["unit", "census_pop", "kontur_pop", "hexes", "area_km2", "geometry"]
    units[cols].to_file(OUT_UNITS, layer="districts", driver="GPKG")
    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    with open(OUT_LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "census_pop_2021", "kontur_pop_2023", "hexes", "area_km2",
                    "kontur_over_census"])
        for r in units.sort_values("unit").itertuples():
            w.writerow([r.unit, r.census_pop, r.kontur_pop, r.hexes, round(r.area_km2, 1),
                        round(r.kontur_pop / r.census_pop, 3)])
    with open(OUT_MUKIMS, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["district", "mukim_geoboundaries", "census_pop_2021", "kontur_pop_2023",
                    "hexes", "area_km2"])
        for r in m2.sort_values(["district", "mukim"]).itertuples():
            w.writerow([r.district, r.mukim, r.census_pop, r.kontur_pop, r.hexes,
                        round(r.area_km2, 2)])
    print(f"\nwrote {OUT_UNITS}\nwrote {OUT_HEXES} ({len(out):,} hexes)\nwrote {OUT_LOOKUP}\n"
          f"wrote {OUT_MUKIMS}")


if __name__ == "__main__":
    main()
