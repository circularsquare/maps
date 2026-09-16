"""Tokelau - the three atolls, and the placement grid.

Writes:
    data/geo/tk/tk_atolls.gpkg   the 3 units (OSM atoll areas), with census and Kontur populations
    data/geo/tk/tk_hexes.gpkg    Kontur 400 m hexes with `unit` and `pop` (`place`)
    data/geo/tk/tk_lookup.csv    unit -> present 2016, de jure 2016, Kontur 2023, hexes

Usage:
    python sources/tk_geo.py --fetch    Kontur population TK (4 KB) and Kontur Boundaries TK (26 KB)
    python sources/tk_geo.py            rebuild from the downloads

## THE UNITS ARE OSM'S THREE ATOLL AREAS

There is no COD-AB for Tokelau (HDX `cod-ps-tkl` says so) and geoBoundaries has ADM0 only.
Kontur Boundaries TK 2023-06-28 (OpenStreetMap, ODbL) carries Tokelau at admin level 2 and
Atafu, Fakaofo and Nukunonu at admin level 8. Each atoll feature is a sea area of 2,000 to 2,700
km2 around its atoll, so it only decides which atoll a hex belongs to; the atolls are 60 to 100
km apart and nothing sits near a boundary.

## A WITNESS THE NAME TAG CANNOT MAKE TRUE: THE PROFILE REPORT'S DISTANCES

The census profile report (printed p.10) says "Nukunonu lies 64 kilometres north-west of
Fakaofo, and Atafu lies 92 kilometres north-west of Nukunonu". The populated hexes of the
feature OSM calls Nukunonu must lie north-west of those in the one it calls Fakaofo at about that
distance, and the same for Atafu. The printed figures are between atolls and the hexes are on the
villages, so the band is wide; a swapped name fails it by direction.

## KONTUR IS THE WEIGHT INSIDE AN ATOLL

17 populated hexes, 5 or 6 per atoll, on the villages (one each on Atafu and Nukunonu, two on
Fakaofo, printed p.10). That is under spec §8.2e's grid floor, and the grid is kept anyway: the
alternative, equal shares over the hexes or the atoll area, puts people on 1-person hexes or in
the lagoon (sources/geo_checks.csv, `grid_floor,tk`). Kontur 2023 holds 1,837 people against the
1,197 present in 2016 and 1,499 de jure; the 2019 population count was higher than 2016 (the scout
read 1,647), so a ratio above one is expected.
"""

import csv
import gzip
import math
import os
import shutil
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from tk import ATOLLS, DEJURE, PRESENT, T58_TOTAL   # noqa: E402

RAW = os.path.join(ROOT, "data", "raw", "tk")
GEO = os.path.join(ROOT, "data", "geo", "tk")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur")

OUT_UNITS = os.path.join(GEO, "tk_atolls.gpkg")
OUT_HEXES = os.path.join(GEO, "tk_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "tk_lookup.csv")

KONTUR_GPKG = os.path.join(KONTUR, "kontur_population_TK_20231101.gpkg")
BOUNDS_GPKG = os.path.join(RAW, "kontur_boundaries_TK_20230628.gpkg")
S3 = "https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
DOWNLOADS = {
    KONTUR_GPKG + ".gz": (S3 + "kontur_population_TK_20231101.gpkg.gz", 4_360,
                          "SNOBIGKEAJP5Y7R2A7TEYRGCX2YS6CIO"),
    BOUNDS_GPKG + ".gz": (S3 + "kontur_boundaries_TK_20230628.gpkg.gz", 25_852,
                          "C6CRWPADY2YQBSM5K72GTZ4JIINR5DZL"),
}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

HEXES = 17
KONTUR_TOTAL = 1_837
UTM = "EPSG:32702"
EQ = "EPSG:6933"
# (from, to, printed km): profile report printed p.10, both "north-west"
PRINTED_KM = [("Fakaofo", "Nukunonu", 64), ("Nukunonu", "Atafu", 92)]
DIST_BAND = (0.80, 1.25)
KONTUR_RATIO = (1.0, 2.0)          # Kontur 2023 over the residents present in 2016
KONTUR_UNIT_BAND = (1.0, 2.2)      # the same, per atoll


def fetch():
    import requests
    from fetch_checks import digest

    os.makedirs(RAW, exist_ok=True)
    os.makedirs(KONTUR, exist_ok=True)
    for dst, (url, size, dig) in DOWNLOADS.items():
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            print(f"  have {os.path.basename(dst)} ({size:,} bytes)")
            continue
        r = requests.get(url, headers=UA, timeout=600)
        r.raise_for_status()
        body = r.content
        if not body.startswith(b"\x1f\x8b") or len(body) != size or digest(body) != dig:
            raise SystemExit(f"{os.path.basename(dst)}: {len(body):,} bytes, digest {digest(body)}, "
                             f"starts {body[:4]!r}; pinned {size:,} and {dig}")
        with open(dst + ".part", "wb") as fh:
            fh.write(body)
        os.replace(dst + ".part", dst)
        print(f"  got  {os.path.basename(dst)} ({len(body):,} bytes)")


def unpack(gpkg):
    gz = gpkg + ".gz"
    if not os.path.exists(gpkg):
        if not os.path.exists(gz):
            raise SystemExit(f"missing {gz}; run with --fetch first")
        with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")


def main():
    import geopandas as gpd
    import numpy as np

    import geo_checks

    if "--fetch" in sys.argv:
        fetch()
    unpack(KONTUR_GPKG)
    unpack(BOUNDS_GPKG)

    # ---- 1. units
    b = geo_checks.read_layer(BOUNDS_GPKG, "Kontur Boundaries TK").to_crs("EPSG:4326")
    atolls = b[b["osm_admin_level"].astype(str) == "8"].copy()
    names = sorted(atolls["name"].astype(str))
    if names != sorted(ATOLLS):
        raise SystemExit(f"admin level 8 features {names}, expected {sorted(ATOLLS)}")
    atolls = atolls.rename(columns={"name": "unit"})[["unit", "geometry"]].reset_index(drop=True)
    geoms = list(atolls.geometry)
    touching = [(atolls.unit[i], atolls.unit[j]) for i in range(3) for j in range(i + 1, 3)
                if geoms[i].intersection(geoms[j]).area > 0]
    if touching:
        raise SystemExit(f"atoll areas overlap: {touching}")
    atolls["area_km2"] = atolls.to_crs(EQ).area / 1e6
    print("Kontur Boundaries TK 2023-06-28 (OSM): 3/3 atolls at admin level 8, no overlap; "
          + ", ".join(f"{r.unit} {r.area_km2:,.0f} km2 of sea and reef" for r in atolls.itertuples()))

    # ---- 2. Kontur hexes into atolls
    hexes = geo_checks.read_layer(KONTUR_GPKG, "Kontur TK")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    total = float(hexes[popcol].sum())
    if len(hexes) != HEXES or round(total) != KONTUR_TOTAL:
        raise SystemExit(f"Kontur TK has {len(hexes)} hexes and {total:,.0f} people, expected "
                         f"{HEXES} and {KONTUR_TOTAL:,}")
    cent = hexes.geometry.centroid                      # EPSG:3857, a projected CRS
    pts = gpd.GeoDataFrame({"pop": hexes[popcol].to_numpy(dtype=float)}, geometry=cent,
                           crs=hexes.crs).to_crs("EPSG:4326")
    j = gpd.sjoin(pts, atolls[["unit", "geometry"]], how="left", predicate="within")
    if j.index.duplicated().any() or j["unit"].isna().any():
        raise SystemExit(f"hexes in no atoll or in two: {j[j['unit'].isna()].index.tolist()}")
    pts["unit"] = j["unit"].to_numpy()

    per = pts.groupby("unit")["pop"].agg(["size", "sum"])
    atolls["present_2016"] = atolls["unit"].map(dict(zip(ATOLLS, T58_TOTAL[2016])))
    atolls["dejure_2016"] = atolls["unit"].map(dict(zip(ATOLLS, DEJURE[2016][:3])))
    atolls["kontur_pop"] = atolls["unit"].map(per["sum"]).round().astype("int64")
    atolls["hexes"] = atolls["unit"].map(per["size"]).astype("int64")

    # ---- 3. the witness: where OSM's named atolls sit against the profile report's distances
    utm = pts.to_crs(UTM)
    where = {}
    for u, g in utm.groupby("unit"):
        w = g["pop"].to_numpy()
        where[u] = (float(np.average(g.geometry.x, weights=w)), float(np.average(g.geometry.y, weights=w)))
    print("\n  witness, profile report printed p.10 (village centres, population-weighted):")
    bad = []
    for a, c, km in PRINTED_KM:
        dx = where[c][0] - where[a][0]
        dy = where[c][1] - where[a][1]
        d = math.hypot(dx, dy) / 1000
        bearing = (math.degrees(math.atan2(dx, dy)) + 360) % 360
        good = DIST_BAND[0] <= d / km <= DIST_BAND[1] and 270 <= bearing <= 360
        print(f"    {c} from {a}: {d:5.1f} km at {bearing:5.1f} deg, printed {km} km north-west "
              f"({d / km:4.2f}x) {'OK' if good else 'BAD'}")
        if not good:
            bad.append(f"{c} from {a}")
    if bad:
        raise SystemExit(f"OSM's atoll names disagree with the printed geography: {bad}")

    # ---- 4. Kontur against the census
    ratio = atolls["kontur_pop"].sum() / PRESENT[2016]
    print(f"\n  Kontur 2023 {atolls['kontur_pop'].sum():,} against {PRESENT[2016]:,} present in 2016: "
          f"ratio {ratio:.3f}; against {sum(DEJURE[2016][:3]):,} de jure on the atolls "
          f"{atolls['kontur_pop'].sum() / sum(DEJURE[2016][:3]):.3f}")
    if not KONTUR_RATIO[0] <= ratio <= KONTUR_RATIO[1]:
        raise SystemExit(f"ratio outside {KONTUR_RATIO}; check the download")
    for r in atolls.itertuples():
        print(f"    {r.unit:<9} present {r.present_2016:>4}  de jure {r.dejure_2016:>4}  Kontur "
              f"{r.kontur_pop:>4} ({r.kontur_pop / r.present_2016:4.2f}x present)  {r.hexes} hexes")
    print(f"  median hexes per atoll {atolls['hexes'].median():.0f}")
    geo_checks.ratio_band(dict(zip(atolls["unit"], atolls["present_2016"])),
                          dict(zip(atolls["unit"], atolls["kontur_pop"])),
                          *KONTUR_UNIT_BAND, what="atoll")

    # ---- write
    os.makedirs(GEO, exist_ok=True)
    atolls[["unit", "present_2016", "dejure_2016", "kontur_pop", "hexes", "area_km2", "geometry"]].to_file(
        OUT_UNITS, layer="atolls", driver="GPKG")
    out = gpd.GeoDataFrame({"unit": pts["unit"].to_numpy(), "pop": pts["pop"].to_numpy()},
                           geometry=hexes.to_crs("EPSG:4326").geometry.to_numpy(), crs="EPSG:4326")
    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    with open(OUT_LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "present_2016", "dejure_2016", "kontur_2023", "hexes", "area_km2"])
        for r in atolls.sort_values("unit").itertuples():
            w.writerow([r.unit, r.present_2016, r.dejure_2016, r.kontur_pop, r.hexes, round(r.area_km2, 1)])
    print(f"\nwrote {OUT_UNITS}\nwrote {OUT_HEXES} ({len(out)} hexes)\nwrote {OUT_LOOKUP}")


if __name__ == "__main__":
    main()
