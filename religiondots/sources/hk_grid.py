"""Hong Kong — the placement grid: Kontur 400 m population hexagons, clipped to districts.

Writes data/geo/hk/hk_hexes.gpkg. `countries.py` uses it to weight where a district's dots
land, never to change how many there are. Hong Kong has no separate unit layer, so **these
hexes are the geography** -- Tonga's and China's wiring.

HONG KONG NEEDS THIS MORE THAN ALMOST ANY COUNTRY HERE. Its 18 districts average 62 km2 but
three quarters of the territory is country park, steep hillside and reservoir: Islands
district is 176 km2 of which the people are on a few square kilometres of Tung Chung and
Cheung Chau, and Sai Kung is mostly a peninsula nobody lives on. Weighting by area would put
dots across the Ma On Shan ridge and the Plover Cove catchment. At 400 m against a population
of 7.4 million the grid is far finer than the counting tier, which is where Kontur pays
([[reference_kontur_resolution_floor]]).

THE STRAYS ARE SNAPPED, NOT DROPPED, and Hong Kong is the strongest case for that rule on
this map. The territory is 263 islands and its shoreline is almost entirely reclaimed
seawall, so on a 400 m grid a large share of the population lives in cells whose centroid can
fall just seaward of an administrative outline drawn at a different resolution. That loss is
not random: it is the waterfront, which in Hong Kong is where the people are. Vanuatu's rule
(§9bg) and Tonga's, for the same reason.

THE VINTAGE GAP IS TWO YEARS, counts 2021 and grid 2023, and it moves dots within a district,
never between districts.

Usage:
    python sources/hk_grid.py --fetch    one ~1 MB gz from Kontur
    python sources/hk_grid.py            rebuild from data/raw/hk/
"""

import gzip
import os
import shutil
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "hk")
GEO = os.path.join(ROOT, "data", "geo", "hk")
DISTRICTS = os.path.join(GEO, "hk_districts.gpkg")
OUT = os.path.join(GEO, "hk_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_HK_20231101.gpkg.gz")
GZ_NAME = "kontur_population_HK_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_HK_20231101.gpkg"

EXPECTED_UNITS = 18
CENSUS_POPULATION = 7_413_070

# A hex centroid this far outside a district is a coastline-resolution artefact and is
# snapped to the nearest district; anything further is dropped. Metres, in the Hong Kong
# 1980 Grid, which is the local projected CRS and avoids a UTM zone edge.
SNAP_M = 600.0
SNAP_CRS = "EPSG:2326"
MAX_SPAN_DEG = 2.0


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 50_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz):
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=1800, stream=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(gz + ".part", "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        os.replace(gz + ".part", gz)
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} — run with --fetch first")
    if not os.path.exists(DISTRICTS):
        raise SystemExit(f"missing {DISTRICTS} — run sources/hk_geo.py first")

    units = gpd.read_file(DISTRICTS).to_crs("EPSG:4326")
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{DISTRICTS} has {len(units)} districts, expected {EXPECTED_UNITS}")
    print(f"districts: {len(units)}, crs={units.crs}")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    kontur_total = float(hexes[popcol].sum())
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, population {kontur_total:,.0f}")
    print(f"  against the 2021 census's {CENSUS_POPULATION:,} "
          f"= {kontur_total / CENSUS_POPULATION:.3f}x")

    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=hexes.geometry.centroid,
                           crs=hexes.crs).to_crs("EPSG:4326")
    span = pts.total_bounds[2] - pts.total_bounds[0]
    print(f"  hex centroids span {span:.3f}° of longitude "
          f"(districts {units.total_bounds[2] - units.total_bounds[0]:.3f}°)")
    if span > MAX_SPAN_DEG:
        raise SystemExit(f"the grid spans {span:.1f}°; something is torn "
                         "[[reference_antimeridian]]")

    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    stray = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid falls outside every district: {int(outside.sum()):,} "
          f"({stray:,.0f} people, {100.0 * stray / kontur_total:.3f}%)")
    if outside.any():
        m_units = units[["unit", "geometry"]].to_crs(SNAP_CRS)
        m_pts = pts.loc[outside].to_crs(SNAP_CRS)
        near = gpd.sjoin_nearest(m_pts, m_units, how="left", max_distance=SNAP_M,
                                 distance_col="_d")
        near = near[~near.index.duplicated(keep="first")]
        joined.loc[near.index, "unit"] = near["unit"]
        snapped = joined.loc[outside, "unit"].notna()
        moved = float(pts.loc[outside][snapped.to_numpy()][popcol].sum())
        print(f"     {int(snapped.sum()):,} are within {SNAP_M:,.0f} m of a district and are "
              f"SNAPPED to the nearest\n     ({moved:,.0f} people, "
              f"{100.0 * moved / stray:.1f}% of the strays) — reclaimed waterfront and "
              "outlying\n     islets, which is where Hong Kong's people are, so dropping them "
              "would pull\n     every shore's dots inland.")

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"     {int(outside.sum()):,} cells remain unplaced ({lost:,.0f} people, "
          f"{100.0 * lost / kontur_total:.3f}%) and are dropped;")
    print("     these are placement WEIGHTS and not counts, so nobody leaves the map.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(),
        crs=hexes.crs).to_crs("EPSG:4326")

    have = set(out["unit"])
    missing = sorted(set(units["unit"]) - have)
    if missing:
        raise SystemExit(f"!! districts with no hex at all: {missing} — at 400 m in Hong "
                         "Kong that is a join failure, not a resolution floor")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, driver="GPKG", layer="hexes")
    print(f"\nwrote {OUT}\n  {len(out):,} hexes, {out['pop'].sum():,.0f} people")

    # ---- the correlation, which checks the JOIN and nothing else ---------------------
    #
    # A 2023 modelled surface sharing no lineage with the 2021 census has to agree with it
    # about roughly how many people are in each of 18 districts. That is the one thing that
    # would catch a boundary file for the wrong vintage, or a letter code paired with the
    # wrong polygon.
    #
    # **IT IS NOT A CHECK ON THE OUTPUT, AND THE BAND IS WIDE ON PURPOSE.** Dots per district
    # come from the census; Kontur only decides where inside a district they land. So a
    # district whose Kontur total is 40% high changes nothing about how many dots it gets --
    # only about which hillside they sit on. Demanding agreement here would fail on an honest
    # modelling difference, which is §9i's principle and `cn_geo.py`'s.
    #
    # AND THE DISAGREEMENT HAS A SHAPE, WHICH IS THE REASSURING PART. Kontur is built from
    # building footprints and settlement rasters, and Hong Kong is the hardest place on earth
    # for that: it reads a 40-storey public housing estate as one footprint. So the districts
    # it undercounts are the vertical ones -- Wong Tai Sin 0.60x, Sham Shui Po 0.75x, Kwun
    # Tong 0.85x, all dense high-rise -- and the ones it overcounts are the spread-out ones,
    # North 1.61x and Central and Western 1.41x. A scrambled join has no such pattern; it
    # pairs a large district with a small one and the correlation collapses toward zero.
    # r = 0.87 over 18 units with every ratio inside 0.6-1.7 is a join that holds.
    #
    # The cost is real and is named in note_public: within a district, dots are pulled
    # slightly away from the tower estates and toward the low-rise.
    import csv as _csv

    norm = os.path.join(ROOT, "data", "normalized", "hk.csv")
    if os.path.exists(norm):
        with open(norm, encoding="utf-8") as fh:
            census = {r["geo_id"]: float(r["count"]) for r in _csv.DictReader(fh)
                      if r["source_category"] == "Total"}
        grid = out.groupby("unit")["pop"].sum().to_dict()
        pairs = [(census[u], grid.get(u, 0.0)) for u in sorted(census)]
        n = len(pairs)
        ma = sum(a for a, _ in pairs) / n
        mb = sum(b for _, b in pairs) / n
        num = sum((a - ma) * (b - mb) for a, b in pairs)
        den = (sum((a - ma) ** 2 for a, _ in pairs)
               * sum((b - mb) ** 2 for _, b in pairs)) ** 0.5
        r = num / den
        ratios = {u: grid.get(u, 0.0) / census[u] for u in census}
        print(f"\n  census population vs Kontur, over {n} districts: r = {r:.4f}")
        print(f"  {'district':<8}{'census':>10}{'Kontur':>10}{'ratio':>8}")
        for u in sorted(census):
            print(f"  {u:<8}{census[u]:>10,.0f}{grid.get(u, 0.0):>10,.0f}"
                  f"{ratios[u]:>8.2f}")
        lo, hi = min(ratios.values()), max(ratios.values())
        print(f"  ratio band {lo:.2f}-{hi:.2f}; Kontur undercounts the high-rise districts "
              f"and overcounts the rural ones,")
        print("  which is a modelling difference and not a join failure — it changes where "
              "dots sit inside")
        print("  a district and never how many it gets. See the note above this check.")
        if r < 0.6 or lo < 0.3 or hi > 3.0:
            raise SystemExit(
                f"!! r = {r:.3f}, band {lo:.2f}-{hi:.2f}. That is not a modelling "
                "difference — a district is paired with the wrong polygon.")


if __name__ == "__main__":
    main()
