"""Japan: the placement layer, Kontur 400 m hexes keyed to the 2020 census's 1,896 municipalities.

    python sources/jp_geo.py      -> data/geo/jp/jp_hexes.gpkg (+ jp_units.gpkg, the polygons)

UNITS. The census table (sources/jp_census.py, table 44-1) prints every municipality with its
5-digit JIS code; designated cities (seireishi) are printed both whole and by ward, and the units
here are the wards (level 0 rows) plus every other city, town and village (levels 2 and 3). Tokyo's
23 special wards are wards like any other. Polygons: MLIT National Land Numerical Information N03
(administrative boundaries, 2021-01-01) as simplified to 1% by SmartNews SMRI's japan-topography
(github.com/smartnews-smri/japan-topography, data/municipality/geojson/s0010), which keeps N03's
`N03_007` JIS code, so the join is on the code, not the name. Asserted both ways: every census unit
has a polygon, and the only polygons with no census unit are the six Northern Territories villages
(01695-01700, under Russian administration, which the census does not count) and four
`所属未定地` (land assigned to no municipality, no code). religiondots' Japan layer is by
prefecture only (47 units) so it is not reused; the Kontur extract is religiondots' (copied).

PLACEMENT is by hex centroid (Kontur's own CRS, then reprojected), as sources/_grid.py. The 1%
simplification cuts coastlines and drops small islets, so a hex whose centroid falls outside every
polygon is SNAPPED to the nearest municipality within MAX_SNAP_KM rather than dropped
([[reference_archipelago_grid_snap]], as religiondots' sources/jp_grid.py); the rest are printed.

CHECKS: units with no populated hex (printed; the build stops if they hold over 5,000 people),
Kontur against the census per unit normalised by the national ratio, and the log
correlation against 500 shuffled joins.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import math
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402

from _grid import kontur_path  # noqa: E402

RAW = ROOT / "data" / "raw" / "jp"
N03 = RAW / "N03-21_210101_s0010.json"
OUT = ROOT / "data" / "geo" / "jp" / "jp_hexes.gpkg"
UNITS_OUT = ROOT / "data" / "geo" / "jp" / "jp_units.gpkg"
NORTHERN_TERRITORIES = {"01695", "01696", "01697", "01698", "01699", "01700"}
MAX_SNAP_KM = 15


def census_units():
    """{code: (name, population)} for the 1,896 units of census table 44-1."""
    sys.path.insert(0, str(HERE))
    from jp_census import read_census
    c = read_census()
    return c


def main():
    cen = census_units()
    pop = dict(zip(cen["code"], cen["total"]))
    g = gpd.read_file(N03)
    g = g[g["N03_007"].notna()].rename(columns={"N03_007": "unit"})
    extra = set(g["unit"]) - set(pop)
    if extra != NORTHERN_TERRITORIES:
        raise SystemExit(f"N03 codes with no census unit: {sorted(extra - NORTHERN_TERRITORIES)}")
    missing = set(pop) - set(g["unit"])
    if missing:
        raise SystemExit(f"census units with no N03 polygon: {sorted(missing)}")
    g = g[g["unit"].isin(set(pop))]
    units = g.dissolve(by="unit", as_index=False)[["unit", "geometry"]]
    assert len(units) == len(pop) == 1896, len(units)
    UNITS_OUT.parent.mkdir(parents=True, exist_ok=True)
    units.to_file(UNITS_OUT, layer="units", driver="GPKG")

    hexes = gpd.read_file(kontur_path("jp"))
    pts = gpd.GeoDataFrame({"pop": hexes["population"].to_numpy(dtype=float)},
                           geometry=hexes.geometry.centroid, crs=hexes.crs)
    u3857 = units.to_crs(hexes.crs)
    j = gpd.sjoin(pts, u3857, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)
    out = j["unit"].isna()
    print(f"  Kontur JP: {len(pts):,} hexes, {pts['pop'].sum():,.0f} people; "
          f"{int(out.sum()):,} hexes ({pts.loc[out, 'pop'].sum():,.0f} people) outside every polygon")
    # snap: distances in 3857 metres are stretched by 1/cos(lat), ~1.2 at Japan's latitudes;
    # scale the limit so it is MAX_SNAP_KM on the ground at 35N
    lim = MAX_SNAP_KM * 1000 / math.cos(math.radians(35))
    nr = gpd.sjoin_nearest(pts[out], u3857, how="left", max_distance=lim, distance_col="d")
    nr = nr[~nr.index.duplicated(keep="first")]
    j.loc[nr.index, "unit"] = nr["unit"]
    still = j["unit"].isna()
    print(f"  snapped {int(out.sum() - still.sum()):,} hexes "
          f"({pts.loc[out & ~still, 'pop'].sum():,.0f} people) to the nearest municipality; "
          f"{int(still.sum()):,} hexes ({pts.loc[still, 'pop'].sum():,.0f} people) beyond "
          f"{MAX_SNAP_KM} km dropped")
    if pts.loc[still, "pop"].sum() > 20_000:
        raise SystemExit("too many people beyond the snap distance: a bad layer")
    keep = ~still
    layer = gpd.GeoDataFrame({"unit": j.loc[keep, "unit"].astype(str).to_numpy(),
                              "pop": pts.loc[keep, "pop"].to_numpy()},
                             geometry=hexes.geometry[keep.to_numpy()].to_numpy(),
                             crs=hexes.crs).to_crs(4326)
    per = layer.groupby("unit")["pop"].sum()
    empty = sorted(set(pop) - set(per.index[per > 0]))
    print(f"  {len(empty)} units with no populated hex: "
          + ", ".join(f"{u} {cen.set_index('code').at[u, 'name']} ({pop[u]:,})" for u in empty))
    if sum(pop[u] for u in empty) > 5_000:
        raise SystemExit("units with no populated hex hold too many people")

    rows = [(u, pop[u], float(per.get(u, 0.0))) for u in pop if pop[u] > 0]
    ratio = sum(k for _, _, k in rows) / sum(c for _, c, _ in rows)
    norm = sorted((k / c / ratio, u) for u, c, k in rows)
    print(f"  Kontur / census nationally {ratio:.3f}; per unit normalised: p10 "
          f"{norm[len(norm) // 10][0]:.2f}  median {norm[len(norm) // 2][0]:.2f}  "
          f"p90 {norm[9 * len(norm) // 10][0]:.2f}")
    nm = cen.set_index("code")["name"]
    print("  lowest: " + ", ".join(f"{nm[u]} {r:.2f}" for r, u in norm[:6]))
    print("  highest: " + ", ".join(f"{nm[u]} {r:.2f}" for r, u in norm[-6:]))
    print(f"  {sum(1 for r, _ in norm if not (1 / 3 <= r <= 3))} of {len(norm)} units outside a "
          "factor of 3")
    ok = [(c, k) for _, c, k in rows if k > 0]
    lc = [math.log(c) for c, _ in ok]
    lk = [math.log(k) for _, k in ok]

    def pear(a, b):
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    r = pear(lc, lk)
    rng = random.Random(0)
    best = max(abs(pear(lc, rng.sample(lk, len(lk)))) for _ in range(500))
    print(f"  log correlation r = {r:.3f} against a best of {best:.3f} over 500 shuffles")
    if r <= best:
        raise SystemExit("THE JOIN IS NOT CARRYING INFORMATION")
    layer.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"  wrote {OUT} ({len(layer):,} hexes)")


if __name__ == "__main__":
    main()
