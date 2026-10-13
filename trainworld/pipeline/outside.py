"""T-276: the outside world for the regional market: everyone who lives in no zone.

The market shares each zone's longer trips among the represented places and this outside world, so
a share of every zone's trips leaves the map (docs/zones.md 5). Kontur Population 2023 (H3
resolution 6, global, CC BY 4.0; https://data.humdata.org/dataset/kontur-population-dataset), with
every hexagon whose centre lies inside a registered zone's boundary left out, aggregated by distance
from the nearest zone: resolution 5 within 300 km, 4 within 1,000 km, 3 beyond, cut at 5,000 km.

    python pipeline/outside.py        -> data/work/outside.json  [[lon, lat, people], ...]

pipeline/intercity.py reads it. Seconds once the gpkg is unpacked in data/raw/kontur/.
"""
import gzip, json, math, os, shutil, sqlite3, sys
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "3")
import h3
import numpy as np
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/kontur"
GPKG = RAW / "kontur_population_20231101_r6.gpkg"
URL = "https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/kontur_population_20231101_r6.gpkg.gz"
REGISTRY = json.loads((ROOT / "app/src/game/registry.json").read_text(encoding="utf-8"))
# the zones: every registered pack in the market (a pack with market -1 is not placed yet)
MARKET_ZONES = [c for c in REGISTRY["cities"] if c["market"] >= 0]
NEAR_KM, MID_KM, CUT_KM = 300.0, 1000.0, 5000.0
R_KM = 6371.0088


def unit(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def main():
    if not GPKG.exists():
        gz = RAW / (GPKG.name + ".gz")
        if not gz.exists():
            sys.exit(f"missing {gz}: download {URL} into data/raw/kontur/")
        with gzip.open(gz, "rb") as f, open(GPKG, "wb") as g:
            shutil.copyfileobj(f, g, 1 << 24)
    rows = sqlite3.connect(GPKG).execute("select h3, population from population where population > 0").fetchall()
    cells = [r[0] for r in rows]
    pop = np.array([r[1] for r in rows], dtype=float)
    ll = np.array([h3.cell_to_latlng(c) for c in cells])  # (lat, lon)
    lat, lon = ll[:, 0], ll[:, 1]
    print(f"Kontur: {len(cells):,} hexagons, {pop.sum() / 1e9:.3f} billion people")
    # inside a zone: left out (those people are the zones' own)
    inside = np.zeros(len(cells), bool)
    zone_pts = []
    for c in MARKET_ZONES:
        gj = json.loads((ROOT / f"data/packs/{c['id']}.boundary.geojson").read_text(encoding="utf-8"))
        geoms = [shape(f["geometry"]) for f in gj["features"]] if "features" in gj else [shape(gj["geometry"] if "geometry" in gj else gj)]
        g = shapely.union_all(geoms)
        x0, y0, x1, y1 = g.bounds
        box = (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)
        idx = np.nonzero(box)[0]
        hit = shapely.contains_xy(g, lon[idx], lat[idx])
        inside[idx[hit]] = True
        zone_pts.append((c["center"][0], c["center"][1]))
        # boundary vertices as well, so "distance from the nearest zone" sees a zone's whole extent
        for part in getattr(g, "geoms", [g]):
            xy = np.asarray(part.exterior.coords)[::5]
            zone_pts.extend(map(tuple, xy))
    print(f"inside the zones: {inside.sum():,} hexagons, {pop[inside].sum() / 1e6:.2f} million people")
    zp = np.array(zone_pts)
    tree = cKDTree(unit(zp[:, 0], zp[:, 1]))
    chord, _ = tree.query(unit(lon, lat))
    dist = 2 * R_KM * np.arcsin(np.clip(chord / 2, 0, 1))
    keep = ~inside & (dist <= CUT_KM)
    res = np.where(dist <= NEAR_KM, 5, np.where(dist <= MID_KM, 4, 3))
    groups = {}
    for i in np.nonzero(keep)[0]:
        p = h3.cell_to_parent(cells[i], int(res[i]))
        g = groups.setdefault(p, [0.0, 0.0, 0.0])
        g[0] += pop[i] * lon[i]
        g[1] += pop[i] * lat[i]
        g[2] += pop[i]
    pts = [[round(a / w, 4), round(b / w, 4), round(w, 1)] for a, b, w in groups.values()]
    out = {"source": "Kontur Population 2023-11-01, H3 resolution 6, CC BY 4.0, " + URL,
           "rule": "hexagons outside every registered zone's boundary, within 5,000 km of one; people-weighted centres of H3 resolution 5 parents within 300 km, 4 within 1,000 km, 3 beyond",
           "zones": [c["id"] for c in MARKET_ZONES],
           "points": pts}
    path = ROOT / "data/work/outside.json"
    path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"outside: {len(pts):,} points, {sum(p[2] for p in pts) / 1e6:.1f} million people within {CUT_KM:.0f} km -> {path}")


if __name__ == "__main__":
    main()
