"""Sweden placement layer: SCB's own 1 km population grid, keyed to the 290 kommuner.

    python sources/se_geo.py --fetch     download the grid (WFS geopackage, about 35 MB)
    python sources/se_geo.py

-> data/geo/se/se_grid1km.gpkg (unit = four-digit kommun code, pop = residents 31 Dec 2025)

WHY THIS GRID AND NOT KONTUR. SCB publishes the register population on 1 km squares as open
data (`stat:befolkning_1km_2025` on geodata.scb.se, "Statistik på rutor", CC0): the same
register as the country-of-birth table, at the same date. Kontur models people from buildings,
and Sweden, like Finland, has some 600,000 fritidshus (sources/fi_geo.py's reason). Small
squares are perturbed by SCB's disclosure control; the layer only weights dots inside a
kommun, every count comes from the table.

THE JOIN. SCB's squares carry no kommun code, so each square goes to the kommun its centre
falls in. Kommun polygons: religiondots' se_lau.gpkg (GISCO LAU 2021, 290 kommuner; the codes
have not changed since 2007), read-only. A square straddling a kommun line goes wholly to one
side; centres outside every polygon (coast, the sea in a square) go to the nearest kommun.

CHECKS (printed): every one of the 290 has squares; grid residents per kommun against SCB's
table (ratio, p1/p50/p99 and the worst five).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
RAW = HERE / "data" / "raw" / "se"
GRID = RAW / "befolkning_1km_2025.gpkg"
URL = ("https://geodata.scb.se/geoserver/stat/wfs?service=WFS&REQUEST=GetFeature&version=1.1.0"
       "&TYPENAMES=stat:befolkning_1km_2025&outputFormat=geopackage")
LAU = HERE.parent / "religiondots" / "data" / "geo" / "se" / "se_lau.gpkg"
OUT = HERE / "data" / "geo" / "se" / "se_grid1km.gpkg"


def fetch():
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=1800, stream=True)
    r.raise_for_status()
    with open(GRID, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)


def build():
    import se_scb
    grid = gpd.read_file(GRID)
    grid = grid[grid["beftotalt"] > 0][["rutid_scb", "beftotalt", "geometry"]].copy()
    assert grid["rutid_scb"].is_unique
    kom = gpd.read_file(LAU).rename(columns={"unit": "nuts3", "lau": "unit"})
    kom = kom[["unit", "geometry"]].to_crs(grid.crs)
    assert len(kom) == 290 and kom["unit"].is_unique
    pts = gpd.GeoDataFrame(grid[["rutid_scb"]], geometry=grid.geometry.centroid, crs=grid.crs)
    j = gpd.sjoin(pts, kom[["unit", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)
    miss = j["unit"].isna()
    if miss.any():
        near = gpd.sjoin_nearest(pts[miss], kom[["unit", "geometry"]], how="left")
        near = near[~near.index.duplicated(keep="first")]
        j.loc[miss, "unit"] = near["unit"]
    print(f"  {len(grid):,} populated squares, {grid['beftotalt'].sum():,} residents; "
          f"{int(miss.sum())} centres outside every kommun ({grid.loc[miss, 'beftotalt'].sum():,} "
          f"people) given to the nearest")
    grid["unit"] = j["unit"].astype(str)
    grid = grid.rename(columns={"beftotalt": "pop", "rutid_scb": "square"})
    tab, _, _, _ = se_scb.regional()
    tot = pd.Series({k: v["TOTfod"] for k, v in tab.items()})
    g = grid.groupby("unit")["pop"].sum()
    assert set(g.index) == set(tot.index), (set(tot.index) - set(g.index))
    r = (g / tot).sort_values()
    print(f"  grid/table per kommun: p1 {r.quantile(.01):.3f}  p50 {r.median():.3f}  "
          f"p99 {r.quantile(.99):.3f}")
    print("  lowest", [(k, round(v, 3)) for k, v in r.head(5).items()])
    print("  highest", [(k, round(v, 3)) for k, v in r.tail(5).items()])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    grid[["square", "unit", "pop", "geometry"]].to_crs(4326).to_file(OUT, driver="GPKG")
    print("  wrote", OUT)


if __name__ == "__main__":
    if "--fetch" in sys.argv or not GRID.exists():
        fetch()
    build()
