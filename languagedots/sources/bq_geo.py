"""Caribbean Netherlands: the placement layer, WorldPop 100 m population summed to ~370 m cells,
keyed to the three islands -> data/geo/bq/bq_cells.gpkg.

    python sources/bq_geo.py [--fetch]

WHY NOT KONTUR OR PLAIN POLYGONS. Kontur's BQ extract has no hexes at all (religiondots,
sources/bq.py), and religiondots spreads its dots evenly over Natural Earth's island polygons,
which puts Bonaire's dots into Washington Slagbaai park and the salt pans. WorldPop's
constrained 2020 raster for BES (`BSGM/BES/bes_ppp_2020_UNadj_constrained.tif`) places people
only on built-up land, so the dots follow Kralendijk, Rincon, Oranjestad, Windwardside and The
Bottom. It moves people only inside their island: the counts stay the survey's
(AGENT_BRIEF Â§4.4), so the absolute level never matters, only the shape.

THE CONTROL. WorldPop's unconstrained 2020 raster (`Global_2000_2020/2020/BES/bes_ppp_2020_UNadj.tif`)
is a different model; both are summed per island and compared with CBS's 1 January 2020
populations (83774NED); WorldPop has Sint Eustatius at 1.5x CBS, a level difference that a
within-island weight never sees. Three units is a weak test; it is there to catch a raster that puts
Saba's people on Sint Eustatius, not to grade the shape.

ISLAND JOIN. Cells are keyed by their centroid's position, with religiondots' own rule for
splitting Natural Earth's NLY unit (sources/bq.py::geometry): west of 65 W is Bonaire, east of
it and north of 17.56 N is Saba, the rest Sint Eustatius. The islands are 60-700 km apart, so a
coastal cell whose centroid is at sea still lands on its own island; asserted by checking every
cell lies within 0.05 degrees of religiondots' polygon for the island it was given.
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import json  # noqa: E402
import urllib.request  # noqa: E402
from pathlib import Path  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

RAW = HERE / "data" / "raw" / "bq"
OUT = HERE / "data" / "geo" / "bq" / "bq_cells.gpkg"
RD_ISLANDS = RD_GEO / "bq" / "bq_islands.gpkg"
WP = "https://data.worldpop.org/GIS/Population/"
DRAWN = (WP + "Global_2000_2020_Constrained/2020/BSGM/BES/bes_ppp_2020_UNadj_constrained.tif",
         RAW / "bes_ppp_2020_UNadj_constrained.tif")
CONTROL = (WP + "Global_2000_2020/2020/BES/bes_ppp_2020_UNadj.tif",
           RAW / "bes_ppp_2020_UNadj.tif")
POP_JSON = RAW / "83774NED_TypedDataSet.json"       # fetched by sources/bq_survey.py
ISLANDS = {"GM9001": "Bonaire", "GM9002": "Sint Eustatius", "GM9003": "Saba"}
BLOCK = 4
# per island, grid / CBS, each normalised by its own national ratio. Measured: constrained 0.92 /
# 1.54 / 1.03, unconstrained below. Sint Eustatius is 1.5x in WorldPop (its UN-adjusted
# island total runs ahead of CBS), which does not matter here: each island's dots come from the
# survey and CBS, and the grid only spreads them inside the island. The band catches a gross
# mis-keying (an island at 0 or 3x), not a level difference.
BAND = 2.0
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for url, path in (DRAWN, CONTROL):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        body = urllib.request.urlopen(req, timeout=300).read()
        if body[:4] not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
            raise SystemExit(f"{url}: not a TIFF ({body[:20]!r})")
        tmp = path.with_suffix(".part")
        tmp.write_bytes(body)
        tmp.replace(path)
        print(f"  {path.name}: {len(body):,} bytes")


def cells(path):
    import geopandas as gpd
    import numpy as np
    import rasterio
    from shapely.geometry import box

    with rasterio.open(path) as src:
        if src.crs is None or src.crs.to_epsg() != 4326:
            raise SystemExit(f"{path}: expected EPSG:4326, got {src.crs}")
        a = src.read(1).astype("float64")
        t, nodata = src.transform, src.nodata
    a[~np.isfinite(a)] = 0.0
    if nodata is not None:
        a[a == nodata] = 0.0
    a[a < 0] = 0.0
    h, w = a.shape
    H, W = -(-h // BLOCK) * BLOCK, -(-w // BLOCK) * BLOCK
    pad = np.zeros((H, W))
    pad[:h, :w] = a
    pop = pad.reshape(H // BLOCK, BLOCK, W // BLOCK, BLOCK).sum(axis=(1, 3))
    dx, dy = t.a * BLOCK, t.e * BLOCK
    r, c = np.nonzero(pop > 0)
    x0, y0 = t.c + c * dx, t.f + r * dy
    geom = [box(x, min(y, y + dy), x + dx, max(y, y + dy)) for x, y in zip(x0, y0)]
    g = gpd.GeoDataFrame({"pop": pop[r, c]}, geometry=geom, crs="EPSG:4326")
    cx, cy = x0 + dx / 2, y0 + dy / 2
    unit = np.where(cx < -65.0, "GM9001", np.where(cy > 17.56, "GM9003", "GM9002"))
    g["unit"] = unit
    return g


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv:
        fetch()
    for _, p in (DRAWN, CONTROL):
        if not p.exists():
            raise SystemExit(f"missing {p}; run with --fetch")
    cbs = {r["CaribischNederland"].strip(): r["BevolkingOp1Januari_1"]
           for r in json.loads(POP_JSON.read_text(encoding="utf-8"))["value"]
           if r["Perioden"] == "2020JJ00"}
    cbs = {u: cbs[u] for u in ISLANDS}
    isl = gpd.read_file(RD_ISLANDS)
    if sorted(isl["unit"]) != sorted(ISLANDS):
        raise SystemExit(f"{RD_ISLANDS}: units {sorted(isl['unit'])}")
    polys = dict(zip(isl["unit"], isl.geometry))

    norm = {}
    for label, (_, path) in (("drawn (constrained)", DRAWN), ("control (unconstrained)", CONTROL)):
        g = cells(path)
        far = [u for u, geom in zip(g["unit"], g.geometry) if geom.distance(polys[u]) > 0.05]
        if far:
            raise SystemExit(f"{label}: {len(far)} cells over 0.05 deg from their island")
        per = g.groupby("unit")["pop"].sum()
        ratio = per.sum() / sum(cbs.values())
        norm[label] = {u: per.get(u, 0) / cbs[u] / ratio for u in ISLANDS}
        print(f"  {label}: {len(g):,} cells, {per.sum():,.0f} people (CBS 2020 "
              f"{sum(cbs.values()):,}, ratio {ratio:.3f})")
        for u, nm in ISLANDS.items():
            print(f"      {nm:<15} grid {per.get(u, 0):>8,.0f}  CBS {cbs[u]:>6,}  "
                  f"norm {norm[label][u]:.2f}  cells {(g['unit'] == u).sum()}")
        bad = {ISLANDS[u]: round(v, 2) for u, v in norm[label].items()
               if not 1 / BAND <= v <= BAND}
        if bad:
            raise SystemExit(f"{label}: islands outside a factor {BAND}: {bad}")
        if label.startswith("drawn"):
            out = g
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out[["unit", "pop", "geometry"]].to_file(OUT, layer="cells", driver="GPKG")
    print(f"  wrote {OUT.relative_to(HERE)}: {len(out):,} cells")


if __name__ == "__main__":
    main()
