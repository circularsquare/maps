"""T-1285: a coarse water mask for the whole world, for the ground outside every pack's own mask.

    python pipeline/world_water.py build [--per-deg 60] [--out data/terrain/world.water] [--png preview.png]

Sources, both global and open:
- the sea: OSM's coastline-derived water polygons (osmdata.openstreetmap.de,
  `data/raw/osmdata/water-polygons-split-4326.fgb`, made by `osm_water.sea_fgb`), ODbL;
- large lakes: Natural Earth 1:10m lakes (`ne_10m_lakes.shp`, public domain), so the Caspian, the
  Great Lakes and the like are water too. Smaller water shows only in the packs' own masks.

A cell is water when at least half of it is: each 30-degree tile is drawn at 4 x 4 subcells a cell
and counted. Cells are `1/per_deg` degrees on the world lattice from 180 W and 90 S.

File (`cost::WorldWater::decode` in the sim): `TWW1`, u32 width, height, cells a degree, then each
row from the south: a LEB128 count of boundaries, then the boundaries as LEB128 gaps from the last
(the first from column 0), alternately the first water column and the first land column after it.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEA_FGB = os.path.join(ROOT, "data", "raw", "osmdata", "water-polygons-split-4326.fgb")
# the maps tree's shared copy (an agent worktree of trainworld reads the main tree's)
LAKES = next((p for p in (os.path.join(os.path.dirname(ROOT), "data", "ne_10m_lakes", "ne_10m_lakes.shp"),
                          r"C:\Users\anita\projects\maps\data\ne_10m_lakes\ne_10m_lakes.shp") if os.path.exists(p)),
             "ne_10m_lakes.shp")
TILE = 30
SUB = 4


def leb(n, out):
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return


def rings(geom):
    from shapely import get_parts
    for p in get_parts(geom):
        if p.geom_type != "Polygon" or p.is_empty:
            continue
        yield np.asarray(p.exterior.coords), [np.asarray(r.coords) for r in p.interiors]


def draw_tile(w, s, per, geoms):
    """Water share in sixteenths for the tile's cells, rows from the north."""
    from PIL import Image, ImageDraw
    n = s + TILE
    k = per * SUB
    size = TILE * k
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    holes = []
    for g in geoms:
        for ext, ins in rings(g):
            if len(ext) < 3:
                continue
            xy = np.column_stack(((ext[:, 0] - w) * k, (n - ext[:, 1]) * k)).ravel().tolist()
            d.polygon(xy, fill=1)
            holes.extend(ins)
    # islands after all the water, so a shared edge goes to the land
    for r in holes:
        if len(r) >= 3:
            d.polygon(np.column_stack(((r[:, 0] - w) * k, (n - r[:, 1]) * k)).ravel().tolist(), fill=0)
    a = np.asarray(img, dtype=np.uint8)
    cells = TILE * per
    return a.reshape(cells, SUB, cells, SUB).sum(axis=(1, 3))


def build(per, out, png):
    import pyogrio
    from shapely import box, simplify
    t0 = time.time()
    W, H = 360 * per, 180 * per
    assert W <= 65535, "column numbers are u16 in the sim"
    water = np.zeros((H, W), dtype=bool)
    tol = 0.5 / (per * SUB)
    lakes = pyogrio.read_dataframe(LAKES, columns=[])
    lake_geoms = simplify(lakes.geometry.values, tol)
    for s in range(-90, 90, TILE):
        for w in range(-180, 180, TILE):
            bb = (w, s, w + TILE, s + TILE)
            sea = pyogrio.read_dataframe(SEA_FGB, bbox=bb, columns=[])
            geoms = list(simplify(sea.geometry.values, tol))
            ext = box(*bb)
            geoms += [g for g in lake_geoms if g.intersects(ext)]
            share = draw_tile(w, s, per, geoms)
            r0, c0 = (s + 90) * per, (w + 180) * per
            # rows from the north in the tile, from the south in the world
            water[r0:r0 + TILE * per, c0:c0 + TILE * per] = (share >= SUB * SUB // 2)[::-1]
            print(f"  tile {w},{s}: {len(sea):,} sea polygons, {time.time() - t0:.0f} s", flush=True)
    buf = bytearray(b"TWW1")
    for v in (W, H, per):
        buf += int(v).to_bytes(4, "little")
    bounds = 0
    for r in range(H):
        row = water[r]
        edges = np.flatnonzero(np.diff(np.concatenate(([False], row, [False])).astype(np.int8)))
        leb(len(edges), buf)
        last = 0
        for e in edges.tolist():
            leb(e - last, buf)
            last = e
        bounds += len(edges)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    part = out + ".part"
    with open(part, "wb") as f:
        f.write(buf)
    os.replace(part, out)
    print(f"{out}: {W} x {H} cells, {water.mean() * 100:.1f}% water, {bounds:,} boundaries, "
          f"{len(buf) / 1e6:.2f} MB in {time.time() - t0:.0f} s", flush=True)
    if png:
        from PIL import Image
        step = max(1, per // 8)
        Image.fromarray((~water[::-step, ::step]).astype(np.uint8) * 255).save(png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--per-deg", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "terrain", "world.water"))
    ap.add_argument("--png")
    a = ap.parse_args()
    build(a.per_deg, a.out, a.png)


if __name__ == "__main__":
    main()
