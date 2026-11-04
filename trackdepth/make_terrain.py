"""Terrain tiles for the page's 3D view, from our own rasters instead of AWS Terrarium (which has
bridges, ramps, building edges and -2,500 m pits around here).

Sources: raw/dem (USGS 3DEP bare earth, 3 m, NAVD88; fetch_dem.py), filled where it has no tile
from 3DEP at 15 m (fetched here into raw/dem15/, blended in over 4 cells at the 3 m edge), and
raw/bathy + raw/bathy_fill (NOAA CUDEM topobathy, NAVD88; fetch_bathy.py [--fill]) for the river
and harbour beds.

Water is the polygon in data/water.json (make_water.py, from the OSM coastline), which the page
draws flat at height 0. Land (outside it) keeps its 3DEP height, never below -0.9 m. Inside it the
terrain is the CUDEM bed, pushed to at most -1 m so it sits clearly under the plane, with a short
feather (about 2.5 cells) from the shore's land height. Water cells with no CUDEM value take the
nearest known bed, fading toward 0 with distance (e-fold 25 cells). The page colours a cell water
where the height is below -1 m, so water and land are told apart by that alone.

Everything is averaged onto a web-mercator grid at zoom 13 (about 14.5 m cells here), land and
water smoothed separately with a gaussian of 1 cell (nodata-aware) so the shore keeps its edge, and
written as Terrarium-encoded PNG tiles, data/terrain/{z}/{x}/{y}.png, zooms 9-13 (lower zooms are
2x2 means of the one above). Cells with no data are written as -10000 m; index.html makes them
invisible. Every tile of the block is written, empty ones too.

Textures. data/terrain/tex/{z}/{x}/{y}.webp (512 px at z12-13, 256 px below; the page draws the
terrain unlit, so these are the colours): near-black land lit by a soft hillshade baked from the
heights (the mesh has no vertex normals, so deck's own lighting facets), parks and woods dark green,
lakes and ponds light blue (OSM, fetch_landuse.py), and under the water polygon the bed tinted by
depth. Colours are written as the page shows them over the dark basemap at its terrain opacity
(OPACITY, FINAL). No-data cells are transparent. The texture is true to the ground; only the
height grid is stretched (next).

Tile edges. deck.gl's terrain mesher takes a 256 px image as 257 x 257 vertices and fills the last
column and row by copying the one before, so neighbouring tiles never agreed along their shared
edge (a crack, black through the transparent terrain). Each tile is therefore written sampled at
256 points spread over the tile's 256 px AND its far edge (stride 256/255): the copied last vertex
is the true height on the edge, equal to the next tile's first vertex. The tile's terrain is
stretched by under one pixel at its far edge, the same for height and colour.

    python make_terrain.py        # a few minutes, one core
"""
import json
import math
import os
import pathlib
import shutil
import warnings

import numpy as np
import rasterio
import shapely
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates, zoom

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "data" / "terrain"
ZMAX, ZMIN = 13, 9
NODATA = -10000.0
HALF = 20037508.342789244
TILE_M = lambda z: 2 * HALF / 2 ** z
# The square the 3D view shows (W, S, E, N), about 52 x 52 km, centred on Long Island City. Inside the
# extent of the data (raw/dem, raw/dem15, CUDEM), so nothing is fetched for it; the page's EXTENT is
# this. Tiles that miss it are not written, and cells outside it are no-data.
EXTENT = (-74.26, 40.508, -73.645, 40.975)
WATER_TOP = -1.0        # highest the bed gets under the water polygon
LAND_FLOOR = -0.9       # lowest land gets
FEATHER = 0.5           # cells from the shore over which the bed comes down from the land height. Was 2.5, but
                        # then a narrow creek was all slope and the page's 0 m contour line ran along its middle
                        # (and a river showed it as a ledge): the bed now starts at the first water cell.
FADE = 25.0             # cells over which a bed with no data nearby fades to 0
BLEND = 4.0             # cells over which 15 m ground takes over from 3 m ground
SIGMA_FLAT = 5.0        # cells (about 70 m on the ground; a cell is 14.5 m): the smoothing of flat land, see smooth_land
DECK_REACH = 4          # cells (about 58 m) of land around a pier that give its deck height
DECK_MIN, DECK_MAX = 1.5, 4.0   # m: the lowest and highest a pier deck is drawn
LAKE_PCT = 20           # percentile of the ground just inside a lake's shore that gives its flat surface
SLOPE_LO, SLOPE_HI = 0.03, 0.12   # ground slope between which flat smoothing gives way to 1 cell
# Texture colours (what the eye sees once the page draws the terrain at OPACITY over the dark basemap)
OPACITY = 0.45
BASEMAP = np.array([16.0, 16.0, 20.0])
FINAL = {"land": (24, 24, 27), "park": (26, 42, 31), "lake": (62, 100, 128)}   # lake: about as light as 0.45 opacity allows
HILL_EXAG = 4.0         # vertical exaggeration used for the shading only (the page default is 3)
HILL_GAIN = 1.5         # multiplicative part of the shading (1 = the slope's own ratio to flat)
HILL_ADD = 160.0        # additive part, in texture levels, per unit of (slope light - flat light)


def tile_xy(lon, lat, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


COARSE = ROOT / "raw" / "dem15"
EXPORT = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"


def fetch_coarse(w, s, e, n):
    """raw/dem only covers 4.5 km squares with track in them; fill the rest of the extent from
    3DEP at 15 m (4 x 4 requests of 1000 px), cached in raw/dem15/."""
    import requests
    from pyproj import Transformer
    COARSE.mkdir(parents=True, exist_ok=True)
    to = Transformer.from_crs(4326, 32618, always_xy=True)
    xs, ys = zip(*[to.transform(x, y) for x in (w, e) for y in (s, n)])
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    nx = ny = 4
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    out = []
    for i in range(nx):
        for j in range(ny):
            p = COARSE / f"dem15_{i}_{j}.tif"
            out.append(p)
            if p.exists() and p.stat().st_size > 1e5:
                continue
            bx = (x0 + i * dx, y0 + j * dy, x0 + (i + 1) * dx, y0 + (j + 1) * dy)
            params = {"bbox": ",".join(f"{v:.1f}" for v in bx), "bboxSR": 32618, "imageSR": 32618,
                      "size": f"{round(dx / 15)},{round(dy / 15)}", "format": "tiff", "pixelType": "F32",
                      "noData": -9999, "interpolation": "RSP_BilinearInterpolation", "compression": "LZ77", "f": "image"}
            for attempt in range(4):
                r = requests.get(EXPORT, params=params, headers={"User-Agent": "trackdepth/0.1"}, timeout=600)
                if r.ok and r.content.startswith((b"II*", b"MM\x00*")):
                    break
                print(f"{p.name}: {r.status_code}, retrying", flush=True)
            else:
                raise RuntimeError(f"gave up on {p.name}")
            p.write_bytes(r.content)
            print(f"{p.name} {len(r.content) / 1e6:.1f} MB", flush=True)
    return out


def sources():
    dem = sorted((ROOT / "raw" / "dem").glob("dem_*.tif"))
    # raw/bathy: blocks around the river tunnels; raw/bathy_fill: the rest of the water inside the
    # extent (fetch_bathy.py --fill). Kept apart so make_riverbed.py reads only the first.
    bathy = sorted((ROOT / "raw" / "bathy").glob("cudem_*.tif")) + sorted((ROOT / "raw" / "bathy_fill").glob("cudem_*.tif"))
    return dem, bathy


def tile_block(w, s, e, n):
    """First tile and tile counts, at the lowest zoom, of the block covering an extent."""
    x0, y1 = tile_xy(w, s, ZMIN)
    x1, y0 = tile_xy(e, n, ZMIN)
    tx0, ty0 = int(x0), int(y0)
    return tx0, ty0, int(x1) - tx0 + 1, int(y1) - ty0 + 1


def extent(paths):
    w, s, e, n = 180, 90, -180, -90
    for p in paths:
        with rasterio.open(p) as src:
            b = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        w, s, e, n = min(w, b[0]), min(s, b[1]), max(e, b[2]), max(n, b[3])
    return w, s, e, n


def mosaic(paths, tx0, ty0, nx, ny):
    """Average every source onto the zoom-13 mercator grid covering tiles tx0.., ty0.."""
    res = TILE_M(ZMAX) / 256
    transform = from_origin(-HALF + tx0 * TILE_M(ZMAX), HALF - ty0 * TILE_M(ZMAX), res, res)
    acc = np.zeros((ny * 256, nx * 256), np.float32)
    cnt = np.zeros(acc.shape, np.float32)
    dst = np.empty(acc.shape, np.float32)
    for k, p in enumerate(paths):
        dst.fill(np.nan)
        with rasterio.open(p) as src:
            reproject(rasterio.band(src, 1), dst, src_nodata=src.nodata, dst_nodata=np.nan,
                      dst_transform=transform, dst_crs="EPSG:3857", resampling=Resampling.average)
        ok = np.isfinite(dst) & (dst > -500)
        acc[ok] += dst[ok]
        cnt[ok] += 1
        if k % 20 == 0:
            print(f"  {p.parent.name}: {k + 1}/{len(paths)}", flush=True)
    out = np.full(acc.shape, np.nan, np.float32)
    has = cnt > 0
    out[has] = acc[has] / cnt[has]
    return out


def water_mask(tx0, ty0, nx, ny, path=None, touched=False):
    """data/water.json (or another GeoJSON) rasterised onto the same grid: True where a cell centre
    is inside (any cell it touches, with touched=True)."""
    res = TILE_M(ZMAX) / 256
    transform = from_origin(-HALF + tx0 * TILE_M(ZMAX), HALF - ty0 * TILE_M(ZMAX), res, res)
    fc = json.loads((path or ROOT / "data" / "water.json").read_text(encoding="utf-8"))

    def merc(xy):
        x = xy[:, 0] * HALF / 180
        y = np.log(np.tan(np.pi / 4 + np.radians(xy[:, 1]) / 2)) * HALF / np.pi
        return np.column_stack([x, y])
    shapes = [(shapely.transform(shapely.geometry.shape(f["geometry"]), merc), 1) for f in fc["features"]]
    if not shapes:
        return np.zeros((ny * 256, nx * 256), bool)
    m = rasterize(shapes, out_shape=(ny * 256, nx * 256), transform=transform, fill=0, dtype="uint8", all_touched=touched)
    return m.astype(bool)


def flatten_decks(z, D, M, land):
    """A flat top for every pier. The bare-earth lidar drops deck cells to about sea level, so a pier
    would sink under the water plane. Each connected deck takes the median of the land height within
    DECK_REACH cells of it on the landward side, at least DECK_MIN and at most DECK_MAX m."""
    from scipy.ndimage import binary_dilation, find_objects, label
    lab, n = label(D, structure=np.ones((3, 3), int))
    for k, sl in enumerate(find_objects(lab), start=1):
        if sl is None:
            continue
        y0, y1 = max(sl[0].start - DECK_REACH - 1, 0), sl[0].stop + DECK_REACH + 1
        x0, x1 = max(sl[1].start - DECK_REACH - 1, 0), sl[1].stop + DECK_REACH + 1
        comp = lab[y0:y1, x0:x1] == k
        ring = binary_dilation(comp, iterations=DECK_REACH) & ~D[y0:y1, x0:x1] & ~M[y0:y1, x0:x1]
        v = land[y0:y1, x0:x1][ring]
        v = v[np.isfinite(v)]
        h = float(np.median(v)) if v.size else DECK_MIN
        z[y0:y1, x0:x1][comp] = min(max(h, DECK_MIN), DECK_MAX)
    return n


def nsmooth(v, ok, sigma=1.0):
    """Gaussian smoothing over the cells where ok is True only (normalised convolution); NaN
    elsewhere beyond reach. Cells outside ok still get a value where ok cells are in reach."""
    num = gaussian_filter(np.where(ok, v, 0).astype(np.float32), sigma)
    den = gaussian_filter(ok.astype(np.float32), sigma)
    out = np.full(v.shape, np.nan, np.float32)
    good = den > 1e-4
    out[good] = num[good] / den[good]
    return out


def smooth_land(v, ok):
    """Land height, smoothed more where the ground is flat. The bare-earth DEM sags under buildings
    and is true on the streets, so a flat city shows a dip in every block and a rise at every street.
    A gaussian of SIGMA_FLAT cells (about 70 m on the ground) takes that out; steep ground (a ridge, the
    Palisades) keeps the light 1-cell smoothing, blended over SLOPE_LO to SLOPE_HI of ground slope."""
    light = nsmooth(v, ok, 1.0)
    strong = nsmooth(v, ok, SIGMA_FLAT)
    gy, gx = np.gradient(np.where(np.isfinite(strong), strong, 0).astype(np.float32), TILE_M(ZMAX) / 256 * 0.758)
    slope = np.hypot(gx, gy)
    del gx, gy
    t = np.clip(1 - (slope - SLOPE_LO) / (SLOPE_HI - SLOPE_LO), 0, 1).astype(np.float32)
    out = np.where(np.isfinite(strong), t * strong + (1 - t) * light, light)
    return out.astype(np.float32)


def encode(e):
    v = np.where(np.isfinite(e), e, NODATA).astype(np.float64) + 32768
    r = np.floor(v / 256)
    g = np.floor(v) - r * 256
    b = np.floor((v - np.floor(v)) * 256)
    return np.dstack([r, g, b]).clip(0, 255).astype(np.uint8)


def lake_labels(tx0, ty0, nx, ny):
    """Every OSM lake, pond and reservoir (the shapes the texture colours) rasterised onto the grid,
    one id per water body (0 = none). Tidal water, which is in data/water.json, is not here."""
    import osm_areas
    res = TILE_M(ZMAX) / 256
    transform = from_origin(-HALF + tx0 * TILE_M(ZMAX), HALF - ty0 * TILE_M(ZMAX), res, res)

    def merc(xy):
        x = xy[:, 0] * HALF / 180
        y = np.log(np.tan(np.pi / 4 + np.radians(xy[:, 1]) / 2)) * HALF / np.pi
        return np.column_stack([x, y])
    shapes = [(shapely.transform(g, merc), i + 1) for i, g in enumerate(osm_areas.load_all("lakes"))
              if g.geom_type in ("Polygon", "MultiPolygon")]
    # biggest first, so a pond drawn inside a lake polygon keeps its own id
    shapes.sort(key=lambda s: -s[0].area)
    return rasterize(shapes, out_shape=(ny * 256, nx * 256), transform=transform, fill=0, dtype="int32")


def flatten_lakes(z, g, LK):
    """One flat surface for each lake. Lidar has few returns over water, so the bare-earth height
    inside a lake is noise and the mesh drew a sawtooth across the Reservoir. Each takes a low
    percentile (LAKE_PCT) of the ground in a ring just inside its shore."""
    from scipy.ndimage import binary_erosion, find_objects
    n = 0
    for k, sl in enumerate(find_objects(LK), start=1):
        if sl is None:
            continue
        comp = LK[sl] == k
        if comp.sum() < 4:
            continue
        ring = comp & ~binary_erosion(comp, iterations=2)
        v = g[sl][ring]
        v = v[np.isfinite(v)]
        if v.size < 3:
            v = g[sl][comp]
            v = v[np.isfinite(v)]
        if not v.size:
            continue
        z[sl][comp] = max(float(np.percentile(v, LAKE_PCT)), LAND_FLOOR)
        n += 1
    return n


def merc_shapes():
    """The shapes that colour the texture, in mercator metres: the water polygon (piers already
    cut out), parks and woods, lakes and ponds."""
    import osm_areas
    from shapely.ops import unary_union

    def merc(xy):
        x = xy[:, 0] * HALF / 180
        y = np.log(np.tan(np.pi / 4 + np.radians(xy[:, 1]) / 2)) * HALF / np.pi
        return np.column_stack([x, y])
    fc = json.loads((ROOT / "data" / "water.json").read_text(encoding="utf-8"))
    out = {"sea": unary_union([shapely.geometry.shape(f["geometry"]) for f in fc["features"]])}
    for key, name in (("park", "parks"), ("lake", "lakes")):
        gs = osm_areas.load_all(name)
        out[key] = unary_union(gs) if gs else shapely.Polygon()
        print(f"{key}: {len(getattr(out[key], 'geoms', [out[key]]))} shapes", flush=True)
    return {k: shapely.transform(v, merc) for k, v in out.items()}


def tex_size(z):
    return 512 if z >= 12 else 256


def to_tex(final):
    """The texture colour that shows as `final` once drawn at OPACITY over the dark basemap."""
    return np.clip((np.asarray(final, float) - (1 - OPACITY) * BASEMAP) / OPACITY, 0, 255)


def bake_tile(win, z, i, j, shapes):
    """RGBA texture of one tile: land near-black lit by a soft hillshade, parks dark green, lakes
    light blue, the sea bed tinted by depth under the water polygon. `win` is the level's
    heights around the tile (6 px halo); the texture is true to the ground (no edge stretch)."""
    S = tex_size(z)
    f = S / 256
    w = np.array(win, np.float32)
    fin = np.isfinite(w)
    fill = np.nanmean(w) if fin.any() else 0.0
    w = np.where(fin, w, fill)
    up = zoom(w, f, order=1, mode="nearest", grid_mode=True) if f != 1 else w
    up = gaussian_filter(up, 0.9 * f)
    h = int(6 * f)
    up = up[h - 1:h + S + 1, h - 1:h + S + 1]       # one more px each side for the gradient
    res = TILE_M(z) / 256 / f * 0.758                # metres on the ground per texture px
    gy, gx = np.gradient(up * HILL_EXAG, res)
    gx, gy = gx[1:-1, 1:-1], gy[1:-1, 1:-1]
    zc = up[1:-1, 1:-1]
    az, alt = math.radians(315), math.radians(42)
    L = np.array([math.cos(alt) * math.sin(az), math.cos(alt) * math.cos(az), math.sin(alt)])
    norm = np.sqrt(gx ** 2 + gy ** 2 + 1)
    rel = (gx * L[0] + (-gy) * L[1] + L[2]) / norm / L[2]      # 1 on the flat
    d = (rel - 1)[..., None]                       # 0 on the flat, + facing the light, - away
    # shapes of this tile
    x0 = -HALF + (tx_of(z, i)) * TILE_M(z)
    y1 = HALF - (ty_of(z, j)) * TILE_M(z)
    tf = from_origin(x0, y1, TILE_M(z) / S, TILE_M(z) / S)
    box = (x0, y1 - TILE_M(z), x0 + TILE_M(z), y1)

    def mask(key):
        g = shapely.clip_by_rect(shapes[key], *box)
        if g.is_empty:
            return np.zeros((S, S), bool)
        return rasterize([(g, 1)], out_shape=(S, S), transform=tf, fill=0, dtype="uint8").astype(bool)
    sea, park, lake = mask("sea"), mask("park"), mask("lake") & ~mask("sea")
    land = to_tex(FINAL["land"]) + np.clip(zc, 0, 60)[..., None] * 0.15      # a little lighter uphill, same in every channel
    rgb = np.broadcast_to(land, (S, S, 3)).copy()
    rgb[park] = to_tex(FINAL["park"]) + np.clip(zc[park], 0, 60)[..., None] * 0.1
    # shading on near-black ground needs an additive part: a pure multiplier of 1.5 on 34 is 17 levels
    rgb = np.clip(rgb * np.clip(1 + HILL_GAIN * d, 0.3, 2.0) + HILL_ADD * d, 0, 255)
    rgb[lake] = to_tex(FINAL["lake"])
    k = np.clip((-zc - 1) / 25, 0, 1)[..., None]
    sea_rgb = (np.array([58, 100, 160.]) * (1 - k) + np.array([30, 58, 112.]) * k)
    rgb = np.where(sea[..., None], sea_rgb, rgb)
    a = np.where(np.kron(fin[6:-6, 6:-6], np.ones((int(f), int(f)), bool)), 255, 0).astype(np.uint8)
    return np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), a])


def tx_of(z, i):
    return _ORIGIN[0] * 2 ** (z - ZMIN) + i


def ty_of(z, j):
    return _ORIGIN[1] * 2 ** (z - ZMIN) + j


_ORIGIN = (0, 0)


def write_level(a, z, tx0, ty0, shapes):
    ny, nx = a.shape[0] // 256, a.shape[1] // 256
    pad = np.pad(a, 2, mode="edge")
    pad6 = np.pad(a, 6, mode="edge")
    k = np.arange(256) * (256 / 255) - 0.5 + 2        # +2: the padding
    n = 0
    ax0, ay1 = tile_xy(EXTENT[0], EXTENT[1], z)
    ax1, ay0 = tile_xy(EXTENT[2], EXTENT[3], z)
    for j in range(ny):
        for i in range(nx):
            # every tile that touches EXTENT, empty cells too: the page's terrain layer fails on a 404.
            if not (tx0 + i + 1 > ax0 and tx0 + i < ax1 and ty0 + j + 1 > ay0 and ty0 + j < ay1):
                continue
            # 256 samples over the tile's width plus its far edge, see the module note.
            yy, xx = np.meshgrid(j * 256 + k, i * 256 + k, indexing="ij")
            t = map_coordinates(pad, [yy, xx], order=1, mode="nearest", prefilter=False)
            d = OUT / str(z) / str(tx0 + i)
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(encode(t)).save(d / f"{ty0 + j}.png", optimize=True)
            td = OUT / "tex" / str(z) / str(tx0 + i)
            td.mkdir(parents=True, exist_ok=True)
            win = pad6[j * 256:j * 256 + 268, i * 256:i * 256 + 268]
            Image.fromarray(bake_tile(win, z, i, j, shapes)).save(td / f"{ty0 + j}.webp", "WEBP", quality=86, method=4)
            n += 1
    print(f"z{z}: {n} tiles", flush=True)


def halve(a):
    h, w = a.shape
    b = a.reshape(h // 2, 2, w // 2, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-nodata blocks stay nodata
        return np.nanmean(b, axis=(1, 3)).astype(np.float32)


def main():
    dem, bathy = sources()
    w, s, e, n = extent(dem)                         # the extent of the data
    tx0, ty0, nx, ny = tile_block(w, s, e, n)       # a block of whole tiles at every zoom, over all of it
    f = 2 ** (ZMAX - ZMIN)
    print(f"data {w:.4f},{s:.4f},{e:.4f},{n:.4f}; z{ZMIN} tiles x {tx0}+{nx}, y {ty0}+{ny}", flush=True)
    print(f"EXTENT = {list(EXTENT)}   (the page's EXTENT constant)", flush=True)
    global _ORIGIN
    _ORIGIN = (tx0, ty0)
    # The two mosaics take about 20 minutes; TERRAIN_CACHE=<folder> keeps them between runs, for
    # tuning the colours and the water (delete the folder when a source changes).
    cache = pathlib.Path(os.environ["TERRAIN_CACHE"]) if os.environ.get("TERRAIN_CACHE") else None
    if cache and (cache / "g.npy").exists() and (cache / "b.npy").exists():
        g, b = np.load(cache / "g.npy"), np.load(cache / "b.npy")
        print("mosaics from", cache, flush=True)
    else:
        g = mosaic(dem, tx0 * f, ty0 * f, nx * f, ny * f)
        own = np.isfinite(g)
        coarse = mosaic(fetch_coarse(w, s, e, n), tx0 * f, ty0 * f, nx * f, ny * f)
        # 3 m ground wins inside its tiles; 15 m ground takes over across BLEND cells at their edge
        k = np.clip(distance_transform_edt(own) / BLEND, 0, 1).astype(np.float32)
        use = np.isfinite(coarse)
        g = np.where(own & use, k * np.nan_to_num(g) + (1 - k) * np.nan_to_num(coarse), np.where(own, g, coarse)).astype(np.float32)
        del coarse, k, own
        b = mosaic(bathy, tx0 * f, ty0 * f, nx * f, ny * f)
        if cache:
            cache.mkdir(parents=True, exist_ok=True)
            np.save(cache / "g.npy", g)
            np.save(cache / "b.npy", b)
        if os.environ.get("CACHE_ONLY"):
            return
    print(f"ground: {np.isfinite(g).mean():.4f} of cells", flush=True)
    M = water_mask(tx0 * f, ty0 * f, nx * f, ny * f)
    print(f"water polygon covers {M.mean():.3f} of the block; CUDEM has a value in {np.isfinite(b)[M].mean():.4f} of that", flush=True)
    # bed: CUDEM where it has a value, else the nearest known bed fading toward 0
    known = M & np.isfinite(b)
    bed = np.where(known, b, np.nan).astype(np.float32)
    unknown = M & ~known
    if unknown.any() and known.any():
        dist, idx = distance_transform_edt(~known, return_indices=True)
        near = b[idx[0][unknown], idx[1][unknown]]
        bed[unknown] = near * np.exp(-dist[unknown] / FADE)
        del idx, dist
    print(f"bed filled by falloff in {int(unknown.sum())} water cells", flush=True)
    bed[M & ~np.isfinite(bed)] = 0.0
    # land and water smoothed apart so the shore keeps its edge; the bed comes down from the
    # shore's land height over FEATHER cells
    gok = np.isfinite(g)
    dpath = ROOT / "raw" / "decks.json"
    D = water_mask(tx0 * f, ty0 * f, nx * f, ny * f, dpath, touched=True) & ~M if dpath.exists() else np.zeros_like(M)
    print(f"deck cells (piers): {int(D.sum())}", flush=True)
    LK = lake_labels(tx0 * f, ty0 * f, nx * f, ny * f)
    LK[M] = 0
    print(f"lake cells: {int((LK > 0).sum())}", flush=True)
    # a deck or a lake is left out of the smoothing, so it does not blur into the shore
    landok = ~M & ~D & (LK == 0) & gok
    g0 = np.nan_to_num(g, nan=0.0)
    shore = smooth_land(np.clip(g0, LAND_FLOOR, 2.0), landok)      # land height reaching into the feather
    land = smooth_land(np.clip(g0, LAND_FLOOR, None), landok)
    water = nsmooth(np.minimum(bed, WATER_TOP), M)
    di = distance_transform_edt(M)
    wgt = np.clip((di - 0.5) / FEATHER, 0, 1).astype(np.float32)
    mixed = wgt * water + (1 - wgt) * np.where(np.isfinite(shore), shore, water)
    z = np.where(M, mixed, land)
    if D.any():
        print(f"{flatten_decks(z, D, M, land)} pier decks given a flat top", flush=True)
    if LK.any():
        print(f"{flatten_lakes(z, g, LK)} lakes given a flat surface", flush=True)
    z[~M & ~gok & ~D & (LK == 0)] = np.nan        # outside every source: empty
    del g, g0, b, bed, shore, land, water, mixed, di, wgt, gok, landok, D, LK
    # the square: no data outside EXTENT (a few cells of margin stay, for the edge of each tile)
    ex0, ey1 = tile_xy(EXTENT[0], EXTENT[1], ZMAX)
    ex1, ey0 = tile_xy(EXTENT[2], EXTENT[3], ZMAX)
    c0, c1 = int((ex0 - tx0 * f) * 256), int(math.ceil((ex1 - tx0 * f) * 256))
    r0, r1 = int((ey0 - ty0 * f) * 256), int(math.ceil((ey1 - ty0 * f) * 256))
    z[:r0, :] = np.nan
    z[r1:, :] = np.nan
    z[:, :c0] = np.nan
    z[:, c1:] = np.nan
    print(f"cells below {WATER_TOP} m: {(z < WATER_TOP + 1e-3).mean():.3f}; outside the polygon below 0: {(z[~M] < 0).mean():.4f}", flush=True)
    shapes = merc_shapes()
    if OUT.exists():
        shutil.rmtree(OUT)
    a = z
    for zz in range(ZMAX, ZMIN - 1, -1):
        kk = 2 ** (zz - ZMIN)
        write_level(a, zz, tx0 * kk, ty0 * kk, shapes)
        if zz > ZMIN:
            a = halve(a)
    total = sum(p.stat().st_size for p in OUT.rglob("*.png"))
    print(f"{OUT}: {total / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
