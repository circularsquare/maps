"""The water polygon for the 3D view: sea, harbour and tidal rivers inside the terrain extent, from
the OpenStreetMap coastline (fetch_coast.py). The page draws it flat at height 0; make_terrain.py
rasterises it to decide which terrain cells are water.

OSM coastline ways run with land on their left. They are cut to the extent box, noded together with
the box outline and polygonised; each piece is land or water by which side of a coastline segment it
lies on (voted over every segment on its boundary). Water = box minus land, minus the piers
(man_made=pier, breakwater and groyne areas and strips from fetch_landuse.py), which the coastline
leaves in the water.

Writes data/water.json (GeoJSON, one Polygon with holes for the land, lon/lat, 5 decimals).

    python make_water.py
"""
import json
import pathlib

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, box, mapping
from shapely.ops import polygonize, unary_union

ROOT = pathlib.Path(__file__).parent
EPS = 3e-6      # degrees, about 0.3 m, to step off a segment


def terrain_extent():
    import make_terrain as mt
    return mt.EXTENT


def build():
    w, s, e, n = terrain_extent()
    pad = 0.002
    bx = box(w - pad, s - pad, e + pad, n + pad)
    ways = json.loads((ROOT / "raw" / "osm_coast.json").read_text(encoding="utf-8"))["elements"]
    lines = []
    for el in ways:
        c = [(p["lon"], p["lat"]) for p in el["geometry"]]
        if len(c) < 2:
            continue
        cut = LineString(c).intersection(bx)
        parts = [cut] if cut.geom_type == "LineString" else list(getattr(cut, "geoms", []))
        lines += [p for p in parts if p.geom_type == "LineString" and p.length > 0]
    print(len(ways), "ways ->", len(lines), "pieces inside the box")
    pieces = list(polygonize(unary_union(lines + [bx.exterior])))
    print(len(pieces), "faces")
    vote = np.zeros((len(pieces), 2), int)      # [land, water]
    # every coastline segment: a point a hair to its left (land) and to its right (water)
    seg = []
    for ln in lines:
        xy = np.asarray(ln.coords)
        s2 = np.column_stack([xy[:-1], xy[1:]])
        seg.append(s2[np.linspace(0, len(s2) - 1, min(len(s2), 5)).astype(int)])     # a few per way is enough
    seg = np.vstack(seg)
    dx, dy = seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]
    L = np.hypot(dx, dy)
    seg, dx, dy, L = seg[L > 0], dx[L > 0], dy[L > 0], L[L > 0]
    mx, my = (seg[:, 0] + seg[:, 2]) / 2, (seg[:, 1] + seg[:, 3]) / 2
    nx, ny = -dy / L * EPS, dx / L * EPS
    print(len(seg), "coastline segments", flush=True)
    for side, sign in ((0, 1), (1, -1)):
        px, py = mx + sign * nx, my + sign * ny
        for i, pc in enumerate(pieces):
            x0, y0, x1, y1 = pc.bounds
            near = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
            if near.any():
                vote[i, side] = int(shapely.contains_xy(pc, px[near], py[near]).sum())
    land = [p for p, v in zip(pieces, vote) if v[0] > v[1]]
    mixed = [(i, v.tolist()) for i, v in enumerate(vote) if v[0] and v[1]]
    print(len(land), "land faces;", len(mixed), "with votes both ways:", mixed[:10])
    water = bx.difference(unary_union(land)).intersection(box(w - 0.001, s - 0.001, e + 0.001, n + 0.001))
    # piers, breakwaters and groynes stand over the water: land, whatever the coastline says
    pf = ROOT / "raw" / "osm_piers.json"
    piers = shapely.Polygon()
    if pf.exists():
        import osm_areas
        piers = unary_union(osm_areas.load(pf, line_m=3.0))
        print(f"{len(piers.geoms) if hasattr(piers, 'geoms') else 1} pier shapes cut out of the water")
        water = water.difference(piers)
    # what the lidar ground shows standing over the water (piers and wharves, which OSM maps only
    # in part: the coastline runs along the esplanade behind them): 3 m ground above 1.2 m
    water = unary_union([water, bed_water(water)])
    water = unary_union([water, tidal_inland(water)])
    st = structures(water)
    water = water.difference(st)
    water = water.simplify(4e-6, preserve_topology=True)
    # the decks (OSM piers and the lidar structures), for make_terrain.py to give them a flat top
    decks = unary_union([piers, st]).intersection(box(w - 0.001, s - 0.001, e + 0.001, n + 0.001))
    return water, decks


def tidal_inland(water, reach=2e-4):
    """Canals and creeks the coastline leaves out (Gowanus Canal, Newtown Creek, Flushing Creek):
    OSM water polygons (raw/osm_lakes.json) that touch the water polygon, within `reach` degrees
    (about 20 m), and so on outward from each one added. They are tidal, so at sea level; a lake
    or reservoir further up the hill touches no sea water and stays a lake (colour only)."""
    from shapely.strtree import STRtree
    import osm_areas
    pieces = [g for g in osm_areas.load_all("lakes") if g.area > 2e-9]
    if not pieces:
        return shapely.Polygon()
    tree = STRtree(pieces)
    taken, grown = set(), water
    for _ in range(12):
        new = [i for i in tree.query(grown, predicate="dwithin", distance=reach) if i not in taken]
        if not new:
            break
        taken.update(int(i) for i in new)
        grown = unary_union([grown] + [pieces[i] for i in new])
    s = unary_union([pieces[i] for i in taken]) if taken else shapely.Polygon()
    print(f"inland tidal water from OSM polygons: {len(taken)} pieces, area {s.area:.5f} deg2")
    return s


def bed_water(water, below=-1.0, min_area=3e-8):
    """Water the coastline does not hold: the OSM coastline ends at Spuyten Duyvil, so the Hudson
    above it (and any other inland tidal water) has no polygon. Anywhere CUDEM has a bed below
    `below` m outside the water polygon is water too; polygons smaller than `min_area` deg2
    (about 300 m2) are dropped."""
    import rasterio
    from rasterio.features import rasterize, shapes
    from scipy.ndimage import binary_opening
    import make_terrain as mt
    _, bathy = mt.sources()
    found = []
    for p in bathy:
        with rasterio.open(p) as src:
            a = src.read(1)
            lb = src.bounds
            wet = np.isfinite(a) & (a > -500) & (a < below)
            if not wet.any():
                continue
            part = shapely.clip_by_rect(water, lb.left - 1e-4, lb.bottom - 1e-4, lb.right + 1e-4, lb.top + 1e-4)
            inw = (rasterize([(part, 1)], out_shape=a.shape, transform=src.transform, fill=0, dtype="uint8").astype(bool)
                   if not part.is_empty else np.zeros(a.shape, bool))
            extra = binary_opening(wet & ~inw, np.ones((3, 3), bool))
            if not extra.any():
                continue
            polys = [shapely.geometry.shape(g) for g, v in shapes(extra.astype("uint8"), mask=extra, transform=src.transform)]
            found += [g for g in polys if g.area > min_area]
    s = unary_union(found).buffer(3e-5).buffer(-3e-5) if found else shapely.Polygon()
    # Only what is joined to the water already, step by step from it (within about 20 m): the
    # Hudson above Spuyten Duyvil, not a rail yard or a highway cut that CUDEM has below -1 m.
    from shapely.strtree import STRtree
    parts = list(getattr(s, "geoms", [s]))
    tree = STRtree(parts)
    taken, grown = set(), water
    for _ in range(40):
        new = [int(i) for i in tree.query(grown, predicate="dwithin", distance=2e-4) if int(i) not in taken]
        if not new:
            break
        taken.update(new)
        grown = unary_union([grown] + [parts[i] for i in new])
    s2 = unary_union([parts[i] for i in taken]) if taken else shapely.Polygon()
    print(f"water from the CUDEM bed outside the coastline polygon: {len(found)} pieces, {len(parts)} parts, "
          f"{len(taken)} joined to the water, area {s2.area:.5f} of {s.area:.5f} deg2")
    return s2


def structures(water, top=1.2):
    """Polygons (lon/lat) of 3DEP 3 m ground higher than `top` m inside the water polygon."""
    import rasterio
    from rasterio.features import rasterize, shapes
    from rasterio.warp import transform_bounds
    from pyproj import Transformer
    from scipy.ndimage import binary_opening
    import make_terrain as mt
    to_utm = Transformer.from_crs(4326, 32618, always_xy=True)
    to_ll = Transformer.from_crs(32618, 4326, always_xy=True)

    def proj(tr):
        return lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1]))
    found = []
    dem, _ = mt.sources()
    for p in dem:
        with rasterio.open(p) as src:
            b = src.bounds
            lb = transform_bounds(src.crs, "EPSG:4326", *b)
            part = shapely.clip_by_rect(water, lb[0] - 1e-4, lb[1] - 1e-4, lb[2] + 1e-4, lb[3] + 1e-4)
            if part.is_empty:
                continue
            a = src.read(1)
            ok = np.isfinite(a) & (a > -500)
            wu = shapely.transform(part, proj(to_utm))
            inw = rasterize([(wu, 1)], out_shape=a.shape, transform=src.transform, fill=0, dtype="uint8").astype(bool)
            hi = inw & ok & (a > top)
            hi = binary_opening(hi, np.ones((3, 3), bool))
            if not hi.any():
                continue
            polys = [shapely.geometry.shape(g) for g, v in shapes(hi.astype("uint8"), mask=hi, transform=src.transform)]
            u = unary_union(polys).buffer(2.0).buffer(-2.0).simplify(1.0)
            found.append(shapely.transform(u, proj(to_ll)))
    s = unary_union(found) if found else shapely.Polygon()
    print(f"lidar structures over the water: {len(found)} tiles, area {s.area:.6f} deg2")
    return s


def rounded(g, nd=5):
    if g.geom_type == "Polygon":
        return Polygon([(round(x, nd), round(y, nd)) for x, y in g.exterior.coords],
                       [[(round(x, nd), round(y, nd)) for x, y in r.coords] for r in g.interiors])
    return g.__class__([rounded(p, nd) for p in g.geoms])


if __name__ == "__main__":
    wt, decks = build()
    # raw/decks.json is an input of make_terrain.py only (not served), so it stays out of data/
    dp = [decks] if decks.geom_type == "Polygon" else list(getattr(decks, "geoms", []))
    (ROOT / "raw" / "decks.json").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": mapping(rounded(p))} for p in dp if p.geom_type == "Polygon" and p.area > 2e-10]},
        separators=(",", ":")), encoding="utf-8")
    print(len(dp), "deck shapes ->", ROOT / "raw" / "decks.json")
    wt = rounded(wt)
    if not wt.is_valid:
        wt = wt.buffer(0)
    polys = [wt] if wt.geom_type == "Polygon" else list(wt.geoms)
    polys = [p for p in polys if p.area > 3e-9]       # slivers left where piers were cut out (under about 30 m2)
    print(wt.geom_type, len(polys), "polygons,", sum(len(p.interiors) for p in polys), "holes, area", round(wt.area, 5))
    out = ROOT / "data" / "water.json"
    out.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": mapping(p)} for p in polys]}, separators=(",", ":")), encoding="utf-8")
    print(out, round(out.stat().st_size / 1e6, 2), "MB")
