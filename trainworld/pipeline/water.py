"""Water mask for a city pack (T-030) and the scanline rasterizer it shares with block spreading
(T-013). Format of the shipped mask: notes/T-004.md, "Water mask".

Source (M1, US only): TIGER/Line 2020 AREAWATER for every county near the city, which covers
rivers, bays, sounds, reservoirs and the sea out to the state line (3 nautical miles), plus the
open sea beyond: the extent minus the TIGER 2020 state polygons (which run out to that line).
OSM's coastline-derived water polygons would generalise to the world but are a 906 MB global
download (osmdata.openstreetmap.de) with no way to fetch one region; see notes/T-030.md.

The grid is in the pack's local frame (metres east and north of the pack origin, equirectangular,
as x_m/y_m). A pixel is water when its centre is inside a water polygon.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import shapefile  # pyshp
import shapely
from shapely.geometry import box, shape
from shapely import make_valid, union_all

R_EARTH = 6371008.8
TIGER_AREAWATER = "https://www2.census.gov/geo/tiger/TIGER2020/AREAWATER/tl_2020_{fips}_areawater.zip"
TIGER_STATES = "https://www2.census.gov/geo/tiger/TIGER2020/STATE/tl_2020_us_state.zip"


# ---------------------------------------------------------------- frame

def to_local(geom, lon0, lat0):
    """lon/lat geometry -> metres east/north of (lon0, lat0), the pack's equirectangular frame."""
    k = np.cos(np.radians(lat0))

    def f(c):
        out = np.empty_like(c)
        out[:, 0] = R_EARTH * np.radians(c[:, 0] - lon0) * k
        out[:, 1] = R_EARTH * np.radians(c[:, 1] - lat0)
        return out
    return shapely.transform(geom, f)


def to_lonlat(x, y, lon0, lat0):
    k = np.cos(np.radians(lat0))
    return lon0 + np.degrees(x / (R_EARTH * k)), lat0 + np.degrees(y / R_EARTH)


# ---------------------------------------------------------------- rasterizer

def polygon_edges(geoms):
    """Every ring edge of a list of (multi)polygons, with the index of the geometry it came from.
    Returns x1, y1, x2, y2, gid (numpy arrays)."""
    parts = shapely.get_parts(np.asarray(geoms, dtype=object), return_index=True)
    polys, gidx = parts
    keep = shapely.get_type_id(polys) == 3  # Polygon
    polys, gidx = polys[keep], gidx[keep]
    rings, pidx = shapely.get_rings(polys, return_index=True)
    coords, ridx = shapely.get_coordinates(rings, return_index=True)
    same = ridx[1:] == ridx[:-1]  # consecutive vertices of one ring make an edge
    a, b = coords[:-1][same], coords[1:][same]
    gid = gidx[pidx[ridx[:-1][same]]]
    return a[:, 0], a[:, 1], b[:, 0], b[:, 1], gid


def scanline_runs(x1, y1, x2, y2, gid, x0, y0, cell, W, H):
    """Even-odd scanline fill per geometry id, sampled at pixel centres.

    Returns (gid, row, c0, c1): pixels [c0, c1) of `row` are inside geometry `gid`. Row 0 is the
    southernmost, column 0 the westernmost; pixel (c, r) has its centre at
    (x0 + (c + 0.5) cell, y0 + (r + 0.5) cell)."""
    ylo, yhi = np.minimum(y1, y2), np.maximum(y1, y2)
    # rows whose centre y_c satisfies ylo <= y_c < yhi (horizontal edges cross nothing)
    r_lo = np.clip(np.ceil((ylo - y0) / cell - 0.5), 0, H).astype(np.int64)
    r_hi = np.clip(np.ceil((yhi - y0) / cell - 0.5), 0, H).astype(np.int64)
    n = r_hi - r_lo
    sel = n > 0
    x1, y1, x2, y2, gid, r_lo, n = x1[sel], y1[sel], x2[sel], y2[sel], gid[sel], r_lo[sel], n[sel]
    total = int(n.sum())
    if total == 0:
        e = np.zeros(0, np.int64)
        return e, e, e, e
    idx = np.repeat(np.arange(len(n)), n)
    start = np.cumsum(n) - n
    row = r_lo[idx] + (np.arange(total) - start[idx])
    yc = y0 + (row + 0.5) * cell
    t = (yc - y1[idx]) / (y2[idx] - y1[idx])
    x = x1[idx] + t * (x2[idx] - x1[idx])
    # first pixel whose centre lies east of the crossing
    col = np.clip(np.floor((x - x0) / cell - 0.5) + 1, 0, W).astype(np.int64)
    g = gid[idx]
    order = np.lexsort((x, row, g))
    g, row, col = g[order], row[order], col[order]
    # crossings come in pairs within (geometry, row): inside between 1st and 2nd, 3rd and 4th...
    assert len(g) % 2 == 0
    a, b = slice(0, None, 2), slice(1, None, 2)
    assert np.all(g[a] == g[b]) and np.all(row[a] == row[b]), "unpaired crossing (ring not closed?)"
    g, row, c0, c1 = g[a], row[a], col[a], col[b]
    keep = c1 > c0
    return g[keep], row[keep], c0[keep], c1[keep]


def union_runs(row, c0, c1):
    """Union of possibly overlapping runs per row -> sorted, disjoint, non-touching runs."""
    if len(row) == 0:
        return row, c0, c1
    ev_row = np.concatenate([row, row])
    ev_col = np.concatenate([c0, c1])
    ev_d = np.concatenate([np.ones(len(row), np.int64), -np.ones(len(row), np.int64)])
    # at one column, process ends after starts so touching runs merge
    order = np.lexsort((-ev_d, ev_col, ev_row))
    ev_row, ev_col, ev_d = ev_row[order], ev_col[order], ev_d[order]
    cov = np.cumsum(ev_d)
    before = cov - ev_d
    starts = (before == 0) & (cov > 0)
    ends = (before > 0) & (cov == 0)
    r, s, e = ev_row[starts], ev_col[starts], ev_col[ends]
    assert len(s) == len(e) and np.all(ev_row[ends] == r)
    return r, s, e


# ---------------------------------------------------------------- sources

def read_shp(path, bbox_ll=None):
    """Shapes of a zipped shapefile as shapely geometries (lon/lat), optionally only those whose
    bounding box meets bbox_ll; returns (geoms, records)."""
    r = shapefile.Reader(path)
    fields = [f[0] for f in r.fields[1:]]
    geoms, recs = [], []
    for sr in r.iterShapeRecords(bbox=bbox_ll):
        if not sr.shape.points:
            continue
        geoms.append(make_valid(shape(sr.shape.__geo_interface__)))
        recs.append(dict(zip(fields, sr.record)))
    return geoms, recs


def water_geometries(bbox_ll, fetch, counties_zip):
    """Water polygons (lon/lat) over bbox_ll = (w, s, e, n): AREAWATER of every county within
    ~30 km of it (sounds and bays are split between counties whose land may be that far away),
    and the open sea outside every state. Returns (geoms, info)."""
    w, s, e, n = bbox_ll
    pad = 0.35
    cgeoms, crecs = read_shp(counties_zip, (w - pad, s - pad, e + pad, n + pad))
    near = box(w - pad, s - pad, e + pad, n + pad)
    fips = sorted(r["GEOID"] for g, r in zip(cgeoms, crecs) if g.intersects(near))
    ext = box(w, s, e, n)
    geoms = []
    for f in fips:
        gs, _ = read_shp(fetch(TIGER_AREAWATER.format(fips=f)), bbox_ll)
        geoms += [g for g in gs if g.intersects(ext)]
    n_area = len(geoms)
    sgeoms, srecs = read_shp(fetch(TIGER_STATES), bbox_ll)
    land_states = union_all([g.intersection(ext) for g in sgeoms])
    sea = ext.difference(land_states)
    if not sea.is_empty:
        geoms.append(sea)
    info = {"counties": fips, "areawater_polygons": n_area,
            "states": sorted(r["STUSPS"] for r in srecs), "open_sea_km2_deg": sea.area}
    return geoms, info


# ---------------------------------------------------------------- the mask

class WaterMask:
    """Row-run water mask: pixels [xs[2k], xs[2k+1]) of row r are water, for the pairs in
    xs[row_off[r]:row_off[r+1]] (sorted, disjoint, never touching)."""

    def __init__(self, x0, y0, cell, W, H, row_off, xs):
        self.x0, self.y0, self.cell, self.W, self.H = x0, y0, cell, W, H
        self.row_off, self.xs = row_off, xs

    @classmethod
    def from_runs(cls, x0, y0, cell, W, H, row, c0, c1):
        row, c0, c1 = union_runs(row, c0, c1)
        order = np.lexsort((c0, row))
        row, c0, c1 = row[order], c0[order], c1[order]
        counts = np.bincount(row, minlength=H) * 2
        row_off = np.zeros(H + 1, np.uint32)
        row_off[1:] = np.cumsum(counts)
        xs = np.empty(2 * len(row), np.uint16)
        xs[0::2], xs[1::2] = c0, c1
        return cls(x0, y0, cell, W, H, row_off, xs)

    @classmethod
    def from_dense(cls, x0, y0, cell, m):
        H, W = m.shape
        p = np.zeros((H, W + 2), np.int8)
        p[:, 1:-1] = m
        dr, dc = np.nonzero(np.diff(p, axis=1))  # row-major: per row, toggles in column order
        counts = np.bincount(dr, minlength=H)
        row_off = np.zeros(H + 1, np.uint32)
        row_off[1:] = np.cumsum(counts)
        return cls(x0, y0, cell, W, H, row_off, dc.astype(np.uint16))

    def without_narrow(self, px):
        """Drop water narrower than about px pixels (creeks, ditches, small ponds): a
        morphological opening with a px x px square. Never adds water."""
        from scipy.ndimage import binary_opening
        m = self.dense()
        o = binary_opening(m, structure=np.ones((px, px), bool))
        assert not np.any(o & ~m), "opening added water"
        return WaterMask.from_dense(self.x0, self.y0, self.cell, o)

    def is_water(self, x, y):
        """Vectorised point lookup in local metres; outside the grid is land."""
        x, y = np.atleast_1d(np.asarray(x, np.float64)), np.atleast_1d(np.asarray(y, np.float64))
        c = np.floor((x - self.x0) / self.cell).astype(np.int64)
        r = np.floor((y - self.y0) / self.cell).astype(np.int64)
        inside = (c >= 0) & (c < self.W) & (r >= 0) & (r < self.H)
        out = np.zeros(len(x), bool)
        for i in np.nonzero(inside)[0]:
            seg = self.xs[self.row_off[r[i]]:self.row_off[r[i] + 1]]
            out[i] = np.searchsorted(seg, c[i], side="right") % 2 == 1
        return out

    def dense(self, factor=1):
        """Boolean raster (H, W), row 0 south; with factor > 1, the water fraction of
        factor x factor blocks (float32, shape (H // factor, W // factor))."""
        d = np.zeros((self.H, self.W + 1), np.int8)
        r = np.repeat(np.arange(self.H), np.diff(self.row_off.astype(np.int64)) // 2)
        np.add.at(d, (r, self.xs[0::2].astype(np.int64)), 1)
        np.add.at(d, (r, self.xs[1::2].astype(np.int64)), -1)
        m = np.cumsum(d, axis=1, dtype=np.int8)[:, :self.W] > 0
        if factor == 1:
            return m
        h, w = self.H // factor, self.W // factor
        return m[:h * factor, :w * factor].reshape(h, factor, w, factor).mean(axis=(1, 3)).astype(np.float32)

    def water_pixels(self):
        return int((self.xs[1::2].astype(np.int64) - self.xs[0::2]).sum())

    def header(self, info):
        return {
            "cell_m": self.cell,
            "origin_m": [self.x0, self.y0],
            "size": [self.W, self.H],
            "encoding": "row runs: water_x[water_row[r]:water_row[r+1]] are pairs (start, end), "
                        "pixels start..end-1 of row r are water; row 0 south, column 0 west",
            "water_share": round(self.water_pixels() / (self.W * self.H), 4),
            **info,
        }


def build_mask(lon0, lat0, bbox_ll, cell, fetch, counties_zip, pad_m=2000.0):
    """Water mask over bbox_ll (plus pad_m), pixel size `cell` metres, grid snapped to `cell`."""
    w, s, e, n = bbox_ll
    k = np.cos(np.radians(lat0))
    xw, xe = R_EARTH * np.radians(w - lon0) * k - pad_m, R_EARTH * np.radians(e - lon0) * k + pad_m
    ys, yn = R_EARTH * np.radians(s - lat0) - pad_m, R_EARTH * np.radians(n - lat0) + pad_m
    x0, y0 = np.floor(xw / cell) * cell, np.floor(ys / cell) * cell
    W, H = int(np.ceil((xe - x0) / cell)), int(np.ceil((yn - y0) / cell))
    assert W < 65536, "columns must fit u16"
    # source polygons over the padded grid, in lon/lat
    gw, gs_ = to_lonlat(np.array([x0]), np.array([y0]), lon0, lat0)
    ge, gn = to_lonlat(np.array([x0 + W * cell]), np.array([y0 + H * cell]), lon0, lat0)
    geoms, info = water_geometries((gw[0], gs_[0], ge[0], gn[0]), fetch, counties_zip)
    local = [to_local(g, lon0, lat0) for g in geoms]
    x1, y1, x2, y2, gid = polygon_edges(local)
    _, row, c0, c1 = scanline_runs(x1, y1, x2, y2, gid, x0, y0, cell, W, H)
    info = {"source": "TIGER/Line 2020 AREAWATER per county + open sea outside TIGER 2020 states",
            "counties": len(info["counties"]), "areawater_polygons": info["areawater_polygons"]}
    return WaterMask.from_runs(float(x0), float(y0), float(cell), W, H, row, c0, c1), info
