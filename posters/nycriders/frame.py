"""
Projection and framing for the nycriders poster.

The subway is a tall, thin thing: 23.3 km wide by 36.4 km high measured on the
network's own vertices, north-up. Unlike japanrail there is nothing to gain by
tilting the sheet — a rotation sweep bottoms out at 10 deg and saves 5% of the
bounding box, which is not worth giving up a north-up map that shares its
orientation with every other map of New York a reader has seen. `rot` is kept in
the API anyway so the machinery matches japanrail's and a tilt stays one flag
away.

The projection is NAD83 / New York Long Island (EPSG:32118), the state plane
zone the city's own data is published in. It is a Lambert conformal conic
centred on 40deg 10' N / 74deg W, so over the ~40 km the sheet covers, scale
error is far below a printed hairline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer

PROJ = CRS.from_epsg(32118)          # NAD83 / New York Long Island, metres
_TF = Transformer.from_crs("EPSG:4326", PROJ, always_xy=True)
_BACK = Transformer.from_crs(PROJ, "EPSG:4326", always_xy=True)

DEFAULT_ROT = 0.0


def project(lon, lat, rot_deg=0.0):
    """lon/lat -> rotated projected metres. Accepts scalars or arrays."""
    x, y = _TF.transform(np.asarray(lon, float), np.asarray(lat, float))
    return rotate(x, y, rot_deg)


def rotate(x, y, rot_deg):
    """Rotate projected metres clockwise about the projection origin."""
    if not rot_deg:
        return x, y
    a = math.radians(rot_deg)
    c, s = math.cos(a), math.sin(a)
    return x * c + y * s, -x * s + y * c


def unrotate(rx, ry, rot_deg):
    if not rot_deg:
        return rx, ry
    a = math.radians(rot_deg)
    c, s = math.cos(a), math.sin(a)
    return rx * c - ry * s, rx * s + ry * c


@dataclass
class Frame:
    """A rectangle of rotated projected metres, and the sheet it prints on."""

    rot: float
    x0: float
    x1: float
    y0: float
    y1: float
    width_in: float
    dpi: int = 300

    # ---- derived -------------------------------------------------------
    @property
    def height_in(self):
        return self.width_in * (self.y1 - self.y0) / (self.x1 - self.x0)

    @property
    def km_per_inch(self):
        return (self.x1 - self.x0) / 1000.0 / self.width_in

    @property
    def m_per_inch(self):
        return (self.x1 - self.x0) / self.width_in

    def size_px(self, scale=1.0):
        w = int(round(self.width_in * self.dpi * scale))
        h = int(round(w * (self.y1 - self.y0) / (self.x1 - self.x0)))
        return w, h

    # ---- mapping -------------------------------------------------------
    def to_inches(self, lon, lat):
        """lon/lat -> inches from the top-left of the map area."""
        rx, ry = project(lon, lat, self.rot)
        return ((rx - self.x0) / self.m_per_inch,
                (self.y1 - ry) / self.m_per_inch)

    def to_px(self, lon, lat, scale=1.0):
        ix, iy = self.to_inches(lon, lat)
        return ix * self.dpi * scale, iy * self.dpi * scale

    def extent(self):
        """matplotlib (x0, x1, y0, y1) in rotated metres."""
        return self.x0, self.x1, self.y0, self.y1

    def lonlat_bbox(self, pad_deg=0.15):
        """A lon/lat bbox that certainly covers this frame, for bbox-filtered
        shapefile reads. A rotated frame's lon/lat hull is bigger than any
        corner-only estimate, so sample the whole boundary."""
        n = 200
        ex = np.concatenate([
            np.linspace(self.x0, self.x1, n), np.linspace(self.x0, self.x1, n),
            np.full(n, self.x0), np.full(n, self.x1)])
        ey = np.concatenate([
            np.full(n, self.y0), np.full(n, self.y1),
            np.linspace(self.y0, self.y1, n), np.linspace(self.y0, self.y1, n)])
        px, py = unrotate(ex, ey, self.rot)
        lon, lat = _BACK.transform(px, py)
        return (lon.min() - pad_deg, lat.min() - pad_deg,
                lon.max() + pad_deg, lat.max() + pad_deg)

    def to_dict(self):
        return {"rot": self.rot, "x0": self.x0, "x1": self.x1,
                "y0": self.y0, "y1": self.y1,
                "width_in": round(self.width_in, 4),
                "height_in": round(self.height_in, 4), "dpi": self.dpi,
                "km_per_inch": round(self.km_per_inch, 3)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["rot"], d["x0"], d["x1"], d["y0"], d["y1"],
                   d["width_in"], d.get("dpi", 300))


def fit(lon, lat, rot, width_in, pad_km=(0, 0, 0, 0), dpi=300):
    """Frame that holds every lon/lat given, plus a per-side margin.

    pad_km is (west, east, south, north) *in the rotated frame*, so "west" is
    the left edge of the printed sheet, not compass west.
    """
    rx, ry = project(lon, lat, rot)
    w, e, s, n = (p * 1000.0 for p in pad_km)
    return Frame(rot, rx.min() - w, rx.max() + e, ry.min() - s, ry.max() + n,
                 width_in, dpi)


def fit_sheet(lon, lat, rot, width_in, height_in, pad_km=0.0, dpi=300):
    """Frame of exactly width_in x height_in that holds every lon/lat given,
    centred, with at least pad_km of slack on the binding axis.

    The sheet size is the thing an order form asks for, so it is the input;
    whichever axis is tight sets the scale and the other gets the slack. That
    slack is not waste — on this map it is New Jersey, the Sound and the ocean,
    which is where the title and legend go.
    """
    rx, ry = project(lon, lat, rot)
    pad = pad_km * 1000.0
    cw, ch = (rx.max() - rx.min()) + 2 * pad, (ry.max() - ry.min()) + 2 * pad
    m_per_in = max(cw / width_in, ch / height_in)
    cx, cy = (rx.min() + rx.max()) / 2, (ry.min() + ry.max()) / 2
    hw, hh = m_per_in * width_in / 2, m_per_in * height_in / 2
    return Frame(rot, cx - hw, cx + hw, cy - hh, cy + hh, width_in, dpi)
