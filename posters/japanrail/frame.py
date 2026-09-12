"""
Projection and framing for the japanrail poster.

Japan is a 1,900 km archipelago running NE–SW, so drawn north-up it needs a
near-square sheet of which most is ocean. Rotating the whole map clockwise lays
the archipelago flat: measured on the rail network's own vertices, the bounding
box shrinks from 1382 x 1560 km at 0 deg to 1879 x 664 km at 51 deg — 41% less
paper for the same content, or 30% more scale on the same paper.

The projection is a Lambert conformal conic for Japan followed by a plain 2D
rotation in projected metres. Conformal rather than equal-area: nothing on this
map is area-proportional (line width is throughput, bubble area is ridership),
while coastline shape and the angles between lines are what a reader navigates
by. Rotating in projected metres rather than in the projection definition keeps
the transform trivially invertible, so insets and locator boxes can be placed in
sheet inches without a second projection.

Positive rotation is CLOCKWISE, which puts Kyushu on the left and Hokkaido on
the right — west-to-east reading order — with north pointing up and to the
right, the Sea of Japan along the top and the Pacific along the bottom.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer

# Lambert conformal conic. Standard parallels bracket the populated latitudes
# (Kagoshima 31.6N .. Wakkanai 45.4N), so scale error stays under about 0.5%
# anywhere on the sheet — well inside the width of a printed scale bar.
PROJ = CRS.from_proj4("+proj=lcc +lat_1=30 +lat_2=44 +lat_0=37 +lon_0=137 "
                      "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

_TF = Transformer.from_crs("EPSG:4326", PROJ, always_xy=True)

# Rail data bounds, from the segments file: lon 127.65..145.60, lat 26.19..45.42.
# Okinawa (the 13 km monorail at 26.2N, 127.7E) is 1,000 km off the southwest
# end and stretches the rotated box from 1,879 km wide to 2,491 km. It gets a
# relocated float instead, the way ancestrydots handles Hawaii.
OKINAWA_MAX_LAT = 28.0

# Chosen by eye from the candidate sweep, not by the bbox-area minimum (51 deg).
# See posters/NOTES.md.
DEFAULT_ROT = 25.0


def project(lon, lat, rot_deg):
    """lon/lat -> rotated projected metres. Accepts scalars or arrays."""
    x, y = _TF.transform(np.asarray(lon, float), np.asarray(lat, float))
    return rotate(x, y, rot_deg)


def rotate(x, y, rot_deg):
    """Rotate projected metres clockwise about the projection origin."""
    a = math.radians(rot_deg)
    c, s = math.cos(a), math.sin(a)
    return x * c + y * s, -x * s + y * c


def unrotate(rx, ry, rot_deg):
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

    def lonlat_bbox(self, pad_deg=0.5):
        """A lon/lat bbox that certainly covers this frame, for bbox-filtered
        shapefile reads. The frame is a rotated rectangle, so its lon/lat hull
        is bigger than any corner-only estimate — sample the whole boundary."""
        n = 200
        ex = np.concatenate([
            np.linspace(self.x0, self.x1, n), np.linspace(self.x0, self.x1, n),
            np.full(n, self.x0), np.full(n, self.x1)])
        ey = np.concatenate([
            np.full(n, self.y0), np.full(n, self.y1),
            np.linspace(self.y0, self.y1, n), np.linspace(self.y0, self.y1, n)])
        px, py = unrotate(ex, ey, self.rot)
        back = Transformer.from_crs(PROJ, "EPSG:4326", always_xy=True)
        lon, lat = back.transform(px, py)
        return (lon.min() - pad_deg, lat.min() - pad_deg,
                lon.max() + pad_deg, lat.max() + pad_deg)

    def to_dict(self):
        return {"rot": self.rot, "x0": self.x0, "x1": self.x1,
                "y0": self.y0, "y1": self.y1, "width_in": round(self.width_in, 4),
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
