"""Small water: which exact-0.06 pixels of the MAP surface are ponds to walk through.

MAP gives 0.06 min/m (1 km/h) to seas and lakes alike. A connected patch of those pixels with
an area under MAX_KM2 counts as land at 1 km/h (docs/resolution.md, "Small water"); build_tiles.py
and build_grid.py both use this, so every level agrees.

Patches are 8-connected, so a diagonal channel stays one body of water. They are found strip by
strip (STRIP rows, the same strips both builds read): each strip is labelled on its own, and
labels that touch across a strip edge, or across the 180 degree seam, are merged before any
area is summed. Usage:

    sw = SmallWater(ds)          # one pass over the tif, ~30 s
    ...
    small = sw.mask(strip_index, a)   # bool, True where a pixel of strip `strip_index` is small water
"""
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

STRIP = 600                 # rows per strip: 5 degrees, a multiple of every level's factor
WATER = np.float32(0.06)
MAX_KM2 = 5.0
R_KM = 6371.0088
PIX_DEG = 1 / 120           # the tif is 43200 x 17400 pixels from 85N to 60S
NORTH = 85.0
STRUCT = np.ones((3, 3), dtype=bool)


def row_area_km2(row0, rows):
    """Area of one pixel in each of rows row0.. (sphere, one pixel wide in longitude)."""
    top = np.radians(NORTH - (row0 + np.arange(rows)) * PIX_DEG)
    bot = np.radians(NORTH - (row0 + 1 + np.arange(rows)) * PIX_DEG)
    return R_KM ** 2 * np.radians(PIX_DEG) * (np.sin(top) - np.sin(bot))


def _label(a):
    lab, n = ndimage.label(a == WATER, structure=STRUCT, output=np.int32)
    return lab, n


def _pairs(p, q):
    """Distinct (label in p, label in q) where both are water; arrays of equal shape."""
    m = (p > 0) & (q > 0)
    return np.unique(np.stack([p[m], q[m]]), axis=1)


class SmallWater:
    def __init__(self, ds, max_km2=MAX_KM2, log=print):
        w, h = ds.width, ds.height
        assert h % STRIP == 0
        self.strips = h // STRIP
        offsets, area, edges = [], [], []   # first global label of each strip, km2 per label, touching pairs
        total = 0
        prev_bottom = None
        for i in range(self.strips):
            a = ds.read(1, window=((i * STRIP, (i + 1) * STRIP), (0, w)))
            lab, n = _label(a)
            ra = row_area_km2(i * STRIP, STRIP)
            wts = np.broadcast_to(ra[:, None], lab.shape)
            area.append(np.bincount(lab.ravel(), weights=wts.ravel(), minlength=n + 1)[1:])
            gl = np.where(lab > 0, lab + total, 0).astype(np.int64)
            offsets.append(total)
            total += n
            # the 180 degree seam: last column against the first, same row and the rows either side
            last, first = gl[:, -1], gl[:, 0]
            edges += [_pairs(last, first), _pairs(last[:-1], first[1:]), _pairs(last[1:], first[:-1])]
            # the strip edge: bottom row of the strip above against the top row of this one
            if prev_bottom is not None:
                for d in (-1, 0, 1):
                    edges.append(_pairs(prev_bottom, np.roll(gl[0], d)))
            prev_bottom = gl[-1]
            log("labelled strip %d / %d: %d patches" % (i + 1, self.strips, n))
        area = np.concatenate(area)
        e = np.concatenate([x for x in edges if x.size] or [np.zeros((2, 0), np.int64)], axis=1)
        e = e - 1                                   # global labels start at 1
        g = coo_matrix((np.ones(e.shape[1], np.int8), (e[0], e[1])), shape=(total, total))
        ncomp, comp = connected_components(g, directed=False)
        comp_area = np.bincount(comp, weights=area, minlength=ncomp)
        small = comp_area < max_km2
        self._small_by_label = small[comp]          # per global label, index label - 1
        self._offsets = offsets
        self.n_patches = total                      # per-strip patches, before merging
        self.n_components = ncomp
        self.comp = comp
        self.comp_area = comp_area
        self.small_components = int(small.sum())
        self.small_area_km2 = float(comp_area[small].sum())

    def mask(self, i, a):
        """True where pixel is exact-0.06 water in a patch under the area limit (strip i, array a)."""
        lab, n = _label(a)
        lut = np.zeros(n + 1, dtype=bool)
        o = self._offsets[i]
        lut[1:] = self._small_by_label[o:o + n]
        return lut[lab]
