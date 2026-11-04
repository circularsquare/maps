"""T-1140 shared helpers: loading a country's data, shapes, compactness measures, unit flows (ACS or
gravity), zone evaluation."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import collections
import math
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TW = r"C:\Users\anita\projects\maps\trainworld"
sys.path.insert(0, TW + r"\pipeline")
import numpy as np  # noqa: E402
import shapely  # noqa: E402
from shapely import union_all  # noqa: E402

import world_plan as wp  # noqa: E402


def load(name):
    with open(os.path.join(HERE, name), "rb") as f:
        return pickle.load(f)


class Shapes:
    def __init__(self, units):
        self.units = units
        self.simple = {}
        self.cache = {}

    def unit(self, i):
        if i not in self.simple:
            self.simple[i] = wp.polygonal(self.units[i].geom.simplify(500))
        return self.simple[i]

    def zone(self, ids):
        k = frozenset(ids)
        if k not in self.cache:
            self.cache[k] = union_all([self.unit(i) for i in ids])
        return self.cache[k]


def hull_ratio(g):
    h = g.convex_hull.area
    return g.area / h if h else 1.0


def polsby(g):
    return 4 * math.pi * g.area / g.length ** 2 if g.length else 1.0


def reock(g):
    c = shapely.minimum_bounding_circle(g)
    return g.area / c.area if c.area else 1.0


def span_ratio(g):
    """The equivalent circle's diameter over the shape's longest span (1 for a disc)."""
    c = shapely.minimum_bounding_radius(g)
    return math.sqrt(g.area / math.pi) / c if c else 1.0


def gravity_flows(cells, ids, km=20, reach_km=200):
    """{unit: {unit: flow}} between distinct units: people x people x the zone curve over the units'
    people cells (aggregated to `km`), pairs within `reach_km`. Symmetric."""
    from scipy.spatial import cKDTree
    xs, ys, ps, us = [], [], [], []
    k = km * 1000.0
    for n, i in enumerate(ids):
        x, y, p = cells[i]
        if not len(p):
            continue
        gx, gy = np.floor(x / k), np.floor(y / k)
        key = gx * 1e6 + gy
        u, inv = np.unique(key, return_inverse=True)
        s = np.bincount(inv, weights=p)
        sx = np.bincount(inv, weights=x * p) / s
        sy = np.bincount(inv, weights=y * p) / s
        xs.append(sx); ys.append(sy); ps.append(s); us.append(np.full(len(s), n))
    x, y, p, u = (np.concatenate(a) for a in (xs, ys, ps, us))
    t = cKDTree(np.c_[x, y])
    sm = t.sparse_distance_matrix(t, reach_km * 1000.0, output_type="coo_matrix")
    a, b, d = sm.row, sm.col, sm.data
    m = u[a] != u[b]
    a, b, d = a[m], b[m], d[m]
    dk = np.maximum(d / 1000.0, 2.0)
    w = p[a] * p[b] * dk ** -0.8 * np.exp(-dk / 30.0)
    key = u[a].astype(np.int64) * len(ids) + u[b]
    uk, inv = np.unique(key, return_inverse=True)
    s = np.bincount(inv, weights=w)
    out = collections.defaultdict(dict)
    for kk, v in zip(uk, s):
        i, j = ids[kk // len(ids)], ids[kk % len(ids)]
        out[i][j] = float(v)
    return out


def acs_sym(flows, ids):
    """ACS home-to-work flows made symmetric (both directions summed), distinct units only."""
    ids = set(ids)
    out = collections.defaultdict(dict)
    for h, row in flows.items():
        if h not in ids:
            continue
        for w, n in row.items():
            if w == h or w not in ids or not n:
                continue
            out[h][w] = out[h].get(w, 0.0) + n
            out[w][h] = out[w].get(h, 0.0) + n
    return out


def area_flows(F, ids, owner):
    """{area: flow} from units `ids` to every other area (owner: unit -> area id)."""
    mine = set(ids)
    to = collections.Counter()
    for i in ids:
        for j, n in F.get(i, {}).items():
            if j in mine:
                continue
            a = owner.get(j)
            if a is not None:
                to[a] += n
    return to
