"""The hot desert mask (docs/hazard.md 2.1), one raster shared by the land hazard's drinkable water
and the canoe's paddleable rivers (docs/canoe.md 4), so both layers agree on where a wadi is dry.

    python isochrone/tools/hazard/build.py hot     # CHELSA -> data/raw/hazard_work/hot_g2.npy   (~1 min)

The raster is the hazard's 2 km distance grid (1/60 degree from 180 W, 85 N, 21600 x 8700): a cell is
hot desert when at least 2 of its 4 L1 pixels have annual rain under HOT_DESERT_MM and the warmer of
the January and July day highs at least HOT_DESERT_C.

There WaterGAP's own discharge is not trusted: it gives dry catchments a mean flow of up to hundreds of
m3/s, often with a driest month to match (Wadi ad-Dawasir 292 m3/s, 98 in its driest month; T-079).
A river in hot desert carries only the water that comes in from upstream outside it (`desert_rivers`):
the mean and driest-month flow of its reaches outside the desert, scaled down where WaterGAP's mean flow
falls, and losing LOSS_M_DAY over its wetted width on every km. It is a river while that is at least
Q_RIVER mean and Q_PERENNIAL in the driest month: the Nile, Niger, Euphrates, Indus, Amu Darya.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(HERE, "data", "raw", "hazard_work", "hot_g2.npy")

HOT_DESERT_MM, HOT_DESERT_C = 250.0, 25.0
Q_RIVER = 1.0                  # m3/s mean: a desert river (the canoe's paddleable threshold)
Q_PERENNIAL = 0.1              # m3/s in the driest month (RiverATLAS dis_m3_pmn)
LOSS_M_DAY = 0.03              # water lost over a desert river's wetted bed, m a day (evaporation and seepage)
W_COEF = 7.2                   # wetted width W = 7.2 Q^0.5 m (Moody and Troutman 2002, docs/canoe.md 2)
HOT_SHARE = 0.5                # a reach is in hot desert when this share of its length is

NORTH, L1W, L1H = 85.0, 43200, 17400
G = 2                          # L1 pixels per cell
GW, GH = L1W // G, L1H // G


def build(read, strip=600, log=print):
    """read(var, month, r0, n) -> CHELSA rows on the L1 grid (tools/hazard/build.py chelsa_reader)."""
    hot = np.zeros((GH, GW), bool)
    for r0 in range(0, L1H, strip):
        n = min(strip, L1H - r0)
        pr = np.nan_to_num(read("bio12", None, r0, n), nan=1000.0)
        tx = np.nan_to_num(np.fmax(read("tasmax", "01", r0, n), read("tasmax", "07", r0, n)), nan=0.0)
        px = (pr < HOT_DESERT_MM) & (tx >= HOT_DESERT_C)
        hot[r0 // G:(r0 + n) // G] = px.reshape(n // G, G, GW, G).sum(axis=(1, 3)) >= 2
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    tmp = PATH + ".tmp.npy"
    np.save(tmp, hot)
    os.replace(tmp, PATH)
    log("hot desert cells", int(hot.sum()), "of", hot.size)
    return hot


def load():
    if not os.path.exists(PATH):
        raise SystemExit("no %s: run python isochrone/tools/hazard/build.py hot" % PATH)
    return np.load(PATH, mmap_mode="r")


def reach_share(hot, co, off):
    """Share of each line's length (vertices co, offsets off) in hot desert, by segment midpoints."""
    R = len(off) - 1
    seg_ok = np.ones(len(co) - 1, bool)
    seg_ok[off[1:-1] - 1] = False
    s0, s1 = co[:-1][seg_ok], co[1:][seg_ok]
    seg_r = np.repeat(np.arange(R), np.diff(off) - 1)
    mid = (s0 + s1) / 2
    km = np.hypot((s1[:, 0] - s0[:, 0]) * np.cos(np.radians(mid[:, 1])), s1[:, 1] - s0[:, 1])
    tot = np.bincount(seg_r, weights=km, minlength=R)
    inh = np.bincount(seg_r, weights=km * at(hot, mid[:, 0], mid[:, 1]), minlength=R)
    first = at(hot, co[off[:-1], 0], co[off[:-1], 1])        # a one-point line: its point
    return np.where(tot > 0, inh / np.where(tot > 0, tot, 1), first.astype(np.float64))


def desert_rivers(share, q, qmin, length_km, down, order):
    """Per reach (the Q >= 1 network, down = index of the next reach or -1, order = upstream first):
    in_hot, the mean and driest-month flow carried in from outside the hot desert (q and qmin outside
    it), and ok = a river (always, outside hot desert)."""
    R = len(q)
    hot = (share >= HOT_SHARE).tolist()
    half_a = LOSS_M_DAY / 86400.0 * W_COEF * 1000.0 / 2      # d sqrt(Q) per km, from dQ = -LOSS x W per m
    ql, qml, ll, dl = q.tolist(), qmin.tolist(), length_km.tolist(), down.tolist()
    uq, ue, um = [0.0] * R, [0.0] * R, [0.0] * R
    qe, qme = [0.0] * R, [0.0] * R
    for k in order.tolist():
        if not hot[k]:
            a, b = ql[k], qml[k]
        else:
            f = min(1.0, ql[k] / uq[k]) if uq[k] > 0 else 0.0    # WaterGAP's own losses, never its gains
            drop = half_a * ll[k]
            a = max(0.0, (ue[k] * f) ** 0.5 - drop) ** 2
            b = min(a, max(0.0, (um[k] * f) ** 0.5 - drop) ** 2)
        qe[k], qme[k] = a, b
        d = dl[k]
        if d >= 0:
            uq[d] += ql[k]
            ue[d] += a
            um[d] += b
    in_hot = np.array(hot)
    qe, qme = np.array(qe), np.array(qme)
    ok = ~in_hot | ((qe >= Q_RIVER) & (qme >= Q_PERENNIAL))
    return in_hot, qe, qme, ok


def at(hot, lon, lat):
    """The mask at points (degrees); False north of 85 N or south of the grid."""
    c = np.floor((np.asarray(lon) + 180.0) * 60).astype(np.int64) % GW
    r = np.floor((NORTH - np.asarray(lat)) * 60).astype(np.int64)
    ok = (r >= 0) & (r < GH)
    out = np.zeros(len(c), bool)
    out[ok] = np.asarray(hot[r[ok], c[ok]])
    return out
