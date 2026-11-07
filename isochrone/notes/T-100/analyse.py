"""T-100 / T-101: RiverATLAS reaches against Hydrography90m (MERIT Hydro) in the sample boxes.

For each paddleable reach k with its midpoint in a box: the stretch from k's midpoint to the
midpoint of the first downstream reach j at least WIN km on (as rivers.py dem_gradient's window).
Both midpoints snapped to the highest-accumulation MERIT cell within SNAP px; kept when both
are in one basin, B is nearer the outlet, and Q / upstream area at both ends is within 4x of the
box's median (else the snap found another river). Per pair: RiverATLAS km, MERIT km (difference
of flow distance to the outlet), MERIT drop (difference of height above the outlet), GMTED drop
on the same two points (3 x 3 minimum, as dem_gradient), and the reach's built gradient.

    python analyse.py            -> pairs.npz + tables
"""
import os
import sys
import glob

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

ISO = r"C:\Users\anita\projects\maps\isochrone"
WORK = os.path.join(ISO, "data", "raw", "rivers_work")
BOXD = os.path.join(ISO, "data", "raw", "hydrography90m")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ISO, "tools", "canoe"))
sys.path.insert(0, os.path.join(ISO, "tools"))
import rivers  # noqa: E402

WIN = 10.0
SNAP = 4


def load_reaches():
    r = dict(np.load(os.path.join(WORK, "reaches.npz")))
    dz = np.load(os.path.join(WORK, "desert.npz"))
    r["q"] = np.where(dz["in_hot"], dz["q_in"], r["q"])
    r = rivers.keep_reaches(r, dz["ok"])
    grad = np.load(os.path.join(WORK, "gradient_gmted.npy"))     # GMTED over RiverATLAS km, as before T-100
    assert len(grad) == len(r["q"])
    return r, grad


def midpoints(co, off, ks):
    out = np.empty((len(ks), 2))
    for i, k in enumerate(ks):
        p = co[off[k]:off[k + 1]].astype(np.float64)
        if len(p) == 1:
            out[i] = p[0]
            continue
        d = np.hypot(np.diff(p[:, 0]) * np.cos(np.radians(p[1:, 1])), np.diff(p[:, 1]))
        c = np.concatenate([[0], np.cumsum(d)])
        h = c[-1] / 2
        j = min(np.searchsorted(c, h), len(p) - 1)
        j = max(j, 1)
        t = (h - c[j - 1]) / max(c[j] - c[j - 1], 1e-12)
        out[i] = p[j - 1] + t * (p[j] - p[j - 1])
    return out


def gmted():
    import rasterio
    from scipy.ndimage import minimum_filter
    with rasterio.open(os.path.join(ISO, "data", "surface", "elev.tif")) as ds:
        z = ds.read(1).astype(np.float32)
    z[z == -32768] = 1e4
    return minimum_filter(z, size=3, mode="wrap")


def main():
    r, grad = load_reaches()
    co, off, L, q = r["co"], r["off"], r["length_km"], r["q"]
    down = rivers.reach_down(r)
    st = co[off[:-1]]
    print("reaches", len(q), flush=True)
    Z = gmted()
    print("gmted read", flush=True)
    rows = []
    for path in sorted(glob.glob(os.path.join(BOXD, "box_*.npz"))):
        name = os.path.basename(path)[4:-4]
        b = np.load(path)
        c0, r0 = int(b["c0"]), int(b["r0"])
        diff, dist, acc, basin = b["diff"], b["dist"], b["acc"], b["basin"]
        H, W = diff.shape
        w, n = c0 / 1200 - 180, 85 - r0 / 1200
        e, s = w + W / 1200, n - H / 1200
        m = (st[:, 0] > w - 0.5) & (st[:, 0] < e + 0.5) & (st[:, 1] > s - 0.5) & (st[:, 1] < n + 0.5)
        ks = np.nonzero(m)[0]
        mid = midpoints(co, off, ks)
        inb = (mid[:, 0] > w) & (mid[:, 0] < e) & (mid[:, 1] > s) & (mid[:, 1] < n)
        ks, mid = ks[inb], mid[inb]
        midof = dict(zip(ks.tolist(), range(len(ks))))
        # snap every midpoint
        cc = np.floor((mid[:, 0] + 180) * 1200).astype(np.int64) - c0
        rr = np.floor((85 - mid[:, 1]) * 1200).astype(np.int64) - r0
        ok = (cc >= SNAP) & (cc < W - SNAP) & (rr >= SNAP) & (rr < H - SNAP)
        sv = {k: np.full(len(ks), np.nan) for k in ("diff", "dist", "acc", "basin")}
        offs = np.arange(-SNAP, SNAP + 1)
        for i in np.nonzero(ok)[0]:
            a = acc[rr[i] + offs[:, None], cc[i] + offs[None, :]]
            a = np.where(a > -1, a, -1)
            j = np.argmax(a)
            y, x = rr[i] + j // len(offs) - SNAP, cc[i] + j % len(offs) - SNAP
            if diff[y, x] == -9999 or dist[y, x] == -9999:
                continue
            sv["diff"][i], sv["dist"][i], sv["acc"][i], sv["basin"][i] = diff[y, x], dist[y, x], acc[y, x], basin[y, x]
        qa = q[ks] / np.where(sv["acc"] > 0, sv["acc"], np.nan)
        plaus = np.isfinite(qa) & (qa > 3e-4) & (qa < 0.3)     # m3/s per km2: 10 to 9,500 mm a year
        med = np.median(qa[plaus])
        good = np.isfinite(qa) & (qa > med / 4) & (qa < med * 4)
        # GMTED at the midpoints
        gc = np.floor((mid[:, 0] + 180) * 120).astype(np.int64) % 43200
        gr = np.floor((85 - mid[:, 1]) * 120).astype(np.int64)
        zg = Z[gr, gc]
        npair = 0
        for i, k in enumerate(ks.tolist()):
            if not good[i]:
                continue
            km, j = L[k] / 2, down[k]
            while j >= 0 and km + L[j] / 2 < WIN:
                km += L[j]
                j = down[j]
            if j < 0 or j not in midof:
                continue
            jj = midof[j]
            if not good[jj] or sv["basin"][i] != sv["basin"][jj]:
                continue
            km += L[j] / 2
            mkm = (sv["dist"][i] - sv["dist"][jj]) / 1000
            if mkm <= 0:
                continue
            rows.append((name, k, q[k], km, mkm, sv["diff"][i] - sv["diff"][jj], zg[i] - zg[jj], grad[k], L[k],
                         mid[i, 0], mid[i, 1]))
            npair += 1
        print("%-20s reaches %6d, snapped %6d, Q/area median %.4f, pairs %6d" % (name, len(ks), good.sum(), med, npair), flush=True)
    names = np.array([x[0] for x in rows])
    arr = np.array([x[1:] for x in rows], np.float64)
    np.savez(os.path.join(HERE, "pairs.npz"), box=names, k=arr[:, 0].astype(np.int64), q=arr[:, 1], ra_km=arr[:, 2],
             merit_km=arr[:, 3], merit_drop=arr[:, 4], gmted_drop=arr[:, 5], grad=arr[:, 6], reach_km=arr[:, 7],
             lon=arr[:, 8], lat=arr[:, 9])
    print("pairs", len(rows))


if __name__ == "__main__":
    main()
