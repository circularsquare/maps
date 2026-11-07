"""MERIT Hydro along the canoe's rivers, read from Hydrography90m (T-100, T-101; docs/canoe.md 4).

    python tools/canoe/h90.py fetch [--reverse]   # tropical forest tiles -> data/raw/hydrography90m/tiles/
                                         #   ~3 min a tile, 132 tiles, ~32 GB; run two at once (one --reverse);
                                         #   after a crash delete tiles/*.claim before starting again
    python tools/canoe/h90.py reaches    # per reach: MERIT drop and length over the gradient window -> work/merit.npz   ~10 min

Hydrography90m (Amatulli et al. 2022) is MERIT Hydro's network with its layers worked out per
cell; its public Nextcloud share serves global cloud-optimised GeoTIFFs that GDAL reads by HTTP
range, so only the tiles used are fetched. Layers: outlet_diff_dw_basin (m above the basin
outlet along the flow path: MERIT Hydro's hydrologically adjusted elevation, canopy removed),
outlet_dist_dw_basin (m to the outlet along the flow path), accumulation (km2), basin (id).

Tiles are the 5-degree speed tiles with at least FOREST_SHARE of their land in LC100 closed
evergreen broadleaf forest (data/surface/lc_class.tif class 112): where GMTED's canopy reads as
drop (notes/T-100.md). Elsewhere GMTED and MERIT agree on the gradient.
"""
import json
import os
import shutil
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(HERE, "data", "raw")
WORK = os.path.join(RAW, "rivers_work")
TILES = os.path.join(RAW, "hydrography90m", "tiles")
URL = "/vsicurl/https://public.igb-berlin.de/public.php/webdav/global/"
# the share's public token, as Hydrography90m's own download scripts use it (no account)
ENV = dict(GDAL_HTTP_USERPWD="agciopgzXjWswF4:", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
           CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MULTIRANGE="YES",
           GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", GDAL_HTTP_MAX_RETRY="8", GDAL_HTTP_RETRY_DELAY="5",
           GDAL_CACHEMAX=256)
LAYERS = {"diff": "outlet_diff_dw_basin_ovr.tif", "dist": "outlet_dist_dw_basin_ovr.tif",
          "acc": "accumulation_ovr.tif", "basin": "basin_ovr.tif"}
PX = 1200                  # MERIT pixels per degree
PAD = 8                    # pixels read round each tile, so a snap near its edge stays inside
FOREST_SHARE = 0.2
MIN_FREE_GB = 45           # Anita's floor is 40 GB free on C:
SNAP = 4                   # pixels: a point takes the highest-accumulation cell within this
WIN_KM = 10.0              # as rivers.py WIN_KM
QA_SPREAD = 4.0            # a snap is kept when Q / upstream area is within this factor of the tile's median


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def forest_tiles():
    import rasterio
    with rasterio.open(os.path.join(HERE, "data", "surface", "lc_class.tif")) as ds:
        lc = ds.read(1)
    T = 600
    out = []
    for tr in range(lc.shape[0] // T):
        for tc in range(lc.shape[1] // T):
            b = lc[tr * T:(tr + 1) * T, tc * T:(tc + 1) * T]
            land = (b != 0) & (b != 80) & (b != 200)
            n = int(land.sum())
            if n >= 0.05 * T * T and (b == 112).sum() >= FOREST_SHARE * n:
                out.append((tr, tc))
    return out


def stage_fetch():
    import rasterio
    from rasterio.windows import Window
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(TILES, exist_ok=True)
    tiles = forest_tiles()
    with open(os.path.join(TILES, "tiles.json"), "w") as fh:
        json.dump(tiles, fh)
    log("%d forest tiles" % len(tiles))

    def read(name, win):
        with rasterio.Env(**ENV):
            with rasterio.open(URL + LAYERS[name]) as ds:
                return ds.read(1, window=win)     # forest tiles never touch 180 degrees

    # several fetchers may run at once (the share serves ~1.3 MB/s a connection): each claims a tile
    order = list(enumerate(tiles))
    if "--reverse" in sys.argv:
        order.reverse()
    for i, (tr, tc) in order:
        dst = os.path.join(TILES, "%d_%d.npz" % (tr, tc))
        if os.path.exists(dst):
            continue
        try:
            os.close(os.open(dst + ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        free = shutil.disk_usage(TILES).free / 1e9
        if free < MIN_FREE_GB:
            raise SystemExit("stopping: %.1f GB free on the drive" % free)
        c0, r0 = tc * 5 * PX - PAD, tr * 5 * PX - PAD
        win = Window(c0, r0, 5 * PX + 2 * PAD, 5 * PX + 2 * PAD)
        t0 = time.time()
        for attempt in range(4):
            try:
                with ThreadPoolExecutor(4) as ex:
                    res = dict(zip(LAYERS, ex.map(lambda k: read(k, win), LAYERS)))
                break
            except Exception as e:          # the share drops a connection now and then
                log("retry", tr, tc, repr(e)[:200])
                time.sleep(30)
        else:
            raise SystemExit("tile %d_%d failed" % (tr, tc))
        tmp = dst[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, c0=c0, r0=r0, **res)
        os.replace(tmp, dst)
        os.remove(dst + ".claim")
        log("%d/%d %d_%d in %.0f s, %.0f MB, %.0f GB free" % (i + 1, len(tiles), tr, tc, time.time() - t0,
                                                              os.path.getsize(dst) / 1e6, free))


def midpoints(co, off):
    """The point halfway along each reach's line."""
    n = len(off) - 1
    p0, p1 = co[:-1].astype(np.float64), co[1:].astype(np.float64)
    d = np.hypot((p1[:, 0] - p0[:, 0]) * np.cos(np.radians(p1[:, 1])), p1[:, 1] - p0[:, 1])
    d[off[1:-1] - 1] = 0                      # no segment between two reaches
    c = np.concatenate([[0], np.cumsum(d)])
    half = (c[off[:-1]] + c[off[1:] - 1]) / 2
    j = np.searchsorted(c, half, side="left")
    j = np.clip(j, off[:-1] + 1, np.maximum(off[1:] - 1, off[:-1] + 1))
    j = np.minimum(j, len(co) - 1)
    seg = np.maximum(c[j] - c[j - 1], 1e-12)
    t = np.clip((half - c[j - 1]) / seg, 0, 1)[:, None]
    out = co[j - 1] + t * (co[j] - co[j - 1])
    one = np.diff(off) == 1
    out[one] = co[off[:-1][one]]
    return out


def stage_reaches():
    """Per reach k (as rivers.py build keeps them): the window from k's midpoint to the midpoint of
    the first downstream reach at least WIN_KM on (or the river's end), with the MERIT drop and
    flow length over it and RiverATLAS's length. NaN where the tiles do not hold both ends or a snap
    fails (two basins, or Q / area off by more than QA_SPREAD: another river)."""
    import glob
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rivers
    r = dict(np.load(os.path.join(WORK, "reaches.npz")))
    dz = np.load(os.path.join(WORK, "desert.npz"))
    r["q"] = np.where(dz["in_hot"], dz["q_in"], r["q"])
    r = rivers.keep_reaches(r, dz["ok"])
    R = len(r["q"])
    q, L = r["q"], r["length_km"]
    mid = midpoints(r["co"], r["off"])
    down = rivers.reach_down(r)
    log("reaches", R)
    sd = np.full(R, np.nan)       # snapped height above the outlet (m)
    sx = np.full(R, np.nan)       # snapped flow distance to the outlet (m)
    sb = np.zeros(R, np.int64)
    tr_, tc_ = np.floor((85 - mid[:, 1]) / 5).astype(np.int64), np.floor((mid[:, 0] + 180) / 5).astype(np.int64)
    files = sorted(glob.glob(os.path.join(TILES, "*_*.npz")))
    offs = np.arange(-SNAP, SNAP + 1)
    for f in files:
        tr, tc = map(int, os.path.basename(f)[:-4].split("_"))
        ks = np.nonzero((tr_ == tr) & (tc_ == tc))[0]
        if not len(ks):
            continue
        b = np.load(f)
        c0, r0 = int(b["c0"]), int(b["r0"])
        diff, dist, acc, basin = b["diff"], b["dist"], b["acc"], b["basin"]
        cc = np.floor((mid[ks, 0] + 180) * PX).astype(np.int64) - c0
        rr = np.floor((85 - mid[ks, 1]) * PX).astype(np.int64) - r0
        yy = rr[:, None, None] + offs[None, :, None]
        xx = cc[:, None, None] + offs[None, None, :]
        a = acc[yy, xx].reshape(len(ks), -1)
        a = np.where(a > -1, a, -1)
        j = np.argmax(a, axis=1)
        y, x = rr + j // len(offs) - SNAP, cc + j % len(offs) - SNAP
        dv, xv, av = diff[y, x].astype(np.float64), dist[y, x].astype(np.float64), acc[y, x].astype(np.float64)
        good = (dv != -9999) & (xv != -9999) & (av > 0)
        qa = q[ks] / np.where(good, av, np.nan)
        med = np.nanmedian(qa) if good.any() else np.nan
        good &= (qa > med / QA_SPREAD) & (qa < med * QA_SPREAD)
        sd[ks[good]], sx[ks[good]], sb[ks[good]] = dv[good], xv[good], basin[y, x][good]
        log("%s: %d reaches, %d snapped, Q/area %.4f" % (os.path.basename(f), len(ks), good.sum(), med))
    # the window: k's midpoint to the midpoint of the first reach j with half k + ... + half j >= WIN_KM
    ra = np.full(R, np.nan)
    drop = np.full(R, np.nan)
    mkm = np.full(R, np.nan)
    dl, Ll = down.tolist(), L.tolist()
    have = np.isfinite(sd)
    for k in np.nonzero(have)[0].tolist():
        km, j = Ll[k] / 2, dl[k]
        while j >= 0 and km + Ll[j] / 2 < WIN_KM:
            km += Ll[j]
            j = dl[j]
        if j < 0:
            # the river ends (sea, a sink): its end is the basin outlet only when the outlet is
            # where RiverATLAS ends too; left out
            continue
        if not have[j] or sb[j] != sb[k]:
            continue
        km += Ll[j] / 2
        m = (sx[k] - sx[j]) / 1000
        if m <= 0:
            continue
        ra[k], drop[k], mkm[k] = km, max(sd[k] - sd[j], 0.0), m
    ok = np.isfinite(ra)
    tmp = os.path.join(WORK, "merit.tmp.npz")
    np.savez(tmp, ra_km=ra.astype(np.float32), drop=drop.astype(np.float32), merit_km=mkm.astype(np.float32))
    os.replace(tmp, os.path.join(WORK, "merit.npz"))
    log("reaches with a MERIT window: %d, %.0f km of %.0f km" % (ok.sum(), L[ok].sum(), L.sum()))


def main():
    os.makedirs(WORK, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    t0 = time.time()
    if stage == "fetch":
        stage_fetch()
    elif stage == "reaches":
        stage_reaches()
    else:
        print(__doc__)
        return 1
    log("done in %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    sys.exit(main())
