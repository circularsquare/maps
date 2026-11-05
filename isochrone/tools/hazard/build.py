"""The land hazard band (docs/hazard.md): a uint8 log-coded rate per walking hour, today's world.

    python isochrone/tools/hazard/build.py ice     # ice outlines north of 60 N -> work/ice_l1.npy         (~1 min)
    python isochrone/tools/hazard/build.py dist    # distance to drinkable water, 2 km grid -> work/dist_g2.npy  (~5 min, ~4 GB)
    python isochrone/tools/hazard/build.py tiles   # rates per L1 pixel -> data/hazard/ l10, l3, l1, index.json  (~25 min, ~3 GB)
    python isochrone/tools/hazard/build.py plot    # the built L1 band -> maps/artifacts/isochrone/t066_world.png

Inputs (tools/hazard/fetch.py): CHELSA V2.1 in data/raw/chelsa/, OSM water points in data/raw/osm_water/,
ice outlines in data/raw/ice/ (RGI 6.0 Arctic regions where present, Natural Earth 10 m glaciated areas);
RiverATLAS vertices and HydroLAKES from the canoe build (data/raw/rivers_work/); MAP's walking surface
(time weights, land, as the speed tiles); data/surface/lc_class.tif and elev.tif. Work files in
data/raw/hazard_work/.
"""
import glob
import gzip
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import rasterio
from rasterio.windows import Window

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(HERE, "tools"))
from smallwater import SmallWater, STRIP   # noqa: E402  (the speed tiles' land: small water is land)

RAW = os.path.join(HERE, "data", "raw")
WORK = os.path.join(RAW, "hazard_work")
RIVERS = os.path.join(RAW, "rivers_work")
OUT = os.path.join(HERE, "data", "hazard")
MAPF = os.path.join(RAW, "unzipped", "2020_walking_only_friction_surface.geotiff")
LC = os.path.join(HERE, "data", "surface", "lc_class.tif")
ELEV = os.path.join(HERE, "data", "surface", "elev.tif")
CH = os.path.join(RAW, "chelsa", "CHELSA_%s_%s_1981-2010_V.2.1.tif")
CHB = os.path.join(RAW, "chelsa", "CHELSA_%s_1981-2010_V.2.1.tif")
ART = os.path.join(os.path.dirname(HERE), "artifacts", "isochrone")

L1W, L1H, NORTH = 43200, 17400, 85.0
G = 2                          # L1 pixels per distance cell (1/60 degree, ~2 km)
GW, GH = L1W // G, L1H // G
ICE_ROWS = 3000                # the ice outlines are read north of 60 N
R_EARTH = 6371.0
WATER = np.float32(0.06)
MIN_LAND = 0.25                # as the speed grid and tiles
TILE = {1: 600, 3: 200}

# ---- the rule (docs/hazard.md 2) ----
Q_DRINK = 0.1                  # m3/s mean discharge: a stream you can drink from
Q_PERENNIAL = 0.1              # m3/s in the driest month (Q >= 1 reaches): water even in a hot desert
HOT_DESERT_MM, HOT_DESERT_C = 250.0, 25.0
V_DAY = 25.0                   # km walked a day, turning survival days into distance
# FM 21-76 (1957) fig. 196: days survived walking at night with 10 US quarts, by max shade temperature
S10_T = np.array([10.0, 15.6, 21.1, 26.7, 32.2, 37.8, 43.3, 48.9], np.float32)
S10_D = np.array([14.0, 14.0, 13.5, 11.5, 8.0, 5.5, 3.5, 3.0], np.float32)
ICE = 1e-3
R_MAX = 1.0


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def thirst(d_km, tmax):
    s = np.interp(tmax, S10_T, S10_D).astype(np.float32)
    k = np.float32(8.477 * 2 / V_DAY) / s           # ln(0.693 / 0.01) / (S / 2), per km of half-gap
    c = np.float32(1.0596 * 0.01 * np.exp(-4.238)) / s
    return np.minimum(np.float32(R_MAX), c * np.expm1(np.minimum(k * d_km, np.float32(40.0))))


def cold(tmin):
    r = np.float32(1e-5) * np.power(np.float32(10.0), (np.float32(-10.0) - tmin) / np.float32(10.0))
    return np.where(tmin >= -10.0, np.float32(0), np.minimum(r, np.float32(1e-2)))


def altitude(z):
    r = np.power(np.float32(10.0), np.float32(-6.0) + (z - np.float32(3000.0)) / np.float32(1000.0))
    return np.where(z >= 3000.0, np.minimum(r, np.float32(1e-2)), np.float32(0))


def code(r):
    """0 = no hazard; c = 1..240 a rate of 10^((c - 240) / 40) per walking hour."""
    c = np.rint(240.0 + 40.0 * np.log10(np.maximum(r, 1e-30)))
    return np.where(r > 0, np.clip(c, 0, 240), 0).astype(np.uint8)


def decode(c):
    return np.where(c > 0, np.power(10.0, (c.astype(np.float32) - 240.0) / 40.0), 0.0).astype(np.float32)


def ice_mask(lc, poly):
    """LC100 snow and ice; where LC100 has no class (north of 80 N, Svalbard, some Arctic islands),
    inside an ice outline (RGI, Natural Earth)."""
    return (lc == 70) | (((lc == 0) | (lc == 255)) & poly)


def ice_rows(poly, r0, n):
    out = np.zeros((n, L1W), bool)
    k = max(0, min(ICE_ROWS, r0 + n) - r0)
    if k:
        out[:k] = poly[r0:r0 + k]
    return out


# ---------------------------------------------------------------- CHELSA
def chelsa_reader():
    dss = {(v, m): rasterio.open(CH % (v, m)) for v in ("tasmax", "tasmin") for m in ("01", "07")}
    dss[("bio12", None)] = rasterio.open(CHB % "bio12")
    ds = dss[("tasmax", "01")]
    top, res = ds.bounds.top, ds.res[1]
    col0 = int(round((-180.0 - ds.bounds.left) / ds.res[0]))

    def read(v, m, r0, n):
        d = dss[(v, m)]
        row0 = int(round((top - (NORTH - r0 / 120.0)) / res))
        rows = np.clip(row0 + np.arange(n), 0, d.height - 1)    # north of CHELSA's 84 N: its top row
        lo, hi = rows.min(), rows.max()
        a = d.read(1, window=Window(col0, lo, L1W, hi - lo + 1))[rows - lo]
        bad = a == d.nodata
        # temperatures: K x 10 (the file's scale tag); bio12: mm a year as stored
        t = (a.astype(np.float32) / np.float32(10.0) - np.float32(273.15)) if v.startswith("tas") else a.astype(np.float32)
        t[bad] = np.nan
        return t
    return read


# ---------------------------------------------------------------- ice
def stage_ice():
    import pyogrio
    import shapely
    from rasterio import features
    from rasterio.transform import from_origin
    os.makedirs(WORK, exist_ok=True)
    out = np.zeros((ICE_ROWS, L1W), np.uint8)
    tr = from_origin(-180.0, NORTH, 1 / 120, 1 / 120)
    shps = sorted(glob.glob(os.path.join(RAW, "ice", "**", "*.shp"), recursive=True))
    for p in shps:
        _, _, geom, _ = pyogrio.raw.read(p, columns=[])
        polys = shapely.from_wkb(geom)
        polys = polys[shapely.bounds(polys)[:, 3] > 60.0]
        features.rasterize(((g, 1) for g in polys), out=out, transform=tr, merge_alg=features.MergeAlg.replace)
        log(os.path.relpath(p, RAW), len(polys), "outlines north of 60 N; ice pixels so far", int(out.sum()))
    np.save(os.path.join(WORK, "ice_l1.npy"), out.astype(bool))


# ---------------------------------------------------------------- distance
def unit(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1)


def stage_dist():
    from scipy.spatial import cKDTree
    poly = np.load(os.path.join(WORK, "ice_l1.npy"))
    lk = np.load(os.path.join(RIVERS, "lakes_l1.npy"), mmap_mode="r")
    land = np.zeros((GH, GW), bool)
    ice = np.zeros((GH, GW), bool)
    lake = np.zeros((GH, GW), bool)
    hot = np.zeros((GH, GW), bool)     # hot desert: streams there are dry most of the time
    read = chelsa_reader()
    with rasterio.open(MAPF) as fds, rasterio.open(LC) as lds:
        sw = SmallWater(fds, log=lambda *a: None)     # the tiles' land: every land pixel needs a distance
        for r0 in range(0, L1H, STRIP):
            n = min(STRIP, L1H - r0)
            g0, gn = r0 // G, n // G
            blk = lambda m: m.reshape(gn, G, GW, G)
            f = fds.read(1, window=Window(0, r0, L1W, n))
            land[g0:g0 + gn] = blk(((f != WATER) & (f > 0)) | sw.mask(r0 // STRIP, f)).any(axis=(1, 3))
            del f
            lc = lds.read(1, window=Window(0, r0, L1W, n))
            ice[g0:g0 + gn] = blk(ice_mask(lc, ice_rows(poly, r0, n))).sum(axis=(1, 3)) >= 2
            del lc
            lake[g0:g0 + gn] = blk(np.asarray(lk[r0:r0 + n]) > 0).any(axis=(1, 3))
            pr = np.nan_to_num(read("bio12", None, r0, n), nan=1000.0)
            tx = np.nan_to_num(np.fmax(read("tasmax", "01", r0, n), read("tasmax", "07", r0, n)), nan=0.0)
            hot[g0:g0 + gn] = blk((pr < HOT_DESERT_MM) & (tx >= HOT_DESERT_C)).sum(axis=(1, 3)) >= 2
            del pr, tx
    del poly
    log("land cells", int(land.sum()), "ice", int(ice.sum()), "hot desert", int((hot & land).sum()))
    a = np.load(os.path.join(RIVERS, "all_lines.npz"))
    co, off, q = a["co"], a["off"], a["q"]
    r = np.load(os.path.join(RIVERS, "reaches.npz"))
    perennial = np.zeros(len(q), bool)
    perennial[r["all_index"][r["qmin"] >= Q_PERENNIAL]] = True
    cnt = np.diff(off)
    keep = np.repeat((q >= Q_DRINK) | perennial, cnt)
    pv = np.repeat(perennial, cnt)[keep]
    sel = co[keep]
    del co, keep, a, r
    gc = np.clip(((sel[:, 0] + 180.0) * 60).astype(np.int64), 0, GW - 1)
    gr = ((NORTH - sel[:, 1]) * 60).astype(np.int64)
    del sel
    ok = (gr >= 0) & (gr < GH)
    gr, gc, pv = gr[ok], gc[ok], pv[ok]
    ok = pv | ~hot[gr, gc]
    wet = np.zeros((GH, GW), bool)
    wet[gr[ok], gc[ok]] = True
    log("river cells", int(wet.sum()), "; vertices dropped in hot desert", int((~ok).sum()), "of", len(ok))
    del gr, gc, pv, ok
    wet |= lake & ~hot
    del lake, hot
    # today's water points: wells, springs, settlements (OSM)
    for fn in sorted(glob.glob(os.path.join(RAW, "osm_water", "*.tsv"))):
        p = np.loadtxt(fn, dtype=np.float64, ndmin=2)
        pr_ = ((NORTH - p[:, 0]) * 60).astype(np.int64)
        pc_ = np.clip(((p[:, 1] + 180.0) * 60).astype(np.int64), 0, GW - 1)
        ok = (pr_ >= 0) & (pr_ < GH)
        wet[pr_[ok], pc_[ok]] = True
        log(os.path.basename(fn), len(p))
    wet &= ~ice
    del ice
    wr, wc = np.nonzero(wet)
    del wet
    t0 = time.time()
    tree = cKDTree(unit(-180.0 + (wc + 0.5) / 60.0, NORTH - (wr + 0.5) / 60.0))
    log("water cells", len(wr), "tree %.0f s" % (time.time() - t0))
    del wr, wc
    d = np.full((GH, GW), 65535, np.uint16)
    ub = 2 * np.sin(2000.0 / R_EARTH / 2)
    BAND = 300                        # distance-grid rows per query batch
    for g0 in range(0, GH, BAND):
        lr, lcc = np.nonzero(land[g0:g0 + BAND])
        if not len(lr):
            continue
        dd, _ = tree.query(unit(-180.0 + (lcc + 0.5) / 60.0, NORTH - (g0 + lr + 0.5) / 60.0), k=1,
                           distance_upper_bound=ub)
        km = np.where(np.isfinite(dd), 2 * R_EARTH * np.arcsin(np.minimum(dd, 2.0) / 2), 2000.0)
        d[g0 + lr, lcc] = np.minimum(np.rint(km), 2000).astype(np.uint16)
    log("queried %.0f s" % (time.time() - t0))
    np.save(os.path.join(WORK, "dist_g2.npy"), d)


# ---------------------------------------------------------------- tiles
def blocksum(a, b):
    h, w = a.shape
    return a.reshape(h // b, b, w // b, b).sum(axis=(1, 3))


def write(path, arr):
    data = gzip.compress(np.ascontiguousarray(arr).tobytes(), compresslevel=9)
    with open(path + ".tmp", "wb") as fh:
        fh.write(data)
    os.replace(path + ".tmp", path)
    return len(data)


def stage_tiles():
    t0 = time.time()
    dist = np.load(os.path.join(WORK, "dist_g2.npy"), mmap_mode="r")
    poly = np.load(os.path.join(WORK, "ice_l1.npy"))
    read = chelsa_reader()
    for lv in TILE:
        os.makedirs(os.path.join(OUT, "l%d" % lv), exist_ok=True)
    l10 = np.zeros((L1H // 10, L1W // 10), np.uint8)
    found = {lv: [] for lv in TILE}
    nbytes = {lv: 0 for lv in TILE}
    tx_n = L1W // STRIP
    with rasterio.open(MAPF) as fds, rasterio.open(LC) as lds, rasterio.open(ELEV) as eds:
        sw = SmallWater(fds, log=lambda *a: None)
        for tr in range(L1H // STRIP):
            r0, n = tr * STRIP, STRIP
            f = fds.read(1, window=Window(0, r0, L1W, n))
            land = ((f != WATER) & (f > 0)) | sw.mask(tr, f)
            fw = np.where(land, f, 0).astype(np.float32)     # time per metre: the coarse levels' weights
            del f
            g0 = r0 // G
            d = np.repeat(np.repeat(np.asarray(dist[g0:g0 + n // G], np.float32), G, 0), G, 1)
            nod = int((land & (d == 65535)).sum())
            if nod:
                log("  %d land pixels with no distance (read as 2000 km)" % nod)
                d[d == 65535] = 2000.0
            r = np.zeros((n, L1W), np.float32)
            jul = None
            for m in ("01", "07"):
                tx = np.nan_to_num(read("tasmax", m, r0, n), nan=15.0)   # no CHELSA value (a few coast pixels): mild
                tn = np.nan_to_num(read("tasmin", m, r0, n), nan=5.0)
                rm = thirst(d, tx) + cold(tn)
                del tx, tn
                if jul is None:
                    r, jul = rm, True
                else:
                    np.minimum(r, rm, out=r)                  # the kinder of January and July
                del rm
            del d
            lc = lds.read(1, window=Window(0, r0, L1W, n))
            r += np.where(ice_mask(lc, ice_rows(poly, r0, n)), np.float32(ICE), np.float32(0))
            del lc
            z = eds.read(1, window=Window(0, r0, L1W, n)).astype(np.float32)
            z[z < -1000] = 0
            r += altitude(z)
            del z
            np.minimum(r, np.float32(R_MAX), out=r)
            r[~land] = 0
            c1 = code(r)
            # coarse levels: time-weighted mean of the L1 land pixels, kept where the speed level has land
            wr = fw * r
            del r
            coarse = {}
            for lv in (3, 10):
                ws, rs = blocksum(fw, lv), blocksum(wr, lv)
                ls = blocksum(land.astype(np.uint16), lv)
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean = np.where(ws > 0, rs / ws, 0.0)
                cc = code(mean)
                cc[ls < MIN_LAND * lv * lv] = 0
                coarse[lv] = cc
            del wr, fw, land
            l10[r0 // 10:(r0 + n) // 10] = coarse[10]
            for lv, size in TILE.items():
                g = c1 if lv == 1 else coarse[lv]
                for tc in range(tx_n):
                    t = g[:, tc * size:(tc + 1) * size]
                    if not t.any():
                        continue
                    nbytes[lv] += write(os.path.join(OUT, "l%d" % lv, "%d_%d.bin.gz" % (tr, tc)), t)
                    found[lv].append(tr * tx_n + tc)
            log("tile row %d / %d (%.0f s)  L1 %d tiles %.1f MB, L3 %d tiles %.1f MB" %
                (tr + 1, L1H // STRIP, time.time() - t0, len(found[1]), nbytes[1] / 1e6, len(found[3]), nbytes[3] / 1e6))
    n10 = write(os.path.join(OUT, "l10.bin.gz"), l10)
    index = {"west": -180.0, "east": 180.0, "north": NORTH, "south": -60.0, "tileDeg": 5,
             "tilesX": tx_n, "tilesY": L1H // STRIP,
             "coding": "uint8: 0 no hazard, c in 1..240 a rate of 10^((c - 240) / 40) per walking hour (docs/hazard.md 4)",
             "world": {"file": "l10.bin.gz", "width": L1W // 10, "height": L1H // 10},
             "levels": {str(lv): {"factor": lv, "width": L1W // lv, "height": L1H // lv, "tile": TILE[lv],
                                  "path": "l%d" % lv, "tiles": found[lv]} for lv in TILE}}
    with open(os.path.join(OUT, "index.json"), "w") as fh:
        json.dump(index, fh, separators=(",", ":"))
    log("L10 %.2f MB; L3 %d tiles %.1f MB; L1 %d tiles %.1f MB; %.0f s" %
        (n10 / 1e6, len(found[3]), nbytes[3] / 1e6, len(found[1]), nbytes[1] / 1e6, time.time() - t0))


# ---------------------------------------------------------------- plot
def stage_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    with open(os.path.join(OUT, "index.json")) as fh:
        idx = json.load(fh)
    S = 4                                               # every 4th L1 pixel: 10800 x 4350
    img = np.full((L1H // S, L1W // S), np.nan, np.float32)
    lv = idx["levels"]["1"]
    for t in lv["tiles"]:
        tr, tc = divmod(t, idx["tilesX"])
        with gzip.open(os.path.join(OUT, "l1", "%d_%d.bin.gz" % (tr, tc))) as fh:
            a = np.frombuffer(fh.read(), np.uint8).reshape(600, 600)[::S, ::S]
        img[tr * 150:(tr + 1) * 150, tc * 150:(tc + 1) * 150] = np.where(a > 0, decode(a), np.nan)
    with gzip.open(os.path.join(HERE, "data", "grid.bin.gz")) as fh:
        grid = np.frombuffer(fh.read(), np.uint8).reshape(L1H // 10, L1W // 10)
    fig, ax = plt.subplots(figsize=(24, 10.5), dpi=120)
    ax.set_facecolor("#20242a")
    ext = [-180, 180, -60, 85]
    ax.imshow(np.where(grid > 0, 1.0, np.nan), extent=ext, cmap="Greys", vmin=0, vmax=3, interpolation="nearest")
    im = ax.imshow(img, extent=ext, cmap="magma", norm=LogNorm(1e-6, 1), interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.7, label="hazard per walking hour")
    ax.set_title("land hazard, built L1 band (every 4th pixel); grey: land with no hazard")
    os.makedirs(ART, exist_ok=True)
    p = os.path.join(ART, "t066_world.png")
    fig.savefig(p, bbox_inches="tight")
    log("wrote", p)


if __name__ == "__main__":
    {"ice": stage_ice, "dist": stage_dist, "tiles": stage_tiles, "plot": stage_plot}[sys.argv[1]]()
    try:
        import psutil
        mi = psutil.Process().memory_info()
        log("peak memory %.1f GB" % (getattr(mi, "peak_wset", mi.rss) / 1e9))
    except ImportError:
        pass
