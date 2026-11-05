"""T-042 sample: the land hazard rate at L10 from the real sources, by the rule in docs/hazard.md.

    python isochrone/notes/T-042/build_l10.py dist     # distance to drinkable water, 2 km grid, two variants  (~5 min)
    python isochrone/notes/T-042/build_l10.py rates    # rates per L1 pixel -> L10 (+ L3, L1 code sizes)       (~10 min)

Inputs: MAP friction (the page's walking surface, for time weights and land), data/surface/lc_class.tif
(ice) and elev.tif (altitude), CHELSA V2.1 Jan/Jul tasmax and tasmin (data/raw/chelsa/), RiverATLAS
vertices and HydroLAKES (data/raw/rivers_work/), OSM wells, springs and settlements (data/raw/osm_water/).
Outputs in data/raw/t042/ (gitignored; delete with the task).
"""
import gzip
import os
import sys
import time
import zlib

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import rasterio
from rasterio.windows import Window

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(HERE, "data", "raw")
WORK = os.path.join(RAW, "rivers_work")
OUT = os.path.join(RAW, "t042")
MAPF = os.path.join(RAW, "unzipped", "2020_walking_only_friction_surface.geotiff")
LC = os.path.join(HERE, "data", "surface", "lc_class.tif")
ELEV = os.path.join(HERE, "data", "surface", "elev.tif")
CH = os.path.join(RAW, "chelsa", "CHELSA_%s_%s_1981-2010_V.2.1.tif")
CHB = os.path.join(RAW, "chelsa", "CHELSA_%s_1981-2010_V.2.1.tif")

L1W, L1H, NORTH = 43200, 17400, 85.0
G = 2                         # L1 pixels per distance cell (1/60 degree, ~2 km)
GW, GH = L1W // G, L1H // G
R_EARTH = 6371.0
Q_DRINK = 0.1                 # m3/s mean discharge: a stream you can drink from
Q_PERENNIAL = 0.1             # m3/s in the driest month (Q >= 1 reaches): water even in a hot desert
HOT_DESERT_MM, HOT_DESERT_C = 250.0, 25.0   # annual rain under this and the warmer of Jan/Jul max at least this
WATER = np.float32(0.06)
STRIP = 300                   # L1 rows per strip: 30 L10 rows, 100 L3 rows

# ---- the rule (docs/hazard.md) ----
V_DAY = 25.0                  # km walked in a day, for turning survival days into distance
# FM 21-76 (1957) fig. 196: days of survival walking at night with 10 US quarts, by max shade temperature
S10_T = np.array([10.0, 15.6, 21.1, 26.7, 32.2, 37.8, 43.3, 48.9])
S10_D = np.array([14.0, 14.0, 13.5, 11.5, 8.0, 5.5, 3.5, 3.0])
ICE = 1e-3
R_MAX = 1.0
# name, distance layer, month (None = the kinder of January and July per pixel)
VARIANTS = [("nat", "nat", None), ("osm", "osm", None), ("natjul", "nat", "07"), ("osmjul", "osm", "07")]


def thirst(d_km, tmax):
    s = np.interp(tmax, S10_T, S10_D).astype(np.float32)
    k = 8.477 / s * 2 / V_DAY                      # ln(0.693 / 0.01) / (S/2), per km of half-gap
    c = 1.0596 / s * 0.01 * np.exp(-4.238)         # (a / 8 h) x 0.01 x e^(-a S/2)
    return np.minimum(R_MAX, c * np.expm1(np.minimum(k * d_km, 40.0)))


def cold(tmin):
    r = 1e-5 * np.power(10.0, (-10.0 - tmin) / 10.0)
    return np.where(tmin >= -10.0, 0.0, np.minimum(r, 1e-2)).astype(np.float32)


def altitude(z):
    r = np.power(10.0, -6.0 + (z - 3000.0) / 1000.0)
    return np.where(z >= 3000.0, np.minimum(r, 1e-2), 0.0).astype(np.float32)


def code(r):
    """uint8: 0 = no hazard, c = 1..240 a rate of 10^((c - 240) / 40) per walking hour."""
    c = np.rint(240.0 + 40.0 * np.log10(np.maximum(r, 1e-30)))
    return np.where(r > 0, np.clip(c, 0, 240), 0).astype(np.uint8)


def decode(c):
    return np.where(c > 0, np.power(10.0, (c.astype(np.float64) - 240.0) / 40.0), 0.0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ice_mask(lc, lat_rows):
    # LC100 snow and ice; no class on land north of 80 N is walked as snow and ice (docs/surface.md)
    return (lc == 70) | (((lc == 0) | (lc == 255)) & (lat_rows[:, None] > 80.0))


# ---------------------------------------------------------------- distance
def unit(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1)


def stage_dist():
    from scipy.spatial import cKDTree
    os.makedirs(OUT, exist_ok=True)
    lk = np.load(os.path.join(WORK, "lakes_l1.npy"), mmap_mode="r")
    land = np.zeros((GH, GW), bool)
    ice = np.zeros((GH, GW), bool)
    lake = np.zeros((GH, GW), bool)
    hot = np.zeros((GH, GW), bool)     # hot desert: streams there are dry most of the time
    read = chelsa_reader()
    with rasterio.open(MAPF) as fds, rasterio.open(LC) as lds:
        for r0 in range(0, L1H, STRIP):
            n = min(STRIP, L1H - r0)
            f = fds.read(1, window=Window(0, r0, L1W, n))
            lc = lds.read(1, window=Window(0, r0, L1W, n))
            lat = NORTH - (r0 + np.arange(n) + 0.5) / 120.0
            g0, gn = r0 // G, n // G
            blk = lambda m: m.reshape(gn, G, GW, G)
            land[g0:g0 + gn] = blk((f != WATER) & (f > 0)).any(axis=(1, 3))
            ice[g0:g0 + gn] = blk(ice_mask(lc, lat)).sum(axis=(1, 3)) >= 2
            lake[g0:g0 + gn] = blk(np.asarray(lk[r0:r0 + n]) > 0).any(axis=(1, 3))
            pr = np.nan_to_num(read("bio12", None, r0, n), nan=1000.0)
            tx = np.fmax(read("tasmax", "01", r0, n), read("tasmax", "07", r0, n))
            hd = (pr < HOT_DESERT_MM) & (np.nan_to_num(tx, nan=0.0) >= HOT_DESERT_C)
            hot[g0:g0 + gn] = blk(hd).sum(axis=(1, 3)) >= 2
    log("land cells", int(land.sum()), "ice", int(ice.sum()), "hot desert", int((hot & land).sum()))
    a = np.load(os.path.join(WORK, "all_lines.npz"))
    co, off, q = a["co"], a["off"], a["q"]
    r = np.load(os.path.join(WORK, "reaches.npz"))
    perennial = np.zeros(len(q), bool)
    perennial[r["all_index"][r["qmin"] >= Q_PERENNIAL]] = True
    qv = np.repeat(q, np.diff(off))
    pv = np.repeat(perennial, np.diff(off))
    keep = (qv >= Q_DRINK) | pv
    sel, pv = co[keep], pv[keep]
    del co, qv
    gc = np.clip(((sel[:, 0] + 180.0) * 60).astype(np.int64), 0, GW - 1)
    gr = ((NORTH - sel[:, 1]) * 60).astype(np.int64)
    ok = (gr >= 0) & (gr < GH)
    gr, gc, pv = gr[ok], gc[ok], pv[ok]
    ok = pv | ~hot[gr, gc]
    wet = np.zeros((GH, GW), bool)
    wet[gr[ok], gc[ok]] = True
    log("river cells", int(wet.sum()), "; vertices dropped in hot desert", int((~ok).sum()), "of", len(ok))
    wet |= lake & ~hot
    nat = wet & ~ice
    log("water cells (natural, ice removed)", int(nat.sum()))
    # OSM water points: wells, springs, settlements
    pts = []
    for fn in sorted(os.listdir(os.path.join(RAW, "osm_water"))):
        if fn.endswith(".tsv"):
            p = np.loadtxt(os.path.join(RAW, "osm_water", fn), dtype=np.float64, ndmin=2)
            log(fn, len(p))
            pts.append(p)
    p = np.concatenate(pts)
    pr = ((NORTH - p[:, 0]) * 60).astype(np.int64)
    pc = np.clip(((p[:, 1] + 180.0) * 60).astype(np.int64), 0, GW - 1)
    ok = (pr >= 0) & (pr < GH)
    osm = nat.copy()
    osm[pr[ok], pc[ok]] = True
    osm &= ~ice
    np.save(os.path.join(OUT, "ice_g2.npy"), ice)
    np.save(os.path.join(OUT, "land_g2.npy"), land)
    lr, lcc = np.nonzero(land)
    qxyz = unit(-180.0 + (lcc + 0.5) / 60.0, NORTH - (lr + 0.5) / 60.0)
    for name, m in (("nat", nat), ("osm", osm)):
        wr, wc = np.nonzero(m)
        t0 = time.time()
        tree = cKDTree(unit(-180.0 + (wc + 0.5) / 60.0, NORTH - (wr + 0.5) / 60.0))
        log(name, "tree", len(wr), "%.0f s" % (time.time() - t0))
        d = np.full((GH, GW), 65535, np.uint16)
        ub = 2 * np.sin(2000.0 / R_EARTH / 2)
        CH_ = 4_000_000
        for i in range(0, len(lr), CH_):
            dd, _ = tree.query(qxyz[i:i + CH_], k=1, distance_upper_bound=ub)
            km = np.where(np.isfinite(dd), 2 * R_EARTH * np.arcsin(np.minimum(dd, 2.0) / 2), 2000.0)
            d[lr[i:i + CH_], lcc[i:i + CH_]] = np.minimum(np.rint(km), 2000).astype(np.uint16)
        log(name, "queried %.0f s" % (time.time() - t0))
        np.save(os.path.join(OUT, "dist_%s_g2.npy" % name), d)
        del tree


# ---------------------------------------------------------------- rates
def chelsa_reader():
    dss = {(v, m): rasterio.open(CH % (v, m)) for v in ("tasmax", "tasmin") for m in ("01", "07")}
    dss[("bio12", None)] = rasterio.open(CHB % "bio12")
    ds = next(iter(dss.values()))
    top, res = ds.bounds.top, ds.res[1]
    col0 = int(round((-180.0 - ds.bounds.left) / ds.res[0]))
    scale, offset = ds.scales[0], ds.offsets[0]
    log("CHELSA", ds.width, ds.height, ds.bounds, "scale", scale, "offset", offset, "nodata", ds.nodata)

    def read(v, m, r0, n):
        d = dss[(v, m)]
        row0 = int(round((top - (NORTH - r0 / 120.0)) / res))
        rows = np.clip(row0 + np.arange(n), 0, d.height - 1)    # north of CHELSA's top: its top row
        lo, hi = rows.min(), rows.max()
        a = d.read(1, window=Window(col0, lo, L1W, hi - lo + 1)).astype(np.float32)
        a = a[rows - lo]
        bad = a == d.nodata if d.nodata is not None else np.zeros(a.shape, bool)
        # temperatures: K x 10 (the file's scale tag); bio12: mm a year as stored
        t = a / 10.0 - 273.15 if v.startswith("tas") else a
        t[bad] = np.nan
        return t
    return read


def blockmean(num, den, b):
    h, w = num.shape
    ns = num.reshape(h // b, b, w // b, b).sum(axis=(1, 3))
    ds = den.reshape(h // b, b, w // b, b).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ds > 0, ns / ds, 0.0)


def stage_rates():
    dn = np.load(os.path.join(OUT, "dist_nat_g2.npy"), mmap_mode="r")
    do = np.load(os.path.join(OUT, "dist_osm_g2.npy"), mmap_mode="r")
    read = chelsa_reader()
    W10, H10 = L1W // 10, L1H // 10
    names = [v for v, _, _ in VARIANTS] + ["thirst_" + v for v, _, _ in VARIANTS] + ["cold", "ice", "alt", "july"]
    o10 = {k: np.zeros((H10, W10), np.float32) for k in names}
    zc = {lv: {v: zlib.compressobj(9, zlib.DEFLATED, 31) for v in ("nat", "osm")} for lv in (1, 3, 10)}
    zn = {lv: {v: 0 for v in ("nat", "osm")} for lv in (1, 3, 10)}
    with rasterio.open(MAPF) as fds, rasterio.open(LC) as lds, rasterio.open(ELEV) as eds:
        for r0 in range(0, L1H, STRIP):
            n = min(STRIP, L1H - r0)
            f = fds.read(1, window=Window(0, r0, L1W, n))
            land = (f != WATER) & (f > 0)
            fw = np.where(land, f, 0).astype(np.float64)
            lat = NORTH - (r0 + np.arange(n) + 0.5) / 120.0
            ice = ice_mask(lds.read(1, window=Window(0, r0, L1W, n)), lat)
            z = eds.read(1, window=Window(0, r0, L1W, n)).astype(np.float32)
            z[z < -1000] = 0
            g0 = r0 // G
            up = lambda a: np.repeat(np.repeat(np.asarray(a[g0:g0 + n // G], np.float32), G, 0), G, 1)
            d = {"nat": up(dn), "osm": up(do)}
            tx = {m: read("tasmax", m, r0, n) for m in ("01", "07")}
            tn = {m: read("tasmin", m, r0, n) for m in ("01", "07")}
            for m in ("01", "07"):     # no CHELSA value (a few coastal pixels): mild
                tx[m] = np.nan_to_num(tx[m], nan=15.0)
                tn[m] = np.nan_to_num(tn[m], nan=5.0)
            alt = altitude(z)
            icer = np.where(ice, ICE, 0.0).astype(np.float32)
            parts = {}
            for v, dv, force in VARIANTS:
                tj, tl = thirst(d[dv], tx["01"]), thirst(d[dv], tx["07"])
                cj, cl = cold(tn["01"]), cold(tn["07"])
                jul = (tl + cl) < (tj + cj) if force is None else np.full(tl.shape, force == "07")
                th = np.where(jul, tl, tj)
                co_ = np.where(jul, cl, cj)
                r = np.minimum(R_MAX, th + co_ + icer + alt)
                r = np.where(land, r, 0.0)
                parts[v] = r
                parts["thirst_" + v] = th
                if v == "nat":
                    parts["cold"], parts["july"] = co_, jul.astype(np.float32)
                if v not in zc[1]:
                    continue
                # sizes: code each level as the band would be, gzip-streamed
                for lv, b in ((1, 1), (3, 3), (10, 10)):
                    if lv == 1:
                        cc = code(r)
                    else:
                        cc = code(blockmean(fw * r, fw, b))
                    zn[lv][v] += len(zc[lv][v].compress(cc.tobytes()))
            parts["ice"], parts["alt"] = icer, alt
            for k in names:
                o10[k][r0 // 10:(r0 + n) // 10] = blockmean(fw * parts[k], fw, 10)
            if (r0 // STRIP) % 6 == 0:
                log("rows", r0 + n, "/", L1H)
    for lv in (1, 3, 10):
        for v in ("nat", "osm"):
            zn[lv][v] += len(zc[lv][v].flush())
    for k in names:
        o10[k].tofile(os.path.join(OUT, "l10_%s.f32" % k))
    for v, _, _ in VARIANTS:
        c = code(o10[v])
        with gzip.open(os.path.join(OUT, "hazard_%s.bin.gz" % v), "wb", compresslevel=9) as fh:
            fh.write(c.tobytes())
    log("gzip bytes of the whole band (one stream, no tiling):", zn)


if __name__ == "__main__":
    {"dist": stage_dist, "rates": stage_rates}[sys.argv[1]]()
