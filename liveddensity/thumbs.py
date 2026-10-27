"""Thumbnails: one 192x192 WebP per place (country, state, metro, city), 2025 population density.

    python -I thumbs.py                    # prep, download what is needed, render what is missing
    python -I thumbs.py --force            # render everything again
    python -I thumbs.py --only USA,fua:7244 --keep   # just these
    python -I thumbs.py clean              # delete the downloads in raw/thumbs/ and raw/worldpop/

Output: data/thumbs/<id>.webp, the id with ":" replaced by "_" (USA.webp, fua_7244.webp).
Resumable: a place with a file is skipped. Progress goes to stdout and logs/thumbs.log.
NOTES.md ("Thumbnails") has the decisions.

Each picture: the place's outline in a local Lambert azimuthal equal-area projection centred on it,
square, with a margin. Every pixel is the mean density of the cells under it, coloured with the
page's ramp (index.html RAMP / rampAt), 10 to 400k people/km2 (fading to the background between 10
and 1 per km2); zero is the background #0a0a0a.
Pixels outside the outline are dimmed to 30%. The outline is a 1 px light-grey line.

Sources: GHS-POP R2023A 2025 as the page uses it (100 m tiles where a pixel is under 1 km, the 1 km
product above that, 5 km and 25 km block means of it for countries), and WorldPop R2025A over the
ten build.WORLDPOP countries (worldpop.py's own resampling onto the GHS-POP grid).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("GDAL_CACHEMAX", "256")

import argparse
import json
import math
import multiprocessing as mp
import pickle
import re
import shutil
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
import rasterio.features
import rasterio.windows
import requests
import shapely
from PIL import Image, ImageDraw
from pyproj import CRS, Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)            # -I leaves the script folder off the path
import build  # noqa: E402

MOLL = "ESRI:54009"
RAW = os.path.join(HERE, "raw", "thumbs")
TILES = os.path.join(RAW, "tiles")
OUT = os.path.join(HERE, "data", "thumbs")
LOGS = os.path.join(HERE, "logs")
PLANS = os.path.join(RAW, "plans.pkl")
UA = {"User-Agent": "liveddensity-thumbs/1.0 (population density map build)"}

X0, Y0 = -18041000.0, 9000000.0      # GHS-POP grid origin (all products share it)
TILE = 10000                          # 100 m cells in one 1000 km tile
N = 192                               # picture side in pixels
SS = 4                                # supersampling for the outline and the mask
MARGIN = 0.08                         # of the frame, on each side
MIN_SIDE = 4000.0                     # smallest frame, metres (about 4.8 px per 100 m cell)
DIM = 0.30                            # brightness outside the outline
BG = np.array([10, 10, 10], dtype=np.float64)
LINE = np.array([205, 205, 205], dtype=np.float64)
LINE_ALPHA = 0.9
CHAIN_FRAC, CHAIN_MIN = 0.15, 100_000.0   # parts of an outline farther than this from the rest are left out

CELL = [100.0, 1000.0, 5000.0, 25000.0]                       # metres per cell at each level
SHAPE = [(180000, 360820), (18000, 36082), (3600, 7217), (720, 1444)]   # rows, cols per level
MAXK = 10                                                     # sub-samples per pixel side

L1_URL = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/"
          "GHS_POP_E2025_GLOBE_R2023A_54009_1000/V1-0/GHS_POP_E2025_GLOBE_R2023A_54009_1000_V1_0.zip")
TILE_URL = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/"
            "GHS_POP_E2025_GLOBE_R2023A_54009_100/V1-0/tiles/"
            "GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_R{r}_C{c}.zip")
L1_ZIP = os.path.join(RAW, "pop1km.zip")
L1_TIF = os.path.join(RAW, "pop1km.tif")
L2_NPY = os.path.join(RAW, "pop5km.npy")
L3_NPY = os.path.join(RAW, "pop25km.npy")

_logf = None
TRACE = open(os.environ["THUMBS_TRACE"], "w") if os.environ.get("THUMBS_TRACE") else None


def log(*a):
    global _logf
    msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    if _logf is None:
        os.makedirs(LOGS, exist_ok=True)
        _logf = open(os.path.join(LOGS, "thumbs.log"), "a", encoding="utf-8")
    _logf.write(msg + "\n")
    _logf.flush()


def fname(pid):
    """adm1:GAZ+00? -> adm1_GAZ_00_.webp (index.html does the same with a regex)"""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", pid) + ".webp"


# ------------------------------------------------------------------ colour ramp (index.html)

RAMP = [(0.3, .91, .12, 140), (1.6, .83, .16, 110), (2.48, .75, .16, 75),
        (3.18, .67, .18, 45), (3.9, .59, .20, 22), (4.6, .51, .20, 340)]
RAMP = np.array([[x, L, C * math.cos(h * math.pi / 180), C * math.sin(h * math.pi / 180)]
                 for x, L, C, h in RAMP])
X_MIN, X_MAX = 1.0, 5.6


def ramp_rgb(x):
    """index.html rampAt(x): OKLCH stops at log10 density -> sRGB bytes, vectorised."""
    x = np.asarray(x, dtype=np.float64)
    i = np.zeros(x.shape, dtype=np.int64)
    for j in range(len(RAMP) - 2):                 # while (i < len-2 && x > RAMP[i+1][0]) i++
        i = np.where((i == j) & (x > RAMP[j + 1, 0]), j + 1, i)
    a, b = RAMP[i], RAMP[i + 1]
    t = np.clip((x - a[..., 0]) / (b[..., 0] - a[..., 0]), 0, 1)
    L = a[..., 1] + (b[..., 1] - a[..., 1]) * t
    A = a[..., 2] + (b[..., 2] - a[..., 2]) * t
    B = a[..., 3] + (b[..., 3] - a[..., 3]) * t
    l = (L + .3963377774 * A + .2158037573 * B) ** 3
    m = (L - .1055613458 * A - .0638541728 * B) ** 3
    s = (L - .0894841775 * A - 1.291485548 * B) ** 3
    rgb = np.stack([4.0767416621 * l - 3.3077115913 * m + .2309699292 * s,
                    -1.2684380046 * l + 2.6097574011 * m - .3413193965 * s,
                    -.0041960863 * l - .7034186147 * m + 1.707614701 * s], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    rgb = np.where(rgb <= .0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - .055)
    return np.floor(rgb * 255 + .5)          # Math.round


# ------------------------------------------------------------------ geometry prep (main process)

def laea(lon, lat):
    return CRS.from_proj4(f"+proj=laea +lat_0={lat:.6f} +lon_0={lon:.6f} +x_0=0 +y_0=0 "
                          f"+ellps=WGS84 +units=m +no_defs")


def xform(geom, tr):
    def f(xy):
        x, y = tr.transform(xy[:, 0], xy[:, 1])
        bad = ~(np.isfinite(x) & np.isfinite(y))
        if bad.any():       # a vertex on the 180 meridian can round to just outside Mollweide's ellipse
            x2, y2 = tr.transform(xy[bad, 0] * (1 - 1e-7), xy[bad, 1])
            x, y = np.array(x), np.array(y)
            x[bad], y[bad] = x2, y2
        return np.column_stack([x, y])
    return shapely.transform(geom, f)


def polys_of(g):
    return [p for p in shapely.get_parts(g) if p.geom_type == "Polygon" and not p.is_empty]


CORE_SHARE = 0.10      # a part with this share of the people is always kept


def part_pops(parts, l2, l1):
    """People in each part (Mollweide polygons): cell centres of the 5 km level (parts over
    50,000 km2) or the 1 km level inside the polygon, times the cell area."""
    out = []
    for p in parts:
        lv, s, arr = (2, 5000.0, l2) if p.area > 5e10 else (1, 1000.0, None)
        x0, y0, x1, y1 = p.bounds
        c0, c1 = int(math.floor((x0 - X0) / s)), int(math.ceil((x1 - X0) / s))
        r0, r1 = int(math.floor((Y0 - y1) / s)), int(math.ceil((Y0 - y0) / s))
        nr, nc = SHAPE[lv]
        c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, nc), min(r1, nr)
        if c0 >= c1 or r0 >= r1:
            out.append(0.0)
            continue
        if arr is not None:
            a = np.asarray(arr[r0:r1, c0:c1], dtype=np.float64)
        else:
            a = clean_arr(l1.read(1, window=rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0))).astype(np.float64)
        tf = rasterio.Affine(s, 0, X0 + c0 * s, 0, -s, Y0 - r0 * s)
        m = rasterio.features.rasterize([(p, 1)], out_shape=a.shape, transform=tf, dtype="uint8") > 0
        out.append(float(a[m].sum() * (s / 1000.0) ** 2))
    return np.array(out)


def choose_parts(parts, inv, pops=None, hop=None):
    """The people's part (most people, else the biggest), every part with CORE_SHARE of the
    people, and every part reached from those by hops of at most max(0.15 x the main part's
    diagonal, 100 km). Distances in a projection centred on the main part."""
    if len(parts) == 1:
        return parts
    areas = np.array([p.area for p in parts])
    main = int(np.argmax(areas))
    core = {main}
    if pops is not None and pops.sum() > 0:
        main = int(np.argmax(pops))
        core = {main} | {int(i) for i in np.nonzero(pops >= CORE_SHARE * pops.sum())[0]}
        if hop:
            core = {main}
    c = parts[main].centroid
    lon, lat = inv.transform(c.x, c.y)
    tr = Transformer.from_crs(MOLL, laea(lon, lat), always_xy=True)
    P = [xform(p, tr) for p in parts]
    for i, q in enumerate(P):        # far side of the globe from the centre: not reachable by hops
        if not np.isfinite(shapely.get_coordinates(q)).all():
            P[i] = shapely.Point(1e12, 1e12) if i != main else q
    x0, y0, x1, y1 = P[main].bounds
    T = hop or max(CHAIN_FRAC * math.hypot(x1 - x0, y1 - y0), CHAIN_MIN)
    P = [shapely.simplify(q, T * 0.01) if q.geom_type == "Polygon" else q for q in P]   # distances need not be exact
    tree = shapely.STRtree(P)
    got, todo = set(core), list(core)
    while todo:
        i = todo.pop()
        for j in tree.query(P[i], predicate="dwithin", distance=T):
            if j not in got:
                got.add(int(j))
                todo.append(int(j))
    return [parts[i] for i in sorted(got)]


def frame_of(parts, inv):
    """Projection centre, then the square frame (centre and side, metres) round the outline."""
    # centred on the area-weighted middle of the parts (Mollweide is equal-area); the frame then
    # sits on the middle of the outline's box in that projection
    a = np.array([p.area for p in parts])
    cs = shapely.centroid(np.array(parts, dtype=object))
    # parts split at 180 sit at both ends of the map, so average the centroids as points on the sphere
    ll = np.array([inv.transform(x, y) for x, y in zip(shapely.get_x(cs), shapely.get_y(cs))])
    la, lo = np.radians(ll[:, 1]), np.radians(ll[:, 0])
    v = np.array([np.average(np.cos(la) * np.cos(lo), weights=a),
                  np.average(np.cos(la) * np.sin(lo), weights=a),
                  np.average(np.sin(la), weights=a)])
    lon, lat = math.degrees(math.atan2(v[1], v[0])), math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))
    tr = Transformer.from_crs(MOLL, laea(lon, lat), always_xy=True)
    g = shapely.union_all([xform(p, tr) for p in parts])
    x0, y0, x1, y1 = g.bounds
    side = max(x1 - x0, y1 - y0) / (1 - 2 * MARGIN)
    return lon, lat, (x0 + x1) / 2, (y0 + y1) / 2, max(side, MIN_SIDE)


def level_of(side):
    p = side / N
    if p < 1000:
        return 0, min(MAXK, max(1, math.ceil(p / CELL[0])))
    for lv in (3, 2, 1):
        if p >= CELL[lv]:
            return lv, min(MAXK, max(1, math.ceil(p / CELL[lv])))


def tiles_of(lon, lat, cx, cy, side):
    """100 m tiles under a frame (a coarse grid over it, a little larger)."""
    tr = Transformer.from_crs(laea(lon, lat), MOLL, always_xy=True)
    h = side * 0.58
    g = np.linspace(-h, h, 160)
    X, Y = np.meshgrid(cx + g, cy + g)
    mx, my = tr.transform(X.ravel(), Y.ravel())
    ok = np.isfinite(mx) & np.isfinite(my)
    c = np.floor((mx[ok] - X0) / 100).astype(np.int64)
    r = np.floor((Y0 - my[ok]) / 100).astype(np.int64)
    k = (r >= 0) & (c >= 0) & (r < SHAPE[0][0]) & (c < SHAPE[0][1])
    return sorted({(int(a), int(b)) for a, b in zip(r[k] // TILE, c[k] // TILE)})


def make_plans():
    t0 = time.time()
    keys, names, cparts, ccid, _ = build.load_countries(MOLL)
    geoms = shapely.from_wkb(cparts)
    kix = {k: i + 1 for i, k in enumerate(keys)}
    tree = shapely.STRtree(geoms)
    by_country = {}
    for g, c in zip(geoms, ccid):
        by_country.setdefault(keys[int(c) - 1], []).append(g)
    log(f"countries loaded ({time.time() - t0:.0f}s)")
    inv = Transformer.from_crs(MOLL, "EPSG:4326", always_xy=True)

    hist = json.load(open(build.OUT, encoding="utf-8"))
    index = json.load(open(os.path.join(HERE, "data", "sub", "index.json"), encoding="utf-8"))["units"]
    c = pickle.load(open(os.path.join(LOGS, "units.pkl"), "rb"))
    geo = {u["id"]: u for u in c["adm1"] + c["fua"]}
    for u in pickle.load(open(os.path.join(LOGS, "city_units.pkl"), "rb")):
        geo[u["id"]] = u

    places = [{"id": k, "kind": "country", "name": v["name"], "parent": None,
               "parts": by_country[k]} for k, v in hist["countries"].items()]
    missing = []
    for e in index:
        u = geo.get(e["id"])
        if u is None:
            missing.append(e["id"])
            continue
        if u.get("whole") or not u["parts"]:
            parts = by_country[e["parent"]]
            clip = False
        else:
            parts = [p for w in u["parts"] for p in polys_of(shapely.make_valid(shapely.from_wkb(w)))]
            clip = True
        places.append({"id": e["id"], "kind": e["kind"], "name": e["name"], "parent": e["parent"],
                       "parts": parts, "clip": clip})
    if missing:
        log("NO GEOMETRY for", len(missing), missing[:20])

    ensure_levels()
    l2 = np.load(L2_NPY, mmap_mode="r")
    l1 = rasterio.open(L1_TIF)
    plans, nclip = [], 0
    for n, pl in enumerate(places, 1):
        tu = time.time()
        if TRACE:
            TRACE.write(pl["id"] + "\n")
            TRACE.flush()
        parts = pl["parts"]
        if pl.get("clip"):
            # as the build did: a unit holds only its own country's cells, so what lies inside
            # another country's polygon is cut off
            g = shapely.union_all(parts)
            cand = tree.query(g)
            others = [geoms[i] for i in cand if ccid[i] != kix[pl["parent"]]]
            if others:
                bb = g.bounds
                oth = [o for o in shapely.clip_by_rect(others, *bb) if not o.is_empty]
                if oth:
                    cut = shapely.difference(g, shapely.make_valid(shapely.union_all(oth)))
                    if not cut.is_empty and polys_of(cut):
                        if cut.area < g.area * 0.999:
                            nclip += 1
                        parts = polys_of(cut)
        total = len(parts)
        pops = None
        if total > 1:
            x0, y0, x1, y1 = shapely.union_all(parts).bounds
            if math.hypot(x1 - x0, y1 - y0) > 200_000:        # nothing far to leave out below that
                pops = part_pops(parts, l2, l1)
        parts = choose_parts(parts, inv, pops)
        area_all = sum(p.area for p in pl["parts"])
        lon, lat, cx, cy, side = frame_of(parts, inv)
        lv, k = level_of(side)
        plan = {"id": pl["id"], "kind": pl["kind"], "name": pl["name"], "parent": pl["parent"],
                "wkb": shapely.to_wkb(shapely.MultiPolygon(parts)), "lon": lon, "lat": lat,
                "cx": cx, "cy": cy, "side": side, "level": lv, "k": k,
                "parts_total": total, "parts_kept": len(parts),
                "area_kept": sum(p.area for p in parts) / area_all if area_all else 1.0}
        if lv == 0:
            plan["tiles"] = tiles_of(lon, lat, cx, cy, side)
        plans.append(plan)
        if time.time() - tu > 3:
            log(f"  slow: {pl['id']} {pl['name']} {time.time() - tu:.1f}s, {total} parts")
        if n % 500 == 0:
            log(f"prep {n}/{len(places)} ({time.time() - t0:.0f}s)")
    log(f"{nclip} units clipped at a border")
    names_ = [fname(p["id"]) for p in plans]
    assert len(set(names_)) == len(names_), "two places share a file name"
    wp = {iso: shapely.to_wkb(shapely.MultiPolygon(polys_of(shapely.union_all(by_country[iso]))))
          for iso in build.WORLDPOP}
    with open(PLANS + ".tmp", "wb") as f:
        pickle.dump({"plans": plans, "wp_parents": wp}, f)
    os.replace(PLANS + ".tmp", PLANS)
    log(f"{len(plans)} plans in {time.time() - t0:.0f}s")


SPARSE = 0.01      # an outline covering less of its frame than this is cut down to its main part


def refine_sparse(d):
    """Scattered atolls (Kiribati, Marshall Islands, Tuvalu, ...) come out as specks in a frame
    hundreds of km wide. Those are framed on the part with most people and parts within 15 km."""
    inv = Transformer.from_crs(MOLL, "EPSG:4326", always_xy=True)
    l2 = np.load(L2_NPY, mmap_mode="r")
    l1 = rasterio.open(L1_TIF)
    for p in d["plans"]:
        if p["parts_kept"] < 2:
            continue
        g = shapely.from_wkb(p["wkb"])
        parts = polys_of(g)
        if g.area / p["side"] ** 2 >= SPARSE:
            continue
        sel = choose_parts(parts, inv, part_pops(parts, l2, l1), hop=15_000.0)
        lon, lat, cx, cy, side = frame_of(sel, inv)
        lv, k = level_of(side)
        new = shapely.MultiPolygon(sel)
        log(f"refine {p['id']} {p['name']}: {len(parts)} -> {len(sel)} parts, frame {p['side'] / 1000:.0f} -> {side / 1000:.0f} km")
        p.update(wkb=shapely.to_wkb(new), lon=lon, lat=lat, cx=cx, cy=cy, side=side, level=lv, k=k,
                 area_kept=p["area_kept"] * new.area / g.area, parts_kept=len(sel))
    d["refined"] = True
    with open(PLANS + ".tmp", "wb") as f:
        pickle.dump(d, f)
    os.replace(PLANS + ".tmp", PLANS)


def load_plans():
    if not os.path.exists(PLANS):
        make_plans()
    with open(PLANS, "rb") as f:
        d = pickle.load(f)
    if not d.get("refined"):
        ensure_levels()
        refine_sparse(d)
    return d


# ------------------------------------------------------------------ downloads

def download(url, dest, tries=6):
    tmp = dest + ".part"
    for a in range(tries):
        try:
            with requests.get(url, headers=UA, stream=True, timeout=(30, 120)) as r:
                if r.status_code == 404:
                    return False
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for b in r.iter_content(1 << 20):
                        f.write(b)
            os.replace(tmp, dest)
            return True
        except (requests.RequestException, OSError) as e:
            log(f"  download retry {a + 1}: {os.path.basename(dest)} {e.__class__.__name__}")
            time.sleep(5 * (a + 1))
    raise RuntimeError("download failed: " + url)


def unzip_tif(zpath, dest):
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".tif")]
        with z.open(names[0]) as src, open(dest + ".part", "wb") as out:
            shutil.copyfileobj(src, out, 1 << 22)
    os.replace(dest + ".part", dest)


def clean_arr(a):
    a = np.nan_to_num(a.astype(np.float32), nan=0.0)
    a[a < 0] = 0
    return a


def coarsen(arr, off_r, off_c, f):
    """Block means over f x f cells on a grid that starts at 0, from an array whose first cell is
    at (off_r, off_c) of the finer grid. Returns the array and its offset on the coarser grid."""
    R0, C0 = off_r // f, off_c // f
    pr, pc = off_r - R0 * f, off_c - C0 * f
    h, w = arr.shape[0] + pr, arr.shape[1] + pc
    H2, W2 = -(-h // f), -(-w // f)
    pad = np.zeros((H2 * f, W2 * f), np.float32)
    pad[pr:pr + arr.shape[0], pc:pc + arr.shape[1]] = arr
    return pad.reshape(H2, f, W2, f).mean(axis=(1, 3)), R0, C0


def ensure_levels():
    if not os.path.exists(L1_TIF):
        if not os.path.exists(L1_ZIP):
            log("downloading the 1 km product")
            download(L1_URL, L1_ZIP)
        unzip_tif(L1_ZIP, L1_TIF)
        os.remove(L1_ZIP)
    if not (os.path.exists(L2_NPY) and os.path.exists(L3_NPY)):
        log("building the 5 km and 25 km levels")
        l2 = np.zeros(SHAPE[2], np.float32)
        with rasterio.open(L1_TIF) as ds:
            assert (ds.height, ds.width) == SHAPE[1], (ds.height, ds.width)
            for r in range(0, ds.height, 1000):
                a = clean_arr(ds.read(1, window=rasterio.windows.Window(0, r, ds.width, 1000)))
                m, _, _ = coarsen(a, 0, 0, 5)
                l2[r // 5:r // 5 + m.shape[0], :m.shape[1]] = m
        l3, _, _ = coarsen(l2, 0, 0, 5)
        assert l3.shape == SHAPE[3], l3.shape
        np.save(L2_NPY, l2)
        np.save(L3_NPY, l3)


def tile_path(tr, tc):
    return os.path.join(TILES, f"R{tr + 1}_C{tc + 1}.tif")


def fetch_tile(t):
    tr, tc = t
    tif = tile_path(tr, tc)
    if os.path.exists(tif) or os.path.exists(tif + ".none"):
        return t, True
    z = tif[:-4] + ".zip"
    if not download(TILE_URL.format(r=tr + 1, c=tc + 1), z):
        open(tif + ".none", "w").close()        # all water: GHSL has no tile
        return t, True
    unzip_tif(z, tif)
    os.remove(z)
    return t, True


def patch_files(iso):
    return (os.path.join(RAW, f"wp_{iso}_100m.tif"), os.path.join(RAW, f"wp_{iso}_1km.npy"),
            os.path.join(RAW, f"wp_{iso}.json"))


def make_patch(iso):
    """WorldPop R2025A 2025 for one country, summed onto the GHS-POP 100 m grid (worldpop.py's
    resampling): a float32 GeoTIFF, plus the 1 km block sums (people/km2) as .npy."""
    import worldpop
    from affine import Affine
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform_bounds
    tif, k1p, meta = patch_files(iso)
    if os.path.exists(meta):
        return
    log(f"WorldPop patch for {iso}")
    path = worldpop.fetch(iso, 2025)
    with rasterio.open(path) as src:
        l, b, r, t = transform_bounds(src.crs, MOLL, *src.bounds, densify_pts=101)
        c0 = int(math.floor((l - X0) / 1000)) * 10 - 10
        c1 = int(math.ceil((r - X0) / 1000)) * 10 + 10
        r0 = int(math.floor((Y0 - t) / 1000)) * 10 - 10
        r1 = int(math.ceil((Y0 - b) / 1000)) * 10 + 10
        from affine import Affine
        tf = Affine(100, 0, X0 + c0 * 100, 0, -100, Y0 - r0 * 100)
        H, W = r1 - r0, c1 - c0
        vrt = WarpedVRT(src, crs=MOLL, transform=tf, width=W, height=H, resampling=Resampling.sum,
                        src_nodata=src.nodata, nodata=-1.0, dtype="float64")
        k1 = np.zeros((H // 10, W // 10), np.float32)
        prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="float32", crs=MOLL,
                    transform=tf, tiled=True, blockxsize=256, blockysize=256, compress="deflate")
        with rasterio.open(tif + ".part", "w", **prof) as dst:
            for row0 in range(0, H, 5000):
                for col0 in range(0, W, 5000):
                    h, w = min(5000, H - row0), min(5000, W - col0)
                    win = rasterio.windows.Window(col0, row0, w, h)
                    pop = vrt.read(1, window=win)
                    pop[~(pop > 0)] = 0.0
                    dst.write(pop.astype(np.float32), 1, window=win)
                    k1[row0 // 10:(row0 + h) // 10, col0 // 10:(col0 + w) // 10] = \
                        pop.reshape(h // 10, 10, w // 10, 10).sum(axis=(1, 3))
        vrt.close()
    os.replace(tif + ".part", tif)
    np.save(k1p, k1)
    json.dump({"r0": r0, "c0": c0, "H": H, "W": W}, open(meta, "w"))


# ------------------------------------------------------------------ readers (density, people/km2)

class ArrReader:
    def __init__(self, arr, off_r=0, off_c=0):
        self.a, self.off_r, self.off_c = arr, off_r, off_c

    def read(self, r0, r1, c0, c1):
        out = np.zeros((r1 - r0, c1 - c0), np.float32)
        a0, a1 = max(r0, self.off_r), min(r1, self.off_r + self.a.shape[0])
        b0, b1 = max(c0, self.off_c), min(c1, self.off_c + self.a.shape[1])
        if a0 < a1 and b0 < b1:
            out[a0 - r0:a1 - r0, b0 - c0:b1 - c0] = \
                self.a[a0 - self.off_r:a1 - self.off_r, b0 - self.off_c:b1 - self.off_c]
        return out


class TifReader:
    def __init__(self, path, off_r=0, off_c=0, mult=1.0):
        self.path, self.off_r, self.off_c, self.mult = path, off_r, off_c, mult

    def read(self, r0, r1, c0, c1):
        out = np.zeros((r1 - r0, c1 - c0), np.float32)
        with rasterio.open(self.path) as ds:
            a0, a1 = max(r0, self.off_r), min(r1, self.off_r + ds.height)
            b0, b1 = max(c0, self.off_c), min(c1, self.off_c + ds.width)
            if a0 < a1 and b0 < b1:
                w = rasterio.windows.Window(b0 - self.off_c, a0 - self.off_r, b1 - b0, a1 - a0)
                out[a0 - r0:a1 - r0, b0 - c0:b1 - c0] = clean_arr(ds.read(1, window=w)) * self.mult
        return out


class TileReader:
    """The 100 m tiles, as density (people per cell x 100)."""

    def read(self, r0, r1, c0, c1):
        out = np.zeros((r1 - r0, c1 - c0), np.float32)
        for tr in range(r0 // TILE, (r1 - 1) // TILE + 1):
            for tc in range(c0 // TILE, (c1 - 1) // TILE + 1):
                p = tile_path(tr, tc)
                if not os.path.exists(p):
                    if not os.path.exists(p + ".none"):
                        raise FileNotFoundError(p)
                    continue
                a0, a1 = max(r0, tr * TILE), min(r1, tr * TILE + TILE)
                b0, b1 = max(c0, tc * TILE), min(c1, tc * TILE + TILE)
                with rasterio.open(p) as ds:
                    a1, b1 = min(a1, tr * TILE + ds.height), min(b1, tc * TILE + ds.width)
                    if a0 >= a1 or b0 >= b1:
                        continue
                    w = rasterio.windows.Window(b0 - tc * TILE, a0 - tr * TILE, b1 - b0, a1 - a0)
                    out[a0 - r0:a1 - r0, b0 - c0:b1 - c0] = clean_arr(ds.read(1, window=w)) * 100.0
        return out


G = {}


def init_worker(wp_parents):
    G["ghsl"] = [TileReader(), TifReader(L1_TIF), ArrReader(np.load(L2_NPY, mmap_mode="r")),
                 ArrReader(np.load(L3_NPY, mmap_mode="r"))]
    G["wp_parents"] = wp_parents
    G["wp"] = {}


def wp_readers(iso):
    if iso not in G["wp"]:
        tif, k1p, meta = patch_files(iso)
        m = json.load(open(meta))
        k1 = np.load(k1p)
        l2, r2, c2 = coarsen(k1, m["r0"] // 10, m["c0"] // 10, 5)
        l3, r3, c3 = coarsen(l2, r2, c2, 5)
        G["wp"][iso] = [TifReader(tif, m["r0"], m["c0"]), ArrReader(k1, m["r0"] // 10, m["c0"] // 10),
                        ArrReader(l2, r2, c2), ArrReader(l3, r3, c3)]
    return G["wp"][iso]


def sample(reader, level, rows, cols):
    """Density at integer cell indices (any shape); 0 outside the grid. A read per half of the
    world, so a frame over the antimeridian does not read the whole width."""
    nr, nc = SHAPE[level]
    out = np.zeros(rows.shape, np.float32)
    valid = (rows >= 0) & (cols >= 0) & (rows < nr) & (cols < nc)
    # 100 m tiles: only the tiles a point falls in are read (a window over the points' box would
    # take in tiles that are not downloaded)
    groups = [valid & (rows // TILE == tr) & (cols // TILE == tc)
              for tr, tc in sorted({(int(a), int(b)) for a, b in zip(rows[valid] // TILE, cols[valid] // TILE)})] \
        if level == 0 else [valid & (cols < nc // 2), valid & (cols >= nc // 2)]
    for m in groups:
        if not m.any():
            continue
        r, c = rows[m], cols[m]
        r0, c0 = int(r.min()), int(c.min())
        arr = reader.read(r0, int(r.max()) + 1, c0, int(c.max()) + 1)
        out[m] = arr[r - r0, c - c0]
    return out


# ------------------------------------------------------------------ rendering

def cover(polys, x0, y1, p, width=0):
    """Supersampled coverage (0..1, N x N) of polygons given in projected metres; width > 0 draws
    their rings as lines of that many output pixels instead."""
    im = Image.new("L", (N * SS, N * SS), 0)
    d = ImageDraw.Draw(im)
    s = SS / p

    def pts(ring):
        a = np.asarray(ring.coords)
        return [(float(x), float(y)) for x, y in np.column_stack([(a[:, 0] - x0) * s, (y1 - a[:, 1]) * s])]
    for poly in polys:
        if width:
            d.line(pts(poly.exterior), fill=255, width=max(1, int(round(width * SS))), joint="curve")
            for r in poly.interiors:
                d.line(pts(r), fill=255, width=max(1, int(round(width * SS))), joint="curve")
        else:
            d.polygon(pts(poly.exterior), fill=255)
            for r in poly.interiors:
                d.polygon(pts(r), fill=0)
    return np.asarray(im.reduce(SS), dtype=np.float64) / 255.0


def render(plan):
    lon, lat, cx, cy, side = plan["lon"], plan["lat"], plan["cx"], plan["cy"], plan["side"]
    lv, k = plan["level"], plan["k"]
    crs = laea(lon, lat)
    to_l = Transformer.from_crs(MOLL, crs, always_xy=True)
    from_l = Transformer.from_crs(crs, MOLL, always_xy=True)
    p = side / N
    x0, y1 = cx - side / 2, cy + side / 2

    g = shapely.union_all(shapely.get_parts(xform(shapely.from_wkb(plan["wkb"]), to_l)))
    eps = max(0.5, p * 0.03)                              # closes the seam of a part split at 180
    g = g.buffer(eps, 2).buffer(-eps, 2)
    polys = polys_of(shapely.simplify(g, p * 0.2))

    # mean density under each pixel: k x k points per pixel, each read from the cell it falls in
    n = N * k
    xs = x0 + (np.arange(n) + 0.5) * (p / k)
    ys = y1 - (np.arange(n) + 0.5) * (p / k)
    X, Y = np.meshgrid(xs, ys)
    mx, my = from_l.transform(X.ravel(), Y.ravel())
    mx, my = np.asarray(mx), np.asarray(my)
    ok = np.isfinite(mx) & np.isfinite(my)
    cols = np.full(mx.shape, -1, np.int64)
    rows = np.full(mx.shape, -1, np.int64)
    cell = CELL[lv]
    cols[ok] = np.floor((mx[ok] - X0) / cell).astype(np.int64)
    rows[ok] = np.floor((Y0 - my[ok]) / cell).astype(np.int64)
    rows, cols = rows.reshape(n, n), cols.reshape(n, n)

    def mean(a):
        return a.reshape(N, k, N, k).mean(axis=(1, 3))

    dens = mean(sample(G["ghsl"][lv], lv, rows, cols))
    if plan["parent"] in G["wp_parents"]:
        wp = mean(sample(wp_readers(plan["parent"])[lv], lv, rows, cols))
        par = shapely.from_wkb(G["wp_parents"][plan["parent"]])
        par = polys_of(shapely.union_all(shapely.get_parts(xform(par, to_l))))
        inside = cover(par, x0, y1, p) > 0.5
        dens = np.where(inside | (wp > 0), wp, dens)

    rgb = np.tile(BG, (N, N, 1))
    pos = dens > 0
    if pos.any():
        x = np.log10(dens[pos])
        # the page's ramp starts at 10/km2; below that the colour fades to the background by 1/km2,
        # so a thinly peopled country is not one flat pale sheet
        w = np.clip(x, 0, 1)[:, None]
        rgb[pos] = BG + (ramp_rgb(np.clip(x, X_MIN, X_MAX)) - BG) * w
    ins = cover(polys, x0, y1, p)
    rgb = BG + (rgb - BG) * (DIM + (1 - DIM) * ins)[..., None]
    a = cover(polys, x0, y1, p, width=1.0) * LINE_ALPHA
    rgb = rgb * (1 - a[..., None]) + LINE * a[..., None]
    return Image.fromarray(np.clip(np.floor(rgb + .5), 0, 255).astype(np.uint8), "RGB")


def work(plan):
    try:
        t0 = time.time()
        im = render(plan)
        path = os.path.join(OUT, fname(plan["id"]))
        im.save(path + ".part", "WEBP", quality=80, method=4)
        os.replace(path + ".part", path)
        return plan["id"], None, time.time() - t0
    except Exception as e:     # noqa: BLE001
        import traceback
        return plan["id"], f"{e.__class__.__name__}: {e}\n{traceback.format_exc(limit=3)}", 0.0


# ------------------------------------------------------------------ main

def run_batch(pool, plans, label, state):
    t0 = time.time()
    for i, (pid, err, dt) in enumerate(pool.imap_unordered(work, plans, chunksize=2), 1):
        state["done"] += 1
        if err:
            state["failed"].append((pid, err))
            log(f"FAILED {pid}: {err}")
        if i % 200 == 0 or i == len(plans):
            log(f"{label}: {i}/{len(plans)} ({time.time() - t0:.0f}s); total {state['done']}/{state['total']}, "
                f"failed {len(state['failed'])}")


def clean():
    for d in (RAW, build.os.path.join(HERE, "raw", "worldpop")):
        if os.path.isdir(d):
            shutil.rmtree(d)
            log("deleted", d)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "clean", "prep"])
    ap.add_argument("--force", action="store_true", help="render places that already have a file")
    ap.add_argument("--only", help="comma-separated place ids")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--reprep", action="store_true", help="rebuild the outline plans")
    ap.add_argument("--max-tiles", type=int, default=40)
    ap.add_argument("--keep", action="store_true", help="keep tiles after each batch")
    a = ap.parse_args()
    if a.cmd == "clean":
        return clean()
    if a.reprep and os.path.exists(PLANS):
        os.remove(PLANS)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TILES, exist_ok=True)
    d = load_plans()
    plans, wp_parents = d["plans"], d["wp_parents"]
    if a.cmd == "prep":
        return
    ids = set(a.only.split(",")) if a.only else None
    todo = [p for p in plans if (ids is None or p["id"] in ids)
            and (a.force or ids is not None or not os.path.exists(os.path.join(OUT, fname(p["id"]))))]
    if ids is not None:
        todo = [p for p in plans if p["id"] in ids]
    for p in todo:          # from the frame, so a change to tiles_of needs no new prep
        if p["level"] == 0:
            p["tiles"] = tiles_of(p["lon"], p["lat"], p["cx"], p["cy"], p["side"])
    by_level = {lv: sum(1 for p in todo if p["level"] == lv) for lv in range(4)}
    log(f"{len(plans)} places, {len(todo)} to render; by level {by_level}; workers {a.workers}")
    if not todo:
        return
    ensure_levels()
    for iso in build.WORLDPOP:
        if any(p["parent"] == iso for p in todo):
            make_patch(iso)
    state = {"done": 0, "total": len(todo), "failed": []}
    t0 = time.time()
    with mp.Pool(a.workers, init_worker, (wp_parents,)) as pool:
        coarse = [p for p in todo if p["level"] > 0]
        if coarse:
            run_batch(pool, coarse, "coarse", state)
        l0 = sorted([p for p in todo if p["level"] == 0], key=lambda p: (p["tiles"][0][1], p["tiles"][0][0]))
        batches, cur, tset = [], [], set()
        for p in l0:
            if cur and len(tset | set(p["tiles"])) > a.max_tiles:
                batches.append((cur, tset))
                cur, tset = [], set()
            cur.append(p)
            tset |= set(p["tiles"])
        if cur:
            batches.append((cur, tset))
        log(f"{len(l0)} places need 100 m tiles, in {len(batches)} batches; "
            f"{len({t for _, s in batches for t in s})} distinct tiles")
        for bi, (batch, tset) in enumerate(batches):
            need = [t for t in sorted(tset) if not (os.path.exists(tile_path(*t))
                                                    or os.path.exists(tile_path(*t) + ".none"))]
            if need:
                log(f"batch {bi + 1}/{len(batches)}: {len(batch)} places, downloading {len(need)} of {len(tset)} tiles")
                with ThreadPoolExecutor(4) as ex:
                    list(ex.map(fetch_tile, need))
            run_batch(pool, batch, f"batch {bi + 1}/{len(batches)}", state)
            if not a.keep:
                nxt = batches[bi + 1][1] if bi + 1 < len(batches) else set()
                for t in tset - nxt:
                    for ext in ("", ".none"):
                        pth = tile_path(*t) + ext
                        if os.path.exists(pth):
                            os.remove(pth)
    log(f"done: {state['done']} rendered, {len(state['failed'])} failed, {(time.time() - t0) / 60:.1f} min")
    if state["failed"]:
        json.dump(state["failed"], open(os.path.join(LOGS, "thumbs_failed.json"), "w"), indent=1)


if __name__ == "__main__":
    mp.freeze_support()
    main()
