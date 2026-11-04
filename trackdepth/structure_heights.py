"""Measured heights of elevated structures and bridges, from the 2017 NYC lidar point cloud.

Stages (python structure_heights.py <stage>):

  samples   sample points every SPACING m along elevated / bridge / embankment OSM ways
  plan      which EPT nodes touch the track corridor, and how big they are
  fetch     download those nodes (plain HTTPS, resumable) to raw/lidar/ept/
  measure   per sample: top of structure from the first returns, ground from the 3 m DEM, rail
  all       samples, plan, fetch, measure, then write evidence/structure_heights.csv

The bare-earth ground the model uses has structures removed, so the lidar surface over the track
centreline minus nothing is the structure top (NAVD88 m; the point cloud is NAVD88 metres). Rail
height = highest clean surface over the track footprint minus a small deck-to-rail offset (see
TOP_TO_RAIL). Bridge rows from published figures are added by bridges.csv (hand-written, sources
in the file) and merged into the output.

Source: https://s3-us-west-2.amazonaws.com/usgs-lidar-public/NY_NewYorkCity/ept.json
(Entwine Point Tiles, 4.75 billion points, EPSG:3857 horizontal, collected 2017).
"""
import collections
import concurrent.futures as cf
import json
import math
import pathlib
import sys
import time

import numpy as np
import requests
from pyproj import Transformer

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
RAW = HERE / "raw"
LIDAR = RAW / "lidar"
EPT_DIR = LIDAR / "ept"
HIER_DIR = EPT_DIR / "hier"
LAZ_DIR = EPT_DIR / "laz"
EVID = HERE / "evidence"
BASE = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/NY_NewYorkCity"
UA = "trackdepth/0.1 (research; public lidar bucket)"

SPACING = 30.0        # m between samples along a way
BUFFER = 18.0         # m around each sample in which EPT nodes are fetched
MAXDEPTH = 12
MINDEPTH = 6          # shallower EPT nodes hold points 8 m or more apart

to_3857 = Transformer.from_crs(4326, 3857, always_xy=True)
to_utm = Transformer.from_crs(4326, 32618, always_xy=True)
to_ll = Transformer.from_crs(32618, 4326, always_xy=True)

_session = None


def session():
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = UA
    return _session


def get(url, dest, tries=4):
    """Download url to dest (skip if present), through a .part file."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for a in range(tries):
        try:
            r = session().get(url, timeout=120)
            if r.status_code == 200:
                tmp = dest.with_suffix(dest.suffix + ".part")
                tmp.write_bytes(r.content)
                tmp.replace(dest)
                return dest
            if r.status_code in (403, 404):
                raise RuntimeError(f"{r.status_code} for {url}")
        except requests.RequestException:
            pass
        time.sleep(2 * (a + 1))
    raise RuntimeError(f"failed: {url}")


# ---------- the ways ----------

def structure_ways():
    """OSM subway/SIR ways that are bridges, viaducts, trestles or embankments (not tunnels)."""
    import build
    out = []
    for w in build.load_tracks():
        t = w["tags"]
        if t.get("tunnel") in ("yes", "building_passage", "culvert"):
            continue
        br = t.get("bridge")
        if (br and br != "no") or t.get("embankment") == "yes":
            out.append(w)
    return out


def sample_way(w):
    """Points every SPACING m along the way: arrays x, y (UTM 18N), lon, lat, dist along, heading."""
    g = w["geometry"]
    lon = np.array([p["lon"] for p in g]); lat = np.array([p["lat"] for p in g])
    x, y = to_utm.transform(lon, lat)
    seg = np.hypot(np.diff(x), np.diff(y))
    d = np.concatenate([[0], np.cumsum(seg)])
    L = d[-1]
    if L < 1:
        return None
    n = max(1, int(round(L / SPACING)))
    s = np.linspace(0, L, n + 1)
    s = (s[:-1] + s[1:]) / 2 if n > 1 else np.array([L / 2])  # mid-bin, so no doubles at junctions
    sx = np.interp(s, d, x); sy = np.interp(s, d, y)
    h = 1.0
    ax = np.interp(np.clip(s + h, 0, L), d, x) - np.interp(np.clip(s - h, 0, L), d, x)
    ay = np.interp(np.clip(s + h, 0, L), d, y) - np.interp(np.clip(s - h, 0, L), d, y)
    nrm = np.hypot(ax, ay); nrm[nrm == 0] = 1
    lo, la = to_ll.transform(sx, sy)
    return dict(x=sx, y=sy, lon=np.asarray(lo), lat=np.asarray(la), s=s, tx=ax / nrm, ty=ay / nrm, L=L)


def all_samples():
    rows = []
    for w in structure_ways():
        sm = sample_way(w)
        if sm is None:
            continue
        for i in range(len(sm["x"])):
            rows.append((w["id"], i, sm["x"][i], sm["y"][i], sm["lon"][i], sm["lat"][i],
                         sm["tx"][i], sm["ty"][i], sm["s"][i], sm["L"]))
    return rows


# ---------- EPT ----------

def ept_info():
    p = EPT_DIR / "ept.json"
    get(BASE + "/ept.json", p)
    return json.loads(p.read_text())


def cube(info):
    b = info["bounds"]
    return b[0], b[1], b[2], b[3] - b[0]  # x0, y0, z0, size


def corridor_cells(px, py, depth, info, buf):
    """Set of (ix, iy) cells of the given depth touched by boxes of +-buf around the points (3857 m)."""
    x0, y0, z0, size = cube(info)
    cs = size / 2 ** depth
    cells = set()
    for dx in (-buf, buf):
        for dy in (-buf, buf):
            ix = np.floor((px + dx - x0) / cs).astype(int); iy = np.floor((py + dy - y0) / cs).astype(int)
            cells.update(zip(ix.tolist(), iy.tolist()))
    # a box can span cells whose corners are not hit only if buf > cell, not so here for depth<=12
    # except at the shallow depths, where the cell list is filled in below
    if cs < 2 * buf * 1.01:
        return cells
    ix = np.floor((px - x0) / cs).astype(int); iy = np.floor((py - y0) / cs).astype(int)
    lo_x = np.floor((px - buf - x0) / cs).astype(int); hi_x = np.floor((px + buf - x0) / cs).astype(int)
    lo_y = np.floor((py - buf - y0) / cs).astype(int); hi_y = np.floor((py + buf - y0) / cs).astype(int)
    for a, b, c, d in zip(lo_x, hi_x, lo_y, hi_y):
        for i in range(a, b + 1):
            for j in range(c, d + 1):
                cells.add((i, j))
    return cells


def load_hierarchy(info, px, py, buf, maxdepth=MAXDEPTH, pool=None):
    """Walk the hierarchy files that touch the corridor. Returns {key: points}."""
    cellsets = {d: corridor_cells(px, py, d, info, buf) for d in range(0, maxdepth + 1)}

    def hit(key):
        d, x, y, z = map(int, key.split("-"))
        return d <= maxdepth and (x, y) in cellsets[d]

    nodes = {}
    todo = ["0-0-0-0"]
    seen = set()
    while todo:
        batch = [k for k in todo if k not in seen]
        seen.update(batch)
        todo = []
        paths = {k: HIER_DIR / f"{k}.json" for k in batch}
        with cf.ThreadPoolExecutor(8) as ex:
            list(ex.map(lambda k: get(f"{BASE}/ept-hierarchy/{k}.json", paths[k]), batch))
        for k in batch:
            h = json.loads(paths[k].read_text())
            for kk, v in h.items():
                if not hit(kk):
                    continue
                if v == -1:
                    todo.append(kk)
                elif v > 0:
                    nodes[kk] = v
    return nodes


def decode_node(key):
    import laspy
    p = LAZ_DIR / f"{key}.laz"
    las = laspy.read(str(p))
    return las


def main_plan():
    info = ept_info()
    rows = all_samples()
    print(len(rows), "samples", len({r[0] for r in rows}), "ways")
    lon = np.array([r[4] for r in rows]); lat = np.array([r[5] for r in rows])
    px, py = to_3857.transform(lon, lat)
    buf = BUFFER * 1.32
    nodes = load_hierarchy(info, np.asarray(px), np.asarray(py), buf)
    by = collections.Counter(); pts = collections.Counter()
    for k, v in nodes.items():
        d = int(k.split("-")[0]); by[d] += 1; pts[d] += v
    for d in sorted(by):
        print(f"depth {d}: {by[d]} nodes, {pts[d]/1e6:.1f} M points")
    print("total", len(nodes), "nodes", sum(nodes.values()) / 1e6, "M points")
    (LIDAR).mkdir(parents=True, exist_ok=True)
    (LIDAR / "nodes.json").write_text(json.dumps(nodes))
    return nodes


def main_fetch():
    nodes = json.loads((LIDAR / "nodes.json").read_text())
    # depths 0-5 have points 8 m or more apart: nothing useful in a 3 m wide footprint
    keys = sorted((k for k in nodes if int(k.split("-")[0]) >= MINDEPTH), key=lambda k: (int(k.split("-")[0]), k))
    todo = [k for k in keys if not (LAZ_DIR / f"{k}.laz").exists()]
    print(len(todo), "of", len(keys), "nodes to download", flush=True)
    done = 0
    t0 = time.time()
    nbytes = 0
    with cf.ThreadPoolExecutor(10) as ex:
        futs = {ex.submit(get, f"{BASE}/ept-data/{k}.laz", LAZ_DIR / f"{k}.laz"): k for k in todo}
        for f in cf.as_completed(futs):
            f.result()
            done += 1
            nbytes += (LAZ_DIR / f"{futs[f]}.laz").stat().st_size
            if done % 100 == 0 or done == len(todo):
                print(f"{done}/{len(todo)} {nbytes/1e6:.0f} MB {time.time()-t0:.0f}s", flush=True)


# ---------- extraction: lidar points in a footprint around every sample ----------

LAT_KEEP = 4.0   # m either side of the centreline kept in the hits file (analysis uses less)
ALONG = SPACING / 2

_S = None


def _init(sx, sy, tx, ty):
    global _S
    _S = (sx, sy, tx, ty)


def _extract_node(key):
    import laspy
    from scipy.spatial import cKDTree
    sx, sy, tx, ty = _S
    las = laspy.read(str(LAZ_DIR / f"{key}.laz"))
    px, py = Transformer.from_crs(3857, 32618, always_xy=True).transform(np.asarray(las.x), np.asarray(las.y))
    px = np.asarray(px); py = np.asarray(py)
    pz = np.asarray(las.z, dtype=np.float32)
    m = ((sx > px.min() - 16) & (sx < px.max() + 16) & (sy > py.min() - 16) & (sy < py.max() + 16))
    idx = np.nonzero(m)[0]
    if not len(idx):
        return None
    cls = np.asarray(las.classification, dtype=np.uint8)
    rn = np.asarray(las.return_number, dtype=np.uint8)
    nr = np.asarray(las.number_of_returns, dtype=np.uint8)
    tree = cKDTree(np.column_stack([px, py]))
    R = math.hypot(ALONG, LAT_KEEP) + 0.1
    lists = tree.query_ball_point(np.column_stack([sx[idx], sy[idx]]), R)
    out = []
    for si, pts in zip(idx, lists):
        if not pts:
            continue
        pts = np.asarray(pts)
        dx = px[pts] - sx[si]; dy = py[pts] - sy[si]
        al = dx * tx[si] + dy * ty[si]
        la = -dx * ty[si] + dy * tx[si]
        k = (np.abs(al) <= ALONG) & (np.abs(la) <= LAT_KEEP)
        if k.any():
            p = pts[k]
            out.append((np.full(k.sum(), si, dtype=np.int32), al[k].astype(np.float32), la[k].astype(np.float32),
                        pz[p], cls[p], rn[p], nr[p]))
    if not out:
        return None
    return tuple(np.concatenate(c) for c in zip(*out))


def main_extract(workers):
    rows = all_samples()
    sx = np.array([r[2] for r in rows]); sy = np.array([r[3] for r in rows])
    tx = np.array([r[6] for r in rows]); ty = np.array([r[7] for r in rows])
    keys = sorted(p.stem for p in LAZ_DIR.glob("*.laz"))
    print(len(rows), "samples,", len(keys), "nodes,", workers, "workers", flush=True)
    parts = []
    t0 = time.time()
    with cf.ProcessPoolExecutor(workers, initializer=_init, initargs=(sx, sy, tx, ty)) as ex:
        for i, r in enumerate(ex.map(_extract_node, keys, chunksize=8)):
            if r is not None:
                parts.append(r)
            if (i + 1) % 200 == 0:
                print(f"{i+1}/{len(keys)} {time.time()-t0:.0f}s", flush=True)
    names = ["si", "along", "lat", "z", "cls", "rn", "nr"]
    cat = {n: np.concatenate([p[j] for p in parts]) for j, n in enumerate(names)}
    np.savez(LIDAR / "hits.npz", **cat)
    print(len(cat["si"]), "hits saved", flush=True)


# ---------- measurement ----------

WIN_LAT = 1.3          # m either side of the centreline: both running rails and the third rail
LAYER_ALONG = 10.0     # m: window along the track for finding surface layers
MIN_LAYER_PTS = 25     # returns in one 0.75 m thick layer for it to count as a surface
BIN = 0.25
SKIP_CLASSES = (7, 18)  # low and high noise
# Highest dense surface over the rails is ballast / tie tops, a few tenths below the railhead
# (tie 0.18 m + rail 0.17 m, minus the ballast or sleeper crib the returns sample): rail = surface + 0.2.
SURFACE_TO_RAIL = 0.20
PLATFORM_ABOVE_RAIL = 1.1   # as build.py
# In a station zone the top flat surface is the track bed in some samples and the platform (1.1 m
# higher) in others, since OSM track positions are a metre or two off. Against 15 elevated platform
# heights from the elevator chains (evidence/station_levels.csv, platform - 1.1 m), the lidar rail
# came out 0.55 m high on median (sd 0.6 without one outlier), so that is taken off.
STATION_BIAS = -0.5
OFFSET_SD = 0.25        # m: uncertainty of that offset (open deck vs ballast vs concrete trough)
LIDAR_SD = 0.12         # m: vertical accuracy of the 2017 lidar over a hard flat surface
STATION_R = 95.0        # m around a station point: platform roofs and canopies hide the track
NEIGH = 4               # samples either side used for the continuity check
NULL_COST = 5.0         # a sample may be given no surface at this cost (in units of one sd squared)
NULL_STEP = 0.5
JUMP_SD = 0.35          # m: step between neighbouring surfaces, beyond the lidar's own grade
JUMP_CAP = 25.0


def layers_of(z, along):
    """Candidate surface layers from returns z (m) at along-track offsets: list of (z at along 0, n, resid sd)."""
    if len(z) < MIN_LAYER_PTS:
        return []
    lo = z.min()
    nb = int(math.ceil((z.max() - lo) / BIN)) + 3
    h = np.bincount(((z - lo) / BIN).astype(int), minlength=nb).astype(float)
    win = np.convolve(h, np.ones(3), mode="same")  # 0.75 m
    cands = []
    for i in range(1, nb - 1):
        if win[i] >= MIN_LAYER_PTS and win[i] >= win[i - 1] and win[i] > win[i + 1]:
            cands.append(i)
    out = []
    for i in cands:
        c = lo + (i + 0.5) * BIN
        m = np.abs(z - c) < 0.6
        a, zz = along[m], z[m]
        if len(zz) < MIN_LAYER_PTS:
            continue
        # linear fit z = z0 + g * along, two clipping rounds (grade across the 20 m window)
        keep = np.ones(len(zz), bool)
        for _ in range(3):
            A = np.column_stack([np.ones(keep.sum()), a[keep]])
            coef = np.linalg.lstsq(A, zz[keep], rcond=None)[0]
            r = zz - (coef[0] + coef[1] * a)
            s = 1.4826 * np.median(np.abs(r[keep])) + 0.03
            keep = np.abs(r) < max(2.5 * s, 0.15)
        coef[1] = float(np.clip(coef[1], -0.06, 0.06))
        out.append((float(coef[0]), int(keep.sum()), float(np.std(r[keep])), float(coef[1])))
    return out


def station_points():
    import pandas as pd
    pts = []
    s = pd.read_csv(RAW / "mta_stations.csv")
    for lo, la in zip(s["GTFS Longitude"], s["GTFS Latitude"]):
        pts.append((lo, la, STATION_R))
    j = json.loads((RAW / "osm_stations.json").read_text(encoding="utf-8"))
    for e in j["elements"]:
        t = e.get("tags", {})
        if e["type"] == "node" and t.get("public_transport") == "stop_position" and "lat" in e:
            pts.append((e["lon"], e["lat"], 70.0))
    lo = np.array([p[0] for p in pts]); la = np.array([p[1] for p in pts])
    x, y = to_utm.transform(lo, la)
    return np.asarray(x), np.asarray(y), np.array([p[2] for p in pts])


EXT_SAMPLES = 14   # samples taken from the ways joined on at each end


def extended_sequences(rows):
    """For each way, its samples in order, with the samples of the ways joined on at either end
    (the most nearly straight-on continuation at each shared end vertex), as (index, position m);
    position 0 is the start of the way itself. Ways are cut at every junction in OSM, so most of them
    are short and the neighbours that matter are in the next way."""
    ways = {w["id"]: w for w in structure_ways()}
    own = collections.defaultdict(list)
    for i, r in enumerate(rows):
        own[r[0]].append(i)
    for w in own:
        own[w].sort(key=lambda i: rows[i][1])
    ends = collections.defaultdict(list)   # vertex -> [(way, 'start'|'end')]
    info = {}
    for wid, w in ways.items():
        if wid not in own:
            continue
        g = w["geometry"]
        x, y = to_utm.transform([p["lon"] for p in g], [p["lat"] for p in g])
        x = np.asarray(x); y = np.asarray(y)
        L = float(np.hypot(np.diff(x), np.diff(y)).sum())
        k0 = (round(g[0]["lon"], 7), round(g[0]["lat"], 7)); k1 = (round(g[-1]["lon"], 7), round(g[-1]["lat"], 7))
        h0 = np.array([x[min(2, len(x) - 1)] - x[0], y[min(2, len(x) - 1)] - y[0]]); h0 /= np.hypot(*h0) or 1
        h1 = np.array([x[-1] - x[max(-3, -len(x))], y[-1] - y[max(-3, -len(x))]]); h1 /= np.hypot(*h1) or 1
        info[wid] = dict(L=L, start=k0, end=k1, hs=h0, he=h1)
        ends[k0].append((wid, "start")); ends[k1].append((wid, "end"))

    def walk(wid, side, budget, visited):
        """Samples beyond the given end of wid: list of (index, distance from that end of wid)."""
        out = []
        base = 0.0
        cur, cside = wid, side
        while len(out) < budget:
            inf = info[cur]
            key = inf["end"] if cside == "end" else inf["start"]
            head = inf["he"] if cside == "end" else -inf["hs"]
            best, bd = None, 0.8
            for ow, oside in ends[key]:
                if ow == cur or ow in visited:
                    continue
                oh = info[ow]["hs"] if oside == "start" else -info[ow]["he"]
                dot = float(head @ oh)
                if dot > bd:
                    best, bd = (ow, oside), dot
            if best is None:
                break
            ow, oside = best
            visited.add(ow)
            ids = own[ow] if oside == "start" else own[ow][::-1]
            for i in ids:
                d = rows[i][8] if oside == "start" else info[ow]["L"] - rows[i][8]
                out.append((i, base + d))
            base += info[ow]["L"]
            cur, cside = ow, ("end" if oside == "start" else "start")
        return out[:budget]

    ext = {}
    for wid in own:
        seq = [(i, rows[i][8]) for i in own[wid]]
        pre = walk(wid, "start", EXT_SAMPLES, {wid})
        post = walk(wid, "end", EXT_SAMPLES, {wid})
        L = info[wid]["L"]
        seq = [(i, -d) for i, d in pre][::-1] + seq + [(i, L + d) for i, d in post]
        ext[wid] = seq
    return ext


def measure():
    import build
    from scipy.spatial import cKDTree
    rows = all_samples()
    n = len(rows)
    d = np.load(LIDAR / "hits.npz")
    si = d["si"]; al = d["along"]; la = d["lat"]; z = d["z"]; cls = d["cls"]
    sel = (np.abs(la) <= WIN_LAT) & ~np.isin(cls, SKIP_CLASSES)
    si, al, z = si[sel], al[sel], z[sel]
    order = np.argsort(si, kind="stable")
    si, al, z = si[order], al[order], z[order].astype(np.float64)
    bounds = np.searchsorted(si, np.arange(n + 1))
    sx = np.array([r[2] for r in rows]); sy = np.array([r[3] for r in rows])
    ground = build.sample_ground(sx, sy)
    stx, sty, strad = station_points()
    tree = cKDTree(np.column_stack([stx, sty]))
    near_station = np.zeros(n, bool)
    for i, (x, y) in enumerate(zip(sx, sy)):
        for j in tree.query_ball_point([x, y], 100.0):
            if math.hypot(stx[j] - x, sty[j] - y) <= strad[j]:
                near_station[i] = True
                break
    cands = []
    for i in range(n):
        a, b = bounds[i], bounds[i + 1]
        m = np.abs(al[a:b]) <= LAYER_ALONG
        cands.append(layers_of(z[a:b][m], al[a:b][m]) if m.sum() >= MIN_LAYER_PTS else [])
    ext = extended_sequences(rows)   # {way: [(sample index, position along the chain m), ...]}
    n_top = np.array([max(c, key=lambda t: t[0])[0] if c else np.nan for c in cands])
    top = n_top.copy()
    rail = np.full(n, np.nan); sd = np.full(n, np.nan); npts = np.zeros(n, int)
    note = [[] for _ in range(n)]
    src = ["lidar"] * n
    # Each way (with the samples of the ways joined on at its ends) is solved as a path through the
    # candidate surfaces of its samples (Viterbi): a surface costs more the further it is from the one
    # before it after allowing for the grade measured in the lidar itself, the topmost surface is
    # preferred, a station zone also offers the platform surface (rail = platform - 1.1 m), and any
    # sample may be given no surface (a train, a canopy, a structure of another height over the track).
    done = {}
    for w, seq in ext.items():
        ids = [q[0] for q in seq]; pos = np.array([q[1] for q in seq])
        states = []
        for i in ids:
            st = []
            order_ = sorted(cands[i], key=lambda t: -t[0])
            for r_, t in enumerate(order_):
                st.append(dict(z=t[0] + SURFACE_TO_RAIL, g=t[3], u=1.0 * min(r_, 2) + (0.0 if r_ == 0 else 0.0),
                               kind="track", rank=r_, n=t[1], rsd=t[2]))
                if near_station[i]:
                    st.append(dict(z=t[0] - PLATFORM_ABOVE_RAIL, g=t[3], u=1.5 + 1.0 * min(r_, 2),
                                   kind="platform", rank=r_, n=t[1], rsd=t[2]))
            st.append(dict(z=np.nan, g=0.0, u=NULL_COST, kind="null", rank=-1, n=0, rsd=0.0))
            states.append(st)
        cost = [np.array([s["u"] for s in states[0]])]
        back = []
        for k in range(1, len(ids)):
            prev, cur = states[k - 1], states[k]
            dpos = pos[k] - pos[k - 1]
            C = np.zeros((len(prev), len(cur)))
            for a, sa in enumerate(prev):
                for b, sb in enumerate(cur):
                    if sa["kind"] == "null" or sb["kind"] == "null":
                        C[a, b] = NULL_STEP
                    else:
                        gbar = 0.5 * (sa["g"] + sb["g"])
                        C[a, b] = min(((sb["z"] - sa["z"] - gbar * dpos) / JUMP_SD) ** 2, JUMP_CAP)
            tot = cost[-1][:, None] + C
            bk = tot.argmin(axis=0)
            cost.append(tot[bk, np.arange(len(cur))] + np.array([s["u"] for s in cur]))
            back.append(bk)
        j = int(cost[-1].argmin())
        path = [j]
        for bk in reversed(back):
            j = int(bk[j]); path.append(j)
        path.reverse()
        for k, i in enumerate(ids):
            if rows[i][0] == w:
                done[i] = (states[k][path[k]], k)
        chain_state = [states[k][path[k]] for k in range(len(ids))]
        # a short run of surfaces between two gaps that sits far from what the samples either side
        # say is an excursion (a stopped train, a canopy) that the path took because gaps reset it
        nullst = states[0][-1]
        for _ in range(3):
            zs = np.array([s["z"] for s in chain_state])
            real = ~np.isnan(zs)
            k = 0
            changed = False
            while k < len(ids):
                if not real[k]:
                    k += 1
                    continue
                e = k
                while e + 1 < len(ids) and real[e + 1]:
                    e += 1
                lo_c = np.nonzero(real[:k])[0]; hi_c = np.nonzero(real[e + 1:])[0]
                if len(lo_c) and len(hi_c) and k > 0 and e + 1 < len(ids) and e - k < 8:
                    a_, b_ = lo_c[-1], e + 1 + hi_c[0]
                    pm = pos[k:e + 1].mean()
                    f = (pm - pos[a_]) / (pos[b_] - pos[a_])
                    exp_ = (1 - f) * zs[a_] + f * zs[b_]
                    if abs(zs[k:e + 1].mean() - exp_) > 1.0:
                        for q in range(k, e + 1):
                            chain_state[q] = nullst
                        changed = True
                k = e + 1
            if not changed:
                break
        # fill the nulls of this way's own samples from the chain
        zs = np.array([s["z"] for s in chain_state])
        for k, i in enumerate(ids):
            if rows[i][0] != w:
                continue
            s = chain_state[k]
            if s["kind"] == "null":
                good_ = np.nonzero(~np.isnan(zs))[0]
                lo_c = good_[good_ < k]; hi_c = good_[good_ > k]
                if len(lo_c) and len(hi_c):
                    a_, b_ = lo_c[-1], hi_c[0]
                    f = (pos[k] - pos[a_]) / (pos[b_] - pos[a_])
                    rail[i] = (1 - f) * zs[a_] + f * zs[b_]
                    gap = pos[b_] - pos[a_]
                    base = 0.35
                elif len(lo_c) or len(hi_c):
                    a_ = lo_c[-1] if len(lo_c) else hi_c[0]
                    rail[i] = zs[a_]
                    gap = 2 * abs(pos[k] - pos[a_])
                    base = 0.35
                else:
                    src[i] = "lidar_unclean"
                    sd[i] = 3.0
                    rail[i] = n_top[i] + SURFACE_TO_RAIL if not np.isnan(n_top[i]) else np.nan
                    note[i].append("no surface found that fits the neighbours: top surface as measured, sd widened")
                    continue
                sd[i] = math.sqrt(0.4 ** 2 + (0.005 * gap) ** 2 + base ** 2)
                src[i] = "lidar_interp"
                note[i].append("no usable surface over the track (%s): rail interpolated between samples %.0f m apart"
                               % ("station zone" if near_station[i] else "train, overpass or canopy?", gap))
            else:
                rail[i] = s["z"]
                npts[i] = s["n"]
                sd0 = math.sqrt(s["rsd"] ** 2 / max(1, min(s["n"], 30)) + LIDAR_SD ** 2 + OFFSET_SD ** 2)
                if s["kind"] == "platform":
                    sd[i] = math.sqrt(sd0 ** 2 + 0.6 ** 2)
                    src[i] = "lidar_station"
                    note[i].append("station zone: platform surface minus %.1f m" % PLATFORM_ABOVE_RAIL)
                elif near_station[i]:
                    rail[i] = s["z"] + STATION_BIAS
                    sd[i] = math.sqrt(sd0 ** 2 + 0.7 ** 2)
                    src[i] = "lidar_station"
                    note[i].append("station zone: top surface is track or platform, %.1f m allowed for it" % STATION_BIAS)
                else:
                    sd[i] = sd0
                if s["kind"] == "track" and s["rank"] > 0:
                    sd[i] = math.sqrt(sd[i] ** 2 + 0.3 ** 2)
                    note[i].append("surface %d above the top one chosen: higher structure over the track" % s["rank"])
    return rows, top, rail, sd, ground, src, note, npts, near_station


# ---------- river bridges ----------

# Tide datum offsets: mean high water above NAVD88 (m), NOAA CO-OPS datums.json, epoch 1983-2001.
MHW_NAVD88 = {"Battery": 0.597,        # station 8518750, for the East River and Harlem River bridges
              "NorthChannel": 0.72}    # station 8517201, Jamaica Bay
FT = 0.3048
FLOOR_TO_RAIL = 2.5   # m from the underside of a truss bridge's lowest deck (the published clearance) to the rail:
                      # floor beam 1.2-1.5 m, stringers 0.8-0.9 m, tie and rail 0.35 m

# The upper roadways of the Manhattan Bridge hide most of its subway tracks from above (the DSM there
# is the roadway, 7 m higher), and on the Williamsburg Bridge one OSM track runs under a higher level.
# One OSM track of each bridge is visible from above for the whole crossing; the other tracks of the
# same structure are at the same height, so wherever their own lidar rail disagrees by more than
# TEMPLATE_TOL the template track's rail at the same place is used.
TEMPLATES = [
    dict(name="Manhattan Bridge", template=817756457,
         targets=[823986481, 46340237, 46346751, 616217282, 823986482]),
    dict(name="Williamsburg Bridge", template=206583332, targets=[180330673]),
]
TEMPLATE_TOL = 1.0
TEMPLATE_MAX_DIST = 40.0   # m in plan from the template track

# Published figures, converted to a rail height: (name, lon, lat, template/nearby way, rail NAVD88, sd, source)
def published_rows():
    out = []
    bat = MHW_NAVD88["Battery"]; nc = MHW_NAVD88["NorthChannel"]
    out.append(dict(
        name="Manhattan Bridge mid-span", lon=-73.99065, lat=40.70725, way=817756457,
        rail=135 * FT + bat + FLOOR_TO_RAIL, sd=1.2,
        note="clearance 135 ft above mean high water at mid-span (en.wikipedia.org/wiki/Manhattan_Bridge, 134 ft in the text) "
             "= underside of the lower deck, which carries the subway tracks under the upper roadways; +%.1f m MHW "
             "(Battery) + %.1f m underside to rail" % (bat, FLOOR_TO_RAIL)))
    out.append(dict(
        name="Williamsburg Bridge mid-span", lon=-73.97175, lat=40.71350, way=206583332,
        rail=135 * FT + bat + FLOOR_TO_RAIL, sd=1.2,
        note="clearance 135 ft above mean high water at mid-river (en.wikipedia.org/wiki/Williamsburg_Bridge); +%.1f m MHW "
             "(Battery) + %.1f m underside to rail" % (bat, FLOOR_TO_RAIL)))
    out.append(dict(
        name="Beach Channel railroad bridge (Jamaica Bay)", lon=-73.8104, lat=40.5958, way=291273055,
        rail=26 * FT + nc + 1.3, sd=1.0,
        note="NYCT swing bridge, vertical clearance closed 26 ft at mean high water (Coast Guard drawbridge notice, "
             "Federal Register 2005-11-02, 05-21855; waterwayguide.com); +%.2f m MHW (North Channel station 8517201) "
             "+ 1.3 m girder and deck to rail" % nc))
    return out


UPPER_DECK_STEP = 3.5   # m: a rail step this big between neighbouring clean samples is a change of deck
UPPER_DECK_SD = 0.8     # m: sd of a rail taken as the upper deck minus the measured deck gap


def fix_template_upper_deck(t, rows, rail, sd, src, note, top, ti):
    """The template track is visible on the main span, but on the Manhattan Bridge's Manhattan
    approach the upper roadway covers every track, the template's too: its lidar 'rail' steps up
    4.5 m within 60 m past the tower (no rail can) and then follows the roadway down at 3%, 6 m
    above the rail, into the portal. Detected as an upward step of more than UPPER_DECK_STEP
    between neighbouring clean samples (and back down by as much, if it comes back); those runs
    get rail = surface - the deck gap, the gap measured where both decks are seen on this bridge
    (samples whose chosen surface lies under a higher one), and source 'lidar_deck'. That is an
    upper bound, not a reading: the tracks are under the roadway, but on the approach the roadway
    ramps down to the Canal St plaza while the tracks dive for their portal, so they may be well
    below it (build.py uses these rows as caps). Interpolated samples between are re-interpolated
    and become bounds too. Found 2026-11-02: the approach read 6 m high (and held as a reading)
    pulled the tunnel at Canal St and Grand St 10-15 m above the published platform depths."""
    gaps = [top[i] + SURFACE_TO_RAIL - rail[i] for i in ti
            if any("above the top one chosen" in s for s in note[i]) and np.isfinite(top[i])]
    gaps = [g for g in gaps if 3.0 < g < 10.0]
    if len(gaps) < 3:
        return 0
    gap = float(np.median(gaps))
    clean = [i for i in ti if src[i] == "lidar" and np.isfinite(rail[i])]
    deck = set()
    on = False
    cz = {}   # corrected rail of the clean samples walked so far
    for k, b in enumerate(clean):
        if k < 2:
            cz[b] = rail[b]
            continue
        a, a2 = clean[k - 1], clean[k - 2]
        # the step is measured against the rail's own trend over the two samples before it, so a
        # long gap (a station zone between) is not read as a step
        da = rows[a][8] - rows[a2][8] or 1.0
        slope = float(np.clip((cz[a] - cz[a2]) / da, -0.05, 0.05))
        resid = (rail[b] - (gap if on else 0.0)) - (cz[a] + slope * (rows[b][8] - rows[a][8]))
        if not on and resid > UPPER_DECK_STEP:
            on = True
        elif on and resid < -UPPER_DECK_STEP:
            on = False
        cz[b] = rail[b] - (gap if on else 0.0)
        if on:
            deck.add(b)
    for i in deck:
        note[i].append("%s: upper deck over the template track itself (a %.1f m step up from the visible rail): "
                       "rail = deck - %.1f m, the gap between the decks measured on this bridge" % (t["name"], UPPER_DECK_STEP, gap))
        rail[i] -= gap
        sd[i] = UPPER_DECK_SD
        src[i] = "lidar_deck"
    # interpolated samples lying between corrected ones follow them
    good = [i for i in ti if src[i] in ("lidar", "lidar_deck", "lidar_station")]
    pos = {i: k for k, i in enumerate(ti)}
    for k, i in enumerate(ti):
        if src[i] != "lidar_interp":
            continue
        lo = [j for j in good if pos[j] < k]; hi = [j for j in good if pos[j] > k]
        if lo and hi and (lo[-1] in deck or hi[0] in deck):
            a, b = lo[-1], hi[0]
            al = [rows[j][8] for j in (a, i, b)]
            f = (al[1] - al[0]) / (al[2] - al[0])
            rail[i] = (1 - f) * rail[a] + f * rail[b]
            src[i] = "lidar_deck"
            note[i].append("re-interpolated after the upper-deck correction")
    print(t["name"], "template samples read as the upper deck:", len(deck), "(gap %.2f m from %d samples)" % (gap, len(gaps)))
    return len(deck)


def way_continuations(hops=2):
    """{way id: ids of structure ways that continue it end to end with the same track_ref, up to
    `hops` ways on}."""
    ways = structure_ways()
    ends = collections.defaultdict(list)
    info = {}
    for w in ways:
        g = w["geometry"]
        k0 = (round(g[0]["lon"], 7), round(g[0]["lat"], 7)); k1 = (round(g[-1]["lon"], 7), round(g[-1]["lat"], 7))
        ref = w["tags"].get("railway:track_ref")
        info[w["id"]] = (k0, k1, ref)
        ends[k0].append(w["id"]); ends[k1].append(w["id"])
    out = {}
    for wid_, (k0, k1, ref) in info.items():
        if ref is None:
            continue
        seen, front = set(), {wid_}
        for _ in range(hops):
            nxt = set()
            for w in front:
                for k in info[w][:2]:
                    for o in ends[k]:
                        if o != wid_ and o not in seen and info[o][2] == ref:
                            nxt.add(o)
            seen |= nxt
            front = nxt
        out[wid_] = seen
    return out


def bridge_adjust(rows, rail, sd, src, note, ground, names, top=None):
    """Replace rails of the hidden tracks, add the published rows. Returns extra rows."""
    sx = np.array([r[2] for r in rows]); sy = np.array([r[3] for r in rows])
    wid = np.array([r[0] for r in rows]); alongs = np.array([r[8] for r in rows])
    cont = way_continuations()
    for t in TEMPLATES:
        ti = np.nonzero(wid == t["template"])[0]
        ti = ti[np.argsort(alongs[ti])]
        if top is not None:
            fix_template_upper_deck(t, rows, rail, sd, src, note, top, ti)
        # The hidden track stays hidden past the end of its OSM way: OSM cuts the bridge into ways
        # at the anchorages, and the Williamsburg's track 2 runs on under the higher level down the
        # Manhattan approach (way 1547317494), where its own lidar read 5 m above track 1 for 300 m
        # and the model drew a bump on one J track (found 2026-11-02). Template and targets both
        # extend through end-to-end continuations of the same track (same track_ref).
        tws = {t["template"]} | cont.get(t["template"], set())
        tgs = set()
        for w in t["targets"]:
            tgs |= {w} | cont.get(w, set())
        tgs -= tws
        ti = np.nonzero(np.isin(wid, list(tws)))[0]
        ti = ti[np.lexsort((alongs[ti], wid[ti]))]
        tx, ty, tz = sx[ti], sy[ti], rail[ti]
        n_fix = 0
        for i in np.nonzero(np.isin(wid, list(tgs)))[0]:
            d = np.hypot(tx - sx[i], ty - sy[i])
            j = np.argsort(d)[:2]
            if d[j[0]] > TEMPLATE_MAX_DIST:
                continue
            w1 = 1 / max(d[j[0]], 1.0); w2 = 1 / max(d[j[1]], 1.0) if abs(j[0] - j[1]) == 1 else 0.0
            z = (w1 * tz[j[0]] + w2 * tz[j[1]]) / (w1 + w2)
            if src[ti[j[0]]] == "lidar_deck":
                # under the same roadway as the template's upper-deck stretch: a bound, as there
                note[i].append("%s: under the upper roadway where the template track is hidden too; rail at most the "
                               "template's bound" % t["name"])
                rail[i] = min(rail[i], z)
                sd[i] = UPPER_DECK_SD
                src[i] = "lidar_deck"
                n_fix += 1
            elif abs(rail[i] - z) > TEMPLATE_TOL:
                note[i].append("%s: this track's own lidar rail was %.1f m %s (an upper level hides or replaces the track); "
                               "rail taken from the track that is visible" % (t["name"], abs(rail[i] - z), "above" if rail[i] > z else "below"))
                rail[i] = z
                sd[i] = math.sqrt(0.5 ** 2 + 0.35 ** 2)
                src[i] = "lidar_template"
                n_fix += 1
        print(t["name"], "samples taken from the template track:", n_fix)
    extra = []
    for p in published_rows():
        extra.append(p)
    return extra


def write_csv(rows, top, rail, sd, ground, src, note, npts, near_station, names, extra=()):
    import csv
    EVID.mkdir(exist_ok=True)
    p = EVID / "structure_heights.csv"
    fmt = lambda v, p=2: "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{p}f}"
    with open(p, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["way_id", "lon", "lat", "top_navd88_m", "ground_navd88_m", "rail_navd88_m", "sd_m",
                     "source", "notes", "line", "rail_above_ground_m", "along_m", "n_points"])
        for i, r in enumerate(rows):
            ag = rail[i] - ground[i]
            wr.writerow([r[0], f"{r[4]:.6f}", f"{r[5]:.6f}", fmt(top[i]), fmt(ground[i]), fmt(rail[i]), fmt(sd[i]),
                         src[i], "; ".join(note[i]), names.get(r[0], ""), fmt(ag), f"{r[8]:.0f}", npts[i]])
        for e in extra:
            wr.writerow([e["way"], f"{e['lon']:.6f}", f"{e['lat']:.6f}", "", fmt(e.get("ground")), fmt(e["rail"]), fmt(e["sd"]),
                         "published", e["name"] + ": " + e["note"], names.get(e["way"], ""),
                         fmt(e["rail"] - e["ground"]) if e.get("ground") is not None else "", "", 0])
    return p


def main_measure():
    import build
    rows, top, rail, sd, ground, src, note, npts, ns = measure()
    names = {w["id"]: w["tags"].get("name", "") for w in structure_ways()}
    extra = bridge_adjust(rows, rail, sd, src, note, ground, names, top)
    # Smith-9 Sts: platform 87.5 ft above ground (en.wikipedia.org/wiki/Smith-Ninth_Streets_station)
    gx, gy = to_utm.transform(-73.9960, 40.6736)
    g0 = float(build.sample_ground(np.array([gx]), np.array([gy]))[0])
    extra.append(dict(name="Smith-9 Sts platform", lon=-73.9960, lat=40.6736, way=46341911, ground=g0,
                      rail=g0 + 87.5 * FT - PLATFORM_ABOVE_RAIL, sd=1.0,
                      note="station platform 87.5 ft above ground level (Wikipedia, Smith-Ninth Streets station) = %.1f m over the "
                           "3 m DEM ground %.2f m, minus %.1f m platform above rail" % (87.5 * FT, g0, PLATFORM_ABOVE_RAIL)))
    for e in extra:
        if "ground" not in e:
            gx, gy = to_utm.transform(e["lon"], e["lat"])
            e["ground"] = float(build.sample_ground(np.array([gx]), np.array([gy]))[0])
    p = write_csv(rows, top, rail, sd, ground, src, note, npts, ns, names, extra)
    print("wrote", p, len(rows), "lidar rows +", len(extra), "published rows")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if stage == "plan":
        main_plan()
    elif stage == "fetch":
        main_fetch()
    elif stage == "measure":
        main_measure()
    elif stage == "extract":
        main_extract(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    elif stage == "all":
        # extract uses several processes: claim cores first (python cores.py claim 4 trackdepth-els ...)
        main_plan(); main_fetch(); main_extract(int(sys.argv[2]) if len(sys.argv) > 2 else 2); main_measure()
