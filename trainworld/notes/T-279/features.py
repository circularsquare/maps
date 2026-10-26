"""T-279 research: per pair features along the straight line and least-cost paths on a coarse
raster (1/60 degree), for every OSRM-routed pair in osrm/tables.json and T-279's 50 pairs.
python -I features.py <scratch dir>
Writes <scratch>/pairs.json (one record per unordered pair)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, math, sys, time
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

TW = r"C:\Users\anita\projects\maps\trainworld"
sys.path.insert(0, os.path.join(TW, "pipeline"))
from zone_terrain import Dem, M_PER_DEG_LAT, M_PER_DEG_LON  # noqa: E402

S = sys.argv[1]
PER = 60
R = 6371.0088
rows = json.load(open(os.path.join(TW, "app", "src", "game", "market-rows.json"), encoding="utf-8"))
tables = json.load(open(os.path.join(S, "osrm", "tables.json")))


def gc(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[1], a[0], b[1], b[0]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * math.asin(min(1, math.sqrt(h)))


# ---------------------------------------------------------------- global layers
def world_water():
    b = open(os.path.join(r"C:\Users\anita\projects\maps\artifacts\trainworld\t1285", "world.water"), "rb").read()
    assert b[:4] == b"TWW1"
    W, H, per = (int.from_bytes(b[4 + 4 * i:8 + 4 * i], "little") for i in range(3))
    out = np.zeros((H, W), bool)
    p = 16

    def leb():
        nonlocal p
        n = s = 0
        while True:
            x = b[p]; p += 1
            n |= (x & 0x7F) << s; s += 7
            if not x & 0x80:
                return n
    for r in range(H):
        k = leb(); last = 0; e = []
        for _ in range(k):
            last += leb(); e.append(last)
        for i in range(0, len(e), 2):
            out[r, e[i]:e[i + 1]] = True
    assert per == PER
    return out


def burn_links(water):
    j = json.load(open(os.path.join(TW, "data", "raw", "world_field", "wikidata_links.json"), encoding="utf-8"))["results"]["bindings"]
    n = 0
    for bnd in j:
        c = bnd["coord"]["value"]
        if not c.startswith("POINT("):
            continue
        x, y = map(float, c[6:-1].split())
        L = float(bnd["m"]["value"]) / 1000 if "m" in bnd else 0.0
        if not (0.3 <= L <= 80) or abs(y) > 85:
            continue
        rad = L / 2 + 1.0
        ry = int(math.ceil(rad / 110.57 * PER)); rx = int(math.ceil(rad / (111.32 * max(math.cos(math.radians(y)), 0.05)) * PER))
        r0 = int((y + 90) * PER); c0 = int((x + 180) * PER)
        ra0, ra1 = max(r0 - ry, 0), min(r0 + ry + 1, water.shape[0])
        ca0, ca1 = max(c0 - rx, 0), min(c0 + rx + 1, water.shape[1])
        sub = water[ra0:ra1, ca0:ca1]
        if sub.any():
            yy = (np.arange(ra0, ra1) - r0)[:, None]; xx = (np.arange(ca0, ca1) - c0)[None, :]
            d = np.hypot(yy / PER * 110.57, xx / PER * 111.32 * math.cos(math.radians(y)))
            sub[d <= rad] = False
            n += 1
    print("links burnt", n, flush=True)


def grip(name):
    cache = os.path.join(S, f"grip_{name}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    sub = {"tp1": "tp1/grip4_tp1_dens_m_km2.asc", "tp2": "tp2/grip4_tp2_dens_m_km2.asc", "total": "total/grip4_total_dens_m_km2.asc"}[name]
    a = pd.read_csv(os.path.join(S, "dl", "grip", sub), skiprows=6, header=None, sep=r"\s+", dtype=np.float32).to_numpy()
    a = np.where(a < 0, 0, a)[::-1].copy()  # rows from the south
    np.save(cache, a)
    return a


t0 = time.time()
WATER = world_water()
burn_links(WATER)
HW, PRI, TOT = grip("tp1"), grip("tp2"), grip("total")
from pathlib import Path  # noqa: E402
DEM = Dem(Path(TW, "data", "raw", "gedtm30", "gedtm_rf_m_240m_v1.2.tif"))
print("layers", round(time.time() - t0), "s", flush=True)


def slope_grid(c0, r0, cw, ch):
    """Mean slope (degrees) per 1/60 degree cell, rows from the south, NaN at sea."""
    w = c0 / PER - 180; s = r0 / PER - 90; e = (c0 + cw) / PER - 180; n = (r0 + ch) / PER - 90
    px = DEM.px
    pc0 = math.floor((w - DEM.lon0) / px) - 1; pc1 = math.ceil((e - DEM.lon0) / px) + 1
    pr0 = math.floor((DEM.lat0 - n) / px) - 1; pr1 = math.ceil((DEM.lat0 - s) / px) + 1
    z = DEM.window(pr0, pr1, pc0, pc1)
    lats = DEM.lat0 - (np.arange(pr0, pr1) + 0.5) * px
    lons = DEM.lon0 + (np.arange(pc0, pc1) + 0.5) * px
    dy = px * M_PER_DEG_LAT
    dx = px * M_PER_DEG_LON * np.cos(np.radians(lats))[:, None]
    gy, gx = np.gradient(z)
    sl = np.degrees(np.arctan(np.hypot(gx / dx, gy / dy)))
    rr = np.floor((lats + 90) * PER).astype(np.int64) - r0
    cc = np.floor((lons + 180) * PER).astype(np.int64) - c0
    ok = np.isfinite(sl) & ((rr >= 0) & (rr < ch))[:, None] & ((cc >= 0) & (cc < cw))[None, :]
    idx = (rr[:, None] * cw + cc[None, :])[ok]
    tot = np.bincount(idx, weights=sl[ok], minlength=cw * ch)
    cnt = np.bincount(idx, minlength=cw * ch)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (tot / cnt).reshape(ch, cw).astype(np.float32)


def unit(p):
    lo, la = math.radians(p[0]), math.radians(p[1])
    return np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])


def line_pts(a, b, step_km=1.0):
    d = gc(a, b)
    n = max(2, int(d / step_km) + 1)
    ua, ub = unit(a), unit(b)
    om = math.acos(max(-1, min(1, float(ua @ ub))))
    t = np.linspace(0, 1, n)
    if om < 1e-9:
        v = np.repeat(ua[None], n, 0)
    else:
        v = (np.sin((1 - t) * om)[:, None] * ua + np.sin(t * om)[:, None] * ub) / math.sin(om)
    lon = np.degrees(np.arctan2(v[:, 1], v[:, 0])); lat = np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))
    return lon, lat


# speed variants for the raster router (km/h): (motorway cell, primary cell, other road, no road)
VARIANTS = {
    "v1": dict(hw=105, pri=75, road=50, none=20, slope=True),
    "v2": dict(hw=105, pri=75, road=50, none=20, slope=False),
    "v3": dict(hw=100, pri=60, road=60, none=60, slope=False),
    "v5": dict(hw=105, pri=75, road=50, none=20, slope=True, water=15),
    "v6": dict(hw=105, pri=75, road=50, none=20, slope=True, water=6),
    "v7": dict(hw=105, pri=75, road=50, none=20, slope=True, slope_k=3, water=6),
    "v8": dict(hw=110, pri=85, road=55, none=20, slope=True, water=6),
}


def speeds(v, hw, pri, tot, sl):
    s = np.full(hw.shape, v["none"], np.float32)
    s = np.where(tot >= 100, v["road"], s)
    s = np.where(hw + pri >= 25, v["pri"], s)
    f = 1 / (1 + np.maximum(np.nan_to_num(sl) - 3, 0) / v.get("slope_k", 6)) if v["slope"] else 1.0
    s = s * f
    s = np.where(hw >= 25, v["hw"] * (f if v.get("slope_all") else 1), s)
    return s


def window_graph(c0, r0, cw, ch, cost):
    """8-neighbour graph over the window; edge weight = km x mean cost a km of its two cells
    (cost inf: no edge)."""
    lat = (r0 + np.arange(ch) + 0.5) / PER - 90
    kx = 111.32 * np.cos(np.radians(lat)) / PER; ky = 110.57 / PER
    idx = np.arange(cw * ch).reshape(ch, cw)
    I, J, Wt = [], [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = idx[0:ch - dr, max(0, -dc):cw - max(0, dc)]
        b = idx[dr:ch, max(0, dc):cw - max(0, -dc)]
        ra = np.arange(ch - dr)[:, None]
        km = np.hypot(dc * (kx[ra] + kx[ra + dr]) / 2, dr * ky) * np.ones_like(a, dtype=np.float64)
        ca, cb = cost.ravel()[a], cost.ravel()[b]
        w = km * (ca + cb) / 2
        ok = np.isfinite(w)
        I.append(a[ok]); J.append(b[ok]); Wt.append(w[ok])
    I, J, Wt = np.concatenate(I), np.concatenate(J), np.concatenate(Wt)
    g = coo_matrix((np.concatenate([Wt, Wt]), (np.concatenate([I, J]), np.concatenate([J, I]))), shape=(cw * ch, cw * ch)).tocsr()
    return g


def path_len(pred, src, dst, cw, r0):
    """Geometric km along the predecessor chain from dst back to src."""
    km = 0.0
    j = dst
    while j != src and j >= 0:
        i = pred[j]
        if i < 0:
            return float("nan")
        ri, ci = divmod(i, cw); rj, cj = divmod(j, cw)
        lat = (r0 + (ri + rj) / 2 + 0.5) / PER - 90
        km += math.hypot((cj - ci) * 111.32 * math.cos(math.radians(lat)) / PER, (rj - ri) * 110.57 / PER)
        j = i
    return km


def group(pairs, places):
    """Features for pairs [(ka, kb, rec)] among the places {k: (lon, lat)} in one window."""
    lons = [p[0] for p in places.values()]; lats = [p[1] for p in places.values()]
    w, e = min(lons) - 1.5, max(lons) + 1.5; s, n = min(lats) - 1.2, max(lats) + 1.2
    c0 = int(math.floor((w + 180) * PER)); c1 = int(math.ceil((e + 180) * PER))
    r0 = int(math.floor((s + 90) * PER)); r1 = int(math.ceil((n + 90) * PER))
    cw, ch = c1 - c0, r1 - r0
    water = WATER[r0:r1, c0:c1]
    rr = (np.arange(r0, r1) // 5)[:, None]; cc = (np.arange(c0, c1) // 5 % 4320)[None, :]
    hw, pri, tot = HW[rr, cc], PRI[rr, cc], TOT[rr, cc]
    sl = slope_grid(c0, r0, cw, ch)
    keys = list(places)
    cell = {k: (int((places[k][1] + 90) * PER) - r0) * cw + int((places[k][0] + 180) * PER) - c0 for k in keys}
    src = np.array([cell[k] for k in keys])
    # endpoints are on land by definition: clear water at the sources
    wflat = water.ravel().copy(); wflat[src] = False
    out = {}
    land_cost = np.where(wflat.reshape(ch, cw), np.inf, 1.0)
    g = window_graph(c0, r0, cw, ch, land_cost)
    dl = dijkstra(g, indices=src)
    res = {"land": dl}
    g = window_graph(c0, r0, cw, ch, np.where(wflat.reshape(ch, cw), 4.0, 1.0))
    res["land4"] = dijkstra(g, indices=src)
    for name, v in VARIANTS.items():
        sp = speeds(v, hw, pri, tot, sl)
        cost = np.where(wflat.reshape(ch, cw), 60.0 / v["water"] if "water" in v else np.inf, 60.0 / sp)
        g = window_graph(c0, r0, cw, ch, cost)
        dm, pred = dijkstra(g, indices=src, return_predecessors=True)
        res[name] = (dm, pred)
    for ka, kb, rec in pairs:
        ia, ib = keys.index(ka), keys.index(kb)
        a, b = places[ka], places[kb]
        lon, lat = line_pts(a, b)
        r = np.clip(np.floor((lat + 90) * PER).astype(int) - r0, 0, ch - 1)
        c = np.clip(np.floor((lon + 180) * PER).astype(int) - c0, 0, cw - 1)
        wl = water[r, c]
        runs = int(np.sum(np.diff(wl.astype(np.int8)) == 1))
        longest = 0; cur = 0
        for x in wl:
            cur = cur + 1 if x else 0; longest = max(longest, cur)
        sll = np.nan_to_num(sl[r, c])
        f = dict(rec)
        d = gc(a, b)
        f.update(gc_km=d, water_frac=float(wl.mean()), water_runs=runs, water_longest_km=longest * d / max(len(wl) - 1, 1),
                 slope_mean=float(sll.mean()), slope_p75=float(np.percentile(sll, 75)), frac_hilly=float((sll >= 5).mean()), frac_mtn=float((sll >= 10).mean()),
                 hw_mean=float(hw[r, c].mean()), hw_frac=float((hw[r, c] >= 25).mean()), pri_mean=float(pri[r, c].mean()), tot_mean=float(tot[r, c].mean()),
                 land_km=float(res["land"][ia, cell[kb]]), land4_km=float(res["land4"][ia, cell[kb]]))
        for name in VARIANTS:
            dm, pred = res[name]
            f[name + "_min"] = float(dm[ia, cell[kb]])
            f[name + "_km"] = path_len(pred[ia], cell[ka], cell[kb], cw, r0)
        out[(ka, kb)] = f
    return out, (cw, ch)


recs = []
for anc, t in tables.items():
    ids = t.get("ids") or []
    if len(ids) < 2:
        continue
    places = {k: (rows[k]["lon"], rows[k]["lat"]) for k in ids}
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = t["dur"][i][j], t["dur"][j][i]
            da, db = t["dist"][i][j], t["dist"][j][i]
            if None in (a, b, da, db):
                continue
            d = gc(places[ids[i]], places[ids[j]])
            if not 20 <= d <= 700:
                continue
            pairs.append((ids[i], ids[j], {"cluster": anc, "a": ids[i], "b": ids[j], "ca": rows[ids[i]]["country"], "cb": rows[ids[j]]["country"],
                                            "osrm_min": (a + b) / 120, "osrm_km": (da + db) / 2000,
                                            "snap_m": max(t["snap"][i] or 0, t["snap"][j] or 0)}))
    t1 = time.time()
    f, shape = group(pairs, places)
    recs += list(f.values())
    print(anc, len(pairs), shape, round(time.time() - t1, 1), "s", flush=True)

# T-279's 50 pairs, each its own window
old = json.load(open(os.path.join(r"C:\Users\anita\projects\maps\artifacts\trainworld\t1147", "osrm_pairs.json")))
for o in old:
    if o["ida"] not in rows or o["idb"] not in rows:
        continue
    places = {o["ida"]: (rows[o["ida"]]["lon"], rows[o["ida"]]["lat"]), o["idb"]: (rows[o["idb"]]["lon"], rows[o["idb"]]["lat"])}
    if rows[o["ida"]]["road"] != rows[o["idb"]]["road"]:
        continue
    f, shape = group([(o["ida"], o["idb"], {"cluster": "t279", "kind": o["kind"], "name": f"{o['a']} - {o['b']}", "a": o["ida"], "b": o["idb"],
                                             "ca": rows[o["ida"]]["country"], "cb": rows[o["idb"]]["country"], "osrm_min": o["osrm_min"], "osrm_km": o["road_km"], "snap_m": 0})], places)
    recs += list(f.values())
    print("t279", o["a"], o["b"], flush=True)
json.dump(recs, open(os.path.join(S, "pairs.json"), "w"), indent=0)
print("pairs", len(recs), round(time.time() - t0), "s", flush=True)
