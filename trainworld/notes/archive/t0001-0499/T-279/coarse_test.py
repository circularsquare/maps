"""T-279 research: full against coarse OSM road graphs on one cached extract, both against OSRM.
python -I coarse_test.py <scratch dir> <extract name> <country> <w> <s> <e> <n>
Speeds by class (OSRM car profile's defaults, rounded); no oneways, no turn costs.
Full = every drive class; coarse = motorway, trunk, primary and their links; coarse2 adds secondary."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, math, sys, time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.spatial import cKDTree

S, EXT, CTY = sys.argv[1], sys.argv[2], sys.argv[3]
BOX = tuple(map(float, sys.argv[4:8]))
TW = r"C:\Users\anita\projects\maps\trainworld"
DRIVE_CLASSES = ("motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
                 "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
                 "residential", "living_street", "road")
KMH = dict(motorway=105, motorway_link=60, trunk=85, trunk_link=50, primary=65, primary_link=45, secondary=55,
           secondary_link=40, tertiary=45, tertiary_link=35, unclassified=30, residential=25, living_street=10, road=25)
SETS = {"full": set(DRIVE_CLASSES), "coarse": {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"},
        "coarse2": {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link"}}
R = 6371.0088

rows = json.load(open(os.path.join(TW, "app", "src", "game", "market-rows.json"), encoding="utf-8"))
tables = json.load(open(os.path.join(S, "osrm", "tables.json")))
t0 = time.time()
z = np.load(os.path.join(TW, "data", "raw", "osm", EXT + ".ways2.npz"))
cls, flags, off, lon, lat = z["cls"], z["flags"], z["off"], z["lon"], z["lat"]
print("ways", len(cls), "points", len(lon), round(time.time() - t0), "s", flush=True)

# truth pairs: both ends in the box and the country
truth = {}
for anc, t in tables.items():
    ids = t.get("ids") or []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            ra, rb = rows[a], rows[b]
            if ra["country"] != CTY or rb["country"] != CTY:
                continue
            if not all(BOX[0] <= r["lon"] <= BOX[2] and BOX[1] <= r["lat"] <= BOX[3] for r in (ra, rb)):
                continue
            if None in (t["dur"][i][j], t["dur"][j][i]):
                continue
            truth[tuple(sorted((a, b)))] = ((t["dur"][i][j] + t["dur"][j][i]) / 120, (t["dist"][i][j] + t["dist"][j][i]) / 2000)
places = sorted({k for p in truth for k in p})
print("truth pairs", len(truth), "places", len(places), flush=True)


def km_of(lo1, la1, lo2, la2):
    x = np.radians(lo2 - lo1) * np.cos(np.radians((la1 + la2) / 2)); y = np.radians(la2 - la1)
    return R * np.hypot(x, y)


out = {}
for name, keep_cls in SETS.items():
    t1 = time.time()
    kc = np.array([i for i, c in enumerate(DRIVE_CLASSES) if c in keep_cls])
    wk = np.nonzero(np.isin(cls, kc) & (flags & 1 > 0))[0]
    # consecutive point pairs of kept ways
    way_of = np.repeat(np.arange(len(cls)), np.diff(off))
    kept = np.zeros(len(cls), bool); kept[wk] = True
    a_idx = np.nonzero((way_of[:-1] == way_of[1:]) & kept[way_of[:-1]])[0]
    seg_cls = cls[way_of[a_idx]]
    b_idx = a_idx + 1
    # node ids by coordinate
    key = np.round(lon * 1e6).astype(np.int64) * 1_000_000_000 + np.round(lat * 1e6).astype(np.int64)
    used = np.unique(np.concatenate([a_idx, b_idx]))
    uk, inv = np.unique(key[used], return_inverse=True)
    node_of = np.full(len(lon), -1, np.int64); node_of[used] = inv
    na, nb = node_of[a_idx], node_of[b_idx]
    km = km_of(lon[a_idx], lat[a_idx], lon[b_idx], lat[b_idx])
    sp = np.array([KMH[DRIVE_CLASSES[c]] for c in range(len(DRIVE_CLASSES))])[seg_cls]
    w = km / sp * 60
    N = len(uk)
    g = coo_matrix((np.concatenate([w, w]), (np.concatenate([na, nb]), np.concatenate([nb, na]))), shape=(N, N)).tocsr()
    gk = coo_matrix((np.concatenate([km, km]), (np.concatenate([na, nb]), np.concatenate([nb, na]))), shape=(N, N)).tocsr()
    ncomp, lab = connected_components(g, directed=False)
    big = np.bincount(lab).argmax()
    nl = np.zeros(N); nlat = np.zeros(N)
    nl[node_of[used]] = lon[used]; nlat[node_of[used]] = lat[used]
    ok = np.nonzero(lab == big)[0]
    tree = cKDTree(np.column_stack([nl[ok] * np.cos(np.radians(50)), nlat[ok]]))
    src, acc_km = [], []
    for k in places:
        d, i = tree.query([rows[k]["lon"] * math.cos(math.radians(50)), rows[k]["lat"]])
        j = ok[i]
        src.append(j); acc_km.append(float(km_of(rows[k]["lon"], rows[k]["lat"], nl[j], nlat[j])))
    dm, pred = dijkstra(g, indices=np.array(src), return_predecessors=True)
    res = {}
    for (a, b), (tm, tk) in truth.items():
        ia, ib = places.index(a), places.index(b)
        # road km along the time-shortest path
        j = src[ib]; kmsum = 0.0
        while j != src[ia] and j >= 0:
            i = pred[ia, j]
            if i < 0:
                kmsum = float("nan"); break
            kmsum += gk[i, j]; j = i
        acc = (acc_km[ia] + acc_km[ib]) * 1.3
        res[f"{a}|{b}"] = {"min": float(dm[ia, src[ib]]) + acc / 30 * 60, "km": kmsum + acc, "osrm_min": tm, "osrm_km": tk,
                           "acc_km": acc_km[ia] + acc_km[ib]}
    out[name] = res
    print(name, "nodes", N, "edges", len(w), "components", ncomp, "biggest", len(ok), round(time.time() - t1), "s", flush=True)
json.dump(out, open(os.path.join(S, f"coarse_{CTY}.json"), "w"), indent=0)
for name, res in out.items():
    r = np.array([v["min"] / v["osrm_min"] for v in res.values()])
    e = np.array([abs(v["min"] - v["osrm_min"]) for v in res.values()])
    print(name, "time ratio median", round(float(np.median(r)), 3), "p10-p90", np.round(np.percentile(r, [10, 90]), 3), "abs err median", round(float(np.median(e)), 1), "p90", round(float(np.percentile(e, 90)), 1))
