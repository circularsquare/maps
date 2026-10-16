"""T-498: real rail minutes door to door between a pack's zones, from OSM track and stations only
(no timetable): ride at a commercial speed by track class, 5 min wait, 2 min each way between a
station and each line within 300 m (so transfers cost 4 min), access by walk or feeder.
    python rail.py <overpass.json> <car.npz> <out.npz>
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json, sys, time, re
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

src, carf, out = sys.argv[1:4]
WAIT, PLAT, SNAP_M, ACC_M = 5.0, 2.0, 300.0, 3000.0
C = np.load(carf)
lon0, lat0 = float(C["lon0"]), float(C["lat0"])
R = 6371008.8
kx = R * np.cos(np.radians(lat0)) * np.pi / 180; ky = R * np.pi / 180

def xy(lon, lat):
    return (np.asarray(lon) - lon0) * kx, (np.asarray(lat) - lat0) * ky

def num(s):
    m = re.match(r"\s*(\d+(\.\d+)?)\s*(mph)?", s or "")
    return float(m.group(1)) * (1.609 if m.group(3) else 1.0) if m else None

def speed(t):
    rw = t.get("railway")
    ms = num(t.get("maxspeed"))
    if t.get("highspeed") == "yes" or (ms and ms >= 200):
        return 0.55 * (ms or 260)
    if rw == "subway":
        return 33.0
    if rw == "monorail":
        return 30.0
    if rw == "light_rail":
        return 25.0
    if rw == "tram":
        return 15.0
    if ms:
        return float(np.clip(0.6 * ms, 30, 110))
    return {"main": 50.0, "branch": 35.0}.get(t.get("usage"), 45.0)

def tags(other):
    return dict(re.findall(r'"([^"]+)"=>"([^"]*)"', other or ""))

t0 = time.time()
RAILS = ("rail", "subway", "light_rail", "monorail", "narrow_gauge", "tram")
ways, stations = [], []
if src.endswith(".pbf"):
  for src in src.split(","):
          import pyogrio, shapely
          cache = src + ".rail.npz"
          if not os.path.exists(cache):
              L = pyogrio.read_dataframe(src, layer="lines", columns=["railway", "other_tags"], where="railway IN ('rail','subway','light_rail','monorail','narrow_gauge','tram')")
              L["other_tags"] = '"railway"=>"' + L["railway"] + '",' + L["other_tags"].fillna("")
              Pn = pyogrio.read_dataframe(src, layer="points", columns=["other_tags"], where="other_tags LIKE '%\"railway\"=>\"station\"%' OR other_tags LIKE '%\"railway\"=>\"halt\"%'")
              co, ix = shapely.get_coordinates(L.geometry.to_numpy(), return_index=True)
              pc = shapely.get_coordinates(Pn.geometry.to_numpy())
              np.savez(cache, co=co, ix=ix, ltags=np.array(L["other_tags"].fillna(""), dtype=object), pc=pc,
                       ptags=np.array(Pn["other_tags"].fillna(""), dtype=object))
              print(f"read pbf {time.time() - t0:.0f} s", flush=True)
          Z_ = np.load(cache, allow_pickle=True)
          co, ix = Z_["co"], Z_["ix"]
          off = np.searchsorted(ix, np.arange(len(Z_["ltags"]) + 1))
          for w, o in enumerate(Z_["ltags"]):
              t = tags(o)
              if t.get("railway") not in RAILS:
                  continue
              if t.get("service") in ("yard", "siding", "spur", "crossover") or t.get("usage") in ("freight", "industrial", "military", "tourism", "test"):
                  continue
              if t.get("railway") == "tram" and t.get("service"):
                  continue
              c = co[off[w]:off[w + 1]]
              keys = [(round(x * 1e7), round(y * 1e7)) for x, y in c]
              ways.append((keys, [{"lon": x, "lat": y} for x, y in c], speed(t)))
          for (x, y), o in zip(Z_["pc"], Z_["ptags"]):
              t = tags(o)
              if any(k.startswith(("disused", "abandoned", "construction", "proposed")) for k in t) or t.get("usage") == "tourism":
                  continue
              stations.append((x, y))
else:
    E = json.load(open(src, encoding="utf-8"))["elements"]
    for e in E:
        t = e.get("tags", {})
        if e["type"] == "node":
            if any(k.startswith(("disused", "abandoned", "construction", "proposed")) for k in t) or t.get("usage") == "tourism":
                continue
            stations.append((e["lon"], e["lat"]))
        elif e["type"] == "way":
            if t.get("service") in ("yard", "siding", "spur", "crossover") or t.get("usage") in ("freight", "industrial", "military", "tourism", "test"):
                continue
            if t.get("railway") == "tram" and t.get("service"):
                continue
            ways.append((e["nodes"], e["geometry"], speed(t)))
print(f"{len(ways):,} ways, {len(stations):,} stations, read {time.time() - t0:.0f} s", flush=True)

# vertices by OSM node id
vid = {}
pts = []
for nodes, geom, _ in ways:
    for n, g in zip(nodes, geom):
        if n not in vid:
            vid[n] = len(pts); pts.append((g["lon"], g["lat"]))
pts = np.array(pts)
vx, vy = xy(pts[:, 0], pts[:, 1])
use = np.zeros(len(pts), np.int32)
wayof = []
for w, (nodes, _, _) in enumerate(ways):
    ids = np.array([vid[n] for n in nodes])
    np.add.at(use, ids, 1)
    use[ids[0]] += 1; use[ids[-1]] += 1
    wayof.append(ids)
keep = use >= 2  # junctions and way ends
# stations: nearest vertex of each way within SNAP_M
st = np.array(stations); sx, sy = xy(st[:, 0], st[:, 1])
vt = cKDTree(np.c_[vx, vy])
vway = np.full(len(pts), -1, np.int64)
for w, ids in enumerate(wayof):
    vway[ids] = w  # one way per vertex is enough to tell lines apart near a station
links = []
for s, cand in enumerate(vt.query_ball_point(np.c_[sx, sy], SNAP_M)):
    best = {}
    for v in cand:
        d = np.hypot(vx[v] - sx[s], vy[v] - sy[s])
        w = vway[v]
        if w not in best or d < best[w][1]:
            best[w] = (v, d)
    for v, d in best.values():
        links.append((s, v)); keep[v] = True
print(f"{len(pts):,} vertices, {int(keep.sum()):,} kept, {len(links):,} station links", flush=True)
kid = -np.ones(len(pts), np.int64); kid[keep] = np.arange(int(keep.sum()))
NK = int(keep.sum())
a_, b_, c_ = [], [], []
for w, ids in enumerate(wayof):
    sp = ways[w][2] / 60.0 * 1000.0  # m per minute
    seg = np.hypot(np.diff(vx[ids]), np.diff(vy[ids])) / sp
    cum = np.concatenate([[0], np.cumsum(seg)])
    k = np.nonzero(keep[ids])[0]
    for p, q in zip(k[:-1], k[1:]):
        a_.append(kid[ids[p]]); b_.append(kid[ids[q]]); c_.append(max(cum[q] - cum[p], 1e-3))
S = len(stations)
for s, v in links:
    a_.append(NK + s); b_.append(kid[v]); c_.append(PLAT)
n = NK + S
G = coo_matrix((c_, (a_, b_)), shape=(n, n)).tocsr()
G = G.minimum(G.T) if False else G
t0 = time.time()
# only stations that link to a line
live = np.unique([s for s, _ in links])
SS = dijkstra(G, directed=False, indices=NK + live)[:, NK + live].astype(np.float32)
print(f"{len(live):,} stations, station to station in {time.time() - t0:.0f} s", flush=True)
lsx, lsy = sx[live], sy[live]

# zones: access from the commuter-weighted centroid
zx, zy = C["mx"], C["my"]
Z = len(zx)
def acc_min(d):
    w = d * 1.3 / 100.0  # 6 km/h = 100 m a minute
    f = 5.0 + d * 1.3 / 250.0  # feeder (bus or bicycle) at 15 km/h after 5 min
    return np.minimum(w, f)
st_t = cKDTree(np.c_[lsx, lsy])
acc = st_t.query_ball_point(np.c_[zx, zy], ACC_M)
INF = np.float32(1e4)
ZS = np.full((Z, len(live)), INF, np.float32)
for i in range(Z):
    if acc[i]:
        ks = np.array(acc[i]); ti = acc_min(np.hypot(lsx[ks] - zx[i], lsy[ks] - zy[i])).astype(np.float32)
        ZS[i] = np.min(ti[:, None] + SS[ks], axis=0)
TR = np.full((Z, Z), INF, np.float32)
for j in range(Z):
    if acc[j]:
        ks = np.array(acc[j]); tj = acc_min(np.hypot(lsx[ks] - zx[j], lsy[ks] - zy[j])).astype(np.float32)
        TR[:, j] = np.min(ZS[:, ks] + tj[None, :], axis=1)
TR += WAIT
print(f"zones with a station within {ACC_M:.0f} m: {sum(1 for x in acc if x)} of {Z}; rail matrix {time.time() - t0:.0f} s", flush=True)
np.savez(out, rail=TR, nstations=len(live))
