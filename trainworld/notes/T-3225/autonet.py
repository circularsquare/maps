"""T-3225 experiment: a city's rail network built with no hand work from noritetsu's local OSM extract
(route relations, railway ways with tunnel/bridge/layer, stop nodes), compared with the game's
hand-made network JSON for the same city. Read only on everything; prints a report.

    python autonet.py <cc> <city>      e.g. ca tor, jp sdj, us nyc
"""
import json
import math
import os
import pickle
import re
import sys
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from shapely.geometry import LineString, shape
from shapely.strtree import STRtree

cc, city = sys.argv[1], sys.argv[2]
MAPS = r"C:\Users\anita\projects\maps"
PROC = os.path.join(MAPS, "noritetsu", "data", "proc", cc)
TW = os.path.join(MAPS, "trainworld")
out = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"report_{city}.txt"), "w", encoding="utf-8")


def p(*a):
    print(*a, file=out)


hand = json.load(open(os.path.join(TW, "data", "work", "realnet", f"{city}_real.json"), encoding="utf-8"))
pack = hand["pack"]
bnd = shape(json.load(open(os.path.join(TW, "data", "packs", pack + ".boundary.geojson")))["features"][0]["geometry"]) \
    if "features" in json.load(open(os.path.join(TW, "data", "packs", pack + ".boundary.geojson"))) else \
    shape(json.load(open(os.path.join(TW, "data", "packs", pack + ".boundary.geojson"))))
bnd = bnd.buffer(0)
W, S, E, N = bnd.bounds
lat0 = (S + N) / 2
KX = 111320 * math.cos(math.radians(lat0))
KY = 110540


def xy(lon, lat):
    return lon * KX, lat * KY


z = np.load(os.path.join(PROC, "coords.npz"))
ids, cx, cy = z["id"], z["x"], z["y"]
order = np.argsort(ids)
ids, cx, cy = ids[order], cx[order], cy[order]
# scale: find it from a stop's known lon
stops = pickle.load(open(os.path.join(PROC, "stops.pkl"), "rb"))
rels = pickle.load(open(os.path.join(PROC, "rels.pkl"), "rb"))
ways = pickle.load(open(os.path.join(PROC, "ways.pkl"), "rb"))
SCALE = 1e7
probe = next(i for i in stops if np.searchsorted(ids, i) < len(ids) and ids[np.searchsorted(ids, i)] == i)
j = np.searchsorted(ids, probe)
for sc in (1e7, 1e6, 1e5):
    if abs(cx[j] / sc - stops[probe][1]) < 1e-3:
        SCALE = sc
p(f"coords scale {SCALE}")


def node_ll(nid):
    if nid in stops:
        return stops[nid][1], stops[nid][2]
    j = np.searchsorted(ids, nid)
    if j < len(ids) and ids[j] == nid:
        return cx[j] / SCALE, cy[j] / SCALE
    return None


KIND = {"subway": "metro", "train": "commuter", "light_rail": "light_rail", "tram": "tram", "monorail": "monorail"}
SKIP_SERVICE = {"high_speed", "long_distance", "night", "tourism"}
masters = {}
for rid, (t, ms) in rels.items():
    if t.get("type") == "route_master":
        for ty, r, _ in ms:
            if ty == "r":
                masters[r] = rid

routes = []  # (rid, master, tags, kind, stops [(name, lon, lat)], way ids)
dropped = Counter()
for rid, (t, ms) in rels.items():
    if t.get("type") != "route":
        continue
    kind = KIND.get(t.get("route"))
    if not kind:
        continue
    if t.get("service") in SKIP_SERVICE:
        dropped["service tag " + t["service"]] += 1
        continue
    st = []
    for ty, r, role in ms:
        if ty == "n" and role.startswith("stop"):
            ll = node_ll(r)
            if ll and bnd.contains(__import__("shapely.geometry", fromlist=["Point"]).Point(ll)):
                nm = stops.get(r, ({}, 0, 0))[0].get("name", "")
                st.append((nm, ll[0], ll[1], r))
    wids = [r for ty, r, role in ms if ty == "w" and role in ("", "forward", "backward") and r in ways]
    if len(st) >= 2:
        routes.append((rid, masters.get(rid, rid), t, kind, st, wids))

p(f"route relations kept: {len(routes)}; dropped {dict(dropped)}")
by_master = defaultdict(list)
for r in routes:
    by_master[r[1]].append(r)
p(f"lines (route masters or lone relations): {len(by_master)}")

# stations: one per name within 400 m, else per node
NAME_M = 400
stn = []  # [name, x, y, n]
stn_of = {}
for r in routes:
    for nm, lon, lat, nid in r[4]:
        if nid in stn_of:
            continue
        x, y = xy(lon, lat)
        k = None
        if nm:
            for i, s in enumerate(stn):
                if s[0] == nm and (s[1] - x) ** 2 + (s[2] - y) ** 2 < NAME_M ** 2:
                    k = i
                    break
        if k is None:
            stn.append([nm, x, y, 1])
            k = len(stn) - 1
        stn_of[nid] = k
p(f"stations (name within {NAME_M} m): {len(stn)}; stop nodes with no name: {sum(1 for r in routes for s in r[4] if not s[0])}")

# the hand network's stations and lines
hs = hand["stations"]
def hxy(s):
    if isinstance(s, dict):
        lon, lat = s.get("lon"), s.get("lat")
    else:
        lon, lat = s[2], s[3]
    return xy(lon, lat)
p(f"hand network: {len(hs)} stations, {len(hand['lines'])} lines; sample station {json.dumps(hs[0], ensure_ascii=False)[:160]}")
H = np.array([hxy(s) for s in hs])
A = np.array([(s[1], s[2]) for s in stn])


def nearest(P, Q):
    d = np.sqrt(((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1))
    return d.min(1)


dh = nearest(H, A)
da = nearest(A, H)
p(f"hand stations with an auto station within 200 m: {np.mean(dh < 200):.0%} ({int((dh < 200).sum())} of {len(H)})")
p(f"auto stations with a hand station within 200 m: {np.mean(da < 200):.0%} ({int((da < 200).sum())} of {len(A)})")

# line match: each hand line against each auto line, share of its stops within 200 m of the auto line's stops
auto_line_pts = {m: np.array([xy(lon, lat) for r in rs for _, lon, lat, _ in r[4]]) for m, rs in by_master.items()}
matched = 0
kinds = Counter()
miss = []
tph_pairs = []
for l in hand["lines"]:
    P = H[l["stops"]]
    best, bm = 0, None
    for m, Q in auto_line_pts.items():
        f = float(np.mean(nearest(P, Q) < 200))
        if f > best:
            best, bm = f, m
    if best >= 0.8:
        matched += 1
        t = rels[by_master[bm][0][0]][0]
        iv = t.get("interval") or rels.get(bm, ({}, []))[0].get("interval")
        tph_pairs.append((l.get("name"), l["tph"][0], iv))
    else:
        miss.append((l.get("name"), l.get("kind"), round(best, 2)))
    kinds[l.get("kind")] += 1
p(f"hand lines whose stops are 80%+ on one auto line: {matched} of {len(hand['lines'])}")
p("  missed (name, kind, best share): " + "; ".join(f"{a} {b} {c}" for a, b, c in miss[:40]))
with_iv = [x for x in tph_pairs if x[2]]
p(f"matched lines whose OSM relation has an interval tag: {len(with_iv)} of {len(tph_pairs)}: " +
  "; ".join(f"{a}: hand peak {b:.1f}/h, OSM interval {c}" for a, b, c in with_iv[:15]))
auto_names = sorted({(rels.get(m, ({}, []))[0].get("ref") or rels.get(m, ({}, []))[0].get("name") or by_master[m][0][2].get("ref") or by_master[m][0][2].get("name", "?")) for m in by_master})
p("auto lines: " + "; ".join(auto_names[:80]))
p(f"auto relations with colour: {sum(1 for r in routes if r[2].get('colour') or rels.get(r[1], ({}, []))[0].get('colour'))} of {len(routes)}; "
  f"with interval: {sum(1 for r in routes if r[2].get('interval'))}; with wikidata: {sum(1 for r in routes if r[2].get('wikidata'))}")

# levels: the ways the kept relations run over
used = {}
for r in routes:
    for w in r[5]:
        used.setdefault(w, set()).add(r[1])
tagc = Counter()
km = Counter()
geoms, glev, gown, gnodes = [], [], [], []
for w, owners in used.items():
    t, nodes = ways[w]
    pts = [node_ll(n) for n in nodes]
    pts = [q for q in pts if q]
    if len(pts) < 2:
        continue
    g = LineString([xy(*q) for q in pts])
    L = g.length / 1000
    tun = t.get("tunnel") in ("yes", "building_passage", "culvert") or t.get("location") == "underground"
    bri = t.get("bridge") not in (None, "no")
    lay = t.get("layer")
    km["all"] += L
    if tun:
        km["tunnel"] += L
    if bri:
        km["bridge"] += L
    if lay not in (None, "0"):
        km["layer set"] += L
    try:
        lv = int(float(lay)) if lay else (-1 if tun else (1 if bri else 0))
    except ValueError:
        lv = -1 if tun else (1 if bri else 0)
    if tun and lv >= 0:
        lv = -1
    if bri and lv <= 0:
        lv = 1
    tagc[lv] += 1
    geoms.append(g)
    glev.append(lv)
    gown.append(owners)
    gnodes.append(set(int(n) for n in nodes))
p(f"track km under the kept lines: {km['all']:.0f}; tunnel {km['tunnel']:.0f}, bridge {km['bridge']:.0f}, layer set {km['layer set']:.0f}")
p(f"ways by level from tags: {dict(sorted(tagc.items()))}")

# crossings between ways of different lines that do not share a node (a real crossing, not a junction)
tree = STRtree(geoms)
cross_same_tags, cross_naive, total = 0, 0, 0
for i, g in enumerate(geoms):
    for j in tree.query(g):
        j = int(j)
        if j <= i or (gown[i] & gown[j]) or (gnodes[i] & gnodes[j]):
            continue
        x = g.intersection(geoms[j])
        if x.is_empty or x.geom_type not in ("Point", "MultiPoint"):
            continue
        n = len(getattr(x, "geoms", [x]))
        total += n
        cross_naive += n
        if glev[i] == glev[j]:
            cross_same_tags += n
p(f"crossings of different lines' ways (no shared node): {total}; flat if all at one level: {cross_naive}; "
  f"flat with levels from layer/tunnel/bridge: {cross_same_tags}")
out.close()
print(open(out.name, encoding="utf-8").read())
