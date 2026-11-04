"""Scan the built model for things that can't be right. Run after build.py and make_platforms.py:

    python checks/scan.py            # counts and the worst cases per check
    python checks/scan.py --all      # every case
    python checks/scan.py --data <dir>   # another data folder (a saved tracks.json / platforms.json)

Reads data/tracks.json, data/platforms.json, data/anchors.csv, raw/osm_tracks.json, raw/osm_stations.json,
raw/osm_platforms.json, raw/mta_stations.csv, evidence/structure_heights.csv. Writes nothing. One check
per class of mistake Anita found on the map (2026-11-03); each prints one count line and the worst cases.

 1 anchors        the model away from an anchor it was given (elevator chain, document, placed row)
 2 cover          underground stations and tunnel track too close under the street for the structure
 3 docs           (in 1: documented depths are anchors)
 4 above-ground   grade-separated crossings of open track (els, bridges, embankments) without room, and
                  bridge-tagged track (not over water) drawn at the ground
 5 open-deep      open cut, embankment and at-grade structure (OSM or the MTA's Structure column) drawn
                  as deep as a tunnel, and MTA Structure against the station dot's height
 6 platform-tie   tracks on the two sides of one OSM platform at different heights
 7 portals        tunnel to open track along a line with no portal
 8 structure      a stretch where the OSM structure tags, the lidar and the model's rail vs ground disagree
 9 dots           OSM stations on the map's tracks with no station dot
10 curvature      vertical curves sharper than a train can run (radius under 600 m)
11 boxes          station boxes on a level with no platform
12 labels         station dots split by level where only one level has a platform
"""
import collections
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = pathlib.Path(sys.argv[sys.argv.index("--data") + 1]) if "--data" in sys.argv else ROOT / "data"
ALL = "--all" in sys.argv
SHOW = 10**9 if ALL else 12
to_utm = Transformer.from_crs(4326, 32618, always_xy=True)
to_ll = Transformer.from_crs(32618, 4326, always_xy=True)
RAILROAD = ("Long Island Rail Road", "Metro-North", "Amtrak", "New Jersey Transit", "NJ Transit")

tj = json.loads((DATA / "tracks.json").read_text(encoding="utf-8"))
pj = json.loads((DATA / "platforms.json").read_text(encoding="utf-8")) if (DATA / "platforms.json").exists() else {"boxes": [], "platforms": []}
raw = {e["id"]: e["tags"] for e in json.loads((ROOT / "raw" / "osm_tracks.json").read_text(encoding="utf-8"))["elements"]}
T = tj["tracks"]
results = {}


def out(name, n, rows, what):
    results[name] = n
    print(f"\n[{name}] {n} {what}")
    for r in rows[:SHOW]:
        print("   ", r)
    if len(rows) > SHOW:
        print(f"    ... {len(rows) - SHOW} more (--all)")


def ll(x, y):
    lo, la = to_ll.transform(x, y)
    return f"{la:.5f},{lo:.5f}"


# ---------- points ----------
P = []   # (way index, k)
X, Y = [], []
for wi, t in enumerate(T):
    x, y = to_utm.transform(np.array(t["lon"]), np.array(t["lat"]))
    for k in range(len(x)):
        P.append((wi, k)); X.append(x[k]); Y.append(y[k])
X = np.array(X); Y = np.array(Y)
Z = np.array([T[w]["z"][k] for w, k in P]); G = np.array([np.nan if T[w]["ground"][k] is None else T[w]["ground"][k] for w, k in P])
# tracks.json `ground` is the surface a person stands on (the street over a tunnel, the surface beside open
# track); open track's own bed is the bare-earth DEM at the track, read here (build.sample_ground)
sys.path.insert(0, str(ROOT))
import build  # noqa: E402
DEM = build.sample_ground(X, Y)
CLS = np.array([T[w]["cls"] for w, k in P]); WAY = np.array([w for w, k in P])
SVC = np.array([bool(T[w]["service"]) for w, k in P])
LVL = [T[w]["level"] for w, k in P]
ptree = cKDTree(np.c_[X, Y])
wet = np.array([("bed" in T[w] and T[w]["bed"][k] is not None) for w, k in P]) | (G < 1.0)
# graph: shared coordinates are joined
node = {}
for i, (w, k) in enumerate(P):
    node.setdefault((T[w]["lon"][k], T[w]["lat"][k]), []).append(i)
open_ = ~np.isin(CLS, ["tunnel"])
# distance in plan to the nearest open (non-tunnel) point: portals and their approaches
otree = cKDTree(np.c_[X[open_], Y[open_]])
d_open, _ = otree.query(np.c_[X, Y])


def stretches(mask, minpts=2):
    """Runs of consecutive flagged points along a way: (way, k0, k1)."""
    runs = []
    start = 0
    for wi, t in enumerate(T):
        n = len(t["z"])
        m = mask[start:start + n]
        k = 0
        while k < n:
            if m[k]:
                j = k
                while j + 1 < n and m[j + 1]:
                    j += 1
                if j - k + 1 >= minpts:
                    runs.append((wi, k, j, start))
                k = j + 1
            else:
                k += 1
        start += n
    return runs


# ---------- 1 anchors ----------
an = pd.read_csv(DATA / "anchors.csv")
rows = []
for _, a in an.iterrows():
    if not np.isfinite(a.get("lon", np.nan)):
        continue
    ax, ay = to_utm.transform(a["lon"], a["lat"])
    idx = ptree.query_ball_point([ax, ay], 100)
    if not idx:
        continue
    if a["source"] == "document":
        tgt = None   # depth below street; compare depth
        zz = np.array([G[i] - Z[i] for i in idx])
        target = a["depth"]
        model = float(np.median(zz))
        # the row's own level is not stored: take the nearest model depth to it
        model = float(zz[np.argmin(np.abs(zz - target))])
    else:
        target = a["rail"]
        zz = Z[idx]
        model = float(zz[np.argmin(np.abs(zz - target))])
    sdu = float(a["sd_used"])
    off = model - target
    if abs(off) > max(2.0, 2 * sdu):
        rows.append((abs(off), f"{a['source']:9} {str(a['station'])[:36]:36} {str(a['line'])[:20]:20} anchor {target:6.1f} model {model:6.1f} (off {off:+.1f}, anchor sd {sdu:.1f})"))
rows.sort(reverse=True)
out("1 anchors", len(rows), [r for _, r in rows], "anchors the model misses by more than max(2 m, 2 sd)")

# ---------- 2 cover ----------
mta = pd.read_csv(ROOT / "raw" / "mta_stations.csv", dtype=str)
sl = pd.read_csv(ROOT / "evidence" / "station_levels.csv", dtype={"station_mrn": str})
with_mezz = set(sl.loc[sl["level_name"].str.contains("mezzanine", na=False), "station"].astype(str))
rows = []
mta_struct0 = {(r["Stop Name"], r["Line"]): r["Structure"] for _, r in mta.iterrows()}
for s in tj["stations"]:
    if s["ground"] is None or mta_struct0.get((s["name"], s["line"])) != "Subway":
        continue
    dep = s["ground"] - s["z"]
    need = 7.0 if s["name"] in with_mezz else 4.5
    if dep < need:
        rows.append((need - dep, f"{s['name'][:30]:30} {s['line'][:22]:22} lvl {s.get('level')} rail {dep:4.1f} m under the street (needs {need:.1f}: {'mezzanine' if need > 5 else 'fare control at the street'})"))
rows.sort(reverse=True)
n_st = len(rows)
tun_shallow = (CLS == "tunnel") & ~SVC & np.isfinite(G) & ~wet & (G - Z < 3.5) & (d_open > 150)
runs = stretches(tun_shallow, 2)
rows2 = sorted(((G[s + k0:s + k1 + 1] - Z[s + k0:s + k1 + 1]).min(), f"tunnel {T[w]['name'][:30]:30} way {T[w]['id']} {ll(X[s + k0], Y[s + k0])} "
                f"{(k1 - k0 + 1) * 25} m, rail {(G[s + k0:s + k1 + 1] - Z[s + k0:s + k1 + 1]).min():.1f} m under the street") for w, k0, k1, s in runs)
out("2 cover", n_st + len(rows2), [r for _, r in rows] + [r for _, r in rows2],
    f"too shallow: {n_st} underground station dots under their structural minimum, {len(rows2)} tunnel stretches (over 150 m from open track) under 3.5 m")

# ---------- 4 above-ground crossings, bridges at the ground ----------
segs = []
for wi, t in enumerate(T):
    if t["service"] or t["cls"] == "tunnel":
        continue
    x, y = to_utm.transform(np.array(t["lon"]), np.array(t["lat"]))
    for k in range(len(x) - 1):
        segs.append((wi, k, x[k], y[k], x[k + 1], y[k + 1]))
allsegs = []
for wi, t in enumerate(T):
    if t["service"]:
        continue
    x, y = to_utm.transform(np.array(t["lon"]), np.array(t["lat"]))
    for k in range(len(x) - 1):
        allsegs.append((wi, k, x[k], y[k], x[k + 1], y[k + 1]))
mid = np.array([((s[2] + s[4]) / 2, (s[3] + s[5]) / 2) for s in allsegs])
half = max(math.hypot(s[4] - s[2], s[5] - s[3]) for s in allsegs) / 2
stree = cKDTree(mid)
open_idx = [i for i, s in enumerate(allsegs) if T[s[0]]["cls"] != "tunnel"]
rows = []
seen = set()


def coords_of(wi):
    return {(a, b) for a, b in zip(T[wi]["lon"], T[wi]["lat"])}


join_cache = {}


def joined(w1, w2):
    """The two ways share a vertex, or each touches a way that touches the other (one switch apart)."""
    k = (min(w1, w2), max(w1, w2))
    if k not in join_cache:
        c1, c2 = coords_of(w1), coords_of(w2)
        join_cache[k] = bool(c1 & c2) or T[w1]["name"] == T[w2]["name"] and T[w1]["level"] == T[w2]["level"]
    return join_cache[k]


for i in open_idx:
    s1 = allsegs[i]
    for j in stree.query_ball_point(mid[i], 2 * half):
        if j == i:
            continue
        s2 = allsegs[j]
        if s1[0] == s2[0]:
            continue
        r = (s1[4] - s1[2], s1[5] - s1[3]); q = (s2[4] - s2[2], s2[5] - s2[3])
        den = r[0] * q[1] - r[1] * q[0]
        if abs(den) < 1e-9 or abs(den) / (math.hypot(*r) * math.hypot(*q) or 1) < 0.42:
            continue
        qp = (s2[2] - s1[2], s2[3] - s1[3])
        tt = (qp[0] * q[1] - qp[1] * q[0]) / den; uu = (qp[0] * r[1] - qp[1] * r[0]) / den
        if not (0 <= tt <= 1 and 0 <= uu <= 1) or joined(s1[0], s2[0]):
            continue
        t1, t2 = T[s1[0]], T[s2[0]]
        z1 = (1 - tt) * t1["z"][s1[1]] + tt * t1["z"][s1[1] + 1]
        z2 = (1 - uu) * t2["z"][s2[1]] + uu * t2["z"][s2[1] + 1]
        key = (min(s1[0], s2[0]), max(s1[0], s2[0]))
        if key in seen:
            continue
        seen.add(key)
        hi, lo = (t1, t2) if z1 > z2 else (t2, t1)
        if lo["cls"] == "tunnel":
            continue   # tunnels under open track: the cover cap's business
        # open track over open track: the lower one's clearance plus the upper deck (build.py's CROSS_CLEAR)
        lo_t = raw.get(lo["id"], {})
        lo_rr = any(o in lo_t.get("operator", "") for o in RAILROAD)
        el = lo_t.get("electrified", "")
        clear = 6.7 if "contact_line" in el else (4.9 if lo_rr else 4.4) if el else 6.1
        need = clear + (1.5 if hi["cls"] != "tunnel" else 1.1)
        gap = abs(z1 - z2)
        x = s1[2] + tt * r[0]; y = s1[3] + tt * r[1]
        if gap < need - 0.5:
            rows.append((need - gap, f"{ll(x, y)} gap {gap:4.1f} need {need:.1f}: {hi['name'][:26]!r} {hi['cls']} z {max(z1, z2):.1f} over "
                                     f"{lo['name'][:26]!r} {lo['cls']} z {min(z1, z2):.1f}"))
rows.sort(reverse=True)
ncross = len(rows)
bridge_low = (CLS == "elevated") & ~wet & np.isfinite(G) & (Z - G < 2.0) & ~SVC
runs = stretches(bridge_low, 2)
rows2 = sorted(((-(Z[s + k0:s + k1 + 1] - G[s + k0:s + k1 + 1]).min()), f"bridge-tagged {T[w]['name'][:30]:30} way {T[w]['id']} {ll(X[s + k0], Y[s + k0])} "
                f"{(k1 - k0 + 1) * 25} m at {(Z[s + k0:s + k1 + 1] - G[s + k0:s + k1 + 1]).min():+.1f} m from the ground") for w, k0, k1, s in runs)
rows2.sort(key=lambda r: r[0])
out("4 above-ground", ncross + len(rows2), [r for _, r in rows] + [r for _, r in rows2],
    f"open-track problems: {ncross} crossings over open track without room, {len(rows2)} bridge-tagged stretches (not over water) within 2 m of the ground")

# ---------- 5 open structure drawn deep; MTA Structure vs the dot ----------
# open track's rail more than 3 m under the bare earth at the track itself (its own bed)
deep_open = np.isin(CLS, ["grade", "embankment", "cutting"]) & np.isfinite(DEM) & ~wet & (DEM - Z > 3.0) & ~SVC
runs = stretches(deep_open, 2)
rows = sorted(((DEM[s + k0:s + k1 + 1] - Z[s + k0:s + k1 + 1]).max(), f"{T[w]['cls']:10} {T[w]['name'][:30]:30} way {T[w]['id']} {ll(X[s + k0], Y[s + k0])} "
               f"{(k1 - k0 + 1) * 25} m, rail down to {(DEM[s + k0:s + k1 + 1] - Z[s + k0:s + k1 + 1]).max():.1f} m under the ground at the track") for w, k0, k1, s in runs)
rows.sort(reverse=True)
STRUCT = {"Open Cut": (1.5, 11.0), "At Grade": (-2.5, 2.5), "Embankment": (-9.0, 1.0), "Elevated": (-40.0, -3.0),
          "Viaduct": (-60.0, -3.0), "Subway": (3.0, 80.0)}   # allowed (ground - rail) range
mta_struct = {(r["Stop Name"], r["Line"]): r["Structure"] for _, r in mta.iterrows()}
rows2 = []
for s in tj["stations"]:
    st_ = mta_struct.get((s["name"], s["line"]))
    if st_ not in STRUCT or s["ground"] is None:
        continue
    dep = s["ground"] - s["z"]
    lo, hi = STRUCT[st_]
    if not (lo <= dep <= hi):
        rows2.append((min(abs(dep - lo), abs(dep - hi)), f"{s['name'][:30]:30} {s['line'][:20]:20} MTA {st_:10} but rail {dep:+.1f} m under the street (allowed {lo:+.0f} to {hi:+.0f})"))
rows2.sort(reverse=True)
out("5 open-deep", len(rows) + len(rows2), [r for _, r in rows] + [r for _, r in rows2],
    f"open structure drawn deep: {len(rows)} OSM open stretches (grade, embankment, cutting) over 3 m under the ground at the track, "
    f"{len(rows2)} station dots against the MTA's Structure column")

# ---------- 6 tracks on two sides of one platform ----------
from shapely.geometry import LineString, Point, Polygon
plats = json.loads((ROOT / "raw" / "osm_platforms.json").read_text(encoding="utf-8"))["elements"]
rows = []
ntie = 0
for e in plats:
    if e["type"] != "way" or len(e.get("geometry", [])) < 2:
        continue
    tg = e.get("tags", {})
    if any(k.startswith(("disused", "abandoned")) for k in tg):
        continue
    xy = [to_utm.transform(p["lon"], p["lat"]) for p in e["geometry"]]
    geom = Polygon(xy).buffer(0) if len(xy) >= 4 and xy[0] == xy[-1] else LineString(xy)
    c = geom.centroid
    near = ptree.query_ball_point([c.x, c.y], max(60.0, geom.length / 2))
    byw = collections.defaultdict(list)
    for i in near:
        if SVC[i]:
            continue
        p_ = geom.distance(Point(X[i], Y[i]))
        if p_ < 6.5:
            byw[WAY[i]].append(i)
    if len(byw) < 2:
        continue
    zs = {w: float(np.median(Z[v])) for w, v in byw.items() if len(v) >= 2}
    if len(zs) < 2:
        continue
    names = {T[w]["name"] for w in zs}
    lvls = {T[w]["level"] for w in zs}
    if len(lvls) > 1:
        continue   # two levels drawn side by side (a stacked station): not one platform's two sides
    ntie += 1
    spread = max(zs.values()) - min(zs.values())
    if spread > 1.0:
        rows.append((spread, f"{tg.get('name', tg.get('ref', ''))[:30]:30} {ll(c.x, c.y)} tracks {sorted(names)[:3]} heights {sorted(round(v, 1) for v in zs.values())}"))
rows.sort(reverse=True)
out("6 platform-tie", len(rows), [r for _, r in rows], f"of {ntie} OSM platforms with track on both sides have them more than 1 m apart")

# ---------- 7 portals ----------
por = tj.get("portals", [])
px, py = to_utm.transform(np.array([p["lon"] for p in por]), np.array([p["lat"] for p in por]))
prt = cKDTree(np.c_[px, py]) if len(por) else None
rows = []
seen = set()
# tunnel runs as build.find_portals sees them: tunnel ways joined at shared vertices, with open runs
# shorter than 50 m touching tunnel at both ends (a gap in the tags) counted as tunnel; runs under 60 m
# are a street over an open cut and need no portal
def wlen(w):
    x, y = to_utm.transform(np.array(T[w]["lon"]), np.array(T[w]["lat"]))
    return float(np.hypot(np.diff(x), np.diff(y)).sum())
parent = list(range(len(T)))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
ends = collections.defaultdict(set)
for w, t in enumerate(T):
    if t["service"]:
        continue
    for k in (0, -1):
        ends[(t["lon"][k], t["lat"][k])].add(w)
for kk, idxs in node.items():
    ws = sorted({WAY[i] for i in idxs if not T[WAY[i]]["service"]})
    tw = [w for w in ws if T[w]["cls"] == "tunnel"]
    for a in tw[1:]:
        parent[find(a)] = find(tw[0])
gap_ways = set()
for w, t in enumerate(T):   # short open gaps between tunnels
    if t["service"] or t["cls"] == "tunnel" or wlen(w) >= 50:
        continue
    tun_ends = [v for k in (0, -1) for v in ends[(t["lon"][k], t["lat"][k])] if T[v]["cls"] == "tunnel"]
    a_end = [v for v in ends[(t["lon"][0], t["lat"][0])] if T[v]["cls"] == "tunnel"]
    b_end = [v for v in ends[(t["lon"][-1], t["lat"][-1])] if T[v]["cls"] == "tunnel"]
    if a_end and b_end:
        parent[find(a_end[0])] = find(b_end[0])
        gap_ways.add(w)
run_len = collections.defaultdict(float)
for w, t in enumerate(T):
    if t["cls"] == "tunnel" and not t["service"]:
        run_len[find(w)] += wlen(w)
for kk, idxs in node.items():
    ws = {WAY[i] for i in idxs}
    tun = [w for w in ws if T[w]["cls"] == "tunnel" and not T[w]["service"] and run_len[find(w)] >= 60]
    opn = [w for w in ws if T[w]["cls"] != "tunnel" and not T[w]["service"] and w not in gap_ways]
    if not tun or not opn:
        continue
    x, y = X[idxs[0]], Y[idxs[0]]
    d = prt.query([x, y])[0] if prt is not None else 1e9
    if d > 60:
        key = (round(x / 50), round(y / 50))
        if key in seen:
            continue
        seen.add(key)
        rows.append((T[tun[0]]["name"], f"{ll(x, y)} {T[tun[0]]['name'][:30]!r} tunnel way {T[tun[0]]['id']} meets {T[opn[0]]['cls']} way {T[opn[0]]['id']}; nearest portal {d:.0f} m"))
rows.sort()
out("7 portals", len(rows), [r for _, r in rows], "tunnel-to-open changes (non-service, merged within 50 m) with no portal within 60 m")

# ---------- 8 structure audit ----------
lid = pd.read_csv(ROOT / "evidence" / "structure_heights.csv")
lid = lid[np.isfinite(lid["rail_navd88_m"]) & lid["source"].isin(["lidar", "lidar_station"])]
lx, ly = to_utm.transform(lid["lon"].to_numpy(), lid["lat"].to_numpy())
dd, ii = ptree.query(np.c_[lx, ly])
lid_off = collections.defaultdict(list)
for d_, i_, zl in zip(dd, ii, lid["rail_navd88_m"].to_numpy()):
    if d_ < 6:
        lid_off[i_].append(zl)
cause = np.array([""] * len(P), dtype=object)
m_tun_up = (CLS == "tunnel") & np.isfinite(G) & ~wet & (Z > G - 1.0) & (d_open > 100) & ~SVC
m_el_low = (CLS == "elevated") & np.isfinite(G) & ~wet & (Z < G + 2.0) & ~SVC
m_open_low = np.isin(CLS, ["grade", "embankment"]) & np.isfinite(DEM) & ~wet & (Z < DEM - 2.5) & ~SVC
m_open_wet_low = np.isin(CLS, ["grade", "embankment"]) & wet & np.isfinite(DEM) & (DEM > 1.0) & (Z < DEM - 2.0) & ~SVC
m_open_high = np.isin(CLS, ["grade", "cutting"]) & np.isfinite(DEM) & ~wet & (Z > DEM + 4.0) & ~SVC
cause[m_open_high] = "open track 4 m+ over the ground at the track (an unmarked bridge?)"
cause[m_open_low] = "grade/embankment 2.5 m+ under the ground at the track"
cause[m_open_wet_low] = "grade/embankment by the water drawn under its causeway"
cause[m_el_low] = "bridge-tagged within 2 m of the ground (not over water)"
cause[m_tun_up] = "tunnel rail above the street (over 100 m from open track)"
for i_, zs in lid_off.items():
    if abs(Z[i_] - np.median(zs)) > 3.0 and not SVC[i_]:
        cause[i_] = "lidar rail 3 m+ off the model"
tab = collections.Counter()
rows = []
start = 0
for wi, t in enumerate(T):
    n = len(t["z"])
    c_ = cause[start:start + n]
    k = 0
    while k < n:
        if c_[k]:
            j = k
            while j + 1 < n and c_[j + 1] == c_[k]:
                j += 1
            if j - k + 1 >= 2:
                tab[c_[k]] += 1
                i0 = start + k
                off = float(np.max(np.abs(Z[start + k:start + j + 1] - G[start + k:start + j + 1])))
                rows.append((c_[k], j - k + 1, f"{c_[k][:44]:44} {t['name'][:28]:28} way {t['id']} {ll(X[i0], Y[i0])} {(j - k + 1) * 25} m, {t['cls']}, rail-ground up to {off:.1f} m"))
            k = j + 1
        else:
            k += 1
    start += n
rows.sort(key=lambda r: -r[1])
out("8 structure", sum(tab.values()), [r for _, _, r in rows], "stretches where OSM structure, lidar and the model's rail vs ground disagree; by cause:")
for k, v in tab.most_common():
    print(f"      {v:4d}  {k}")

# ---------- 9 station dots ----------
els = json.loads((ROOT / "raw" / "osm_stations.json").read_text(encoding="utf-8"))["elements"]
dots = tj["stations"]
dx, dy = to_utm.transform(np.array([d["lon"] for d in dots]), np.array([d["lat"] for d in dots]))
dtree = cKDTree(np.c_[dx, dy])
rows = []
nst = 0
W, S_, E_, N_ = -74.26, 40.508, -73.645, 40.975
# any passenger operator's track in the OSM download (yards too: terminal platform tracks are often
# service=yard), so a station whose tracks the build left out still counts
PAX = RAILROAD + ("Transit Authority", "Trans-Hudson", "Port Authority", "Staten Island", "Alstom", "AirTrain")
rx, ry = [], []
for e in json.loads((ROOT / "raw" / "osm_tracks.json").read_text(encoding="utf-8"))["elements"]:
    t = e["tags"]
    if any(o in t.get("operator", "") + t.get("name", "") for o in PAX):
        for p in e.get("geometry") or []:
            rx.append(p["lon"]); ry.append(p["lat"])
rxu, ryu = to_utm.transform(np.array(rx), np.array(ry))
rtree = cKDTree(np.c_[rxu, ryu])
seen = set()
for e in els:
    t = e.get("tags", {})
    if t.get("railway") not in ("station", "halt") or not t.get("name"):
        continue
    if t.get("station") == "light_rail" or "Light Rail" in t.get("network", "") + t.get("name", ""):
        if "AirTrain" not in t.get("name", "") + t.get("network", ""):
            continue
    lon_, lat_ = (e["lon"], e["lat"]) if "lon" in e else (e["center"]["lon"], e["center"]["lat"])
    if not (W < lon_ < E_ and S_ < lat_ < N_):
        continue
    sx, sy = to_utm.transform(lon_, lat_)
    if rtree.query([sx, sy])[0] > 250:
        continue   # no passenger track near it in OSM (a line the map leaves out)
    nm = t["name"]
    key = (nm, round(lon_, 2), round(lat_, 2))
    if key in seen:
        continue
    seen.add(key)
    nst += 1
    d = dtree.query([sx, sy])[0]
    if d > 300:
        rows.append((nm, f"{nm[:34]:34} {lat_:.5f},{lon_:.5f} {t.get('network', t.get('operator', ''))[:24]:24} nearest dot {d:.0f} m, nearest track {ptree.query([sx, sy])[0]:.0f} m"))
rows.sort()
gcm = [d for d in dots if "Madison" in d["name"]]
out("9 dots", len(rows), [r for _, r in rows], f"of {nst} OSM stations on the map's tracks have no station dot within 300 m"
    + ("" if gcm else " (and Grand Central Madison has no dot of its own)"))

# ---------- 10 vertical curvature ----------
R_MIN = 600.0
rows = []
nflag = 0
for wi, t in enumerate(T):
    if t["service"] or len(t["z"]) < 3:
        continue
    x, y = to_utm.transform(np.array(t["lon"]), np.array(t["lat"]))
    h = np.hypot(np.diff(x), np.diff(y)); h[h < 1] = 1
    gr = np.diff(np.array(t["z"])) / h
    dg = np.abs(np.diff(gr)); L = (h[:-1] + h[1:]) / 2
    with np.errstate(divide="ignore"):
        R = L / np.maximum(dg, 1e-9)
    bad = np.where((R < R_MIN) & (dg > 0.01))[0]   # z is rounded to 0.1 m: under 1% change of grade is rounding
    nflag += len(bad)
    if len(bad):
        k = bad[np.argmin(R[bad])] + 1
        rows.append((R[k - 1], f"{t['name'][:30]:30} tr {t['track']:4} way {t['id']} {t['lat'][k]:.5f},{t['lon'][k]:.5f} radius {R[k - 1]:4.0f} m "
                                f"(grade {gr[k - 1]*100:+.1f}% -> {gr[k]*100:+.1f}%), {len(bad)} points on this way"))
rows.sort()
out("10 curvature", nflag, [r for _, r in rows], f"track points on a vertical curve tighter than {R_MIN:.0f} m radius (worst per way)")

# ---------- 11 boxes on levels with no platform ----------
rows = []
for b in pj["boxes"]:
    if b.get("fallback"):
        rows.append(f"{b['station'][:30]:30} {b['line'][:22]:22} lvl {b['level']} rail {b['rail_z']:.1f}: fallback box, no OSM platform on this level")
out("11 boxes", len(rows), rows, "station boxes with no OSM platform on their level")

# ---------- 12 labels: level split with one real platform level ----------
rows = []
by = collections.defaultdict(list)
for k, d in enumerate(dots):
    if d.get("tier"):
        by[(d["name"], d["line"], d["lon"])].append(k)
real = collections.Counter(b["dot"] for b in pj["boxes"] if not b.get("fallback"))
for key_, ks in by.items():
    n_real = sum(1 for k in ks if real.get(k))
    if n_real < len(ks):
        rows.append(f"{key_[0][:30]:30} {key_[1][:22]:22} {len(ks)} level dots, {n_real} with an OSM platform")
out("12 labels", len(rows), rows, "stations labelled 'upper/lower level' where a level has no OSM platform")

# ---------- 13 a station against the track it sits on ----------
# A dot reports its rail and its ground (the page's depth is the difference); its tracks report theirs. At
# New Dorp (SIR, an open cut) the track said a few metres under the surface and the dot +1 m (2026-11-03):
# two grounds. Every dot against the nearest track points within 60 m at its own height (within 3 m of its
# rail), and every box against the tracks under it.
rows = []
n13 = 0
for k, d in enumerate(dots):
    if d["ground"] is None:
        continue
    x0, y0 = to_utm.transform(d["lon"], d["lat"])
    near = [i for i in ptree.query_ball_point([x0, y0], 60.0) if abs(Z[i] - d["z"]) < 3.0 and np.isfinite(G[i]) and not SVC[i]]
    if not near:
        continue
    n13 += 1
    i = min(near, key=lambda i: (X[i] - x0) ** 2 + (Y[i] - y0) ** 2)
    dg, dz = d["ground"] - G[i], d["z"] - Z[i]
    if abs(dg) > 1.0 or abs(dz) > 1.0:
        rows.append((max(abs(dg), abs(dz)), f"dot {d['name'][:30]:30} {d['line'][:20]:20} ground {d['ground']:6.1f} vs track {G[i]:6.1f}, "
                                            f"rail {d['z']:6.1f} vs {Z[i]:6.1f}  (depth shown {d['z'] - d['ground']:+.1f} vs {Z[i] - G[i]:+.1f})"))
nbx = 0
for b in pj["boxes"]:
    if b.get("dot") is None or b["dot"] >= len(dots) or dots[b["dot"]]["ground"] is None:
        continue
    lo_ = np.mean([p[0] for p in b["ring"]]); la_ = np.mean([p[1] for p in b["ring"]])
    x0, y0 = to_utm.transform(lo_, la_)
    near = [i for i in ptree.query_ball_point([x0, y0], 40.0) if abs(Z[i] - b["rail_z"]) < 3.0 and np.isfinite(G[i]) and not SVC[i]]
    if not near:
        continue
    nbx += 1
    d = dots[b["dot"]]
    bg = b.get("ground", d["ground"])   # the box's own ground (make_platforms), else its dot's
    if bg is None:
        continue
    i = min(near, key=lambda i: (X[i] - x0) ** 2 + (Y[i] - y0) ** 2)   # the track at the box's middle
    tg, tz = float(G[i]), float(Z[i])
    dg, dz = bg - tg, b["rail_z"] - tz
    if abs(dg) > 1.0 or abs(dz) > 1.0:
        rows.append((max(abs(dg), abs(dz)), f"box {b['station'][:30]:30} {b['line'][:20]:20} ground {bg:6.1f} vs its tracks {tg:6.1f}, "
                                            f"rail {b['rail_z']:6.1f} vs {tz:6.1f}"))
rows.sort(reverse=True)
out("13 station-track", len(rows), [r for _, r in rows], f"of {n13} dots and {nbx} boxes disagree with the track under them by over 1 m in rail or ground")

# ---------- 14 isolated short track ----------
# connected pieces of the map's track (shared vertices join) under 2 km, joined to nothing, carrying no
# passenger service (services.json): a scrap OSM keeps (the piece at a creek bridge near Elizabeth, NJ)
svc_ids = set()
sjp = DATA / "services.json"
if sjp.exists():
    for s in json.loads(sjp.read_text(encoding="utf-8")).get("services", []):
        for v in s.get("variants", []):
            svc_ids.update(int(w[0]) for w in v.get("ways", []))
par = list(range(len(T)))
def fnd(a):
    while par[a] != a:
        par[a] = par[par[a]]; a = par[a]
    return a
for kk, idxs in node.items():
    ws = sorted({WAY[i] for i in idxs})
    for a in ws[1:]:
        par[fnd(a)] = fnd(ws[0])
comp = collections.defaultdict(list)
for w in range(len(T)):
    comp[fnd(w)].append(w)
rows = []
for c, ws in comp.items():
    L = sum(wlen(w) for w in ws)
    if L >= 2000 or any(T[w]["id"] in svc_ids for w in ws):
        continue
    t0 = T[ws[0]]
    rows.append((L, f"{len(ws):3d} ways {L:6.0f} m  {t0['name'][:30]!r} {raw.get(t0['id'], {}).get('operator', '')[:28]!r} "
                    f"{t0['lat'][0]:.5f},{t0['lon'][0]:.5f}  classes {sorted({T[w]['cls'] for w in ws})}  ways {[T[w]['id'] for w in ws][:4]}"))
rows.sort()
out("14 isolated", len(rows), [r for _, r in rows], "pieces of track under 2 km joined to no other track and carrying no service")

print("\nsummary: " + ", ".join(f"{k} {v}" for k, v in results.items()))
