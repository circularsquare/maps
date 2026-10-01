"""Russia stage 1: can Tariff Guide No. 4's sections be laid on OSM track? A dry run of the reader.

    python probe_ru_trace.py > data/raw/ru/probe_trace.txt

Needs data/proc/ru (extract.py), data/raw/ru/tr4_sections.json (probe_ru_tr4.py) and
data/raw/ru/osm_esr.json (probe_ru_esr.py). For every Russian-administration tariff section
(2022 annexations left out):

1. Each point is found in OSM by its ESR code (`esr:user` on a station or halt node); failing
   that, by name among OSM stops within reach of its two found neighbours.
2. Between consecutive found points the shortest path over OSM heavy-rail track (yards and
   sidings excluded) is compared with the difference of their tariff km. A point claims every
   track vertex within 200 m and the nearest vertex of every way within 600 m, and the path is
   charged from the point itself (Korea's two lessons; claiming one vertex sent 10-25 gaps per
   Trans-Siberian section round a crossover). Tariff km are whole kilometres, so pairs under
   5 km are only counted in the section totals.

    python probe_ru_trace.py 92-002 01-011     # also print every gap of these sections
3. A section is "traced" when its two ends are found and every gap along it is bridged by a
   path within 15% + 3 km of the tariff figure.

Prints match rates, the distribution of path/tariff ratios, and the worst sections.
"""
import heapq
import json
import math
import pickle
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
D = ROOT / "data" / "proc" / "ru"
RAW = ROOT / "data" / "raw" / "ru"
OCCUPIED_2022 = {"Донец (Р)", "ЛУГАН (Р)", "МЕЛИТ (Р)"}
SNAP_M = 600          # a station node to the nearest track vertex
CLAIM_M = 200         # ... or to every track vertex this close
HEAVY = {"rail", "narrow_gauge", "preserved"}
SKIP_SERVICE = {"yard", "siding"}
SHOW = set(sys.argv[1:])          # section ids whose every gap is printed


def nkey(s):
    s = unicodedata.normalize("NFKC", s or "").lower().replace("ё", "е")
    s = re.sub(r"^(оп|о\.п\.|ост\.?\s*пункт|остановочный пункт|платформа)\s+", "", s)
    s = re.sub(r"\((рзд|бп|пп|п|обп|эксп\.?|перев\.?)\)", "", s)
    s = re.sub(r"[\s\-–—.]+", " ", s).strip()
    return s


def main():
    from scipy.spatial import cKDTree

    secs = [s for s in json.loads((RAW / "tr4_sections.json").read_text("utf-8"))
            if s["sheet"] not in OCCUPIED_2022]
    osm_esr = json.loads((RAW / "osm_esr.json").read_text("utf-8"))
    with open(D / "ways.pkl", "rb") as f:
        ways = pickle.load(f)
    with open(D / "stops.pkl", "rb") as f:
        stops = pickle.load(f)
    c = np.load(D / "coords.npz")
    cid, cx, cy = c["id"], c["x"] / 1e7, c["y"] / 1e7

    # --- track graph
    adj = defaultdict(list)
    used = set()
    vway = {}
    for wid, (tags, nodes) in ways.items():
        if tags["railway"] not in HEAVY or tags.get("service") in SKIP_SERVICE:
            continue
        pos = np.searchsorted(cid, nodes)
        np.clip(pos, 0, cid.size - 1, out=pos)
        pos = pos[cid[pos] == nodes]
        if pos.size < 2:
            continue
        lon, lat = cx[pos], cy[pos]
        d = np.hypot(np.diff(lon) * np.cos(np.radians(lat[:-1])) * 111.32, np.diff(lat) * 110.57)
        for a, b, w in zip(pos[:-1].tolist(), pos[1:].tolist(), d.tolist()):
            adj[a].append((b, w))
            adj[b].append((a, w))
            used.add(a)
            used.add(b)
        for v in pos.tolist():
            vway.setdefault(v, wid)
    vid = np.array(sorted(used))
    lat0 = np.radians(55)
    tree = cKDTree(np.c_[cx[vid] * 111.32 * np.cos(np.radians(cy[vid])), cy[vid] * 110.57])
    print(f"track graph: {vid.size:,} vertices")

    def snap(lon, lat):
        """Every track vertex within CLAIM_M of the point (so each track of a double or
        quadruple line through it), or failing that the nearest within SNAP_M. One vertex per
        track would make the search cross over to the other track and back: HANDOFF's
        'a station must cut every track through it'."""
        q = np.array([lon * 111.32 * math.cos(math.radians(lat)), lat * 110.57])
        near = tree.query_ball_point(q, SNAP_M / 1000)
        if not near:
            return None
        d = np.hypot(*(tree.data[near] - q).T)
        keep, best = {}, {}
        for i, dd in zip(near, d.tolist()):
            v = int(vid[i])
            if dd * 1000 <= CLAIM_M:
                keep[v] = dd
            w = vway[v]                      # the nearest vertex of every track in reach,
            if w not in best or dd < best[w][0]:   # however sparse its vertices
                best[w] = (dd, v)
        for dd, v in best.values():
            keep[v] = dd
        # Charged from the point itself, not from whichever vertex the search leaves by, or
        # every gap comes out short by up to two claim radii.
        return keep

    def path_km(a, b, limit):
        """Shortest path from point a to point b, each a {vertex: km from the point} claim."""
        dist = dict(a)
        h = [(o, u) for u, o in a.items()]
        heapq.heapify(h)
        best = None
        while h:
            d, u = heapq.heappop(h)
            if best is not None and d >= best:
                return best
            if u in b:
                best = d + b[u] if best is None else min(best, d + b[u])
                continue
            if d > limit:
                return best
            if d > dist.get(u, 1e18):
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd < dist.get(v, 1e18):
                    dist[v] = nd
                    heapq.heappush(h, (nd, v))
        return best

    # --- OSM stops by name key, for the fallback
    by_name = defaultdict(list)
    for nid, (tags, lon, lat) in stops.items():
        if tags.get("railway") in ("station", "halt") or tags.get("public_transport") == "station":
            if tags.get("name"):
                by_name[nkey(tags["name"])].append((lon, lat))

    def esr_point(code):
        rows = [r for r in osm_esr.get(code, ()) if r[5] in ("station", "halt", "stop")] \
            or osm_esr.get(code, ())
        return (rows[0][2], rows[0][3]) if rows else None

    stat = Counter()
    ratios = []
    sec_res = []
    for s in secs:
        pts = s["points"]
        found = []
        for p in pts:
            q = esr_point(p["esr"])
            how = "esr" if q else None
            found.append([q, how])
        # name fallback between found neighbours
        for i, p in enumerate(pts):
            if found[i][0]:
                continue
            cand = by_name.get(nkey(p["name"]), [])
            prev = next((found[j][0] for j in range(i - 1, -1, -1) if found[j][0]), None)
            nxt = next((found[j][0] for j in range(i + 1, len(pts)) if found[j][0]), None)
            anchors = [a for a in (prev, nxt) if a]
            best = None
            for lon, lat in cand:
                dmin = min((math.hypot((lon - a[0]) * 111.32 * math.cos(math.radians(lat)),
                                       (lat - a[1]) * 110.57) for a in anchors), default=0)
                reach = 40 if anchors else 0
                if dmin <= max(reach, (p["km"][0] or 0) + 5) and (best is None or dmin < best[0]):
                    best = (dmin, (lon, lat))
            if best and anchors:
                found[i] = [best[1], "name"]
        for q, how in found:
            stat[how or "none"] += 1
        # paths between consecutive found points
        km0 = [p["km"][0] for p in pts]
        idx = [i for i, (q, _h) in enumerate(found) if q and km0[i] is not None]
        ok_all = len(idx) >= 2 and idx[0] == 0 and idx[-1] == len(pts) - 1
        path_tot, tar_tot, bad = 0.0, 0, 0
        for i, j in zip(idx, idx[1:]):
            tar = km0[j] - km0[i]
            a, b = snap(*found[i][0]), snap(*found[j][0])
            if a is None or b is None:
                stat["pair_unsnapped"] += 1
                ok_all = False
                continue
            pk = path_km(a, b, max(tar * 1.6, tar + 15))
            if pk is None:
                stat["pair_nopath"] += 1
                ok_all = False
                bad += 1
                continue
            stat["pair_ok"] += 1
            if s["id"] in SHOW:
                print(f"   {s['id']} {pts[i]['name']:<26} -> {pts[j]['name']:<26} tariff {tar:4d}"
                      f"  path {pk:7.1f}  {found[i][1]}/{found[j][1]}"
                      f"{'  <--' if abs(pk - tar) > 0.15 * tar + 3 else ''}")
            if tar >= 5:
                ratios.append(pk / tar)
            if abs(pk - tar) > 0.15 * tar + 3:
                bad += 1
                ok_all = False
            path_tot += pk
            tar_tot += tar
        length = max((k for k in km0 if k is not None), default=0)
        sec_res.append((s["id"], s["name"], s["type"], length, len(pts),
                        sum(1 for q, _ in found if q), ok_all, path_tot, tar_tot, bad, s["sheet"]))

    n = sum(stat[k] for k in ("esr", "name", "none"))
    print(f"\npoints on {len(secs)} sections (a point on two sections counts twice): {n:,}")
    print(f"  found by ESR code {stat['esr']:,} ({100 * stat['esr'] / n:.1f}%), by name "
          f"{stat['name']:,} ({100 * stat['name'] / n:.1f}%), not found {stat['none']:,}")
    print(f"  consecutive pairs: path found {stat['pair_ok']:,}, no path within limit "
          f"{stat['pair_nopath']:,}, a point not within {SNAP_M} m of track "
          f"{stat['pair_unsnapped']:,}")
    r = np.array(ratios)
    if r.size:
        print(f"  pairs 5 km or more apart: {r.size:,}; path/tariff median {np.median(r):.3f}, "
              f"within 5% {100 * np.mean(np.abs(r - 1) <= 0.05):.1f}%, within 10% "
              f"{100 * np.mean(np.abs(r - 1) <= 0.10):.1f}%, over 1.25 "
              f"{100 * np.mean(r > 1.25):.1f}%")
    traced = [x for x in sec_res if x[6]]
    km_all = sum(x[3] for x in sec_res)
    print(f"\nsections fully traced: {len(traced)} of {len(sec_res)}, "
          f"{sum(x[3] for x in traced):,} of {km_all:,} tariff km "
          f"({100 * sum(x[3] for x in traced) / km_all:.1f}%)")
    by_type = defaultdict(lambda: [0, 0, 0, 0])
    for x in sec_res:
        t = by_type[x[2]]
        t[0] += 1
        t[1] += x[6]
        t[2] += x[3]
        t[3] += x[3] if x[6] else 0
    for t, v in sorted(by_type.items(), key=lambda kv: -kv[1][2]):
        print(f"   {t:<34} {v[1]:4d}/{v[0]:<4d} sections, {v[3]:7,}/{v[2]:<7,} km")
    by_sheet = defaultdict(lambda: [0, 0])
    for x in sec_res:
        by_sheet[x[10]][0] += x[3]
        by_sheet[x[10]][1] += x[3] if x[6] else 0
    print("   by railway (traced km / km):",
          ", ".join(f"{k} {v[1]:,}/{v[0]:,}" for k, v in sorted(by_sheet.items())))
    tot_p = sum(x[7] for x in sec_res)
    tot_t = sum(x[8] for x in sec_res)
    print(f"  over every bridged gap: path {tot_p:,.0f} km against tariff {tot_t:,} km "
          f"({tot_p / tot_t:.3f})")
    worst = sorted((x for x in sec_res if not x[6]), key=lambda x: -x[3])[:30]
    print("\n  longest sections NOT traced (id, km, points found/total, bad gaps):")
    for x in worst:
        print(f"    {x[0]:<7} {x[3]:5d} km  {x[5]:3d}/{x[4]:<3d} bad {x[9]:2d}  {x[1]}  [{x[2]}]")


if __name__ == "__main__":
    main()
