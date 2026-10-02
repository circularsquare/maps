"""Ownership statistics for one prototype run. python report.py <tag>"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "2"
sys.dont_write_bytecode = True
import json, pickle, math
from collections import defaultdict, Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
ROOT = Path(r"C:\Users\anita\projects\maps\noritetsu")
tag = sys.argv[1]
m = pickle.load(open(HERE / "out" / tag / "model.pkl", "rb"))
cc = m["meta"]["cc"]
L = json.load(open(ROOT / "dist" / "data" / cc / "lines.json", encoding="utf-8"))["lines"]
S = json.load(open(ROOT / "dist" / "data" / cc / "stations.json", encoding="utf-8"))["stations"]
byid = {l["id"]: l for l in L}
W = m["ways"]
secs = m["secs"]


def lab(lid):
    l = byid[lid]
    return f"{l.get('ref') or ''}|{l['name'][:40]}"


def near_station(lon, lat):
    best = min(S.values(), key=lambda s: (s["x"] - lon) ** 2 * math.cos(math.radians(lat)) ** 2 + (s["y"] - lat) ** 2)
    return best["n"]


# ways by status, km (one-way length; parallel tracks each counted)
km = Counter()
for j, w in enumerate(W):
    km[m["wstat"].get(j, "none")] += w["len"] / 1000
print("drawn way km by owner kind:", {k: round(v, 1) for k, v in km.items()})


def components(js, near_m=30.0):
    """Places: ways sharing a node or lying within near_m of each other (both tracks of a
    double-track street are one place)."""
    from shapely import STRtree
    from shapely.geometry import LineString
    js = list(js)
    if not js:
        return []
    geo = [LineString(W[j]["xy"]) for j in js]
    tree = STRtree(geo)
    adj = defaultdict(set)
    for a, g in enumerate(geo):
        for b in tree.query(g.buffer(near_m, quad_segs=2)):
            if a != b:
                adj[a].add(int(b))
    seen, comps = set(), []
    for a in range(len(js)):
        if a in seen:
            continue
        stack, comp = [a], []
        seen.add(a)
        while stack:
            u = stack.pop()
            comp.append(js[u])
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)
    return comps


def show_contest(rows, title, with_exact=False):
    js = [r[0] for r in rows]
    tot = sum(W[j]["len"] for j in js) / 1000
    # a place: ways shared by the same set of lines, touching or within 30 m of each other
    byset = defaultdict(list)
    for r in rows:
        byset[(r[1],) + tuple(r[2])].append(r[0])
    comps = [c for v in byset.values() for c in components(v)]
    print(f"\n{title}: {len(js)} ways, {tot:.1f} km of way length, {len(comps)} places")
    info = {r[0]: r for r in rows}
    out = []
    for comp in comps:
        k = sum(W[j]["len"] for j in comp) / 1000
        sets = Counter()
        for j in comp:
            r = info[j]
            sets[(r[1], tuple(r[2]))] += W[j]["len"] / 1000
        (o, others), _ = sets.most_common(1)[0]
        lon = sum(W[j]["lon"] for j in comp) / len(comp)
        lat = sum(W[j]["lat"] for j in comp) / len(comp)
        ex = sum(W[j]["len"] for j in comp if with_exact and info[j][3]) / 1000
        out.append((k, o, others, lon, lat, ex, len(sets)))
    out.sort(key=lambda x: -x[0])
    if with_exact:
        print(f"   of which exact ties (both lines within 0.5 m: the ref rule decides): "
              f"{sum(W[r[0]]['len'] for r in rows if r[3]) / 1000:.1f} km")
    for k, o, others, lon, lat, ex, nsets in out[:25]:
        print(f"   {k:6.2f} km  owner {lab(o):<32} over {', '.join(lab(x) for x in others)[:90]}"
              f"  near {near_station(lon, lat)} ({lon:.4f},{lat:.4f})"
              + (f" exact {ex:.2f}" if with_exact and ex else "") + (f" [{nsets} sets]" if nsets > 1 else ""))
    return out


osm_rows = m["osm_contest"]
o1 = show_contest(osm_rows, "SHARED NON-REGISTER TRACK (several OSM lines, fixed rule picks one)")
# route-km view: distinct track km, counting a double-track pair once: use owner sections' own spans
# the size that matters for totals: how much owned km moves to the owner from the others.
by_kind = Counter()
for j, o, others in osm_rows:
    by_kind[byid[o]["kind"]] += W[j]["len"] / 1000
print("   by kind of owner:", {k: round(v, 1) for k, v in by_kind.items()})

# route-km of shared non-register track: owned spans another OSM line's footprint also covers
def merge(iv, tol=0.0):
    if not iv:
        return []
    iv = sorted([list(x) for x in iv])
    out = [iv[0]]
    for lo, hi in iv[1:]:
        if lo <= out[-1][1] + tol:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def inter(a, b):
    out, i, j = 0.0, 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


shared = defaultdict(list)
pair_km = defaultdict(list)
for g, s in secs.items():
    if s["reg"] or s["service"]:
        continue
    for t, lo, hi in m["foot"][g]:
        ts = secs[t]
        if ts["reg"] or ts["line"] == s["line"]:
            continue
        shared[t].append((lo, hi))
        pair_km[(ts["line"], s["line"])].append((t, lo, hi))
tot = 0.0
by_owner = defaultdict(float)
for t, iv in shared.items():
    k = secs[t]["km"] * inter(merge(iv), m["own"].get(t, []))
    tot += k
    by_owner[secs[t]["line"]] += k
print(f"   ROUTE-KM of shared non-register track (owner's sections, each stretch once): {tot:.1f} km "
      f"over {len(by_owner)} owner lines")
lost = defaultdict(float)
for (o, x), rows in pair_km.items():
    per = defaultdict(list)
    for t, lo, hi in rows:
        per[t].append((lo, hi))
    lost[x] += sum(secs[t]["km"] * inter(merge(iv), m["own"].get(t, [])) for t, iv in per.items())
print("   lines that give the most shared track to a lower-numbered line (route-km):")
for x, k in sorted(lost.items(), key=lambda r: -r[1])[:12]:
    print(f"     {k:6.2f} km  {lab(x)}")

reg_rows = m["reg_contest"]
print(f"\nregister lines within 8 m of another on one way (nearest wins, no rule needed): "
      f"{len(reg_rows)} ways, {sum(W[r[0]]['len'] for r in reg_rows)/1000:.1f} km of way length")
o2 = show_contest([(r[0], r[1], r[4]) for r in reg_rows if r[3]],
                  "REGISTER LINES DRAWN ON THE SAME WAY (both within 0.5 m: the ref rule decides)")

# OSM lines' composition
print("\nOSM SECTIONS: km of section geometry by what the track under it is")
agg = {"line": Counter(), "service": Counter()}
for g, s in secs.items():
    if s["reg"]:
        continue
    k = "service" if s["service"] else "line"
    for st, v in m["comp"][g].items():
        agg[k][st] += v / 1000
for k, c in agg.items():
    print(f"   {k:8}", {a: round(b, 1) for a, b in c.most_common()})

# per OSM line owned km
own_km = defaultdict(float)
for g, sp in m["own"].items():
    s = secs[g]
    if s["reg"]:
        continue
    own_km[s["line"]] += s["km"] * sum(h - l for l, h in sp)
rows = []
for l in L:
    if l.get("src", "osm") != "osm" or l["service"]:
        continue
    if not any(sec[3] in secs for sec in l["sections"]):
        continue
    ok = own_km.get(l["id"], 0.0)
    rows.append((ok, l))
tot_own = sum(r[0] for r in rows)
sliv = [(ok, l) for ok, l in rows if 0.05 < ok < 0.4 * l["km"]]
print(f"\nOSM lines (not named trains): {len(rows)}; owned km total {tot_own:.1f}")
print(f"   lines owning some track but under 40% of their length: {len(sliv)}, {sum(r[0] for r in sliv):.1f} km between them")
for ok, l in sorted(sliv, key=lambda r: -r[0])[:30]:
    print(f"     {ok:6.2f} of {l['km']:6.1f} km  {l['id']} {l.get('ref','')} {l['name'][:60]} [{l['kind']}]")
big = [(ok, l) for ok, l in rows if ok >= 0.4 * l["km"] and ok > 0.05]
print(f"   lines owning >= 40%: {len(big)}, {sum(r[0] for r in big):.1f} km")
for ok, l in sorted(big, key=lambda r: -r[0])[:15]:
    print(f"     {ok:6.2f} of {l['km']:6.1f} km  {l['id']} {l.get('ref','')} {l['name'][:60]} [{l['kind']}]")

# services-only track
sv = [j for j, st in m["wstat"].items() if st == "svc"]
print(f"\nTrack only named trains run over (owned by nobody): {len(sv)} ways, {sum(W[j]['len'] for j in sv)/1000:.1f} km")
comps = components(sv)
comps.sort(key=lambda c: -sum(W[j]['len'] for j in c))
for comp in comps[:10]:
    k = sum(W[j]["len"] for j in comp) / 1000
    users = Counter(u for j in comp for u in m["way_users"].get(j, []))
    lon = sum(W[j]["lon"] for j in comp) / len(comp)
    lat = sum(W[j]["lat"] for j in comp) / len(comp)
    print(f"   {k:6.2f} km near {near_station(lon, lat)}: {', '.join(lab(u) for u, _ in users.most_common(3))}")
