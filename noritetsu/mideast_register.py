"""Saudi Arabia, the UAE, Qatar, Iraq and Jordan: each country's main-line passenger lines as
a hand-written station list (mideast_lines.py), traced over OSM track by rinf.py, with
nafrica_register.py's code (this module hands it the lists; nothing of nafrica's changes).

    python tools/slot.py 2 -- python extract.py --region sa --pbf data/raw/gcc-states-latest.osm.pbf --bbox 34.40,16.30,55.70,32.20 --station-areas
    python mideast_register.py --clip sa          # after every extract (drops other countries'
                                                  # ground and the NOT_SERVICE routes)
    python build_model.py --region sa --register mideast_register:data/raw/rinf/sa
    python mideast_register.py --trace sa "Riyadh Railway Station" "Abqaiq"   # to write lists
    python mideast_register.py --fork sa A B C    # where the trace A -> C leaves A -> B

Extract bboxes (west,south,east,north) from Geofabrik's one gcc-states file:
    sa 34.40,16.30,55.70,32.20   ae 51.50,22.60,56.45,26.10   qa 50.70,24.45,51.70,26.20
Iraq and Jordan have their own files (iraq-latest, jordan-latest), no bbox. After --clip,
Iraq also takes `--join iq` (its track is in pieces a few metres apart) and Jordan `--fill jo`
(OSM has no Al Jeezah station).

WHAT IS A REGISTER LINE HERE.  Only main-line railways: SAR's East and North Trains and the
Haramain line (sa), Etihad Rail's passenger route (ae), IRR's lines (iq), the Hejaz Railway's
excursion line (jo). No operator publishes a line register with chainage, so the km are our
own traces (`no_chain`) and check_model.REGISTER's published lengths are the outside check.
Metros, trams and monorails (Riyadh, Dubai, Doha, Lusail, the Palm) are OSM lines, as in
North Africa, Argentina, the US and the UK: rinf.py traces rail track only, and their OSM route
relations are complete. Qatar has nothing else, so it builds with no register at all
(`build_model.py --region qa`).

Sources, what runs and the numbers: sa_, ae_, qa_, iq_, jo_sources.md.
"""
import nafrica_register as nr
from mideast_lines import BORDERS, LINES, NOT_SERVICE

# nafrica_register's tables, extended in this process only.
nr.LINES.update(LINES)
nr.NOT_SERVICE.update(NOT_SERVICE)
nr.BORDERS.update(BORDERS)
nr.LANGS.update({"sa": ["ar", "en"], "ae": ["ar", "en"], "qa": ["ar", "en"],
                 "iq": ["ar", "en"], "jo": ["ar", "en"]})
nr.ISO3.update({"sa": "SAU", "ae": "ARE", "qa": "QAT", "iq": "IRQ", "jo": "JOR"})


GULF = {"sa", "ae", "qa"}
# The Gulf's three extracts are bboxes of one file, so each holds its neighbours' coasts: Dubai
# Marina, the Palm and Lusail's marina lie on reclaimed land outside the simplified outlines,
# in no country, and nafrica's clip ("inside ANOTHER country's outline") kept them in Saudi
# Arabia's extract. Here "abroad" is the ground within GULF_BUFFER_DEG (~15 km) of another
# country's outline, less this country's own outline: a coastal stretch is cut only near a
# neighbour. Saudi Arabia's nearest track to a neighbour, Dammam's, is ~30 km from Bahrain.
GULF_BUFFER_DEG = 0.15
_nafrica_abroad = nr.abroad


def abroad(cc):
    if cc not in GULF:
        return _nafrica_abroad(cc)
    import json
    import shapely
    from shapely.geometry import shape
    from shapely.ops import unary_union
    feats = json.loads(nr.SHAPES.read_text("utf-8"))["features"]
    home = unary_union([shape(f["geometry"]) for f in feats if f["properties"]["cc"] == cc])
    others = [shape(f["geometry"]) for f in feats if f["properties"]["cc"] != cc]
    reach = home.buffer(1.0)
    near = [g.buffer(GULF_BUFFER_DEG) for g in others if g.intersects(reach)]
    g = unary_union(others + near).difference(home)
    shapely.prepare(g)
    return g


nr.abroad = abroad


JOIN_M = 40                    # `--join`: a loose track end this close to other track
JOIN_BASE = -6_000_000_000_000  # synthetic joining ways: this - k (never an OSM id)


def join(cc, log=print):
    """Join track OSM leaves a few metres apart. Iraq's Southern Line is mapped as two single
    tracks that each stop 17-31 m short of the other at Rumaitha and south of it, so no trace
    could pass from Rumaitha to Samawah. A loose end (a node only one way of running line
    track ends at) within JOIN_M of a node of running line track in another connected piece
    gets a two-node way between them, tagged as the way it leaves. Yard, siding, spur and
    crossover track neither gives nor takes a join. Rewrites data/proc/<cc>/ways.pkl; run
    after --clip. Earlier joins are dropped first, so a rerun rebuilds them."""
    import pickle
    from collections import defaultdict
    import numpy as np
    d = nr.ROOT / "data" / "proc" / cc
    ways = pickle.load(open(d / "ways.pkl", "rb"))
    ways = {k: v for k, v in ways.items() if not k <= JOIN_BASE}
    c = np.load(d / "coords.npz")
    idx = {int(n): i for i, n in enumerate(c["id"])}
    X, Y = c["x"] / 1e7, c["y"] / 1e7
    run = {w: (t, [int(n) for n in ns]) for w, (t, ns) in ways.items()
           if t.get("railway") in ("rail", "narrow_gauge") and not t.get("service")}
    deg = defaultdict(int)
    by_node = defaultdict(set)
    for w, (t, ns) in run.items():
        deg[ns[0]] += 1
        deg[ns[-1]] += 1
        for n in ns:
            by_node[n].add(w)
    comp, k = {}, 0
    for w in run:
        if w in comp:
            continue
        k += 1
        comp[w], st = k, [w]
        while st:
            u = st.pop()
            for n in run[u][1]:
                for v in by_node[n]:
                    if v not in comp:
                        comp[v] = k
                        st.append(v)
    nodes = [n for n in by_node if n in idx]
    nx = np.array([X[idx[n]] for n in nodes])
    ny = np.array([Y[idx[n]] for n in nodes])
    ncomp = np.array([comp[next(iter(by_node[n]))] for n in nodes])
    made, seen = 0, set()
    for w, (t, ns) in run.items():
        for end in (ns[0], ns[-1]):
            if deg[end] != 1 or len(by_node[end]) != 1 or end not in idx:
                continue
            x, y = X[idx[end]], Y[idx[end]]
            kx = 111320 * np.cos(np.radians(y))
            dist = np.hypot((nx - x) * kx, (ny - y) * 110570)
            ok = (dist <= JOIN_M) & (ncomp != comp[w])
            if not ok.any():
                continue
            j = int(np.argmin(np.where(ok, dist, np.inf)))
            other = nodes[j]
            key = frozenset((end, other))
            if key in seen:
                continue
            seen.add(key)
            made += 1
            ways[JOIN_BASE - made] = (dict(t, note="mideast_register join"),
                                      np.asarray([end, other], dtype=np.int64))
            log(f"  joined {t.get('name') or t.get('railway')} at {x:.5f},{y:.5f} "
                f"({dist[j]:.0f} m)")
    tmp = d / "ways.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(ways, f, protocol=4)
    import os
    os.replace(tmp, d / "ways.pkl")
    log(f"{cc.upper()} join: {made} loose track ends joined to track within {JOIN_M} m")


def country_conf(cc):
    return nr.country_conf(cc)


def build(path, log):
    return nr.build(path, log)


def split_pieces(lines, stations, geoms, reg_ways, state, log):
    nr.split_pieces(lines, stations, geoms, reg_ways, state, log)


if __name__ == "__main__":
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", metavar="CC")
    ap.add_argument("--fill", metavar="CC")
    ap.add_argument("--join", metavar="CC")
    ap.add_argument("--convert", metavar="CC")
    ap.add_argument("--dry", metavar="CC", help="convert without writing")
    ap.add_argument("--trace", nargs="+", metavar="ARG")
    ap.add_argument("--fork", nargs=4, metavar=("CC", "A", "B", "C"))
    a = ap.parse_args()
    t = time.time()
    lg = lambda m: print(f"[{time.time() - t:6.1f}s] {m}", flush=True)  # noqa: E731
    if a.clip:
        nr.clip(a.clip, lg)
    if a.join:
        join(a.join, lg)
    if a.fill:
        nr.fill(a.fill, lg)
    if a.convert:
        nr.convert(a.convert, lg)
    if a.dry:
        nr.convert(a.dry, lg, write=False)
    if a.trace:
        nr.trace_cmd(a.trace[0], a.trace[1:])
    if a.fork:
        nr.fork_cmd(*a.fork)
