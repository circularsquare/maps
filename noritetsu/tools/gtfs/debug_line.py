"""python debug_line.py <cc> <line ref or name substring> : every section of the line with its
state, its ends' feed matches, and the consecutive-call pairs at each end with their paths."""
import os
import pickle
import sys

sys.path.insert(0, r"C:\Users\anita\projects\maps\noritetsu")
sys.stdout.reconfigure(encoding="utf-8")
import gtfs_served as gs  # noqa: E402

cc, want = sys.argv[1], sys.argv[2]
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, f"state_{cc}.pkl"), "rb") as f:
    lines, stations, geoms, route_share = pickle.load(f)
rs0 = dict(route_share)
gs.check(cc, lines, stations, route_share, lambda m: None)
L = gs.LAST
nodes, fmatch, feed, pairs, g, edges, eref, state = (L[k] for k in (
    "nodes", "fmatch", "feed", "pairs", "graph", "edges", "eref", "state"))
fst = feed["stations"]
by_node = {}
for f, n in fmatch.items():
    by_node.setdefault(n, []).append(fst[f][0])
for i, (l, key) in enumerate(eref):
    if not (l.get("ref") == want or want in l["name"]):
        continue
    a, b, km = edges[i]
    print(f"{state[i]:9s} {l.get('ref')} {nodes[a][0]}{'(j)' if nodes[a][3] else ''} - "
          f"{nodes[b][0]}{'(j)' if nodes[b][3] else ''} {km:.1f} km  osm {rs0.get((l['id'], key), 0):.2f}"
          f"  trips {L['trips'][i]}  feed: {by_node.get(a)} / {by_node.get(b)}  known "
          f"{a in L['known']}/{b in L['known']}")
    if len(sys.argv) > 3 and sys.argv[3] == "via":
        for (u, v), rec in pairs.items():
            if i in rec[3]:
                print(f"      via: {nodes[u][0]} - {nodes[v][0]} path {rec[0]:.1f} km, trips {rec[1]}")
    elif len(sys.argv) > 3:
        for n in (a, b):
            for (u, v), rec in pairs.items():
                if n in (u, v):
                    o = v if n == u else u
                    dist, prev = g.dijkstra(n, rec[0])
                    path = []
                    x = o
                    if x in dist:
                        while x != n:
                            path.append(nodes[x][0])
                            x = prev[x]
                    print(f"      pair {nodes[n][0]} - {nodes[o][0]}: cap {rec[0]:.1f}, trips {rec[1]}, "
                          f"path {dist.get(o, -1):.1f}: {' < '.join(path[:12])}")
