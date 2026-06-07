"""Betweenness centrality over the FR24 route graph.

Edge cost models travel time: flight distance + a fixed per-connection penalty
(layover + ground time, expressed in km-equivalent). Minimising it approximates
real door-to-door time, so shortest paths avoid both pointless detours and
needless extra stops. Betweenness then = the share of those shortest paths that
pass through each airport — i.e. how much of a structural connector it is.

Writes the result back into routes_fr24.json:
  - each airport entry's 6th element [5] = normalised betweenness (float)

The viewer pairs this with traffic to highlight airports that are *more* central
than their size would predict ("punches above its weight").

Run AFTER build_fr24.py. Order vs build_clusters.py doesn't matter — each script
only writes its own airport slot ([4] cluster, [5] betweenness).

    python build_centrality.py
"""
import json
import math
import networkx as nx

DATA       = "routes_fr24.json"
HOP_PENALTY = 1000   # km-equivalent of one connection's layover+ground overhead
K_PIVOTS    = 1500   # sampled-betweenness pivots (exact is too slow in pure NX);
                     # ~54% of nodes — stable ranking, reproducible via SEED
SEED        = 1


def haversine(a, b):
    r = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, (a[1], a[0], b[1], b[0]))
    dla, dlo = la2 - la1, lo2 - lo1
    x = math.sin(dla/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlo/2)**2
    return 2 * r * math.asin(min(1.0, math.sqrt(x)))


def main():
    with open(DATA, encoding="utf-8") as f:
        d = json.load(f)
    ap, rts = d["airports"], d["routes"]

    G = nx.Graph()
    for a, b, *_ in rts:
        if a in ap and b in ap and a != b:
            G.add_edge(a, b, w=haversine(ap[a], ap[b]) + HOP_PENALTY)

    print(f"betweenness on {G.number_of_nodes()} nodes / {G.number_of_edges()} "
          f"edges (k={K_PIVOTS}, penalty={HOP_PENALTY}km)...")
    bw = nx.betweenness_centrality(G, k=K_PIVOTS, seed=SEED, weight="w",
                                   normalized=True)

    # write back: airport element [5] = betweenness; preserve [4] (cluster)
    for code, info in ap.items():
        base = list(info)
        while len(base) < 5:
            base.append(-1)          # cluster slot (default if not yet computed)
        while len(base) < 6:
            base.append(0.0)         # betweenness slot
        base[5] = round(bw.get(code, 0.0), 8)
        ap[code] = base

    with open(DATA, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, separators=(",", ":"))

    # report: top connectors vs their traffic rank (the "above its weight" story)
    from collections import defaultdict
    traf = defaultdict(float)
    for a, b, pax, *_ in rts:
        if a in ap and b in ap:
            traf[a] += pax
            traf[b] += pax
    trank = {c: i+1 for i, c in enumerate(sorted(traf, key=lambda x: -traf[x]))}
    print(f"baked betweenness for {len(bw)} airports. top connectors:")
    for c in sorted(bw, key=lambda x: -bw[x])[:15]:
        print(f"  {c:4s} {ap[c][2][:18]:18s} {ap[c][3][:14]:14s} "
              f"| traffic #{trank.get(c, '?')}")


if __name__ == "__main__":
    main()
