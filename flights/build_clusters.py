"""Cluster detection over the FR24 route graph.

Reads routes_fr24.json, runs weighted Louvain (airports = nodes, passengers =
edge weight), and writes the result back into the SAME file so the viewer ships
one data file:

  - each airport entry gains a 5th element: its cluster id (int, -1 = "other")
  - meta.clusters = [{id, hub, hubCity, hubCountry, n, countries}, ...]
                    ordered by total cluster traffic (id 0 = biggest)

Idempotent: strips any prior cluster data first, so re-running is safe. Run it
AFTER build_fr24.py (which regenerates routes_fr24.json with 4-element airport
entries and no cluster meta).

    python build_clusters.py
"""
import json
import networkx as nx
from collections import Counter, defaultdict

DATA       = "routes_fr24.json"
RESOLUTION = 1.75   # higher = more, smaller clusters; 1.75 = clean ~11 regions
                    # (1.5 leaves Africa as a thin S-Africa-only cluster; 2.0
                    #  fragments Africa and splits Japan off from SE Asia)
SEED       = 42     # Louvain is order-dependent; fix for reproducible colours
MIN_SIZE   = 10     # clusters smaller than this fall into "other" (id -1)
MAX_CLUST  = 12     # cap on coloured clusters (palette size in the viewer)
MIN_COHESION = 0.25  # repair clusters whose internal traffic share is below this


def _internal_pct(comm, rts):
    """Each community's share of touching traffic that stays inside it."""
    intp, crossp = defaultdict(float), defaultdict(float)
    for a, b, pax, *_ in rts:
        if a not in comm or b not in comm:
            continue
        if comm[a] == comm[b]:
            intp[comm[a]] += pax
        else:
            crossp[comm[a]] += pax
            crossp[comm[b]] += pax
    return {k: intp[k] / ((intp[k] + crossp[k]) or 1)
            for k in set(intp) | set(crossp)}


def _repair(parts, rts, max_passes=5):
    """Eject 'swing' misfits from low-cohesion communities.

    Louvain occasionally drops a hub whose traffic is split across many regions
    (LHR, JFK) into a small incoherent community next to a near-isolated pocket
    (the French-Polynesia island network). Each pass moves ONLY members of a
    low-cohesion community whose traffic actually points at a different, cohesive
    community — so genuine members and all healthy communities are untouched.
    """
    comm = {c: i for i, p in enumerate(parts) for c in p}
    for _ in range(max_passes):
        pct = _internal_pct(comm, rts)
        sizes = Counter(comm.values())
        bad = {k for k in pct if pct[k] < MIN_COHESION and sizes[k] >= 5}
        if not bad:
            break
        nbr = defaultdict(lambda: defaultdict(float))
        for a, b, pax, *_ in rts:
            if a in comm and b in comm:
                nbr[a][comm[b]] += pax
                nbr[b][comm[a]] += pax
        moved = 0
        for n in list(comm):
            if comm[n] not in bad or not nbr[n]:
                continue
            best = max(nbr[n], key=nbr[n].get)
            if best != comm[n] and best not in bad:
                comm[n] = best
                moved += 1
        if not moved:
            break
    groups = defaultdict(set)
    for n, k in comm.items():
        groups[k].add(n)
    return list(groups.values())


def main():
    with open(DATA, encoding="utf-8") as f:
        d = json.load(f)
    ap, rts = d["airports"], d["routes"]

    # --- graph + per-airport traffic (for hub-picking and ordering) ----------
    G = nx.Graph()
    traf = defaultdict(float)
    for a, b, pax, *_ in rts:
        if a in ap and b in ap:
            G.add_edge(a, b, weight=pax)
            traf[a] += pax
            traf[b] += pax

    parts = nx.community.louvain_communities(
        G, weight="weight", resolution=RESOLUTION, seed=SEED)
    parts = _repair(parts, rts)   # eject swing-hub misfits from junk communities

    # order by total cluster traffic so id 0 = the heaviest cluster
    parts.sort(key=lambda c: sum(traf[x] for x in c), reverse=True)

    airport_cluster = {}   # code -> cluster id (or -1)
    meta_clusters = []
    cid = 0
    for c in parts:
        if len(c) < MIN_SIZE or cid >= MAX_CLUST:
            for code in c:
                airport_cluster[code] = -1
            continue
        hub = max(c, key=lambda x: traf.get(x, 0))
        countries = Counter(ap[x][3] for x in c if x in ap)
        meta_clusters.append({
            "id": cid,
            "hub": hub,
            "hubCity": ap[hub][2],
            "hubCountry": ap[hub][3],
            "n": len(c),
            "countries": [[k, v] for k, v in countries.most_common(4)],
        })
        for code in c:
            airport_cluster[code] = cid
        cid += 1

    # --- write back: airport element [4] = cluster id; meta.clusters ---------
    # Set only our own slot; preserve any later slots (e.g. [5] betweenness from
    # build_centrality.py) so the two enrichment scripts are order-independent.
    for code, info in ap.items():
        base = list(info)
        while len(base) < 5:
            base.append(-1)
        base[4] = airport_cluster.get(code, -1)
        ap[code] = base
    d["meta"]["clusters"] = meta_clusters

    with open(DATA, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, separators=(",", ":"))

    n_colored = sum(1 for v in airport_cluster.values() if v >= 0)
    print(f"{len(meta_clusters)} clusters coloured, "
          f"{n_colored}/{len(ap)} airports assigned "
          f"({sum(1 for v in airport_cluster.values() if v < 0)} -> other)")
    for m in meta_clusters:
        tops = ", ".join(f"{k}({v})" for k, v in m["countries"][:3])
        print(f"  C{m['id']:2d} n={m['n']:4d} hub={m['hub']} "
              f"[{m['hubCountry']}] :: {tops}")


if __name__ == "__main__":
    main()
