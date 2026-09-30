"""What is actually in an extracted region: tag coverage, LOD bands, route relation quality.

Run this on every new region before trusting anything built from it.  The LOD table in
build_tiles.py and the line model in build_model.py both rest on tags that are conventions,
not guarantees -- `usage=main` is well kept in some countries and absent in others, and the
share of route relations that carry a colour or a route_master decides how much of the line
model has to be synthesised.

    python inspect_region.py --region jp
"""
import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

# This console is cp1252 and the data is not: operator and station names here are Japanese,
# Korean, Cyrillic and much else, and without this the script dies inside a print() after all
# the real work succeeded, which reads like a pipeline failure and is not one.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent


def pct(n, d):
    return f"{100*n/d:5.1f}%" if d else "    -"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    d = ROOT / "data" / "proc" / args.region

    with open(d / "ways.pkl", "rb") as f:
        ways = pickle.load(f)
    with open(d / "rels.pkl", "rb") as f:
        rels = pickle.load(f)
    with open(d / "stops.pkl", "rb") as f:
        stops = pickle.load(f)

    import build_tiles as bt

    print(f"=== {args.region}: {len(ways)} track ways\n")

    kind_rank = Counter()
    usage = Counter()
    service = Counter()
    for tags, _ in ways.values():
        kind = bt.KIND[tags["railway"]]
        kind_rank[(kind, bt.rank_of(kind, tags))] += 1
        usage[tags.get("usage", "(none)")] += 1
        service[tags.get("service", "(none)")] += 1

    print("ways by kind and LOD rank (0 main, 1 branch/unstated, 2 industrial, 3 siding)")
    for (kind, rank), n in sorted(kind_rank.items(), key=lambda kv: -kv[1]):
        mz = bt.MINZOOM.get((kind, rank), 12)
        print(f"  {kind:<13} rank {rank}  z{mz:<2}  {n:>7}  {pct(n, len(ways))}")

    print("\nusage tag")
    for k, n in usage.most_common(args.top):
        print(f"  {k:<20} {n:>7}  {pct(n, len(ways))}")
    print("service tag")
    for k, n in service.most_common(6):
        print(f"  {k:<20} {n:>7}  {pct(n, len(ways))}")

    print(f"\n=== stops: {len(stops)}")
    sk = Counter()
    named = 0
    latin = 0
    for tags, _, _ in stops.values():
        sk[tags.get("railway") or f"pt={tags.get('public_transport')}"] += 1
        if tags.get("name"):
            named += 1
        if tags.get("name:en"):
            latin += 1
    for k, n in sk.most_common(args.top):
        print(f"  {k:<22} {n:>7}")
    print(f"  with name             {named:>7}  {pct(named, len(stops))}")
    print(f"  with name:en          {latin:>7}  {pct(latin, len(stops))}")

    masters = {i: (t, m) for i, (t, m) in rels.items() if t.get("type") == "route_master"}
    routes = {i: (t, m) for i, (t, m) in rels.items() if t.get("type") == "route"}
    print(f"\n=== relations: {len(masters)} route_master, {len(routes)} route")

    in_master = set()
    for t, members in masters.values():
        for ty, ref, _ in members:
            if ty == "r":
                in_master.add(ref)
    orphan = [i for i in routes if i not in in_master]
    print(f"  routes inside a route_master  {len(routes)-len(orphan):>6}  "
          f"{pct(len(routes)-len(orphan), len(routes))}")
    print(f"  orphan routes                 {len(orphan):>6}  "
          f"{pct(len(orphan), len(routes))}")

    have = Counter()
    for t, members in routes.values():
        for key in ("colour", "color", "ref", "name", "operator", "network", "service"):
            if t.get(key):
                have[key] += 1
        if t.get("public_transport:version") == "2":
            have["ptv2"] += 1
        roles = {r for _, _, r in members if r}
        if roles & {"stop", "stop_entry_only", "stop_exit_only"}:
            have["stop roles"] += 1
        if any(ty == "n" for ty, _, _ in members):
            have["node members"] += 1
    print("  route relations carrying")
    for key in ("name", "ref", "colour", "color", "operator", "network", "service",
                "ptv2", "stop roles", "node members"):
        print(f"    {key:<14} {have[key]:>6}  {pct(have[key], len(routes))}")

    print("\n  route kinds")
    for k, n in Counter(t.get("route") for t, _ in routes.values()).most_common():
        print(f"    {k:<14} {n:>6}")

    print("\n  biggest operators by route relation count")
    ops = Counter(t.get("operator", "(none)") for t, _ in routes.values())
    for k, n in ops.most_common(args.top):
        print(f"    {k:<44} {n:>5}")

    passenger_coverage(args.region, ways, routes)


def passenger_coverage(region, ways, routes):
    """How much track a passenger route relation actually runs over.

    This is the honest test of "show only passenger lines".  OSM has no passenger flag: the
    strongest available evidence that a piece of track carries passengers is that a
    route=train|subway|... relation runs over it.  Where that covers most of the network,
    filtering on it is safe; where it does not, filtering on it would delete real passenger
    lines, and the usage tags have to carry the decision instead.
    """
    import numpy as np

    import build_tiles as bt

    d = ROOT / "data" / "proc" / region
    c = np.load(d / "coords.npz")
    cid, cx, cy = c["id"], c["x"], c["y"]

    on_route = set()
    for _t, members in routes.values():
        for ty, ref, role in members:
            # Empty role is the route's own path; platform and stop members are the furniture.
            if ty == "w" and (not role or role.startswith(("forward", "backward"))):
                on_route.add(ref)

    km = Counter()
    n_by = Counter()
    for wid, (tags, nodes) in ways.items():
        pos = np.searchsorted(cid, nodes)
        np.clip(pos, 0, cid.size - 1, out=pos)
        ok = cid[pos] == nodes
        pos = pos[ok]
        if pos.size < 2:
            continue
        lon, lat = cx[pos] / 1e7, cy[pos] / 1e7
        dx = np.diff(lon) * np.cos(np.radians(lat[:-1])) * 111.32
        dy = np.diff(lat) * 110.57
        length = float(np.hypot(dx, dy).sum())
        kind = bt.KIND[tags["railway"]]
        rank = bt.rank_of(kind, tags)
        hit = wid in on_route
        km[(rank, hit)] += length
        n_by[(rank, hit)] += 1

    print("\n=== passenger coverage: track a route relation runs over")
    print("  rank                      ways on a route        km on a route")
    total_km = total_hit = 0.0
    for rank in range(4):
        n_hit, n_miss = n_by[(rank, True)], n_by[(rank, False)]
        k_hit, k_miss = km[(rank, True)], km[(rank, False)]
        total_km += k_hit + k_miss
        total_hit += k_hit
        label = ["0 main", "1 branch", "2 industrial", "3 yard/siding"][rank]
        print(f"  {label:<16} {n_hit:>7}/{n_hit+n_miss:<7} {pct(n_hit, n_hit+n_miss)}"
              f"   {k_hit:>8.0f}/{k_hit+k_miss:<8.0f} {pct(k_hit, k_hit+k_miss)}")
    print(f"  {'all track':<16} {'':>15} {'':>6}   {total_hit:>8.0f}/{total_km:<8.0f} "
          f"{pct(total_hit, total_km)}")


if __name__ == "__main__":
    main()
