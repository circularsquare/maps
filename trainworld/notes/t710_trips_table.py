"""T-710: the trips, routes with the same named lines and stations taken as one (a line's service
patterns are separate lines in the real networks)."""
import json
import os
import sys
from collections import defaultdict

OUT = r"C:\Users\anita\projects\maps\artifacts\trainworld\t710"


NETS = {}


def stations_only(legs, city):
    """The legs without their line names: the stations ridden between."""
    if city not in NETS:
        net = json.load(open(rf"C:\Users\anita\projects\maps\trainworld\data\work\realnet\{city}_real.json"))
        NETS[city] = sorted({l.get("name", "?") for l in net["lines"]}, key=len, reverse=True)
    out = []
    for leg in legs.split("; "):
        if leg.startswith("walk to "):
            out.append(leg)
            continue
        for n in NETS[city]:
            if leg.startswith(n + " "):
                leg = leg[len(n) + 1:]
                break
        out.append(leg)
    return "; ".join(out)


def main():
    tags = sys.argv[1].split(",")
    cities = sys.argv[2].split(",") if len(sys.argv) > 2 else ["nyc", "lon", "tor"]
    for c in cities:
        rows = defaultdict(dict)
        for t in tags:
            p = os.path.join(OUT, f"trips_{c}_{t}.json")
            if not os.path.exists(p):
                continue
            for tr in json.load(open(p))["trips"]:
                g = defaultdict(float)
                gs = defaultdict(float)
                cost = {}
                for r in tr["routes"]:
                    g[r["legs"]] += r["share"]
                    gs[stations_only(r["legs"], c)] += r["share"]
                    cost[r["legs"]] = min(cost.get(r["legs"], 1e9), r["cost"])
                top = max(g, key=g.get) if g else ""
                fast = min(cost, key=cost.get) if cost else ""
                rows[tr["trip"]][t] = (tr["rail"], tr["pairs_80"], tr["busiest_pair"], g.get(top, 0.0), top, g.get(fast, 0.0), cost.get(fast, 0.0), cost.get(top, 0.0), max(gs.values()) if gs else 0.0)
        print(f"\n{c}: trip | per setting: rail share, pairs for 80%, busiest pair %, biggest route % (its cost), fastest listed route % (its cost), biggest station path %")
        for trip, d in rows.items():
            cells = []
            for t in tags:
                if t in d:
                    r = d[t]
                    cells.append(f"{t}: {r[0]:.0%} {r[1]} {r[2]:.0%} {r[3]:.0%}({r[7]:.0f}) {r[5]:.0%}({r[6]:.0f}) {r[8]:.0%}")
            print(f"  {trip}: " + " | ".join(cells))
            base = d.get(tags[0]) or next(iter(d.values()))
            last = d.get(tags[-1]) or base
            print(f"      biggest {tags[0]}: {base[4]}\n      biggest {tags[-1]}: {last[4]}")


if __name__ == "__main__":
    main()
