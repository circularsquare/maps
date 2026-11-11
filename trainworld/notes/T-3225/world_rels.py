"""T-3225: what noritetsu's local OSM extracts hold for an automatic real-network build, world wide:
route relations by kind, how many carry colour, interval and stop members, and subway networks."""
import glob
import os
import pickle
from collections import Counter, defaultdict

base = r"C:\Users\anita\projects\maps\noritetsu\data\proc"
tot = Counter()
nets = defaultdict(set)  # kind -> {(cc, network or operator)}
rows = []
for d in sorted(glob.glob(os.path.join(base, "*"))):
    cc = os.path.basename(d)
    p = os.path.join(d, "rels.pkl")
    if cc == "fixture" or not os.path.exists(p):
        continue
    rels = pickle.load(open(p, "rb"))
    c = Counter()
    for rid, (t, ms) in rels.items():
        if t.get("type") != "route":
            continue
        k = t.get("route") or "?"
        c[k] += 1
        tot[k] += 1
        stops = sum(1 for ty, _, role in ms if ty == "n" and role.startswith("stop"))
        plats = sum(1 for _, _, role in ms if role.startswith("platform"))
        ways = sum(1 for ty, _, role in ms if ty == "w" and role in ("", "forward", "backward"))
        tot[k + " with stops"] += stops >= 2
        tot[k + " with platforms only"] += stops < 2 and plats >= 2
        tot[k + " with ways"] += ways > 0
        tot[k + " colour"] += bool(t.get("colour") or t.get("color"))
        tot[k + " interval"] += bool(t.get("interval"))
        tot[k + " ptv2"] += t.get("public_transport:version") == "2"
        if k in ("subway", "light_rail", "monorail"):
            nets[k].add((cc, t.get("network") or t.get("operator") or "?"))
    rows.append((cc, c.get("subway", 0), c.get("light_rail", 0), c.get("train", 0), c.get("tram", 0), c.get("monorail", 0)))

print("region subway light_rail train tram monorail (route relations, one per direction)")
for r in sorted(rows, key=lambda r: -r[1])[:40]:
    print(*r)
print()
for k in ("subway", "light_rail", "train", "tram", "monorail"):
    n = tot[k]
    print(f"{k}: {n} relations; stops>=2 {tot[k + ' with stops'] / max(n, 1):.0%}, platforms only {tot[k + ' with platforms only'] / max(n, 1):.0%}, "
          f"ways {tot[k + ' with ways'] / max(n, 1):.0%}, colour {tot[k + ' colour'] / max(n, 1):.0%}, interval {tot[k + ' interval'] / max(n, 1):.0%}, PTv2 {tot[k + ' ptv2'] / max(n, 1):.0%}")
for k, s in nets.items():
    print(f"distinct (region, network) with {k} relations: {len(s)}")
print("regions with any subway relation:", sum(1 for r in rows if r[1]))
