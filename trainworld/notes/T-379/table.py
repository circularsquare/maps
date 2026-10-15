"""T-379: the review's pair table (research/intercity/pair_table.py) on today's world market, after
air: longer trips and over 80 km from bin/market_pack's output, commute legs from the rows, the
plane's share of the trips not by rail, and the rail share of one centre-to-centre trip on the
review's trains with the car as pair_table prices it and the plane as T-378's logit weighs it.

    python -I table.py <trainworld root> <market_pack out.json>
"""
import json, math, sys
import numpy as np
root, src = sys.argv[1], sys.argv[2]
doc = json.load(open(src))
rows = doc["rows"]
place = {k: v["place"] for k, v in rows.items()}
pairs = {(p["a"], p["b"]): p for p in doc["pairs"]}
pack = json.load(open(f"{root}/data/packs/intercity.json", encoding="utf-8"))
R = 6371008.8
lon0, lat0 = pack["frame"]["lon"], pack["frame"]["lat"]
S = np.array([s[:2] for s in pack["sites"]])
park = pack["site_park"]
CENTRES = {"phl": (-75.1636, 39.9526), "bos": (-71.0589, 42.3601), "wdc": (-77.0365, 38.8977), "nhv": (-72.9279, 41.3083), "hfd": (-72.6851, 41.7637),
           "pvd": (-71.4128, 41.824), "bal": (-76.6122, 39.2904)}


def park_at(c):
    lon, lat = CENTRES[c]
    x, y = R * math.radians(lon - lon0) * math.cos(math.radians(lat0)), R * math.radians(lat - lat0)
    return park[int(np.argmin((S[:, 0] - x) ** 2 + (S[:, 1] - y) ** 2))]


def commute(a, b):
    n = sum(t for j, t in rows[a]["commute"] if j == place[b]) + sum(t for j, t in rows[b]["commute"] if j == place[a])
    return 2 * n


NEC = {("nyc", "phl"): 57000, ("nyc", "bos"): 50000, ("nyc", "wdc"): 20000, ("bos", "wdc"): 4900, ("wdc", "bal"): 6300}
REVIEW = [("nyc", "phl", 134, 167), ("nyc", "bos", 302, 396), ("nyc", "wdc", 330, 355), ("phl", "wdc", 196, 224), ("nyc", "nhv", 108, 178),
          ("nhv", "hfd", 56, 112), ("bos", "pvd", 65, 114), ("bos", "wdc", 630, None), ("wdc", "bal", 55, None)]
print("| Pair | NEC FUTURE (a weekday) | Longer trips | Over 80 km | x NEC FUTURE (over 80 km) | Commute legs | Plane share of non-rail | Rail share, one trip |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for a, b, km, rail_min in REVIEW:
    p = pairs[(a, b)]
    nec = NEC.get((a, b))
    wk = nec / ((5 + 2 * 1.58) / 7) if nec else None
    share = ""
    if rail_min:
        drive = 15 + km * 1.25 / 85 * 60 + (km * 1.25 * 0.15 * 3 + park_at(b)) / 2
        rest = 1 / (1 - p["air"]) if p["air"] < 1 else 1e9
        share = f"{100 / (1 + rest * math.exp(-0.025 * (drive - rail_min))):.0f}%"
    necs = f"~{nec:,} ({wk:,.0f})" if nec else ""
    x = f"{p['over_80km'] / wk:.1f}" if nec else ""
    print(f"| {a} - {b} | {necs} | {p['both_ways']:,.0f} | {p['over_80km']:,.0f} | {x} | {commute(a, b):,.0f} | {100 * p['air']:.0f}% | {share} |")
