"""T-276: the intercity review's city-pair table (research/intercity-review.md 2) re-run on a market
pack: all-mode trips a day between places (longer trips from the derived block, commutes between
zones from the commute list), and the rail share of one centre-to-centre trip by hand on the same
trains as the review (notes/T-268.md), with the car as the pack's formula prices it.

    python -I pair_table.py <intercity.json> [--old]

--old prices the car as before T-276 (time only).
"""
import json, math, struct, sys
import numpy as np

pack = json.load(open(sys.argv[1], encoding="utf-8"))
old = "--old" in sys.argv
names = pack["markets"]
m = len(names)
corr = [struct.unpack("<d", struct.pack("<Q", v))[0] for v in pack["derived"]["corridor_trips"]]
idx, k = {}, 0
for a in range(m):
    for b in range(a, m):
        idx[(a, b)] = k
        k += 1
P = {n: i for i, n in enumerate(names)}
P["Trenton"] = P.get("Trenton", 2)
# commutes between zones by place pair (both legs: 2 a commuter)
sites = pack["sites"]
comm = np.zeros((m, m))
for h, w, t in zip(pack.get("commute_home", []), pack.get("commute_work", []), pack.get("commute_trips", [])):
    a, b = sites[h][4], sites[w][4]
    comm[min(a, b), max(a, b)] += 2 * t
print(f"day: {struct.unpack('<d', struct.pack('<Q', pack['derived']['daily']))[0]:,.0f} longer trips, {comm.sum():,.0f} commute legs between zones")

# the review's trains and rail perceived minutes (notes/T-268.md, by hand)
REVIEW = [("New York", "Philadelphia", 134, 167, 16.91), ("New York", "Boston", 302, 396, 36.28), ("New York", "Washington, DC", 330, 355, 39.41),
          ("Philadelphia", "Washington, DC", 196, 224, 24.04), ("New York", "New Haven", 108, 178, 13.94), ("New Haven", "Hartford", 56, 112, 7.97),
          ("Boston", "Providence", 65, 114, 8.99), ("New York", "Trenton", 88, 131, 11.56), ("Washington, DC", "Baltimore", 55, None, None), ("Washington, DC", "Boston", 630, None, None)]
# parking at each centre's 600 m site (the local rule), for the destination end
CENTRES = {"New York": (-73.985, 40.758), "Philadelphia": (-75.1636, 39.9526), "Boston": (-71.0589, 42.3601), "Washington, DC": (-77.0365, 38.8977),
           "New Haven": (-72.9279, 41.3083), "Hartford": (-72.6851, 41.7637), "Providence": (-71.4128, 41.824), "Trenton": (-74.7597, 40.2206), "Baltimore": (-76.6122, 39.2904)}
R = 6371008.8
lon0, lat0 = pack["frame"]["lon"], pack["frame"]["lat"]
park = pack.get("site_park", [])
S = np.array([s[:2] for s in sites])


def park_at(name):
    if not park:
        return 0.0
    lon, lat = CENTRES[name]
    x, y = R * math.radians(lon - lon0) * math.cos(math.radians(lat0)), R * math.radians(lat - lat0)
    return park[int(np.argmin((S[:, 0] - x) ** 2 + (S[:, 1] - y) ** 2))]


print("| Pair | km | longer trips | commute legs | all | rail share, one trip |")
print("|---|---:|---:|---:|---:|---:|")
for a, b, km, rail_min, fare in REVIEW:
    ia, ib = P[a], P[b]
    key = (min(ia, ib), max(ia, ib))
    longer = corr[idx[key]]
    c = comm[key]
    share = ""
    if rail_min:
        drive = 15 + km * 1.25 / 85 * 60
        if not old:
            drive += (km * 1.25 * 0.15 * 3 + park_at(b)) / 2
        r = 1 / (1 + math.exp(-0.025 * (drive - rail_min)))
        share = f"{100 * r:.0f}%"
    print(f"| {a} - {b} | {km} | {longer:,.0f} | {c:,.0f} | {longer + c:,.0f} | {share} |")
