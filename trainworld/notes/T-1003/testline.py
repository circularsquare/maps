"""T-1003: a straight test line through a pack's origin (east-west), as a calib network.

    python testline.py <pack> <length km> <spacing km> <out.json> [angle deg]
Each hop is timed at 35 km/h average including dwell; 20 trains an hour all day; 8 cars.
"""
import json, math, sys

pack, L, s, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
ang = math.radians(float(sys.argv[5])) if len(sys.argv) > 5 else 0.0
n = int(round(L / s)) + 1
xs = [(-L / 2 + i * s) * 1000 for i in range(n)]
ox = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
oy = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
st = [(ox + x * math.cos(ang), oy + x * math.sin(ang)) for x in xs]
hop = s / 35 * 3600
times = ([hop] * (n - 1) + [0.0]) * 3 + ([0.0] + [hop] * (n - 1)) * 3
net = {
    "what": "T-1003 test line", "pack": pack,
    "stations": [{"key": f"t:{i}", "name": f"S{i}", "feed": "t", "x": x, "y": y} for i, (x, y) in enumerate(st)],
    "lines": [{"feed": "t", "name": "test", "operator": "t", "stops": list(range(n)), "tph": [20, 20, 20], "cars": 8}],
    "demand": {"st_xy": [v for p in st for v in p], "line_n": [n], "stops": list(range(n)), "times": times, "tph": [20.0, 20.0, 20.0], "cars": [8.0]},
}
json.dump(net, open(out, "w"))
print(out, n, "stations")
