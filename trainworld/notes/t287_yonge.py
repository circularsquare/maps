"""T-287: a Yonge subway test line (Union to Eglinton, 12 stations, 20/10/6 trains an hour, 6 cars)
as a `realnet` network JSON in Toronto's pack frame, so the sim's own solver can run the same line
on the real pack and on tier 0's (both packs share the origin).

    ..\\venv\\Scripts\\python.exe trainworld/notes/t287_yonge.py
    cargo run --release --features demand --bin realnet -- --net data/work/t287/yonge.json --pack <pack.json> --out <result.json> --rounds 3
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LON0, LAT0 = -79.3832, 43.6534  # Toronto's origin (build_city.CITIES["tor"])
R = 6371008.8
STATIONS = [("Union", 43.6453, -79.3806), ("King", 43.6490, -79.3779), ("Queen", 43.6524, -79.3793),
            ("Dundas", 43.6561, -79.3803), ("College", 43.6613, -79.3830), ("Wellesley", 43.6655, -79.3838),
            ("Bloor-Yonge", 43.6710, -79.3857), ("Rosedale", 43.6768, -79.3888), ("Summerhill", 43.6822, -79.3909),
            ("St Clair", 43.6880, -79.3934), ("Davisville", 43.6977, -79.3972), ("Eglinton", 43.7057, -79.3983)]


def main():
    xy = []
    for _, la, lo in STATIONS:
        xy += [R * math.radians(lo - LON0) * math.cos(math.radians(LAT0)), R * math.radians(la - LAT0)]
    n = len(STATIONS)
    hop = []
    for k in range(n - 1):
        d = math.hypot(xy[2 * k + 2] - xy[2 * k], xy[2 * k + 3] - xy[2 * k + 1])
        hop.append(30.0 + d / 11.0)  # 30 s dwell and acceleration, 40 km/h between
    times = []
    for run in range(2):
        for _ in range(3):
            for k in range(n):
                if run == 0:
                    times.append(hop[k] if k < n - 1 else 0.0)
                else:
                    times.append(hop[k - 1] if k > 0 else 0.0)
    net = {"demand": {"st_xy": xy, "line_n": [n], "stops": list(range(n)), "times": times, "tph": [20.0, 10.0, 6.0], "cars": [6.0]},
           "stations": [s[0] for s in STATIONS]}
    out = os.path.join(ROOT, "data", "work", "t287", "yonge.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(net, f)
    print(f"{n} stations, {sum(hop) / 60:.1f} min end to end -> {out}")


if __name__ == "__main__":
    main()
