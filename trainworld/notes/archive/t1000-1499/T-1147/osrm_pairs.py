"""T-279: routed drives (OSRM's public demo, slowly) between market place centres against the
market's car (1.25 x the great circle at 85 km/h, plus 15 minutes).
python -I osrm_pairs.py <market-rows.json> <out dir>"""
import json
import math
import os
import sys
import time
import urllib.request

rows = json.load(open(sys.argv[1], encoding="utf-8"))
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
cache_f = os.path.join(OUT, "osrm_cache.json")
cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
R = 6371.0088

C = {
    "Dallas": (32.78, -96.80), "Houston": (29.76, -95.37), "Chicago": (41.88, -87.63), "Detroit": (42.33, -83.05),
    "Berlin": (52.52, 13.40), "Hamburg": (53.55, 9.99), "Paris": (48.86, 2.35), "Lille": (50.63, 3.06),
    "Warsaw": (52.23, 21.01), "Poznan": (52.41, 16.93), "Madrid": (40.42, -3.70), "Valencia": (39.47, -0.38),
    "Milan": (45.46, 9.19), "Bologna": (44.49, 11.34), "Beijing": (39.90, 116.40), "Tianjin": (39.13, 117.20),
    "Shanghai": (31.23, 121.47), "Nanjing": (32.06, 118.80), "Amsterdam": (52.37, 4.90), "Brussels": (50.85, 4.35),
    "Zurich": (47.37, 8.54), "Munich": (48.14, 11.58), "Innsbruck": (47.27, 11.39), "Vienna": (48.21, 16.37),
    "Graz": (47.07, 15.44), "Tokyo": (35.68, 139.69), "Nagano": (36.65, 138.19), "Bilbao": (43.26, -2.93),
    "Athens": (37.98, 23.73), "Thessaloniki": (40.64, 22.94), "Taipei": (25.03, 121.56), "Hualien": (23.99, 121.60),
    "Chengdu": (30.66, 104.07), "Chongqing": (29.56, 106.55), "Xian": (34.34, 108.94), "Kunming": (25.04, 102.71),
    "Guiyang": (26.65, 106.63), "Sofia": (42.70, 23.32), "Lyon": (45.76, 4.84), "Turin": (45.07, 7.69),
    "Seattle": (47.61, -122.33), "Portland": (45.52, -122.68), "Phoenix": (33.45, -112.07), "Los Angeles": (34.05, -118.24),
    "New York": (40.76, -73.95), "Boston": (42.36, -71.06), "San Francisco": (37.77, -122.42), "Copenhagen": (55.68, 12.57),
    "Gothenburg": (57.71, 11.97), "Oslo": (59.91, 10.75), "Stockholm": (59.33, 18.07), "Helsinki": (60.17, 24.94),
    "Tallinn": (59.44, 24.75), "London": (51.51, -0.13), "Dublin": (53.35, -6.26), "Sapporo": (43.06, 141.35),
    "Hong Kong": (22.32, 114.17), "Guangzhou": (23.13, 113.26), "Heraklion": (35.34, 25.13), "Osaka": (34.69, 135.50),
    "Fukuoka": (33.59, 130.40), "Aarhus": (56.16, 10.20), "Barcelona": (41.39, 2.17), "Palma": (39.57, 2.65),
    "Rome": (41.90, 12.50), "Naples": (40.85, 14.27), "Marseille": (43.30, 5.37), "Denver": (39.74, -104.99),
    "Salt Lake City": (40.76, -111.89), "Atlanta": (33.75, -84.39), "Washington": (38.90, -77.04), "Glasgow": (55.86, -4.25),
    "Manchester": (53.48, -2.24), "Nagoya": (35.18, 136.91), "Hiroshima": (34.39, 132.46), "Matsuyama": (33.84, 132.77),
    "Kaohsiung": (22.63, 120.30), "Lisbon": (38.72, -9.14), "Porto": (41.15, -8.61), "Seoul": (37.57, 126.98),
}
PAIRS = [
    ("flat", "Dallas", "Houston"), ("flat", "Chicago", "Detroit"), ("flat", "Berlin", "Hamburg"), ("flat", "Paris", "Lille"),
    ("flat", "Warsaw", "Poznan"), ("flat", "Madrid", "Valencia"), ("flat", "Milan", "Bologna"), ("flat", "Beijing", "Tianjin"),
    ("flat", "Shanghai", "Nanjing"), ("flat", "Amsterdam", "Brussels"), ("flat", "Rome", "Naples"), ("flat", "Lisbon", "Porto"),
    ("flat", "Taipei", "Kaohsiung"), ("flat", "Atlanta", "Washington"),
    ("mountain", "Zurich", "Milan"), ("mountain", "Munich", "Innsbruck"), ("mountain", "Vienna", "Graz"), ("mountain", "Tokyo", "Nagano"),
    ("mountain", "Madrid", "Bilbao"), ("mountain", "Athens", "Thessaloniki"), ("mountain", "Taipei", "Hualien"),
    ("mountain", "Chengdu", "Chongqing"), ("mountain", "Xian", "Chengdu"), ("mountain", "Kunming", "Guiyang"),
    ("mountain", "Sofia", "Thessaloniki"), ("mountain", "Lyon", "Turin"), ("mountain", "Denver", "Salt Lake City"),
    ("mountain", "Tokyo", "Nagoya"),
    ("water", "New York", "Boston"), ("water", "San Francisco", "Los Angeles"), ("water", "Seattle", "Portland"),
    ("water", "Copenhagen", "Gothenburg"), ("water", "Copenhagen", "Aarhus"), ("water", "Helsinki", "Tallinn"),
    ("water", "London", "Paris"), ("water", "London", "Dublin"), ("water", "Tokyo", "Sapporo"), ("water", "Hong Kong", "Guangzhou"),
    ("water", "Athens", "Heraklion"), ("water", "Osaka", "Fukuoka"), ("water", "Barcelona", "Palma"), ("water", "Hiroshima", "Matsuyama"),
    ("sparse", "Phoenix", "Los Angeles"), ("sparse", "Oslo", "Stockholm"), ("sparse", "Glasgow", "Manchester"),
    ("long", "New York", "Chicago"), ("long", "Paris", "Marseille"), ("long", "Madrid", "Barcelona"), ("long", "Beijing", "Shanghai"),
    ("long", "Munich", "Berlin"),
]


def gc(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * math.asin(min(1, math.sqrt(h)))


places = [(k, (r["lat"], r["lon"]), r["country"]) for k, r in rows.items()]


def place_of(name):
    p = C[name]
    best = min(places, key=lambda x: gc(p, x[1]))
    d = gc(p, best[1])
    return (best, d) if d < 40 else (None, d)


def osrm(a, b):
    key = f"{a[1]:.4f},{a[0]:.4f};{b[1]:.4f},{b[0]:.4f}"
    if key in cache:
        return cache[key]
    url = f"https://router.project-osrm.org/route/v1/driving/{key}?overview=false"
    req = urllib.request.Request(url, headers={"User-Agent": "trainworld-research/1.0 (road factor study)"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as f:
                j = json.load(f)
            break
        except Exception as e:  # noqa: BLE001
            print("retry", key, e, flush=True)
            time.sleep(10)
    else:
        return None
    rt = j["routes"][0] if j.get("routes") else None
    cache[key] = None if rt is None else {"m": rt["distance"], "s": rt["duration"]}
    json.dump(cache, open(cache_f, "w"))
    time.sleep(2.0)
    return cache[key]


out = []
for kind, a, b in PAIRS:
    (pa, da), (pb, db) = place_of(a), place_of(b)
    if not pa or not pb:
        print("no place near", a if not pa else b)
        continue
    km = gc(pa[1], pb[1])
    r = osrm(pa[1], pb[1])
    if not r:
        print("no route", a, b)
        continue
    road_km, road_min = r["m"] / 1000, r["s"] / 60
    model_min = 15 + km * 1.25 / 85 * 60
    out.append({"kind": kind, "a": a, "b": b, "ida": pa[0], "idb": pb[0], "gc_km": km, "road_km": road_km, "factor": road_km / km,
                "osrm_min": road_min, "osrm_kmh": road_km / road_min * 60, "model_min": model_min, "model_over_osrm": model_min / road_min})
    o = out[-1]
    print(f"{kind:8s} {a}-{b} ({pa[0]}-{pb[0]}): {km:.0f} km, road {road_km:.0f} km (x{o['factor']:.2f}), OSRM {road_min:.0f} min ({o['osrm_kmh']:.0f} km/h), model {model_min:.0f} min ({o['model_over_osrm']:.2f}x)", flush=True)
json.dump(out, open(os.path.join(OUT, "osrm_pairs.json"), "w"), indent=1)
