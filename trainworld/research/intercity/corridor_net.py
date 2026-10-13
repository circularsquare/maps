"""T-276: a test network of five commuter-rail corridors across zone lines, in the format
`zones_day` takes (DemandApi.set_network arrays, metres in the New York equirectangular frame):
the New Haven Line to Grand Central, the Hartford Line, the MBTA Providence Line, the MARC Penn
Line and the Northeast Corridor Philadelphia - Trenton - New York. Stations at the real stops,
run times about today's, trains an hour (peak, midday, night) about today's. Judgement, a check
network only.

    python -I corridor_net.py <out.json>
"""
import json, math, sys

R, LON0, LAT0 = 6371008.8, -73.985, 40.758
LINES = [
    ("New Haven Line", (3, 2, 1), 10, [("New Haven", -72.9267, 41.2968, 0), ("Bridgeport", -73.1886, 41.1786, 22), ("Stamford", -73.5420, 41.0466, 25),
                                       ("New Rochelle", -73.7826, 40.9113, 20), ("Harlem-125 St", -73.9381, 40.8052, 18), ("Grand Central", -73.9772, 40.7527, 10)]),
    ("Hartford Line", (1, 1, 0.5), 4, [("Springfield", -72.5933, 42.1060, 0), ("Hartford", -72.6818, 41.7686, 27), ("Meriden", -72.8019, 41.5381, 20),
                                       ("Wallingford", -72.8240, 41.4560, 8), ("New Haven", -72.9267, 41.2968, 17)]),
    ("Providence Line", (3, 1, 1), 8, [("Providence", -71.4133, 41.8295, 0), ("Attleboro", -71.2850, 41.9398, 15), ("Mansfield", -71.2189, 42.0332, 8),
                                       ("Route 128", -71.1478, 42.2103, 15), ("Back Bay", -71.0756, 42.3474, 12), ("South Station", -71.0552, 42.3522, 4)]),
    ("MARC Penn", (4, 1, 1), 8, [("Baltimore Penn", -76.6155, 39.3074, 0), ("BWI", -76.6945, 39.1922, 15), ("Odenton", -76.7004, 39.0858, 8),
                                 ("Bowie State", -76.7457, 39.0035, 7), ("New Carrollton", -76.8714, 38.9480, 8), ("Union Station", -77.0063, 38.8973, 13)]),
    ("Northeast Corridor", (3, 2, 1), 10, [("30th Street", -75.1817, 39.9566, 0), ("North Philadelphia", -75.1555, 39.9975, 7), ("Cornwells Heights", -74.9468, 40.0716, 18),
                                           ("Trenton", -74.7546, 40.2178, 17), ("Princeton Junction", -74.6238, 40.3166, 12), ("New Brunswick", -74.4459, 40.4964, 13),
                                           ("Newark Penn", -74.1645, 40.7346, 20), ("Penn Station", -73.9935, 40.7506, 17)]),
]


def xy(lon, lat):
    return R * math.radians(lon - LON0) * math.cos(math.radians(LAT0)), R * math.radians(lat - LAT0)


st, key, line_n, stops, times, tph, cars = [], {}, [], [], [], [], []
for name, t, c, stations in LINES:
    for s, lon, lat, _ in stations:
        if s not in key:
            key[s] = len(st)
            st.append(xy(lon, lat))
        stops.append(key[s])
    n = len(stations)
    hop = [stations[k + 1][3] * 60.0 for k in range(n - 1)]
    # per run (0 forward, 1 back) and level (high, medium, low): n numbers, run 0 the hop to the
    # next stop at k, run 1 the hop back from stop k to k - 1 at k
    for run in range(2):
        for _ in range(3):
            times += [hop[k] if k < n - 1 else 0.0 for k in range(n)] if run == 0 else [0.0] + hop
    line_n.append(n)
    tph += list(t)
    cars.append(c)
out = {"lines": [l[0] for l in LINES], "stations": list(key), "demand": {"st_xy": [v for p in st for v in p], "line_n": line_n, "stops": stops, "times": times, "tph": tph, "cars": cars}}
json.dump(out, open(sys.argv[1], "w"), indent=0)
print(len(st), "stations", len(line_n), "lines")
