"""T-007: New York's real rail network as a demand test set, and the measured numbers to check it.

    python pipeline/realnet.py net        # GTFS -> data/work/realnet/nyc_real.json
    python pipeline/realnet.py targets    # measured ridership and mode share -> targets.json
    python pipeline/realnet.py counties   # county of every pack cell -> cell_county.u8
    (the sim crate's `realnet` bin solves a day on a network: notes/T-007.md)
    python pipeline/realnet.py compare <result.json> [--tag x] [--net n.json] [--no-plot]
    python pipeline/realnet.py single     # the batch-3 single Manhattan line as a test network
    python pipeline/realnet.py geom       # track geometry from GTFS shapes, for the `realsave` bin
    python pipeline/realnet.py gamesave   # the `realsave` output as a game file (T-007-real-nyc.save)

The network is built from the operators' GTFS (weekday schedule): New York City Subway with the
Staten Island Railway, PATH, LIRR, Metro-North and NJ Transit rail. Stations: the MTA's station
complexes for the subway (one station per complex, so in-complex transfers are free of walking,
as the MTA's own counts treat them); one station per stop elsewhere, stops of different
operators within 150 m merged (Penn Station, Hoboken, Newark Penn). Stations outside the city
boundary are dropped.

Lines: per route, the weekday daytime stopping patterns. A pattern with at least `min_share` of
its route's trips (and `min_trips` a day) is its own line; a pattern that is a contiguous piece of
a kept one (a short turn) adds its trips to that line; anything else goes to the kept pattern it
overlaps most. Trains per hour per demand level: high = the mean of 7-9 and 17-19, medium =
10-16, low = 0-6, counted at each trip's stop nearest Midtown, both directions halved. Run times:
per consecutive station pair, the median of every trip of the route that runs it in the level's
hours (departure to departure, so a stop's dwell is in the hop that leaves it).

Raw GTFS: the subway feed is maps/riders/nycriders/data/gtfs_subway (MTA, Feb 2026); the others in
data/raw/gtfs/ (downloaded once, see FEEDS).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import json
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS = os.path.dirname(ROOT)
RAW = os.path.join(ROOT, "data", "raw")
WORK = os.path.join(ROOT, "data", "work", "realnet")
PACKS = os.path.join(ROOT, "data", "packs")
R_EARTH = 6371008.8

NYCRIDERS = os.path.join(MAPS, "riders", "nycriders")
FEEDS = {
    # name: (folder, url it came from, min_share, min_trips, merge group)
    "subway": (os.path.join(NYCRIDERS, "data", "gtfs_subway"), "http://web.mta.info/developers/data/nyct/subway/google_transit.zip", 0.15, 20),
    "path": (os.path.join(RAW, "gtfs", "path-nj-us"), "http://data.trilliumtransit.com/gtfs/path-nj-us/path-nj-us.zip", 0.15, 20),
    "lirr": (os.path.join(RAW, "gtfs", "gtfslirr"), "https://rrgtfsfeeds.s3.amazonaws.com/gtfslirr.zip", 0.05, 4),
    "mnr": (os.path.join(RAW, "gtfs", "gtfsmnr"), "https://rrgtfsfeeds.s3.amazonaws.com/gtfsmnr.zip", 0.05, 4),
    "njt": (os.path.join(RAW, "gtfs", "njt_rail"), "https://www.njtransit.com/rail_data.zip", 0.05, 4),
}
MTA_STATIONS = os.path.join(MAPS, "data", "nystreets", "MTA_Subway_Stations.csv")
OD_SUBWAY = os.path.join(NYCRIDERS, "data", "od_wednesday_oct.csv")
NYCRIDERS_STATS = os.path.join(NYCRIDERS, "stats.json")

# Train length in metres per route (cars x car length), for the game's 20 m cars. Subway: A
# division 51 ft cars (10, the 7 11), B division 60 ft (600 ft trains; J, Z, M, L 480 ft; G 300 ft),
# shuttles short; SIR 4 x 75 ft. PATH 7 x 51 ft. Commuter rail: 10 x 85 ft as a typical peak train.
TRAIN_M = {
    "subway": {"1": 155, "2": 155, "3": 155, "4": 155, "5": 155, "5X": 155, "6": 155, "6X": 155,
               "7": 171, "7X": 171, "GS": 93, "FS": 37, "H": 73, "G": 91, "J": 146, "Z": 146,
               "M": 146, "L": 146, "SI": 91, "_": 183},
    "path": {"_": 109},
    # NJT's feed also has its light rail: Hudson-Bergen (route 4) 2-car, Newark (13) 2-car trains
    "lirr": {"_": 260}, "mnr": {"_": 260}, "njt": {"_": 260, "4": 60, "13": 56, "17": 60},
}
CAR_M = 20.0
LEVEL_WINDOWS = {0: [(7, 9), (17, 19)], 1: [(10, 16)], 2: [(0, 6)]}
DAY = (6, 20)
MERGE_M = 150.0


def origin():
    h = json.load(open(os.path.join(PACKS, "nyc.json")))
    return h["origin"]["lon"], h["origin"]["lat"]


def to_xy(lon, lat, o):
    lon0, lat0 = o
    x = R_EARTH * np.radians(np.asarray(lon, float) - lon0) * np.cos(np.radians(lat0))
    y = R_EARTH * np.radians(np.asarray(lat, float) - lat0)
    return x, y


def secs(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def boundary():
    from shapely.geometry import shape
    g = json.load(open(os.path.join(PACKS, "nyc.boundary.geojson")))
    geoms = [shape(f["geometry"]) for f in g["features"]] if "features" in g else [shape(g)]
    from shapely import union_all
    return union_all(geoms).buffer(0.01)  # ~1 km in degrees


def read(folder, name, **kw):
    p = os.path.join(folder, name)
    return pd.read_csv(p, dtype=str, keep_default_na=False, **kw) if os.path.exists(p) else None


def service_ids(folder):
    """Service ids running on the weekday (a Wednesday) with the most trips in the feed."""
    cal = read(folder, "calendar.txt")
    cd = read(folder, "calendar_dates.txt")
    trips = read(folder, "trips.txt", usecols=["service_id"])
    per_service = trips["service_id"].value_counts().to_dict()
    dates = set()
    if cal is not None and len(cal):
        for _, r in cal.iterrows():
            d0, d1 = pd.Timestamp(r["start_date"]), pd.Timestamp(r["end_date"])
            dates.update(pd.date_range(d0, min(d1, d0 + pd.Timedelta(days=120)), freq="W-WED"))
    if cd is not None and len(cd):
        dates.update(pd.to_datetime(cd["date"].unique()))
    dates = sorted(d for d in dates if d.dayofweek == 2)
    best, best_n = None, -1
    for d in dates:
        s = set()
        if cal is not None and len(cal):
            ds = d.strftime("%Y%m%d")
            m = (cal["wednesday"] == "1") & (cal["start_date"] <= ds) & (cal["end_date"] >= ds)
            s |= set(cal.loc[m, "service_id"])
        if cd is not None and len(cd):
            ds = d.strftime("%Y%m%d")
            x = cd[cd["date"] == ds]
            s |= set(x.loc[x["exception_type"] == "1", "service_id"])
            s -= set(x.loc[x["exception_type"] == "2", "service_id"])
        n = sum(per_service.get(v, 0) for v in s)
        if n > best_n:
            best, best_n = (d, s), n
    return best


def load_feed(name, o, inside):
    """Stations and trips of one feed: (stations {key: (name, lon, lat)}, trips [(route, dir,
    [keys], [dep secs])])."""
    folder = FEEDS[name][0]
    date, sids = service_ids(folder)
    stops = read(folder, "stops.txt")
    parent = {}
    if "parent_station" in stops:
        parent = {r.stop_id: r.parent_station for r in stops.itertuples() if r.parent_station}
    pos = {r.stop_id: (r.stop_name, float(r.stop_lon), float(r.stop_lat)) for r in stops.itertuples()}
    if name == "subway":
        mta = pd.read_csv(MTA_STATIONS, dtype=str)
        cx = dict(zip(mta["GTFS Stop ID"], mta["Complex ID"]))
        cname = mta.groupby("Complex ID")["Stop Name"].agg(lambda s: "/".join(dict.fromkeys(s)))
        cpos = mta.assign(lon=mta["GTFS Longitude"].astype(float), lat=mta["GTFS Latitude"].astype(float)).groupby("Complex ID")[["lon", "lat"]].mean()

        def key(sid):
            p = parent.get(sid, sid)
            c = cx.get(p)
            return ("mta:" + c) if c else ("subway:" + p)
        stations = {}
        for sid in pos:
            k = key(sid)
            if k.startswith("mta:"):
                c = k[4:]
                stations[k] = (cname[c], float(cpos.loc[c, "lon"]), float(cpos.loc[c, "lat"]))
            else:
                stations[k] = pos[parent.get(sid, sid)]
    else:
        def key(sid):
            return name + ":" + parent.get(sid, sid)
        stations = {key(s): pos[parent.get(s, s)] for s in pos}
    keep = {k for k, (_, lon, lat) in stations.items() if inside(lon, lat)}
    trips = read(folder, "trips.txt")
    trips = trips[trips["service_id"].isin(sids)]
    tinfo = {r.trip_id: (r.route_id, r.direction_id) for r in trips.itertuples()}
    st = read(folder, "stop_times.txt", usecols=["trip_id", "stop_id", "arrival_time", "departure_time", "stop_sequence"])
    st = st[st["trip_id"].isin(tinfo)]
    st = st.assign(seq=st["stop_sequence"].astype(int)).sort_values(["trip_id", "seq"])
    out = []
    for tid, g in st.groupby("trip_id", sort=False):
        ks, ts = [], []
        for sid, dep, arr in zip(g["stop_id"], g["departure_time"], g["arrival_time"]):
            k = key(sid)
            if k not in keep:
                continue
            t = secs(dep or arr)
            if ks and ks[-1] == k:
                ts[-1] = t
                continue
            ks.append(k)
            ts.append(t)
        if len(ks) >= 2:
            r, d = tinfo[tid]
            out.append((r, d, ks, ts))
    print(f"{name}: {date.date()}, {len(sids)} services, {len(out)} trips, {len(keep)} of {len(stations)} stations inside")
    return {k: v for k, v in stations.items() if k in keep}, out


def is_piece(small, big):
    """`small` is a contiguous run of `big`."""
    n = len(small)
    return any(tuple(big[i:i + n]) == tuple(small) for i in range(len(big) - n + 1))


def build_lines(name, trips, xy):
    _, _, min_share, min_trips = FEEDS[name]
    lines = []
    by_route = defaultdict(list)
    for t in trips:
        by_route[t[0]].append(t)
    for route, ts in sorted(by_route.items()):
        # canonical orientation: direction 0 as given, direction 1 reversed
        canon = []
        for r, d, ks, tt in ts:
            c = tuple(ks) if d != "1" else tuple(reversed(ks))
            canon.append(c)
        mid_t = []
        for (_, _, ks, tt) in ts:
            # time at the stop nearest Midtown (the pack origin)
            dist = [xy[k][0] ** 2 + xy[k][1] ** 2 for k in ks]
            mid_t.append(tt[int(np.argmin(dist))] / 3600.0 % 24)
        day = [DAY[0] <= h < DAY[1] for h in mid_t]
        cnt = Counter(c for c, dd in zip(canon, day) if dd)
        if not cnt:
            cnt = Counter(canon)
        total = sum(cnt.values())
        kept = []
        for c, n in cnt.most_common():
            if n < max(min_share * total, min_trips) or any(is_piece(c, k) for k in kept):
                continue
            # the same ends and nearly the same stops as a busier pattern: a variant of it (a stop
            # skipped one way for works, say), not a service of its own; splitting it off would
            # halve the frequency riders see
            if any(c[0] == k[0] and c[-1] == k[-1] and len(set(c) & set(k)) / len(set(c) | set(k)) >= 0.8 for k in kept):
                continue
            kept.append(c)
        if not kept:
            kept = [cnt.most_common(1)[0][0]]
        # every trip goes to a kept pattern: one it is a contiguous piece of (most stops first),
        # else the one it shares most stops with
        assign = {}
        for c in set(canon):
            hosts = [k for k in kept if is_piece(c, k)]
            if hosts:
                assign[c] = max(hosts, key=lambda k: cnt.get(k, 0))
                continue
            ov = [(len(set(c) & set(k)) / len(c), cnt.get(k, 0), k) for k in kept]
            best = max(ov)
            assign[c] = best[2] if best[0] >= 0.5 else None
        # hop times by station pair and level, over every trip of the route
        hop = defaultdict(lambda: defaultdict(list))
        for (_, _, ks, tt), h in zip(ts, mid_t):
            levs = [lv for lv, ws in LEVEL_WINDOWS.items() if any(a <= h < b for a, b in ws)]
            for i in range(len(ks) - 1):
                dt = tt[i + 1] - tt[i]
                if dt <= 0:
                    continue
                for lv in levs:
                    hop[(ks[i], ks[i + 1])][lv].append(dt)
                hop[(ks[i], ks[i + 1])]["all"].append(dt)
        for k in kept:
            n_lev = {lv: 0.0 for lv in LEVEL_WINDOWS}
            for c, h in zip(canon, mid_t):
                if assign.get(c) != k:
                    continue
                for lv, ws in LEVEL_WINDOWS.items():
                    if any(a <= h < b for a, b in ws):
                        n_lev[lv] += 1
            hours = {lv: sum(b - a for a, b in ws) for lv, ws in LEVEL_WINDOWS.items()}
            tph = [round(n_lev[lv] / hours[lv] / 2.0, 2) for lv in range(3)]
            if max(tph) == 0:
                continue

            def hop_t(a, b, lv):
                for src in ((a, b), (b, a)):
                    h = hop.get(src)
                    if not h:
                        continue
                    for want in (lv, 1, 0, "all"):
                        if h.get(want):
                            return float(np.median(h[want]))
                return None
            n = len(k)
            t0 = [[0.0] * n for _ in range(3)]
            t1 = [[0.0] * n for _ in range(3)]
            ok = True
            for lv in range(3):
                for i in range(n - 1):
                    f, b = hop_t(k[i], k[i + 1], lv), hop_t(k[i + 1], k[i], lv)
                    if f is None or b is None:
                        ok = False
                    t0[lv][i] = f or b or 0.0
                    t1[lv][i + 1] = b or f or 0.0
            if not ok:
                print(f"  {name} {route}: a hop has no time in one direction; mirrored")
            m = TRAIN_M[name]
            cars = max(1, round(m.get(route, m["_"]) / CAR_M))
            lines.append({"feed": name, "route": route, "stops": list(k), "tph": tph, "t0": t0, "t1": t1, "cars": cars,
                          "trips_day": sum(1 for c in canon if assign.get(c) == k)})
    return lines


def route_names(name):
    r = read(FEEDS[name][0], "routes.txt")
    col = "route_short_name" if "route_short_name" in r and (r["route_short_name"] != "").all() else "route_long_name"
    return dict(zip(r["route_id"], r[col]))


def cmd_net():
    os.makedirs(WORK, exist_ok=True)
    o = origin()
    from shapely import prepared
    from shapely.geometry import Point
    bnd = prepared.prep(boundary())

    def inside(lon, lat):
        return bnd.contains(Point(lon, lat))
    stations, lines = {}, []
    for name in FEEDS:
        st, trips = load_feed(name, o, inside)
        for k, (nm, lon, lat) in st.items():
            x, y = to_xy(lon, lat, o)
            stations[k] = {"key": k, "name": nm, "feed": name, "lon": lon, "lat": lat, "x": float(x), "y": float(y)}
        xy = {k: (v["x"], v["y"]) for k, v in stations.items()}
        ls = build_lines(name, trips, xy)
        rn = route_names(name)
        for l in ls:
            l["name"] = rn.get(l["route"], l["route"])
        print(f"  {name}: {len(ls)} lines from {len(set(l['route'] for l in ls))} routes")
        lines += ls
    # merge stations of different non-subway feeds within MERGE_M (one interchange)
    keys = [k for k in stations if not k.startswith("mta:") and not k.startswith("subway:")]
    alias = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            sa, sb = stations[a], stations[b]
            if sa["feed"] != sb["feed"] and (sa["x"] - sb["x"]) ** 2 + (sa["y"] - sb["y"]) ** 2 < MERGE_M ** 2:
                ra, rb = alias.get(a, a), alias.get(b, b)
                if ra != rb:
                    alias[rb] = ra
                    for k, v in list(alias.items()):
                        if v == rb:
                            alias[k] = ra
    for l in lines:
        l["stops"] = [alias.get(k, k) for k in l["stops"]]
    used = sorted({k for l in lines for k in l["stops"]})
    for b, a in alias.items():
        if b in stations and a in stations:
            stations[a]["name"] = stations[a]["name"] + " / " + stations[b]["name"]
    idx = {k: i for i, k in enumerate(used)}
    st_list = [stations[k] for k in used]
    for l in lines:
        l["stops"] = [idx[k] for k in l["stops"]]
    # drop lines that collapsed (merged stops next to each other)
    good = []
    for l in lines:
        s = l["stops"]
        if any(s[i] == s[i + 1] for i in range(len(s) - 1)) or len(s) < 2:
            print(f"  dropped {l['feed']} {l['name']}: repeated stop after merging")
            continue
        good.append(l)
    lines = good
    # the arrays DemandApi::set_network takes
    st_xy, line_n, stops, times, tph, cars = [], [], [], [], [], []
    for s in st_list:
        st_xy += [s["x"], s["y"]]
    for l in lines:
        n = len(l["stops"])
        line_n.append(n)
        stops += l["stops"]
        for run in (l["t0"], l["t1"]):
            for lv in range(3):
                times += run[lv]
        tph += l["tph"]
        cars.append(l["cars"])
    out = {"what": "New York's rail network from GTFS (T-007)", "feeds": {k: v[1] for k, v in FEEDS.items()},
           "stations": st_list, "lines": lines,
           "demand": {"st_xy": st_xy, "line_n": line_n, "stops": stops, "times": times, "tph": tph, "cars": cars}}
    p = os.path.join(WORK, "nyc_real.json")
    json.dump(out, open(p, "w"))
    by = Counter(l["feed"] for l in lines)
    print(f"{len(st_list)} stations, {len(lines)} lines {dict(by)} -> {p}")


# ---------------------------------------------------------------- measured targets

ACS_B08301 = "https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData/acsdt5y2023-b08301.dat"
# B08301 (2019+): 001 workers 16+, 002 car/truck/van, 010 public transport (no taxi), 011 bus,
# 012 subway or elevated rail, 013 long-distance train or commuter rail, 014 light rail,
# streetcar or trolley, 015 ferry, 016 taxi, 017 motorcycle, 018 bicycle, 019 walked, 020 other,
# 021 worked from home
B08301_COLS = [f"B08301_E{i:03d}" for i in range(1, 22)]
COUNTIES = {
    "36005": "Bronx", "36047": "Kings", "36061": "New York", "36081": "Queens", "36085": "Richmond",
    "36059": "Nassau", "36103": "Suffolk", "36119": "Westchester", "36087": "Rockland", "36079": "Putnam",
    "34003": "Bergen", "34013": "Essex", "34017": "Hudson", "34019": "Hunterdon", "34023": "Middlesex",
    "34025": "Monmouth", "34027": "Morris", "34029": "Ocean", "34031": "Passaic", "34035": "Somerset",
    "34037": "Sussex", "34039": "Union", "09001": "Fairfield CT", "36027": "Dutchess", "36071": "Orange NY",
}
COUNTY_ORDER = sorted(COUNTIES)
# Measured weekday totals the subway OD cannot give (October 2024, a weekday): MTA Daily Ridership
# (data.ny.gov vxuj-8kew, the Wednesdays of October 2024, fetched by `targets`); PATH 211,525 on an
# average October 2024 weekday (Port Authority, via Progressive Railroading, 2024-11); NJ Transit
# rail has no recent published weekday figure found (FY2018: 311,250 boardings with 28,400
# rail-to-rail transfers; 2024 at roughly 80% of that is ~250k), so it is a rough number only.
PATH_WEEKDAY = 211525
NJT_WEEKDAY_ROUGH = 250000
MTA_DAILY = "https://data.ny.gov/resource/vxuj-8kew.csv?$where=date%20between%20'2024-10-01T00:00:00'%20and%20'2024-10-31T00:00:00'&$limit=100"
PERIOD_HOURS = [(6, 10), (10, 16), (16, 20), (20, 24), (0, 6)]


def period_of_hour(h):
    for q, (a, b) in enumerate(PERIOD_HOURS):
        if a <= h < b:
            return q


def cmd_counties():
    """One byte per pack cell: its county's index in COUNTY_ORDER (255 = none)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import read_pack
    import shapefile
    from shapely import contains_xy, make_valid
    from shapely.geometry import shape
    header, a = read_pack.load_pack("nyc")
    lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
    x, y = a["x_m"].astype(float), a["y_m"].astype(float)
    lon = lon0 + np.degrees(x / (R_EARTH * np.cos(np.radians(lat0))))
    lat = lat0 + np.degrees(y / R_EARTH)
    r = shapefile.Reader(os.path.join(RAW, "cb_2020_us_county_500k.zip"))
    gi = [f[0] for f in r.fields[1:]].index("GEOID")
    out = np.full(len(x), 255, np.uint8)
    polys = {}
    for sr in r.iterShapeRecords():
        g = sr.record[gi]
        if g in COUNTIES:
            polys[g] = make_valid(shape(sr.shape.__geo_interface__))
    for i, g in enumerate(COUNTY_ORDER):
        m = contains_xy(polys[g], lon, lat)
        out[m & (out == 255)] = i
    # cells just off the generalised coast: the nearest county
    miss = np.where(out == 255)[0]
    if len(miss):
        from shapely.geometry import Point
        for c in miss:
            p = Point(lon[c], lat[c])
            out[c] = min(range(len(COUNTY_ORDER)), key=lambda i: polys[COUNTY_ORDER[i]].distance(p))
    p = os.path.join(WORK, "cell_county.u8")
    out.tofile(p)
    print(f"{len(x)} cells, {len(miss)} placed by nearest county -> {p}")


def acs_counties():
    """B08301 by our 25 counties (2020 boundaries). Connecticut's 2023 rows are planning regions,
    so Fairfield is the sum of its 2020 tracts (tract codes are unchanged; read from the 2020
    block file)."""
    cache = os.path.join(RAW, "acs2023_5y_b08301_counties.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype={"fips": str}).set_index("fips")
    src = os.path.join(RAW, "acsdt5y2023-b08301.dat")
    ct_tracts = set()
    dbf = os.path.join(RAW, "tl_2020_09_tabblock20.dbf")
    import shapefile
    rd = shapefile.Reader(dbf=open(dbf, "rb"))
    names = [f[0] for f in rd.fields[1:]]
    ci, ti = names.index("COUNTYFP20"), names.index("TRACTCE20")
    for rec in rd.iterRecords(fields=["COUNTYFP20", "TRACTCE20"]):
        if rec[0] == "001":
            ct_tracts.add(rec[1])
    rows = []
    for ch in pd.read_csv(src, sep="|", usecols=["GEO_ID"] + B08301_COLS, dtype={"GEO_ID": str}, chunksize=200_000):
        cty = ch[ch["GEO_ID"].str.startswith("0500000US")]
        cty = cty.assign(fips=cty["GEO_ID"].str[9:14])
        rows.append(cty[cty["fips"].isin(COUNTIES)])
        tr = ch[ch["GEO_ID"].str.startswith("1400000US09")]
        tr = tr[tr["GEO_ID"].str[14:20].isin(ct_tracts)]
        if len(tr):
            s = tr[B08301_COLS].sum().to_frame().T
            s["fips"] = "09001"
            rows.append(s)
    d = pd.concat(rows, ignore_index=True).groupby("fips")[B08301_COLS].sum()
    assert len(d) == len(COUNTIES), d.index.tolist()
    d.to_csv(cache)
    return d


def cmd_targets():
    os.makedirs(WORK, exist_ok=True)
    net = json.load(open(os.path.join(WORK, "nyc_real.json")))
    st_idx = {s["key"]: i for i, s in enumerate(net["stations"])}
    out = {"sources": {}}
    # 1. subway entries and exits per complex per period: MTA OD estimate, Wednesdays of Oct 2024
    od = pd.read_csv(OD_SUBWAY, usecols=["hour_of_day", "origin_station_complex_id", "destination_station_complex_id", "estimated_average_ridership"],
                     dtype={"origin_station_complex_id": str, "destination_station_complex_id": str})
    od["q"] = od["hour_of_day"].map(period_of_hour)
    ent = od.groupby(["origin_station_complex_id", "q"])["estimated_average_ridership"].sum().unstack(fill_value=0)
    ext = od.groupby(["destination_station_complex_id", "q"])["estimated_average_ridership"].sum().unstack(fill_value=0)
    by_hour = od.groupby("hour_of_day")["estimated_average_ridership"].sum()
    mta = pd.read_csv(MTA_STATIONS, dtype=str)
    boro = mta.groupby("Complex ID")["Borough"].first().to_dict()
    sub = {}
    unmatched = 0.0
    for cid in sorted(set(ent.index) | set(ext.index)):
        k = "mta:" + cid
        e = [float(ent.loc[cid].get(q, 0)) if cid in ent.index else 0.0 for q in range(5)]
        x = [float(ext.loc[cid].get(q, 0)) if cid in ext.index else 0.0 for q in range(5)]
        if k not in st_idx:
            unmatched += sum(e)
            continue
        sub[k] = {"complex": cid, "borough": boro.get(cid, "?"), "entries": e, "exits": x}
    out["subway_od"] = {"stations": sub, "by_hour": by_hour.round(0).tolist(), "total": float(od["estimated_average_ridership"].sum()),
                        "unmatched_entries": unmatched}
    out["sources"]["subway_od"] = "MTA Subway Origin-Destination Ridership Estimate 2024 (data.ny.gov jsu2-fbtj), Wednesdays of October 2024, via maps/riders/nycriders"
    print(f"subway OD: {out['subway_od']['total']:,.0f} trips a day, {len(sub)} complexes matched, {unmatched:,.0f} entries unmatched")
    # 2. busiest hour per directed station pair: nycriders' RAPTOR assignment of the same OD
    stats = json.load(open(NYCRIDERS_STATS))
    mta_cx = dict(zip(mta["GTFS Stop ID"], mta["Complex ID"]))
    links = defaultdict(lambda: np.zeros(24))
    cover = defaultdict(set)
    for sgm in stats["segments"]:
        a, b = mta_cx.get(sgm["from"]), mta_cx.get(sgm["to"])
        if not a or not b or a == b:
            continue
        ia, ib = "mta:" + a, "mta:" + b
        if ia not in st_idx or ib not in st_idx:
            continue
        links[(ia, ib)] += np.array([h[0] for h in sgm["by_hour"]])
        # nycriders splits an express hop over the local stretches it runs along ("parents"),
        # so a link's riders are everyone passing there; the model's hops are spread the same way
        for pa, pb in sgm["parents"]:
            xa, xb = mta_cx.get(pa), mta_cx.get(pb)
            if xa and xb and xa != xb:
                cover[("mta:" + xa, "mta:" + xb)].add((ia, ib))
    out["subway_links"] = [{"a": a, "b": b, "by_hour": v.round(0).tolist()} for (a, b), v in links.items()]
    out["subway_cover"] = [{"hop": list(k), "links": sorted(list(v))} for k, v in cover.items()]
    out["sources"]["subway_links"] = "maps/riders/nycriders stats.json: the same OD routed over the GTFS timetable (RAPTOR), riders per segment per hour"
    print(f"subway links: {len(links)} directed station pairs")
    # 3. commute mode share by county of residence: ACS 2019-2023 B08301
    acs = acs_counties()
    out["acs"] = {f: {"name": COUNTIES[f], **{c[-3:]: float(acs.loc[f, c]) for c in B08301_COLS}} for f in COUNTY_ORDER}
    out["sources"]["acs"] = "ACS 2019-2023 5-year B08301, table-based summary file; Fairfield CT from its 2020 tracts"
    # 4. weekday totals by operator
    import io
    import requests
    resp = requests.get(MTA_DAILY, headers={"User-Agent": "trainworld-pipeline/1.0"}, timeout=60)
    resp.raise_for_status()
    md = pd.read_csv(io.StringIO(resp.text))
    md["date"] = pd.to_datetime(md["date"])
    wed = md[md["date"].dt.dayofweek == 2]
    out["operators"] = {
        "subway": float(wed["subways_total_estimated_ridership"].mean()),
        "sir": float(wed["staten_island_railway_total_estimated_ridership"].mean()),
        "lirr": float(wed["lirr_total_estimated_ridership"].mean()),
        "mnr": float(wed["metro_north_total_estimated_ridership"].mean()),
        "bus_mta": float(wed["buses_total_estimated_ridersip"].mean()),
        "path": PATH_WEEKDAY,
        "njt_rough": NJT_WEEKDAY_ROUGH,
    }
    out["sources"]["operators"] = "MTA Daily Ridership (data.ny.gov vxuj-8kew), Wednesdays of October 2024; PATH Oct 2024 weekday 211,525 (PANYNJ via Progressive Railroading); NJT rail rough"
    print("operators:", {k: round(v) for k, v in out["operators"].items()})
    p = os.path.join(WORK, "targets.json")
    json.dump(out, open(p, "w"))
    print("->", p)


# ---------------------------------------------------------------- comparison

PEAK_HOUR_FACTOR = [1.4, 1.15, 1.4, 1.3, 1.5]


def compare(result_path, tag=None, plot=True, quiet=False, net_path=None):
    """Model (realnet bin output) against the measured targets. Returns a dict of headline
    numbers; prints a report; with `plot`, writes T-007<tag>.png in the project root."""
    net = json.load(open(net_path or os.path.join(WORK, "nyc_real.json")))
    tg = json.load(open(os.path.join(WORK, "targets.json")))
    res = json.load(open(result_path))
    P = res["periods"]
    say = (lambda *a: None) if quiet else print
    # everything is compared per demand station (the model merges stations within
    # `station_merge_m`; measured numbers are summed the same way)
    smap = res.get("st_map") or list(range(len(net["stations"])))
    key_idx = {s["key"]: i for i, s in enumerate(net["stations"])}
    nd = max(smap) + 1
    st = [None] * nd
    for i, s in enumerate(net["stations"]):
        d = smap[i]
        if st[d] is None or (s["key"].startswith("mta:") and not st[d]["key"].startswith("mta:")):
            st[d] = dict(s)
    tsub = {}
    for k, v in tg["subway_od"]["stations"].items():
        d = smap[key_idx[k]]
        if d in tsub:
            tsub[d] = {**tsub[d], "entries": [a + b for a, b in zip(tsub[d]["entries"], v["entries"])],
                       "exits": [a + b for a, b in zip(tsub[d]["exits"], v["exits"])]}
        else:
            tsub[d] = v
    tlinks = defaultdict(lambda: np.zeros(24))
    for L in tg["subway_links"]:
        a, b = smap[key_idx[L["a"]]], smap[key_idx[L["b"]]]
        if a != b:
            tlinks[(a, b)] += np.array(L["by_hour"])
    h = {}
    day = res["day"]
    h["model_trips"] = day["trips"]
    h["model_rail"] = day["rail"]
    h["model_rail_share"] = day["rail"] / day["trips"]
    h["model_walk_share"] = day["walk"] / day["trips"]
    h["other_leg_share"] = day.get("rail_other", 0) / max(day["rail"], 1)  # before T-078 only
    # --- mode share by county of residence (ACS: commuters, i.e. workers less work from home)
    acs = tg["acs"]
    rows = []
    for i, f in enumerate(COUNTY_ORDER):
        a = acs[f]
        comm = a["001"] - a["021"]
        rail_acs = (a["012"] + a["013"] + a["014"]) / comm
        r = sum(p["county"]["rail"][i] for p in P)
        t = sum(p["county"]["trips"][i] for p in P)
        rows.append((COUNTIES[f], comm, rail_acs, r / t if t else 0, a["011"] / comm, a["019"] / comm, a["012"] / comm, a["013"] / comm, t / 2))
    cty = pd.DataFrame(rows, columns=["county", "commuters", "acs_rail", "model_rail", "acs_bus", "acs_walk", "acs_subway", "acs_cr", "model_commuters"])
    acs_rail_trips = (cty.acs_rail * cty.commuters).sum() * 2
    h["acs_rail_share"] = (cty.acs_rail * cty.commuters).sum() / cty.commuters.sum()
    h["acs_walk_share"] = (cty.acs_walk * cty.commuters).sum() / cty.commuters.sum()
    h["acs_rail_commute_trips"] = acs_rail_trips
    say(f"\nDay: model {day['rail']:,.0f} rail commute trips = {100 * h['model_rail_share']:.1f}% of {day['trips']:,.0f} "
        f"(walk {100 * h['model_walk_share']:.1f}%" + (f", other leg {100 * h['other_leg_share']:.0f}% of rail)" if h['other_leg_share'] else ")"))
    say(f"ACS: rail {100 * h['acs_rail_share']:.1f}% of commuters (subway, commuter rail, light rail), walk {100 * h['acs_walk_share']:.1f}%; "
        f"{acs_rail_trips:,.0f} rail commute trips a day if each commuter goes and returns")
    cty["diff"] = cty.model_rail - cty.acs_rail
    say("\nRail share by county of residence (ACS 2019-2023 vs model):")
    say(cty.sort_values("commuters", ascending=False).to_string(index=False, formatters={c: "{:.1%}".format for c in ["acs_rail", "model_rail", "acs_bus", "acs_walk", "acs_subway", "acs_cr", "diff"]} | {"commuters": "{:,.0f}".format, "model_commuters": "{:,.0f}".format}))
    w = cty.commuters
    h["county_rail_mae_pts"] = float((cty["diff"].abs() * w).sum() / w.sum() * 100)
    h["county_rail_logr"] = float(np.corrcoef(np.log(cty.acs_rail.clip(1e-3)), np.log(cty.model_rail.clip(1e-3)))[0, 1])
    # --- subway entries per complex
    sub = tsub
    idx = sorted(sub)
    meas_day = np.array([sum(sub[i]["entries"]) for i in idx])
    meas_am = np.array([sub[i]["entries"][0] for i in idx])
    mod_q = np.array([[P[q]["entries"][i] for i in idx] for q in range(5)])
    mod_day, mod_am = mod_q.sum(0), mod_q[0]
    is_sub = np.array([s["key"].startswith("mta:") for s in st])
    mod_sub_all = sum(np.array(P[q]["entries"])[is_sub].sum() for q in range(5))
    h["subway_entries_model"] = float(mod_sub_all)
    h["subway_entries_measured"] = float(meas_day.sum())
    h["subway_am_model"] = float(mod_am.sum())
    h["subway_am_measured"] = float(meas_am.sum())
    say(f"\nSubway entries a day: model {mod_sub_all:,.0f} (commutes only) vs MTA OD {meas_day.sum():,.0f} (all trips); "
        f"6-10: model {mod_am.sum():,.0f} vs {meas_am.sum():,.0f}")
    by_hour = np.array(tg["subway_od"]["by_hour"])
    meas_q = [sum(by_hour[a:b]) for a, b in PERIOD_HOURS]
    say("By period, measured share of the day: " + " ".join(f"{100 * v / sum(meas_q):.0f}%" for v in meas_q) +
        "; model: " + " ".join(f"{100 * mod_q[q].sum() / mod_q.sum():.0f}%" for q in range(5)))
    good = (meas_day > 0)
    lr = np.corrcoef(np.log(meas_day[good]), np.log(np.maximum(mod_day[good], 1)))[0, 1]
    lr_am = np.corrcoef(np.log(np.maximum(meas_am[good], 1)), np.log(np.maximum(mod_am[good], 1)))[0, 1]
    h["station_logr_day"], h["station_logr_am"] = float(lr), float(lr_am)
    scale = meas_am.sum() / max(mod_am.sum(), 1)
    h["station_am_misplaced"] = float(0.5 * np.abs(meas_am / meas_am.sum() - mod_am / mod_am.sum()).sum())
    say(f"Station entries, log correlation over {good.sum()} complexes: day {lr:.3f}, 6-10 {lr_am:.3f}; "
        f"share of 6-10 entries in the wrong station {100 * h['station_am_misplaced']:.1f}%")
    names = [st[i]["name"] for i in idx]
    boro = [sub[i]["borough"] for i in idx]
    df = pd.DataFrame({"station": names, "borough": boro, "meas_day": meas_day, "model_day": mod_day, "meas_am": meas_am, "model_am": mod_am})
    df["ratio_am"] = df.model_am / df.meas_am.clip(1)
    top = df.sort_values("meas_day", ascending=False).head(30)
    say("\nTop 30 complexes by measured entries (day; 6-10):")
    say(top.to_string(index=False, formatters={c: "{:,.0f}".format for c in ["meas_day", "model_day", "meas_am", "model_am"]} | {"ratio_am": "{:.2f}".format, "station": lambda s: s[:34]}))
    bb = df.groupby("borough")[["meas_day", "model_day", "meas_am", "model_am"]].sum()
    bb["share_meas"] = bb.meas_day / bb.meas_day.sum()
    bb["share_model"] = bb.model_day / bb.model_day.sum()
    say("\nSubway entries by borough:")
    say(bb.to_string(formatters={c: "{:,.0f}".format for c in ["meas_day", "model_day", "meas_am", "model_am"]} | {"share_meas": "{:.1%}".format, "share_model": "{:.1%}".format}))
    # --- busiest links: measured busiest hour (8-9) vs model 6-10 load x 1.4 / 4
    rs = np.array(res["rn_station"])
    seg0 = np.array(P[0]["seg"])
    # model load per directed station pair: route node r -> next route node of the same line run
    rl = np.array(res["rn_line"])
    hop_flow = defaultdict(float)
    for r in range(len(rs) - 1):
        if rl[r] == rl[r + 1] and seg0[r] > 0:
            hop_flow[(int(rs[r]), int(rs[r + 1]))] += seg0[r]
    cover = defaultdict(set)
    for c in tg.get("subway_cover", []):
        x, y = smap[key_idx[c["hop"][0]]], smap[key_idx[c["hop"][1]]]
        for a, b in c["links"]:
            a, b = smap[key_idx[a]], smap[key_idx[b]]
            if a != b:
                cover[(x, y)].add((a, b))
    model_link = defaultdict(float)
    for hop, f in hop_flow.items():
        for L in (cover.get(hop) or {hop}):
            model_link[L] += f
    lk = []
    for (a, b), hb in tlinks.items():
        lk.append((a, b, hb[8], hb[6:10].sum(), model_link.get((a, b), 0.0)))
    lk = pd.DataFrame(lk, columns=["a", "b", "meas_8to9", "meas_6to10", "model_6to10"])
    lk["model_peak_hour"] = lk.model_6to10 * PEAK_HOUR_FACTOR[0] / 4
    lk["name"] = [f"{st[a]['name'][:20]} > {st[b]['name'][:20]}" for a, b in zip(lk.a, lk.b)]
    topl = lk.sort_values("meas_8to9", ascending=False).head(20)
    say("\nBusiest subway links, riders in the busiest hour (measured 8-9; model 6-10 x 1.4 / 4):")
    say(topl[["name", "meas_8to9", "model_peak_hour", "meas_6to10", "model_6to10"]].to_string(index=False, formatters={c: "{:,.0f}".format for c in ["meas_8to9", "model_peak_hour", "meas_6to10", "model_6to10"]}))
    h["top20_links_meas"] = float(topl.meas_8to9.sum())
    h["top20_links_model"] = float(topl.model_peak_hour.sum())
    m = lk.meas_6to10 > 100
    h["link_logr_am"] = float(np.corrcoef(np.log(lk.meas_6to10[m]), np.log(lk.model_6to10[m].clip(1)))[0, 1])
    h["max_link_model"] = float(lk.model_peak_hour.max())
    h["max_link_meas"] = float(lk.meas_8to9.max())
    # --- line totals by operator (boardings incl. transfers within the line's operator)
    lines = net["lines"]
    ent_feed = defaultdict(float)
    for q in range(5):
        e = np.array(P[q]["entries"])
        for i, s in enumerate(st):
            ent_feed[s["feed"]] += e[i]
    h["entries_by_feed"] = dict(ent_feed)
    ops = tg["operators"]
    say("\nTrips starting at each operator's stations, model (commutes) vs operator weekday ridership (all trips):")
    say(f"  subway incl SIR {ent_feed['subway']:,.0f} vs {ops['subway'] + ops['sir']:,.0f}; LIRR {ent_feed['lirr']:,.0f} vs {ops['lirr']:,.0f}; "
        f"MNR {ent_feed['mnr']:,.0f} vs {ops['mnr']:,.0f}; PATH {ent_feed['path']:,.0f} vs {ops['path']:,.0f}; NJT {ent_feed['njt']:,.0f} vs ~{ops['njt_rough']:,.0f}")
    # --- crowding
    lc0 = np.array(P[0]["load_of_crush"])
    h["worst_crush_am"] = float(lc0.max())
    worst = np.argsort(-lc0)[:10]
    say("\nFullest segments, morning busiest hour, share of crush: " + "; ".join(
        f"{lines[rl[r]]['feed']} {lines[rl[r]]['name']} at {st[rs[r]]['name'][:18]} {100 * lc0[r]:.0f}%" for r in worst))
    h["rounds_am"] = [r[2] for r in P[0]["rounds"]]
    if plot:
        plot_compare(df, cty, lk, h, tag)
    return h


def plot_compare(df, cty, lk, h, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    a = ax[0]
    a.scatter(df.meas_am.clip(10), df.model_am.clip(10), s=8, alpha=0.6, color="#3b6fb6")
    lim = [10, max(df.meas_am.max(), df.model_am.max()) * 1.3]
    a.plot(lim, lim, color="#999", lw=1)
    a.set_xscale("log"), a.set_yscale("log"), a.set_xlim(lim), a.set_ylim(lim)
    a.set_xlabel("Measured entries 6-10 (MTA OD, Oct 2024)"), a.set_ylabel("Model entries 6-10 (commutes)")
    a.set_title(f"Subway entries by complex, morning peak (log r {h['station_logr_am']:.2f})")
    for _, r in df.sort_values("meas_am", ascending=False).head(8).iterrows():
        a.annotate(r.station.split("/")[0][:18], (max(r.meas_am, 10), max(r.model_am, 10)), fontsize=7)
    a = ax[1]
    a.scatter(cty.acs_rail * 100, cty.model_rail * 100, s=np.sqrt(cty.commuters) / 3, alpha=0.6, color="#c0504d")
    m = max(cty.acs_rail.max(), cty.model_rail.max()) * 105
    a.plot([0, m], [0, m], color="#999", lw=1)
    for _, r in cty.iterrows():
        a.annotate(r.county, (r.acs_rail * 100, r.model_rail * 100), fontsize=7)
    a.set_xlabel("ACS rail share of commuters, % (subway, commuter rail, light rail)"), a.set_ylabel("Model rail share, %")
    a.set_title("Rail commute share by county of residence")
    a = ax[2]
    s = lk[lk.meas_8to9 > 0]
    a.scatter(s.meas_8to9.clip(10), s.model_peak_hour.clip(10), s=8, alpha=0.6, color="#4f9a5a")
    lim = [10, max(s.meas_8to9.max(), s.model_peak_hour.max()) * 1.3]
    a.plot(lim, lim, color="#999", lw=1)
    a.set_xscale("log"), a.set_yscale("log"), a.set_xlim(lim), a.set_ylim(lim)
    a.set_xlabel("Measured riders 8-9 (nycriders routing of the OD)"), a.set_ylabel("Model busiest hour (6-10 x 1.4 / 4)")
    a.set_title(f"Subway link loads, morning busiest hour (log r {h['link_logr_am']:.2f})")
    fig.tight_layout()
    p = os.path.join(ROOT, f"T-007{('-' + tag) if tag else ''}.png")
    fig.savefig(p, dpi=110)
    print("->", p)


def simplify(pts, tol):
    """Douglas-Peucker on an (n, 2) array; keeps the ends."""
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        ab = b - a
        L = np.hypot(*ab)
        seg = pts[i + 1:j] - a
        d = np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / L if L > 0 else np.hypot(seg[:, 0], seg[:, 1])
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return pts[keep]


def cmd_geom():
    """Track geometry for the save (T-007): per line, a node per stop on the line's own GTFS shape
    nearest its station, and the shape between consecutive stops, simplified to 15 m. A hop with no
    shape near both ends is a straight line."""
    o = origin()
    net = json.load(open(os.path.join(WORK, "nyc_real.json")))
    st = net["stations"]
    shapes_by_route = {}
    for name in FEEDS:
        folder = FEEDS[name][0]
        sh = read(folder, "shapes.txt", usecols=["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"])
        sh["seq"] = sh["shape_pt_sequence"].astype(float)
        sh = sh.sort_values(["shape_id", "seq"])
        x, y = to_xy(sh["shape_pt_lon"].astype(float).to_numpy(), sh["shape_pt_lat"].astype(float).to_numpy(), o)
        sh["x"], sh["y"] = x, y
        pts = {sid: g[["x", "y"]].to_numpy() for sid, g in sh.groupby("shape_id", sort=False)}
        tr = read(folder, "trips.txt", usecols=["route_id", "shape_id"])
        for r, g in tr.groupby("route_id"):
            ids = [s for s in g["shape_id"].unique() if s in pts]
            # longest shapes first; a short turn's shape is a piece of a longer one
            shapes_by_route[(name, r)] = sorted((pts[s] for s in ids), key=len, reverse=True)
        print(f"{name}: {len(pts)} shapes")
    out = []
    n_straight = 0
    for l in net["lines"]:
        shapes = shapes_by_route.get((l["feed"], l["route"]), [])
        xy = [np.array([st[k]["x"], st[k]["y"]]) for k in l["stops"]]
        # the shape that passes nearest all stops of the line
        best, best_d = None, np.inf
        for s in shapes[:40]:
            d = sum(np.min(np.hypot(*(s - p).T)) for p in xy)
            if d < best_d:
                best, best_d = s, d
        nodes, hops = [], []
        idx = []
        for p in xy:
            if best is not None:
                dd = np.hypot(*(best - p).T)
                i = int(np.argmin(dd))
                idx.append(i if dd[i] < 400 else None)
                nodes.append(best[i].tolist() if dd[i] < 400 else p.tolist())
            else:
                idx.append(None)
                nodes.append(p.tolist())
        for k in range(len(xy) - 1):
            i, j = idx[k], idx[k + 1]
            if i is not None and j is not None and abs(j - i) >= 2:
                seg = best[min(i, j):max(i, j) + 1]
                if j < i:
                    seg = seg[::-1]
                seg = simplify(seg, 15.0)
                hops.append(seg[1:-1].round(1).tolist())
            else:
                hops.append([])
                if i is None or j is None:
                    n_straight += 1
        out.append({"nodes": nodes, "hops": hops})
    # one level per line, chosen so that as few line pairs as possible cross on the same level
    # (each line has its own track here; a same-level crossing would be a flat crossing)
    from shapely import STRtree
    from shapely.geometry import LineString
    geoms = []
    for g in out:
        pts = [g["nodes"][0]]
        for k, h in enumerate(g["hops"]):
            pts += h + [g["nodes"][k + 1]]
        geoms.append(LineString(pts))
    tree = STRtree(geoms)
    cross = defaultdict(int)
    for i, g in enumerate(geoms):
        for j in tree.query(g):
            j = int(j)
            if j <= i:
                continue
            x = g.intersection(geoms[j])
            n = len(getattr(x, "geoms", [x])) if not x.is_empty else 0
            if n:
                cross[(i, j)] = n
                cross[(j, i)] = n
    nbr = defaultdict(list)
    for (i, j), n in cross.items():
        nbr[i].append((j, n))
    levels = [None] * len(out)
    palette = [-1, -2, -3, 1, 2, 3]
    for i in sorted(range(len(out)), key=lambda i: -sum(n for _, n in nbr[i])):
        cost = {lv: sum(n for j, n in nbr[i] if levels[j] == lv) for lv in palette}
        levels[i] = min(palette, key=lambda lv: (cost[lv], palette.index(lv)))
    left = sum(n for (i, j), n in cross.items() if i < j and levels[i] == levels[j])
    for g, lv in zip(out, levels):
        g["level"] = lv
    print(f"{sum(1 for (i, j) in cross if i < j)} crossing line pairs; {left} crossing points left on one level")
    p = os.path.join(WORK, "nyc_real_geom.json")
    json.dump({"lines": out}, open(p, "w"))
    print(f"{len(out)} lines, {n_straight} hops straight for want of a shape -> {p}")


def cmd_gamesave():
    """Wrap the track save (`realsave` bin) as a game file the app's "Load a file" reads (T-029's
    format: gzip of "TWG1", u32 header length, JSON header, TWT2): day 1 07:00, $6B, the fleet the
    lines need, the starting fares."""
    import gzip
    import struct
    track = open(os.path.join(WORK, "nyc_real.twt2"), "rb").read()
    tj = json.load(open(os.path.join(WORK, "nyc_real_track.json")))
    header = {"game": "anitabuilder", "format": 1, "city": "nyc", "clock": 86400 + 7 * 3600, "cash": 6000.0,
              "fleet": int(tj["fleet"]), "fares": {"base": 1.5, "perKm": 0.1}, "ledger": []}
    hj = json.dumps(header).encode()
    raw = b"TWG1" + struct.pack("<I", len(hj)) + hj + track
    p = os.path.join(ROOT, "T-007-real-nyc.save")
    with open(p, "wb") as f:
        f.write(gzip.compress(raw, mtime=0))
    print(f"{len(raw):,} bytes raw, {os.path.getsize(p):,} gzipped, fleet {header['fleet']:,} cars -> {p}")


SINGLE_LINE = [  # the batch-3 test line: 12.4 km under Manhattan, 7 stations, 14 trains an hour
    ("Bowling Green", -74.0141, 40.7046), ("Canal St", -74.0005, 40.7190), ("Union Sq", -73.9903, 40.7347),
    ("Grand Central", -73.9772, 40.7520), ("59 St", -73.9672, 40.7626), ("86 St", -73.9555, 40.7795),
    ("116 St", -73.9417, 40.7985),
]


def cmd_single():
    """The single-line case from notes/T-025.md as a network for the realnet bin: 7 stations about
    2 km apart, 124 s a hop with the dwell (its 30m51s round trip less two 3-minute turnarounds),
    14 / 8 / 4 trains an hour, 10 cars."""
    o = origin()
    xy = [to_xy(lon, lat, o) for _, lon, lat in SINGLE_LINE]
    n = len(xy)
    d = [float(np.hypot(xy[i + 1][0] - xy[i][0], xy[i + 1][1] - xy[i][1])) for i in range(n - 1)]
    t0 = [124.0] * (n - 1) + [0.0]
    t1 = [0.0] + [124.0] * (n - 1)
    times = t0 * 3 + t1 * 3
    net = {"stations": [{"key": f"single:{i}", "name": s[0], "feed": "single", "x": float(p[0]), "y": float(p[1])} for i, (s, p) in enumerate(zip(SINGLE_LINE, xy))],
           "lines": [{"feed": "single", "name": "single", "stops": list(range(n)), "tph": [14, 8, 4], "cars": 10}],
           "demand": {"st_xy": [float(v) for p in xy for v in p], "line_n": [n], "stops": list(range(n)), "times": times, "tph": [14.0, 8.0, 4.0], "cars": [10.0]}}
    p = os.path.join(WORK, "single_line.json")
    json.dump(net, open(p, "w"))
    print(f"{sum(d) / 1000:.1f} km, hops {[round(v) for v in d]} m -> {p}")


def cmd_compare():
    args = sys.argv[2:]
    tag = None
    if "--tag" in args:
        tag = args[args.index("--tag") + 1]
    h = compare(args[0], tag, plot="--no-plot" not in args, net_path=args[args.index("--net") + 1] if "--net" in args else None)
    print("\nheadline:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items() if k != "entries_by_feed"}))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "net"
    {"net": cmd_net, "counties": cmd_counties, "targets": cmd_targets, "compare": cmd_compare, "single": cmd_single, "geom": cmd_geom, "gamesave": cmd_gamesave}[cmd]()
