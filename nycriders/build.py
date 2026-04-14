"""
Build train animation data using RAPTOR timetable routing + O-D ridership.

Pipeline:
1. Parse GTFS into route patterns and timetables
2. For each O-D pair, find shortest journey via RAPTOR (supports transfers)
3. Assign riders to specific train runs
4. Output trains.json with train timelines and route geometries

Replaces the old build_paths.py (shapefile graph) + build_trains.py (direct-only routing).
"""
import argparse
import math
import multiprocessing as mp
import pickle
import tempfile
import pandas as pd
import json
import csv
import time as time_module
import os
import random
from collections import defaultdict
from bisect import bisect_left

GTFS_DIR = "data/gtfs_subway"
STATIONS_CSV = "../data/nystreets/MTA_Subway_Stations.csv"
OD_CSV = "data/od_wednesday_sep.csv"
OUT_JSON = "trains.json"
GEOMETRY_JSON = "geometry.json"
SERVICE_ID = "Weekday"
DEFAULT_SAMPLE_RATE = 1.0  # 0.1 for testing, 1.0 for full data
SAMPLE_RATE = DEFAULT_SAMPLE_RATE
HOUR_RANGE = None  # e.g. (7, 12) for 7am-noon
TRANSFER_TIME = 180  # default transfer penalty (seconds)
BASE_TRANSFER_TIME = 60  # seconds (stairs, doors, overhead)
WALK_SPEED = 1.2  # m/s average indoor walking speed
MAX_ROUNDS = 3  # max trips per journey (= max transfers + 1)
DEP_BIN = 120  # group departure times into 2-minute bins for RAPTOR caching
SHAPE_SIMPLIFY_EPSILON = 0.0001  # ~11m, for Douglas-Peucker simplification of waypoints

INF = float('inf')


# ---------------------------------------------------------------------------
# Data loading (reused from build_trains.py)
# ---------------------------------------------------------------------------

def load_gtfs():
    """Load GTFS weekday stop_times, trips, and route colors."""
    print("Loading GTFS data...")

    trips = pd.read_csv(f"{GTFS_DIR}/trips.txt")
    trips = trips[trips.service_id == SERVICE_ID]
    print(f"  {len(trips)} weekday trips")

    import os
    st_path = f"{GTFS_DIR}/stop_times_fixed.txt"
    if not os.path.exists(st_path):
        st_path = f"{GTFS_DIR}/stop_times.txt"
    print(f"  using {st_path}")
    stop_times = pd.read_csv(st_path)
    stop_times = stop_times[stop_times.trip_id.isin(trips.trip_id)]

    def time_to_secs(t):
        parts = t.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

    stop_times['arr_secs'] = stop_times.arrival_time.apply(time_to_secs)
    stop_times['dep_secs'] = stop_times.departure_time.apply(time_to_secs)
    stop_times['parent_stop'] = stop_times.stop_id.str.replace(r'[NS]$', '', regex=True)
    stop_times = stop_times.sort_values(['trip_id', 'stop_sequence'])
    print(f"  {len(stop_times)} stop_time entries")

    routes = pd.read_csv(f"{GTFS_DIR}/routes.txt")
    route_colors = dict(zip(routes.route_id, routes.route_color.apply(lambda c: f'#{c}')))

    trip_route = dict(zip(trips.trip_id, trips.route_id))
    trip_dir = dict(zip(trips.trip_id, trips.direction_id))
    stop_times['route_id'] = stop_times.trip_id.map(trip_route)
    stop_times['direction_id'] = stop_times.trip_id.map(trip_dir)

    return stop_times, trips, route_colors


def load_stop_to_complex():
    """GTFS parent stop ID -> station complex ID."""
    stations = pd.read_csv(STATIONS_CSV)
    return dict(zip(stations['GTFS Stop ID'].astype(str),
                    stations['Complex ID'].astype(int)))


def load_stop_coords():
    """Parent station coordinates from GTFS stops.txt."""
    stops = pd.read_csv(f"{GTFS_DIR}/stops.txt")
    parent = stops[stops.location_type == 1]
    return {str(row.stop_id): (row.stop_lat, row.stop_lon)
            for _, row in parent.iterrows()}


def load_od_data():
    """Load O-D ridership keyed by (origin_complex, dest_complex, hour)."""
    label = f"sample rate: {SAMPLE_RATE}"
    if HOUR_RANGE:
        label += f", hours {HOUR_RANGE[0]}-{HOUR_RANGE[1]}"
    print(f"Loading O-D data ({label})...")
    od = defaultdict(float)
    random.seed(123)
    with open(OD_CSV) as f:
        for row in csv.DictReader(f):
            if random.random() > SAMPLE_RATE:
                continue
            hour = int(row['hour_of_day'])
            if HOUR_RANGE and not (HOUR_RANGE[0] <= hour < HOUR_RANGE[1]):
                continue
            oid = int(row['origin_station_complex_id'])
            did = int(row['destination_station_complex_id'])
            riders = float(row['estimated_average_ridership'])
            if oid != did:
                od[(oid, did, hour)] += riders
    print(f"  {len(od)} O-D-hour tuples")
    return od


def load_complex_info(stop_to_complex, stop_coords):
    """Build complex_id -> {name, lat, lon} from stations CSV + GTFS coords."""
    stations = pd.read_csv(STATIONS_CSV)
    info = {}
    for _, row in stations.iterrows():
        cid = int(row['Complex ID'])
        if cid in info:
            continue
        stop_id = str(row['GTFS Stop ID'])
        if stop_id in stop_coords:
            lat, lon = stop_coords[stop_id]
        else:
            lat, lon = row['GTFS Latitude'], row['GTFS Longitude']
        info[cid] = {'name': row['Stop Name'], 'lat': float(lat), 'lon': float(lon)}
    return info


def load_geometry():
    """Load geometry.json (from build_geometry.py) for curved lines + station-level coords."""
    if not os.path.exists(GEOMETRY_JSON):
        print(f"  {GEOMETRY_JSON} not found, using straight-line fallback")
        return None
    print(f"Loading {GEOMETRY_JSON}...")
    with open(GEOMETRY_JSON) as f:
        geo = json.load(f)
    print(f"  {len(geo['lines'])} line geometries, "
          f"{len(geo['stop_coords'])} station coords, "
          f"{len(geo['stop_route_coords'])} per-route coords")
    return geo


def load_shapes():
    """Load GTFS shapes into dict of shape_id -> [(lat, lon), ...]."""
    print("Loading GTFS shapes...")
    shapes_raw = defaultdict(list)
    with open(f"{GTFS_DIR}/shapes.txt") as f:
        for row in csv.DictReader(f):
            shapes_raw[row['shape_id']].append((
                int(row['shape_pt_sequence']),
                float(row['shape_pt_lat']),
                float(row['shape_pt_lon']),
            ))
    shapes = {}
    for sid, pts in shapes_raw.items():
        pts.sort(key=lambda p: p[0])
        shapes[sid] = [(lat, lon) for _, lat, lon in pts]
    print(f"  {len(shapes)} shapes")
    return shapes


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def approx_distance_m(lat1, lon1, lat2, lon2):
    """Approximate distance in meters between two lat/lon points."""
    dlat = (lat2 - lat1) * 111_000
    dlon = (lon2 - lon1) * 111_000 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.sqrt(dlat * dlat + dlon * dlon)


def snap_to_polyline(lat, lon, polyline):
    """Snap (lat, lon) to nearest point on polyline. Returns (lat, lon, dist, seg_idx, t)."""
    best_dist = float('inf')
    best_lat, best_lon = lat, lon
    best_seg, best_t = 0, 0.0
    for i in range(len(polyline) - 1):
        lat1, lon1 = polyline[i]
        lat2, lon2 = polyline[i + 1]
        dx, dy = lat2 - lat1, lon2 - lon1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-14:
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((lat - lat1) * dx + (lon - lon1) * dy) / seg_len_sq))
        proj_lat = lat1 + t * dx
        proj_lon = lon1 + t * dy
        dist = (lat - proj_lat) ** 2 + (lon - proj_lon) ** 2
        if dist < best_dist:
            best_dist = dist
            best_lat, best_lon = proj_lat, proj_lon
            best_seg, best_t = i, t
    return best_lat, best_lon, math.sqrt(best_dist), best_seg, best_t


def extract_subline(polyline, seg1, t1, seg2, t2):
    """Extract polyline from (seg1, t1) to (seg2, t2)."""
    if seg1 > seg2 or (seg1 == seg2 and t1 > t2):
        pts = extract_subline(polyline, seg2, t2, seg1, t1)
        pts.reverse()
        return pts
    points = []
    lat1, lon1 = polyline[seg1]
    lat2, lon2 = polyline[min(seg1 + 1, len(polyline) - 1)]
    points.append((lat1 + t1 * (lat2 - lat1), lon1 + t1 * (lon2 - lon1)))
    for i in range(seg1 + 1, min(seg2 + 1, len(polyline))):
        points.append(polyline[i])
    if seg2 < len(polyline) - 1:
        lat1, lon1 = polyline[seg2]
        lat2, lon2 = polyline[seg2 + 1]
        points.append((lat1 + t2 * (lat2 - lat1), lon1 + t2 * (lon2 - lon1)))
    return points


def simplify_polyline(points, epsilon):
    """Douglas-Peucker polyline simplification."""
    if len(points) <= 2:
        return points
    max_dist = 0
    max_idx = 0
    lat1, lon1 = points[0]
    lat2, lon2 = points[-1]
    dx, dy = lat2 - lat1, lon2 - lon1
    line_len_sq = dx * dx + dy * dy
    for i in range(1, len(points) - 1):
        lat, lon = points[i]
        if line_len_sq < 1e-14:
            dist = math.sqrt((lat - lat1) ** 2 + (lon - lon1) ** 2)
        else:
            t = ((lat - lat1) * dx + (lon - lon1) * dy) / line_len_sq
            proj_lat = lat1 + t * dx
            proj_lon = lon1 + t * dy
            dist = math.sqrt((lat - proj_lat) ** 2 + (lon - proj_lon) ** 2)
        if dist > max_dist:
            max_dist = dist
            max_idx = i
    if max_dist > epsilon:
        left = simplify_polyline(points[:max_idx + 1], epsilon)
        right = simplify_polyline(points[max_idx:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


def precompute_shape_segments(patterns, shapes, trip_shape, get_stop_coord):
    """Pre-compute simplified shape sub-polylines between consecutive stops per pattern."""
    print("Pre-computing shape segments for train animation...")

    # build index: (route, direction) -> [shape_id, ...] for fallback matching
    route_dir_shape_list = defaultdict(list)
    for pattern in patterns:
        for trip in pattern['trips']:
            sid = trip_shape.get(trip['trip_id'])
            if sid and sid in shapes:
                key = (pattern['route'], pattern['direction'])
                if sid not in route_dir_shape_list[key]:
                    route_dir_shape_list[key].append(sid)
                break

    pattern_segments = {}
    matched = 0
    fallback_matched = 0
    for pid, pattern in enumerate(patterns):
        # find a shape for this pattern
        shape_id = None
        for trip in pattern['trips']:
            sid = trip_shape.get(trip['trip_id'])
            if sid and sid in shapes:
                shape_id = sid
                break

        # fallback: find a shape from the same route+direction that covers
        # this pattern's stops (e.g. short-turn trips with no shape_id)
        if not shape_id:
            route = pattern['route']
            direction = pattern['direction']
            stops = pattern['stops']
            candidate_shapes = route_dir_shape_list.get((route, direction), [])
            best_sid = None
            best_max_dist = float('inf')
            for sid in candidate_shapes:
                polyline = shapes[sid]
                max_snap_dist = 0
                for stop_id in stops:
                    coord = get_stop_coord(stop_id, route)
                    if coord:
                        _, _, dist, _, _ = snap_to_polyline(coord[0], coord[1], polyline)
                        max_snap_dist = max(max_snap_dist, dist)
                if max_snap_dist < best_max_dist:
                    best_max_dist = max_snap_dist
                    best_sid = sid
            # accept if worst snap is under ~500m (0.005 degrees)
            if best_sid and best_max_dist < 0.005:
                shape_id = best_sid
                fallback_matched += 1

        if not shape_id:
            continue

        polyline = shapes[shape_id]
        stops = pattern['stops']
        route = pattern['route']

        # snap each stop to shape
        snaps = []
        for stop_id in stops:
            coord = get_stop_coord(stop_id, route)
            if coord:
                _, _, _, seg, t = snap_to_polyline(coord[0], coord[1], polyline)
                snaps.append((seg, t))
            else:
                snaps.append(None)

        # extract + simplify sub-polylines between consecutive stops
        segments = []
        for i in range(len(stops) - 1):
            if snaps[i] is None or snaps[i + 1] is None:
                segments.append(None)
                continue
            seg0, t0 = snaps[i]
            seg1, t1 = snaps[i + 1]
            sub = extract_subline(polyline, seg0, t0, seg1, t1)
            if len(sub) > 2:
                sub = simplify_polyline(sub, SHAPE_SIMPLIFY_EPSILON)
            segments.append(sub if len(sub) > 2 else None)

        pattern_segments[pid] = segments
        matched += 1

    print(f"  {matched}/{len(patterns)} patterns matched to shapes"
          f" ({fallback_matched} via fallback)")
    return pattern_segments


# ---------------------------------------------------------------------------
# RAPTOR preprocessing
# ---------------------------------------------------------------------------

def build_route_patterns(stop_times):
    """
    Group trips into route patterns (unique route + direction + stop sequence).
    Returns list of patterns, each with:
        route, direction, stops (list of parent_stop_id),
        trips (list of {trip_id, times: [(arr, dep), ...]}) sorted by first departure.
    Also returns precomputed departure-time arrays for binary search.
    """
    print("Building route patterns...")

    pattern_groups = defaultdict(list)
    for trip_id, group in stop_times.groupby('trip_id'):
        route = group.iloc[0]['route_id']
        direction = group.iloc[0]['direction_id']
        stops = group['parent_stop'].tolist()
        times = list(zip(group['arr_secs'].tolist(), group['dep_secs'].tolist()))
        key = (route, direction, tuple(stops))
        pattern_groups[key].append({'trip_id': trip_id, 'times': times})

    patterns = []
    for (route, direction, stop_tuple), trip_list in pattern_groups.items():
        # for trips crossing midnight (times >= 86400), add a wrapped copy
        # shifted by -86400 so RAPTOR finds them for riders departing near 00:00
        wrapped = []
        for trip in trip_list:
            if any(t >= 86400 for _, t in trip['times']):
                wrapped.append({
                    'trip_id': '_wrap_' + trip['trip_id'],
                    'times': [(a - 86400, d - 86400) for a, d in trip['times']],
                })
        trip_list = wrapped + trip_list
        trip_list.sort(key=lambda t: t['times'][0][1])  # sort by first stop departure
        patterns.append({
            'route': route,
            'direction': direction,
            'stops': list(stop_tuple),
            'trips': trip_list,
        })

    # precompute departure times at each stop for binary search
    pattern_deps = []
    for pattern in patterns:
        deps_by_stop = []
        for i in range(len(pattern['stops'])):
            deps_by_stop.append([t['times'][i][1] for t in pattern['trips']])
        pattern_deps.append(deps_by_stop)

    print(f"  {len(patterns)} route patterns from {sum(len(p['trips']) for p in patterns)} trips")
    return patterns, pattern_deps


def build_stop_routes(patterns):
    """Inverse index: stop_id -> [(pattern_idx, stop_position)]."""
    stop_routes = defaultdict(list)
    for pid, pattern in enumerate(patterns):
        for pos, stop in enumerate(pattern['stops']):
            stop_routes[stop].append((pid, pos))
    return stop_routes


def build_transfers(stop_to_complex, geometry=None):
    """Transfer edges from GTFS transfers.txt + implicit same-complex transfers.

    When geometry is available, computes distance-based transfer times for
    same-complex transfers instead of using a flat constant.
    """
    print("Building transfer graph...")
    transfers = defaultdict(list)

    # build station coords lookup from geometry for distance calculations
    station_coords = {}
    if geometry:
        for sid, c in geometry['stop_coords'].items():
            station_coords[sid] = (c['lat'], c['lon'])

    # explicit GTFS transfers
    tf = pd.read_csv(f"{GTFS_DIR}/transfers.txt")
    explicit_pairs = set()
    for _, row in tf.iterrows():
        a = str(row.from_stop_id)
        b = str(row.to_stop_id)
        t = int(row.min_transfer_time) if pd.notna(row.min_transfer_time) else TRANSFER_TIME
        if a != b:
            transfers[a].append((b, t))
            transfers[b].append((a, t))
            explicit_pairs.add((a, b))
            explicit_pairs.add((b, a))

    # implicit same-complex transfers
    complex_to_stops = defaultdict(set)
    for stop_id, cid in stop_to_complex.items():
        complex_to_stops[cid].add(stop_id)

    added = 0
    for cid, stop_set in complex_to_stops.items():
        stop_list = list(stop_set)
        for i in range(len(stop_list)):
            for j in range(len(stop_list)):
                if i != j and (stop_list[i], stop_list[j]) not in explicit_pairs:
                    # distance-based transfer time if we have station coords
                    a, b = stop_list[i], stop_list[j]
                    if a in station_coords and b in station_coords:
                        lat1, lon1 = station_coords[a]
                        lat2, lon2 = station_coords[b]
                        dist = approx_distance_m(lat1, lon1, lat2, lon2)
                        t = int(BASE_TRANSFER_TIME + dist / WALK_SPEED)
                    else:
                        t = TRANSFER_TIME
                    transfers[a].append((b, t))
                    added += 1

    print(f"  {len(tf)} GTFS transfers + {added} implicit same-complex transfers")
    if station_coords:
        # show some stats
        implicit_times = [t for a in transfers for b, t in transfers[a]
                         if (a, b) not in explicit_pairs]
        if implicit_times:
            print(f"  Distance-based transfer times: "
                  f"min {min(implicit_times)}s, median {sorted(implicit_times)[len(implicit_times)//2]}s, "
                  f"max {max(implicit_times)}s")
    return transfers, complex_to_stops


# ---------------------------------------------------------------------------
# RAPTOR core
# ---------------------------------------------------------------------------

def raptor_query(origin_stops, dep_time, patterns, pattern_deps, stop_routes, transfers_map):
    """
    Find earliest arrival at all stops from origin_stops departing at dep_time.

    Returns:
        tau: dict[stop_id -> earliest_arrival_secs]
        journey: dict[stop_id -> backpointer]
            Trip leg:     (board_stop, pattern_idx, trip_idx, 'trip')
            Transfer leg: (from_stop, -1, -1, 'transfer')
    """
    tau = {}
    journey = {}

    # round 0: initialize origin stops
    marked = set()
    for stop in origin_stops:
        tau[stop] = dep_time
        marked.add(stop)

    # apply initial footpath transfers from origin
    for stop in list(marked):
        for to_stop, transfer_time in transfers_map.get(stop, []):
            new_arr = dep_time + transfer_time
            if new_arr < tau.get(to_stop, INF):
                tau[to_stop] = new_arr
                marked.add(to_stop)
                journey[to_stop] = (stop, -1, -1, 'transfer')

    tau_prev = dict(tau)

    for k in range(MAX_ROUNDS):
        # collect routes serving marked stops
        Q = {}
        for stop in marked:
            for pid, pos in stop_routes.get(stop, []):
                if pid not in Q or pos < Q[pid]:
                    Q[pid] = pos

        new_marked = set()

        # scan each route
        for pid, board_pos in Q.items():
            pattern = patterns[pid]
            stops = pattern['stops']
            trips = pattern['trips']
            deps = pattern_deps[pid]

            current_trip_idx = None
            board_stop = None

            for i in range(board_pos, len(stops)):
                stop = stops[i]

                # can we arrive earlier at this stop via current trip?
                if current_trip_idx is not None:
                    arr = trips[current_trip_idx]['times'][i][0]
                    if arr < tau.get(stop, INF):
                        tau[stop] = arr
                        journey[stop] = (board_stop, pid, current_trip_idx, 'trip')
                        new_marked.add(stop)

                # can we board an earlier trip here?
                prev_arr = tau_prev.get(stop, INF)
                if prev_arr < INF:
                    idx = bisect_left(deps[i], prev_arr)
                    if idx < len(trips):
                        if current_trip_idx is None or idx < current_trip_idx:
                            current_trip_idx = idx
                            board_stop = stop

        # apply footpath transfers
        for stop in list(new_marked):
            for to_stop, transfer_time in transfers_map.get(stop, []):
                new_arr = tau[stop] + transfer_time
                if new_arr < tau.get(to_stop, INF):
                    tau[to_stop] = new_arr
                    new_marked.add(to_stop)
                    journey[to_stop] = (stop, -1, -1, 'transfer')

        tau_prev = dict(tau)
        marked = new_marked
        if not marked:
            break

    return tau, journey


def extract_journey(dest_stop, journey, origin_stops):
    """
    Backtrack from dest_stop to extract trip legs.
    Returns list of (pattern_idx, trip_idx, board_stop, alight_stop) or None.
    """
    legs = []
    current = dest_stop
    seen = set()

    while current not in origin_stops:
        if current in seen or current not in journey:
            return None
        seen.add(current)

        bp = journey[current]
        from_stop, pid, tidx, kind = bp

        if kind == 'transfer':
            current = from_stop
        else:
            legs.append((pid, tidx, from_stop, current))
            current = from_stop

    legs.reverse()
    return legs if legs else None


# ---------------------------------------------------------------------------
# Rider assignment
# ---------------------------------------------------------------------------

NUM_WORKERS = 8

# shared read-only data for worker processes
_worker_data = {}

def _worker_init(shared_data_path):
    """Load read-only shared data from temp file (avoids pipe buffer issues on Windows)."""
    with open(shared_data_path, 'rb') as f:
        _worker_data.update(pickle.load(f))


def _route_chunk(chunk):
    """Process a list of (oid, tuples) in a worker. Returns local results."""
    patterns = _worker_data['patterns']
    pattern_deps = _worker_data['pattern_deps']
    stop_routes = _worker_data['stop_routes']
    transfers_map = _worker_data['transfers_map']
    complex_to_stops = _worker_data['complex_to_stops']

    train_boardings = defaultdict(lambda: defaultdict(float))
    train_alightings = defaultdict(lambda: defaultdict(float))
    transfer_events = []
    origin_walk_events = []
    transfer_walk_events = []
    dest_walk_events = []
    assigned = 0
    unassigned = 0

    for oid, tuples in chunk:
        origin_stops = complex_to_stops.get(oid, set())
        if not origin_stops:
            unassigned += len(tuples)
            continue

        by_bin = defaultdict(list)
        for did, dep_time, riders in tuples:
            bin_time = (dep_time // DEP_BIN) * DEP_BIN
            by_bin[bin_time].append((did, riders))

        for bin_time, dest_tuples in by_bin.items():
            tau, journey_bp = raptor_query(
                origin_stops, bin_time, patterns, pattern_deps,
                stop_routes, transfers_map)

            for did, riders in dest_tuples:
                dest_stops = complex_to_stops.get(did, set())
                best_stop = None
                best_arr = INF
                for s in dest_stops:
                    arr = tau.get(s, INF)
                    if arr < best_arr:
                        best_arr = arr
                        best_stop = s

                if best_stop is None or best_arr == INF:
                    unassigned += 1
                    continue

                legs = extract_journey(best_stop, journey_bp, origin_stops)
                if legs is None:
                    unassigned += 1
                    continue

                for pid, tidx, board_stop, alight_stop in legs:
                    trip_id = patterns[pid]['trips'][tidx]['trip_id']
                    board_pos = patterns[pid]['stops'].index(board_stop)
                    alight_pos = patterns[pid]['stops'].index(alight_stop)
                    train_boardings[trip_id][board_pos] += riders
                    train_alightings[trip_id][alight_pos] += riders

                # record origin walk + waiting
                pid_0, tidx_0, board_stop_0, _ = legs[0]
                board_pos_0 = patterns[pid_0]['stops'].index(board_stop_0)
                board_time_0 = patterns[pid_0]['trips'][tidx_0]['times'][board_pos_0][1]
                if board_time_0 > bin_time:
                    route_0 = patterns[pid_0]['route']
                    transfer_events.append((board_stop_0, bin_time, board_time_0, riders, route_0))
                    origin_walk_events.append((board_stop_0, bin_time, riders))

                # record transfer walk + waiting between consecutive legs
                for li in range(len(legs) - 1):
                    pid_a, tidx_a, _, alight_stop = legs[li]
                    pid_b, tidx_b, board_stop, _ = legs[li + 1]
                    alight_pos = patterns[pid_a]['stops'].index(alight_stop)
                    board_pos = patterns[pid_b]['stops'].index(board_stop)
                    alight_time = patterns[pid_a]['trips'][tidx_a]['times'][alight_pos][0]
                    board_time = patterns[pid_b]['trips'][tidx_b]['times'][board_pos][1]
                    if board_time > alight_time:
                        route_b = patterns[pid_b]['route']
                        walk_time = min(60, board_time - alight_time)
                        transfer_walk_events.append((alight_stop, board_stop, alight_time, walk_time, riders))
                        wait_start = alight_time + walk_time
                        if board_time > wait_start:
                            transfer_events.append((board_stop, wait_start, board_time, riders, route_b))

                # record destination walk (riders leaving final station)
                pid_last, tidx_last, _, alight_stop_last = legs[-1]
                alight_pos_last = patterns[pid_last]['stops'].index(alight_stop_last)
                alight_time_last = patterns[pid_last]['trips'][tidx_last]['times'][alight_pos_last][0]
                dest_walk_events.append((alight_stop_last, alight_time_last, riders))

                assigned += 1

    # convert defaultdicts to plain dicts for pickling
    boardings = {tid: dict(stops) for tid, stops in train_boardings.items()}
    alightings = {tid: dict(stops) for tid, stops in train_alightings.items()}
    return (boardings, alightings, transfer_events,
            origin_walk_events, transfer_walk_events, dest_walk_events,
            assigned, unassigned)


def route_all_riders(od_data, complex_to_stops, stop_to_complex, patterns,
                     pattern_deps, stop_routes, transfers_map):
    """
    For each O-D-hour tuple, run RAPTOR to find journey, assign riders to trips.
    Parallelized across NUM_WORKERS processes.
    """
    print(f"Routing riders ({NUM_WORKERS} workers)...")
    t0 = time_module.time()
    random.seed(42)

    # group O-D by origin complex, skewing departure times within each hour
    # so that the rate ramps smoothly between adjacent hours instead of
    # jumping at hour boundaries.
    #
    # For each (oid, did, hour) we sample a departure time from a linear
    # distribution whose density at the start of the hour is proportional to
    # (prev_hour_riders + this_hour_riders) and at the end is proportional to
    # (this_hour_riders + next_hour_riders).  Total riders per OD-hour is
    # unchanged -- only the timing within the hour shifts.

    # aggregate total riders per hour across all OD pairs
    hour_totals = defaultdict(float)
    for (oid, did, hour), riders in od_data.items():
        hour_totals[hour] += riders

    CHUNK_MAX = 600  # split large OD groups into chunks of this size

    def skewed_dep_time(hour, a, b):
        u = random.random()
        if abs(b - a) < 1e-9:
            t = u
        else:
            t = (-a + math.sqrt(a * a + u * (b * b - a * a))) / (b - a)
        return hour * 3600 + int(t * 3599)

    by_origin = defaultdict(list)
    for (oid, did, hour), riders in od_data.items():
        r = hour_totals[hour]
        prev_r = hour_totals.get((hour - 1) % 24, 0.0)
        next_r = hour_totals.get((hour + 1) % 24, 0.0)
        a = prev_r + r               # density weight at start of hour
        b = r + next_r               # density weight at end of hour
        remaining = riders
        while remaining > CHUNK_MAX:
            dep_time = skewed_dep_time(hour, a, b)
            by_origin[oid].append((did, dep_time, CHUNK_MAX))
            remaining -= CHUNK_MAX
        dep_time = skewed_dep_time(hour, a, b)
        by_origin[oid].append((did, dep_time, remaining))

    # split origins into chunks for workers
    origin_items = list(by_origin.items())
    chunk_size = (len(origin_items) + NUM_WORKERS - 1) // NUM_WORKERS
    chunks = [origin_items[i:i + chunk_size]
              for i in range(0, len(origin_items), chunk_size)]

    print(f"  {len(origin_items)} origins split into {len(chunks)} chunks")

    # write shared data to temp file (avoids pipe deadlock on Windows spawn)
    shared = {
        'patterns': patterns,
        'pattern_deps': pattern_deps,
        'stop_routes': stop_routes,
        'transfers_map': transfers_map,
        'complex_to_stops': complex_to_stops,
    }
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
    pickle.dump(shared, tmp)
    tmp.close()
    print(f"  Shared data written to temp file ({os.path.getsize(tmp.name) / 1024 / 1024:.0f} MB)")

    try:
        with mp.Pool(NUM_WORKERS, initializer=_worker_init,
                     initargs=(tmp.name,)) as pool:
            results = pool.map(_route_chunk, chunks)
    finally:
        os.unlink(tmp.name)

    # merge results from all workers
    train_boardings = defaultdict(lambda: defaultdict(float))
    train_alightings = defaultdict(lambda: defaultdict(float))
    transfer_events = []
    origin_walk_events = []
    transfer_walk_events = []
    dest_walk_events = []
    assigned = 0
    unassigned = 0

    for (wb, wa, wte, wowe, wtwe, wdwe, wa_count, wu_count) in results:
        for tid, stops in wb.items():
            for pos, riders in stops.items():
                train_boardings[tid][pos] += riders
        for tid, stops in wa.items():
            for pos, riders in stops.items():
                train_alightings[tid][pos] += riders
        transfer_events.extend(wte)
        origin_walk_events.extend(wowe)
        transfer_walk_events.extend(wtwe)
        dest_walk_events.extend(wdwe)
        assigned += wa_count
        unassigned += wu_count

    elapsed = time_module.time() - t0
    print(f"  Assigned {assigned}/{assigned+unassigned} O-D tuples, "
          f"{unassigned} unassigned ({elapsed:.0f}s)")
    print(f"  {len(transfer_events)} waiting events, "
          f"{len(origin_walk_events)} origin walks, "
          f"{len(transfer_walk_events)} transfer walks, "
          f"{len(dest_walk_events)} dest walks")
    return (train_boardings, train_alightings, transfer_events,
            origin_walk_events, transfer_walk_events, dest_walk_events)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def build_lines(patterns, stop_coords, route_colors, geometry=None):
    """Build subway line geometries for visualization.

    If geometry.json is loaded, uses curved GTFS shape geometries.
    Otherwise falls back to straight lines between stops.
    """
    if geometry:
        # use curved lines from geometry.json (already [lon, lat] order)
        lines = []
        for line in geometry['lines']:
            lines.append({
                'route': line['route'],
                'color': line['color'],
                'coords': [[round(c[0], 5), round(c[1], 5)] for c in line['coords']],
            })
        return lines

    # fallback: straight lines between stops
    best = {}
    for p in patterns:
        key = (p['route'], p['direction'])
        if key not in best or len(p['stops']) > len(best[key]['stops']):
            best[key] = p

    lines = []
    for (route, direction), pattern in best.items():
        coords = []
        for stop_id in pattern['stops']:
            if stop_id in stop_coords:
                lat, lon = stop_coords[stop_id]
                coords.append([round(lon, 5), round(lat, 5)])
        if len(coords) >= 2:
            color = route_colors.get(route, '#808183')
            lines.append({'route': route, 'color': color, 'coords': coords})

    return lines


def build_waiting(transfer_events, complex_info, geometry=None):
    """Aggregate transfer events into per-station waiting timelines.

    Transfer events are keyed by stop_id (platform-level). Uses station-level
    coords from geometry.json when available, falls back to complex coords.
    """
    # build stop_id -> coords and name lookup
    stop_coords_map = {}
    stop_name_map = {}
    if geometry:
        for sid, c in geometry['stop_coords'].items():
            stop_coords_map[sid] = (c['lat'], c['lon'])
        for sid, s in geometry.get('stations', {}).items():
            stop_name_map[sid] = s.get('name', '')

    # collect deltas per stop: (time, +/- riders)
    station_deltas = defaultdict(list)
    station_routes = defaultdict(set)
    for stop_id, arr_time, dep_time, riders, route in transfer_events:
        station_deltas[stop_id].append((arr_time, riders))
        station_deltas[stop_id].append((dep_time, -riders))
        station_routes[stop_id].add(str(route))

    waiting_out = []
    for stop_id, deltas in station_deltas.items():
        deltas.sort()
        timeline = []
        cumulative = 0
        for t, d in deltas:
            cumulative += d
            if timeline and timeline[-1][0] == t:
                timeline[-1][1] = round(cumulative)
            else:
                timeline.append([t, round(cumulative)])

        if max(row[1] for row in timeline) < 0.5:
            continue

        # get coords: prefer station-level from geometry, fall back to complex
        if stop_id in stop_coords_map:
            lat, lon = stop_coords_map[stop_id]
        else:
            # no station coord, skip
            continue

        entry = {
            'lat': round(lat, 5),
            'lon': round(lon, 5),
            'timeline': timeline,
        }
        name = stop_name_map.get(stop_id)
        if name:
            entry['name'] = name
        routes = sorted(station_routes.get(stop_id, []))
        if routes:
            entry['routes'] = routes
        waiting_out.append(entry)

    return waiting_out


ORIGIN_WALK_TIME = 120   # seconds for riders to walk to origin station (~1.7-3.3 m/s over 200-400m)
DEST_WALK_TIME = 120     # seconds for riders to walk away from dest station
WALK_BIN = 60            # aggregate walks into 1-minute bins


def build_walks(origin_walk_events, transfer_walk_events, geometry=None,
                dest_walk_events=None):
    """Build walk animation data: small bubbles moving toward/away from stations.

    Origin walks: random nearby point -> station (arrive at bin_time).
    Transfer walks: alight platform -> board platform.
    Destination walks: station -> random nearby point (depart at alight_time).

    Output: list of [from_lon, from_lat, to_lon, to_lat, t0, t1, riders],
            sorted by t0.
    """
    print("Building walk animations...")

    stop_coords_map = {}
    if geometry:
        for sid, c in geometry['stop_coords'].items():
            stop_coords_map[sid] = (c['lat'], c['lon'])

    # --- aggregate origin walks by (board_stop, 1-min bin) ---
    origin_agg = defaultdict(float)
    for board_stop, arrive_time, riders in origin_walk_events:
        t_bin = (arrive_time // WALK_BIN) * WALK_BIN
        origin_agg[(board_stop, t_bin)] += riders

    # --- aggregate transfer walks by (from_stop, to_stop, 1-min bin) ---
    transfer_agg = defaultdict(lambda: [0.0, 0])  # riders, max walk_time
    for from_stop, to_stop, start_time, walk_time, riders in transfer_walk_events:
        t_bin = (start_time // WALK_BIN) * WALK_BIN
        key = (from_stop, to_stop, t_bin)
        transfer_agg[key][0] += riders
        transfer_agg[key][1] = max(transfer_agg[key][1], walk_time)

    # --- aggregate dest walks by (alight_stop, 1-min bin) ---
    dest_agg = defaultdict(float)
    for alight_stop, alight_time, riders in (dest_walk_events or []):
        t_bin = (alight_time // WALK_BIN) * WALK_BIN
        dest_agg[(alight_stop, t_bin)] += riders

    origin_walks = []
    transfer_walks = []
    dest_walks = []
    rng = random.Random(99)

    # --- origin walks ---
    for (stop_id, t_bin), riders in origin_agg.items():
        if riders < 10:
            continue
        if stop_id not in stop_coords_map:
            continue
        to_lat, to_lon = stop_coords_map[stop_id]
        # random direction and distance per (stop, time) for variety
        angle = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(0.002, 0.004)  # ~200-400m
        from_lat = to_lat + dist * math.cos(angle)
        from_lon = to_lon + dist * math.sin(angle) / math.cos(math.radians(to_lat))
        t1 = t_bin
        t0 = t1 - ORIGIN_WALK_TIME
        origin_walks.append([round(from_lon, 5), round(from_lat, 5),
                             round(to_lon, 5), round(to_lat, 5),
                             t0, t1, round(riders)])

    # --- transfer walks ---
    for (from_stop, to_stop, t_bin), (riders, walk_time) in transfer_agg.items():
        if riders < 20:
            continue
        if from_stop not in stop_coords_map or to_stop not in stop_coords_map:
            continue
        from_lat, from_lon = stop_coords_map[from_stop]
        to_lat, to_lon = stop_coords_map[to_stop]
        t0 = t_bin
        t1 = t0 + walk_time
        transfer_walks.append([round(from_lon, 5), round(from_lat, 5),
                               round(to_lon, 5), round(to_lat, 5),
                               t0, t1, round(riders)])

    # --- destination walks ---
    for (stop_id, t_bin), riders in dest_agg.items():
        if riders < 10:
            continue
        if stop_id not in stop_coords_map:
            continue
        from_lat, from_lon = stop_coords_map[stop_id]
        angle = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(0.002, 0.004)  # ~200-400m
        to_lat = from_lat + dist * math.cos(angle)
        to_lon = from_lon + dist * math.sin(angle) / math.cos(math.radians(from_lat))
        t0 = t_bin
        t1 = t0 + DEST_WALK_TIME
        dest_walks.append([round(from_lon, 5), round(from_lat, 5),
                           round(to_lon, 5), round(to_lat, 5),
                           t0, t1, round(riders)])

    origin_walks.sort(key=lambda w: w[4])
    transfer_walks.sort(key=lambda w: w[4])
    dest_walks.sort(key=lambda w: w[4])

    print(f"  {len(origin_walks)} origin walk animations, "
          f"{len(transfer_walks)} transfer walk animations, "
          f"{len(dest_walks)} dest walk animations")
    return origin_walks, transfer_walks, dest_walks


def build_output(patterns, train_boardings, train_alightings, transfer_events,
                 route_colors, stop_coords, complex_info, geometry=None,
                 shapes=None, trip_shape=None,
                 origin_walk_events=None, transfer_walk_events=None,
                 dest_walk_events=None):
    """Build trains.json output with train timelines and line geometries."""
    print("Building output...")

    # load per-route station coords from geometry.json if available
    geo_stop_coords = {}
    geo_route_coords = {}
    if geometry:
        for sid, c in geometry['stop_coords'].items():
            geo_stop_coords[sid] = (c['lat'], c['lon'])
        for key, c in geometry['stop_route_coords'].items():
            geo_route_coords[key] = (c['lat'], c['lon'])
        print(f"  Using station-level coords from {GEOMETRY_JSON}")

    def get_stop_coord(stop_id, route=None):
        """Get best coords for a stop: per-route > station-level > GTFS parent."""
        if route:
            key = f"{stop_id}:{route}"
            if key in geo_route_coords:
                return geo_route_coords[key]
        if stop_id in geo_stop_coords:
            return geo_stop_coords[stop_id]
        return stop_coords.get(stop_id)

    # pre-compute shape segments for curved train animation
    pattern_segments = {}
    if shapes and trip_shape:
        pattern_segments = precompute_shape_segments(
            patterns, shapes, trip_shape, get_stop_coord)

    # map trip_id -> (pattern_idx, trip_idx) for looking up stops/times
    trip_to_pattern = {}
    for pid, pattern in enumerate(patterns):
        for tidx, trip in enumerate(pattern['trips']):
            trip_to_pattern[trip['trip_id']] = (pid, tidx)

    # merge wrapped midnight trips back onto originals
    for trip_id in list(train_boardings.keys()):
        if trip_id.startswith('_wrap_'):
            orig_id = trip_id[6:]
            for pos, riders in train_boardings[trip_id].items():
                train_boardings.setdefault(orig_id, {})[pos] = \
                    train_boardings.get(orig_id, {}).get(pos, 0) + riders
            for pos, riders in train_alightings.get(trip_id, {}).items():
                train_alightings.setdefault(orig_id, {})[pos] = \
                    train_alightings.get(orig_id, {}).get(pos, 0) + riders
            del train_boardings[trip_id]
            if trip_id in train_alightings:
                del train_alightings[trip_id]

    trains_out = []
    total_waypoints = 0
    for trip_id in train_boardings:
        pid, tidx = trip_to_pattern[trip_id]
        pattern = patterns[pid]
        trip = pattern['trips'][tidx]
        route = pattern['route']

        riders_on = 0
        timeline = []
        for i, stop_id in enumerate(pattern['stops']):
            riders_on += train_boardings[trip_id].get(i, 0)
            riders_on -= train_alightings[trip_id].get(i, 0)
            riders_on = max(0, riders_on)

            coord = get_stop_coord(stop_id, route)
            if coord:
                lat, lon = coord
                t_secs = trip['times'][i][0]
                timeline.append([t_secs, round(lat, 5), round(lon, 5), round(riders_on)])

        if not timeline or max(row[3] for row in timeline) < 0.5:
            continue

        # expand timeline with shape waypoints between stations
        segments = pattern_segments.get(pid)
        if segments:
            expanded = [timeline[0]]
            for i in range(len(timeline) - 1):
                t0, lat0, lon0, riders0 = timeline[i]
                t1, lat1, lon1, riders1 = timeline[i + 1]

                # pattern stop index for this timeline entry
                # (timeline may skip stops without coords, but usually 1:1)
                seg = segments[i] if i < len(segments) else None
                if seg and len(seg) > 2:
                    # compute cumulative distances along sub-polyline
                    dists = [0.0]
                    for j in range(1, len(seg)):
                        d = math.sqrt((seg[j][0] - seg[j - 1][0]) ** 2 +
                                      (seg[j][1] - seg[j - 1][1]) ** 2)
                        dists.append(dists[-1] + d)
                    total_dist = dists[-1]

                    if total_dist > 1e-10:
                        # insert intermediate waypoints (skip first/last = stations)
                        travel_time = t1 - t0
                        for j in range(1, len(seg) - 1):
                            frac = dists[j] / total_dist
                            t = t0 + frac * travel_time
                            # waypoints: 3 elements (no riders = no dwell)
                            expanded.append([round(t), round(seg[j][0], 5), round(seg[j][1], 5)])
                            total_waypoints += 1

                expanded.append(timeline[i + 1])
            timeline = expanded

        color = route_colors.get(route, '#808183')
        trains_out.append({
            'route': str(route),
            'color': color,
            'timeline': timeline,
        })

    print(f"  {len(trains_out)} trains with riders")
    if total_waypoints:
        print(f"  {total_waypoints} shape waypoints inserted")

    lines = build_lines(patterns, stop_coords, route_colors, geometry)
    print(f"  {len(lines)} subway line geometries")

    waiting = build_waiting(transfer_events, complex_info, geometry)
    print(f"  {len(waiting)} stations with waiting")

    origin_walks = []
    transfer_walks = []
    dest_walks = []
    if origin_walk_events is not None or transfer_walk_events is not None or dest_walk_events is not None:
        origin_walks, transfer_walks, dest_walks = build_walks(
            origin_walk_events or [], transfer_walk_events or [], geometry,
            dest_walk_events or [])

    # stations dict for the UI
    stations = {}
    for cid, info in complex_info.items():
        stations[str(cid)] = info

    out = {'stations': stations, 'lines': lines, 'trains': trains_out,
           'waiting': waiting}
    if transfer_walks:
        out['transfer_walks'] = transfer_walks
    if origin_walks:
        out['origin_walks'] = origin_walks
    if dest_walks:
        out['dest_walks'] = dest_walks
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build NYC subway ridership animation data.")
    parser.add_argument('--sample', type=float, default=DEFAULT_SAMPLE_RATE,
                        help=f"O-D sampling rate, 0.0-1.0 (default: {DEFAULT_SAMPLE_RATE})")
    parser.add_argument('--hours', type=str, default=None,
                        help="Hour range to include, e.g. '7-12' (default: all hours)")
    args = parser.parse_args()

    global SAMPLE_RATE, HOUR_RANGE
    SAMPLE_RATE = args.sample
    if args.hours:
        lo, hi = args.hours.split('-')
        HOUR_RANGE = (int(lo), int(hi))
    else:
        HOUR_RANGE = None

    stop_times, trips, route_colors = load_gtfs()
    stop_to_complex = load_stop_to_complex()
    stop_coords = load_stop_coords()
    od_data = load_od_data()
    complex_info = load_complex_info(stop_to_complex, stop_coords)
    geometry = load_geometry()
    shapes = load_shapes() if geometry else None
    trip_shape = dict(zip(trips.trip_id, trips.shape_id)) if shapes else {}

    patterns, pattern_deps = build_route_patterns(stop_times)
    stop_routes = build_stop_routes(patterns)
    transfers_map, complex_to_stops = build_transfers(stop_to_complex, geometry)

    (train_boardings, train_alightings, transfer_events,
     origin_walk_events, transfer_walk_events, dest_walk_events) = route_all_riders(
        od_data, complex_to_stops, stop_to_complex, patterns, pattern_deps,
        stop_routes, transfers_map)

    out = build_output(patterns, train_boardings, train_alightings,
                       transfer_events, route_colors, stop_coords, complex_info,
                       geometry, shapes, trip_shape,
                       origin_walk_events, transfer_walk_events,
                       dest_walk_events)

    print(f"Writing {OUT_JSON}...")
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f)

    size_mb = os.path.getsize(OUT_JSON) / 1024 / 1024
    print(f"Done! {OUT_JSON} is {size_mb:.1f} MB")


if __name__ == '__main__':
    main()
