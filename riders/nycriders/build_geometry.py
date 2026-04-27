"""
Build improved geometry for NYC subway visualization.

Replaces straight-line station-to-station segments with:
1. Curved track geometries from GTFS shapes
2. Station-level coordinates (not complex-level) from MTA stations CSV
3. Stations snapped to their nearest line geometry

Outputs geometry.json for use by build.py / index.html.
"""
import csv
import json
import math
from collections import defaultdict

GTFS_DIR = "data/gtfs_subway"
STATIONS_CSV = "../../data/nystreets/MTA_Subway_Stations.csv"
OUT_JSON = "geometry.json"
SERVICE_ID = "Weekday"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_shapes():
    """Load GTFS shapes into dict of shape_id -> [(lat, lon), ...]."""
    print("Loading GTFS shapes...")
    shapes = defaultdict(list)
    with open(f"{GTFS_DIR}/shapes.txt") as f:
        for row in csv.DictReader(f):
            shapes[row['shape_id']].append((
                int(row['shape_pt_sequence']),
                float(row['shape_pt_lat']),
                float(row['shape_pt_lon']),
            ))
    # sort by sequence, keep only coords
    out = {}
    for sid, pts in shapes.items():
        pts.sort(key=lambda p: p[0])
        out[sid] = [(lat, lon) for _, lat, lon in pts]
    print(f"  {len(out)} shapes, {sum(len(v) for v in out.values())} total points")
    return out


def load_trips():
    """Load weekday trips with route and shape mapping."""
    print("Loading GTFS trips...")
    trips = {}
    with open(f"{GTFS_DIR}/trips.txt") as f:
        for row in csv.DictReader(f):
            if row['service_id'] != SERVICE_ID:
                continue
            trips[row['trip_id']] = {
                'route_id': str(row['route_id']),
                'shape_id': row['shape_id'],
                'direction_id': int(row['direction_id']),
            }
    print(f"  {len(trips)} weekday trips")
    return trips


def load_route_colors():
    """Load route colors from GTFS routes.txt."""
    colors = {}
    with open(f"{GTFS_DIR}/routes.txt") as f:
        for row in csv.DictReader(f):
            colors[str(row['route_id'])] = f"#{row['route_color']}"
    return colors


def load_mta_stations():
    """Load station-level data from MTA stations CSV.

    Returns dict keyed by GTFS Stop ID with per-station coords, complex ID,
    and which daytime routes serve it.
    """
    print("Loading MTA station data...")
    stations = {}
    with open(STATIONS_CSV) as f:
        for row in csv.DictReader(f):
            gtfs_id = str(row['GTFS Stop ID']).strip()
            stations[gtfs_id] = {
                'name': row['Stop Name'],
                'station_id': int(row['Station ID']),
                'complex_id': int(row['Complex ID']),
                'lat': float(row['GTFS Latitude']),
                'lon': float(row['GTFS Longitude']),
                'routes': row['Daytime Routes'].split(),
            }
    print(f"  {len(stations)} stations")
    return stations


def load_stop_times_stops():
    """Load stop_times to find which parent stops each trip visits (in order)."""
    print("Loading stop_times for stop-shape mapping...")
    trip_stops = defaultdict(list)
    with open(f"{GTFS_DIR}/stop_times.txt") as f:
        for row in csv.DictReader(f):
            trip_stops[row['trip_id']].append((
                int(row['stop_sequence']),
                row['stop_id'],
            ))
    # sort by sequence, strip N/S to get parent stop
    out = {}
    for tid, stops in trip_stops.items():
        stops.sort(key=lambda s: s[0])
        out[tid] = [sid.rstrip('NS') for _, sid in stops]
    print(f"  {len(out)} trips with stop sequences")
    return out


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def snap_to_polyline(lat, lon, polyline):
    """Snap a point to the nearest position on a polyline.

    Returns (snapped_lat, snapped_lon, distance, segment_index, t).
    Distance is approximate (Euclidean in degree space).
    """
    best_dist = float('inf')
    best_lat, best_lon = lat, lon
    best_seg, best_t = 0, 0.0

    for i in range(len(polyline) - 1):
        lat1, lon1 = polyline[i]
        lat2, lon2 = polyline[i + 1]

        dx = lat2 - lat1
        dy = lon2 - lon1
        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq < 1e-14:
            t = 0.0
        else:
            t = ((lat - lat1) * dx + (lon - lon1) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))

        proj_lat = lat1 + t * dx
        proj_lon = lon1 + t * dy
        dist = (lat - proj_lat) ** 2 + (lon - proj_lon) ** 2

        if dist < best_dist:
            best_dist = dist
            best_lat = proj_lat
            best_lon = proj_lon
            best_seg = i
            best_t = t

    return best_lat, best_lon, math.sqrt(best_dist), best_seg, best_t


def polyline_position(polyline, seg_idx, t):
    """Convert (segment_index, t) to a scalar 'distance along' for ordering."""
    dist = 0.0
    for i in range(seg_idx):
        lat1, lon1 = polyline[i]
        lat2, lon2 = polyline[i + 1]
        dist += math.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)
    if seg_idx < len(polyline) - 1:
        lat1, lon1 = polyline[seg_idx]
        lat2, lon2 = polyline[seg_idx + 1]
        dist += t * math.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)
    return dist


def extract_subline(polyline, seg1, t1, seg2, t2):
    """Extract the portion of a polyline between two snapped positions.

    Returns list of (lat, lon) from (seg1, t1) to (seg2, t2).
    """
    if seg1 > seg2 or (seg1 == seg2 and t1 > t2):
        # reverse direction - swap and reverse result
        pts = extract_subline(polyline, seg2, t2, seg1, t1)
        pts.reverse()
        return pts

    points = []
    # start point
    lat1, lon1 = polyline[seg1]
    lat2, lon2 = polyline[seg1 + 1] if seg1 + 1 < len(polyline) else polyline[seg1]
    points.append((lat1 + t1 * (lat2 - lat1), lon1 + t1 * (lon2 - lon1)))

    # intermediate vertices
    for i in range(seg1 + 1, min(seg2 + 1, len(polyline))):
        points.append(polyline[i])

    # end point (if not already at a vertex)
    if seg2 < len(polyline) - 1:
        lat1, lon1 = polyline[seg2]
        lat2, lon2 = polyline[seg2 + 1]
        end = (lat1 + t2 * (lat2 - lat1), lon1 + t2 * (lon2 - lon1))
        points.append(end)

    return points


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_route_shapes(shapes, trips, trip_stops):
    """Map each (route, direction) to representative shape_ids, including branches.

    Detects branches by finding shapes with stops not covered by longer shapes.
    """
    route_dir_shapes = defaultdict(set)
    shape_to_stops = defaultdict(set)

    for trip_id, trip in trips.items():
        key = (trip['route_id'], trip['direction_id'])
        sid = trip['shape_id']
        if sid in shapes:
            route_dir_shapes[key].add(sid)
            if trip_id in trip_stops:
                shape_to_stops[sid] |= set(trip_stops[trip_id])

    best = {}
    for (route, direction), shape_ids in route_dir_shapes.items():
        # sort by length (longest first)
        sorted_ids = sorted(shape_ids, key=lambda sid: len(shapes[sid]), reverse=True)

        selected = []
        covered_stops = set()
        for sid in sorted_ids:
            stops = shape_to_stops.get(sid, set())
            unique = stops - covered_stops
            # keep if it has enough unique stops (branch) or it's the first
            if not selected or len(unique) >= 3:
                selected.append(sid)
                covered_stops |= stops

        best[(route, direction)] = selected

    total = sum(len(v) for v in best.values())
    print(f"  {len(best)} route-direction pairs, {total} representative shapes")
    return best, route_dir_shapes


def map_stops_to_shapes(trips, trip_stops, shapes):
    """For each parent_stop, find which shape_ids visit it.

    Returns dict: parent_stop_id -> set of shape_ids.
    """
    stop_shapes = defaultdict(set)
    for trip_id, trip in trips.items():
        if trip_id not in trip_stops:
            continue
        shape_id = trip['shape_id']
        if shape_id not in shapes:
            continue
        for parent_stop in trip_stops[trip_id]:
            stop_shapes[parent_stop].add(shape_id)
    return dict(stop_shapes)


def snap_stations_to_lines(mta_stations, shapes, stop_shapes, best_shapes, trips):
    """Snap each MTA station to its route's representative shape geometry.

    Returns dict: gtfs_stop_id -> {lat, lon, snapped_lat, snapped_lon, complex_id, name,
                                    route, shape_id, seg_idx, t}
    """
    print("Snapping stations to line geometries...")

    # collect all representative shape_ids into a set for priority snapping
    representative_shapes = set()
    for shape_list in best_shapes.values():
        representative_shapes.update(shape_list)

    snapped = {}
    snap_distances = []

    for gtfs_id, station in mta_stations.items():
        lat, lon = station['lat'], station['lon']
        candidate_shapes = stop_shapes.get(gtfs_id, set())

        if not candidate_shapes:
            # no trip visits this stop - use original coords
            snapped[gtfs_id] = {
                'name': station['name'],
                'complex_id': station['complex_id'],
                'lat': lat,
                'lon': lon,
                'snapped_lat': lat,
                'snapped_lon': lon,
                'shape_id': None,
            }
            continue

        # find the best shape to snap to: prefer representative shapes
        best_dist = float('inf')
        best_result = None
        best_shape = None

        for shape_id in candidate_shapes:
            if shape_id not in shapes:
                continue
            polyline = shapes[shape_id]
            s_lat, s_lon, dist, seg, t = snap_to_polyline(lat, lon, polyline)

            is_representative = shape_id in representative_shapes
            adj_dist = dist * (0.5 if is_representative else 1.0)

            if adj_dist < best_dist:
                best_dist = adj_dist
                best_result = (s_lat, s_lon, seg, t)
                best_shape = shape_id

        if best_result:
            s_lat, s_lon, seg, t = best_result
            snap_distances.append(best_dist)
            snapped[gtfs_id] = {
                'name': station['name'],
                'complex_id': station['complex_id'],
                'lat': lat,
                'lon': lon,
                'snapped_lat': s_lat,
                'snapped_lon': s_lon,
                'shape_id': best_shape,
                'seg_idx': seg,
                't': t,
            }
        else:
            snapped[gtfs_id] = {
                'name': station['name'],
                'complex_id': station['complex_id'],
                'lat': lat,
                'lon': lon,
                'snapped_lat': lat,
                'snapped_lon': lon,
                'shape_id': None,
            }

    if snap_distances:
        # convert rough degree distance to meters (1 degree ~ 111km)
        meters = [d * 111_000 for d in snap_distances]
        print(f"  Snapped {len(snap_distances)} stations")
        print(f"  Snap distance: median {sorted(meters)[len(meters)//2]:.0f}m, "
              f"max {max(meters):.0f}m, mean {sum(meters)/len(meters):.0f}m")

    return snapped


def build_parent_stop_coords(mta_stations, snapped, stop_shapes, trips, trip_stops):
    """Build a mapping from parent_stop_id to snapped coordinates.

    When a parent stop appears on multiple routes, we need per-route coords.
    Returns:
        stop_coords: parent_stop_id -> (lat, lon)  (best single coord)
        stop_route_coords: (parent_stop_id, route_id) -> (lat, lon)
    """
    # map parent_stop -> gtfs_stop_id from MTA stations CSV
    # a parent_stop like "127" maps to GTFS Stop ID "127" in the CSV
    gtfs_to_parent = {}
    for gtfs_id in mta_stations:
        # parent stop = gtfs_id (the MTA CSV uses the same IDs as GTFS parent stations)
        gtfs_to_parent[gtfs_id] = gtfs_id

    # figure out which route each parent_stop is on, per trip
    stop_route = defaultdict(set)
    for trip_id, trip in trips.items():
        if trip_id not in trip_stops:
            continue
        for parent_stop in trip_stops[trip_id]:
            stop_route[parent_stop].add(trip['route_id'])

    stop_coords = {}
    stop_route_coords = {}

    for gtfs_id, info in snapped.items():
        parent_stop = gtfs_id
        lat, lon = info['snapped_lat'], info['snapped_lon']
        # use snapped coord as default for this parent stop
        stop_coords[parent_stop] = (lat, lon)

        # also store per-route
        for route in mta_stations[gtfs_id].get('routes', []):
            stop_route_coords[(parent_stop, route)] = (lat, lon)

    return stop_coords, stop_route_coords


def build_output(shapes, best_shapes, route_colors, snapped, mta_stations,
                 stop_coords, stop_route_coords):
    """Build the output geometry.json."""
    print("Building output...")

    # 1. Line geometries from representative shapes (including branches)
    lines = []
    for (route, direction), shape_ids in sorted(best_shapes.items()):
        color = route_colors.get(route, '#808183')
        for shape_id in shape_ids:
            polyline = shapes[shape_id]
            lines.append({
                'route': str(route),
                'direction': direction,
                'color': color,
                'shape_id': shape_id,
                'coords': [[lon, lat] for lat, lon in polyline],
            })
    print(f"  {len(lines)} line geometries")

    # 2. Station data with snapped coordinates
    stations = {}
    for gtfs_id, info in snapped.items():
        stations[gtfs_id] = {
            'name': info['name'],
            'complex_id': info['complex_id'],
            'lat': round(info['snapped_lat'], 6),
            'lon': round(info['snapped_lon'], 6),
            'original_lat': round(info['lat'], 6),
            'original_lon': round(info['lon'], 6),
        }

    # 3. Parent stop -> snapped coords (for build.py integration)
    # This replaces stop_coords in build.py
    stop_coords_out = {}
    for parent_stop, (lat, lon) in stop_coords.items():
        stop_coords_out[parent_stop] = {
            'lat': round(lat, 6),
            'lon': round(lon, 6),
        }

    # 4. Per-route stop coords (for when a stop serves multiple routes)
    stop_route_coords_out = {}
    for (parent_stop, route), (lat, lon) in stop_route_coords.items():
        key = f"{parent_stop}:{route}"
        stop_route_coords_out[key] = {
            'lat': round(lat, 6),
            'lon': round(lon, 6),
        }

    # 5. Complex info (station-level, not complex-level)
    # Group stations by complex for reference
    complexes = defaultdict(list)
    for gtfs_id, info in snapped.items():
        complexes[info['complex_id']].append({
            'gtfs_id': gtfs_id,
            'name': info['name'],
            'lat': round(info['snapped_lat'], 6),
            'lon': round(info['snapped_lon'], 6),
        })

    return {
        'lines': lines,
        'stations': stations,
        'stop_coords': stop_coords_out,
        'stop_route_coords': stop_route_coords_out,
        'complexes': {str(k): v for k, v in complexes.items()},
    }


def main():
    shapes = load_shapes()
    trips = load_trips()
    route_colors = load_route_colors()
    mta_stations = load_mta_stations()
    trip_stops = load_stop_times_stops()

    best_shapes, all_route_shapes = build_route_shapes(shapes, trips, trip_stops)
    stop_shapes = map_stops_to_shapes(trips, trip_stops, shapes)

    snapped = snap_stations_to_lines(mta_stations, shapes, stop_shapes, best_shapes, trips)
    stop_coords, stop_route_coords = build_parent_stop_coords(
        mta_stations, snapped, stop_shapes, trips, trip_stops)

    out = build_output(shapes, best_shapes, route_colors, snapped, mta_stations,
                       stop_coords, stop_route_coords)

    print(f"Writing {OUT_JSON}...")
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f)

    import os
    size_mb = os.path.getsize(OUT_JSON) / 1024 / 1024
    print(f"Done! {OUT_JSON} is {size_mb:.1f} MB")

    # summary
    print(f"\nSummary:")
    print(f"  {len(out['lines'])} line geometries (curved, from GTFS shapes)")
    print(f"  {len(out['stations'])} stations (snapped to lines)")
    print(f"  {len(out['stop_coords'])} parent stop coords")
    print(f"  {len(out['stop_route_coords'])} per-route stop coords")
    print(f"  {len(out['complexes'])} station complexes")


if __name__ == '__main__':
    main()
