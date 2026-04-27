"""
Build train-based animation data from GTFS schedules + O-D ridership.

Instead of animating individual trips, we:
1. Parse GTFS to get every train run (trip) with its stop times
2. For each O-D pair, figure out which train(s) riders would take
3. Aggregate riders onto train runs
4. Output: for each train run, a timeline of (time, station, riders_on_board)

Output: trains.json
"""
import pandas as pd
import numpy as np
import json
import csv
import time
import os
import random
from collections import defaultdict

GTFS_DIR = "data/gtfs_subway"
STATIONS_CSV = "../../data/nystreets/MTA_Subway_Stations.csv"
OD_CSV = "data/od_wednesday_sep.csv"
PATHS_JSON = "data.json"  # existing pathfinding output (for route info)
OUT_JSON = "trains.json"

SERVICE_ID = "Weekday"  # Wednesday


def load_gtfs():
    """Load and parse GTFS data for weekday subway service."""
    print("Loading GTFS data...")

    trips = pd.read_csv(f"{GTFS_DIR}/trips.txt")
    trips = trips[trips.service_id == SERVICE_ID]
    print(f"  {len(trips)} weekday trips")

    stop_times = pd.read_csv(f"{GTFS_DIR}/stop_times.txt")
    stop_times = stop_times[stop_times.trip_id.isin(trips.trip_id)]
    # parse times to seconds since midnight
    def time_to_secs(t):
        parts = t.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    stop_times['arr_secs'] = stop_times.arrival_time.apply(time_to_secs)
    stop_times['dep_secs'] = stop_times.departure_time.apply(time_to_secs)
    # extract parent stop_id (strip N/S suffix)
    stop_times['parent_stop'] = stop_times.stop_id.str.replace(r'[NS]$', '', regex=True)
    stop_times = stop_times.sort_values(['trip_id', 'stop_sequence'])
    print(f"  {len(stop_times)} stop_time entries")

    routes = pd.read_csv(f"{GTFS_DIR}/routes.txt")
    route_colors = dict(zip(routes.route_id, routes.route_color.apply(lambda c: f'#{c}')))

    # merge route_id onto stop_times
    trip_route = dict(zip(trips.trip_id, trips.route_id))
    stop_times['route_id'] = stop_times.trip_id.map(trip_route)

    return stop_times, trips, route_colors


def load_stop_to_complex():
    """Build mapping from GTFS parent stop ID -> station complex ID."""
    stations = pd.read_csv(STATIONS_CSV)
    return dict(zip(stations['GTFS Stop ID'].astype(str),
                    stations['Complex ID'].astype(int)))


SAMPLE_RATE = 0.1  # use 1.0 for full data

def load_od_data():
    """Load O-D ridership, keyed by (origin_complex, dest_complex, hour)."""
    print(f"Loading O-D data (sample rate: {SAMPLE_RATE})...")
    od = defaultdict(float)
    random.seed(123)
    with open(OD_CSV) as f:
        for row in csv.DictReader(f):
            if random.random() > SAMPLE_RATE:
                continue
            oid = int(row['origin_station_complex_id'])
            did = int(row['destination_station_complex_id'])
            hour = int(row['hour_of_day'])
            riders = float(row['estimated_average_ridership'])
            if oid != did:
                od[(oid, did, hour)] += riders
    print(f"  {len(od)} O-D-hour tuples")
    return od


def build_train_runs(stop_times, stop_to_complex):
    """
    Build train runs: for each trip, a list of (time_secs, complex_id).
    """
    print("Building train runs...")
    runs = {}  # trip_id -> { route, stops: [(time_secs, complex_id), ...] }

    for trip_id, group in stop_times.groupby('trip_id'):
        route = group.iloc[0]['route_id']
        stops = []
        for _, row in group.iterrows():
            cid = stop_to_complex.get(row['parent_stop'])
            if cid is not None:
                stops.append((int(row['arr_secs']), int(cid)))
        if len(stops) >= 2:
            runs[trip_id] = {'route': route, 'stops': stops}

    print(f"  {len(runs)} valid train runs")
    return runs


def assign_riders(runs, od_data, stop_to_complex):
    """
    For each O-D-hour tuple, pick a random departure time within the hour,
    then find the next train at the origin that also stops at the destination.
    All riders for that tuple board that single train.
    """
    print("Assigning riders to trains...")
    t0 = time.time()
    random.seed(42)

    # index: complex_id -> [(trip_id, stop_idx, time_secs)] sorted by time
    station_trains = defaultdict(list)
    for trip_id, run in runs.items():
        for idx, (t, cid) in enumerate(run['stops']):
            station_trains[cid].append((trip_id, idx, t))
    for cid in station_trains:
        station_trains[cid].sort(key=lambda x: x[2])

    train_boardings = defaultdict(lambda: defaultdict(float))
    train_alightings = defaultdict(lambda: defaultdict(float))
    assigned = 0
    unassigned = 0

    for (oid, did, hour), riders in od_data.items():
        # random departure time within the hour
        depart_time = hour * 3600 + random.randint(0, 3599)

        # find trains at origin departing at or after depart_time
        candidates = station_trains.get(oid, [])
        # binary search for first train >= depart_time
        lo, hi = 0, len(candidates)
        while lo < hi:
            mid = (lo + hi) // 2
            if candidates[mid][2] < depart_time:
                lo = mid + 1
            else:
                hi = mid

        # check trains from this point forward, find first that stops at dest
        found = False
        for j in range(lo, min(lo + 20, len(candidates))):  # check up to 20 upcoming trains
            tid, o_idx, o_time = candidates[j]
            run = runs[tid]
            for d_idx in range(o_idx + 1, len(run['stops'])):
                if run['stops'][d_idx][1] == did:
                    train_boardings[tid][o_idx] += riders
                    train_alightings[tid][d_idx] += riders
                    assigned += 1
                    found = True
                    break
            if found:
                break

        if not found:
            unassigned += 1

    print(f"  Assigned {assigned}/{assigned+unassigned} O-D tuples, {unassigned} unassigned ({time.time()-t0:.0f}s)")
    return train_boardings, train_alightings


def build_output(runs, train_boardings, train_alightings, route_colors, complex_info):
    """
    Build JSON output: each train run with its timeline of rider counts.
    Only include trains that actually carry riders.
    """
    print("Building output...")

    # load station info for coordinates
    trains_out = []
    for trip_id, run in runs.items():
        if trip_id not in train_boardings:
            continue

        boardings = train_boardings[trip_id]
        alightings = train_alightings[trip_id]

        # compute riders on board at each stop
        riders_on = 0
        timeline = []
        for idx, (t_secs, cid) in enumerate(run['stops']):
            riders_on += boardings.get(idx, 0)
            riders_on -= alightings.get(idx, 0)
            riders_on = max(0, riders_on)
            if cid in complex_info:
                station = complex_info[cid]
                timeline.append([
                    t_secs,
                    station['lat'],
                    station['lon'],
                    round(riders_on, 1),
                ])

        if not timeline or max(row[3] for row in timeline) < 0.5:
            continue

        color = route_colors.get(run['route'], '#808183')
        trains_out.append({
            'route': run['route'],
            'color': color,
            'timeline': timeline,  # [[secs, lat, lon, riders_on_board], ...]
        })

    print(f"  {len(trains_out)} trains with riders")
    return trains_out


def main():
    stop_times, trips, route_colors = load_gtfs()
    stop_to_complex = load_stop_to_complex()
    od_data = load_od_data()
    runs = build_train_runs(stop_times, stop_to_complex)

    # load station info from existing data.json
    with open(PATHS_JSON) as f:
        existing = json.load(f)
    complex_info = {}
    for cid_str, info in existing['stations'].items():
        complex_info[int(cid_str)] = info

    train_boardings, train_alightings = assign_riders(runs, od_data, stop_to_complex)
    trains_out = build_output(runs, train_boardings, train_alightings, route_colors, complex_info)

    # also include segments for drawing subway lines
    out = {
        'stations': existing['stations'],
        'segments': existing['segments'],
        'trains': trains_out,
    }

    print(f"Writing {OUT_JSON}...")
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f)

    size_mb = os.path.getsize(OUT_JSON) / 1024 / 1024
    print(f"Done! {OUT_JSON} is {size_mb:.1f} MB")


if __name__ == '__main__':
    main()
