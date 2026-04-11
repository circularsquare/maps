"""
Check inter-stop train speeds from GTFS timetables.
Prints a summary distribution, flags segments where trains are unrealistically
fast, and proposes timetable fixes by stealing time from slower neighbors.

Usage:
  python check_speeds.py              # dry-run: show proposed fixes
  python check_speeds.py --apply      # apply fixes to stop_times.txt
"""
import argparse
import math
import pandas as pd
from collections import defaultdict

GTFS_DIR = "data/gtfs_subway"
SERVICE_ID = "Weekday"

SPEED_THRESHOLD_KMH = 85
DWELL_TIME = 20
ACCEL_DECEL_PENALTY = 20  # 20s accel + 20s decel at half speed = 20s lost
OVERHEAD = DWELL_TIME + ACCEL_DECEL_PENALTY


def time_to_secs(t):
    parts = t.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def secs_to_time(s):
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def approx_distance_m(lat1, lon1, lat2, lon2):
    dlat = (lat2 - lat1) * 111_000
    dlon = (lon2 - lon1) * 111_000 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.sqrt(dlat * dlat + dlon * dlon)


def cruise_speed_kmh(dist_m, dt):
    cruise_dt = dt - OVERHEAD
    if cruise_dt <= 0:
        return 9999.0
    return (dist_m / cruise_dt) * 3.6


def time_needed_for_speed(dist_m, target_kmh):
    """Total inter-stop time needed so cruise speed = target_kmh."""
    cruise_dt = dist_m / (target_kmh / 3.6)
    return cruise_dt + OVERHEAD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true',
                        help='Apply fixes to stop_times.txt')
    args = parser.parse_args()

    # load stops
    stops = pd.read_csv(f"{GTFS_DIR}/stops.txt")
    parent = stops[stops.location_type == 1]
    stop_coords = {str(row.stop_id): (row.stop_lat, row.stop_lon)
                   for _, row in parent.iterrows()}

    # load trips + stop_times
    trips = pd.read_csv(f"{GTFS_DIR}/trips.txt")
    trips = trips[trips.service_id == SERVICE_ID]
    trip_route = dict(zip(trips.trip_id, trips.route_id))

    st = pd.read_csv(f"{GTFS_DIR}/stop_times.txt")
    st = st[st.trip_id.isin(trips.trip_id)]
    st['arr_secs'] = st.arrival_time.apply(time_to_secs)
    st['dep_secs'] = st.departure_time.apply(time_to_secs)
    st['parent_stop'] = st.stop_id.str.replace(r'[NS]$', '', regex=True)
    st = st.sort_values(['trip_id', 'stop_sequence']).reset_index(drop=True)

    # build per-trip ordered stop list with distances
    # trip_stops[trip_id] = [(idx_in_st, parent_stop, arr_secs, dep_secs, dist_to_next_m), ...]
    trip_stops = defaultdict(list)
    prev = None
    for i, row in st.iterrows():
        if prev is not None and row.trip_id == prev.trip_id:
            from_stop = prev.parent_stop
            to_stop = row.parent_stop
            if from_stop in stop_coords and to_stop in stop_coords:
                lat1, lon1 = stop_coords[from_stop]
                lat2, lon2 = stop_coords[to_stop]
                dist = approx_distance_m(lat1, lon1, lat2, lon2)
            else:
                dist = None
            # attach dist_to_next to the previous entry
            trip_stops[prev.trip_id][-1] = (*trip_stops[prev.trip_id][-1][:4], dist)
        trip_stops[row.trip_id].append((i, row.parent_stop, row.arr_secs, row.dep_secs, None))
        prev = row

    # find fixes: for each trip, find fast segments and steal from neighbors
    # adjustments[st_index] = delta_seconds (positive = shift later)
    adjustments = defaultdict(int)
    fix_proposals = []  # for dry-run display

    for trip_id, stop_list in trip_stops.items():
        route = trip_route.get(trip_id, '?')
        n = len(stop_list)

        for j in range(n - 1):
            idx, fr_stop, fr_arr, fr_dep, dist = stop_list[j]
            idx_next, to_stop, to_arr, to_dep, _ = stop_list[j + 1]

            if dist is None:
                continue
            dt = to_arr - fr_dep
            if dt <= 0:
                continue

            speed = cruise_speed_kmh(dist, dt)
            if speed <= SPEED_THRESHOLD_KMH:
                continue

            # how much extra time do we need?
            needed_dt = math.ceil(time_needed_for_speed(dist, SPEED_THRESHOLD_KMH))
            extra = needed_dt - dt

            # try to steal from the slower neighbor (before or after)
            # check the segment before (j-1 -> j)
            before_avail = 0
            if j > 0:
                prev_idx, prev_stop, prev_arr, prev_dep, prev_dist = stop_list[j - 1]
                if prev_dist is not None:
                    prev_dt = fr_arr - prev_dep
                    if prev_dt > OVERHEAD + 1:
                        # how much can we steal while keeping neighbor above threshold?
                        prev_min_dt = math.ceil(time_needed_for_speed(prev_dist, SPEED_THRESHOLD_KMH))
                        before_avail = max(0, prev_dt - prev_min_dt)

            # check the segment after (j+1 -> j+2)
            after_avail = 0
            if j + 2 < n:
                _, _, _, next_dep, next_dist = stop_list[j + 1]
                next2_idx, _, next2_arr, _, _ = stop_list[j + 2]
                if next_dist is not None:
                    next_dt = next2_arr - next_dep
                    if next_dt > OVERHEAD + 1:
                        next_min_dt = math.ceil(time_needed_for_speed(next_dist, SPEED_THRESHOLD_KMH))
                        after_avail = max(0, next_dt - next_min_dt)

            total_avail = before_avail + after_avail
            if total_avail == 0:
                fix_proposals.append((route, trip_id, fr_stop, to_stop, speed,
                                      extra, 0, "NO DONOR"))
                continue

            take = min(extra, total_avail)
            # split proportionally between before and after
            if total_avail > 0:
                take_before = round(take * before_avail / total_avail)
                take_after = take - take_before
            else:
                take_before = take_after = 0

            new_speed = cruise_speed_kmh(dist, dt + take)

            fix_proposals.append((route, trip_id, fr_stop, to_stop, speed,
                                  extra, take,
                                  f"{speed:.0f} -> {new_speed:.0f} km/h  "
                                  f"(steal {take_before}s before, {take_after}s after)"))

            if take_before > 0:
                # shift the middle stop (j) later by take_before
                # this slows down segment j-1->j and speeds up j->j+1
                adjustments[idx] += take_before

            if take_after > 0:
                # shift stop j+1 earlier by take_after
                # this speeds up j->j+1 and slows down j+1->j+2
                adjustments[idx_next] -= take_after

    # dedupe proposals for display: group by (route, from, to), show worst
    seen = set()
    unique = []
    for p in sorted(fix_proposals, key=lambda x: -x[4]):
        key = (p[0], p[2], p[3])
        if key not in seen:
            seen.add(key)
            unique.append(p)

    print(f"Segments above {SPEED_THRESHOLD_KMH} km/h: {len(fix_proposals)} "
          f"({len(unique)} unique route segments)\n")

    for route, trip, fr, to, speed, needed, got, detail in unique[:60]:
        status = "FIXED" if got >= needed else "PARTIAL" if got > 0 else "NO FIX"
        print(f"  [{status:>7}] {route:>3}  {fr} -> {to}  {detail}")

    if not args.apply:
        adj_count = sum(1 for v in adjustments.values() if v != 0)
        print(f"\n{adj_count} stop times would be adjusted. Run with --apply to write.")
        return

    # apply adjustments to the full stop_times file
    print(f"\nApplying {sum(1 for v in adjustments.values() if v != 0)} adjustments...")
    full_st = pd.read_csv(f"{GTFS_DIR}/stop_times.txt")

    # we need to map our filtered index back to the full file
    # rebuild: only adjust weekday trips
    weekday_trip_ids = set(trips.trip_id)
    filtered = full_st[full_st.trip_id.isin(weekday_trip_ids)].copy()
    filtered['arr_secs'] = filtered.arrival_time.apply(time_to_secs)
    filtered['dep_secs'] = filtered.departure_time.apply(time_to_secs)
    filtered = filtered.sort_values(['trip_id', 'stop_sequence'])

    # map filtered positional index -> full dataframe index
    filtered_indices = filtered.index.tolist()

    changes = 0
    for pos, delta in adjustments.items():
        if delta == 0:
            continue
        if pos >= len(filtered_indices):
            continue
        real_idx = filtered_indices[pos]
        old_arr = time_to_secs(full_st.at[real_idx, 'arrival_time'])
        old_dep = time_to_secs(full_st.at[real_idx, 'departure_time'])
        full_st.at[real_idx, 'arrival_time'] = secs_to_time(old_arr + delta)
        full_st.at[real_idx, 'departure_time'] = secs_to_time(old_dep + delta)
        changes += 1

    full_st.to_csv(f"{GTFS_DIR}/stop_times.txt", index=False)
    print(f"Wrote {changes} changes to {GTFS_DIR}/stop_times.txt")


if __name__ == '__main__':
    main()
