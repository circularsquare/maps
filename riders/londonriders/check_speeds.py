"""
Inspect inter-stop train speeds in the parsed London trip data.

Same idea as nycriders/check_speeds.py: timetables are minute-rounded, which
makes some segments look impossibly fast and adjacent ones look impossibly
slow. The actual smoothing logic lives in build.py (`smooth_trip_speeds`)
and runs automatically; this script is a report-only inspection tool.

Usage:
  python check_speeds.py                 # report fast segments
  python check_speeds.py --after         # apply smoothing and report after
  python check_speeds.py --after -v      # also print example fixes
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build
from build import (
    SPEED_THRESHOLDS, DEFAULT_SPEED_THRESHOLD,
    haversine_m, cruise_speed_kmh, smooth_trip_speeds, jitter_trip_times,
)


def load_trips_and_coords():
    """Load the same trip set build.py uses, plus naptan->coord lookup."""
    geo = build.load_geometry()
    coords = geo.get("naptan_coords") or {}
    coords = {nid: (v[0], v[1]) for nid, v in coords.items()}

    trips = []
    if os.path.isdir("data/timetables"):
        from railwaydata import parse_all_pdfs
        trips.extend(parse_all_pdfs())
    cif_lines_seen = set()
    if os.path.exists("data/ezl.json.gz") or os.path.exists("data/og.json.gz"):
        from cif import parse_all_cif
        cif_trips = parse_all_cif()
        cif_lines_seen = {t["line"] for t in cif_trips}
        trips.extend(cif_trips)
    if os.path.isdir("data/timetables"):
        from tfl_pdf import parse_all_tfl_pdfs
        pdf_trips = [t for t in parse_all_tfl_pdfs() if t["line"] not in cif_lines_seen]
        trips.extend(pdf_trips)
    trips = build.dedupe_prefix_trips(trips)
    return trips, coords


def report(trips, coords, label):
    """Print speed distribution and segment-count stats."""
    by_line = defaultdict(lambda: {"total": 0, "over": 0, "max_speed": 0.0})
    for trip in trips:
        line = trip["line"]
        threshold = SPEED_THRESHOLDS.get(line, DEFAULT_SPEED_THRESHOLD)
        stops = trip["stops"]
        for j in range(len(stops) - 1):
            a, b = stops[j][0], stops[j + 1][0]
            ca, cb = coords.get(a), coords.get(b)
            if not ca or not cb:
                continue
            d = haversine_m(ca[0], ca[1], cb[0], cb[1])
            dt = stops[j + 1][1] - stops[j][1]
            if dt <= 0:
                continue
            speed = cruise_speed_kmh(d, dt)
            entry = by_line[line]
            entry["total"] += 1
            if speed > threshold:
                entry["over"] += 1
            if speed > entry["max_speed"]:
                entry["max_speed"] = speed

    print(f"\n=== {label} ===")
    print(f"{'line':<6} {'thresh':>7} {'segs':>8} {'> thresh':>14} {'max km/h':>10}")
    for line in sorted(by_line):
        e = by_line[line]
        thr = SPEED_THRESHOLDS.get(line, DEFAULT_SPEED_THRESHOLD)
        pct = 100 * e["over"] / max(1, e["total"])
        over_str = f"{e['over']:>5} ({pct:>4.1f}%)"
        print(f"{line:<6} {thr:>7} {e['total']:>8} {over_str:>14} {e['max_speed']:>10.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--after", action="store_true",
                        help="Apply smoothing in-memory and report the result")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print example fixes")
    args = parser.parse_args()

    trips, coords = load_trips_and_coords()
    print(f"Loaded {len(trips)} trips, {len(coords)} naptan coords.")

    report(trips, coords, "BEFORE smoothing")

    if args.after:
        jitter_trip_times(trips)
        stats = smooth_trip_speeds(trips, coords, verbose=args.verbose)
        if args.verbose:
            for line, fr, to, sp_b, sp_a, tb, ta in stats["examples"][:30]:
                print(f"  {line:<5} {fr} -> {to}  {sp_b:>5.0f} -> {sp_a:>5.0f} km/h  "
                      f"(steal {tb}s before, {ta}s after)")
        report(trips, coords, "AFTER smoothing")


if __name__ == "__main__":
    main()
