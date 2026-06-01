"""Fast diagnostic: categorize why each unassigned OD row failed.

Single-pass version that mirrors assign_riders() — for each row we do at most
one path scan and one time scan; failures are bucketed by reason.

Buckets:
  no_naptan_o    - mnlc_o has no naptan ID in our (line, mnlc) map for this line
  no_naptan_d    - mnlc_d has no naptan
  no_trip_path   - naptans exist on the line, but no trip's pattern includes
                   o then d in order
  no_trip_time   - a trip's pattern covers o->d but none depart o in the qhr
                   15-min window
"""
import csv
import json
import os
from bisect import bisect_left
from collections import defaultdict

from railwaydata import parse_all_pdfs
from cif import parse_all_cif

GEOMETRY_JSON = "geometry.json"
OD_PER_LINE_DIR = "data/od_per_line"
QHR_LEN_S = 15 * 60

PER_LINE_TO_CODES = {
    "bak": ["BAK"], "bakwdc": ["BAK", "WEL"], "cen": ["CEN"], "dis": ["DIS"],
    "dlr": ["DLR"], "ezl": ["EZL"], "ham": ["HAM", "CIR"], "jub": ["JUB"],
    "loa": ["WAG"], "loe": ["ELL"], "log": ["GOB"], "lon": ["NLL"],
    "lor": ["URL"], "met": ["MET"], "nor": ["NOR"], "pic": ["PIC"],
    "ssl": ["DIS", "HAM", "MET", "CIR"],
    "ssp": ["PIC", "DIS", "HAM", "MET", "CIR"],
    "vic": ["VIC"], "wat": ["WAC"],
}


def main():
    with open(GEOMETRY_JSON, encoding="utf-8") as f:
        geo = json.load(f)
    mnlc_to_naptans = {int(k): v for k, v in geo["mnlc_to_naptans"].items()}
    mnlc_to_naptans_per_line = {
        line: {int(m): v for m, v in by_mnlc.items()}
        for line, by_mnlc in geo.get("mnlc_to_naptans_per_line", {}).items()
    }
    stations = geo["stations"]

    trips = parse_all_pdfs()
    cif_trips = parse_all_cif()
    cif_lines_seen = {t["line"] for t in cif_trips}
    # Mirror build.py: PDFs are dropped for any line CIF supplies (EZL/OG).
    trips = [t for t in trips if t["line"] not in cif_lines_seen] + cif_trips
    print(f"Indexing {len(trips)} trips...")
    trip_index = defaultdict(list)
    for ti, t in enumerate(trips):
        for pos, (n, s) in enumerate(t["stops"]):
            trip_index[(t["line"], n)].append((s, ti, pos))
    for k in trip_index:
        trip_index[k].sort()
    dep_arrays = {k: [v[0] for v in lst] for k, lst in trip_index.items()}

    # Pre-compute set of stops per trip for O(1) membership checks
    trip_stop_sets = [set(n for n, _ in t["stops"]) for t in trips]

    print(f"\n{'file':14s} {'no_napt_o':>10s} {'no_napt_d':>10s} {'no_path':>10s} {'no_time':>10s}  | top unmatched stations (riders)")

    for fname in sorted(os.listdir(OD_PER_LINE_DIR)):
        if not fname.endswith(".csv"):
            continue
        base = fname[:-4]
        line_codes = PER_LINE_TO_CODES.get(base)
        if not line_codes:
            continue

        bucket_riders = defaultdict(float)
        unmatched_o = defaultdict(float)
        unmatched_d = defaultdict(float)
        path_fails = defaultdict(float)   # (o, d) -> riders

        with open(f"{OD_PER_LINE_DIR}/{fname}", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    o = int(row["mode_mnlc_o"])
                    d = int(row["mode_mnlc_d"])
                    qhr = int(row["qhr"]) - 1
                    vol = float(row["vol"])
                except (KeyError, ValueError):
                    continue
                if vol <= 0 or o == d or not (0 <= qhr < 104):
                    continue

                naptans_o_set = set()
                naptans_d_set = set()
                for line_code in line_codes:
                    pl = mnlc_to_naptans_per_line.get(line_code)
                    if not pl:
                        continue
                    naptans_o_set.update(pl.get(o, []))
                    naptans_d_set.update(pl.get(d, []))
                if not naptans_o_set:
                    naptans_o_set.update(mnlc_to_naptans.get(o, []))
                if not naptans_d_set:
                    naptans_d_set.update(mnlc_to_naptans.get(d, []))

                if not naptans_o_set:
                    bucket_riders["no_naptan_o"] += vol
                    unmatched_o[o] += vol
                    continue
                if not naptans_d_set:
                    bucket_riders["no_naptan_d"] += vol
                    unmatched_d[d] += vol
                    continue

                window_start = qhr * QHR_LEN_S
                window_end = window_start + QHR_LEN_S

                # Single pass: find any trip that covers o->d. If found, then
                # check if any in window. Track best outcome.
                any_path = False
                any_time = False
                for line_code in line_codes:
                    for nid_o in naptans_o_set:
                        lst = trip_index.get((line_code, nid_o))
                        if not lst:
                            continue
                        # Path test: scan trips at this naptan, check stops set
                        for dep, ti, pos_o in lst:
                            if naptans_d_set & trip_stop_sets[ti]:
                                # Verify d is *after* o in the stop order
                                stops = trips[ti]["stops"]
                                for j in range(pos_o + 1, len(stops)):
                                    if stops[j][0] in naptans_d_set:
                                        any_path = True
                                        if window_start <= dep < window_end:
                                            any_time = True
                                        break
                                if any_time:
                                    break
                        if any_time:
                            break
                    if any_time:
                        break

                if not any_path:
                    bucket_riders["no_trip_path"] += vol
                    path_fails[(o, d)] += vol
                elif not any_time:
                    bucket_riders["no_trip_time"] += vol

        # Top failures for the verbose summary
        top_o = sorted(unmatched_o.items(), key=lambda kv: -kv[1])[:2]
        top_d = sorted(unmatched_d.items(), key=lambda kv: -kv[1])[:2]
        top_path = sorted(path_fails.items(), key=lambda kv: -kv[1])[:2]

        def name(m):
            return stations.get(str(m), {}).get("name", f"mnlc{m}")

        details = []
        for m, v in top_o:
            details.append(f"o:{name(m)}({v:.0f})")
        for m, v in top_d:
            details.append(f"d:{name(m)}({v:.0f})")
        for (o, d), v in top_path:
            details.append(f"{name(o)}->{name(d)}({v:.0f})")

        print(f"{fname:14s} {bucket_riders['no_naptan_o']:>10,.0f} "
              f"{bucket_riders['no_naptan_d']:>10,.0f} "
              f"{bucket_riders['no_trip_path']:>10,.0f} "
              f"{bucket_riders['no_trip_time']:>10,.0f}  | "
              + "  ".join(details))


if __name__ == "__main__":
    main()
