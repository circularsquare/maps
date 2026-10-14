"""
Fetch MTA Subway Origin-Destination ridership data from Socrata API.
Pulls one month and one day of the week (an average day of that type), all
O-D pairs by hour.

    python fetch_data.py --year 2026 --month 5 --day wednesday   # what build.py uses
    python fetch_data.py --year 2026 --month 5 --day saturday
    python fetch_data.py --year 2026 --month 5 --day sunday

Writes data/od_<day>_<mon>.csv for 2025 (the names build.py already uses) and
data/od_<day>_<year>-<mm>.csv for other years. Expect ~1.2-1.6M rows.
"""
import argparse
import calendar
import csv
import time

import requests

# one dataset per year; 2026 is "Beginning 2026" and grows month by month
DATASETS = {2023: "uhf3-t34z", 2024: "jsu2-fbtj", 2025: "y2qv-fytt", 2026: "28vm-gjqr"}

FIELDS = (
    "hour_of_day,"
    "origin_station_complex_id,origin_station_complex_name,"
    "origin_latitude,origin_longitude,"
    "destination_station_complex_id,destination_station_complex_name,"
    "destination_latitude,destination_longitude,"
    "estimated_average_ridership"
)

EXPECTED_ROWS = 1588000
LIMIT = 500000  # big pages: ordered paging with large offsets is slow on Socrata
TIMEOUT = 300


def fetch_all(base_url, where):
    rows = []
    offset = 0
    session = requests.Session()
    t0 = time.time()
    while True:
        pct = min(100, int(offset / EXPECTED_ROWS * 100))
        elapsed = time.time() - t0
        print(f"  [{pct:3d}%] fetching rows {offset}-{offset+LIMIT} ({elapsed:.0f}s elapsed)...", end=" ", flush=True)
        for attempt in range(3):
            try:
                resp = session.get(base_url, params={
                    "$select": FIELDS,
                    "$where": where,
                    "$limit": LIMIT,
                    "$offset": offset,
                    "$order": "hour_of_day,estimated_average_ridership DESC,"
                              "origin_station_complex_id,destination_station_complex_id",
                }, timeout=TIMEOUT)
                resp.raise_for_status()
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                if attempt == 2:
                    raise
                print(f"retry {attempt+1}...", end=" ", flush=True)
        batch = resp.json()
        print(f"got {len(batch)} rows (total: {len(rows) + len(batch)})")
        if not batch:
            break
        rows.extend(batch)
        offset += LIMIT
    print(f"  Done! Fetched {len(rows)} rows in {time.time() - t0:.0f}s")
    return rows


def write_csv(rows, out_file):
    if not rows:
        print("No data!")
        return
    keys = list(rows[0].keys())
    with open(out_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2025, choices=sorted(DATASETS))
    parser.add_argument('--month', type=int, default=9)
    parser.add_argument('--day', default='wednesday',
                        help="day of the week, e.g. wednesday, saturday, sunday")
    args = parser.parse_args()

    day = args.day.capitalize()
    mon = calendar.month_abbr[args.month].lower()
    if args.year == 2025:
        out_file = f"data/od_{day.lower()}_{mon}.csv"
    else:
        out_file = f"data/od_{day.lower()}_{args.year}-{args.month:02d}.csv"
    base_url = f"https://data.ny.gov/resource/{DATASETS[args.year]}.json"
    where = f"month={args.month} AND day_of_week='{day}'"

    print(f"Fetching O-D data for {day}s in {calendar.month_name[args.month]} {args.year} -> {out_file}")
    print(f"Expecting ~{EXPECTED_ROWS:,} rows")
    rows = fetch_all(base_url, where)
    write_csv(rows, out_file)
