"""
Fetch MTA Subway Origin-Destination ridership data from Socrata API.
Pulls a single representative day (Wednesday in October 2024) and filters
to all O-D pairs for a single day type.

Expected: ~1.6M rows, may take a while.
"""
import requests
import csv
import time

DATASET_ID = "jsu2-fbtj"
BASE_URL = f"https://data.ny.gov/resource/{DATASET_ID}.json"

EXPECTED_ROWS = 1588000
WHERE = "month=10 AND day_of_week='Wednesday'"
FIELDS = (
    "hour_of_day,"
    "origin_station_complex_id,origin_station_complex_name,"
    "origin_latitude,origin_longitude,"
    "destination_station_complex_id,destination_station_complex_name,"
    "destination_latitude,destination_longitude,"
    "estimated_average_ridership"
)

OUT_FILE = "data/od_wednesday_oct.csv"
LIMIT = 10000
TIMEOUT = 300


def fetch_all():
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
                resp = session.get(BASE_URL, params={
                    "$select": FIELDS,
                    "$where": WHERE,
                    "$limit": LIMIT,
                    "$offset": offset,
                    "$order": "hour_of_day,estimated_average_ridership DESC",
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


def write_csv(rows):
    if not rows:
        print("No data!")
        return
    keys = list(rows[0].keys())
    with open(OUT_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_FILE}")


if __name__ == "__main__":
    print(f"Fetching O-D data for Wednesdays in October 2024 (ridership >= 10)...")
    print(f"Expecting ~{EXPECTED_ROWS:,} rows")
    rows = fetch_all()
    write_csv(rows)
