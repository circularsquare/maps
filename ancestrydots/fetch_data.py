"""
Fetch ACS B04006 (People Reporting Ancestry) data for all census tracts.
Also fetches B02015 (detailed Asian), B02016 (Pacific Islander),
B02020 (American Indian and Alaska Native), and B03001 (detailed Hispanic).

Usage:
    python fetch_data.py --state 36          # NY only (for NYC dev)
    python fetch_data.py --all               # all 50 states
    python fetch_data.py --state 36 --table B04006

Set CENSUS_API_KEY in .env or as env var (optional but recommended).
"""

import os
import json
import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CENSUS_API_KEY", "")
BASE_URL = "https://api.census.gov/data/2024/acs/acs5"

# All 50 state FIPS codes + DC
STATE_FIPS = [
    "01","02","04","05","06","08","09","10","11","12","13","15","16","17","18",
    "19","20","21","22","23","24","25","26","27","28","29","30","31","32","33",
    "34","35","36","37","38","39","40","41","42","44","45","46","47","48","49",
    "50","51","53","54","55","56"
]

TABLES = ["B04006", "B02015", "B03001", "B02016", "B02020", "B02008", "B02009"]


def fetch_table(table: str, state_fips: str) -> pd.DataFrame:
    params = {
        "get": f"group({table})",
        "for": "tract:*",
        "in": f"state:{state_fips}",
    }
    if API_KEY:
        params["key"] = API_KEY

    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    return df


def fetch_state(state_fips: str, out_dir: Path):
    for table in TABLES:
        out_path = out_dir / f"{table}_{state_fips}.csv"
        if out_path.exists():
            print(f"  {table} state {state_fips}: already exists, skipping")
            continue
        print(f"  Fetching {table} for state {state_fips}...")
        try:
            df = fetch_table(table, state_fips)
            df.to_csv(out_path, index=False)
            print(f"    Saved {len(df)} tracts to {out_path}")
        except Exception as e:
            print(f"    ERROR: {e}")
        time.sleep(0.5)  # be polite to census API


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", help="Single state FIPS code (e.g. 36 for NY)")
    parser.add_argument("--all", action="store_true", help="Fetch all 50 states")
    args = parser.parse_args()

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.state:
        states = [args.state.zfill(2)]
    elif args.all:
        states = STATE_FIPS
    else:
        parser.print_help()
        return

    for fips in states:
        print(f"State {fips}:")
        fetch_state(fips, out_dir)


if __name__ == "__main__":
    main()
