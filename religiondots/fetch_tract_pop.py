"""
Fetch 2020 decennial census tract populations — the weight used to place dots inside a
county (spec.md §8).

Decennial PL 94-171, not ACS, deliberately: ASARB's own county population column is the
2020 census count, so weighting with the same universe means the tract weights sum to the
county total the religion data was measured against. An ACS 5-year estimate would not.

Set CENSUS_API_KEY in the environment or a .env file (optional; the API serves modest
volumes without one).

Usage:
    python fetch_tract_pop.py
"""
import os
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.census.gov/data/2020/dec/pl"
OUT = Path(__file__).parent / "data" / "geo" / "tract_pop_2020.csv"

STATE_FIPS = [
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13", "15", "16", "17",
    "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31",
    "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "44", "45", "46",
    "47", "48", "49", "50", "51", "53", "54", "55", "56", "72",
]

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
API_KEY = os.getenv("CENSUS_API_KEY", "")


def fetch_state(fips: str) -> pd.DataFrame:
    params = {"get": "P1_001N", "for": "tract:*", "in": f"state:{fips}"}
    if API_KEY:
        params["key"] = API_KEY
    r = requests.get(BASE, params=params, timeout=60)
    r.raise_for_status()
    rows = r.json()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    df["pop"] = df["P1_001N"].astype(int)
    return df[["GEOID", "pop"]]


def main():
    frames = []
    for i, fips in enumerate(STATE_FIPS, 1):
        try:
            df = fetch_state(fips)
        except Exception as exc:                        # noqa: BLE001
            print(f"  {fips}: FAILED — {exc}")
            continue
        frames.append(df)
        print(f"  [{i:>2}/{len(STATE_FIPS)}] state {fips}: {len(df):>6,} tracts, "
              f"{df['pop'].sum():>12,} people")
        time.sleep(0.2)

    out = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\n{len(out):,} tracts, {out['pop'].sum():,} people -> {OUT}")
    print(f"tracts with zero population: {(out['pop'] == 0).sum():,}")


if __name__ == "__main__":
    main()
