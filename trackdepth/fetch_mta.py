"""Fetch MTA open-data tables that bear on depth.

- raw/mta_elevators.csv : Subway Elevator and Escalator Asset Inventory (94fv-bak7);
  has Elevator Rise / Escalator Rise in feet for some machines.
- raw/mta_stations.csv  : MTA Subway Stations (39hk-dx4f); has Structure
  (Subway / Elevated / At Grade / Open Cut / Embankment / Viaduct).
"""
import pathlib

import requests

ROOT = pathlib.Path(__file__).parent
RAW = ROOT / "raw"
UA = "trackdepth/0.1"
SETS = {
    "mta_elevators.csv": "94fv-bak7",
    "mta_stations.csv": "39hk-dx4f",
}

if __name__ == "__main__":
    RAW.mkdir(exist_ok=True)
    for name, ident in SETS.items():
        url = f"https://data.ny.gov/api/views/{ident}/rows.csv?accessType=DOWNLOAD"
        r = requests.get(url, headers={"User-Agent": UA}, timeout=300)
        r.raise_for_status()
        (RAW / name).write_bytes(r.content)
        rows = r.content.count(b"\n") - 1
        print(f"{name}: {rows} rows")
