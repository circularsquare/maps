"""Download the raw US inputs for citydirections. All keyless, ~30 MB total.

  CenPop2020_Mean_BG.txt  population-weighted centroid + population of every
                          2020 census block group (geometry, no shapefiles needed)
  acsdt5y2023-b19013.dat  ACS 2019-2023 median household income, all geo levels
  list1_2023.xlsx         county -> CBSA (metro area) crosswalk
"""
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"

SOURCES = {
    "CenPop2020_Mean_BG.txt":
        "https://www2.census.gov/geo/docs/reference/cenpop2020/blkgrp/CenPop2020_Mean_BG.txt",
    "acsdt5y2023-b19013.dat":
        "https://www2.census.gov/programs-surveys/acs/summary_file/2023/"
        "table-based-SF/data/5YRData/acsdt5y2023-b19013.dat",
    "list1_2023.xlsx":
        "https://www2.census.gov/programs-surveys/metro-micro/geographies/"
        "reference-files/2023/delineation-files/list1_2023.xlsx",
}


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        dest = DATA / name
        if dest.exists():
            print(f"  have {name} ({dest.stat().st_size:,} bytes)")
            continue
        print(f"fetching {name} ...", flush=True)
        urllib.request.urlretrieve(url, dest)
        print(f"  wrote {name} ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
