"""Fetch Turkmenistan's raw files into helper1m/data/turkmenistan/raw/, skipping any present.

- census section 1 PDF (stat.gov.tm), copied from religiondots' cache when there
- Kontur Boundaries 2023-06-28 (HDX), copied from religiondots' cache when there
- Geofabrik OSM extract of 2026-10-05 (25 MB)
- UN WPP 2024 total population CSV (17 MB)
religiondots is only read.
"""
import shutil
import urllib.request
from pathlib import Path

HELPER = Path(__file__).resolve().parents[2]
RAW = HELPER / "data" / "turkmenistan" / "raw"
RD = HELPER.parent / "religiondots" / "data" / "raw" / "tm"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) helper1m-build/1.0"}

FILES = [
    ("tm_census2022_results_en_1.pdf", RD / "tm_census2022_results_en_1.pdf",
     "https://www.stat.gov.tm/population-census-pdfs/results/en/1.pdf"),
    ("kontur_boundaries_TM_20230628.gpkg", RD / "kontur_boundaries_TM_20230628.gpkg", None),
    ("turkmenistan-261005.osm.pbf", None,
     "https://download.geofabrik.de/asia/turkmenistan-261005.osm.pbf"),
    ("WPP2024_TotalPopulationBySex.csv.gz", None,
     "https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/"
     "CSV_FILES/WPP2024_TotalPopulationBySex.csv.gz"),
]


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, local, url in FILES:
        out = RAW / name
        if out.exists():
            print(f"have {name}")
        elif local and local.exists():
            shutil.copyfile(local, out)
            print(f"copied {name} from religiondots")
        elif url:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r:
                out.write_bytes(r.read())
            print(f"downloaded {name}")
        else:
            print(f"MISSING {name}: Kontur Boundaries are on HDX "
                  "(kontur-boundaries-turkmenistan), release 2023-06-28")


if __name__ == "__main__":
    main()
