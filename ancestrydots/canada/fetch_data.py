"""
Fetch StatCan 2021 Census Profile (98-401-X2021006) ethnic-or-cultural-origin
data at the Dissemination Area level, for one or more regions.

The profile CSV is a huge long-format file (~2600 characteristics per geography,
every geography level from Country down to DA). We stream it and keep only:
  - GEO_LEVEL == "Dissemination area"
  - CHARACTERISTIC_ID in 1699..1948  (the 250 individual ethnic/cultural origins;
    1698 is the "Total" row, kept too for reference)
and write a compact long CSV: dguid, cid, origin, count.

Regions (GEONO suffix): Territories_Territoires, Atlantic, Quebec, Ontario,
Prairies, British_Columbia.  Territories is tiny (11MB) and good for testing.

Usage:
    python fetch_data.py --region Territories_Territoires
    python fetch_data.py --region Ontario
    python fetch_data.py --all
    python fetch_data.py --region Ontario --from-zip path/to/already.zip
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

BASE = ("https://www12.statcan.gc.ca/census-recensement/2021/dp-pd/prof/details/"
        "download-telecharger/comp/GetFile.cfm?Lang=E&FILETYPE=CSV&GEONO=006_")

# Short region key -> exact GEONO suffix on the StatCan download page.
REGION_GEONO = {
    "Territories": "Territories_Territoires",
    "Atlantic": "Atlantic_Atlantique",
    "Quebec": "Quebec",
    "Ontario": "Ontario",
    "Prairies": "Prairies",
    "BC": "BC_CB",
}
REGIONS = list(REGION_GEONO)

# Ethnic-or-cultural-origin characteristic IDs (verified against the 2021 profile
# metadata; 1698 = total, 1699-1948 = 250 individual origins, 1949 = next topic).
ORIGIN_ID_LO, ORIGIN_ID_HI = 1699, 1948
TOTAL_ID = 1698

RAW_DIR = Path(__file__).parent / "data" / "raw"


def region_url(region: str) -> str:
    return BASE + REGION_GEONO[region]


def download_zip(region: str, dest: Path) -> Path:
    url = region_url(region)
    print(f"  Downloading {region} profile...\n    {url}")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    size = dest.stat().st_size
    with open(dest, "rb") as f:
        magic = f.read(2)
    if magic != b"PK" or size < 10_000:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Download for {region} was not a valid zip ({size} bytes) — "
                           f"check the GEONO suffix for this region.")
    print(f"    saved {dest} ({size/1e6:.0f} MB)")
    return dest


def parse_zip_to_origins(zip_path: Path, out_csv: Path) -> int:
    """Stream the profile CSV inside the zip, keep DA-level origin rows, write long CSV."""
    with zipfile.ZipFile(zip_path) as zf:
        data_name = next(n for n in zf.namelist() if n.endswith(".csv") and "CSV_data" in n)
        print(f"  Parsing {data_name} ...")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        kept = 0
        with zf.open(data_name) as raw, open(out_csv, "w", newline="", encoding="utf-8") as out:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="latin-1"))
            header = next(reader)
            col = {name: i for i, name in enumerate(header)}
            i_lvl = col["GEO_LEVEL"]; i_dg = col["DGUID"]
            i_cid = col["CHARACTERISTIC_ID"]; i_name = col["CHARACTERISTIC_NAME"]
            i_cnt = col["C1_COUNT_TOTAL"]
            w = csv.writer(out)
            w.writerow(["dguid", "cid", "origin", "count"])
            for row in reader:
                if row[i_lvl] != "Dissemination area":
                    continue
                try:
                    cid = int(row[i_cid])
                except ValueError:
                    continue
                if not (cid == TOTAL_ID or ORIGIN_ID_LO <= cid <= ORIGIN_ID_HI):
                    continue
                cnt = row[i_cnt].strip()
                # blanks / suppression symbols -> 0
                count = int(cnt) if cnt.isdigit() else 0
                if count == 0 and cid != TOTAL_ID:
                    continue
                w.writerow([row[i_dg], cid, row[i_name].strip(), count])
                kept += 1
        print(f"    wrote {kept} DA-origin rows -> {out_csv}")
        return kept


def process(region: str, from_zip: str | None = None):
    print(f"Region {region}:")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = Path(from_zip) if from_zip else RAW_DIR / f"profile_{region}.zip"
    if not zip_path.exists():
        download_zip(region, zip_path)
    else:
        print(f"  Using existing {zip_path}")
    out_csv = RAW_DIR / f"origins_{region}.csv"
    parse_zip_to_origins(zip_path, out_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", help="One region GEONO suffix, e.g. Ontario")
    ap.add_argument("--all", action="store_true", help="All six regions")
    ap.add_argument("--from-zip", help="Parse an already-downloaded zip instead of fetching")
    args = ap.parse_args()

    if args.all:
        regions = REGIONS
    elif args.region:
        regions = [args.region]
    else:
        ap.print_help(); sys.exit(1)

    for region in regions:
        process(region, from_zip=args.from_zip if args.region else None)


if __name__ == "__main__":
    main()
