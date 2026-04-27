"""
Download NUMBAT 2024 source data from TfL crowding bucket.

Files pulled into data/:
  - PTSP Oasis for NUMBAT definitions.xlsx  (stations, links, lines)
  - NBT24TWT_outputs.xlsx                   (Link_Loads, Link_Frequencies)
  - NBT24TWT5d_od_network_tb_lf_o.csv       (network OD by 15-min band)
  - OD Data/NBT24TWT4c_od_<line>_qhr_lf_o.csv  (per-line route-choice OD)

TWT = Tue/Wed/Thu typical weekday (closest analog to NYC's "Wednesday").
"""
import os
import sys
import time
import urllib.request

BASE = "https://s3-eu-west-1.amazonaws.com/crowding.data.tfl.gov.uk/NUMBAT/NUMBAT%202024"
OUT_DIR = "data"

# Tube + DLR + Elizabeth + Overground line codes used in OD per-line filenames.
# Trams (TRM) excluded — runs in Croydon and isn't part of the rail OD CSVs.
LINE_CODES = [
    "bak", "bakwdc", "cen", "dis", "dlr", "ezl", "ham", "jub",
    "loa", "loe", "log", "lon", "lor",  # overground branches (no low/lor for TWT)
    "met", "nor", "pic", "ssl", "ssp", "vic", "wat",
]


def download(url, dest):
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1024 / 1024
        print(f"  skip (exists, {size_mb:.1f} MB) {dest}")
        return
    print(f"  GET {url}")
    t0 = time.time()
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.rename(tmp, dest)
    size_mb = os.path.getsize(dest) / 1024 / 1024
    print(f"       {size_mb:.1f} MB in {time.time() - t0:.1f}s")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/od_per_line", exist_ok=True)

    print("Definitions...")
    download(f"{BASE}/PTSP%20Oasis%20for%20NUMBAT%20definitions.xlsx",
             f"{OUT_DIR}/numbat_definitions.xlsx")

    print("Outputs (link freq, link loads)...")
    download(f"{BASE}/NBT24TWT_outputs.xlsx",
             f"{OUT_DIR}/numbat_twt_outputs.xlsx")

    print("Network OD by 15-min band...")
    download(f"{BASE}/OD%20Data/NBT24TWT5d_od_network_tb_lf_o.csv",
             f"{OUT_DIR}/od_network_twt.csv")

    print(f"Per-line OD ({len(LINE_CODES)} files)...")
    for line in LINE_CODES:
        download(f"{BASE}/OD%20Data/NBT24TWT4c_od_{line}_qhr_lf_o.csv",
                 f"{OUT_DIR}/od_per_line/{line}.csv")

    print("Done.")


if __name__ == "__main__":
    main()
