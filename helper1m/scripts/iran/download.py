"""Download the raw census files for helper1m Iran, through the Wayback Machine.

amar.org.ir (Statistical Centre of Iran, SCI) resets every TLS connection from here, so every
file comes from web.archive.org's `id_` copies (the original bytes, no Wayback banner).

  raw/abadi90/osNN.xls       1390 (2011) census population by settlement (abadi), one per province,
                             on the 1390 divisions. Path: /Portals/0/sarshomari90/Files/ABADY-90/.
                             Fars (07), Mazandaran (02) and West Azerbaijan (04) exist only as
                             `osNN-r.xls` (revised).
  raw/abadi95/CN95_HouseholdPopulationVillage_NN_r.xlsx
                             1395 (2016) census by settlement, on the 1395 divisions, with every
                             province, county (shahrestan), district (bakhsh), rural district
                             (dehestan) and city row. `_r` is the revised file; Kurdistan (12)
                             exists only in that form, so `_r` is used for all.

  raw/sci/GEO1400.xlsx       SCI's settlement file for the 1400 divisions (bakhsh.py)
  raw/sci/iod-06124.html     SCI's 1403 provincial estimate, Iran Open Data's page (Wayback)

Not fetched here: the Geofabrik extract for the district polygons and place nodes,
https://download.geofabrik.de/asia/iran-latest.osm.pbf (230 MB) into raw/osm/, then
osm_admin.py and osm_places.py; the .pbf can be deleted once those two have run.

Usage: C:\\Python39\\python.exe helper1m\\scripts\\iran\\download.py
Idempotent: files already on disk are skipped.
"""
import sys
import time
from pathlib import Path

import requests

HELPER = Path(__file__).resolve().parents[2]
RAW = HELPER / "data" / "iran" / "raw"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) helper1m-build/1.0"}

A90 = "http://www.amar.org.ir/Portals/0/sarshomari90/Files/ABADY-90/"
A95 = "https://www.amar.org.ir/Portals/0/census/1395/results/abadi/"
REVISED_90 = {"02", "04", "07"}


def jobs():
    for i in range(31):
        nn = f"{i:02d}"
        name = f"os{nn}-r.xls" if nn in REVISED_90 else f"os{nn}.xls"
        yield A90 + name, RAW / "abadi90" / name
        name = f"CN95_HouseholdPopulationVillage_{nn}_r.xlsx"
        yield A95 + name, RAW / "abadi95" / name
    # SCI's settlement file for the 1400 divisions (469 counties): every village with its
    # 1395-census code, under its 1400 county and district (bakhsh.py's bridge)
    yield ("https://amar.org.ir/Portals/0/GEO/2024/GEO1400.xlsx", RAW / "sci" / "GEO1400.xlsx",
           "2025")
    # SCI's 1403 provincial estimate, as reproduced by Iran Open Data (Cloudflare-walled live)
    yield ("https://iranopendata.org/fa/dataset/iod-06124-estimated-population-iran-province-2024/",
           RAW / "sci" / "iod-06124.html", "20250614210551")


def get(url, dst, ts="2020"):
    if dst.exists() and dst.stat().st_size > 5000:
        return "have"
    dst.parent.mkdir(parents=True, exist_ok=True)
    wb = f"http://web.archive.org/web/{ts}id_/" + url
    for attempt in range(6):
        try:
            r = requests.get(wb, headers=UA, timeout=180)
            if r.status_code == 200 and len(r.content) > 5000:
                head = r.content[:8]
                if not (head.startswith(b"PK") or head.startswith(b"\xd0\xcf\x11\xe0")
                        or dst.suffix == ".html"):
                    raise RuntimeError(f"unexpected content {head!r}")
                tmp = dst.with_suffix(dst.suffix + ".part")
                tmp.write_bytes(r.content)
                tmp.replace(dst)
                return f"got {len(r.content):,}"
            print(f"    {r.status_code} {len(r.content)} bytes, retry", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"    {type(e).__name__}: {e}, retry", flush=True)
        time.sleep(15 * (attempt + 1))
    return "FAILED"


def main():
    failed = []
    for job in jobs():
        url, dst = job[0], job[1]
        res = get(url, dst, *job[2:])
        print(f"{dst.relative_to(RAW)}: {res}", flush=True)
        if res == "FAILED":
            failed.append(url)
        elif res != "have":
            time.sleep(3)
    if failed:
        print("failed:", *failed, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
