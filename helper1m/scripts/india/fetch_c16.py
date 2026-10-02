"""Download Census 2011 table C-16, population by mother tongue, for every state.

C-16 is published per state on the census NADA catalogue, one XLSX each, at state and
district level (catalogue 10191 is all-India, 10192-10226 the 35 states and UTs). The
download URL carries a per-file resource id that is not derivable from the state code, so
each catalogue page is fetched and the link read off it.

censusindia.gov.in serves an incomplete certificate chain, so curl, requests and certifi
all fail with "unable to get local issuer certificate". Verification is turned off for this
one host and the payload is checked structurally instead (zip magic, minimum size).

Writes helper1m/data/india/c16/DDW-C16-STMT-MDDS-<ss>00.XLSX  (36 files, gitignored)

Usage:
    python scripts/india/fetch_c16.py        # from helper1m/; skips files already present
"""
import re
import sys
from pathlib import Path

import requests
import urllib3

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HELPER = Path(__file__).resolve().parents[2]
RAW = HELPER / "data" / "india" / "c16"

CAT_FIRST, CAT_LAST = 10191, 10226
CAT_URL = "https://censusindia.gov.in/nada/index.php/catalog/{cid}"
DL_RE = re.compile(r'href="(https://censusindia\.gov\.in/nada/index\.php/catalog/'
                   r'\d+/download/[^"]+)"')
NAME_RE = re.compile(r"DDW-C16-STMT-MDDS-(\d\d)00\.XLSX$", re.I)
MIN_BYTES = 10_000


def main():
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    s = requests.Session()
    s.verify = False
    s.headers["User-Agent"] = "helper1m/1.0 (map project)"
    RAW.mkdir(parents=True, exist_ok=True)

    codes = []
    for cid in range(CAT_FIRST, CAT_LAST + 1):
        page = s.get(CAT_URL.format(cid=cid), timeout=120)
        page.raise_for_status()
        links = sorted(set(DL_RE.findall(page.text)))
        if len(links) != 1:
            sys.exit(f"catalog {cid}: expected 1 download link, found {len(links)}")
        name = links[0].rsplit("/", 1)[-1]
        m = NAME_RE.search(name)
        if not m:
            sys.exit(f"catalog {cid}: unexpected file {name}")
        codes.append(m.group(1))
        dest = RAW / name
        if dest.exists() and dest.stat().st_size >= MIN_BYTES:
            print(f"  have {name}")
            continue
        r = s.get(links[0], timeout=300)
        r.raise_for_status()
        if len(r.content) < MIN_BYTES or r.content[:2] != b"PK":
            sys.exit(f"{name}: not an xlsx ({len(r.content)} bytes, {r.content[:8]!r})")
        dest.write_bytes(r.content)
        print(f"  got  {name}  ({len(r.content):,} bytes)")

    expected = [f"{i:02d}" for i in range(36)]
    if sorted(codes) != expected:
        sys.exit(f"state codes wrong: missing {sorted(set(expected) - set(codes))}")
    print(f"{len(codes)} C-16 files, state codes 00-35 complete")


if __name__ == "__main__":
    main()
