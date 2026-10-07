"""Download Census of India 2011 table C-16 (population by mother tongue), one xlsx per state.

The tables sit on the census NADA catalogue, one catalogue entry per state, ids from 10191
(India) upwards. Each entry's related-materials page names its DDW-C16-STMT-MDDS-<ss>00 file
and a download id. censusindia.gov.in has a broken TLS chain, so certificate checks are off
for this host only.

    python sources/in_fetch.py            # -> data/raw/in/DDW-C16-STMT-MDDS-<ss>00.xlsx
"""
import re
import ssl
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "in"
BASE = "https://censusindia.gov.in/nada/index.php/catalog/{}/related-materials"
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
UA = {"User-Agent": "Mozilla/5.0"}


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, context=CTX, timeout=120) as r:
        return r.read()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    found = {}
    misses = 0
    cat = 10191
    while misses < 6 and cat < 10300:
        try:
            html = get(BASE.format(cat)).decode("utf-8", "replace")
        except Exception as e:
            print(f"  {cat}: {e}")
            misses += 1
            cat += 1
            continue
        title = re.search(r"<title>([^<]*)", html)
        title = title.group(1).strip() if title else ""
        name = re.search(r"(DDW-C16-STMT-MDDS-\d{4})\.xlsx", html, re.I)
        link = re.search(r'href="(https://censusindia\.gov\.in/nada/index\.php/catalog/\d+/download/\d+)"', html)
        if not (name and link) or "C-16" not in title:
            print(f"  {cat}: not a C-16 entry ({title[:60]})")
            misses += 1
            cat += 1
            continue
        misses = 0
        fn = name.group(1).upper() + ".xlsx"
        out = RAW / fn
        if not out.exists():
            data = get(link.group(1))
            if data[:2] != b"PK":
                raise SystemExit(f"{cat}: {fn} did not download as an xlsx ({data[:60]!r})")
            out.write_bytes(data)
            time.sleep(0.5)
        found[fn] = title
        print(f"  {cat}: {fn}  {title}")
        cat += 1
    print(f"{len(found)} files in {RAW}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
