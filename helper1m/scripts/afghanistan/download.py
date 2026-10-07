"""Download the GSIA (formerly NSIA) population estimate files and COD-PS 2021
into helper1m/data/afghanistan/raw/. Skips files already there.

gsia.gov.af:8443 serves its certificate without the intermediate, so a plain
verified request fails. Rather than switch verification off, this fetches the
missing intermediate (Certum DV TLS G2 R39 CA) from the address printed in the
server's own certificate and verifies against certifi's roots plus that.

The site is WordPress behind an Angular front end; its media API lists every
upload (wp-json/wp/v2/media?search=...), which is how these were found.
"""
import ssl
import urllib.request
from pathlib import Path

import certifi
import requests

HELPER = Path(__file__).resolve().parents[2]
RAW = HELPER / "data" / "afghanistan" / "raw"
BUNDLE = RAW / "ca_bundle.pem"
INTERMEDIATE = "http://certumdvtlsg2r39ca.repository.certum.pl/certumdvtlsg2r39ca.cer"
UA = {"User-Agent": "Mozilla/5.0"}
G = "https://gsia.gov.af:8443/wp-content/uploads/"
FILES = {
    "nsia_1405_v4.pdf": G + "2026/08/براورد-نفوس-1405-1.pdf",          # 2026-08-25, used
    "nsia_1404.xlsx": G + "2023/01/براورد-نفوس-کشور-1404.xlsx",
    "nsia_1404.pdf": G + "2025/09/براورد-نفوس-کشور-سال-1404.pdf",
    "nsia_1403.xlsx": G + "2023/01/براورد-نفوس-کشور-بابت-سال-1403.xlsx",
    "nsia_1403.pdf": G + "2024/10/براورد-نفوس-کشور-سال-1403.pdf",
    **{f"nsia_{y}.xlsx": G + f"2024/01/برآورد-نفوس-کشور-بابت-سال-{y}.xlsx" for y in range(1390, 1403)},
    "afg_admpop_2021_v2.xlsx": "https://data.humdata.org/dataset/79484094-6e4a-4fbd-9152-66b42a9b32e0/"
                               "resource/6ee580c6-26f2-4eb1-82ba-5a7bac30f2ce/download/afg_admpop_2021_v2.xlsx",
}


def bundle():
    if not BUNDLE.exists():
        der = urllib.request.urlopen(INTERMEDIATE, timeout=60).read()
        BUNDLE.write_text(Path(certifi.where()).read_text() + "\n" + ssl.DER_cert_to_PEM_cert(der))
    return str(BUNDLE)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    verify = bundle()
    for name, url in FILES.items():
        out = RAW / name
        if out.exists():
            continue
        r = requests.get(url, headers=UA, verify=verify, timeout=300)
        print(name, r.status_code, len(r.content))
        r.raise_for_status()
        out.write_bytes(r.content)


if __name__ == "__main__":
    main()
