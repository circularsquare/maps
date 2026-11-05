"""CHELSA V2.1 1981-2010 monthly climatologies for the land hazard (docs/hazard.md), into data/raw/chelsa/.

    python isochrone/notes/T-042/fetch_chelsa.py

January and July mean daily maximum and minimum air temperature, ~150 MB each, and annual
precipitation (bio12, 349 MB). CC0 (EnviDat,
doi:10.16904/envidat.228). Files already on disk are skipped.
"""
import os
import sys
import time

import requests

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(HERE, "data", "raw", "chelsa")
BASE = "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/climatologies/%s/1981-2010/CHELSA_%s_%s_1981-2010_V.2.1.tif"
FILES = [(v, m) for v in ("tasmax", "tasmin") for m in ("01", "07")]
BIO = "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/bioclim/%s/1981-2010/CHELSA_%s_1981-2010_V.2.1.tif"
EXTRA = [BIO % ("bio12", "bio12")]      # annual precipitation, for "hot desert" (349 MB)


def main():
    os.makedirs(OUT, exist_ok=True)
    for url in [BASE % (v, v, m) for v, m in FILES] + EXTRA:
        path = os.path.join(OUT, os.path.basename(url))
        if os.path.exists(path):
            print(path, "on disk", flush=True)
            continue
        t0 = time.time()
        with requests.get(url, stream=True, timeout=600, headers={"User-Agent": "isochrone-map-build/1.0"}) as r:
            r.raise_for_status()
            n = 0
            with open(path + ".part", "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
                    n += len(chunk)
        os.replace(path + ".part", path)
        print("%s %.0f MB %.0f s" % (os.path.basename(path), n / 1e6, time.time() - t0), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
