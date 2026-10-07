"""Downloads for helper1m Sri Lanka. Everything lands in helper1m/data/srilanka/raw/;
files already there are kept, so a rerun is offline.

  GN_population.xlsx      DCS, CPH 2024, GN-level population by sex and age (14,008 rows)
  CPH2024_Final_Eng.pdf   DCS, CPH 2024 final report (administrative-structure tables)
  cph2012/<District>_A1.pdf   CPH 2012 district report Table A1, population by DS division
  kontur_population_LK_20231101.gpkg   Kontur 400 m population hexes, for check.py only
  arcgis/<layer>.geojson  DCS cartography unit's public ArcGIS Online layers (2024 census
                          geography): DS divisions (340) and the GN layer's attributes

Usage:  C:\\Python39\\python.exe helper1m\\scripts\\srilanka\\download.py
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HELPER = Path(__file__).resolve().parents[2]
RAW = HELPER / "data" / "srilanka" / "raw"
UA = {"User-Agent": "Mozilla/5.0"}

DCS = "https://www.statistics.gov.lk"
FILES = {
    "GN_population.xlsx": DCS + "/Population/StaticalInformation/CPH2024/GN_population_excel",
    "CPH2024_Final_Eng.pdf": DCS + "//Resource/en/Population/CPH_2024/CPH2024_Final_Eng.pdf",
}
# District folder names as DCS spells them for the 2012 reports.
DISTRICTS_2012 = [
    "Colombo", "Gampaha", "Kalutara", "Kandy", "Matale", "NuwaraEliya", "Galle", "Matara",
    "Hambantota", "Jaffna", "Mannar", "Vavuniya", "Mullaitivu", "Kilinochchi", "Batticaloa",
    "Ampara", "Trincomalee", "Kurunegala", "Puttalam", "Anuradhapura", "Polonnaruwa",
    "Badulla", "Moneragala", "Ratnapura", "Kegalle"]
A1_URL = DCS + "/PopHouSat/CPH2011/Pages/Activities/Reports/District/{}/{}.pdf"
A1_NAMES = ["A1", "Table%20A1"]          # Kandy's tables are "Table A1.pdf"
A1_FOLDER = {"Moneragala": "Monaragala"}
KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
              "kontur_population_LK_20231101.gpkg.gz")

AGOL = "https://services6.arcgis.com/v2E6mIH6KqVu8t9L/arcgis/rest/services/{}/FeatureServer/0/query"
LAYERS = {
    "dsd": ("DSD_POP_DATA_NEW_Update", True),     # 340 DS polygons + 2024 population
    "gnd_attr": ("GND_POP_DATA_update", False),   # 14,008 GN rows, attributes only
}


def get(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if k == tries - 1 or getattr(e, "code", None) == 404:
                raise
            print(f"    retry {url} ({e})")
            time.sleep(3 * (k + 1))


def save(path, url, magic=None):
    if path.exists() and path.stat().st_size > 1000:
        return
    urls = url if isinstance(url, list) else [url]
    for i, u in enumerate(urls):
        print("GET", u)
        try:
            b = get(u)
            break
        except Exception as e:  # noqa: BLE001 - try the next spelling
            if i == len(urls) - 1:
                raise
            print(f"    {e}; trying the next spelling")
    if magic and not b.startswith(magic):
        raise SystemExit(f"{url}: not the expected file type ({b[:40]!r})")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(b)
    tmp.replace(path)


def agol(name, service, geometry):
    out = RAW / "arcgis" / f"{name}.geojson"
    if out.exists():
        return
    feats, offset = [], 0
    while True:
        q = {"where": "1=1", "outFields": "*", "returnGeometry": str(geometry).lower(),
             "outSR": "4326", "f": "geojson", "resultOffset": offset,
             "resultRecordCount": 1000, "orderByFields": "FID"}
        body = json.loads(get(AGOL.format(service) + "?" + urllib.parse.urlencode(q)))
        if "features" not in body:
            raise SystemExit(f"{service}: {body}")
        feats += body["features"]
        print(f"  {service}: {len(feats)}")
        if len(body["features"]) < 1000:
            break
        offset += 1000
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}),
                   encoding="utf-8")


def main():
    for fn, url in FILES.items():
        save(RAW / fn, url)
    for d in DISTRICTS_2012:
        save(RAW / "cph2012" / f"{d}_A1.pdf",
             [A1_URL.format(A1_FOLDER.get(d, d), n) for n in A1_NAMES], b"%PDF")
    gz = RAW / "kontur_population_LK_20231101.gpkg.gz"
    gpkg = gz.with_suffix("")
    if not gpkg.exists():
        import gzip
        import shutil
        save(gz, KONTUR_URL)
        with gzip.open(gz) as src, open(gpkg, "wb") as dst:
            shutil.copyfileobj(src, dst)
        gz.unlink()
    for name, (service, geom) in LAYERS.items():
        agol(name, service, geom)
    print("downloads complete")


if __name__ == "__main__":
    main()
