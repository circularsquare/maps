"""Downloads for the land hazard band (docs/hazard.md), into data/raw/. Files on disk are skipped.

    python isochrone/tools/hazard/fetch.py [chelsa] [osm] [ice]     (all three by default)

chelsa  CHELSA V2.1 1981-2010: Jan/Jul tasmax and tasmin, annual rain bio12 -> data/raw/chelsa/ (990 MB, ~4 min)
osm     OSM wells, springs, settlements from QLever's planet endpoint -> data/raw/osm_water/ (~2 min)
ice     RGI 6.0 glacier outlines for the high Arctic regions (OGGM's open mirror) and Natural Earth
        10 m glaciated areas (the Greenland ice sheet, which RGI leaves out) -> data/raw/ice/ (~100 MB)
"""
import os
import re
import sys
import time
import zipfile

import requests

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(HERE, "data", "raw")
UA = {"User-Agent": "isochrone-map-build/1.0 (a travel-time map)"}

CHELSA = "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/"
CHELSA_FILES = (["climatologies/%s/1981-2010/CHELSA_%s_%s_1981-2010_V.2.1.tif" % (v, v, m)
                 for v in ("tasmax", "tasmin") for m in ("01", "07")] +
                ["bioclim/bio12/1981-2010/CHELSA_bio12_1981-2010_V.2.1.tif"])

QLEVER = "https://qlever.dev/api/osm-planet"
QUERY = """PREFIX osmkey: <https://www.openstreetmap.org/wiki/Key:>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
SELECT (geof:centroid(?wkt) AS ?c) WHERE {
  ?osm osmkey:%s "%s" .
  ?osm geo:hasGeometry/geo:asWKT ?wkt .
}
"""
OSM_TAGS = [("man_made", "water_well"), ("natural", "spring"), ("place", "city"), ("place", "town"),
            ("place", "village"), ("place", "hamlet")]
POINT = re.compile(r"POINT\(([-0-9.eE]+) ([-0-9.eE]+)\)")

RGI = "https://cluster.klima.uni-bremen.de/~oggm/rgi/www.glims.org/RGI/rgi60_files/"
RGI_FILES = ["03_rgi60_ArcticCanadaNorth.zip", "05_rgi60_GreenlandPeriphery.zip", "07_rgi60_Svalbard.zip",
             "09_rgi60_RussianArctic.zip"]
NE_GLACIERS = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_glaciated_areas.zip"


def get(url, path):
    if os.path.exists(path):
        print(path, "on disk", flush=True)
        return
    t0 = time.time()
    n = 0
    with requests.get(url, stream=True, timeout=600, headers=UA) as r:
        r.raise_for_status()
        with open(path + ".part", "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
                n += len(chunk)
    os.replace(path + ".part", path)
    print("%s %.0f MB %.0f s" % (os.path.basename(path), n / 1e6, time.time() - t0), flush=True)


def chelsa():
    out = os.path.join(RAW, "chelsa")
    os.makedirs(out, exist_ok=True)
    for f in CHELSA_FILES:
        get(CHELSA + f, os.path.join(out, os.path.basename(f)))


def osm():
    out = os.path.join(RAW, "osm_water")
    os.makedirs(out, exist_ok=True)
    for key, value in OSM_TAGS:
        path = os.path.join(out, "%s_%s.tsv" % (key, value))
        if os.path.exists(path):
            print(path, "on disk", flush=True)
            continue
        for attempt in range(4):
            t0 = time.time()
            try:
                r = requests.post(QLEVER, data={"query": QUERY % (key, value)},
                                  headers=dict(UA, Accept="text/tab-separated-values"), timeout=1800)
            except requests.RequestException as ex:
                print("  %s=%s error %s" % (key, value, ex), flush=True)
                time.sleep(60 * (attempt + 1))
                continue
            if r.status_code != 200:
                print("  %s=%s failed %d: %s" % (key, value, r.status_code, r.text[:200]), flush=True)
                time.sleep(60 * (attempt + 1))
                continue
            rows = [m.group(2) + "\t" + m.group(1) for m in map(POINT.search, r.text.splitlines()[1:]) if m]
            with open(path + ".tmp", "w", encoding="utf-8") as fh:
                fh.write("\n".join(rows) + "\n")     # lat, lon
            os.replace(path + ".tmp", path)
            print("%s=%s: %d points, %.0f s" % (key, value, len(rows), time.time() - t0), flush=True)
            break
        else:
            print("GAVE UP %s=%s" % (key, value), flush=True)


def ice():
    out = os.path.join(RAW, "ice")
    os.makedirs(out, exist_ok=True)
    for f in RGI_FILES:
        try:
            get(RGI + f, os.path.join(out, f))
        except requests.RequestException as ex:      # the mirror is sometimes down; NE alone still covers the ice caps
            print("  RGI %s not fetched: %s" % (f, type(ex).__name__), flush=True)
    p = os.path.join(out, os.path.basename(NE_GLACIERS))
    get(NE_GLACIERS, p)
    for f in RGI_FILES + [os.path.basename(NE_GLACIERS)]:
        d = os.path.join(out, f[:-4])
        if os.path.exists(os.path.join(out, f)) and not os.path.isdir(d):
            with zipfile.ZipFile(os.path.join(out, f)) as z:
                z.extractall(d)


def main():
    which = sys.argv[1:] or ["chelsa", "osm", "ice"]
    for w in which:
        {"chelsa": chelsa, "osm": osm, "ice": ice}[w]()
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
