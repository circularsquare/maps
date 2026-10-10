"""Netherlands: NS in- en uitstappers per station, average working day, 2000-2024.

Published by the province of Zuid-Holland as "In en uitstappers treinstations t/m 2024"
(CC0), NS's own figures: passengers getting on + off on an average working day (Monday to
Friday), all stations in the country from 2013, NS trains only (stations served only by
Arriva, Keolis or Qbuzz are not in it). The latest year per station; positions from the file.
"""
import json

KEY = "ns"
CC = "nl"
FOLDER = "ns"
COMBINE = "max"
MODES = {"rail"}
META = {
    "label": "NS station counts",
    "name": "In- en uitstappers treinstations t/m 2024 (NS figures), Provincie Zuid-Holland",
    "url": "https://opendata.zuid-holland.nl/geonetwork/srv/dut/catalog.search#/metadata/"
           "BB83DE3B-C29F-4F7E-A017-189F9622170B",
    "licence": "CC0 1.0",
    "counts": "passengers getting on + off on an average working day, NS trains",
    "per": "weekday",
    "note": "",
}
FILE = "in_en_uitstappers.geojson"


def records(raw):
    d = json.loads((raw / FILE).read_text(encoding="utf-8"))
    out = []
    for f in d["features"]:
        p = f["properties"]
        year, n = None, None
        for y in range(2030, 1999, -1):
            v = p.get(f"jaar_{y}")
            if v:
                year, n = y, float(v)
                break
        if not n:
            continue
        x, y = f["geometry"]["coordinates"][:2]
        name = p["station"].strip()
        alt = [name.replace("a/d", "aan den").replace(" v ", " van ")]
        out.append({"name": name, "alt": alt, "x": x, "y": y, "n": n, "year": year})
    return out
