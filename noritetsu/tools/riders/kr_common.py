"""Shared by the Korean sources: positions for a station name on a line, from koreariders'
metro_stations.geojson (read only), and the boxes a name-only match is confined to."""
import json
import re
from pathlib import Path

KR = Path(r"C:\Users\anita\projects\maps\riders\koreariders\data")
SEOUL_RIDERS = Path(r"C:\Users\anita\projects\maps\riders\seoulriders\data")

BOX = {
    "capital": (126.3, 36.7, 127.95, 38.1),
    "busan": (128.75, 35.0, 129.45, 35.6),
    "daegu": (128.2, 35.65, 128.85, 36.2),
    "daejeon": (127.2, 36.2, 127.6, 36.5),
    "gwangju": (126.65, 35.05, 127.05, 35.3),
}

_POS = None


def strip(name):
    name = re.sub(r"\(.*?\)", "", name or "").strip()
    if len(name) > 2 and name.endswith("역"):
        name = name[:-1]
    return name.replace(" ", "")


def positions():
    global _POS
    if _POS is None:
        _POS = {}
        for f in json.loads((KR / "metro_stations.geojson").read_text(encoding="utf-8"))["features"]:
            p = f["properties"]
            x, y = f["geometry"]["coordinates"][:2]
            for line in p["lines"]:
                _POS[(strip(p["station"]), line)] = (x, y)
    return _POS


def where(name, lines):
    """(lon, lat) of `name` on any of koreariders' `lines`, or None."""
    pos = positions()
    for line in lines:
        xy = pos.get((strip(name), line))
        if xy:
            return xy
    return None
