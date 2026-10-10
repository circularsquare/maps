"""South Korea, intercity: Korail's 역별 승하차 from the 2023 철도통계연보, via riders/koreariders.

koreariders' data/stations.geojson (build_stations.py, read only): sheet 8 of the yearbook's
"4. 수송(여객)", every station's boardings and alightings in both directions summed and divided
by 365, for the intercity trains Korail counts there (KTX, SRT, ITX, 무궁화 ...; the
commuter 광역철도 lines are a separate count, korail_gw). Positions are koreariders' OSM nodes.
"""
import json
from pathlib import Path

KEY = "korail"
CC = "kr"
FOLDER = "korail"
COMBINE = "max"
MODES = {"rail"}
META = {
    "label": "Korail statistics yearbook",
    "name": "철도통계연보 2023 (Korail railway statistics yearbook), 역별 승하차, via koreariders",
    "url": "https://info.korail.com/",
    "licence": "Korail public statistics (공공누리 type 1, free use with attribution)",
    "counts": "boardings + alightings per day, intercity trains (annual / 365)",
    "note": "",
}
SRC = Path(r"C:\Users\anita\projects\maps\riders\koreariders\data\stations.geojson")
YEAR = 2023


def records(raw):
    out = []
    for f in json.loads(SRC.read_text(encoding="utf-8"))["features"]:
        p = f["properties"]
        if not p.get("riders"):
            continue
        x, y = f["geometry"]["coordinates"][:2]
        out.append({"name": p["station"], "x": x, "y": y, "n": float(p["riders"]),
                    "year": YEAR})
    return out
