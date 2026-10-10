"""Taiwan: Taiwan Railway (TRA) daily gate entries and exits per station.

"臺鐵每日各站點進出站人數" (data.gov.tw 8792): gateInComingCnt + gateOutGoingCnt per station and
day. The newest file is the current year so far (2026, from 1 January to the last day in it);
the figure is the mean over the days in the file, so "year" 2026 is a part year. Station
names and positions from TRA's station list (車站基本資料集, data.gov.tw 33425).
"""
import json
from collections import defaultdict

KEY = "tra"
CC = "tw"
FOLDER = "tra"
COMBINE = "max"
MODES = {"rail"}
META = {
    "label": "Taiwan Railway gate counts",
    "name": "臺鐵每日各站點進出站人數 (daily entries and exits per station), Taiwan Railway",
    "url": "https://data.gov.tw/dataset/8792",
    "licence": "Open Government Data License, version 1.0 (Taiwan)",
    "counts": "gate entries + exits per day, mean over the days published",
    "note": "2026 is January to the latest month published",
}
DAILY = "daily-2026.json"


def records(raw):
    st = {s["stationCode"]: s for s in
          json.loads((raw / "stations.json").read_text(encoding="utf-8"))}
    tot = defaultdict(float)
    days = defaultdict(set)
    for r in json.loads((raw / DAILY).read_text(encoding="utf-8")):
        try:
            n = float(r["gateInComingCnt"]) + float(r["gateOutGoingCnt"])
        except (TypeError, ValueError):
            continue
        tot[r["staCode"]] += n
        days[r["staCode"]].add(r["trnOpDate"])
    alldays = set().union(*days.values())
    year = int(max(alldays)[:4])
    out = []
    for code, t in tot.items():
        s = st.get(code)
        if not s or t <= 0:
            continue
        try:
            lat, lon = (float(v) for v in s["gps"].split())
        except ValueError:
            lat = lon = None
        # a station's mean is over every day in the file: a day it is missing is a day
        # nobody passed its gates (an unstaffed halt), not a gap
        out.append({"name": s["stationName"], "alt": [s.get("stationEName") or ""],
                    "x": lon, "y": lat, "n": t / len(alldays), "year": year,
                    "code": code})
    return out
