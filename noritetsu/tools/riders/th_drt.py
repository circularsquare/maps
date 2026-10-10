"""Thailand, Bangkok: Department of Rail Transport (DRT) passengers per station, Airport Rail
Link only, 2024.

"ปริมาณผู้โดยสารระบบรถไฟฟ้าขนส่งมวลชนในเขตกรุงเทพมหานครและปริมณฑล รายสถานี" (datagov.mot.go.th
drt2566_02, Open Data Common; compiled from SRT reports): Passenger_Arrival and
Passenger_Departure per station and day. Despite the title, every row is the Airport Rail
Link (A1 Suvarnabhumi to A8 Phaya Thai); DRT publishes BTS, MRT and the SRT Red Lines only as
per-operator totals (stat_pass_rail), not per station. The figure is arrivals + departures
(exits + entries), summed over 2024 and divided by the days published in 2024 (365: one
August day is missing).

The resource's CSV (drt.gdcatalog.go.th, updated May 2026) did not connect from here; this
reads the portal's datastore copy, which the portal itself marks incomplete and which ends
on 28 February 2025, saved as drt2566_02.json by
  https://datagov.mot.go.th/api/3/action/datastore_search?resource_id=266f7db8-6f66-4fce-8f31-103eaf336a0b&limit=32000
so 2024 is the latest full year in it. The file has Thai names and codes A1-A8, no positions;
names are matched inside Bangkok among stations a rail-class line stops at (the Airport Rail
Link is a "train" line in th).
"""
import json
from collections import defaultdict

KEY = "th_drt"
CC = "th"
FOLDER = "th_drt"
COMBINE = "max"
MODES = {"rail"}
META = {
    "label": "Thai Department of Rail Transport counts",
    "name": "DRT ปริมาณผู้โดยสารระบบรถไฟฟ้าขนส่งมวลชน รายสถานี (drt2566_02), Airport Rail Link, 2024",
    "url": "https://datagov.mot.go.th/dataset/drt2566_02",
    "licence": "Open Data Common (Department of Rail Transport, Thailand)",
    "counts": "arrivals + departures (exits + entries) per day, mean over the 2024 days "
              "published; Airport Rail Link only",
    "note": "BTS, MRT and the SRT Red Lines have no per-station figures in DRT's open data",
}
FILE = "drt2566_02.json"
YEAR = 2024
BOX = (100.3, 13.5, 100.95, 14.1)       # Bangkok and vicinity
FORCE = {
    "มักกะสัน": "n6711023184",     # ARL Makkasan (A6), not SRT's Makkasan on the Eastern line
}


def records(raw):
    rows = json.loads((raw / FILE).read_text(encoding="utf-8"))["result"]["records"]
    tot = defaultdict(float)
    names = {}
    days = set()
    for r in rows:
        d = str(r.get("Date"))[:10]
        if d[:4] != str(YEAR):
            continue
        days.add(d)
        try:
            n = float(r["Passenger_Arrival"]) + float(r["Passenger_Departure"])
        except (TypeError, ValueError):
            continue
        tot[r["Station_Code"]] += n
        names[r["Station_Code"]] = r["Station_Name"].strip()
    return [{"name": names[c], "code": c, "box": BOX, "n": t / len(days), "year": YEAR}
            for c, t in sorted(tot.items()) if t > 0]
