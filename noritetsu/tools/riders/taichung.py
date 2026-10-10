"""Taiwan, Taichung MRT: 臺中捷運各站旅運量 (data.gov.tw 175718), monthly entries and exits per
station, from Taichung City's open data centre.

Some months publish only exits; a station's figure uses only the months that have both its
entries and its exits (2025-11 to 2026-08 downloaded; those with both: 2026-02, 03, 05, 06,
07, 08 when this was written), added and divided by the days in those months: gate entries +
exits on an average day, a part year labelled 2026. Names are matched inside Taichung's box
(the file has no positions).
"""
import calendar
import json
from collections import defaultdict

KEY = "taichung"
CC = "tw"
FOLDER = "taichung"
COMBINE = "max"
MODES = {"metro", "rail"}
META = {
    "label": "Taichung MRT station counts",
    "name": "臺中捷運各站旅運量 (Taichung MRT entries and exits per station), Taichung City",
    "url": "https://data.gov.tw/dataset/175718",
    "licence": "Open Government Data License, version 1.0 (Taiwan)",
    "counts": "gate entries + exits per day, mean over the months with both published",
    "note": "a part year",
}
BOX = (120.55, 24.05, 120.8, 24.25)


def records(raw):
    tot = defaultdict(lambda: defaultdict(dict))     # station -> month -> {in, out}
    for p in sorted(raw.glob("*.json")):
        for r in json.loads(p.read_text(encoding="utf-8")):
            item = r.get("項目", "")
            kind = "in" if "入站" in item else "out" if "出站" in item else None
            if not kind:
                continue
            try:
                v = float(r["數值"])
            except (TypeError, ValueError):
                continue
            month = r["資料時間日期"][:7]
            tot[r["欄位名稱"].strip()][month][kind] = v
    out = []
    for name, months in tot.items():
        both = {m: d for m, d in months.items() if "in" in d and "out" in d}
        if not both:
            continue
        days = sum(calendar.monthrange(int(m[:4]), int(m[5:7]))[1] for m in both)
        n = sum(d["in"] + d["out"] for d in both.values()) / days
        if n <= 0:
            continue
        out.append({"name": name, "box": BOX, "n": n, "year": int(max(both)[:4])})
    return out
