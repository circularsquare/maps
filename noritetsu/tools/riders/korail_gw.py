"""South Korea, commuter lines: Korail's 광역철도 승하차 인원 per station, 2025.

The 광역철도 board's 2025 workbook (koreariders' data/gyeongchun/gwangyeok_2025.xlsx, read
only), sheet 승하차인원, part "2) 역별": boardings and alightings per line and station, with
Korail's own daily mean (일평균, the year over 365). One record per line and station; a
station on several Korail commuter lines is the sum of its lines' rows (COMBINE sum).
Positions from koreariders' metro station layer where it has the station on the matching
line; otherwise a name unique inside the line's region.
"""
import openpyxl

from . import kr_common as K

KEY = "korail_gw"
CC = "kr"
FOLDER = "korail_gw"
COMBINE = "sum"
MODES = {"rail", "metro"}
META = {
    "label": "Korail commuter line counts",
    "name": "광역철도 역별 승하차 인원 2025 (Korail commuter lines), via koreariders",
    "url": "https://info.korail.com/",
    "licence": "Korail public statistics (공공누리 type 1, free use with attribution)",
    "counts": "boardings + alightings per day (Korail's daily mean over 365 days), "
              "Korail commuter (광역) lines",
    "note": "",
}
FILE = K.KR / "gyeongchun" / "gwangyeok_2025.xlsx"
YEAR = 2025
ABBREV = {"디엠시": ["디지털미디어시티"]}
# 동해선 names shared with Busan Metro stations nearby (checked by hand 2026-10-09)
FORCE = {"동래": "k5757138369", "좌천": "k1668440061"}
# Korail line -> koreariders' metro layer lines and the region box
LINES = {
    "경부선": (["수도권 1호선"], "capital"), "경인선": (["수도권 1호선"], "capital"),
    "경원선": (["수도권 1호선"], "capital"), "장항선": (["수도권 1호선"], "capital"),
    "경의선": (["수도권 경의중앙선"], "capital"), "중앙선": (["수도권 경의중앙선"], "capital"),
    "분당선": (["수도권 수인분당선"], "capital"), "수인선": (["수도권 수인분당선"], "capital"),
    "과천선": (["수도권 4호선"], "capital"), "안산선": (["수도권 4호선"], "capital"),
    "일산선": (["수도권 3호선"], "capital"), "경춘선": (["수도권 경춘선"], "capital"),
    "ITX-청춘": (["수도권 경춘선"], "capital"), "경강선": (["수도권 경강선"], "capital"),
    "서해선": (["수도권 서해선"], "capital"), "동해선": ([], "busan"), "대경선": ([], "daegu"),
}


def records(raw):
    wb = openpyxl.load_workbook(FILE, read_only=True, data_only=True)
    ws = wb["승하차인원"]
    rows = list(ws.iter_rows(values_only=True))
    start = next(i for i, r in enumerate(rows) if r[0] and "역별" in str(r[0]))
    head = rows[start + 1]
    c_mean = head.index("일평균")
    acc = {}
    line = None
    for r in rows[start + 2:]:
        if r[0]:
            line = str(r[0]).strip()
        name = r[1]
        if not name or r[2] not in ("승차", "하차"):
            continue
        name = str(name).strip()
        try:
            v = float(r[c_mean])
        except (TypeError, ValueError):
            continue
        acc.setdefault((line, name), 0.0)
        acc[(line, name)] += v
    out = []
    for (line, name), n in acc.items():
        if n <= 0:
            continue
        lines, region = LINES.get(line, ([], "capital"))
        xy = K.where(name, lines)
        out.append({"name": name, "alt": ABBREV.get(name, []), "line": line,
                    "x": xy[0] if xy else None,
                    "y": xy[1] if xy else None, "box": K.BOX[region], "n": n, "year": YEAR})
    return out
