"""South Korea, the other cities' metros: gate counts per station and day (data.go.kr), via
riders/koreariders (read only: data/<city>/counts.csv, CP949).

  busan          부산교통공사 일별 역별 시간대별 승하차 (3057229), 2026-01..07
  daegu          대구교통공사 역별 일별 시간대별 승하차 (15002503), 2026-01..06
  daejeon        대전교통공사 역별 일별 시간대별 통행량 (15060591), 2026-01..07
  gwangju        광주교통공사 역 일 시간대별 승하차 (15060048), 2026-01..07
  busan_gimhae   부산김해경전철 역사별 시간대별 승하차 (15105181), 2024-2025 (2025 used)

Boardings + alightings added over every day in the file and divided by the number of days:
an average over all days, weekends included (koreariders' own layer uses Monday-Friday). The
2026 files are a part year. A transfer station counted per line (Busan's 서면 on lines 1 and 2)
is added (COMBINE sum). Positions from koreariders' metro station layer.
"""
import csv
import re
from collections import defaultdict

from . import kr_common as K

KEY = "kr_metro"
CC = "kr"
FOLDER = "kr_metro"
COMBINE = "sum"
MODES = {"metro", "rail"}
META = {
    "label": "City metro gate counts from data.go.kr",
    "name": "Busan, Daegu, Daejeon and Gwangju metros and the Busan-Gimhae light rail, daily "
            "gate counts (data.go.kr), via koreariders",
    "url": "https://www.data.go.kr/",
    "licence": "공공누리 type 1 (free use with attribution)",
    "counts": "gate entries + exits per day, mean over the days published",
    "note": "2026 is a part year (January to June or July)",
}


def read(path):
    with open(path, encoding="cp949", errors="replace", newline="") as f:
        return list(csv.reader(f))


def num(s):
    try:
        return float(str(s).strip() or 0)
    except ValueError:
        return 0.0


def tally(rows, date, name, kind, total=None, hours=None, keep=lambda d: True):
    tot, days = defaultdict(float), set()
    for r in rows[1:]:
        if len(r) < 4 or r[kind] not in ("승차", "하차"):
            continue
        d = date(r)
        if not keep(d):
            continue
        days.add(d)
        v = num(r[total]) if total is not None else sum(num(c) for c in r[hours:])
        tot[r[name].strip()] += v
    return tot, days


def tally_daegu(rows, *args, **kw):
    """Daegu counts a transfer station per line with the line's digit ("반월당1",
    "반월당2"): one station."""
    tot, days = tally(rows, *args, **kw)
    out = defaultdict(float)
    for k, v in tot.items():
        out[re.sub(r"(?<=[가-힣])[1-3]$", "", k)] += v
    return out, days


CITIES = {
    # city: (file, date, name col, kind col, total col, first hour col, layer prefix, year, keep)
    "busan": ("busan", lambda r: r[2], 1, 4, 5, None, "부산 ", None),
    "daegu": ("daegu", lambda r: r[0] + r[1], 3, 4, len, None, "대구 ", None),
    "daejeon": ("daejeon", lambda r: r[0], 2, 3, None, 4, "대전 ", None),
    "gwangju": ("gwangju", lambda r: r[0], 2, 3, None, 4, "광주 ", None),
    "busan_gimhae": ("busan_gimhae", lambda r: r[1], 2, 0, 3, None, "부산김해경전철",
                     lambda d: d.startswith("2025")),
}
YEARS = {"busan": 2026, "daegu": 2026, "daejeon": 2026, "gwangju": 2026, "busan_gimhae": 2025}
RENAMED = {"어린이회관": ["어린이세상"]}     # Daegu line 3, renamed
REGION = {"busan": "busan", "daegu": "daegu", "daejeon": "daejeon", "gwangju": "gwangju",
          "busan_gimhae": "busan"}


def records(raw):
    pos = K.positions()
    out = []
    for city, (folder, date, c_name, c_kind, c_total, c_hour, prefix, keep) in CITIES.items():
        rows = read(K.KR / folder / "counts.csv")
        if c_total is len:                 # daegu: the last column is the day's total
            c_total = len(rows[0]) - 1
        count = tally_daegu if city == "daegu" else tally
        tot, days = count(rows, date, c_name, c_kind, c_total, c_hour, keep or (lambda d: True))
        for name, t in tot.items():
            if t <= 0:
                continue
            xy = None
            for (st, line), p in pos.items():
                if st == K.strip(name) and line.startswith(prefix):
                    xy = p
                    break
            out.append({"name": name, "alt": RENAMED.get(name, []), "city": city, "op": city,
                        "x": xy[0] if xy else None,
                        "y": xy[1] if xy else None, "box": K.BOX[REGION[city]],
                        "n": t / len(days), "year": YEARS[city]})
    return out
