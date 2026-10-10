"""South Korea, Seoul-area city railways: KRIC 철도통계 역별 승강차실적, 2024.

riders/seoulriders' data/kric_station_monthly.csv (read only; KRIC www.kric.go.kr, 도시철도
여객수송 > 역별 승강차실적(월)): boardings and alightings per operator, line, station and
month for 13 city-railway operators of the capital area, 2023-2024. The 2024 months are added
and divided by the days in them for an average day (366, or fewer for a station that opened
during the year). KRIC splits a transfer complex per line (고속터미널,
고속터미널(7), 고속터미널(9)); those rows land on one station and are added (COMBINE sum).
Korail's own commuter lines are korail_gw, not this.
"""
import calendar
import csv
from collections import defaultdict

from . import kr_common as K

KEY = "kric"
CC = "kr"
FOLDER = "kric"
COMBINE = "sum"
MODES = {"rail", "metro"}
META = {
    "label": "Korea Rail Information Center counts",
    "name": "KRIC 철도통계, 도시철도 역별 승강차실적 2024, via seoulriders",
    "url": "https://www.kric.go.kr/",
    "licence": "KRIC public statistics (free use with attribution)",
    "counts": "boardings + alightings per day (2024 total / 366), capital-area city railways",
    "note": "",
}
FILE = K.SEOUL_RIDERS / "kric_station_monthly.csv"
YEAR = 2024
DAYS = 366
# 2024 names of stations renamed since, -> the name kr's stations carry
RENAMED = {"뚝섬유원지": ["자양"], "당고개": ["불암산"]}
# 5호선 양평, not 중앙선 양평 (checked by hand 2026-10-09)
FORCE = {"양평": "k6039313683"}


def koreariders_lines(op, line):
    if op == "서울메트로9":
        return ["수도권 9호선"]
    if op == "공항철도":
        return ["수도권 공항철도"]
    if op == "우이신설도시철도":
        return ["수도권 우이신설선"]
    if op == "남서울경전철":
        return ["수도권 신림선"]
    if op in ("네오트랜스(주)", "경기철도", "새서울철도"):
        return ["수도권 신분당선"]
    if op == "김포골드라인":
        return ["수도권 김포골드라인"]
    if op == "의정부경전철":
        return ["수도권 의정부경전철"]
    if op == "인천교통공사" and line in ("1호선", "2호선"):
        return [f"수도권 인천 {line}"]
    if line.endswith("호선"):
        return [f"수도권 {line}"]
    return []


def records(raw):
    acc = defaultdict(float)
    months = defaultdict(set)
    with open(FILE, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["year"] != str(YEAR):
                continue
            try:
                key = (r["operator"], r["line"], r["station"])
                acc[key] += float(r["boardings"]) + float(r["alightings"])
                months[key].add(int(r["month"]))
            except ValueError:
                continue
    out = []
    for (op, line, name), tot in acc.items():
        if tot <= 0:
            continue
        # a station opened during the year (8호선's 별내 extension, August 2024) is averaged
        # over the months it has, not the whole year
        days = sum(calendar.monthrange(YEAR, m)[1] for m in months[(op, line, name)])
        tot = tot * DAYS / days
        half = len(name) // 2
        if len(name) % 2 == 0 and name[:half] == name[half:]:
            name = name[:half]                  # "부평삼거리부평삼거리"
        xy = K.where(name, koreariders_lines(op, line))
        out.append({"name": K.strip(name) or name, "alt": RENAMED.get(K.strip(name), []),
                    "op": op, "line": line,
                    "x": xy[0] if xy else None, "y": xy[1] if xy else None,
                    "box": K.BOX["capital"], "n": tot / DAYS, "year": YEAR})
    return out
