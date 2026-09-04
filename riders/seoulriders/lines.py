# -*- coding: utf-8 -*-
"""The line registry: one place that knows what lines exist and what they are called.

Four names for the same line, and every one of them is load-bearing:

  id        ours. Short, ASCII, safe as a JSON key and a CSS class.
  kric      what data/kric_urbanrail_timetable.xlsx calls it.
  od        what data/od_2023-12-31.csv calls it -- sometimes two labels, because
            the OD files by *operator* and we file by line.
  osm       a regex for the OSM route relation names, anchored so that
            "수도권 전철 1호선" cannot swallow "인천 도시철도 1호선", and so that
            공항철도 does not pick up the terminal shuttle or the maglev.
  display   what the page shows in Korean.
  en        what the page shows in English, from the operators' own signage.
            Station names come out of OSM's name:en, but there are only 22
            lines and their English names are a matter of house style rather
            than data -- 경의중앙선 is signed "Gyeongui-Jungang Line", not
            "Gyeonguijungang" -- so they are written out here.

Lines 1-9 come from the 서울교통공사 timetable and keep their bare numbers, which
is what the rest of the pipeline already assumes. Everything else arrives via
kric.py.

Colours are the operators' own, from the 수도권 전철 signage scheme. They are
meant to be hand-tuned rather than generated -- keep them stable so the map
reads the same way between builds.
"""

import collections
import re

Line = collections.namedtuple(
    "Line", "id kric od display display_en color capacity source osm")


def _L(id, kric, od, display, en, color, capacity, osm, source="kric"):
    return Line(id, kric, tuple(od), display, en, color, capacity, source, osm)


# Rated capacity (정원) at roughly 160 per car, the figure Seoul Metro
# publishes. Used only to flag a crush load on the map.
LINES = [
    # -- 서울교통공사 timetable, already in the pipeline ---------------------
    _L("1", None, ("1호선",), "1호선", "Line 1", "#0052A4", 1600,
       r"^(수도권 전철|서울 지하철)\s*1호선", "seoul"),
    _L("2", None, ("2호선",), "2호선", "Line 2", "#00A84D", 1600,
       r"^(수도권 전철|서울 지하철)\s*2호선", "seoul"),
    _L("3", None, ("3호선",), "3호선", "Line 3", "#EF7C1C", 1600,
       r"^(수도권 전철|서울 지하철)\s*3호선", "seoul"),
    # 진접선 is the 2022 branch beyond 당고개; the OD files it separately but
    # the trains are line 4 trains and our timetable already carries its stops.
    _L("4", None, ("4호선", "진접선"), "4호선", "Line 4", "#00A5DE", 1600,
       r"^(수도권 전철|서울 지하철)\s*4호선", "seoul"),
    _L("5", None, ("5호선",), "5호선", "Line 5", "#996CAC", 1280,
       r"^(수도권 전철|서울 지하철)\s*5호선", "seoul"),
    _L("6", None, ("6호선",), "6호선", "Line 6", "#CD7C2F", 1280,
       r"^(수도권 전철|서울 지하철)\s*6호선", "seoul"),
    # 7호선(인천) is the 부평구청-석남 stretch that 인천교통공사 runs. Same line,
    # different operator, and the OD labels by operator.
    _L("7", None, ("7호선", "7호선(인천)"), "7호선", "Line 7", "#747F00", 1280,
       r"^(수도권 전철|서울 지하철)\s*7호선", "seoul"),
    _L("8", None, ("8호선",), "8호선", "Line 8", "#E6186C", 960,
       r"^(수도권 전철|서울 지하철)\s*8호선", "seoul"),
    _L("9", None, ("9호선",), "9호선", "Line 9", "#BDB092", 960,
       r"^(수도권 전철|서울 지하철)\s*9호선", "seoul"),

    # -- KRIC 전체_도시철도운행정보 ------------------------------------------
    # 분당선 and 수인선 were joined into one through service in 2020; the OD
    # still files them apart, the timetable does not.
    _L("SB", "수인분당선", ("분당선", "수인선"),
       "수인분당선", "Suin-Bundang Line", "#F5A200", 1600, r"^수인.분당선"),
    _L("GJ", "경의중앙선", ("경의중앙선",),
       "경의중앙선", "Gyeongui-Jungang Line", "#77C4A3", 1600, r"^경의.중앙선"),
    _L("AR", "인천국제공항선", ("공항철도1호선",),
       "공항철도", "Airport Railroad", "#0090D2", 1000, r"^인천국제공항철도"),
    # 신분당선(연장2) is the 신사-신논현 stretch opened 2022.
    _L("SN", "신분당선", ("신분당선", "신분당선(연장2)"),
       "신분당선", "Shinbundang Line", "#D4003B", 960, r"신분당선"),
    _L("UI", "수도권 경량도시철도 우이신설선", ("우이신설선",),
       "우이신설선", "Ui-Sinseol Line", "#B0CE18", 350, r"우이신설선"),
    _L("SL", "수도권 경량도시철도 신림선", ("신림선",),
       "신림선", "Sillim Line", "#6789CA", 350, r"신림선"),
    _L("I1", "인천지하철 1호선", ("인천1호선",),
       "인천 1호선", "Incheon Line 1", "#7CA8D5", 1280, r"^인천 도시철도 1호선"),
    # The KRIC file carries only a headway for this one, so it comes from
    # 인천교통공사's own four CSVs instead. See kric.py.
    _L("I2", None, ("인천2호선",),
       "인천 2호선", "Incheon Line 2", "#ED8B00", 400,
       r"^인천 도시철도 2호선", "incheon2"),
    _L("GP", "김포골드라인", ("김포골드라인",),
       "김포골드라인", "Gimpo Goldline", "#A17800", 350, r"^김포\s*골드라인"),
    _L("GC", "경춘선", ("경춘선",), "경춘선", "Gyeongchun Line",
       "#0C8E72", 1280, r"경춘선"),
    _L("SH", "서해선", ("서해선",), "서해선", "Seohae Line",
       "#8FC740", 960, r"서해선"),
    _L("GG", "경강선", ("경강선",), "경강선", "Gyeonggang Line",
       "#003DA5", 960, r"경강선"),
    _L("UJ", "의정부", ("의정부선",), "의정부경전철", "Uijeongbu LRT",
       "#FDA600", 230, r"^의정부경전철"),
]

BY_ID = collections.OrderedDict((l.id, l) for l in LINES)
BY_KRIC = dict((l.kric, l) for l in LINES if l.kric)

# OD 호선 label -> our line id. This is the mapping that quietly recovers
# 7호선(인천) and 진접선, which were never missing lines at all.
OD_TO_ID = {}
for _l in LINES:
    for _label in _l.od:
        OD_TO_ID[_label] = _l.id

SEOUL_IDS = [l.id for l in LINES if l.source == "seoul"]
EXTRA_IDS = [l.id for l in LINES if l.source != "seoul"]
ALL_IDS = [l.id for l in LINES]

COLORS = dict((l.id, l.color) for l in LINES)
DISPLAY = dict((l.id, l.display) for l in LINES)
DISPLAY_EN = dict((l.id, l.display_en) for l in LINES)
CAPACITY = dict((l.id, l.capacity) for l in LINES)

_OSM = [(l.id, re.compile(l.osm)) for l in LINES if l.osm]


def osm_line_of(relname):
    """Which of our lines an OSM route relation belongs to, or None."""
    n = (relname or "").strip()
    for lid, rx in _OSM:
        if rx.search(n):
            return lid
    return None


def order_key(line_id):
    """Lines 1-9 first in numeric order, then the rest as listed."""
    return (0, int(line_id), "") if line_id.isdigit() else (1, 0, line_id)


# 에버라인 is deliberately absent. The KRIC file gives it operating windows and
# a headway rather than a timetable ("운영시간 - 17:00 ~ 20:00 운행간격 - 4분"),
# 국가철도공단 publishes its stations but not its schedule, and 용인시 publishes
# only ridership. It is 0.05% of the day's trips. Rather than invent stop times
# for it, leave it out and say so. See README, "Extending past lines 1-9".
NO_SCHEDULE = {"에버라인선": "headway only, no published stop times"}
