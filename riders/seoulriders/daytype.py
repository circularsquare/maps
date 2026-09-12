# -*- coding: utf-8 -*-
"""Which day the map is drawing, in one place.

`build_od.py --day X` stamps the choice into `data/od_hourly.npz`, and
`build.py` and `validate.py` read it back from there rather than taking a flag
of their own. That is deliberate: the timetable day type has to agree with the
ridership day type, and a mismatch is silent -- weekday riders routed over the
Sunday timetable simply come out looking like a thinner weekday. Making it one
decision, recorded in the file, is the only way it cannot drift.

**What the day types actually are.** The OD file measures station-to-station
pairs on 2023-12-31 and nothing else. Every other day type takes its *pair
structure* from that file and its *volumes* from the day it names -- see
"Getting off New Year's Eve" in README.md. So `nye` is the only fully measured
option; the rest are that measurement re-levelled onto a different day.

`2023-11-16` is 수능, the national university entrance exam. Subway service is
shifted for it, offices open late, and it is the quietest Tue-Thu of the month.
Excluded rather than argued with.
"""

# Weekdays are Tue/Wed/Thu. Monday and Friday both differ from the middle of
# the week enough to be worth leaving out of a "typical weekday", which is the
# usual convention and holds here: in 2023-11 Monday runs ~4% below and Friday
# ~3% above the Tue-Thu mean.
TUE_WED_THU = (1, 2, 3)
SATURDAY = (5,)
SUNDAY = (6,)

# Reference month for the daily marginals. 2023-11 is the ordinary month
# nearest the OD date, so the level and the pair structure come from the same
# season; no public holidays fall in it.
REF_MONTH = "202311"
EXCLUDE_DATES = ("2023-11-16",)

DAYS = {
    "weekday": {
        "service": "DAY",           # 주중주말 code in the timetables
        "congestion": "평일",        # 구분 value in congestion_raw.csv
        "dows": TUE_WED_THU,
        "extend_late": False,
        "label": "a typical weekday",
    },
    "saturday": {
        "service": "SAT",
        "congestion": "토요일",
        "dows": SATURDAY,
        "extend_late": False,
        "label": "a typical Saturday",
    },
    "sunday": {
        "service": "END",
        "congestion": "일요일",
        "dows": SUNDAY,
        "extend_late": False,
        "label": "a typical Sunday",
    },
    "nye": {
        "service": "END",
        "congestion": "일요일",
        "dows": None,               # one named date, not a day-of-week class
        "date": "2023-12-31",
        "extend_late": True,        # the reconstructed post-countdown service
        "label": "New Year's Eve 2023, a Sunday",
    },
}

DEFAULT = "weekday"


def get(name):
    if name not in DAYS:
        raise SystemExit("unknown --day %r; choose from %s"
                         % (name, ", ".join(sorted(DAYS))))
    return DAYS[name]
