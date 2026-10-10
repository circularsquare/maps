"""New South Wales: Opal entries and exits per station, Sydney Trains, NSW TrainLink and Metro.

"Train, Metro and Light Rail Station Monthly Usage from November 2024 - Based on Opal Batch
Data" (Transport for NSW open data, dataset train-station-entries-and-exits-data, CC BY):
one row per station, month and direction (Entry / Exit) with the month's Opal taps. Taken
here: the financial year July 2025 - June 2026 ("year" 2025), Station_Type "Train", "Metro"
and "Metro Shared" (Central, Chatswood, Epping, Martin Place, Sydenham: one gateline for
Sydney Trains and Sydney Metro, so one figure for both). Light rail stops are the source
nsw_lr, from the same file.

n = (entries + exits) / days, where days are the days of the months counted. A station with
rows for only part of the year (opened, closed or renamed) is averaged over its months
without its first and last month, which are likely part months (Bank Street light rail stop:
February - June 2026). The Bankstown line stations closed for metro conversion since
September 2024 keep only stray taps, all "less than 50", and get nothing. Opal taps only.

Months below 50 taps are published as "Less than 50". Such a month counts as 0 where all
of them together could add at most a tenth to the station's figure; otherwise the station
gets a band "lo-hi" a day (lo: those months as 0; hi: as 49) instead of a figure; a station
with every month masked gets nothing.

The file has no positions; names are matched by whole name inside a box around NSW (FORCE
where that is not unique or wrong). COMBINE "sum": records share an id only through FORCE,
where au's one node stands for two separately gated stations.
"""
import calendar
import csv
from collections import defaultdict

KEY = "nsw"
CC = "au"
FOLDER = "nsw"
MODES = {"rail", "metro"}
COMBINE = "sum"
META = {
    "label": "Transport for NSW Opal counts",
    "name": "Train, Metro and Light Rail Station Monthly Usage (Opal), Transport for NSW",
    "url": "https://opendata.transport.nsw.gov.au/data/dataset/"
           "train-station-entries-and-exits-data",
    "licence": "Creative Commons Attribution 4.0 (Transport for NSW)",
    "counts": "Opal entries + exits per day, July 2025 - June 2026 (days of the months the "
              "station was open), Sydney Trains, NSW TrainLink (Opal area) and Sydney Metro; "
              "Central, Chatswood, Epping, Martin Place and Sydenham are one gateline for "
              "trains and metro",
    "note": "financial year 2025-26; a month under 50 taps is published only as "
            "'less than 50', so very quiet stations get a band",
}
FILE = "entry_exit.csv"
YEAR = 2025
FY = [(2025, m) for m in range(7, 13)] + [(2026, m) for m in range(1, 7)]
BOX = (140.9, -37.6, 153.7, -28.1)
TYPES = {"Train", "Metro", "Metro Shared"}
FORCE = {
    # au names them "Domestic Airport" / "International Airport" (Brisbane has the same two)
    "Domestic": "n433584509",
    "International": "n1572445621",
    # Gadigal (Sydney Metro, under Pitt Street) has no node of its own: au's Town Hall
    # node carries the Metro line. Its own gateline, so added to Town Hall's
    "Gadigal": "n1763901273",
}


def clean(name):
    name = " ".join(name.split())
    for t in (" Light Rail", " Station"):
        if name.endswith(t):
            name = name[: -len(t)]
    return name


def parse(raw, types):
    tot = defaultdict(int)
    masked = defaultdict(int)
    months = defaultdict(set)
    part = defaultdict(int)             # (station, month) -> taps
    part_masked = defaultdict(int)      # (station, month) -> "less than 50" cells
    with open(raw / FILE, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["Station_Type"] not in types or r["Station"] == "UNKNOWN":
                continue
            ym = (int(r["MonthYear"][:4]), int(r["MonthYear"][5:7]))
            if ym not in FY:
                continue
            st = r["Station"]
            months[st].add(ym)
            t = r["Trip"].replace(",", "").strip()
            if t.isdigit():
                tot[st] += int(t)
                part[(st, ym)] += int(t)
            else:
                masked[st] += 1
                part_masked[(st, ym)] += 1
    out = []
    for st, ms in months.items():
        # a station open for only part of the year: its first month (if after July) and
        # last month (if before June) are likely part months, so they are left out
        ms = sorted(ms)
        if len(ms) < len(FY):
            keep = [ym for ym in ms if not (ym == ms[0] and ym != FY[0])
                    and not (ym == ms[-1] and ym != FY[-1])]
            if not keep:
                continue
            for ym in set(ms) - set(keep):
                tot[st] -= part[(st, ym)]
                masked[st] -= part_masked[(st, ym)]
            ms = keep
        days = sum(calendar.monthrange(y, m)[1] for y, m in ms)
        cells = 2 * len(ms)
        if masked[st] >= cells:
            continue                    # every month under 50 taps: nothing to show
        lo = tot[st] / days
        hi = (tot[st] + 49 * masked[st]) / days
        rec = {"name": clean(st), "box": BOX, "year": YEAR}
        if masked[st] and hi > lo * 1.1:
            rec["band"] = f"{int(lo)}-{max(int(hi + 0.999), int(lo) + 1)}"
        else:
            rec["n"] = lo
        out.append(rec)
    return out


def records(raw):
    return parse(raw, TYPES)
