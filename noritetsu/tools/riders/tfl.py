"""London: TfL annualised station entries and exits, 2025 counts (AC2025).

Taken only for the London Underground and the DLR, and only rows counted as whole-station
entry/exit ("Station entry/exit"): Overground and Elizabeth line stations are National Rail
stations ORR already counts, and TfL's boarding/alighting rows are platform counts at them.
Annualised entries + exits / 365 for an average day. FILL_ONLY: where the matched station
already has ORR's figure (one id for a shared station), ORR's stays and TfL's is not added;
TfL's "Paddington TfL" row includes the Elizabeth line, which ORR also counts.

The file has no positions; names are matched inside Greater London's box, where they are
unique (FORCE for the few that are not).
"""
import openpyxl

KEY = "tfl"
CC = "gb"
FOLDER = "tfl"
FILL_ONLY = True
MODES = {"metro", "rail"}
COMBINE = "sum"     # Bank and Monument rows land on one station only through FORCE
META = {
    "label": "Transport for London station counts",
    "name": "TfL station entry/exit counts 2025 (annualised), Transport for London",
    "url": "https://crowding.data.tfl.gov.uk/",
    "licence": "TfL open data licence (Open Government Licence based), "
               "\"Powered by TfL Open Data\"",
    "counts": "entries + exits per day (annualised / 365), Underground and DLR gatelines "
              "or counters",
    "note": "only where ORR has no figure for the station",
}
FILE = "AC2025_AnnualisedEntryExit_public.xlsx"
YEAR = 2025
BOX = (-0.62, 51.25, 0.35, 51.72)
TFL_MODES = {"LU", "DLR"}
FORCE = {
    # two Hammersmith stations, gb has both with the lines in brackets
    "Hammersmith (District)": "n6195731433",
    "Hammersmith (Hammersmith & City)": "n5474926328",
    # TfL counts the joined complex once; gb has Bank (5 lines) and two Monument nodes
    "Bank and Monument": "n1637578440",
}


def clean(name):
    for t in (" TfL", " LU", " NR", " DLR"):
        if name.endswith(t):
            name = name[: -len(t)]
    return (name.replace("(Bak)", "(Bakerloo)").replace("(DIS)", "(District)")
            .replace("(H&C)", "(Hammersmith & City)"))


def records(raw):
    wb = openpyxl.load_workbook(raw / FILE, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    hi = next(i for i, r in enumerate(rows) if r and r[0] == "Mode")
    out = []
    for r in rows[hi + 2:]:
        if not r or r[0] not in TFL_MODES or r[4] != "Station entry/exit":
            continue
        try:
            n = float(r[18])
        except (TypeError, ValueError):
            continue
        name = clean(str(r[3]).strip())
        alt = []
        if " and " in name:                # "Bank and Monument", "Heathrow Terminals 2 and 3"
            alt = [p.strip() for p in name.split(" and ")]
        out.append({"name": name, "alt": alt, "box": BOX, "mode": r[0],
                    "n": n / 365, "year": YEAR})
    return out
