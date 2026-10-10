"""Great Britain: ORR Estimates of station usage, table 1410, April 2024 - March 2025.

"Entries and exits: All tickets" per National Rail station (interchanges left out), an
annual figure divided by 365 for an average day. Stations are named and carry their
three-letter CRS code but no position; the position comes from Wikidata's UK railway station
code (P4755) with its coordinates (P625), cached in data/raw/riders/orr/wd_crs.json.
"""
import csv
import io

from . import wd

KEY = "orr"
CC = "gb"
FOLDER = "orr"
RADIUS_KM = 1.0
COMBINE = "max"
MODES = {"rail"}
META = {
    "label": "Office of Rail and Road station usage estimates",
    "name": "Estimates of station usage 2024-25, table 1410, Office of Rail and Road",
    "url": "https://dataportal.orr.gov.uk/statistics/usage/estimates-of-station-usage/",
    "licence": "Open Government Licence v3.0",
    "counts": "entries + exits per day (annual total / 365), National Rail tickets only",
    "note": "April-March year by its first year",
}
FILE = "table-1410-2024-25.csv"
# source name -> station id, or None to leave it out (checked by hand 2026-10-09)
FORCE = {
    # the only Ravenglass in gb's stations is the 15-inch Ravenglass & Eskdale terminus; the
    # National Rail station's figure is not that railway's
    "Ravenglass for Eskdale": None,
    # Ewenny Road is its own station 0.7 km away; the name match took it to Maesteg
    "Maesteg (Ewenny Road)": "g256427348",
}
YEAR = 2024


def num(s):
    s = (s or "").replace(",", "").strip()
    return float(s) if s and s[0].isdigit() else None


def records(raw):
    rows = wd.sparql("""SELECT ?item ?crs ?coord WHERE {
        ?item wdt:P4755 ?crs ; wdt:P625 ?coord . }""", raw / "wd_crs.json")
    pos = {}
    for r in rows:
        p = wd.point(r.get("coord"))
        if p:
            pos.setdefault(r["crs"].upper(), p)
    text = (raw / FILE).read_text(encoding="utf-8-sig")
    rd = list(csv.reader(io.StringIO(text)))
    hi = next(i for i, r in enumerate(rd) if r and r[0] == "Station name")
    head = [h.replace("\n", " ").strip() for h in rd[hi]]
    c_all = next(i for i, h in enumerate(head) if h.startswith("Entries and exits:")
                 and "All tickets" in h)
    c_tlc = next(i for i, h in enumerate(head) if h.startswith("Three Letter Code"))
    out = []
    for r in rd[hi + 1:]:
        if not r or not r[0].strip():
            continue
        n = num(r[c_all])
        if n is None or n <= 0:
            continue
        crs = r[c_tlc].strip().upper()
        xy = pos.get(crs)
        out.append({"name": r[0].strip(), "code": crs,
                    "x": xy[0] if xy else None, "y": xy[1] if xy else None,
                    "n": n / 365, "year": YEAR})
    return out
