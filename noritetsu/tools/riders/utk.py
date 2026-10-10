"""Poland: UTK "wymiana pasażerska" (average daily boardings + alightings per station).

NOT YET RUN. dane.utk.gov.pl and utk.gov.pl answered every request on 2026-10-09 with an
Incapsula bot-protection page (HTTP 403), and the table is not on dane.gov.pl (UTK's datasets
there are line and operator statistics, no per-station file). Until a file is in
data/raw/riders/utk/ this module is skipped (READY). Download the latest year's station table
(XLSX or CSV) from
    https://dane.utk.gov.pl/sts/wymiana-pasazerska
into data/raw/riders/utk/ and run `python tools/station_riders.py utk`; then read
data/raw/riders/_reports/utk.txt, since the layout below is a guess at the file, not read from
it.

Expected: one row per station or halt, a name column (header containing "stacj", "przystan"
or "nazwa") and a figure column (header containing "wymian", "średni" or "dobow"). UTK gives
big stations an exact average-day figure of people getting on + off (taken as n, no
conversion) and small ones a band ("0-9", "10-19", "20-49" ...), kept exactly as published as
"band". The year is the 20xx in the file name. The file carries no positions, so a name
matches only where exactly one Polish rail station has that whole name; the rest are in the
report's misses for FORCE.
"""
import csv
import re
from pathlib import Path

KEY = "utk"
CC = "pl"
FOLDER = "utk"
COMBINE = "max"
MODES = {"rail"}
META = {
    "label": "Office of Rail Transport (UTK) station counts",
    "name": "Wymiana pasażerska na stacjach i przystankach, Urząd Transportu Kolejowego",
    "url": "https://dane.utk.gov.pl/sts/wymiana-pasazerska",
    "licence": "UTK open data (CC0 1.0 on dane.gov.pl for UTK's statistics)",
    "counts": "people getting on + off per average day (UTK's own figure, no conversion); "
              "small stations as UTK's bands, kept as published",
    "note": "",
}
ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw" / "riders" / FOLDER
READY = RAW.is_dir() and any(p.suffix.lower() in (".xlsx", ".csv") for p in RAW.iterdir())
FORCE = {}
BAND = re.compile(r"^\s*(?:\d[\d\s]*\s*[-–]\s*\d[\d\s]*|[<>≤≥]\s*\d+|\d+\s*\+|pow\.?\s*\d+)\s*$")


def rows_of(p):
    if p.suffix.lower() == ".csv":
        text = p.read_text(encoding="utf-8-sig")
        delim = ";" if text.count(";") > text.count(",") else ","
        return [list(r) for r in csv.reader(text.splitlines(), delimiter=delim)]
    import openpyxl
    wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        out += [list(r) for r in ws.iter_rows(values_only=True)]
    return out


def records(raw):
    files = sorted((p for p in raw.iterdir() if p.suffix.lower() in (".xlsx", ".csv")),
                   key=lambda p: (re.findall(r"20\d\d", p.name) or ["0"])[-1])
    p = files[-1]
    m = re.findall(r"20\d\d", p.name)
    if not m:
        raise ValueError(f"{p.name}: no year in the file name")
    year = int(m[-1])
    rows = rows_of(p)
    c_name = c_val = hi = None
    for i, r in enumerate(rows[:50]):
        cells = [str(c or "").lower() for c in r]
        names = [j for j, c in enumerate(cells) if re.search(r"stacj|przystan|nazwa", c)]
        vals = [j for j, c in enumerate(cells) if re.search(r"wymian|średni|dobow", c)]
        if names and vals:
            c_name, c_val, hi = names[0], vals[-1], i
            break
    if hi is None:
        raise ValueError(f"{p.name}: no header row with a station name and a figure column")
    out = []
    for r in rows[hi + 1:]:
        if len(r) <= max(c_name, c_val) or not r[c_name]:
            continue
        name = " ".join(str(r[c_name]).split())
        v = r[c_val]
        if isinstance(v, (int, float)):
            out.append({"name": name, "n": float(v), "year": year})
            continue
        s = str(v or "").strip()
        if re.fullmatch(r"\d[\d\s]*([.,]\d+)?", s):
            out.append({"name": name, "n": float(s.replace(" ", "").replace(",", ".")),
                        "year": year})
        elif BAND.match(s):
            out.append({"name": name, "band": s, "year": year})
    return out
