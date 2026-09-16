"""Gibraltar: Census of Gibraltar 2022, Table 42, religion by major residential area.

Reads (or fetches) the census report into data/raw/gi/ and the Kontur extract into
data/raw/micro/, and writes

    data/normalized/gi.csv          the territory's eight answers, geo_level `country`
    data/geo/gi/gi_hexes.gpkg       Kontur 400 m population hexagons cut to Gibraltar's land,
                                    unit = GI
    data/geo/gi/gi_land.gpkg        that land: OSM relation 1278736 less the OSM water polygons

`sources/gi.md` is the write-up; `taxonomy/gi2022.py` is the mapping.

## THE HEXES ARE CUT TO GIBRALTAR'S LAND

Kontur's GI extract is 20 hexes of 0.78 km2, and the ones along the isthmus reach into La Línea:
the first build put a Roman Catholic dot at -5.3504, 36.1585, north of the frontier (sources/gi.md
§11). So each hex is intersected with Gibraltar's land, as `mt_geo.py` cuts Malta's hexes to its
localities, and keeps the share of its people that its land inside Gibraltar is of all its land.
Land is the part of the hex outside the OSM water polygons, which is also what `water.py` takes off
in the scatter; dividing by the whole hex would hand the sea a share of the people. The outline
is OpenStreetMap's (ODbL): Natural Earth's 10m Gibraltar is 7 vertices and 3.7 km2 with 20 of the
first build's 34 dots outside it, and geoBoundaries' (a Sentinel-2 land-cover trace) stops about
200 m short of the frontier on the isthmus. `python sources/gi.py --check-dots` asserts every dot
and ring of both editions is on that land.

## THE TABLE

HM Government of Gibraltar, *Census of Gibraltar 2022*, report published 2025 (528 pages).
Table 42, *Population by Major Residential Area, Religion and Sex*, printed p.174: eight rows
(seven residential areas and `Institutions`) plus a total, by eight answers, persons, males and
females. Table 43 prints the same by the 78 enumeration areas (EAs 1-70 and the institutional
EAs 80-87), pp.175-177. The total row is the report's usually-resident population, 37,936,
which is the whole universe: families of UK servicemen (260) and visitors (1,240) are counted as
persons present only, and UK servicemen are outside the census altogether (p.44).

## DRAWN AS ONE UNIT, AND THE AREA TABLE IS CHECKED BUT NOT DRAWN

Only the total row reaches the normalized file. The seven areas are defined in Appendix 9 as
lists of enumeration areas, and the enumeration areas in Appendix 8 as lists of streets and
housing estates; no map of either is published, and three EAs (11, 63, 64) are listed under two
areas each. The placement layer cannot tell them apart either: Kontur covers Gibraltar with 20
hexes of 0.78 km2, two of which hold 7,119 and 6,875 people, so one hex spans Town Area, Upper
Town and the reclamation estates. At 38 dots this is the microstate ruling's case (Anita,
2026-09-08). `sources/gi.md` §5 has the whole reasoning and the area shares.

## THE CHECKS

`check()` parses Table 42 off the page and asserts it equal to the transcription below; persons
= males + females in every cell; every row and column closes. It then crosses three other
printings: the report's 1970-2022 religion series on p.52 (2012 and 2022 columns), Table 43's
total row, and Table 43's 78 enumeration-area rows, which must rebuild every area in Table 42
through Appendix 9, with the three shared EAs split between their two areas in a way that is
non-negative in every cell. Two things that rebuild turned up are asserted rather than
absorbed: Table 42 counts institutional EA 85 in South District (`EA_IN_SOUTH`), and one cell,
Church of England males in EA 64, is 4 people apart between the two tables (pinned in
`check()`; sources/gi.md §3). The 2012 and 2001 columns of the series equal UNSD table 28 to the
person, other and not stated merged. Question 11 on the form (p.480) is asserted to offer the
eight boxes and no not-stated box.

Usage:
    python sources/gi.py --fetch    the report (gibraltar.gov.gi, parliament.gi copy as fallback)
                                    and Kontur's GI extract
    python sources/gi.py            normalise from data/raw/gi/ and build the hex layer
"""

import csv
import os
import re
import sys
import unicodedata

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "gi")
OUT = os.path.join(ROOT, "data", "normalized", "gi.csv")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import micro                                                   # noqa: E402  Kontur paths, COLUMNS
from fetch_checks import FetchCheckError, check_body, digest   # noqa: E402  shared, not copied

SOURCE_ID = "hmgog_census2022_t42"
YEAR = 2022
BASIS = "self_id"
COLUMNS = micro.COLUMNS

URLS = (
    "https://www.gibraltar.gov.gi/uploads/statistics/2025/Census/"
    "Census%20of%20Gibraltar%202022%20-%20Report.pdf",
    # Laid before Parliament; the same 18,264,524 bytes by Content-Length on 2026-09-15.
    "https://www.parliament.gi/uploads/contents/papers_laid/2025/"
    "census_of_gibraltar_report_2022.pdf",
)
PDF = os.path.join(RAW, "census_of_gibraltar_2022_report.pdf")
SIZE = 18_264_524
DIGEST = "MMRHWPMMCBEKBISSO2X4V2SH7QRZWKSR"   # SHA-1, base32, as the Wayback CDX writes it
PAGES = 528

# 0-based page indices (printed page = index + 1 throughout this report).
PAGE_HEADLINE = 43     # p.44, "usually-resident population ... 37,936"
PAGE_SERIES = 51       # p.52, religion 1970-2022 (Figures 8 and 9)
PAGE_T42 = 173         # p.174
PAGES_T43 = (174, 175, 176)   # pp.175-177
PAGE_FORM = 479        # p.480, individual questions 6-13
PAGE_APP9 = 522        # p.523

# Gibraltar's outline, for cutting the hexes (docstring, THE HEXES ARE CUT TO GIBRALTAR'S LAND).
# The relation takes in the territorial sea as well; the water polygons take that back off.
OSM_RELATION = 1278736
OSM_JSON = os.path.join(RAW, f"osm_relation_{OSM_RELATION}.json")
WATER = os.path.join(os.path.dirname(ROOT), "data", "water-polygons-split-4326",
                     "water_polygons.shp")
LAND_KM2 = (6.3, 7.0)            # Gibraltar is 6.7-6.8 km2 of land; this outline measured 6.59
FRONTIER_LON = -5.35
FRONTIER_LAT = (36.153, 36.157)  # the land's north edge at 5.35 W, the fence; measured 36.1549
EQUAL_AREA = "EPSG:3035"

CATS = ["Roman Catholic", "Church of England", "Other Christian", "Muslim", "Jewish", "Hindu",
        "No Religion", "Other/Not stated"]
AREAS = ["Eastside", "North District", "Reclamation Areas", "Town Area", "Upper Town",
         "Sandpits Area", "South District", "Institutions"]

# Table 42, transcribed: (eight answers in CATS order, total), persons / males / females.
T42 = {
    "Eastside": {
        "T": (356, 55, 13, 14, 4, 2, 52, 1, 497),
        "M": (168, 31, 4, 9, 1, 1, 32, 0, 246),
        "F": (188, 24, 9, 5, 3, 1, 20, 1, 251)},
    "North District": {
        "T": (3826, 436, 286, 450, 85, 99, 1200, 175, 6557),
        "M": (1859, 234, 135, 220, 44, 50, 686, 77, 3305),
        "F": (1967, 202, 151, 230, 41, 49, 514, 98, 3252)},
    "Reclamation Areas": {
        "T": (10345, 906, 580, 338, 330, 380, 1812, 274, 14965),
        "M": (4937, 452, 284, 171, 157, 197, 1044, 130, 7372),
        "F": (5408, 454, 296, 167, 173, 183, 768, 144, 7593)},
    "Town Area": {
        "T": (1432, 289, 142, 658, 490, 117, 581, 74, 3783),
        "M": (645, 157, 74, 369, 246, 74, 323, 36, 1924),
        "F": (787, 132, 68, 289, 244, 43, 258, 38, 1859)},
    "Upper Town": {
        "T": (1753, 206, 110, 234, 24, 5, 466, 62, 2860),
        "M": (868, 98, 52, 121, 9, 2, 258, 30, 1438),
        "F": (885, 108, 58, 113, 15, 3, 208, 32, 1422)},
    "Sandpits Area": {
        "T": (1532, 114, 78, 43, 50, 43, 209, 28, 2097),
        "M": (721, 64, 37, 23, 27, 21, 119, 12, 1024),
        "F": (811, 50, 41, 20, 23, 22, 90, 16, 1073)},
    "South District": {
        "T": (4511, 520, 275, 116, 80, 43, 965, 115, 6625),
        "M": (2222, 252, 132, 63, 41, 23, 519, 64, 3316),
        "F": (2289, 268, 143, 53, 39, 20, 446, 51, 3309)},
    "Institutions": {
        "T": (343, 15, 19, 56, 7, 4, 58, 50, 552),
        "M": (132, 10, 7, 49, 2, 4, 39, 23, 266),
        "F": (211, 5, 12, 7, 5, 0, 19, 27, 286)},
}
TOTAL = {
    "T": (24098, 2541, 1503, 1909, 1070, 693, 5343, 779, 37936),
    "M": (11552, 1298, 725, 1025, 527, 372, 3020, 372, 18891),
    "F": (12546, 1243, 778, 884, 543, 321, 2323, 407, 19045),
}
CENSUS_TOTAL = 37_936

# Appendix 9, transcribed. EA 11, 63 and 64 are listed under two areas each; Institutions is
# EAs 80-87 (Appendix 8: religious institutions, hotels, hostels, hospitals, prison, marinas,
# old people's homes, other institutions), which Appendix 9 does not list.
APPENDIX9 = {
    "Eastside": [1],
    "North District": [2, 3, 4, 5, 6, 7, 8, 9, 12],
    "Reclamation Areas": [64, 65, 66, 67, 68, 69, 70],
    "Town Area": [10, 11] + list(range(13, 34)),
    "Upper Town": [11] + list(range(34, 46)),
    "Sandpits Area": [46, 48, 49, 50, 51, 63],
    "South District": [47] + list(range(52, 65)),
    "Institutions": list(range(80, 88)),
}
APPENDIX9_PRINTED = {        # the label and the list as the page prints them
    "Eastside": ("EAST SIDE:", "1"),
    "North District": ("NORTH DISTRICT:", "2, 3, 4, 5, 6, 7, 8, 9, 12."),
    "Reclamation Areas": ("RECLAMATION AREAS:", "64, 65, 66, 67, 68, 69, 70."),
    "Town Area": ("TOWN AREA:", "10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, "
                                "26, 27, 28, 29, 30, 31, 32, 33"),
    "Upper Town": ("UPPER TOWN:", "11, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45."),
    "Sandpits Area": ("SANDPITS AREA:", "46, 48, 49, 50, 51, 63."),
    "South District": ("SOUTH DISTRICT:", "47, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, "
                                          "63, 64."),
}
SHARED = {11: ("Town Area", "Upper Town"), 63: ("Sandpits Area", "South District"),
          64: ("Reclamation Areas", "South District")}
EA_IDS = list(range(1, 71)) + list(range(80, 88))

# TABLE 42 COUNTS INSTITUTIONAL EA 85 IN SOUTH DISTRICT, NOT IN INSTITUTIONS. Appendix 9 does not
# say so and nothing else in the report does; the arithmetic does. Institutions is short of EAs
# 80-87 by exactly EA 85's row (70 people) in all 19 columns, and South District is over the
# remainders of EAs 63 and 64 by the same row. It is the only placement that closes: EA 11, 63
# and 64's splits are fixed by Town Area and Upper Town, Sandpits and the Reclamation Areas,
# which leaves South District as the only row with room. `check()` finds the EA from the
# shortfall rather than trusting this constant. Appendix 8 names EAs 80-87 only as a group.
EA_IN_SOUTH = 85
TABLE42_EAS = {a: [e for e in lst if e != EA_IN_SOUTH] for a, lst in APPENDIX9.items()}
TABLE42_EAS["South District"] = APPENDIX9["South District"] + [EA_IN_SOUTH]

# The report's religion series, p.52: {label: (2001, 2012, 2022)}. The 2012 and 2001 columns
# are asserted against UNSD table 28 below, other and not stated merged.
SERIES_LABELS = ["Roman Catholic", "Church of England", "Other Christian", "Muslim", "Jewish",
                 "Hindu", "No Religion", "Other/Not Stated", "Total"]
UNSD_LABELS = {"Roman Catholic": "Roman Catholic", "Church of England": "Church of England",
               "Other Christians": "Other Christian", "Muslim": "Muslim", "Jewish": "Jewish",
               "Hindu": "Hindu", "No Religion": "No Religion", "Other": "Other/Not Stated",
               "Not Stated": "Other/Not Stated"}

FORM_BOXES = ["Roman Catholic", "Church of England", "Other Christian", "Muslim", "Jewish",
              "Hindu", "Other", "No religion"]

SPACES = dict.fromkeys([0x00A0, 0x2007, 0x2008, 0x2009, 0x202F, 0x205F], " ")


def despace(s):
    """Copied from sources/gw.py (not shared yet)."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s)).translate(SPACES)).strip()


def _num(tok):
    if tok == "-":
        return 0
    if not re.fullmatch(r"\d{1,3}(?:,\d{3})*", tok):
        raise ValueError(f"not a count: {tok!r}")
    return int(tok.replace(",", ""))


def _text(doc, pno):
    return despace(doc.load_page(pno).get_text())


def fetch():
    import urllib.request

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(PDF) and os.path.getsize(PDF) == SIZE:
        print("already have", PDF)
    else:
        for url in URLS:
            try:
                req = urllib.request.Request(url, headers=micro.UA)
                with urllib.request.urlopen(req, timeout=600) as r:
                    body = r.read()
                check_body(body, "pdf", where=f"{url[:60]}...", pin_digest=DIGEST)
            except (FetchCheckError, OSError) as e:
                print(f"  {url[:60]}...: {e}")
                continue
            with open(PDF + ".part", "wb") as fh:
                fh.write(body)
            os.replace(PDF + ".part", PDF)                    # [[reference_wb_truncates]]
            print(f"wrote {PDF} ({len(body):,} bytes) from {url[:40]}")
            break
        else:
            raise SystemExit("neither gibraltar.gov.gi nor parliament.gi returned the report; "
                             "give Anita the URL (AGENT_BRIEF, blocked downloads)")
    micro.fetch(["gi"])       # Kontur GI 2023-11-01, into data/raw/micro/ as the tier keeps it
    fetch_outline()


def fetch_outline():
    """OSM relation 1278736 in full (the relation, its ways and their nodes) into data/raw/gi/.

    From the main OSM API rather than Overpass: on 2026-09-15 overpass-api.de gave a 504 and two
    mirrors timed out on a one-relation query, and a browser User-Agent got a 406. Not pinned by
    digest, since any OSM edit changes the bytes; `land_outline()` checks what matters instead,
    the relation's tags, one closed outer ring, the land area and the frontier."""
    import json
    import urllib.request

    if os.path.exists(OSM_JSON) and os.path.getsize(OSM_JSON) > 0:
        print("already have", OSM_JSON)
        return
    url = f"https://api.openstreetmap.org/api/0.6/relation/{OSM_RELATION}/full.json"
    req = urllib.request.Request(url, headers={"User-Agent": "religiondots/1.0 (research dot map)"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            body = r.read()
        if not json.loads(body).get("elements"):
            raise ValueError("no elements")
    except (OSError, ValueError) as e:
        raise SystemExit(f"{url}: {e}; give Anita that URL (AGENT_BRIEF, blocked downloads) and "
                         f"save it as {OSM_JSON}")
    with open(OSM_JSON + ".part", "wb") as fh:
        fh.write(body)
    os.replace(OSM_JSON + ".part", OSM_JSON)
    print(f"wrote {OSM_JSON} ({len(body):,} bytes)")


def read_t42(doc):
    """Table 42 off p.174: {area: {"T"|"M"|"F": 9-tuple}}, anchored on the row labels."""
    text = _text(doc, PAGE_T42)
    at = text.find("Table 42. Population by Major Residential Area, Religion and Sex")
    if at < 0:
        raise SystemExit(f"page index {PAGE_T42} is not Table 42")
    labels = AREAS + ["Total"]
    spans, pos = [], text.find("Other/Not stated Total", at) + len("Other/Not stated Total")
    for lab in labels:
        m = re.compile(r"(?<![A-Za-z])" + re.escape(lab) + r"(?![A-Za-z])").search(text, pos)
        if not m:
            raise SystemExit(f"Table 42: no row {lab!r}")
        spans.append((lab, m.start(), m.end()))
        pos = m.end()
    out = {}
    for k, (lab, _s, e) in enumerate(spans):
        end = spans[k + 1][1] if k + 1 < len(spans) else text.find("Note:", e)
        toks = text[e:end].split()
        if len(toks) != 29 or toks[9] != "M" or toks[19] != "F":
            raise SystemExit(f"Table 42 row {lab!r}: unexpected tokens {toks}")
        out[lab] = {"T": tuple(map(_num, toks[0:9])), "M": tuple(map(_num, toks[10:19])),
                    "F": tuple(map(_num, toks[20:29]))}
    return out


def read_t43(doc):
    """Table 43 off pp.175-177: ({EA: (18 M/F counts, T)}, total 19-tuple).

    Each page's text layer holds its printed page number, the M/F/T header letters, the column
    labels and the caption as well as the counts. Counts and dashes are the only tokens matching
    `_num`, less each page's leading page number and the caption's `43.`, so the rows are read
    as a stream of 20 tokens (EA, 9 M/F pairs, T) and the last 19 are the printed total row.
    """
    stream = []
    for pno in PAGES_T43:
        text = _text(doc, pno)
        # The last page's footnote reads "Enumeration Areas - see Appendix 8.", and that bare
        # dash would read as a zero cell.
        toks = text[:text.find("Note:")].split() if "Note:" in text else text.split()
        if toks[0] != str(pno + 1):
            raise SystemExit(f"Table 43 page index {pno}: first token {toks[0]!r} is not its "
                             "printed page number")
        body = [t for t in toks[1:] if t == "-" or re.fullmatch(r"\d{1,3}(?:,\d{3})*", t)]
        stream.extend(body)
    want = len(EA_IDS) * 20 + 19
    if len(stream) != want:
        raise SystemExit(f"Table 43: {len(stream)} count tokens, expected {want}")
    rows = {}
    for i, ea in enumerate(EA_IDS):
        chunk = stream[i * 20:(i + 1) * 20]
        if chunk[0] != str(ea):
            raise SystemExit(f"Table 43: row {i} starts {chunk[0]!r}, expected EA {ea}")
        rows[ea] = tuple(map(_num, chunk[1:]))
    total = tuple(map(_num, stream[-19:]))
    return rows, total


def read_series(doc):
    """p.52's 1970-2022 religion table: {label: (2001, 2012, 2022)}."""
    text = _text(doc, PAGE_SERIES)
    start = text.find("1970 1981 1991 2001 2012 2022 Roman Catholic")
    stop = text.find("0% 10%", start)
    if start < 0 or stop < 0:
        raise SystemExit(f"page index {PAGE_SERIES}: no 1970-2022 religion table")
    body = text[start:stop]
    out, pos = {}, 0
    spans = []
    for lab in SERIES_LABELS:
        i = body.find(lab, pos)
        if i < 0:
            raise SystemExit(f"religion series: no {lab!r}")
        spans.append((lab, i, i + len(lab)))
        pos = i + len(lab)
    for k, (lab, _s, e) in enumerate(spans):
        end = spans[k + 1][1] if k + 1 < len(spans) else len(body)
        nums = re.findall(r"n/a|\d{1,3}(?:,\d{3})*", body[e:end])
        # The Total row's 2001 cell is split in the text layer (`27,49 5`); only the last
        # three columns are read, and 2001 is asserted only for the eight answers.
        out[lab] = tuple(nums[-3:])
    return out


def unsd(year):
    try:
        import oracle
        got = oracle.oracle("Gibraltar", year)
    except SystemExit:
        return None
    if not got or oracle.TOTAL not in got:
        return None
    cats, stated, _exact = oracle.partition(got[oracle.TOTAL])
    merged = {}
    for c, n in cats.items():
        merged[UNSD_LABELS[c]] = merged.get(UNSD_LABELS[c], 0) + n
    merged["Total"] = stated
    return merged


def solve_shared(t43):
    """Rebuild each area from its EAs and split the three shared EAs; return the splits.

    Eastside, North District and Institutions have no shared EA and must equal their EA sums.
    For the rest, in every one of the 19 cells: Town Area's EA-11 part is Town Area less its
    own EAs, Sandpits' EA-63 part likewise, Reclamation's EA-64 part likewise, Upper Town and
    South District take the remainders, and South District must then close exactly.
    """
    def row42(area):
        t, m, f = T42[area]["T"], T42[area]["M"], T42[area]["F"]
        return tuple(x for pair in zip(m[:8], f[:8]) for x in pair) + (m[8], f[8], t[8])

    def ea_cells(ea):
        r = t43[ea]                         # 18 pairs then T
        return r[:16] + (r[16], r[17], r[18])

    def add(*vs):
        return tuple(sum(x) for x in zip(*vs))

    def sub(a, b):
        return tuple(x - y for x, y in zip(a, b))

    def own(area):
        return add(*[ea_cells(e) for e in TABLE42_EAS[area] if e not in SHARED]) \
            if any(e not in SHARED for e in TABLE42_EAS[area]) else (0,) * 19

    # Which institutional EA is not in Institutions: the one whose row is the shortfall.
    short = sub(add(*[ea_cells(e) for e in APPENDIX9["Institutions"]]), row42("Institutions"))
    short_eas = [e for e in APPENDIX9["Institutions"] if ea_cells(e) == short]

    exact = {a: sub(row42(a), own(a)) for a in ("Eastside", "North District", "Institutions")}
    town11 = sub(row42("Town Area"), own("Town Area"))
    upper11 = sub(row42("Upper Town"), own("Upper Town"))
    sand63 = sub(row42("Sandpits Area"), own("Sandpits Area"))
    recl64 = sub(row42("Reclamation Areas"), own("Reclamation Areas"))
    south_rest = sub(row42("South District"), own("South District"))
    south_want = add(sub(ea_cells(63), sand63), sub(ea_cells(64), recl64))
    return dict(short_eas=short_eas, ea85=ea_cells(EA_IN_SOUTH),
                exact=exact, town11=town11, upper11=upper11, ea11=ea_cells(11),
                sand63=sand63, ea63=ea_cells(63), recl64=recl64, ea64=ea_cells(64),
                south_rest=south_rest, south_want=south_want)


def check(doc):
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("Gibraltar: Census of Gibraltar 2022, Table 42\n")
    say(doc.page_count == PAGES, f"the report is {doc.page_count} pages (expected {PAGES})")
    with open(PDF, "rb") as fh:
        body = fh.read()
    say(digest(body) == DIGEST, f"file digest {digest(body)} (expected {DIGEST})")

    # 1. parsed = transcription
    t42 = read_t42(doc)
    bad = [a for a in AREAS if t42[a] != T42[a]]
    say(not bad and t42["Total"] == TOTAL,
        "Table 42 parsed off p.174 equals the transcription: 9 rows x 9 columns x T/M/F"
        + (f"; differ: {bad + ([] if t42['Total'] == TOTAL else ['Total'])}" if bad or
           t42["Total"] != TOTAL else ""))

    # 2. the table's own arithmetic
    rows = dict(T42, Total=TOTAL)
    say(all(r["T"][i] == r["M"][i] + r["F"][i] for r in rows.values() for i in range(9)),
        "persons = males + females in all 81 cells")
    say(all(sum(r[s][:8]) == r[s][8] for r in rows.values() for s in "TMF"),
        "every row's eight answers sum to its total, for persons, males and females")
    say(all(sum(T42[a][s][i] for a in AREAS) == TOTAL[s][i] for s in "TMF" for i in range(9)),
        f"the eight rows sum to the total row in every column; total {TOTAL['T'][8]:,}")
    say(TOTAL["T"][8] == CENSUS_TOTAL, f"Table 42's total is the census's {CENSUS_TOTAL:,}")

    # 3. the headline: 37,936 is the usually-resident population, the whole universe
    head = _text(doc, PAGE_HEADLINE)
    say("usually-resident population of Gibraltar was recorded as 37,936" in head,
        "p.44: 37,936 is the usually-resident population (UK servicemen excluded from the "
        "census; their families, 260, and visitors, 1,240, are persons present only)")

    # 4. the 1970-2022 series on p.52, and UNSD for 2012 and 2001
    series = read_series(doc)
    s22 = {lab: _num(v[2]) for lab, v in series.items()}
    want22 = dict(zip(SERIES_LABELS, TOTAL["T"]))
    say(s22 == want22, "p.52's religion series, 2022 column, equals Table 42's total row"
        + ("" if s22 == want22 else f": {s22}"))
    for year, col in ((2012, 1), (2001, 0)):
        u = unsd(year)
        if u is None:
            print(f"  -- oracle cache not present; UNSD {year} check skipped")
            continue
        got = {lab: (None if v[col] == "n/a" else _num(v[col])) for lab, v in series.items()}
        if year == 2001:
            got.pop("Total")                   # the text layer splits 2001's total cell
            u = {k: v for k, v in u.items() if k != "Total"}
        say(got == u, f"p.52's {year} column equals UNSD table 28 Gibraltar {year} to the "
            "person, other and not stated merged" + ("" if got == u else f": {got} vs {u}"))

    # 5. Table 43: its total row, and the 78 EAs rebuild every area through Appendix 9
    t43, t43_total = read_t43(doc)
    want_total = tuple(x for pair in zip(TOTAL["M"][:8], TOTAL["F"][:8]) for x in pair) \
        + (TOTAL["M"][8], TOTAL["F"][8], TOTAL["T"][8])
    say(t43_total == want_total, "Table 43's printed total row equals Table 42's (M/F by "
        "answer, then M, F, T)")
    say(all(r[16 + 2] == r[16] + r[17] and all(0 <= x for x in r) for r in t43.values())
        and all(sum(r[2 * c] + r[2 * c + 1] for c in range(8)) == r[18] for r in t43.values()),
        "every one of the 78 EA rows closes: M + F = T, answers sum to T")
    say(tuple(sum(r[i] for r in t43.values()) for i in range(19)) == want_total,
        "the 78 EA rows sum to the total row in all 19 columns")

    app9 = _text(doc, PAGE_APP9)
    say(all(lab in app9 and lst in app9 for lab, lst in APPENDIX9_PRINTED.values()),
        "Appendix 9 (p.523) prints the area-to-EA lists as transcribed")
    s = solve_shared(t43)
    say(s["short_eas"] == [EA_IN_SOUTH],
        f"Institutions is short of EAs 80-87 by exactly one EA's row in all 19 columns: EA "
        f"{s['short_eas']} ({s['ea85'][18]:,} people), which Table 42 counts in South District "
        f"(expected EA {EA_IN_SOUTH}; Appendix 9 does not say this)")
    say(all(v == (0,) * 19 for v in s["exact"].values()),
        "Eastside (EA 1), North District (EAs 2-9, 12) and Institutions (EAs 80-84, 86, 87) "
        "equal their EA sums in all 19 columns")
    split_ok = all(0 <= a <= b and 0 <= u <= b and a + u == b
                   for a, u, b in zip(s["town11"], s["upper11"], s["ea11"]))
    say(split_ok, f"EA 11 ({s['ea11'][18]:,} people) splits into Town Area {s['town11'][18]:,} "
        f"and Upper Town {s['upper11'][18]:,}, non-negative and closing in every column")
    # ONE CELL DISAGREES BY FOUR PEOPLE, AND IT IS PINNED. The Reclamation Areas are EAs 65-70
    # plus part of EA 64, so in every column their part of EA 64 is fixed; for Church of England
    # males it is 168 (452 in Table 42 less 284 in EAs 65-70), and EA 64 has 164. Every other
    # split of EAs 11, 63 and 64, in all 19 columns, lies inside its EA. Either 4 of those men
    # are tabulated from somewhere Appendix 9 does not put in the Reclamation Areas (EA 85, with
    # 12, is the only unlisted piece with that many), or the two tables differ by 4 in that
    # cell. Nothing in the report settles it and nothing drawn depends on it (sources/gi.md §3).
    cols = [f"{c} {x}" for c in CATS for x in ("males", "females")] + ["males", "females",
                                                                         "persons"]
    over = [(piece, cols[i], part - ea if part > ea else part)
            for piece, parts, eas in (("EA 63 to Sandpits", s["sand63"], s["ea63"]),
                                      ("EA 64 to the Reclamation Areas", s["recl64"], s["ea64"]))
            for i, (part, ea) in enumerate(zip(parts, eas)) if not 0 <= part <= ea]
    pinned = [("EA 64 to the Reclamation Areas", "Church of England males", 4)]
    say(over == pinned and s["south_rest"] == s["south_want"],
        f"EA 63 ({s['ea63'][18]:,}) gives Sandpits {s['sand63'][18]:,} and EA 64 "
        f"({s['ea64'][18]:,}) the Reclamation Areas {s['recl64'][18]:,}; every one of the 38 "
        f"cells fits inside its EA except the pinned one, {pinned[0][1]} over by 4"
        + ("" if over == pinned else f"; found {over}"))

    # 6. the form
    form = _text(doc, PAGE_FORM)
    q = form.find("11 Religion?")
    boxes = form[q:q + 200]
    say(q >= 0 and all(b in boxes for b in FORM_BOXES)
        and not re.search(r"not stated|prefer not|write in", boxes, re.I),
        "form question 11 (p.480) offers the eight boxes " + ", ".join(FORM_BOXES)
        + ", and no not-stated box or write-in line")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def emit():
    return [dict(geo_id="GI", geo_level="country", geo_name="Gibraltar", source_category=c,
                 count=n, basis=BASIS, year=YEAR, source_id=SOURCE_ID,
                 note="Census of Gibraltar 2022, Table 42 (printed p.174), total row; "
                      f"{100.0 * n / CENSUS_TOTAL:.2f}% of the usually-resident population")
            for c, n in sorted(zip(CATS, TOTAL["T"][:8]), key=lambda kv: -kv[1]) if n > 0]


def report():
    print(f"\n  {'area':<18}{'people':>7}  " + "  ".join(f"{c[:6]:>6}" for c in CATS))
    for a in AREAS + ["Total"]:
        r = T42[a]["T"] if a in T42 else TOTAL["T"]
        print(f"  {a:<18}{r[8]:>7,}  " + "  ".join(f"{100.0 * n / r[8]:6.2f}" for n in r[:8]))
    for c in ("Muslim", "Jewish", "Hindu", "No Religion"):
        i = CATS.index(c)
        top = max(AREAS, key=lambda a: T42[a]["T"][i])
        print(f"  {c}: {T42[top]['T'][i]:,} of {TOTAL['T'][i]:,} "
              f"({100.0 * T42[top]['T'][i] / TOTAL['T'][i]:.1f}%) in {top}")


def land_outline():
    """Gibraltar's land in EPSG:4326: OSM relation 1278736 less the OSM water polygons.

    The relation's outer ways close into one ring that takes in the territorial sea. The water
    polygons are the layer `water.py` subtracts in the scatter, so the coast here is the coast the
    scatter cuts to anyway."""
    import json

    import geopandas as gpd
    import shapely
    from shapely.ops import linemerge, polygonize, unary_union

    if not os.path.exists(OSM_JSON):
        raise SystemExit(f"missing {OSM_JSON}; run: python sources/gi.py --fetch")
    with open(OSM_JSON, encoding="utf-8") as fh:
        els = json.load(fh).get("elements", [])
    # the OSM API's `full` shape: the relation, its member ways as node-id lists, and the nodes
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in els if e.get("type") == "node"}
    way_nodes = {e["id"]: e["nodes"] for e in els if e.get("type") == "way"}
    rels = [e for e in els if e.get("type") == "relation" and e.get("id") == OSM_RELATION]
    if len(rels) != 1:
        raise SystemExit(f"{OSM_JSON} does not hold relation {OSM_RELATION}")
    want = {"ISO3166-1": "GI", "admin_level": "2", "boundary": "administrative"}
    got = {k: rels[0].get("tags", {}).get(k) for k in want}
    if got != want:
        raise SystemExit(f"relation {OSM_RELATION} is no longer Gibraltar's national boundary: {got}")
    outer = [m["ref"] for m in rels[0]["members"]
             if m.get("type") == "way" and m.get("role") == "outer"]
    lacking = [w for w in outer if w not in way_nodes or any(n not in nodes for n in way_nodes[w])]
    if lacking:
        raise SystemExit(f"{OSM_JSON} lacks the nodes of outer ways {lacking[:5]}; delete it and "
                         "run --fetch")
    ways = [shapely.LineString([nodes[n] for n in way_nodes[w]]) for w in outer]
    rings = list(polygonize(linemerge(unary_union(ways))))
    if len(rings) != 1:
        raise SystemExit(f"relation {OSM_RELATION}'s {len(ways)} outer ways close into "
                         f"{len(rings)} polygons, expected 1")
    if not os.path.exists(WATER):
        raise SystemExit(f"missing {WATER}, the OSM water polygons water.py clips with")
    water = gpd.read_file(WATER, bbox=rings[0].bounds)
    return shapely.make_valid(shapely.difference(rings[0], unary_union(list(water.geometry))))


def geometry():
    """Kontur hexes cut to Gibraltar's land -> data/geo/gi/gi_hexes.gpkg, all on the one unit.

    Each piece keeps the share of its hex's people that the piece is of the hex's LAND, Spanish
    land included; the sea gets none (docstring, THE HEXES ARE CUT TO GIBRALTAR'S LAND)."""
    import gzip
    import shutil

    import geopandas as gpd
    import numpy as np
    import shapely
    from shapely.ops import unary_union

    from geo_checks import read_layer

    gz, gpkg = micro._kontur_paths("gi")
    if not (os.path.exists(gpkg) and os.path.getsize(gpkg) > 0):
        if not os.path.exists(gz):
            raise SystemExit(f"missing {gz}; run with --fetch")
        with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(gpkg + ".part", gpkg)
    hexes = read_layer(gpkg, "Kontur GI")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"gi: no population column in {list(hexes.columns)}")
    hexes = hexes.rename(columns={popcol: "pop"})
    hexes = hexes[hexes["pop"] > 0].copy().to_crs("EPSG:4326").reset_index(drop=True)
    hexes["unit"] = "GI"
    hexes["cellcode"] = "GI:" + hexes.index.astype(str)

    w, s, e, n = hexes.total_bounds
    if not (-5.40 < w < e < -5.30 and 36.08 < s < n < 36.18):
        raise SystemExit(f"gi: Kontur bbox {w:.3f},{s:.3f},{e:.3f},{n:.3f} is not Gibraltar")

    def km2(geoms):
        return gpd.GeoSeries(geoms, crs="EPSG:4326").to_crs(EQUAL_AREA).area.to_numpy() / 1e6

    land = land_outline()
    land_km2 = float(km2([land])[0])
    if not LAND_KM2[0] <= land_km2 <= LAND_KM2[1]:
        raise SystemExit(f"gi: Gibraltar's land measures {land_km2:.2f} km2, outside {LAND_KM2}")
    ray = shapely.intersection(shapely.LineString([(FRONTIER_LON, 36.13), (FRONTIER_LON, 36.18)]),
                               land)
    north = ray.bounds[3] if not ray.is_empty else float("nan")
    if not FRONTIER_LAT[0] <= north <= FRONTIER_LAT[1]:
        raise SystemExit(f"gi: the land's north edge at {FRONTIER_LON} is {north:.5f}, outside "
                         f"{FRONTIER_LAT}; the frontier in the outline has moved")

    # Each hex's land, Gibraltar's and Spain's: the hex less the sea, over the hexes' own bounds
    # (they reach further north than the outline does).
    sea = gpd.read_file(WATER, bbox=tuple(hexes.total_bounds))
    hexes["land_km2"] = km2(shapely.difference(hexes.geometry.values,
                                               unary_union(list(sea.geometry))))
    cut = gpd.overlay(hexes[["cellcode", "unit", "pop", "land_km2", "geometry"]],
                      gpd.GeoDataFrame(geometry=[land], crs="EPSG:4326"),
                      how="intersection", keep_geom_type=True)
    cut["share"] = np.minimum(1.0, np.divide(km2(cut.geometry.values), cut["land_km2"],
                                             out=np.zeros(len(cut)),
                                             where=cut["land_km2"].to_numpy() > 0))
    cut["kontur"] = cut["pop"]
    cut["pop"] = cut["kontur"] * cut["share"]
    cut = cut[cut["pop"] > 0].reset_index(drop=True)
    if not cut.geometry.within(land.buffer(1e-7)).all():
        raise SystemExit("gi: a cut piece reaches outside Gibraltar's land")

    k, kept = float(hexes["pop"].sum()), float(cut["pop"].sum())
    print(f"\n  Gibraltar's land (OSM relation {OSM_RELATION} less the OSM water polygons): "
          f"{land_km2:.2f} km2, frontier at {north:.5f} N on {FRONTIER_LON}")
    part = cut[cut["share"] < 0.999]
    for r in part.itertuples():
        print(f"    {r.cellcode:<6} {r.kontur:>6,.0f} people, {100 * r.share:5.1f}% of its land in "
              f"Gibraltar -> {r.pop:,.0f}")
    gone = sorted(set(hexes["cellcode"]) - set(cut["cellcode"]))
    print(f"  {len(cut)} of {len(hexes)} hexes have land in Gibraltar ({len(part)} of them only "
          f"partly; with none: {', '.join(gone) or '-'}). {kept:,.0f} of Kontur's {k:,.0f} "
          f"people kept; the rest are on the Spanish land of those hexes")

    ratio = kept / CENSUS_TOTAL
    # Kontur 2023-11 against a count of 14 November 2022: a year apart, so a narrow band.
    if not 0.8 <= ratio <= 1.2:
        raise SystemExit(f"gi: Kontur {kept:,.0f} in Gibraltar against the census's "
                         f"{CENSUS_TOTAL:,}, ratio {ratio:.2f}, outside 0.8-1.2")
    geo = os.path.join(ROOT, "data", "geo", "gi")
    os.makedirs(geo, exist_ok=True)
    lp = os.path.join(geo, "gi_land.gpkg")
    gpd.GeoDataFrame({"source": [f"OSM relation {OSM_RELATION} less OSM water polygons"]},
                     geometry=[land], crs="EPSG:4326").to_file(lp + ".part.gpkg", layer="land",
                                                                driver="GPKG")
    os.replace(lp + ".part.gpkg", lp)
    out = os.path.join(geo, "gi_hexes.gpkg")
    cut[["cellcode", "unit", "pop", "geometry"]].to_file(out + ".part.gpkg", layer="hexes",
                                                         driver="GPKG")
    os.replace(out + ".part.gpkg", out)
    print(f"  Kontur {kept:,.0f} vs census {CENSUS_TOTAL:,} ({micro.KONTUR_VINTAGE}, ratio "
          f"{ratio:.2f}); largest piece {cut['pop'].max():,.0f} people; wrote {out} and {lp}")


def check_dots():
    """Every dot and ring of both editions on Gibraltar's land, or stop. Run after scattering."""
    import geopandas as gpd

    # scatter.py writes coordinates to 4 decimals, up to 0.00005 degrees off in each axis, which
    # at 36 N is at most 7.1 m. A dot sampled inside the land can be written that far outside it.
    tol_m = 7.2
    lp = os.path.join(ROOT, "data", "geo", "gi", "gi_land.gpkg")
    if not os.path.exists(lp):
        raise SystemExit(f"missing {lp}; run: python sources/gi.py")
    land = gpd.read_file(lp).to_crs(EQUAL_AREA).geometry.iloc[0]
    bad = 0
    for fn in ("dots_gi.geojson", "dots_gi_10k.geojson", "rings_gi.geojson", "rings_gi_10k.geojson"):
        p = os.path.join(ROOT, "data", "processed", fn)
        if not os.path.exists(p):
            if fn.startswith("dots"):
                raise SystemExit(f"missing {p}; scatter both editions first")
            print(f"  --  {fn}: none written")
            continue
        d = gpd.read_file(p)
        dist = d.geometry.to_crs(EQUAL_AREA).distance(land) if len(d) else d.geometry
        far = int((dist > tol_m).sum()) if len(d) else 0
        bad += far
        print(f"  {'OK ' if not far else 'BAD'} {fn}: {len(d)} marks, {far} more than {tol_m:g} m "
              f"off Gibraltar's land" + (f" (farthest {dist.max():.2f} m)" if len(d) else ""))
    if bad:
        raise SystemExit(f"gi: {bad} marks outside Gibraltar's land")


def main():
    if "--check-dots" in sys.argv:
        check_dots()
        return
    import fitz

    if "--fetch" in sys.argv:
        fetch()
    if not os.path.exists(PDF):
        raise SystemExit(f"{PDF} missing; run: python sources/gi.py --fetch")
    doc = fitz.open(PDF)
    check(doc)
    report()
    rows = emit()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".part", "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLUMNS)
        wr.writeheader()
        wr.writerows(rows)
    os.replace(OUT + ".part", OUT)
    print(f"\nwrote {OUT} ({len(rows)} rows, {sum(r['count'] for r in rows):,} people)")
    geometry()


if __name__ == "__main__":
    main()
