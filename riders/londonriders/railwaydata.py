"""
Parse railwaydata.co.uk timetable PDFs into trip dicts.

Each PDF is a single line's full timetable. Pages contain wide tables where
columns are individual trains and rows are stations:

    Table BAK _ Bakerloo Line (...)                                Mondays to Fridays
                         Line BAK BAK BAK BAK BAK BAK BAK BAK ... (~26 trains)
       Elephant & Castle d --- --- --- 05:37 05:46 05:56 06:03 ...
            Lambeth North d --- --- --- 05:39 05:48 05:58 06:05 ...
                  Waterloo a --- --- --- 05:41 05:50 06:00 06:07 ...
                             d --- --- --- 05:41 05:50 06:00 06:07 ...
            ...

We extract text via pdftotext -layout (preserves columnar alignment), split by
day-type markers, walk tables column-by-column, and emit trips with the same
shape build.py expects.

Filename convention: `<LINE-CODE> <date>.pdf` (e.g. "BAK 12jan26.pdf").
"""
import json
import os
import re
import subprocess
from collections import defaultdict

PDF_DIR = "data/timetables"
TXT_CACHE_DIR = "data/timetables/_txt"

# PDF filename code -> (NUMBAT line code, TfL API line id, header_code).
# `header_code` is the token used in the "Line X X X ..." column header row,
# which sometimes differs from the table-marker code (e.g. DLR1 PDF marks
# tables as "Table DLR1 _" but the column header still says "Line DLR DLR DLR").
PDF_LINE_TO_INFO = {
    "BAK":  ("BAK", "bakerloo",         "BAK"),
    "CEN":  ("CEN", "central",          "CEN"),
    "CIR":  ("CIR", "circle",           "CIR"),
    "DIS":  ("DIS", "district",         "DIS"),
    "HAM":  ("HAM", "hammersmith-city", "HAM"),
    "JUB":  ("JUB", "jubilee",          "JUB"),
    "MET":  ("MET", "metropolitan",     "MET"),
    "NOR":  ("NOR", "northern",         "NOR"),
    "NTN":  ("NOR", "northern",         "NTN"),
    "PIC":  ("PIC", "piccadilly",       "PIC"),
    "VIC":  ("VIC", "victoria",         "VIC"),
    "WAC":  ("WAC", "waterloo-city",    "WAC"),
    "DLR":  ("DLR", "dlr",              "DLR"),
    "DLR1": ("DLR", "dlr",              "DLR"),
    "DLR2": ("DLR", "dlr",              "DLR"),
    "DLR3": ("DLR", "dlr",              "DLR"),
    "DLR4": ("DLR", "dlr",              "DLR"),
}


# Reuse the same name-normalizer as build_geometry so PDF station names match
# the TfL StopPoints catalog. Local copy to avoid coupling.
TFL_STATION_SUFFIXES = (
    " underground station", " dlr station", " rail station",
    " overground station", " elizabeth line station",
    "-underground",   # 'Paddington (H&C Line)-Underground' uses a dash
    " underground",   # bare-Underground variants
    " station", " (london)",
)
NUMBAT_OPERATOR_SUFFIXES = (" lu", " nr", " lo", " hex", " tfl", " dlr")


def normalize_name(s):
    if not s:
        return ""
    n = s.lower().strip()
    n = n.replace(".", "").replace("'", "")  # St. Paul's / Queen's Park -> bare
    # Strip uninformative " (DLR)" parens that PDF DLR uses on shared stations.
    n = re.sub(r"\s*\(dlr\)\s*", "", n)
    for suffix in TFL_STATION_SUFFIXES:
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    for suffix in NUMBAT_OPERATOR_SUFFIXES:
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    n = n.replace(" & ", " and ")
    n = n.replace("&", "and")
    n = re.sub(r"\s+", " ", n).strip()
    return n


def extract_text(pdf_path):
    """Run pdftotext -layout, cache result alongside the PDF."""
    os.makedirs(TXT_CACHE_DIR, exist_ok=True)
    base = os.path.basename(pdf_path)[:-4] + ".txt"
    txt_path = os.path.join(TXT_CACHE_DIR, base)
    if os.path.exists(txt_path) and os.path.getmtime(txt_path) >= os.path.getmtime(pdf_path):
        with open(txt_path, encoding="utf-8") as f:
            return f.read()
    subprocess.run(
        ["pdftotext", "-layout", "-enc", "UTF-8", pdf_path, txt_path],
        check=True,
    )
    with open(txt_path, encoding="utf-8", errors="replace") as f:
        return f.read()


# Day-type markers as seen in railwaydata PDFs.
DAYTYPE_RE = re.compile(
    r"(Mondays?\s+to\s+Fridays?|"
    r"Saturdays?\s+\(also\s+Good\s+Friday\)|"
    r"Saturdays\s+and\s+Public\s+Holidays|"
    r"Saturdays?|"
    r"Sundays?\s+and\s+Public\s+Holidays|"
    r"Sundays?)",
    re.IGNORECASE,
)


def split_by_daytype(text):
    """Yield (daytype, section_text) pairs walking top-to-bottom.

    daytype is one of: "weekday", "sat", "sun".
    """
    parts = DAYTYPE_RE.split(text)
    # parts alternates [pre-text, marker1, content1, marker2, content2, ...]
    # We discard pre-text. For each (marker, content), classify the marker.
    for i in range(1, len(parts), 2):
        marker = parts[i].lower()
        content = parts[i + 1] if i + 1 < len(parts) else ""
        if "mon" in marker:
            dt = "weekday"
        elif "sat" in marker:
            dt = "sat"
        elif "sun" in marker:
            dt = "sun"
        else:
            continue
        yield dt, content


def split_tables(section_text, line_code):
    """Yield each `Table <CODE> _ ...` block in a day-type section."""
    pattern = re.compile(rf"Table\s+{re.escape(line_code)}\s+_", re.I)
    matches = list(pattern.finditer(section_text))
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(section_text)
        yield section_text[start:end]


# A data row looks like:
#   "         Charing Cross d --- 05:43 05:53 06:02 ..."
# or for continuation rows after an 'a':
#   "                       d --- 05:41 05:50 06:00 ..."
# Capture: name (may be empty) + flag (a|d) + a tail of HH:MM/--- tokens.
ROW_RE = re.compile(
    r"^(?P<name>.*?)\s+(?P<flag>[ad])\s+(?P<tail>(?:(?:\d{2}:\d{2}|---)\s*)+)\s*$"
)


def parse_table_block(block, header_code):
    """Return list of trips (each a list of (station_name, secs_since_midnight))
    extracted from one Table block. `header_code` is the code used in the
    Line column header — usually the PDF filename code (e.g. NTN, DLR)."""
    lines = block.splitlines()

    # Find the column header: "Line <CODE> <CODE> ..."
    n_cols = None
    header_idx = None
    for i, ln in enumerate(lines):
        toks = ln.split()
        if len(toks) >= 5 and toks[0] == "Line" and all(t == header_code for t in toks[1:]):
            n_cols = len(toks) - 1
            header_idx = i
            break
    if n_cols is None:
        return []

    # Walk subsequent rows. Track last-seen station name so 'continuation'
    # rows (the second `d` of an a/d pair on a terminus) inherit it.
    columns = [[] for _ in range(n_cols)]   # each = list of (name, time_str, flag)
    last_name = None

    for ln in lines[header_idx + 1:]:
        m = ROW_RE.match(ln)
        if not m:
            continue
        name_part = m.group("name").strip()
        flag = m.group("flag")
        toks = m.group("tail").split()
        if len(toks) != n_cols:
            continue
        if not all(t == "---" or re.match(r"^\d{2}:\d{2}$", t) for t in toks):
            continue
        if name_part:
            last_name = name_part
        if not last_name:
            continue
        for ci, t in enumerate(toks):
            if t == "---":
                continue
            columns[ci].append((last_name, t, flag))

    # Convert columns to trips. If a station appears twice in a column (an
    # 'a' followed by a 'd'), keep the last entry — that's the dep we want.
    trips = []
    for col in columns:
        if len(col) < 2:
            continue
        stops = []  # list of (name, time_str)
        for name, t, flag in col:
            if stops and stops[-1][0] == name:
                stops[-1] = (name, t)
            else:
                stops.append((name, t))
        if len(stops) < 2:
            continue
        # Convert HH:MM to secs since midnight, handle past-midnight wrap.
        sec_stops = []
        prev_secs = -1
        for name, t in stops:
            h, mm = t.split(":")
            secs = int(h) * 3600 + int(mm) * 60
            if secs < prev_secs:
                secs += 86400
            sec_stops.append((name, secs))
            prev_secs = secs
        trips.append(sec_stops)
    return trips


def build_name_to_naptan(tfl_id):
    """Load stops_<tfl_id>.json and build {normalized_name: naptan_id}."""
    path = f"data/api_cache/stops_{tfl_id}.json"
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        sps = json.load(f)
    out = {}
    for s in sps:
        nid = s.get("naptanId")
        name = s.get("commonName", "")
        if nid and name:
            out[normalize_name(name)] = nid
    return out


# Edge cases where the PDF station name doesn't normalize to the same string
# as the TfL API's commonName. Map normalized PDF -> normalized API.
# PDFs use shorter paren tags than the API's commonName.
PDF_NAME_OVERRIDES = {
    "hammersmith (handc)": "hammersmith (handc line)",
    "hammersmith (distandpicc)": "hammersmith (distandpicc line)",
    "paddington (handc)": "paddington (handc line)",
    "edgware road (circle)": "edgware road (circle line)",
    "edgware road (bak)": "edgware road (bakerloo)",
    # DLR station name quirks
    "custom house for excel": "custom house (for excel)",
    "cutty sark": "cutty sark (for maritime greenwich)",
}


def parse_pdf(pdf_path):
    """Returns list of trip dicts in build.py's expected shape."""
    base = os.path.basename(pdf_path)
    pdf_code = base.split()[0].upper()
    info = PDF_LINE_TO_INFO.get(pdf_code)
    if not info:
        return []
    line_code, tfl_id, header_code = info

    text = extract_text(pdf_path)
    name_to_naptan = build_name_to_naptan(tfl_id)
    if not name_to_naptan:
        return []

    # Walk Table markers directly. Day-type indicators (e.g. "Mondays to
    # Fridays") often share a line with the Table caption, so splitting the
    # text by them severs each table from its data rows. Instead, find each
    # Table block and attribute the most recent day-type marker seen anywhere
    # in the document up to that block's end.
    # The first Table on every line PDF lacks an inline day-type — the page
    # header above it carries the "Mondays to Fridays" marker. Searching only
    # within the block (as a naive split would) drops that first table, which
    # holds the entire early-morning 05:00–06:30 service for one direction.
    # The Table caption in the PDF uses the PDF code (e.g. "NTN" for Northern,
    # "DLR" within DLR1.pdf), which may differ from the internal line code.
    table_marker = re.compile(rf"Table\s+{re.escape(pdf_code)}\s+_", re.I)
    matches = list(table_marker.finditer(text))

    daytype_positions = []  # [(offset, kind)] with kind in "weekday"/"sat"/"sun"
    for dm in DAYTYPE_RE.finditer(text):
        marker = dm.group(0).lower()
        if "mon" in marker:
            kind = "weekday"
        elif "sat" in marker:
            kind = "sat"
        elif "sun" in marker:
            kind = "sun"
        else:
            continue
        daytype_positions.append((dm.start(), kind))

    out = []
    unmatched = defaultdict(int)
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        # Latest day-type marker at offset < end is the table's effective
        # day-type — captures both inline captions and the page-header
        # marker that precedes the first Table on a page. Stray "Saturday" /
        # "Sunday" tokens in the legend at the top of the PDF are correctly
        # overridden by the "Mondays to Fridays" page-header marker that
        # follows them.
        applicable = None
        for off, kind in daytype_positions:
            if off < end:
                applicable = kind
            else:
                break
        if applicable != "weekday":
            continue
        for stop_seq in parse_table_block(block, header_code):
            resolved = []
            for name, secs in stop_seq:
                norm = normalize_name(name)
                nid = name_to_naptan.get(norm)
                if nid is None and norm in PDF_NAME_OVERRIDES:
                    nid = name_to_naptan.get(PDF_NAME_OVERRIDES[norm])
                if nid is None:
                    unmatched[name] += 1
                    continue
                resolved.append((nid, secs))
            if len(resolved) < 2:
                continue
            out.append({
                "line": line_code,
                "tfl_id": tfl_id,
                "origin": resolved[0][0],
                "dep": resolved[0][1],
                "stops": resolved,
            })
    if unmatched:
        items = sorted(unmatched.items(), key=lambda kv: -kv[1])[:10]
        print(f"  {base}: {len(unmatched)} unmatched station names, top: {items[:5]}")
    return out


def parse_all_pdfs(pdf_dir=PDF_DIR):
    """Walk PDF_DIR, parse every <CODE> <date>.pdf, return concatenated trips."""
    print(f"Parsing PDFs in {pdf_dir}/...")
    if not os.path.isdir(pdf_dir):
        return []
    pdfs = [f for f in sorted(os.listdir(pdf_dir)) if f.lower().endswith(".pdf")]
    trips = []
    by_line = defaultdict(int)
    for fname in pdfs:
        path = os.path.join(pdf_dir, fname)
        ts = parse_pdf(path)
        trips.extend(ts)
        for t in ts:
            by_line[t["line"]] += 1
        print(f"  {fname}: {len(ts)} trips")
    print(f"  total: {len(trips)} trips")
    print(f"  trips per line: {dict(sorted(by_line.items()))}")
    return trips


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        trips = parse_pdf(sys.argv[1])
        print(f"\n{len(trips)} trips. First trip sample:")
        if trips:
            t = trips[0]
            print(f"  line={t['line']} dep={t['dep']}s ({t['dep']//3600:02d}:{(t['dep']%3600)//60:02d})")
            for n, s in t["stops"][:8]:
                print(f"    {n}  {s//3600:02d}:{(s%3600)//60:02d}")
            if len(t["stops"]) > 8:
                print(f"    ... +{len(t['stops']) - 8} more")
    else:
        parse_all_pdfs()
