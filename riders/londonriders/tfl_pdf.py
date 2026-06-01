"""
Parse official TfL PDF timetables for Elizabeth + 6 Overground lines.

Different format than railwaydata.co.uk's PDFs:
  - No "Table" markers; sections start with a direction header like
    "Elizabeth line – Westbound" or "Liberty line eastbound".
  - No `Line CODE CODE ...` column header; columns are character-position
    anchored. A station row has time tokens at fixed character offsets, and
    blank where a train doesn't stop.
  - No `d`/`a` flag per row; the time IS the dwell/departure.
  - No `---` for skip; just blank space.
  - Day type appears as a section like "Mondays to Fridays" /
    "Mondays to Saturdays" / "Saturdays" / "Sundays".
  - Some compact PDFs (e.g. Liberty) use "then at the X past Y until Z"
    headway-compression notation that we don't expand. For lines using
    that format, we extract only the explicit columns.

Output: list of trip dicts in build.py's expected shape.
"""
import json
import os
import re
import subprocess
from collections import defaultdict


PDF_DIR = "data/timetables"
TXT_CACHE_DIR = "data/timetables/_txt"


# Filename pattern -> (NUMBAT line code, TfL API line id).
TFL_PDF_LINES = {
    "elizabeth-line": ("EZL", "elizabeth"),
    "lo-liberty-line": ("URL", "liberty"),
    "lo-lioness-line": ("WEL", "lioness"),
    "lo-mildmay-line": ("NLL", "mildmay"),
    "lo-suffragette-line": ("GOB", "suffragette"),
    "lo-weaver-line": ("WAG", "weaver"),
    "lo-windrush-line": ("ELL", "windrush"),
}


TFL_STATION_SUFFIXES = (
    " underground station", " dlr station", " rail station",
    " overground station", " elizabeth line station",
    "-underground", " underground",
    " station", " (london)",
)
# Operator suffixes used in NUMBAT (and sometimes API names) to disambiguate
# multiple physical sites — strip them from the station-name portion before
# matching. e.g. 'New Cross ELL' (Overground) -> 'new cross'.
NUMBAT_OPERATOR_SUFFIXES = (
    " lu", " nr", " lo", " hex", " tfl", " dlr", " ell", " ezl"
)


def normalize_name(s):
    if not s:
        return ""
    n = s.lower().strip()
    # Strip apostrophes, periods, and the Unicode replacement char that shows
    # up when pdftotext mishandles certain en-dashes / typography.
    n = (n.replace(".", "")
          .replace("'", "")
          .replace("’", "")
          .replace("–", "-")
          .replace("—", "-")
          .replace("�", ""))
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


# Map from normalized PDF station name -> normalized API common name.
TFL_PDF_NAME_OVERRIDES = {
    # Elizabeth: Liverpool Street and Paddington each have multiple platform
    # naptans in the API but the PDF tags trains by platform. Collapse all
    # to the simple form so we hit the 910GLIVSTLL / 910GPADTLL parent.
    "liverpool st plts a and b": "liverpool street",
    "liverpool st plts 15-17": "liverpool street",
    "liverpool st plts 1517": "liverpool street",
    "paddington plts 11 and 12": "paddington",
    "paddington plts a and b": "paddington",
    "paddington tfl": "paddington",
    # Burnham/Langley have "(Berks)" disambiguator in API but not PDF.
    "burnham": "burnham (berks)",
    "langley": "langley (berks)",
    # Weaver's API uses 'London Liverpool Street Rail Station'; PDF says bare.
    "liverpool street": "london liverpool street",
}


def extract_text(pdf_path):
    os.makedirs(TXT_CACHE_DIR, exist_ok=True)
    base = os.path.basename(pdf_path)[:-4] + ".txt"
    txt_path = os.path.join(TXT_CACHE_DIR, base)
    if os.path.exists(txt_path) and os.path.getmtime(txt_path) >= os.path.getmtime(pdf_path):
        with open(txt_path, encoding="utf-8", errors="replace") as f:
            return f.read()
    subprocess.run(
        ["pdftotext", "-layout", "-enc", "UTF-8", pdf_path, txt_path],
        check=True,
    )
    with open(txt_path, encoding="utf-8", errors="replace") as f:
        return f.read()


def build_name_to_naptan(tfl_id):
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


def build_naptan_coords(tfl_id):
    """naptan_id -> (lat, lon) for distance sanity checks."""
    path = f"data/api_cache/stops_{tfl_id}.json"
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        sps = json.load(f)
    out = {}
    for s in sps:
        nid = s.get("naptanId")
        lat, lon = s.get("lat"), s.get("lon")
        if nid and lat is not None and lon is not None:
            out[nid] = (float(lat), float(lon))
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    import math
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# Maximum plausible average speed between consecutive stops in km/h. Elizabeth
# Line trains top out at ~145 km/h, so even the fastest legitimate inter-stop
# average (Reading <-> Heathrow direct, ~115 km/h) sits well under 150. Above
# this, we're seeing column-merge artifacts like "Reading 09:45 -> Acton Main
# Line 10:00" — 49 km in 15 min, 196 km/h, faster than the train can run.
MAX_AVG_SPEED_KMH = 150.0

# Hard floor on average inter-station travel time. A train physically can't
# move from one station to its neighbour in less than ~30 seconds (acceleration
# alone). When a "trip" claims to skip N stations on its route in less than
# N * MIN_SEC_PER_STATION seconds, the column merged two trains' times and
# we split there. See split_by_route_subsequence.
MIN_SEC_PER_STATION = 30
# Soft ceiling on average inter-station travel time. A typical London rail
# inter-station hop is 2-3 min including dwell; 8 min is a generous upper
# bound that accommodates skip-stop / express patterns. Also catches the
# inverse column-merge case where two trains' times got grouped: e.g. on
# Weaver, "Clapton 09:05 -> Walthamstow Central 09:37" implies 32 min for a
# 2-station hop, which is the next train's Walthamstow time getting glued
# onto this train's Clapton time.
MAX_SEC_PER_STATION = 8 * 60
# Slack added to the max-time check for terminal dwell or schedule padding.
DWELL_SLACK_S = 5 * 60


def load_routes(tfl_id):
    """Return list of valid naptan sequences from the TfL Unified API.

    Each entry is one orderedLineRoute's full naptan list in physical order
    (e.g. Mildmay's 'Richmond ↔ Stratford' = 23 naptans Richmond..Stratford).
    A trip parsed from the PDF is valid only if its stop sequence is a
    subsequence of one of these routes — that's how we reject column-merge
    chimeras like Richmond-branch + Clapham-branch in one "trip".
    """
    path = f"data/api_cache/seq_{tfl_id}.json"
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            seq = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    routes = []
    for r in seq.get("orderedLineRoutes", []) or []:
        nids = r.get("naptanIds") or []
        if len(nids) >= 2:
            routes.append(nids)
    return routes


def _longest_route_run(resolved, start, route):
    """Walk `route` left-to-right matching resolved[start:]. Track the route
    position of each match; bail out at the first consecutive matched pair
    whose time delta is below the per-station floor (route_dist_in_stations
    * MIN_SEC_PER_STATION). Returns the number of resolved entries accepted.
    """
    j = start
    n = len(resolved)
    last_route_pos = None
    last_secs = None
    ri = 0
    while ri < len(route) and j < n:
        nid_j, secs_j = resolved[j]
        if route[ri] == nid_j:
            if last_route_pos is not None:
                route_dist = ri - last_route_pos
                delta = secs_j - last_secs
                if delta < route_dist * MIN_SEC_PER_STATION:
                    break
                if delta > route_dist * MAX_SEC_PER_STATION + DWELL_SLACK_S:
                    break
            last_route_pos = ri
            last_secs = secs_j
            j += 1
        ri += 1
    return j - start


def split_by_route_subsequence(resolved, routes):
    """Greedy split of a stop sequence by route topology.

    For each starting position i, find the longest k such that resolved[i:i+k]
    is a subsequence of at least one route AND every consecutive matched pair
    respects MIN_SEC_PER_STATION on that route. Emit that sub-segment, advance
    to i+k, repeat. Stops that don't fit any route are dropped.

    This catches PDF column-merge artifacts where times from two physically
    distinct branches got grouped into one column: each branch's stations
    show up as a valid sub-route, the inter-branch jump fails subsequence
    matching, and we split there.
    """
    if not resolved or not routes:
        return [resolved] if resolved else []
    out = []
    i = 0
    n = len(resolved)
    while i < n:
        best = 1
        for route in routes:
            k = _longest_route_run(resolved, i, route)
            if k > best:
                best = k
        if best >= 2:
            out.append(resolved[i:i + best])
        i += max(best, 1)
    return out


# Direction markers vary slightly per line.
DIR_RE = re.compile(
    r"(Elizabeth\s+line\s+[–-]\s+(?:Westbound|Eastbound)|"
    r"(?:Liberty|Lioness|Mildmay|Suffragette|Weaver|Windrush)\s+line\s+(?:eastbound|westbound|northbound|southbound|inbound|outbound))",
    re.IGNORECASE,
)

# Day-type markers.
DAYTYPE_WEEKDAY_RE = re.compile(
    r"(Mondays?\s+to\s+Fridays?|Mondays?\s+to\s+Saturdays?|Mondays?\s+to\s+Thursdays?)",
    re.IGNORECASE,
)
DAYTYPE_END_RE = re.compile(
    r"(Saturdays?|Sundays?|Mondays?\s+to\s+Fridays?|Mondays?\s+to\s+Saturdays?|"
    r"Mondays?\s+to\s+Thursdays?|Valid\s+from)",
    re.IGNORECASE,
)


# Match a time token (HH:MM or HHMM) and validate it's a real time.
TIME_RE = re.compile(r"\b(\d{2})(\d{2})\b")


def is_valid_time(h, m):
    return 0 <= h < 30 and 0 <= m < 60   # allow up to 29:xx for past-midnight


def find_data_rows(section_text):
    """A 'data row' is a line starting with a station name (capital letter)
    and having one or more 4-digit time tokens. Returns list of (line_text)."""
    out = []
    for line in section_text.splitlines():
        # Skip lines that look like headers, captions, or formatting
        if not line.strip():
            continue
        if not re.match(r"^\s*[A-Z]", line):
            continue
        # Must contain at least one valid HHMM time token
        matched = False
        for m in TIME_RE.finditer(line):
            h, mm = int(m.group(1)), int(m.group(2))
            if is_valid_time(h, mm):
                matched = True
                break
        if matched:
            out.append(line)
    return out


def parse_table_rows(rows):
    """Walk rows, extract (station_name, [(col_pos, time_token), ...]) per row.

    Station name = everything before the first time token, with trailing
    operator/service flags (single capital letters like 'LN SN') trimmed.
    """
    parsed = []
    for line in rows:
        # Find first valid time token
        first_time_pos = None
        first_time_match = None
        for m in TIME_RE.finditer(line):
            h, mm = int(m.group(1)), int(m.group(2))
            if not is_valid_time(h, mm):
                continue
            first_time_pos = m.start()
            first_time_match = m
            break
        if first_time_pos is None:
            continue
        # Station name = text before first time, with trailing service flags
        # trimmed. Flags are 2-3 letter uppercase tokens (LN, SN, SX, BHX,...).
        # Single-letter strip is unsafe — it would chew through "Plts A & B"
        # by eating the trailing "B". Only strip 2-3 letter flags, and never
        # if the preceding char is '&' (in case "& B" or "& C" appears).
        name_part = line[:first_time_pos].rstrip()
        name_part = re.sub(r"(?<![&])(?:\s+[A-Z]{2,3})+\s*$", "", name_part)
        name = name_part.strip()
        if not name:
            continue

        # Detect side-by-side direction layouts (Liberty puts eastbound and
        # westbound on the same source line). After the first time, look for
        # another Capitalised-Word — that's the next station starting another
        # column-block. Stop collecting times there, otherwise we fold the
        # next direction's times into the wrong station.
        truncate_pos = len(line)
        for nm in re.finditer(r"\b[A-Z][a-z]{2,}\b", line[first_time_match.end():]):
            truncate_pos = first_time_match.end() + nm.start()
            break

        # Collect all valid time tokens between first_time_pos and truncate_pos.
        times_pos = []
        for m in TIME_RE.finditer(line):
            h, mm = int(m.group(1)), int(m.group(2))
            if not is_valid_time(h, mm):
                continue
            if m.start() < first_time_pos or m.start() >= truncate_pos:
                continue
            times_pos.append((m.start(), f"{h:02d}{mm:02d}"))
        if not times_pos:
            continue
        parsed.append((name, times_pos))
    return parsed


def hms_to_secs(t):
    """'0611' -> 22260 (seconds since midnight)."""
    return int(t[:2]) * 3600 + int(t[2:]) * 60


def secs_to_hms(s):
    s = s % 86400
    return f"{s // 3600:02d}{(s % 3600) // 60:02d}"


# Compression marker phrases that appear in headway-compressed PDFs.
HEADWAY_PHRASES_RE = re.compile(
    r"\b(?:then\s+at|each\s+hour|same\s+time|same\s+minutes\s+past|past\s+each)\b",
    re.IGNORECASE,
)
# In Lioness / Mildmay / etc. the phrase "then at the same minutes past each
# hour until" is rendered ONE WORD PER ROW (right-aligned at the end of each
# station's data line). Detect by any rows ending with a compression keyword
# right after a time token.
COMPRESSION_LINE_END_RE = re.compile(
    r"\d{4}\s+(?:then|at|the|same|past|each|hour|until|time|minutes)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def section_has_compression(section_text):
    if HEADWAY_PHRASES_RE.search(section_text):
        return True
    # If at least 2 station rows end with a standalone compression keyword,
    # we're in the multi-row-fragmented layout.
    matches = COMPRESSION_LINE_END_RE.findall(section_text)
    return len(matches) >= 2


def expand_headway(times_secs):
    """Sorted times list -> sorted list with gap-fills inserted.

    Detects ONE large gap (>2x median delta), fills it at the median-delta
    cadence. Doesn't extrapolate beyond the explicit endpoints — only fills
    between them. So Liberty's [0611, 0641, 0711, 0741, 2111, 2141] becomes
    a 32-entry list at 30-min cadence; Lioness's row that ends mid-day with
    only "then" indicator is left unchanged (no late explicit, nothing to
    fill toward).
    """
    if len(times_secs) < 4:
        return list(times_secs)
    s = sorted(times_secs)
    deltas = [s[i + 1] - s[i] for i in range(len(s) - 1)]
    median_delta = sorted(deltas)[len(deltas) // 2]
    if median_delta <= 0 or median_delta > 30 * 60:
        return s
    out = [s[0]]
    for i in range(1, len(s)):
        gap = s[i] - s[i - 1]
        if gap > 2 * median_delta and gap < 14 * 3600:
            t = s[i - 1] + median_delta
            while t < s[i] - median_delta // 2:
                out.append(t)
                t += median_delta
        out.append(s[i])
    return out


def parse_compressed_section(rows, line_code, tfl_id, name_to_naptan, naptan_coords, routes):
    """Index-aligned alternative to the column-position parser.

    Used when the section's text contains headway-compression phrases. We
    extract each row's times in left-to-right order (= time order within the
    row), gap-fill, then align rows by INDEX. Column N across all rows is
    the Nth-departing train of the day.
    """
    # Re-use parse_table_rows for name/time extraction, then sort each row's
    # times by char-position (which mirrors time order in TfL OG PDFs) and
    # gap-fill.
    parsed = parse_table_rows(rows)
    if not parsed:
        return []

    expanded_rows = []
    for name, times_pos in parsed:
        times_pos_sorted = sorted(times_pos, key=lambda x: x[0])
        secs_list = [hms_to_secs(t) for _, t in times_pos_sorted]
        secs_list = expand_headway(secs_list)
        expanded_rows.append((name, secs_list))

    if not expanded_rows:
        return []

    # Index-align: column N = the Nth train of the day. Each row contributes
    # its Nth time at column N. Rows that ran out of times stop contributing.
    max_cols = max(len(times) for _, times in expanded_rows)
    cols = [[] for _ in range(max_cols)]
    for name, times in expanded_rows:
        for i, t in enumerate(times):
            cols[i].append((name, secs_to_hms(t)))

    out = []
    for col in cols:
        out.extend(column_to_trips(col, line_code, tfl_id, name_to_naptan, naptan_coords, routes))
    return out


def cluster_column_positions(parsed_rows, tolerance=3):
    """Return sorted list of canonical column start positions."""
    all_pos = []
    for _, times_pos in parsed_rows:
        all_pos.extend(p for p, _ in times_pos)
    all_pos = sorted(set(all_pos))
    if not all_pos:
        return []
    anchors = [all_pos[0]]
    for p in all_pos[1:]:
        if p - anchors[-1] > tolerance:
            anchors.append(p)
    return anchors


def assign_to_columns(parsed_rows, anchors, tolerance=3):
    """Each column anchor -> ordered list of (station_name, time_token)."""
    n_cols = len(anchors)
    cols = [[] for _ in range(n_cols)]
    for name, times_pos in parsed_rows:
        for pos, t in times_pos:
            best_i = None
            best_d = tolerance + 1
            for i, a in enumerate(anchors):
                d = abs(pos - a)
                if d <= tolerance and d < best_d:
                    best_d = d
                    best_i = i
            if best_i is None:
                continue
            cols[best_i].append((name, t))
    return cols


def column_to_trips(col, line_code, tfl_id, name_to_naptan, naptan_coords, routes):
    """Convert a column's (name, time) sequence to one or more trip dicts.

    A column from the PDF can mix segments from two different trains when the
    same character position is reused across branches (Elizabeth's Shenfield-
    and Abbey-Wood-branch trains share col 92, Mildmay's Richmond-branch and
    Clapham-Junction-branch likewise overlap). We:
      1. Split at consecutive pairs that imply impossible speed or a >2h gap.
      2. Within each speed-valid segment, split further so each emitted trip
         is a subsequence of one official orderedLineRoute (rejects the
         Richmond+Clapham chimera).
    """
    if len(col) < 2:
        return []
    resolved = []
    prev = -1
    for name, t in col:
        norm = normalize_name(name)
        nid = name_to_naptan.get(norm)
        if nid is None and norm in TFL_PDF_NAME_OVERRIDES:
            nid = name_to_naptan.get(TFL_PDF_NAME_OVERRIDES[norm])
        if nid is None:
            continue
        h, m = int(t[:2]), int(t[2:])
        secs = h * 3600 + m * 60
        # Wrap only on plausible midnight crossings — a small backward jump is
        # almost always a column-merge artifact, and route validation below
        # catches it. A real midnight wrap is a near-24h jump back.
        if secs < prev and (prev - secs) > 18 * 3600:
            secs += 86400
        resolved.append((nid, secs))
        prev = secs
    if len(resolved) < 2:
        return []
    # Collapse consecutive same-naptan stops (Elizabeth's PDF tags trains by
    # platform — "Liverpool St Plts A & B" then "Liverpool St Plts 15-17" both
    # normalize to the same naptan; without collapsing the trip dwells in
    # place for several minutes).
    collapsed = [resolved[0]]
    for nid, secs in resolved[1:]:
        if nid == collapsed[-1][0]:
            collapsed[-1] = (nid, secs)
        else:
            collapsed.append((nid, secs))
    resolved = collapsed
    if len(resolved) < 2:
        return []
    # Pass 1: split at impossible speeds, >2h gaps, or backward time jumps
    # (the latter two indicate cross-train column merge that survived the
    # wrap-suppression above).
    speed_segments = []
    cur = [resolved[0]]
    for i in range(1, len(resolved)):
        prev_nid, prev_t = resolved[i - 1]
        cur_nid, cur_t = resolved[i]
        dt = cur_t - prev_t
        bad = dt > 2 * 3600 or dt < 0
        if not bad:
            a = naptan_coords.get(prev_nid)
            b = naptan_coords.get(cur_nid)
            if a and b and dt > 0:
                dist = haversine_km(a[0], a[1], b[0], b[1])
                if (dist / dt) * 3600 > MAX_AVG_SPEED_KMH:
                    bad = True
        if bad:
            if len(cur) >= 2:
                speed_segments.append(cur)
            cur = [resolved[i]]
        else:
            cur.append(resolved[i])
    if len(cur) >= 2:
        speed_segments.append(cur)
    # Pass 2: route-subsequence split. Each emitted trip must be a subsequence
    # of one of the line's official orderedLineRoutes. If routes data isn't
    # available for this line (no seq_<tfl_id>.json), skip and trust pass 1.
    final_segments = []
    if routes:
        for seg in speed_segments:
            final_segments.extend(split_by_route_subsequence(seg, routes))
    else:
        final_segments = speed_segments
    trips = []
    for stops in final_segments:
        if len(stops) < 2:
            continue
        trips.append({
            "line": line_code,
            "tfl_id": tfl_id,
            "origin": stops[0][0],
            "dep": stops[0][1],
            "stops": stops,
        })
    return trips


def split_into_subsections(text):
    """Yield (direction, daytype, section_text) for each weekday subsection.

    A subsection is a (direction, daytype) pair with the lines belonging to it.
    Walks text top-to-bottom tracking the most recent direction and day-type
    markers.
    """
    lines = text.splitlines()
    cur_direction = None
    cur_daytype = None
    buf = []
    for ln in lines:
        if DIR_RE.search(ln):
            if buf and cur_direction and cur_daytype == "weekday":
                yield (cur_direction, cur_daytype, "\n".join(buf))
            cur_direction = DIR_RE.search(ln).group(1)
            buf = []
            continue
        m = DAYTYPE_END_RE.search(ln)
        if m:
            if buf and cur_direction and cur_daytype == "weekday":
                yield (cur_direction, cur_daytype, "\n".join(buf))
            buf = []
            label = m.group(1).lower()
            if "mon" in label:
                cur_daytype = "weekday"
            elif "sat" in label:
                cur_daytype = "sat"
            elif "sun" in label:
                cur_daytype = "sun"
            elif "valid" in label:
                cur_daytype = None
            continue
        buf.append(ln)
    if buf and cur_direction and cur_daytype == "weekday":
        yield (cur_direction, cur_daytype, "\n".join(buf))


def parse_pdf(pdf_path):
    base = os.path.basename(pdf_path)
    # Filename pattern: e.g. 'lo-liberty-line-timetable-december-2025.pdf'
    # or 'elizabeth-line-december-2025.pdf'
    info = None
    for prefix, val in TFL_PDF_LINES.items():
        if base.lower().startswith(prefix):
            info = val
            break
    if not info:
        return []
    line_code, tfl_id = info

    text = extract_text(pdf_path)
    name_to_naptan = build_name_to_naptan(tfl_id)
    naptan_coords = build_naptan_coords(tfl_id)
    routes = load_routes(tfl_id)
    if not name_to_naptan:
        return []

    out = []
    unmatched_names = defaultdict(int)
    for direction, daytype, section in split_into_subsections(text):
        rows = find_data_rows(section)
        if not rows:
            continue
        if section_has_compression(section):
            # Headway-compressed section (Liberty, occasional Overground rows
            # using "then at the same time past each hour until X Y" notation).
            # Expand each row's times then index-align.
            out.extend(parse_compressed_section(
                rows, line_code, tfl_id, name_to_naptan, naptan_coords, routes))
            continue
        parsed = parse_table_rows(rows)
        if not parsed:
            continue
        anchors = cluster_column_positions(parsed)
        if not anchors:
            continue
        cols = assign_to_columns(parsed, anchors)
        for col in cols:
            trips_from_col = column_to_trips(col, line_code, tfl_id, name_to_naptan, naptan_coords, routes)
            if trips_from_col:
                out.extend(trips_from_col)
            else:
                for name, _ in col:
                    norm = normalize_name(name)
                    if norm not in name_to_naptan and norm not in TFL_PDF_NAME_OVERRIDES:
                        unmatched_names[name] += 1
    if unmatched_names:
        items = sorted(unmatched_names.items(), key=lambda kv: -kv[1])[:5]
        print(f"  {base}: unmatched: {items}")
    return out


def parse_all_tfl_pdfs(pdf_dir=PDF_DIR):
    """Walk PDF_DIR for PDFs whose filenames match TFL_PDF_LINES prefixes."""
    if not os.path.isdir(pdf_dir):
        return []
    trips = []
    by_line = defaultdict(int)
    for fname in sorted(os.listdir(pdf_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        if not any(fname.lower().startswith(p) for p in TFL_PDF_LINES):
            continue
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
        ts = parse_pdf(sys.argv[1])
        print(f"\n{len(ts)} trips. First 3 samples:")
        for t in ts[:3]:
            print(f"  line={t['line']} dep={t['dep']//3600:02d}:{(t['dep']%3600)//60:02d} stops={len(t['stops'])}")
    else:
        parse_all_tfl_pdfs()
