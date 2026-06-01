"""
Parse TfL Journey Planner Timetables (TransXChange v2.1) into trip dicts
matching the shape build.py expects.

For each rail file in the TXC zip:
  - JourneyPatternSection -> [(from_stop, to_stop, run_time_secs), ...]
  - JourneyPattern        -> ordered list of section refs (-> flattened stops)
  - VehicleJourney        -> {dep_time, journey_pattern_ref, operating_profile}

We emit one trip per VehicleJourney scheduled on a typical Tue/Wed/Thu:
  {
    "line":   "BAK" / "CEN" / ...,
    "tfl_id": "bakerloo" / ... (cosmetic; for compatibility with API path),
    "origin": naptan_id,
    "dep":    seconds since midnight,
    "stops":  [(naptan_id, secs_since_midnight), ...]   ordered, includes origin
  }

TXC trip shape mirrors build.py's existing flatten_trips() output exactly so
the rest of the pipeline (trip_index, assign_riders, etc.) keeps working.
"""
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict


NS = "{http://www.transxchange.org.uk/}"


# TXC ServiceCode prefix -> NUMBAT line code.
# Demo zip uses these; live feed should use the same convention.
SERVICE_CODE_TO_LINE = {
    "01BAK": "BAK",
    "01CEN": "CEN",
    "01CIR": "CIR",
    "01DIS": "DIS",
    "01HAM": "HAM",
    "01JUB": "JUB",
    "01MET": "MET",
    "01NTN": "NOR",
    "01PIC": "PIC",
    "01VIC": "VIC",
    "01WAC": "WAC",
    "25DLR": "DLR",
}


# OperatingProfile day flags we accept as "typical Tue/Wed/Thu":
ACCEPT_DAYS = {
    "MondayToFriday",
    "MondayToSaturday",  # rare, but means it runs on weekdays too
    "Tuesday",
    "Wednesday",
    "Thursday",
}


# Some 2010 TXC files reference Tube stations using a bus-stop NaPTAN code
# rather than the rail StopPointRef. Map them to the station-parent form
# explicitly. Found by scanning rail files for non-9400 prefixes.
TXC_NAPTAN_OVERRIDES = {
    "490000254009": "940GZZLUWLO",  # Waterloo (Waterloo & City)
    "490000276005": "940GZZLUWOP",  # Woodside Park (Northern)
}


def normalize_naptan(ref):
    """TXC '9400ZZLUEAC1' (platform-level) -> '940GZZLUEAC' (station-parent).

    The TfL Unified API uses the station-parent NaPTAN form; TXC uses
    platform-level (with a trailing platform digit). They share the trunk
    id, just different prefix codes (9400 vs 940G) and a trailing digit.
    A few stops in the demo TXC use bus-stop NaPTANs entirely; those are
    looked up via TXC_NAPTAN_OVERRIDES.
    """
    if not ref or len(ref) < 4:
        return ref
    if ref in TXC_NAPTAN_OVERRIDES:
        return TXC_NAPTAN_OVERRIDES[ref]
    s = ref.rstrip("0123456789")
    if s[:4] == "9400":
        return "940G" + s[4:]
    return s


def parse_iso_duration_seconds(s):
    """PT60S -> 60. PT2M30S -> 150. PT1H -> 3600."""
    if not s:
        return 0
    m = re.match(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$", s)
    if not m:
        return 0
    h, mm, ss = m.groups()
    return int(h or 0) * 3600 + int(mm or 0) * 60 + int(float(ss or 0))


def parse_hms_seconds(s):
    """HH:MM:SS -> seconds since midnight."""
    if not s:
        return 0
    parts = s.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 3600 + int(parts[1]) * 60
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def operating_profile_runs_on_typical_weekday(elem):
    """Walk an OperatingProfile element; return True if it runs Tue/Wed/Thu.

    Looks at RegularDayType/DaysOfWeek for any of the ACCEPT_DAYS flags.
    Ignores BankHolidayOperation (NUMBAT TWT excludes holidays anyway).
    """
    # Find the DaysOfWeek subtree
    days_elem = elem.find(f".//{NS}RegularDayType/{NS}DaysOfWeek")
    if days_elem is None:
        # No constraint = runs every day
        # But also no day flags = runs all days (TXC convention)
        return True
    for child in days_elem:
        tag = child.tag.replace(NS, "")
        if tag in ACCEPT_DAYS:
            return True
        # Allow "DaysOfNonOperation" combos: if the DaysOfWeek is empty of
        # accepted days, fall through to False.
    return False


def line_code_from_service_code(service_code):
    """Look up 'SId_01BAK0' / 'SId_25DLRS0' -> NUMBAT line code, or None.

    TfL TXC ServiceCode format: 'SId_<5-char-line>0'.
    """
    if not service_code:
        return None
    s = service_code
    if s.startswith("SId_"):
        s = s[4:]
    # Try first 5 chars (e.g. "01BAK", "25DLR")
    return SERVICE_CODE_TO_LINE.get(s[:5])


def parse_txc_file(zf, name):
    """Parse one TXC XML file. Returns list of trip dicts (or [] if not rail)."""
    section_links = {}              # section_id -> [(from_naptan, to_naptan, run_secs)]
    journey_patterns = {}           # jp_id -> {section_refs: [...], direction: str}
    vehicle_journeys = []           # list of {dep, jp_ref, op_profile_elem}
    service_code = None
    line_name = None

    # iterparse for memory efficiency — Tube files can be 12MB
    with zf.open(name) as f:
        for ev, elem in ET.iterparse(f, events=("end",)):
            tag = elem.tag.replace(NS, "")

            if tag == "JourneyPatternSection":
                sid = elem.get("id")
                links = []
                for link in elem.findall(f"{NS}JourneyPatternTimingLink"):
                    from_ref = link.find(f"{NS}From/{NS}StopPointRef")
                    to_ref = link.find(f"{NS}To/{NS}StopPointRef")
                    rt = link.find(f"{NS}RunTime")
                    if from_ref is None or to_ref is None or rt is None:
                        continue
                    secs = parse_iso_duration_seconds(rt.text)
                    links.append((
                        normalize_naptan(from_ref.text),
                        normalize_naptan(to_ref.text),
                        secs,
                    ))
                if links:
                    section_links[sid] = links
                elem.clear()

            elif tag == "JourneyPattern":
                jp_id = elem.get("id")
                refs_elem = elem.find(f"{NS}JourneyPatternSectionRefs")
                refs = []
                if refs_elem is not None:
                    # SectionRefs may be under .text or as one ref each child
                    text = refs_elem.text
                    if text and text.strip():
                        refs = text.split()
                    else:
                        refs = [c.text for c in refs_elem if c.text]
                # JourneyPattern.SectionRefs may be inline (one per child) too:
                if not refs:
                    refs = [c.text for c in elem.findall(f"{NS}JourneyPatternSectionRefs")]
                # Or worst-case stored as <JourneyPatternSectionRefs> repeated:
                if not refs:
                    refs = [c.text for c in elem.iter() if c.tag.endswith("JourneyPatternSectionRefs") and c.text]
                if jp_id and refs:
                    journey_patterns[jp_id] = {"section_refs": refs}
                # don't clear yet — we may revisit

            elif tag == "VehicleJourney":
                dep_elem = elem.find(f"{NS}DepartureTime")
                jpref_elem = elem.find(f"{NS}JourneyPatternRef")
                op_elem = elem.find(f"{NS}OperatingProfile")
                if dep_elem is None or jpref_elem is None:
                    elem.clear()
                    continue
                vehicle_journeys.append({
                    "dep_secs": parse_hms_seconds(dep_elem.text),
                    "jp_ref": jpref_elem.text,
                    "runs_weekday": (
                        operating_profile_runs_on_typical_weekday(op_elem)
                        if op_elem is not None else True
                    ),
                })
                elem.clear()

            elif tag == "ServiceCode" and service_code is None:
                service_code = (elem.text or "").strip()

            elif tag == "LineName" and line_name is None:
                line_name = (elem.text or "").strip()

            elif tag in ("Service", "TransXChange"):
                # safe to clear the big root once everything's collected
                pass

    line_code = line_code_from_service_code(service_code)
    if line_code is None:
        return []

    # Flatten each JourneyPattern's section sequence into an ordered stop list
    # with cumulative time offsets from the pattern's first stop.
    pattern_stops = {}  # jp_id -> [(naptan, offset_secs)]
    for jp_id, jp in journey_patterns.items():
        ordered = []
        cum = 0
        first_from = None
        for sref in jp["section_refs"]:
            links = section_links.get(sref)
            if not links:
                continue
            for i, (fr, to, rt) in enumerate(links):
                if not ordered:
                    ordered.append((fr, 0))
                    first_from = fr
                # Each link contributes the run time to reach `to`
                cum += rt
                ordered.append((to, cum))
        if len(ordered) >= 2:
            pattern_stops[jp_id] = ordered

    # Emit a trip per VehicleJourney that runs on typical weekday
    out = []
    for vj in vehicle_journeys:
        if not vj["runs_weekday"]:
            continue
        ps = pattern_stops.get(vj["jp_ref"])
        if not ps:
            continue
        dep = vj["dep_secs"]
        stops = [(naptan, dep + off) for naptan, off in ps]
        out.append({
            "line": line_code,
            "tfl_id": "",  # TXC has its own service ids; not used downstream
            "origin": ps[0][0],
            "dep": dep,
            "stops": stops,
        })
    return out


def parse_txc_zip(zip_path):
    """Walk every TXC file in the zip; return list of trip dicts for rail lines."""
    print(f"Parsing TXC zip {zip_path}...")
    trips = []
    by_line = defaultdict(int)
    skipped_non_rail = 0
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".xml")]
        for i, name in enumerate(names):
            if (i % 50) == 0:
                print(f"  [{i}/{len(names)}] parsed {len(trips)} trips so far")
            ts = parse_txc_file(z, name)
            if not ts:
                skipped_non_rail += 1
                continue
            trips.extend(ts)
            for t in ts:
                by_line[t["line"]] += 1
    print(f"  {len(trips)} trips total ({len(names) - skipped_non_rail} rail files, "
          f"{skipped_non_rail} skipped non-rail)")
    print(f"  trips per line: {dict(sorted(by_line.items()))}")
    return trips


if __name__ == "__main__":
    # Smoke test: parse a single file and print summary
    import sys
    zip_path = sys.argv[1] if len(sys.argv) > 1 else "data/tfl-txc-demo.zip"
    trips = parse_txc_zip(zip_path)
    if trips:
        s = trips[0]
        print(f"\nFirst trip sample (line={s['line']}, origin={s['origin']}, dep={s['dep']}s):")
        for n, t in s["stops"][:8]:
            print(f"  {n}  +{t - s['dep']:>5}s ({t//3600:02d}:{(t%3600)//60:02d}:{t%60:02d})")
        if len(s["stops"]) > 8:
            print(f"  ... +{len(s['stops']) - 8} more stops")
