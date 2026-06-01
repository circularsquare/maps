"""
Parse Network Rail SCHEDULE feed JSON for Elizabeth + 6 Overground lines.

Network Rail publishes per-TOC schedule extracts via openraildata.com — these
contain JsonScheduleV1 records with full stop sequences, treating each train
as a first-class entity with its real calling pattern. No PDF column-merge
ambiguity; through-running services like Reading→Abbey Wood appear as one
schedule with the full stop list.

Files expected:
  data/ezl.json.gz   CIF_EX_TOC_FULL_DAILY  (Elizabeth, ATOC=XR, business=EX)
  data/og.json.gz    CIF_EK_TOC_FULL_DAILY  (Overground,  ATOC=LO, business=EK)

Each is gzipped newline-JSON with five record types:
  JsonTimetableV1     — one header
  TiplocV1            — global location reference (~12k entries)
  JsonScheduleV1      — one train schedule (the meat)
  JsonAssociationV1   — split/join links between schedules (rare, ignored v1)
  EOF                 — terminator

Output: trip dicts in build.py's expected shape:
  {line: NUMBAT_code, tfl_id, origin: naptan, dep: secs, stops: [(naptan, secs)]}

OG line classification: Network Rail still tags everything atoc='LO'; TfL split
the Overground into 6 named lines in 2024. We infer the named line by checking
which line's station set best contains the train's stops.
"""
import datetime
import gzip
import json
import os
import re
from collections import Counter, defaultdict


CIF_EZL = "data/ezl.json.gz"
CIF_OG = "data/og.json.gz"

# OG named lines in TfL Unified API form
OG_LINES = ["liberty", "lioness", "mildmay", "suffragette", "weaver", "windrush"]

# tfl_id -> NUMBAT line code (matches build.py / PER_LINE_TO_CODES)
TFL_TO_NUMBAT = {
    "elizabeth": "EZL",
    "liberty": "URL",
    "lioness": "WEL",
    "mildmay": "NLL",
    "suffragette": "GOB",
    "weaver": "WAG",
    "windrush": "ELL",
}

# STP indicator priority: C (cancellation) wins, then N (STP), then O (overlay),
# then P (permanent). For any (UID, day) pair, the lowest letter wins.
STP_ORDER = {"C": 0, "N": 1, "O": 2, "P": 3}


def parse_hhmm(s):
    """'0552' or '0552H' -> seconds since midnight. 'H' suffix = +30s."""
    if not s:
        return None
    h, m = int(s[:2]), int(s[2:4])
    extra = 30 if len(s) > 4 and s[4].upper() == "H" else 0
    return h * 3600 + m * 60 + extra


def parse_iso_date(s):
    if not s:
        return None
    return datetime.date.fromisoformat(s[:10])


def normalize_tfl_name(s):
    """TfL commonName -> normalized form for matching."""
    if not s:
        return ""
    n = s.lower().strip()
    n = re.sub(r"[\.\']", "", n)
    n = n.replace("&", "and")
    for tail in (" rail station", " station", " (london)",
                 " underground station", " overground station",
                 " elizabeth line station", " dlr station"):
        if n.endswith(tail):
            n = n[: -len(tail)]
    n = re.sub(r"\s+", " ", n).strip()
    return n


def normalize_tiploc_name(s):
    """TIPLOC description -> normalized form. CIF uses heavy abbreviations
    ('LIVRPL ST' = Liverpool Street, 'KENTISH TN WEST' = Kentish Town West)
    plus branch/operator suffixes ('CANONBURY ELL', 'BUSHEY DC', 'RICHMOND NLL')
    that mark which physical platform-set within a station. Strip both."""
    n = s.lower().strip()
    n = re.sub(r"[\.\']", "", n)
    n = n.replace("&", "and")
    # Strip platform/track disambiguators first ("PLATS 0-2", "PLT A")
    n = re.sub(r"\s+plats?\s+\S+.*$", "", n)
    n = re.sub(r"\s+plt\s+\S+.*$", "", n)
    # Strip suffixes (operator/branch markers + generic "station"/"london" tail).
    # Repeat until stable so chains like "BUSHEY DC STATION" collapse fully.
    suffix_tails = (" ell", " nll", " dc", " el", " lo", " lu", " nr",
                    " mtn", " l m", " lm", " hand c", " cs", " tube",
                    " rail", " station", " london", " jn",
                    " high level", " low level")
    changed = True
    while changed:
        changed = False
        for tail in suffix_tails:
            if n.endswith(tail):
                n = n[: -len(tail)].rstrip()
                changed = True
    # Common CIF word-level abbreviations
    n = n.replace("livrpl", "liverpool")
    n = n.replace("shoredtch", "shoreditch")
    n = n.replace("harlngtn", "harlington")
    n = n.replace("harrngay", "harringay")
    n = n.replace("kingslnd", "kingsland")
    n = n.replace("kensngtn", "kensington")
    # st / rd / hgh / tn expansions (whole-word)
    n = re.sub(r"\bst\b", "street", n)
    n = re.sub(r"\brd\b", "road", n)
    n = re.sub(r"\bhgh\b", "high", n)
    n = re.sub(r"\btn\b", "town", n)
    n = re.sub(r"\bctl\b", "central", n)
    n = re.sub(r"\bjn\b", "junction", n)
    n = re.sub(r"\bpk\b", "park", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


# Hand-curated overrides for TIPLOCs whose tps_description doesn't normalize
# to the matching TfL commonName. Most stations are caught by
# normalize_tiploc_name; this dict picks up the rest where:
#   - The CIF name uses a different word entirely (TIPLOC says "PADDINGTON EL",
#     TfL says "Paddington")
#   - The CIF name omits a TfL disambiguator the API requires
#     ("FINCHLEY RD & FL" -> "Finchley Road & Frognal", "CALEDONIAN ROAD" ->
#     "Caledonian Road & Barnsbury", "HIGH AND IS ELL" -> "Highbury & Islington")
TIPLOC_TO_TFL_NAME = {
    # Elizabeth platform-disambiguated TIPLOCs. PADTLL/LIVSTLL are the
    # Crossrail (low-level) platforms; PADTON is the mainline (terminal)
    # platforms that some Reading/Heathrow short-turns terminate at without
    # through-running to the central tunnel. Both map to the same TfL
    # commonName; the geometry NAPTAN_ALIASES re-unifies the 910GPADTON
    # naptan back to the canonical mnlc.
    "TOTCTRD": "tottenham court road",
    "PADTLL":  "paddington",
    "PADTON":  "paddington",
    "LIVSTLL": "liverpool street",
    # Windrush (East London Line) — joined Underground/Overground stations
    "CNDAW":   "canada water",
    "BTHNLGR": "bethnal green",
    "SURREYQ": "surrey quays",
    "DALS":    "dalston junction",
    "HAGGERS": "haggerston",
    "SHADWEL": "shadwell",
    "WCHAPEL": "whitechapel",
    "HOXTON":  "hoxton",
    # Heathrow naming: TfL uses 'Heathrow Terminals 2 & 3' / 'Terminal 4' /
    # 'Terminal 5'. NR uses HTRWAPT for T2&3, and per-terminal HTRWTM4/HTRWTM5
    # ("Terminal Module") for the branches. The older HTRWAP4/AP5 codes are
    # legacy Heathrow Express identifiers, not in the current Elizabeth feed.
    "HTRWAPT": "heathrow terminals 2 and 3",
    "HTRWTM4": "heathrow terminal 4",
    "HTRWTM5": "heathrow terminal 5",
    # Burnham/Langley have "(Berks)" disambig in TfL but not CIF. Note the
    # actual TIPLOC for Burnham is BNHAM (BURNHAM is unused in the feed).
    "LANGLEY": "langley (berks)",
    "BNHAM":   "burnham (berks)",
    # CIF compresses station names that TfL spells out
    "HAYESAH": "hayes and harlington",
    "CLDNNRB": "caledonian road and barnsbury",   # CIF strips "& Barnsbury"
    "FNCHLYR": "finchley road and frognal",        # "FL" = Frognal
    "STJMSST": "st james street",                  # CIF appends "WSTW" suffix
    # Windrush (ELL) / Mildmay (NLL) / Lioness (DC) branch variants whose
    # suffix-stripped form still doesn't match because the base TIPLOC name
    # is heavily abbreviated:
    "HIGHBYE": "highbury and islington",           # "HIGH AND IS ELL"
    "NEWXGEL": "new cross gate",                   # "NEW CROSS GATE STATION ELL"
    # Windrush terminating service at New Cross (the SE-shared station, mnlc
    # 5150 — different from New Cross Gate, mnlc 5345). normalize_tiploc_name
    # strips the " ell" suffix, leaving "new cross", but the TfL commonName
    # is "New Cross ELL Rail Station" → norm "new cross ell". Override
    # preserves the " ell" so name_index lookup matches.
    "NWCRELL": "new cross ell",
    "NWCROSS": "new cross ell",
    # Suffragette (Gospel Oak — Barking) — CIF abbreviates the descriptive
    # words TfL spells out ("Mid" → "Midland", "H" → "High", "Grn Lns" →
    # "Green Lanes", "Q" → "Queens").
    "LEYTNMR": "leyton midland road",
    "LYTNSHR": "leytonstone high road",
    "HRGYGL":  "harringay green lanes",
    "WLTHQRD": "walthamstow queens road",
    # Mildmay (NLL) — CIF strips "Junction" suffix as a noise word, so
    # "WILLESDEN JN. HIGH LEVEL" / "WILLESDEN JN LOW LEVEL" both normalize
    # to bare "willesden" and fail to match the TfL name. WEL (Lioness)
    # uses the same station.
    "WLSDJHL": "willesden junction",
    "WLSDNJL": "willesden junction",
    # NLL — TfL preserves the paren disambiguator but the CIF norm drops it.
    "KENOLYM": "kensington (olympia)",
    # Windrush (ELL) — CIF abbreviates "Peckham" to "Peckhm" and tags
    # Denmark Hill with a " Lon" suffix that the normalizer doesn't strip.
    "PCKHMQD": "queens road peckham",
    "DENMRKH": "denmark hill",
}


def load_tiploc_info(path):
    """tiploc_code -> {crs, name}. Reads only the TiplocV1 records."""
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            t = rec.get("TiplocV1")
            if not t:
                continue
            tc = t.get("tiploc_code")
            if not tc:
                continue
            out[tc] = {
                "crs": t.get("crs_code"),
                "name": (t.get("description") or t.get("tps_description") or "").strip(),
            }
    return out


def load_tfl_name_index():
    """normalized_name -> set of (tfl_id, naptan).
    Per-line, since some stations have different naptans across lines."""
    name_index = defaultdict(set)
    line_stop_sets = {}
    for tfl_id in ["elizabeth"] + OG_LINES:
        path = f"data/api_cache/stops_{tfl_id}.json"
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            sps = json.load(f)
        s_set = set()
        for s in sps:
            nid = s.get("naptanId")
            nm = s.get("commonName", "")
            if not nid or not nm:
                continue
            name_index[normalize_tfl_name(nm)].add((tfl_id, nid))
            s_set.add(nid)
        line_stop_sets[tfl_id] = s_set
    return name_index, line_stop_sets


def build_tiploc_to_naptan(tiploc_info, name_index, target_tfl_id):
    """tiploc_code -> naptan, restricted to stations on target_tfl_id."""
    out = {}
    for tc, info in tiploc_info.items():
        # Hand-curated override first
        norm = TIPLOC_TO_TFL_NAME.get(tc) or normalize_tiploc_name(info.get("name", ""))
        if not norm:
            continue
        for tfl_id, nid in name_index.get(norm, ()):
            if tfl_id == target_tfl_id:
                out[tc] = nid
                break
    return out


def days_runs_bit(days_runs, weekday):
    """schedule_days_runs is a 7-char bitstring MTWTFSS. weekday: 0=Mon .. 6=Sun."""
    if not days_runs or len(days_runs) < 7:
        return False
    return days_runs[weekday] == "1"


def select_schedules_for_date(schedules, date):
    """Group raw schedules by UID, pick the lowest-STP one valid on `date`,
    drop cancellations. Returns a list of resolved schedules."""
    weekday = date.weekday()
    by_uid = defaultdict(list)
    for s in schedules:
        sd = parse_iso_date(s.get("schedule_start_date"))
        ed = parse_iso_date(s.get("schedule_end_date"))
        if sd and date < sd:
            continue
        if ed and date > ed:
            continue
        if not days_runs_bit(s.get("schedule_days_runs", ""), weekday):
            continue
        uid = s.get("CIF_train_uid")
        by_uid[uid].append(s)
    out = []
    for uid, items in by_uid.items():
        items.sort(key=lambda s: STP_ORDER.get(s.get("CIF_stp_indicator"), 9))
        chosen = items[0]
        if chosen.get("CIF_stp_indicator") == "C":
            continue
        out.append(chosen)
    return out


def schedule_to_stops(s, tiploc_to_naptan):
    """Walk schedule_location -> [(naptan, secs)]. Skips pass-only locations
    (no public time) and TIPLOCs with no naptan mapping (junctions etc.)."""
    seg = s.get("schedule_segment") or {}
    out = []
    for loc in seg.get("schedule_location") or []:
        tc = loc.get("tiploc_code")
        nid = tiploc_to_naptan.get(tc)
        if not nid:
            continue
        # Prefer public_departure (when train leaves stop). Fall back to public_arrival.
        secs = parse_hhmm(loc.get("public_departure")) or parse_hhmm(loc.get("public_arrival"))
        if secs is None:
            continue
        # Midnight wrap: train departing 23:50 then arriving 00:05 next day means
        # the second time should be +24h. Detect by big backward jump.
        if out and secs < out[-1][1] - 6 * 3600:
            secs += 24 * 3600
        # Collapse consecutive same-naptan entries (split/join hubs sometimes
        # list the same station twice).
        if out and out[-1][0] == nid:
            out[-1] = (nid, secs)
        else:
            out.append((nid, secs))
    return out


def classify_og_line(schedule, line_to_tiplocs):
    """Pick the OG line whose TIPLOC set best contains this train's stops.
    Returns (tfl_id, overlap_fraction) or (None, 0.0) if no line dominates."""
    train_tiplocs = set()
    for loc in (schedule.get("schedule_segment") or {}).get("schedule_location") or []:
        if loc.get("public_departure") or loc.get("public_arrival"):
            tc = loc.get("tiploc_code")
            if tc:
                train_tiplocs.add(tc)
    if not train_tiplocs:
        return (None, 0.0)
    best = (None, 0.0)
    for tfl_id, line_tlocs in line_to_tiplocs.items():
        if not line_tlocs:
            continue
        # Fraction of train's stops that lie on this line
        overlap = len(train_tiplocs & line_tlocs) / len(train_tiplocs)
        if overlap > best[1]:
            best = (tfl_id, overlap)
    return best


def next_typical_weekday(today):
    """Next Wednesday (or today if it's Wed). Used as the canonical 'TWT' day
    for selecting which schedules are 'currently active typical weekday'."""
    delta = (2 - today.weekday()) % 7
    return today + datetime.timedelta(days=delta)


def parse_cif_file(path, target_tfl_ids, og_classify=False, target_date=None):
    """Read one TOC's gzipped JSON feed and emit trip dicts."""
    if target_date is None:
        target_date = next_typical_weekday(datetime.date.today())
    print(f"  {os.path.basename(path)}: target date {target_date} ({target_date.strftime('%a')})")

    tiploc_info = load_tiploc_info(path)
    name_index, _ = load_tfl_name_index()

    # Per-tfl_id TIPLOC->naptan maps. If classifying OG, we build all six up-front.
    tiploc_maps = {tfl_id: build_tiploc_to_naptan(tiploc_info, name_index, tfl_id)
                   for tfl_id in target_tfl_ids}
    for tfl_id in target_tfl_ids:
        print(f"    {tfl_id:12s}: {len(tiploc_maps[tfl_id])} TIPLOCs mapped to naptans")

    line_to_tiplocs = {tfl_id: set(m.keys()) for tfl_id, m in tiploc_maps.items()} if og_classify else None

    # Stream all schedules (~25k for OG, ~15k for EZL — small enough to hold)
    schedules = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            s = rec.get("JsonScheduleV1")
            if s:
                schedules.append(s)
    print(f"    {len(schedules)} raw schedules in feed")

    chosen = select_schedules_for_date(schedules, target_date)
    print(f"    {len(chosen)} after STP+date+weekday resolution")

    trips = []
    skipped_short = 0
    skipped_unclassified = 0
    weak_classification = 0
    by_line = Counter()
    for s in chosen:
        if og_classify:
            tfl_id, frac = classify_og_line(s, line_to_tiplocs)
            if not tfl_id or frac < 0.5:
                skipped_unclassified += 1
                continue
            if frac < 0.85:
                weak_classification += 1
        else:
            tfl_id = target_tfl_ids[0]

        stops = schedule_to_stops(s, tiploc_maps[tfl_id])
        if len(stops) < 2:
            skipped_short += 1
            continue

        line_code = TFL_TO_NUMBAT[tfl_id]
        trips.append({
            "line": line_code,
            "tfl_id": tfl_id,
            "origin": stops[0][0],
            "dep": stops[0][1],
            "stops": stops,
        })
        by_line[line_code] += 1

    print(f"    {len(trips)} trips emitted "
          f"({skipped_short} too short, {skipped_unclassified} unclassified, "
          f"{weak_classification} weakly classified)")
    print(f"    per line: {dict(by_line)}")
    return trips


def parse_all_cif():
    """Entry point used by build.py."""
    trips = []
    if os.path.exists(CIF_EZL):
        trips.extend(parse_cif_file(CIF_EZL, ["elizabeth"], og_classify=False))
    if os.path.exists(CIF_OG):
        trips.extend(parse_cif_file(CIF_OG, OG_LINES, og_classify=True))
    return trips


if __name__ == "__main__":
    today = datetime.date.today()
    print(f"Today: {today} ({today.strftime('%a')}), "
          f"target: {next_typical_weekday(today)}")
    trips = parse_all_cif()
    print(f"\nTotal: {len(trips)} trips")
    if trips:
        s = trips[0]
        print(f"sample: line={s['line']} tfl_id={s['tfl_id']} "
              f"dep={s['dep']//3600:02d}:{(s['dep']%3600)//60:02d} "
              f"stops={len(s['stops'])}")
        print(f"  first stop: {s['stops'][0]}")
        print(f"  last stop:  {s['stops'][-1]}")
