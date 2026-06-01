"""
Build trains.json from cached TfL Unified API timetables + NUMBAT per-line OD.

Pipeline:
  1. Load geometry.json (stations + naptan<->mnlc mapping + line colors).
  2. Walk data/api_cache/tt_*.json -> flatten into trip objects with real
     stop sequences and absolute departure/arrival times.
  3. Index trips by (numbat_line_code, naptan_o) -> sorted by dep time.
  4. For each per-line OD row (line, mnlc_o, mnlc_d, qhr, vol), translate
     the mnlcs to naptan ids, find trips on the line whose path includes
     o then d departing in the qhr window, distribute `vol` across them.
  5. Write trains.json (same schema as nycriders viewer expects).

This replaces the old NUMBAT-only synthesis: trains spawn at *real* schedule
times and inter-station travel comes from the API's `stationIntervals`, so we
get genuine per-trip variation instead of mode-average speed.
"""
import argparse
import csv
import json
import math
import os
import random
import time as time_module
from bisect import bisect_left
from collections import defaultdict

GEOMETRY_JSON = "geometry.json"
OUT_JSON = "trains.json"
STATS_JSON = "stats.json"
CACHE_DIR = "data/api_cache"
OD_PER_LINE_DIR = "data/od_per_line"
TXC_ZIP = "data/tfl-txc-demo.zip"   # if present, used as primary trip source

# We model a typical Tue/Wed/Thu. Different lines use different schedule
# labels: Bakerloo "Monday - Friday", Central "Monday - Thursday" + "Friday",
# Metropolitan "Mondays" + "Tuesdays" (single-day per route). Pick exactly one
# schedule per route in priority order:
#   1. "Monday - Thursday" / "Mondays" range  (closest analog to NUMBAT TWT)
#   2. "Monday - Friday" / "Mondays - Fridays" / "Weekday(s)"
#   3. "Tuesdays" / "Wednesdays" / "Thursdays" (single-day TWT-like fallback)
# Skip "Friday" / "Mondays" alone — atypical days NUMBAT TWT excludes.
def select_weekday_schedule(schedules):
    """Return the single weekday-typical schedule, or None."""
    pref1 = pref2 = pref3 = None
    for s in schedules:
        n = (s.get("name") or "").lower()
        if "mon" in n and "thu" in n and "fri" not in n:
            pref1 = pref1 or s
        elif ("mon" in n and "fri" in n) or "weekday" in n:
            pref2 = pref2 or s
        elif n in ("tuesdays", "wednesdays", "thursdays"):
            pref3 = pref3 or s
    return pref1 or pref2 or pref3

QHR_LEN_S = 15 * 60  # 900s — quarter-hour bin width
# If no train departs in a rider's qhr window, look forward by this much for
# the next train. Models the rider waiting a few extra minutes rather than
# vanishing. Forward-only — never matches a train that departed before the qhr.
# At the start of service most lines are still ramping up — trains from the
# termini haven't reached central yet, and the headway is large. NUMBAT TWT
# OD has rider entries in this window that real Tue-Wed-Thu schedules can't
# strictly cover; we stretch the slack so first-train waiters get matched
# rather than disappearing. Visually they trickle onto the platform over the
# first ~12 minutes (per WAIT_SUB_SPACING_S) then accumulate until the train.
QHR_SLACK_S = 2 * 60
RAMP_UP_SLACK_S = 30 * 60      # 06:00-08:00 — service propagating from termini
PRE_PEAK_SLACK_S = 60 * 60     # 04:00-06:00 — first trains starting; sparse headways


def qhr_slack(qhr):
    """Per-qhr forward search window. Larger during early-morning service
    ramp-up; default otherwise."""
    if 16 <= qhr < 24:    # 04:00 to 06:00
        return PRE_PEAK_SLACK_S
    if 24 <= qhr < 32:    # 06:00 to 08:00
        return RAMP_UP_SLACK_S
    return QHR_SLACK_S
# Max riders from a single OD row assigned to a single train. Chosen at ~20%
# of typical tube capacity (~900-1000) and ~40% of W&C (506) and ~70% of DLR
# (~280). Big OD rows get split into ceil(vol/CHUNK_MAX) chunks, each given a
# random dep_time within the qhr and routed to the *next* matching train. This
# breaks up the homogeneity of the old equal-split-across-all-matches scheme.
CHUNK_MAX = 200

# Each chunk's `per` riders fan out over WAIT_N_SUB sub-arrivals at
# WAIT_SUB_SPACING_S cadence so the station's waiting bubble grows smoothly
# instead of pulsing in lumps each qhr. If the wait is shorter than the full
# span, sub-arrivals compress evenly so the last one still lands at train_dep.
WAIT_N_SUB = 5
WAIT_SUB_SPACING_S = 3 * 60

# Walk-bubble timings: how long riders walk to/from a station from a
# random nearby point.
ORIGIN_WALK_S = 120
DEST_WALK_S = 120
WALK_BIN_S = 60          # aggregate walks into 1-minute bins
WALK_MIN_RIDERS = 10     # drop tiny aggregations to keep file size sane

# Speed smoothing — minute-rounded timetables produce some "impossibly fast"
# inter-stop segments next to "implausibly slow" ones. We redistribute time
# from slow donors to fast segments so each segment's cruise speed stays under
# its line threshold. See `smooth_trip_speeds`.
SPEED_THRESHOLDS = {
    # Tube: real max ~80 km/h
    "BAK": 90, "CEN": 90, "CIR": 90, "DIS": 90, "HAM": 90, "JUB": 90,
    "MET": 110, "NOR": 90, "PIC": 90, "VIC": 90, "WAC": 90,
    "DLR": 90,
    # Crossrail tunnel + outer National Rail
    "EZL": 160,
    # Class 378/710 on NR track ~120 km/h
    "NLL": 130, "ELL": 130,
    "WEL": 130, "GOB": 130, "URL": 130, "WAG": 130,
}
DEFAULT_SPEED_THRESHOLD = 100
SMOOTH_DWELL_S = 20
SMOOTH_ACCEL_DECEL_S = 20            # 20s lost to accel + decel
SMOOTH_OVERHEAD_S = SMOOTH_DWELL_S + SMOOTH_ACCEL_DECEL_S
SMOOTH_PASSES = 3                    # multi-pass to clean chains of fast segs
# Pre-smoothing jitter — most published timetables round to the minute, so
# without this every train arrives exactly on :00, and identically-scheduled
# trains stack visually. A small random per-stop offset (±JITTER_S seconds)
# breaks the on-the-minute pattern *before* speed smoothing absorbs any
# unrealistic gaps the jitter introduces. Deterministic seed for reproducibility.
JITTER_S = 10

# After the strict hard-threshold passes some hard violators remain stuck:
# their fast neighbors can't donate (already at threshold) and their slow
# neighbors are saturated. The soft "rescue" pass relaxes the donor cap so a
# hard-violator can steal from a donor whose speed is in (soft, hard] — at
# the cost of pushing the donor slightly past hard. We only take half the
# deficit per try and iterate, so the violation cascades out from chain edges
# toward the middle. Donor cap = hard / SOFT_FACTOR ≈ 1.18 * hard.
SOFT_THRESHOLD_FACTOR = 0.85
SOFT_DONOR_FACTOR = 1.0 / SOFT_THRESHOLD_FACTOR
SOFT_TAKE_FRACTION = 0.5
SOFT_RESCUE_PASSES = 4

# 2-leg interchange routing. Many lines have branches whose services don't
# through-run (DLR Stratford↔Lewisham, Northern Battersea↔Bank, EZL Abbey
# Wood↔Stratford). When no direct trip serves O→D, we try to find a single
# same-line interchange X where the rider alights one trip and boards
# another. TRANSFER_MIN_S is a floor for platform-change time; the cap is
# loose enough to absorb timetable rounding plus a short walk.
TRANSFER_MIN_S = 60
TRANSFER_MAX_S = 10 * 60

# Per-line interchange stations — only these naptans qualify as a transfer
# point for 2-leg routing. Curated to actual branch-junction stops that
# every service in both directions passes through. Lines absent from this
# map skip 2-leg routing entirely (no Bakerloo-style trunk lines need it).
LINE_INTERCHANGES = {
    "DLR": {
        "940GZZDLCAN",   # Canary Wharf — Bank↔Lewisham passes through, Stratford↔CW terminates
        "940GZZDLPOP",   # Poplar — Bank/Stratford/Beckton/Woolwich branches converge
        "940GZZDLWFE",   # Westferry — Bank trunk; Stratford branch joins via WIQ
        "940GZZDLWIQ",   # West India Quay — same trunk
        "940GZZDLCGT",   # Canning Town — Stratford International ↔ Beckton/Woolwich
    },
    "NOR": {
        "940GZZLUKNG",   # Kennington — Battersea ext joins Bank & Charing X branches (south)
        "940GZZLUCTN",   # Camden Town — Bank/Charing X split (north)
        "940GZZLUEUS",   # Euston — both branches stop here
    },
    "EZL": {
        "910GPADTLL",    # Paddington (Crossrail) — western↔eastern through-run boundary
        "910GPADTON",    # Paddington mainline — Reading short-turn terminus
        "910GLIVSTLL",   # Liverpool Street (Crossrail) — Shenfield↔central boundary
        "910GLIVST",     # Liverpool Street mainline
        "910GWCHAPXR",   # Whitechapel — Abbey Wood vs Shenfield branch split
        "910GHAYESAH",   # Hayes & Harlington — Heathrow branch junction
    },
    "ELL": {
        "910GCNDAW",     # Canada Water — Clapham Junction vs Crystal Palace split
        "910GSURREYQ",   # Surrey Quays — New Cross vs Crystal Palace split
        "910GWCHAPEL",   # Whitechapel — northern trunk transfer point
        "910GNEWXGTE",   # New Cross Gate — Crystal Palace vs West Croydon split
    },
    "DIS": {
        "940GZZLUECT",   # Earl's Court — Wimbledon / Edgware Rd / Richmond / Ealing junction
        "940GZZLUTNG",   # Turnham Green — Richmond vs Ealing Broadway branch
        "940GZZLUHSD",   # Hammersmith (Dist) — Edgware Rd branch
    },
    "PIC": {
        "940GZZLUACT",   # Acton Town — Heathrow/Hounslow vs Uxbridge branch split
        "940GZZLUHNX",   # Hatton Cross — Heathrow T4 loop vs T5 branch
        "940GZZLUHSD",   # Hammersmith (Picc) — shared with District
    },
    "MET": {
        "940GZZLUHOH",   # Harrow-on-the-Hill — Watford/Amersham/Chesham/Uxbridge converge
        "940GZZLUMPK",   # Moor Park — Watford vs Amersham/Chesham split
    },
    "WAG": {
        "910GHAKNYNM",   # Hackney Downs — Chingford / Enfield / Cheshunt branches split
    },
    "HAM": {
        "940GZZLUBST",   # Baker Street — Edgware Road branch joins H&C trunk
    },
    "CIR": {
        "940GZZLUBST",   # Baker Street — Circle loop "ends" meet here
    },
}


# Per-line CSV basename -> list of NUMBAT line codes it covers.
#
# IMPORTANT: ssl.csv and ssp.csv are NUMBAT *aggregate* views — empirically
# sum(ssl.vol) ~= sum(dis + ham + met), and sum(ssp.vol) ~= sum(pic + ssl).
# They represent the same riders already counted in the per-line files,
# pivoted differently. Including them as additional OD sources triple-counts
# District/H&C/Met boardings (it's why the District line was showing 1.8M
# riders/day instead of TfL's reported ~660k). Use only individual-line files.
#
# Same redundancy on the Bakerloo + Watford DC corridor — but inverted:
# bak.csv (291k vol) is a strict subset of bakwdc.csv (330k vol, BAK + WEL
# aggregate), and WEL doesn't have its own file because Lioness shares track
# with Bakerloo and gate-tap data can't always tell which service was taken.
# Including both files double-counts every BAK boarding (inflates Oxford
# Circus, Lambeth North, etc. ~2x). Keep only bakwdc.csv — it's the single
# source of truth covering both lines. Build assigns each row to whichever
# of [BAK, WEL] has a matching trip, so no rider is counted twice.
PER_LINE_TO_CODES = {
    "bakwdc":  ["BAK", "WEL"],
    "cen":     ["CEN"],
    "dis":     ["DIS"],
    "dlr":     ["DLR"],
    "ezl":     ["EZL"],
    "ham":     ["HAM", "CIR"],
    "jub":     ["JUB"],
    "loa":     ["WAG"],
    "loe":     ["ELL"],
    "log":     ["GOB"],
    "lon":     ["NLL"],
    "lor":     ["URL"],
    "met":     ["MET"],
    "nor":     ["NOR"],
    "pic":     ["PIC"],
    "vic":     ["VIC"],
    "wat":     ["WAC"],
}


# Hand-curated geometry patches. NUMBAT OD references MNLCs/naptans that fall
# through build_geometry's TfL-API-driven pipeline:
#   * Split-complex stations (Paddington / Liverpool Street) — NUMBAT labels
#     EZL boardings under the LU MNLC because riders tap through the LU
#     gateline to reach the Crossrail platforms.
#   * Naptan code variants (910GLIVST mainline platforms vs 910GLIVSTLL
#     Crossrail platforms) — TfL line endpoints return different codes for
#     the same physical station.
#   * Stations missing from NUMBAT's Branches sheet (Burnham, Langley, etc.)
#     so build_geometry filters them out of stations[].

# Naptan aliases — alternate naptan codes for the same physical station.
# For each, we register the alias in naptan_to_mnlc (so trip stops emitted
# with the alias form resolve to an MNLC) and add it to the listed lines'
# per-line naptan maps (so OD lookups find trips at that naptan).
# Format: { alias_naptan: (mnlc, [lines_using_this_form]) }
NAPTAN_ALIASES = {
    # Liverpool Street mainline (high-level) platforms — every Weaver train
    # uses these, plus some EZL terminators. build_geometry only knew the
    # Crossrail (low-level) variant 910GLIVSTLL.
    "910GLIVST":  (6965, ["EZL", "WAG"]),
    # Paddington mainline (GWR-side) platforms, adjacent to Crossrail box.
    "910GPADTON": (3087, ["EZL"]),
    # Paddington (H&C Line) Underground — physically distinct platform group
    # from the Bakerloo/District/Circle Paddington (940GZZLUPAC). H&C and
    # Circle trains stop here; NUMBAT OD labels all Paddington boardings
    # under mnlc 670 ("Paddington TfL") regardless, so we union the PAH
    # naptan into the same MNLC on the H&C and Circle per-line maps.
    "940GZZLUPAH": (670, ["HAM", "CIR"]),
}

# Station-complex groups — MNLCs that share a physical gateline. NUMBAT
# labels boardings under whichever MNLC the rider tapped, but at split-mode
# complexes the rider may board a different line's platforms (LU gate → EZL
# train). Per-line naptan sets are unioned across each group so an OD lookup
# by any member MNLC resolves to trips at any member naptan on every line
# serving the complex.
COMPLEX_GROUPS = [
    [670, 3087],   # Paddington TfL (LU) ⟷ Paddington NR (Crossrail/mainline)
    [634, 6965],   # Liverpool Street LU ⟷ Liverpool Street NR
    [780, 7090],   # Heathrow T1,2,3 LU (legacy — T1 closed 2015) ⟷ T2&3 EL
]

# Stations NUMBAT references in OD data but are absent from its Branches
# sheet (so build_geometry omits them from stations[] entirely). Adding them
# here gives OD lookups a naptan to match on, and gives the train renderer
# coords to place the train at each stop.
# Format: { mnlc: (name, asc, lat, lon, {line: naptan_on_that_line}) }
EXTRA_STATIONS = {
    700:  ("Shepherd's Bush",  "SBCu", 51.504358, -0.218324, {"CEN": "940GZZLUSBC"}),
    1444: ("Euston NR",        "EUSr", 51.528805, -0.134257, {"WEL": "910GEUSTON"}),
    3171: ("Langley",          "LNYr", 51.507855, -0.541864, {"EZL": "910GLANGLEY"}),
    3176: ("Burnham",          "BNMr", 51.523281, -0.646568, {"EZL": "910GBNHAM"}),
    5150: ("New Cross",        "NWXr", 51.476372, -0.032627, {"ELL": "910GNWCRELL"}),
    5376: ("Norwood Junction", "NWDr", 51.397171, -0.074698, {"ELL": "910GNORWDJ"}),
}


def patch_geometry(geo):
    """Apply the patches above to geo in place.

    Order matters: stations and aliases run first so the per-line maps are
    populated, then complex-group unions copy the populated entries to their
    split-complex siblings.
    """
    stations = geo["stations"]
    n2m = geo["naptan_to_mnlc"]
    m2n = geo["mnlc_to_naptans"]
    m2n_pl = geo.setdefault("mnlc_to_naptans_per_line", {})

    n_extra = 0
    for mnlc, (name, asc, lat, lon, line_naptans) in EXTRA_STATIONS.items():
        key = str(mnlc)
        if key not in stations:
            stations[key] = {
                "name": name, "asc": asc,
                "lat": round(lat, 6), "lon": round(lon, 6),
                "naptan_ids": sorted(set(line_naptans.values())),
                "lines": sorted(line_naptans),
            }
            n_extra += 1
        for line_code, naptan in line_naptans.items():
            n2m.setdefault(naptan, mnlc)
            entries = m2n.setdefault(key, [])
            if naptan not in entries:
                entries.append(naptan)
            line_bucket = m2n_pl.setdefault(line_code, {})
            line_entries = line_bucket.setdefault(key, [])
            if naptan not in line_entries:
                line_entries.append(naptan)

    n_alias = 0
    for alias, (mnlc, lines) in NAPTAN_ALIASES.items():
        key = str(mnlc)
        if alias not in n2m:
            n2m[alias] = mnlc
            n_alias += 1
        entries = m2n.setdefault(key, [])
        if alias not in entries:
            entries.append(alias)
        for line_code in lines:
            line_bucket = m2n_pl.setdefault(line_code, {})
            line_entries = line_bucket.setdefault(key, [])
            if alias not in line_entries:
                line_entries.append(alias)

    n_groups = 0
    for group in COMPLEX_GROUPS:
        keys = [str(m) for m in group]
        for line_code, bucket in m2n_pl.items():
            unioned = set()
            for k in keys:
                unioned.update(bucket.get(k, []))
            if not unioned:
                continue
            for k in keys:
                bucket[k] = sorted(unioned)
        n_groups += 1

    print(f"  patched geometry: +{n_extra} stations, +{n_alias} naptan aliases, "
          f"{n_groups} complex groups unioned")


# Trip-level patches for stops that upstream timetable sources drop. Currently
# railwaydata.co.uk's NTN PDF omits Burnt Oak as a row entirely — every other
# Northern Line station appears, but the Edgware-branch tables jump straight
# from Colindale to Edgware. We insert it with a linearly-interpolated time.
# Each entry: { line: { (prev_naptan, next_naptan): (insert_naptan, fraction) } }
# where fraction is where along the prev→next leg to place the inserted stop.
MISSING_STOP_PATCHES = {
    "NOR": {
        # Edgware branch order: Edgware → Burnt Oak → Colindale. Burnt Oak
        # sits ~55% along the Colindale→Edgware leg (1.4 km of 2.5 km total).
        ("940GZZLUCND", "940GZZLUEGW"): ("940GZZLUBTK", 0.55),
        ("940GZZLUEGW", "940GZZLUCND"): ("940GZZLUBTK", 0.45),
    },
}


def patch_missing_stops(trips):
    """Insert stops that upstream sources drop from individual trip patterns.
    Mutates each trip's stop list in place. See MISSING_STOP_PATCHES."""
    n_inserted = 0
    for trip in trips:
        patches = MISSING_STOP_PATCHES.get(trip["line"])
        if not patches:
            continue
        stops = trip["stops"]
        new_stops = []
        for i, (n, s) in enumerate(stops):
            new_stops.append((n, s))
            if i + 1 >= len(stops):
                continue
            patch = patches.get((n, stops[i + 1][0]))
            if not patch:
                continue
            insert_n, frac = patch
            next_s = stops[i + 1][1]
            new_stops.append((insert_n, int(s + (next_s - s) * frac)))
            n_inserted += 1
        trip["stops"] = new_stops
    if n_inserted:
        print(f"  inserted {n_inserted} missing stops "
              f"(see MISSING_STOP_PATCHES)")
    return trips


def load_geometry():
    print(f"Loading {GEOMETRY_JSON}...")
    with open(GEOMETRY_JSON, encoding="utf-8") as f:
        geo = json.load(f)
    print(f"  {len(geo['stations'])} stations, "
          f"{len(geo['branches'])} branches, "
          f"{len(geo['lines'])} polylines")
    patch_geometry(geo)
    return geo


def parse_hm(j):
    h = int(j["hour"])
    m = int(j["minute"])
    return h * 3600 + m * 60


def flatten_trips(numbat_to_tfl):
    """Walk every cached tt_<line>__<stopId>.json and emit trip dicts.

    Each trip:
      {
        "line": numbat_code,         # e.g. "BAK"
        "tfl_id": tfl_id,            # e.g. "bakerloo"
        "origin": naptan_id,
        "dep": int(seconds_since_midnight),
        "stops": [(naptan_id, secs), ...],   # ordered, includes origin at offset 0
      }
    """
    print("Flattening cached timetables into trips...")
    tfl_to_numbat = {v: k for k, v in numbat_to_tfl.items()}

    trips = []
    files = sorted(os.listdir(CACHE_DIR))
    n_files = sum(1 for f in files if f.startswith("tt_") and f.endswith(".json"))
    seen_keys = set()
    skipped_no_line = 0
    skipped_other_sched = 0
    skipped_bad = 0

    for fname in files:
        if not fname.startswith("tt_") or not fname.endswith(".json"):
            continue
        # tt_<tfl_id>__<naptan>.json
        rest = fname[3:-5]  # strip prefix/suffix
        if "__" not in rest:
            continue
        tfl_id, origin_naptan = rest.split("__", 1)
        line_code = tfl_to_numbat.get(tfl_id)
        if not line_code:
            skipped_no_line += 1
            continue
        try:
            with open(f"{CACHE_DIR}/{fname}", encoding="utf-8") as f:
                tt = json.load(f)
        except (json.JSONDecodeError, OSError):
            skipped_bad += 1
            continue
        if "error" in tt:
            skipped_bad += 1
            continue

        ts = tt.get("timetable", {}) or {}
        for route in ts.get("routes", []) or []:
            # Map intervalId -> list of (stopId, time_offset_seconds)
            iv_by_id = {}
            for si in route.get("stationIntervals", []) or []:
                ivs = []
                for it in si.get("intervals", []) or []:
                    sid = it.get("stopId")
                    tta = it.get("timeToArrival")
                    if sid is None or tta is None:
                        continue
                    ivs.append((sid, int(round(float(tta) * 60))))
                if ivs:
                    iv_by_id[int(si.get("id", -1))] = ivs

            sched = select_weekday_schedule(route.get("schedules", []) or [])
            if sched is None:
                skipped_other_sched += 1
            else:
                for kj in sched.get("knownJourneys", []) or []:
                    iv_id = int(kj.get("intervalId", -1))
                    ivs = iv_by_id.get(iv_id)
                    if not ivs:
                        continue
                    dep_secs = parse_hm(kj)
                    stops = [(origin_naptan, dep_secs)]
                    for sid, off in ivs:
                        stops.append((sid, dep_secs + off))
                    # Dedupe trips that get re-emitted from different origins
                    # by hashing the full stop sequence + dep
                    key = (line_code, origin_naptan, dep_secs, iv_id, len(ivs))
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    trips.append({
                        "line": line_code,
                        "tfl_id": tfl_id,
                        "origin": origin_naptan,
                        "dep": dep_secs,
                        "stops": stops,
                    })
    print(f"  {len(trips)} unique weekday trips "
          f"({n_files} cache files, {skipped_no_line} skipped no-line, "
          f"{skipped_bad} bad/empty)")
    by_line = defaultdict(int)
    for t in trips:
        by_line[t["line"]] += 1
    print(f"  trips per line: {dict(sorted(by_line.items()))}")
    # Diagnostic: every distinct schedule name we saw, so we can tell if a line
    # uses a label our matcher doesn't recognize.
    sched_names = defaultdict(int)
    for fname in sorted(os.listdir(CACHE_DIR)):
        if not (fname.startswith("tt_") and fname.endswith(".json")):
            continue
        try:
            with open(f"{CACHE_DIR}/{fname}", encoding="utf-8") as f:
                tt = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if "error" in tt:
            continue
        for route in (tt.get("timetable", {}) or {}).get("routes", []) or []:
            for s in route.get("schedules", []) or []:
                nm = s.get("name") or "<none>"
                sched_names[nm] += 1
    if sched_names:
        print(f"  schedule names seen across all cache files (count = #routes):")
        for nm, ct in sorted(sched_names.items(), key=lambda kv: -kv[1]):
            print(f"    {ct:>5}  {nm!r}")
    return trips


def build_trip_index(trips):
    """For each (line_code, naptan_id), list of (dep_at_naptan, trip_idx, pos_in_trip).

    Sorted by dep_at_naptan so we can binary-search by qhr window.
    """
    print("Indexing trips by (line, stop)...")
    idx = defaultdict(list)
    for ti, t in enumerate(trips):
        line = t["line"]
        for pos, (nid, secs) in enumerate(t["stops"]):
            idx[(line, nid)].append((secs, ti, pos))
    for k in idx:
        idx[k].sort()
    print(f"  {len(idx)} (line, stop) keys")
    return idx


def assign_riders(trips, trip_index, geo, sample_rate=1.0):
    """Walk per-line OD CSVs; assign vol to matching trips."""
    print(f"Assigning riders (sample_rate={sample_rate})...")
    mnlc_to_naptans = {int(k): v for k, v in geo["mnlc_to_naptans"].items()}
    # Per-line index: line_code -> {mnlc -> [naptans]}. When a per-line OD row
    # asks about a mnlc on the Overground branch, prefer the Overground naptan
    # over the colocated Underground one.
    mnlc_to_naptans_per_line = {
        line: {int(m): v for m, v in by_mnlc.items()}
        for line, by_mnlc in geo.get("mnlc_to_naptans_per_line", {}).items()
    }

    # Per-trip boarding/alighting accumulators (parallel to `trips`)
    boardings = [defaultdict(float) for _ in trips]
    alightings = [defaultdict(float) for _ in trips]
    # Per-chunk waiting events: (origin_naptan, t_arrive, t_board, riders).
    # Aggregated downstream into per-station waiting timelines.
    waiting_events = []
    # Per-chunk walk events. Origin: rider walks to platform, arriving at
    # dep_time. Dest: rider walks away from platform, starting at alight_time.
    origin_walk_events = []  # (naptan_o, dep_time, riders)
    dest_walk_events = []    # (naptan_d, alight_time, riders)

    rng = random.Random(42)

    # Pre-extract dep arrays for fast bisect per (line, naptan_o)
    dep_arrays = {k: [v[0] for v in lst] for k, lst in trip_index.items()}

    total_assigned = 0
    total_unassigned = 0
    total_riders_assigned = 0.0
    total_riders_unassigned = 0.0
    n_2leg_chunks = 0   # diagnostic: how many chunks landed via 2-leg routing

    for fname in sorted(os.listdir(OD_PER_LINE_DIR)):
        if not fname.endswith(".csv"):
            continue
        base = fname[:-4]
        line_codes = PER_LINE_TO_CODES.get(base)
        if not line_codes:
            continue

        n_assigned = 0
        n_unassigned = 0
        riders_assigned = 0.0
        riders_unassigned = 0.0

        with open(f"{OD_PER_LINE_DIR}/{fname}", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if rng.random() > sample_rate:
                    continue
                try:
                    o = int(row["mode_mnlc_o"])
                    d = int(row["mode_mnlc_d"])
                    qhr = int(row["qhr"]) - 1   # 1-indexed from midnight
                    vol = float(row["vol"])
                except (KeyError, ValueError):
                    continue
                if vol <= 0 or o == d:
                    continue
                if not (0 <= qhr < 104):
                    continue

                # Per-line lookup first (preferred): captures cases like
                # Bethnal Green LO vs LU sharing a site but different MNLCs.
                # Multiple line codes per OD row (e.g. ssl=DIS+HAM+MET+CIR) so
                # union the per-line entries; fall back to flat if neither line
                # has a per-line entry for this mnlc.
                naptans_o_set = set()
                naptans_d_set = set()
                for line_code in line_codes:
                    pl = mnlc_to_naptans_per_line.get(line_code)
                    if not pl:
                        continue
                    naptans_o_set.update(pl.get(o, []))
                    naptans_d_set.update(pl.get(d, []))
                if not naptans_o_set:
                    naptans_o_set.update(mnlc_to_naptans.get(o, []))
                if not naptans_d_set:
                    naptans_d_set.update(mnlc_to_naptans.get(d, []))
                naptans_o = list(naptans_o_set)
                naptans_d = naptans_d_set
                if not naptans_o or not naptans_d:
                    n_unassigned += 1
                    riders_unassigned += vol
                    continue

                window_start = qhr * QHR_LEN_S
                window_end = window_start + QHR_LEN_S

                # Find the earliest matching trip whose dep_time is in
                # [t_start, t_start + slack] and whose route serves O then D.
                # Returns (ti, pos_o, pos_d) or None.
                def find_next(t_start, slack):
                    best = None  # (dep_time, ti, pos_o, pos_d)
                    horizon = t_start + slack
                    for line_code in line_codes:
                        for nid_o in naptans_o:
                            key = (line_code, nid_o)
                            deps = dep_arrays.get(key)
                            if not deps:
                                continue
                            lst = trip_index[key]
                            lo_i = bisect_left(deps, t_start)
                            for k in range(lo_i, len(deps)):
                                d_t = deps[k]
                                if d_t > horizon:
                                    break
                                if best is not None and d_t >= best[0]:
                                    break
                                _, ti, pos_o = lst[k]
                                stops = trips[ti]["stops"]
                                for j in range(pos_o + 1, len(stops)):
                                    if stops[j][0] in naptans_d:
                                        best = (d_t, ti, pos_o, j)
                                        break
                    return best

                # Fallback when no direct trip serves O→D: same-line 2-leg via
                # a LINE_INTERCHANGES station X. Picks (T1, X, T2) minimizing
                # arrival time at D, subject to T1 dep in [t_start,t_start+slack]
                # and T2 dep at X in [t_arr_x_1 + TRANSFER_MIN_S, +MAX_S].
                def find_next_2leg(t_start, slack):
                    horizon_1 = t_start + slack
                    best = None  # (t_arr_d, ti_1, pos_o_1, pos_x_1, t_arr_x_1,
                                 #  ti_2, pos_x_2, pos_d_2, t_dep_x_2)
                    for line_code in line_codes:
                        interchanges = LINE_INTERCHANGES.get(line_code)
                        if not interchanges:
                            continue
                        for nid_o in naptans_o:
                            key_o = (line_code, nid_o)
                            deps_o = dep_arrays.get(key_o)
                            if not deps_o:
                                continue
                            lst_o = trip_index[key_o]
                            i_lo = bisect_left(deps_o, t_start)
                            for ki in range(i_lo, len(deps_o)):
                                t_dep_o = deps_o[ki]
                                if t_dep_o > horizon_1:
                                    break
                                if best is not None and t_dep_o >= best[0]:
                                    break
                                _, ti_1, pos_o_1 = lst_o[ki]
                                stops_1 = trips[ti_1]["stops"]
                                for jx in range(pos_o_1 + 1, len(stops_1)):
                                    nid_x, t_arr_x_1 = stops_1[jx]
                                    if nid_x not in interchanges:
                                        continue
                                    key_x = (line_code, nid_x)
                                    deps_x = dep_arrays.get(key_x)
                                    if not deps_x:
                                        continue
                                    lst_x = trip_index[key_x]
                                    t_x_lo = t_arr_x_1 + TRANSFER_MIN_S
                                    t_x_hi = t_arr_x_1 + TRANSFER_MAX_S
                                    k2_lo = bisect_left(deps_x, t_x_lo)
                                    for k2 in range(k2_lo, len(deps_x)):
                                        t_dep_x_2 = deps_x[k2]
                                        if t_dep_x_2 > t_x_hi:
                                            break
                                        _, ti_2, pos_x_2 = lst_x[k2]
                                        if ti_2 == ti_1:
                                            continue
                                        stops_2 = trips[ti_2]["stops"]
                                        for jd in range(pos_x_2 + 1, len(stops_2)):
                                            if stops_2[jd][0] in naptans_d:
                                                t_arr_d = stops_2[jd][1]
                                                if best is None or t_arr_d < best[0]:
                                                    best = (t_arr_d, ti_1, pos_o_1, jx,
                                                            t_arr_x_1, ti_2, pos_x_2, jd,
                                                            t_dep_x_2)
                                                break
                    return best

                # Chunk the OD vol so a single big row doesn't pile onto one
                # train. Each chunk picks a *stratified* random dep_time —
                # we slice the qhr into n_chunks equal sub-windows and draw
                # one t per slice — so chunks deterministically spread across
                # the qhr instead of colliding by birthday-paradox chance.
                # Slack spans the qhr plus a per-qhr forward window — wider
                # during early-morning ramp-up so first-train waiters get
                # matched. See qhr_slack().
                chunk_slack = QHR_LEN_S + qhr_slack(qhr)
                n_chunks = max(1, math.ceil(vol / CHUNK_MAX))
                per = vol / n_chunks
                sub_width = QHR_LEN_S / n_chunks
                row_assigned = 0.0
                for i in range(n_chunks):
                    dep_time = window_start + (i + rng.random()) * sub_width
                    m = find_next(dep_time, chunk_slack)
                    if m is not None:
                        train_dep, ti, pos_o, pos_d = m
                        boardings[ti][pos_o] += per
                        alightings[ti][pos_d] += per
                        row_assigned += per
                        line = trips[ti]["line"]
                        nid_o = trips[ti]["stops"][pos_o][0]
                        nid_d = trips[ti]["stops"][pos_d][0]
                        train_alight = trips[ti]["stops"][pos_d][1]
                        # Origin walk: rider arrives at platform at dep_time
                        # (after walking from somewhere nearby).
                        origin_walk_events.append((line, nid_o, dep_time, per))
                        # Destination walk: rider departs station at the alight time.
                        dest_walk_events.append((line, nid_d, train_alight, per))
                        # Spread the chunk's `per` riders over WAIT_N_SUB sub-arrivals
                        # at WAIT_SUB_SPACING_S cadence (compressed if the wait is
                        # shorter than the full span). Each sub-arrival adds per/N
                        # to the platform; boarding at train_dep removes all `per`.
                        wait = train_dep - dep_time
                        if wait > 0:
                            max_span = WAIT_SUB_SPACING_S * (WAIT_N_SUB - 1)
                            spacing = min(max_span, wait) / (WAIT_N_SUB - 1)
                            sub_per = per / WAIT_N_SUB
                            for k in range(WAIT_N_SUB):
                                waiting_events.append(
                                    (line, nid_o, dep_time + k * spacing, train_dep, sub_per)
                                )
                        continue
                    # No direct trip — try a same-line 2-leg via interchange X.
                    m2 = find_next_2leg(dep_time, chunk_slack)
                    if m2 is None:
                        continue
                    (t_arr_d, ti_1, pos_o_1, pos_x_1, t_arr_x_1,
                     ti_2, pos_x_2, pos_d_2, t_dep_x_2) = m2
                    boardings[ti_1][pos_o_1] += per
                    alightings[ti_1][pos_x_1] += per
                    boardings[ti_2][pos_x_2] += per
                    alightings[ti_2][pos_d_2] += per
                    row_assigned += per
                    n_2leg_chunks += 1
                    line = trips[ti_1]["line"]
                    nid_o = trips[ti_1]["stops"][pos_o_1][0]
                    nid_x = trips[ti_1]["stops"][pos_x_1][0]
                    nid_d = trips[ti_2]["stops"][pos_d_2][0]
                    t_dep_o_1 = trips[ti_1]["stops"][pos_o_1][1]
                    origin_walk_events.append((line, nid_o, dep_time, per))
                    dest_walk_events.append((line, nid_d, t_arr_d, per))
                    # Waiting at O: same fan-out pattern as the direct case.
                    wait_o = t_dep_o_1 - dep_time
                    if wait_o > 0:
                        max_span = WAIT_SUB_SPACING_S * (WAIT_N_SUB - 1)
                        spacing = min(max_span, wait_o) / (WAIT_N_SUB - 1)
                        sub_per = per / WAIT_N_SUB
                        for k in range(WAIT_N_SUB):
                            waiting_events.append(
                                (line, nid_o, dep_time + k * spacing, t_dep_o_1, sub_per)
                            )
                    # Waiting at X: all riders alight T1 together at t_arr_x_1
                    # and board T2 at t_dep_x_2. Single waiting event spans
                    # the transfer interval.
                    if t_dep_x_2 > t_arr_x_1:
                        waiting_events.append(
                            (line, nid_x, t_arr_x_1, t_dep_x_2, per)
                        )

                if row_assigned <= 0:
                    n_unassigned += 1
                    riders_unassigned += vol
                else:
                    n_assigned += 1
                    riders_assigned += row_assigned
                    riders_unassigned += (vol - row_assigned)

        total_assigned += n_assigned
        total_unassigned += n_unassigned
        total_riders_assigned += riders_assigned
        total_riders_unassigned += riders_unassigned
        coverage = riders_assigned / max(1.0, riders_assigned + riders_unassigned) * 100
        line_label = "+".join(line_codes)
        print(f"  {fname:14s} -> {line_label:20s} "
              f"{n_assigned:7d} ok, {n_unassigned:7d} unassigned  "
              f"({riders_assigned:>11,.0f} / {riders_assigned + riders_unassigned:>11,.0f} riders, "
              f"{coverage:5.1f}% covered)"
        )

    overall = total_riders_assigned / max(1.0, total_riders_assigned + total_riders_unassigned) * 100
    print(f"  TOTAL: {total_assigned} assigned, {total_unassigned} unassigned, "
          f"{total_riders_assigned:,.0f} riders ({overall:.1f}% coverage)")
    print(f"  2-leg chunks: {n_2leg_chunks}")
    print(f"  {len(waiting_events)} waiting events generated")
    print(f"  {len(origin_walk_events)} origin + {len(dest_walk_events)} dest walk events")
    return boardings, alightings, waiting_events, origin_walk_events, dest_walk_events


def build_line_adjacency(branches_for_line):
    """naptan -> set of adjacent naptans (across all branches of the line)."""
    adj = defaultdict(set)
    for br in branches_for_line:
        ids = br["naptan_ids"]
        for i in range(len(ids) - 1):
            adj[ids[i]].add(ids[i + 1])
            adj[ids[i + 1]].add(ids[i])
    return adj


def shortest_naptan_path(adj, src, dst):
    """BFS through the line graph; return [src, ..., dst] or None."""
    if src == dst:
        return [src]
    parent = {src: None}
    frontier = [src]
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj.get(u, ()):
                if v in parent:
                    continue
                parent[v] = u
                if v == dst:
                    path = [v]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    return path[::-1]
                nxt.append(v)
        frontier = nxt
    return None


def resolve_link_shape(line, n_a, n_b, link_shapes, line_adj, cache):
    """Return the polyline following the line's track from n_a to n_b. Tries
    direct lookup first. On miss — skip-stop services like Piccadilly express
    Acton Town→Hammersmith or Elizabeth fast Reading→Slough — runs BFS across
    the line's full adjacency graph (trunk + all branches) and concatenates
    per-edge shapes. Caches results so repeat skip patterns are O(1)."""
    direct = link_shapes.get(f"{line}:{n_a}:{n_b}")
    if direct:
        return direct
    cached = cache.get((line, n_a, n_b))
    if cached is not None:
        return cached or None
    chain = None
    path = shortest_naptan_path(line_adj.get(line, {}), n_a, n_b)
    if path and len(path) >= 2:
        pieces = []
        ok = True
        for i in range(len(path) - 1):
            fwd = link_shapes.get(f"{line}:{path[i]}:{path[i+1]}")
            if fwd:
                seg = fwd
            else:
                rev = link_shapes.get(f"{line}:{path[i+1]}:{path[i]}")
                # Reverse so seg always reads path[i] → path[i+1]; without
                # this, mixing forward and reverse shapes produces visible
                # zig-zags through each backwards-stored sub-link.
                seg = list(reversed(rev)) if rev else None
            if not seg:
                ok = False
                break
            if pieces and pieces[-1] == seg[0]:
                pieces.extend(seg[1:])
            else:
                pieces.extend(seg)
        if ok and len(pieces) > 2:
            chain = pieces
    cache[(line, n_a, n_b)] = chain or []
    return chain


def expand_with_link_shapes(timeline, naptans_at, line, link_shapes,
                             line_adj, chain_cache):
    """Insert intermediate shape waypoints between consecutive station entries
    so trains follow the OSM track curve rather than hopping in straight lines.
    Waypoints are 3-element [secs, lat, lon] (no rider counts); the renderer
    distinguishes them by length."""
    if len(timeline) < 2 or not link_shapes:
        return timeline
    out = [timeline[0]]
    for i in range(len(timeline) - 1):
        n_a = naptans_at[i]
        n_b = naptans_at[i + 1]
        poly = resolve_link_shape(line, n_a, n_b, link_shapes,
                                   line_adj, chain_cache)
        if poly and len(poly) > 2:
            t0 = timeline[i][0]
            t1 = timeline[i + 1][0]
            travel = t1 - t0
            dists = [0.0]
            for j in range(1, len(poly)):
                dx = poly[j][0] - poly[j - 1][0]
                dy = poly[j][1] - poly[j - 1][1]
                dists.append(dists[-1] + math.hypot(dx, dy))
            total = dists[-1]
            if total > 1e-12 and travel > 0:
                for j in range(1, len(poly) - 1):
                    frac = dists[j] / total
                    t = t0 + frac * travel
                    out.append([round(t), round(poly[j][1], 5), round(poly[j][0], 5)])
        out.append(timeline[i + 1])
    return out


def build_walks(walk_events, per_line_station_coords, naptan_coords,
                 naptan_to_mnlc, stations, kind, seed):
    """Aggregate raw walk events into the viewer's walk-record format.

    Each walk record is [from_lon, from_lat, to_lon, to_lat, t0, t1, riders],
    where (from→to) is street→station for kind='origin' and station→street
    for kind='dest'. Aggregates by (line, naptan, 1-min bin) and uses the
    per-line OSM-track-snapped coord so walks land at the right physical
    platform group (Bakerloo entrance vs Northern entrance at Waterloo).
    """
    agg = defaultdict(float)
    for line, nid, t, riders in walk_events:
        t_bin = (int(t) // WALK_BIN_S) * WALK_BIN_S
        agg[(line, nid, t_bin)] += riders

    rng = random.Random(seed)
    out = []
    skipped_no_coord = 0
    for (line, nid, t_bin), riders in agg.items():
        if riders < WALK_MIN_RIDERS:
            continue
        # Coord priority: per-line OSM snap > per-naptan TfL > mnlc-aggregated
        c = per_line_station_coords.get(line, {}).get(nid) or naptan_coords.get(nid)
        if not c:
            mnlc = naptan_to_mnlc.get(nid)
            s = stations.get(str(mnlc)) if mnlc is not None else None
            if not s:
                skipped_no_coord += 1
                continue
            c = (s["lat"], s["lon"])
        st_lat, st_lon = c[0], c[1]
        # Random anchor 200-400m from station (degrees: ~0.002-0.004 lat).
        angle = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(0.002, 0.004)
        d_lat = dist * math.cos(angle)
        d_lon = dist * math.sin(angle) / math.cos(math.radians(st_lat))
        anchor_lat = st_lat + d_lat
        anchor_lon = st_lon + d_lon
        if kind == "origin":
            t1 = t_bin
            t0 = t1 - ORIGIN_WALK_S
            out.append([round(anchor_lon, 5), round(anchor_lat, 5),
                        round(st_lon, 5), round(st_lat, 5),
                        t0, t1, round(riders)])
        else:  # dest
            t0 = t_bin
            t1 = t0 + DEST_WALK_S
            out.append([round(st_lon, 5), round(st_lat, 5),
                        round(anchor_lon, 5), round(anchor_lat, 5),
                        t0, t1, round(riders)])
    out.sort(key=lambda w: w[4])
    print(f"  {kind} walks: {len(out)} records (skipped {skipped_no_coord} no-coord)")
    return out


WAIT_TIME_BUCKET_S = 30


def merge_count_timelines(timelines):
    """Sum N cumulative-count timelines into one. Each input is
    `[[t, count], ...]` already filtered to count-change events; we convert
    to deltas, interleave, and re-emit count-change-only entries."""
    deltas = []
    for tl in timelines:
        prev = 0
        for t, c in tl:
            deltas.append((t, c - prev))
            prev = c
    deltas.sort(key=lambda x: x[0])
    out = []
    cum = 0
    last_kept = -1
    for i, (t, d) in enumerate(deltas):
        cum += d
        # Skip until the last delta at this timestamp, so we only emit one
        # entry per t and it reflects the full settled count
        if i + 1 < len(deltas) and deltas[i + 1][0] == t:
            continue
        rc = max(0, cum)
        if rc != last_kept:
            out.append([t, rc])
            last_kept = rc
    return out


def build_waiting_timelines(waiting_events):
    """Aggregate (line, naptan, t_arrive, t_board, riders) events into
    per-(line, naptan) [[t, count], ...] timelines.

    One bubble per physical platform group: at Waterloo each line has its own
    naptan-snapped coord (Bakerloo, Northern, Jubilee, W&C); at Temple the
    Circle and District platforms share a coord and the bubbles overlap
    visually as one. Arrival times are bucketed to WAIT_TIME_BUCKET_S to merge
    near-simultaneous fan-out sub-arrivals; only integer-count changes get an
    entry, keeping JSON small.

    Boarding times are NOT bucketed — t_brd is the train's exact (jittered)
    stop time, identical to the secs value the train's own timeline records.
    Keeping them exact guarantees the bubble's drop fires at the same instant
    the train's rider count picks up, so dwell-phase animations stay in sync."""
    deltas_by_key = defaultdict(lambda: defaultdict(float))
    for line, nid_o, t_arr, t_brd, riders in waiting_events:
        b_arr = int(t_arr / WAIT_TIME_BUCKET_S) * WAIT_TIME_BUCKET_S
        deltas_by_key[(line, nid_o)][b_arr] += riders
        deltas_by_key[(line, nid_o)][int(t_brd)] -= riders
    out = {}
    total_entries = 0
    for key, bucketed in deltas_by_key.items():
        keys = sorted(bucketed)
        tl = []
        cum = 0.0
        last_kept = 0
        for t in keys:
            cum += bucketed[t]
            rc = max(0, round(cum))
            if rc != last_kept:
                tl.append([t, rc])
                last_kept = rc
        if tl:
            out[key] = tl
            total_entries += len(tl)
    print(f"  waiting timelines: {len(out)} (line, platform) bubbles, "
          f"{total_entries} timeline entries "
          f"(avg {total_entries // max(1, len(out))}/bubble)")
    return out


# ---------------------------------------------------------------------------
# Static-mode stats output (stats.json) — daily aggregates for the
# "static views" toggle in the front-end. Mirrors nycriders/build.py's
# schema so the viewer's static-mode code ports cleanly.
# ---------------------------------------------------------------------------


def build_segment_stats(trips, boardings, alightings):
    """Per-segment ridership and crowdedness, bucketed by the hour the train
    passes through the upstream stop.

    Direction is synthesized per-physical-link: a→b in trip order gets
    direction=0 if naptan_a < naptan_b else 1, so opposite-direction trips on
    the same link always end up with opposite direction values. Front-end's
    offset rendering groups by sorted(from, to) anyway.

    Returns: dict[(line, direction, naptan_a, naptan_b)] -> list of 24
        [sum_riders, n_trains] entries (one per hour-of-day).
    """
    print("Building segment stats...")
    scheduled = defaultdict(lambda: defaultdict(int))
    loads = defaultdict(lambda: defaultdict(float))
    for ti, t in enumerate(trips):
        line = t["line"]
        stops = t.get("stops") or []
        if len(stops) < 2:
            continue
        bs = boardings[ti]
        als = alightings[ti]
        riders_on = 0.0
        for i in range(len(stops) - 1):
            riders_on += bs.get(i, 0.0)
            riders_on -= als.get(i, 0.0)
            if riders_on < 0:
                riders_on = 0
            a, sec_a = stops[i]
            b, _ = stops[i + 1]
            direction = 0 if a < b else 1
            key = (line, direction, a, b)
            hour = (int(sec_a) // 3600) % 24
            scheduled[key][hour] += 1
            loads[key][hour] += riders_on

    seg = {}
    for key, hour_counts in scheduled.items():
        load_for_key = loads.get(key, {})
        by_hour = []
        for h in range(24):
            n = hour_counts.get(h, 0)
            r = load_for_key.get(h, 0.0)
            by_hour.append([round(r, 1), n])
        seg[key] = by_hour
    print(f"  {len(seg)} segments")
    return seg


def build_segment_polylines(seg_stats, geo):
    """Return [(lon, lat), ...] polylines per segment. Resolution order:
      1. link_shapes[line:a:b] (OSM track, forward)
      2. link_shapes[line:b:a] reversed
      3. BFS chain through line adjacency, concatenating per-edge shapes —
         catches express skip-stop services (Met fast, Pic express, EZL fast)
         where no direct shape exists between the express's adjacent stops.
      4. Straight line between station coords.
    """
    print("Building segment polylines...")
    link_shapes = geo.get("link_shapes") or {}
    per_line_station_coords = geo.get("per_line_station_coords") or {}
    naptan_coords = geo.get("naptan_coords") or {}
    naptan_to_mnlc = geo["naptan_to_mnlc"]
    stations = geo["stations"]

    # per-line adjacency for chain resolution on skip-stop segments
    branches_by_line = defaultdict(list)
    for br in geo.get("branches", []):
        branches_by_line[br["line"]].append(br)
    line_adj = {line: build_line_adjacency(brs)
                for line, brs in branches_by_line.items()}
    # Mutable copies of link_shapes so we can splice in phantom edges below
    # without leaking changes into the caller's geometry dict.
    link_shapes = dict(link_shapes)
    chain_cache = {}

    def fallback_coord(line, nid):
        c = per_line_station_coords.get(line, {}).get(nid) or naptan_coords.get(nid)
        if c:
            return (c[1], c[0])  # (lon, lat)
        mnlc = naptan_to_mnlc.get(nid)
        s = stations.get(str(mnlc)) if mnlc is not None else None
        if s:
            return (s["lon"], s["lat"])
        return None

    # Some trip-data naptans never appear in `branches` (e.g. Paddington H&C
    # 940GZZLUPAH vs Circle's 940GZZLUPAC at the same physical platform group,
    # or extra-station codes like Burnham/Langley patched in for NUMBAT OD
    # lookups). BFS can't route through them without an adjacency entry, so
    # they fall back to straight lines. Splice in a phantom edge to whichever
    # in-graph naptan on the same line is closest (and likely co-located) so
    # the BFS hops onto the line's main graph and follows the OSM track from
    # there. Phantom edges only land in this function's local copies of
    # link_shapes / line_adj.
    for key in seg_stats.keys():
        line, _, a, b = key
        adj = line_adj.setdefault(line, {})
        for endpoint in (a, b):
            if endpoint in adj:
                continue
            ep_xy = fallback_coord(line, endpoint)
            if not ep_xy:
                continue
            best = None
            best_d2 = float("inf")
            for nid in adj:
                nid_xy = fallback_coord(line, nid)
                if not nid_xy:
                    continue
                dx = ep_xy[0] - nid_xy[0]
                dy = ep_xy[1] - nid_xy[1]
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best = nid
            # ~0.003° ≈ 330m at London's latitude — generous enough for
            # split-platform complexes, tight enough to refuse arbitrary
            # naptans from across a branch.
            if best is None or best_d2 > 0.003 * 0.003:
                continue
            adj.setdefault(endpoint, set()).add(best)
            adj.setdefault(best, set()).add(endpoint)
            best_xy = fallback_coord(line, best)
            phantom = [[round(ep_xy[0], 5), round(ep_xy[1], 5)],
                       [round(best_xy[0], 5), round(best_xy[1], 5)]]
            link_shapes.setdefault(f"{line}:{endpoint}:{best}", phantom)
            link_shapes.setdefault(f"{line}:{best}:{endpoint}",
                                    list(reversed(phantom)))

    # Second pass: ghosts whose nearest sibling is too far away (Shepherd's
    # Bush on Central is ~1km from White City — beyond the close-aliasing
    # threshold above) usually sit BETWEEN their topological neighbors along
    # an existing OSM track shape. Find that shape, split it at the ghost's
    # coord, and register two new sub-shapes. Then BFS resolves both adjacent
    # segments via the synthesized geometry. Also adds phantom adjacency so
    # downstream multi-hop chains can use the same ghost.
    ghost_neighbors_per_line = defaultdict(lambda: defaultdict(set))
    for key in seg_stats.keys():
        line, _, a, b = key
        adj = line_adj.get(line, {})
        if a not in adj:
            ghost_neighbors_per_line[line][a].add(b)
        if b not in adj:
            ghost_neighbors_per_line[line][b].add(a)

    n_splits = 0
    for line, ghosts in ghost_neighbors_per_line.items():
        adj = line_adj.setdefault(line, {})
        for ghost, neighbors in ghosts.items():
            ghost_xy = fallback_coord(line, ghost)
            if not ghost_xy:
                continue
            neighbors_list = sorted(neighbors)
            for i in range(len(neighbors_list)):
                for j in range(i + 1, len(neighbors_list)):
                    n1, n2 = neighbors_list[i], neighbors_list[j]
                    shape = link_shapes.get(f"{line}:{n1}:{n2}")
                    if not shape:
                        rev = link_shapes.get(f"{line}:{n2}:{n1}")
                        if rev:
                            shape = list(reversed(rev))
                    if not shape or len(shape) < 3:
                        continue
                    best_idx = 0
                    best_d2 = float("inf")
                    for k, p in enumerate(shape):
                        dx = p[0] - ghost_xy[0]
                        dy = p[1] - ghost_xy[1]
                        d2 = dx * dx + dy * dy
                        if d2 < best_d2:
                            best_d2 = d2
                            best_idx = k
                    # closest waypoint must be interior (so ghost actually
                    # falls between the two neighbors along the shape) AND
                    # close enough to the track that it makes sense to split
                    # (~550m); both guards reject the MDNHEAD↔TAPLOW shape
                    # being used to "place" Burnham, which is past TAPLOW.
                    if best_idx <= 0 or best_idx >= len(shape) - 1:
                        continue
                    if best_d2 > 0.005 * 0.005:
                        continue
                    first = shape[:best_idx + 1]
                    second = shape[best_idx:]
                    if len(first) < 2 or len(second) < 2:
                        continue
                    link_shapes.setdefault(f"{line}:{n1}:{ghost}", first)
                    link_shapes.setdefault(f"{line}:{ghost}:{n2}", second)
                    link_shapes.setdefault(f"{line}:{ghost}:{n1}",
                                            list(reversed(first)))
                    link_shapes.setdefault(f"{line}:{n2}:{ghost}",
                                            list(reversed(second)))
                    adj.setdefault(n1, set()).add(ghost)
                    adj.setdefault(ghost, set()).add(n1)
                    adj.setdefault(n2, set()).add(ghost)
                    adj.setdefault(ghost, set()).add(n2)
                    n_splits += 1
                    break   # only one split per (n1, n2) pair
                else:
                    continue
                break       # split found; move to next ghost

    poly = {}
    chained = 0
    fallback = 0
    skipped = 0
    for key in seg_stats.keys():
        line, _, a, b = key
        shape = link_shapes.get(f"{line}:{a}:{b}")
        if shape and len(shape) >= 2:
            poly[key] = [[round(c[0], 5), round(c[1], 5)] for c in shape]
            continue
        shape_r = link_shapes.get(f"{line}:{b}:{a}")
        if shape_r and len(shape_r) >= 2:
            poly[key] = [[round(c[0], 5), round(c[1], 5)] for c in reversed(shape_r)]
            continue
        chain = resolve_link_shape(line, a, b, link_shapes, line_adj, chain_cache)
        if not chain:
            chain_r = resolve_link_shape(line, b, a, link_shapes, line_adj, chain_cache)
            if chain_r:
                chain = list(reversed(chain_r))
        if chain and len(chain) >= 2:
            poly[key] = [[round(c[0], 5), round(c[1], 5)] for c in chain]
            chained += 1
            continue
        ca = fallback_coord(line, a)
        cb = fallback_coord(line, b)
        if ca and cb:
            poly[key] = [[round(ca[0], 5), round(ca[1], 5)],
                         [round(cb[0], 5), round(cb[1], 5)]]
            fallback += 1
        else:
            skipped += 1
    print(f"  {len(poly)} polylines ({chained} skip-stop chains, "
          f"{fallback} straight-line fallback, "
          f"{skipped} segments dropped for missing coords)")
    return poly


def build_station_stats(trips, boardings, alightings, geo):
    """Per-station boardings & alightings keyed by (mnlc, line, hour).

    4-column schema kept for forward-compat with nycriders' viewer:
        [origin_entry, transfer_entry, dest_exit, transfer_exit]
    London is direct-route only (no RAPTOR transfers in this pipeline) so
    every boarding is an origin_entry and every alighting is a dest_exit;
    transfer columns stay 0. Front-end collapses the metric toggle to
    boardings/alightings/all.
    """
    print("Building station stats...")
    naptan_to_mnlc = geo["naptan_to_mnlc"]
    agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for ti, t in enumerate(trips):
        line = t["line"]
        stops = t.get("stops") or []
        bs = boardings[ti]
        als = alightings[ti]
        for pos, (nid, secs) in enumerate(stops):
            mnlc = naptan_to_mnlc.get(nid)
            if mnlc is None:
                continue
            hour = (int(secs) // 3600) % 24
            b = bs.get(pos, 0.0)
            a = als.get(pos, 0.0)
            if b > 0:
                agg[(int(mnlc), line, hour)][0] += b
            if a > 0:
                agg[(int(mnlc), line, hour)][2] += a
    print(f"  {len(agg)} (complex, line, hour) station tuples")
    return agg


def build_stops_meta(seg_stats, geo):
    """{naptan: {name, complex_id, lat, lon}} for every naptan referenced by
    a segment endpoint. Used by the front-end to label segment endpoints in
    drill-downs."""
    naptan_to_mnlc = geo["naptan_to_mnlc"]
    stations = geo["stations"]
    naptan_coords = geo.get("naptan_coords") or {}
    seen = set()
    for (line, direction, a, b) in seg_stats.keys():
        seen.add(a)
        seen.add(b)
    out = {}
    for nid in seen:
        mnlc = naptan_to_mnlc.get(nid)
        if mnlc is None:
            continue
        s = stations.get(str(mnlc))
        if not s:
            continue
        c = naptan_coords.get(nid)
        if c:
            lat, lon = c[0], c[1]
        else:
            lat, lon = s["lat"], s["lon"]
        out[nid] = {
            "name": s["name"],
            "complex_id": int(mnlc),
            "lat": round(lat, 5),
            "lon": round(lon, 5),
        }
    return out


def build_stats_output(seg_stats, seg_polylines, station_stats, stops_meta, geo):
    """Combine segment + station stats into stats.json structure (matches
    nycriders schema exactly)."""
    print("Building stats output...")
    stations = geo["stations"]

    segments_out = []
    for key, by_hour in seg_stats.items():
        coords = seg_polylines.get(key)
        if not coords or len(coords) < 2:
            continue
        if sum(r for r, _ in by_hour) < 1:
            continue
        line, direction, a, b = key
        segments_out.append({
            "route": line,
            "direction": int(direction),
            "from": a,
            "to": b,
            "coords": coords,
            "by_hour": [[round(r, 1), n] for r, n in by_hour],
            "parents": [[a, b]],
        })

    nested = defaultdict(lambda: defaultdict(lambda: [[0.0] * 4 for _ in range(24)]))
    for (mnlc, line, hour), vals in station_stats.items():
        slot = nested[mnlc][line][hour]
        for i in range(4):
            slot[i] += vals[i]

    stations_out = []
    for mnlc, by_route in nested.items():
        s = stations.get(str(mnlc))
        if not s:
            continue
        by_route_hour = {}
        for line, arr in by_route.items():
            if any(any(v > 0.5 for v in row) for row in arr):
                by_route_hour[line] = [[round(v, 1) for v in row] for row in arr]
        if not by_route_hour:
            continue
        stations_out.append({
            "complex_id": int(mnlc),
            "name": s["name"],
            "lat": round(s["lat"], 5),
            "lon": round(s["lon"], 5),
            "by_route_hour": by_route_hour,
        })

    print(f"  {len(segments_out)} segments, {len(stations_out)} stations, "
          f"{len(stops_meta)} stop endpoints")
    return {"segments": segments_out, "stations": stations_out, "stops": stops_meta}


def build_output(trips, boardings, alightings, waiting_events,
                  origin_walk_events, dest_walk_events, geo):
    print("Building trains.json...")
    naptan_to_mnlc = geo["naptan_to_mnlc"]
    stations = geo["stations"]
    line_colors = geo["line_colors"]
    link_shapes = geo.get("link_shapes") or {}
    # Per-naptan TfL coords — preserves platform-level placement at
    # interchanges where mnlc-averaging would hide the spread.
    naptan_coords = geo.get("naptan_coords") or {}
    # Per-(line, naptan) snap to the line's OSM track. Splits platforms
    # at single-naptan complexes (e.g. Embankment) where TfL gives all
    # lines the same coord but tracks are physically distinct.
    per_line_station_coords = geo.get("per_line_station_coords") or {}

    branches_by_line = defaultdict(list)
    for br in geo.get("branches", []):
        branches_by_line[br["line"]].append(br)
    line_adj = {line: build_line_adjacency(brs)
                for line, brs in branches_by_line.items()}
    chain_cache = {}

    trains_out = []
    total_waypoints = 0
    chain_hits = 0
    for ti, t in enumerate(trips):
        bs = boardings[ti]
        if not bs:
            continue
        als = alightings[ti]
        riders_on = 0.0
        timeline = []
        naptans_at = []  # naptan id parallel to each timeline station entry
        for pos, (nid, secs) in enumerate(t["stops"]):
            mnlc = naptan_to_mnlc.get(nid)
            if mnlc is None:
                continue
            s = stations.get(str(mnlc))
            if not s:
                continue
            # Prefer per-line OSM-track snap (splits platforms at single-naptan
            # interchanges), then per-naptan TfL coord, then mnlc average.
            plc = per_line_station_coords.get(t["line"], {}).get(nid)
            coord = naptan_coords.get(nid)
            if plc:
                lat, lon = plc[0], plc[1]
            elif coord:
                lat, lon = coord[0], coord[1]
            else:
                lat, lon = s["lat"], s["lon"]
            b = bs.get(pos, 0.0)
            a = als.get(pos, 0.0)
            riders_on += b - a
            riders_on = max(0.0, riders_on)
            timeline.append([
                int(secs),
                round(lat, 5),
                round(lon, 5),
                round(riders_on),
                round(b),
            ])
            naptans_at.append(nid)
        if not timeline or max(r[3] for r in timeline) < 1:
            continue
        before = len(timeline)
        timeline = expand_with_link_shapes(timeline, naptans_at, t["line"],
                                            link_shapes, line_adj, chain_cache)
        total_waypoints += len(timeline) - before
        trains_out.append({
            "route": t["line"],
            "color": line_colors.get(t["line"], "#808183"),
            "timeline": timeline,
        })
    cache_hits = sum(1 for v in chain_cache.values() if v)
    cache_misses = sum(1 for v in chain_cache.values() if not v)
    print(f"  {len(trains_out)} trains with riders"
          f" ({total_waypoints} shape waypoints inserted; "
          f"chained shapes: {cache_hits} unique, {cache_misses} unchainable)")

    # Per-(line, naptan) waiting timelines — one bubble per platform group,
    # placed at the line's OSM-track snap. See build_waiting_timelines.
    waiting_timelines = build_waiting_timelines(waiting_events)
    # Group by rounded coord so lines that share a platform (Circle+District at
    # Temple, Circle+H&C+Met at King's Cross) merge into one bubble with summed
    # waiting count and a combined route list — instead of stacking visually.
    by_coord = defaultdict(list)
    for (line, nid), tl in waiting_timelines.items():
        c = per_line_station_coords.get(line, {}).get(nid) or naptan_coords.get(nid)
        if not c:
            mnlc = naptan_to_mnlc.get(nid)
            s = stations.get(str(mnlc)) if mnlc is not None else None
            if not s:
                continue
            lat, lon = s["lat"], s["lon"]
            name = s["name"]
        else:
            lat, lon = c[0], c[1]
            mnlc = naptan_to_mnlc.get(nid)
            name = stations.get(str(mnlc), {}).get("name", "") if mnlc is not None else ""
        by_coord[(round(lat, 5), round(lon, 5))].append((line, name, tl))

    waiting_out = []
    for (lat, lon), entries in sorted(by_coord.items()):
        routes = sorted({line for line, _, _ in entries})
        names = [n for _, n, _ in entries if n]
        name = names[0] if names else ""
        if len(entries) == 1:
            tl = entries[0][2]
        else:
            tl = merge_count_timelines([e[2] for e in entries])
        waiting_out.append({
            "name": name,
            "lat": lat,
            "lon": lon,
            "routes": routes,
            "timeline": tl,
        })

    origin_walks = build_walks(origin_walk_events, per_line_station_coords,
                                 naptan_coords, naptan_to_mnlc, stations,
                                 "origin", seed=99)
    dest_walks = build_walks(dest_walk_events, per_line_station_coords,
                              naptan_coords, naptan_to_mnlc, stations,
                              "dest", seed=199)

    return {
        "lines": geo["lines"],
        "link_shapes": geo.get("link_shapes") or {},
        "trains": trains_out,
        "waiting": waiting_out,
        "transfer_walks": [],
        "origin_walks": origin_walks,
        "dest_walks": dest_walks,
        "line_colors": line_colors,
    }


def dedupe_prefix_trips(trips):
    """Drop trips whose (naptan, secs) stop sequence is a contiguous slice of
    another trip's — i.e. matches as a prefix, suffix, or interior substring.

    The railwaydata DLR PDFs split into 4 books, and DLR4 ("Bank & Tower Gateway
    - Westferry & Poplar") is a *summary* of trains already listed in their full
    form by DLR1/2/3, in BOTH directions: the eastbound DLR4 entries are
    prefixes of the longer eastbound Beckton/Lewisham/Woolwich runs; the
    westbound entries are suffixes of the same trips going back. Either way the
    same physical train gets emitted twice and shadows itself in the viewer.

    Two real distinct trains on a single-track segment can't share several
    consecutive (naptan, exact-minute) tuples, so a contiguous match is
    overwhelmingly a listing duplicate rather than a coincidence. We always
    keep the longer trip (it carries the same riders plus more route).

    Bucket by every (naptan, secs) tuple in each trip so we only compare pairs
    that share at least one station-time, keeping this near-linear in practice.
    """
    print("Deduping overlapping trips...")

    # (line, naptan, secs) -> set of trip indices passing through that point.
    # Two trips can only overlap as a substring if they share a tuple, so this
    # gives us a tight candidate set without an N^2 scan.
    by_tuple = defaultdict(set)
    for ti, t in enumerate(trips):
        line = t["line"]
        for nid, secs in t.get("stops", ()):
            by_tuple[(line, nid, secs)].add(ti)

    def is_contiguous_substring(short_stops, long_stops):
        n, m = len(short_stops), len(long_stops)
        if n >= m:
            return False
        first = short_stops[0]
        for start in range(m - n + 1):
            if long_stops[start] != first:
                continue
            if all(long_stops[start + i] == short_stops[i] for i in range(1, n)):
                return True
        return False

    drop = set()
    # Process shortest trips first — once a short trip is matched into a longer
    # one we never need to consider it as the "long" side of any future pair.
    order = sorted(range(len(trips)), key=lambda i: len(trips[i].get("stops", ())))
    for ti in order:
        if ti in drop:
            continue
        stops_i = trips[ti].get("stops") or []
        if len(stops_i) < 2:
            continue
        line = trips[ti]["line"]
        first_nid, first_secs = stops_i[0]
        # Candidate longer trips: those passing through stops_i[0] on this line.
        candidates = by_tuple.get((line, first_nid, first_secs), set()) - {ti} - drop
        for tj in candidates:
            if len(trips[tj].get("stops", ())) <= len(stops_i):
                continue
            if is_contiguous_substring(stops_i, trips[tj]["stops"]):
                drop.add(ti)
                break

    if drop:
        per_line = defaultdict(int)
        for ti in drop:
            per_line[trips[ti]["line"]] += 1
        print(f"  dropped {len(drop)} overlap-duplicate trips: "
              f"{dict(sorted(per_line.items()))}")
    else:
        print("  no overlap duplicates")
    return [t for ti, t in enumerate(trips) if ti not in drop]


def dedupe_same_endpoint_trips(trips):
    """Drop trips with identical naptan sequences and matching origin AND
    destination times, even when intermediate dwell times differ by a minute
    or two.

    The Northern PDF lists ~500 trains twice in adjacent columns where every
    station matches except one intermediate stop, off by 1-2 minutes — a data
    artifact (same train, two listings). Two physically distinct trains can't
    share both endpoints at the exact minute on the same route: that'd mean
    they're occupying the same platform at the same minute on both ends.

    Stricter than dedupe_prefix_trips, which only catches contiguous-substring
    overlap (DLR4 summary tables); this catches equal-length near-twins."""
    print("Deduping same-endpoint trips...")
    by_sig = defaultdict(list)
    for ti, t in enumerate(trips):
        stops = t.get("stops") or []
        if len(stops) < 2:
            continue
        sig = (t["line"],
               tuple(nid for nid, _ in stops),
               stops[0][1],
               stops[-1][1])
        by_sig[sig].append(ti)

    drop = set()
    for idxs in by_sig.values():
        if len(idxs) > 1:
            drop.update(idxs[1:])

    if drop:
        per_line = defaultdict(int)
        for ti in drop:
            per_line[trips[ti]["line"]] += 1
        print(f"  dropped {len(drop)} same-endpoint duplicate trips: "
              f"{dict(sorted(per_line.items()))}")
    else:
        print("  no same-endpoint duplicates")
    return [t for ti, t in enumerate(trips) if ti not in drop]


def jitter_trip_times(trips, jitter_s=JITTER_S):
    """Add ±jitter_s random offsets to each stop time, then enforce monotonicity.

    Run before speed smoothing — any unrealistic gaps the jitter introduces
    are absorbed by the smoothing passes. Trips parsed from minute-rounded
    timetables otherwise have every stop at a multiple of 60s, which makes
    co-scheduled trains stack visually on the map.
    """
    print(f"Jittering stop times by +/-{jitter_s}s...")
    rng = random.Random(7)
    n_trips = 0
    for trip in trips:
        stops = trip["stops"]
        if not stops:
            continue
        new_stops = []
        prev = -10 ** 9
        for nid, t in stops:
            jt = t + rng.randint(-jitter_s, jitter_s)
            if jt <= prev:
                jt = prev + 1
            new_stops.append((nid, jt))
            prev = jt
        trip["stops"] = new_stops
        n_trips += 1
    print(f"  jittered {n_trips} trips")


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def cruise_speed_kmh(dist_m, dt_s):
    cruise_dt = dt_s - SMOOTH_OVERHEAD_S
    if cruise_dt <= 0:
        return 9999.0
    return (dist_m / cruise_dt) * 3.6


def _time_needed_for_speed(dist_m, target_kmh):
    return dist_m / (target_kmh / 3.6) + SMOOTH_OVERHEAD_S


def _smooth_pass(trips, coords, threshold_factor=1.0, take_fraction=1.0,
                 donor_factor=1.0, verbose_examples=None):
    """One smoothing pass. Returns counts dict.

    For each segment whose speed exceeds `threshold_factor * line_threshold`,
    steal time from the slower neighbors so its speed drops to that target.
    Donors must remain at speed ≤ `donor_factor * line_threshold` afterward.
    Only `take_fraction` of the deficit is taken (1.0 = full fix, 0.5 = nudge).

      hard pass: threshold=1.0, take=1.0, donor=1.0 (donor stays ≤ hard)
      soft pass: threshold=0.85, take=0.5, donor=1.0 (target speed soft, donor stays ≤ hard)

    Stops are shifted by moving the boundary stop's timestamp, which transfers
    seconds between inter-stop gaps without changing trip duration.
    """
    n_fast = n_fixed = n_partial = n_no_donor = 0
    for trip in trips:
        line = trip["line"]
        hard = SPEED_THRESHOLDS.get(line, DEFAULT_SPEED_THRESHOLD)
        seg_target = hard * threshold_factor
        donor_cap = hard * donor_factor
        stops = trip["stops"]
        n = len(stops)
        if n < 2:
            continue

        dists = [None] * (n - 1)
        for j in range(n - 1):
            ca = coords.get(stops[j][0])
            cb = coords.get(stops[j + 1][0])
            if ca and cb:
                dists[j] = haversine_m(ca[0], ca[1], cb[0], cb[1])

        times = [s[1] for s in stops]
        for j in range(n - 1):
            d = dists[j]
            if d is None:
                continue
            dt = times[j + 1] - times[j]
            if dt <= 0:
                continue
            speed = cruise_speed_kmh(d, dt)
            if speed <= seg_target:
                continue
            n_fast += 1
            full_extra = math.ceil(_time_needed_for_speed(d, seg_target)) - dt
            want = max(1, math.ceil(full_extra * take_fraction))

            before_avail = 0
            if j > 0 and dists[j - 1] is not None:
                prev_dt = times[j] - times[j - 1]
                prev_min = math.ceil(_time_needed_for_speed(dists[j - 1], donor_cap))
                before_avail = max(0, prev_dt - prev_min)

            after_avail = 0
            if j + 2 < n and dists[j + 1] is not None:
                next_dt = times[j + 2] - times[j + 1]
                next_min = math.ceil(_time_needed_for_speed(dists[j + 1], donor_cap))
                after_avail = max(0, next_dt - next_min)

            total_avail = before_avail + after_avail
            if total_avail == 0:
                n_no_donor += 1
                continue

            take = min(want, total_avail)
            take_before = round(take * before_avail / total_avail)
            take_after = take - take_before
            if take_before:
                times[j] -= take_before
            if take_after:
                times[j + 1] += take_after

            if take >= want:
                n_fixed += 1
            else:
                n_partial += 1

            if verbose_examples is not None and len(verbose_examples) < 30:
                new_speed = cruise_speed_kmh(d, times[j + 1] - times[j])
                verbose_examples.append((line, stops[j][0], stops[j + 1][0],
                                         speed, new_speed, take_before, take_after))

        trip["stops"] = [(stops[j][0], times[j]) for j in range(n)]

    return {"fast": n_fast, "fixed": n_fixed,
            "partial": n_partial, "no_donor": n_no_donor}


def smooth_trip_speeds(trips, coords, verbose=False):
    """Hard passes followed by soft 'rescue' passes for chain-blocked violators.

    Hard passes: target speed > hard, donors must stay ≤ hard, full fix.
    Soft passes: target speed > hard (same), donors can be pushed up to
      hard/SOFT_FACTOR ≈ 1.18·hard, take half the deficit. This unblocks
      chains: edge violators steal from previously-saturated neighbors who
      are now allowed to drift slightly past hard, then the next pass lets
      the now-slower edges become donors for inner violators, etc.

    Returns {'passes': [...], 'examples': [...]}.
    """
    print(f"Smoothing trip speeds ({SMOOTH_PASSES} hard + up to "
          f"{SOFT_RESCUE_PASSES} soft passes)...")
    examples = [] if verbose else None
    pass_stats = []
    for i in range(SMOOTH_PASSES):
        s = _smooth_pass(trips, coords, verbose_examples=examples)
        pass_stats.append(s)
        print(f"  hard pass {i+1}: fast={s['fast']:>5} fixed={s['fixed']:>5} "
              f"partial={s['partial']:>5} no-donor={s['no_donor']:>5}")
        if s["fast"] == 0:
            break
    for i in range(SOFT_RESCUE_PASSES):
        s = _smooth_pass(trips, coords,
                         donor_factor=SOFT_DONOR_FACTOR,
                         take_fraction=SOFT_TAKE_FRACTION,
                         verbose_examples=examples)
        pass_stats.append(s)
        # Each rescue pass can surface new mild violators (donors pushed past
        # hard), so the count may rise before falling — don't bail on count.
        # Run all SOFT_RESCUE_PASSES; the half-take fraction shrinks the new
        # excess each round so the cascade converges geometrically.
        print(f"  soft pass {i+1}: fast={s['fast']:>5} fixed={s['fixed']:>5} "
              f"partial={s['partial']:>5} no-donor={s['no_donor']:>5}")
        if s["fast"] == 0:
            break
    return {"passes": pass_stats, "examples": examples or []}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=1.0)
    args = parser.parse_args()
    random.seed(123)

    t0 = time_module.time()
    geo = load_geometry()
    # Trip source priority — sources are complementary, not exclusive:
    #   - data/timetables/<CODE> *.pdf      -> railwaydata.co.uk Tube + DLR PDFs
    #   - data/{ezl,og}.json.gz             -> Network Rail SCHEDULE feed
    #     (canonical source for Elizabeth + 6 Overground lines)
    #   - data/timetables/elizabeth-* /
    #     data/timetables/lo-* PDFs         -> TfL PDF fallback (for any line
    #     CIF didn't supply, e.g. you only downloaded one TOC feed)
    #   - data/tfl-txc-demo.zip             -> 2010 TXC fallback
    #   - data/api_cache/tt_*               -> Unified API per-stop calls
    trips = []
    if os.path.isdir("data/timetables"):
        from railwaydata import parse_all_pdfs as _railwaydata
        trips.extend(_railwaydata())
    cif_lines_seen = set()
    if os.path.exists("data/ezl.json.gz") or os.path.exists("data/og.json.gz"):
        from cif import parse_all_cif
        cif_trips = parse_all_cif()
        cif_lines_seen = {t["line"] for t in cif_trips}
        trips.extend(cif_trips)
    if os.path.isdir("data/timetables"):
        from tfl_pdf import parse_all_tfl_pdfs
        pdf_trips = [t for t in parse_all_tfl_pdfs() if t["line"] not in cif_lines_seen]
        if pdf_trips:
            print(f"  PDF fallback: {len(pdf_trips)} trips for lines without CIF coverage")
            trips.extend(pdf_trips)
    if not trips and os.path.exists(TXC_ZIP):
        from txc import parse_txc_zip
        trips = parse_txc_zip(TXC_ZIP)
    if not trips:
        trips = flatten_trips(geo["numbat_line_to_tfl"])
    if not trips:
        print("No trips found; did you run fetch_schedules.py or place a TXC zip?")
        return
    patch_missing_stops(trips)
    trips = dedupe_prefix_trips(trips)
    trips = dedupe_same_endpoint_trips(trips)
    naptan_coords = {nid: (v[0], v[1]) for nid, v in (geo.get("naptan_coords") or {}).items()}
    jitter_trip_times(trips)
    smooth_trip_speeds(trips, naptan_coords)
    trip_index = build_trip_index(trips)
    boardings, alightings, waiting_events, origin_walk_events, dest_walk_events = \
        assign_riders(trips, trip_index, geo, sample_rate=args.sample)
    out = build_output(trips, boardings, alightings, waiting_events,
                        origin_walk_events, dest_walk_events, geo)
    print(f"Writing {OUT_JSON}...")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f)
    size_mb = os.path.getsize(OUT_JSON) / 1024 / 1024
    print(f"  {OUT_JSON} is {size_mb:.1f} MB")

    seg_stats = build_segment_stats(trips, boardings, alightings)
    seg_polylines = build_segment_polylines(seg_stats, geo)
    station_stats = build_station_stats(trips, boardings, alightings, geo)
    stops_meta = build_stops_meta(seg_stats, geo)
    stats_out = build_stats_output(seg_stats, seg_polylines, station_stats,
                                     stops_meta, geo)
    print(f"Writing {STATS_JSON}...")
    with open(STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats_out, f)
    size_mb = os.path.getsize(STATS_JSON) / 1024 / 1024
    print(f"  {STATS_JSON} is {size_mb:.1f} MB")

    print(f"Done! ({time_module.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
