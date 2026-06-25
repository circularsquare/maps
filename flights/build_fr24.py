#!/usr/bin/env python3
"""
build_fr24.py — aggregate the FR24 pull into routes_fr24.json (the schema the
viewer's routes.json uses).

Reads the per-airport checkpoint files written by fr24_pull.py. Point PULL_DIRS
at one or more pull/<DAY>/ folders; set WINDOW_HOURS to the window each folder
covers (1 for the first-hour test, 24 for a full day).

The aircraft-type table is AIRLINERS ONLY: a flight whose type is not in it
(business jets, helicopters, light GA, sub-~20-seat commuters) is dropped — the
size filter (see CLAUDE.md).
"""
import json, os, csv, collections, math, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# pull/<DAY>/ folders to aggregate, and the window (hours) each one covers.
PULL_DIRS    = [os.path.join(HERE, "pull", "2026-05-15"),
                os.path.join(HERE, "pull", "2026-05-16")]
WINDOW_HOURS = 24.0         # 1 = first-hour validation pull; 24 = a full day

# Load factor. Real load factors aren't flat: thick, high-frequency routes run
# fuller and thin/remote routes emptier. assign_load_factors() spreads LF by
# route density (offered seats/day) on an S-curve, then rescales so the CAPACITY-
# WEIGHTED MEAN equals LOAD_FACTOR — i.e. it redistributes passengers across
# routes without changing the system total. The anchor is set to IATA's real
# global passenger load factor (~0.83).
LOAD_FACTOR  = 0.83
LF_MIN, LF_MAX, LF_K = 0.55, 0.90, 1.3   # S-curve floor / ceiling / steepness
LF_CLAMP = (0.50, 0.92)                   # hard bounds after rescaling

# observed-flight count -> per-year estimate. 8760 hours/year.
YEAR_FACTOR  = 8760.0 / (WINDOW_HOURS * len(PULL_DIRS))


def assign_load_factors(caps):
    """Per-route load factor from route density (annual offered seats -> seats/day
    on an S-curve), rescaled so the capacity-weighted mean is exactly LOAD_FACTOR.
    Returns a list parallel to `caps`."""
    if not caps:
        return []
    xs = [math.log10(max(c / 365.0, 1.0)) for c in caps]      # log10 seats/day
    x0 = statistics.median(xs)
    raw = [LF_MIN + (LF_MAX - LF_MIN) / (1.0 + math.exp(-LF_K * (x - x0))) for x in xs]
    tot = sum(caps)
    mean = sum(c * l for c, l in zip(caps, raw)) / tot if tot else LOAD_FACTOR
    scale = LOAD_FACTOR / mean if mean else 1.0
    lo, hi = LF_CLAMP
    return [min(hi, max(lo, l * scale)) for l in raw]

# Hand-curated routes for airports FR24's flight-summary API can't see
# (Kuwait OKBK, all of Uzbekistan, Kyrgyzstan — confirmed blank both directions
# across both pull days). These are SCHEDULE-derived estimates, not observed
# flights, and are tagged source="sched" in the output so the viewer / unexpected
# engine can treat them differently. See data/backfill_routes.csv.
BACKFILL_CSV = os.path.join(DATA, "backfill_routes.csv")

# ALL-CARGO airline ICAO codes (operating_as). These carry zero passengers, but
# fly the same airframes as pax jets (a 747-8F shares ICAO type B748 with a pax
# 748), so the SEATS table would otherwise count each freighter as a full jumbo
# of passengers and invent huge phantom corridors (Cargolux's LUX hub was the
# tell: LUX-ASB/CGO/HKG/LAX all showed ~410 "seats"). We can only drop these by
# operator — the flight-summary/light record has no cargo/pax flag. Vetted
# against the 2026-05 pull by tail-reg + route so no passenger carrier is caught
# (e.g. MAC=Air Arabia Maroc, SBI=S7, LGL=Luxair are pax and deliberately NOT
# here). MIXED pax+freight carriers (Emirates/Qatar/Korean fly freighters under
# their pax code) are left in — dropping them would delete real pax routes.
CARGO_OPERATORS = frozenset({
    # global integrators / majors
    "FDX", "UPS", "CLX", "GTI", "GEC", "CKS", "PAC", "ABX", "CJT", "WGN",
    "CLU", "ABW", "SQC", "GSS", "SOO", "NCA", "BOX", "AZG", "AIH", "ATN",
    "AJT", "CSB",
    # DHL network
    "BCS", "DHK", "DHX", "AHK", "DAE", "DHA",
    # China cargo
    "CAO", "CKK", "CYZ", "CSS",
    # other verified all-cargo
    "MPH", "TAY", "LCO", "TPA", "LTG", "MSX", "ABD", "RUN", "SNC", "ICL",
    "LAE", "BDA", "SOP", "CHZ", "CHG",   # Challenge Airlines group (767F/747F)
    "HGO", "LHA", "SWN", "NPT", "LSI",   # One Air, Longhao, West Atlantic SE/UK, MSC Air Cargo
    # NOT cargo despite freighter-heavy hubs — these are mixed and fly real pax,
    # so they stay IN: YZR (Suparna, pax 737s), MFX (My Freighter, now pax
    # A320/A330 — also fills the Uzbekistan hole), SWT (Swiftair, Balearic pax
    # hops), AAE (Maleth-type ACMI pax 777-300ER), ABR (ASL Ireland pax charter).
})

# Shared type codes that are freighter-dominant but ALSO have a passenger variant,
# flown by combination carriers under their passenger operator code — so they can't
# be dropped by operator without deleting real pax routes. We can't separate the
# two, so we down-weight: count each such flight at a fraction of its seats. B77L
# is the 777F freighter sharing its code with the rare passenger 777-200LR
# (Emirates/Ethiopian ultra-long-haul); 0.5 splits the difference. The flight
# still counts toward route frequency (flights/day) — only its pax estimate is
# discounted.
TYPE_PAX_WEIGHT = {"B77L": 0.5}

# ICAO aircraft typecode -> typical seats. AIRLINERS ONLY (~29+ seats). A flight
# whose type is absent here is dropped: that excludes business jets, helicopters
# and small planes that never carry "real" passenger volume.
SEATS = {
    "A318":110,"A319":140,"A320":165,"A321":200,"A19N":140,"A20N":165,"A21N":210,
    "A306":260,"A310":220,"A332":250,"A333":290,"A338":280,"A339":300,
    "A342":260,"A343":280,"A345":290,"A346":320,"A359":300,"A35K":350,"A388":525,
    "B712":110,"B722":130,"B732":120,"B733":140,"B734":150,"B735":110,"B736":110,
    "B737":140,"B738":189,"B739":178,"B37M":160,"B38M":178,"B39M":190,
    "B741":410,"B742":410,"B743":410,"B744":416,"B748":410,
    "B752":200,"B753":230,"B762":210,"B763":245,"B764":245,
    "B772":300,"B77L":300,"B773":350,"B77W":365,"B788":242,"B789":290,"B78X":330,
    "E170":76,"E75S":76,"E75L":82,"E190":100,"E195":120,"E290":110,"E295":130,
    "E135":37,"E145":50,"CRJ1":50,"CRJ2":50,"CRJ7":70,"CRJ9":90,
    "BCS1":110,"BCS3":140,
    "C919":160,"ARJ21":90,"AJ27":90,"C909":90,"SU95":98,"RJ85":100,"RJ1H":100,
    "F70":80,"F100":100,
    "AT43":48,"AT45":48,"AT72":70,"AT73":70,"AT75":70,"AT76":70,
    "DH8A":37,"DH8B":37,"DH8C":50,"DH8D":78,"SF34":34,"E120":30,"JS41":29,
    "AN24":48,"AN26":40,
}


def load_airports():
    """ICAO -> (display_code, lon, lat, city, country). Display falls back to ICAO."""
    out = {}
    with open(os.path.join(DATA, "airports.dat"), encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 8:
                continue
            icao, iata = row[5].strip(), row[4].strip()
            if not icao or icao == "\\N" or len(icao) != 4:
                continue
            try:
                lat, lon = float(row[6]), float(row[7])
            except ValueError:
                continue
            disp = iata if (iata and iata != "\\N" and len(iata) == 3) else icao
            out[icao] = (disp, round(lon, 4), round(lat, 4), row[2], row[3])
    return out


def read_flights():
    """All flight records from every per-airport file in the PULL_DIRS."""
    flights, files, skipped_airports = [], 0, 0
    for d in PULL_DIRS:
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json") or fn.startswith("_"):
                continue
            doc = json.load(open(os.path.join(d, fn), encoding="utf-8"))
            if not doc.get("ok"):           # skip suspect / incomplete airports
                skipped_airports += 1
                continue
            flights += doc.get("records", [])
            files += 1
    print(f"airport files read    : {files}  (skipped {skipped_airports} not-ok)")
    return flights


def load_backfill():
    """Hand-curated schedule routes -> list of (icao_a, icao_b, flights_per_week,
    aircraft_type). Rows are ICAO-keyed. Lines starting with # are comments.
    Returns [] if the file is absent."""
    if not os.path.exists(BACKFILL_CSV):
        return []
    out = []
    with open(BACKFILL_CSV, encoding="utf-8") as f:
        rows = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    for row in csv.DictReader(rows):
        try:
            wk = float(row["flights_per_week"])
        except (KeyError, ValueError):
            continue
        o, d = row["orig_icao"].strip(), row["dest_icao"].strip()
        if not o or not d or o == d or wk <= 0:
            continue
        seats = None
        if row.get("seats", "").strip():
            try:
                seats = int(float(row["seats"]))
            except ValueError:
                seats = None
        out.append((o, d, wk, row.get("aircraft", "").strip(), seats))
    return out


def merge_backfill(routes, used, airports, fr24_pairs):
    """Append schedule-derived routes for the FR24 holes. Skips any pair FR24
    already covers (observed always wins), any unknown airport, and any pair
    already added by an earlier backfill row (first wins)."""
    added = skipped_dup = skipped_apt = 0
    seen = set()
    for o, d, wk, ac, seats_override in load_backfill():
        if o not in airports or d not in airports:
            skipped_apt += 1; continue
        key = tuple(sorted((o, d)))
        if key in fr24_pairs or key in seen:
            skipped_dup += 1; continue
        seen.add(key)
        seats = seats_override or SEATS.get(ac, 180)   # explicit > type lookup > narrowbody default
        annual_flights = wk * (365.0 / 7.0)             # weekly -> yearly, same daily-rate x365 basis as FR24
        cap = annual_flights * seats                    # offered seats; LF applied globally below
        routes.append([airports[o][0], airports[d][0],
                       cap, round(wk / 7.0, 1), seats, "sched"])
        used.add(o); used.add(d)
        added += 1
    print(f"  backfill added        : {added}  (dup {skipped_dup}, unknown apt {skipped_apt})")
    return added


def main():
    airports = load_airports()
    flights = read_flights()
    print(f"flight records        : {len(flights)}")

    pairs = collections.defaultdict(lambda: {"n": 0, "seats": 0.0})
    kept = skip_cancel = skip_ga = skip_apt = skip_cargo = 0
    for r in flights:
        o, d, t = r.get("orig_icao"), r.get("dest_icao"), r.get("type")
        if not r.get("datetime_takeoff"):       # cancelled / never flew
            skip_cancel += 1; continue
        if r.get("operating_as") in CARGO_OPERATORS:   # freighter -> no pax
            skip_cargo += 1; continue
        if not o or not d or o == d:
            continue
        seats = SEATS.get(t)
        if seats is None:                       # not an airliner -> size filter
            skip_ga += 1; continue
        if o not in airports or d not in airports:
            skip_apt += 1; continue
        rec = pairs[tuple(sorted((o, d)))]
        rec["n"] += 1
        rec["seats"] += seats * TYPE_PAX_WEIGHT.get(t, 1.0)   # discount likely-freight shared types
        kept += 1

    print(f"  skipped cancelled     : {skip_cancel}")
    print(f"  skipped all-cargo op  : {skip_cargo}")
    print(f"  skipped non-airliner  : {skip_ga}")
    print(f"  skipped unknown apt   : {skip_apt}")
    print(f"  kept                  : {kept}")

    routes, used = [], set()
    for (a, b), rec in pairs.items():
        avg_seats = rec["seats"] / rec["n"]
        annual_flights = rec["n"] * YEAR_FACTOR
        cap = annual_flights * avg_seats                # offered seats; LF applied globally below
        routes.append([airports[a][0], airports[b][0],
                       cap, round(annual_flights / 365, 1), round(avg_seats)])
        used.add(a); used.add(b)

    n_backfill = merge_backfill(routes, used, airports, pairs)

    # density-based load factor: slot [2] currently holds offered capacity; turn
    # it into a passenger estimate — fuller on thick routes, emptier on thin ones,
    # with the capacity-weighted mean pinned to LOAD_FACTOR (total pax unchanged).
    for r, lf in zip(routes, assign_load_factors([r[2] for r in routes])):
        r[2] = round(r[2] * lf)

    routes.sort(key=lambda r: -r[2])
    out = {
        "meta": {
            "source": f"FR24 pull — {', '.join(os.path.basename(d) for d in PULL_DIRS)} "
                      f"({WINDOW_HOURS:g}h window)",
            "note": f"Observed flights annualised x{YEAR_FACTOR:.0f} "
                    f"({WINDOW_HOURS:g}h x {len(PULL_DIRS)} day(s) -> 8760h/yr). "
                    f"Day-of-week / seasonal mix is whatever was pulled. "
                    f"Load factor scales with route density (capacity-weighted "
                    f"mean {LOAD_FACTOR:.2f}). "
                    f"{n_backfill} schedule-derived routes (source=\"sched\") "
                    f"backfill FR24's coverage holes.",
            "n_routes": len(routes),
            "n_backfill": n_backfill,
            "n_airports": len(used),
        },
        "airports": {airports[k][0]: list(airports[k][1:]) for k in used},
        "routes": routes,
    }
    with open(os.path.join(HERE, "routes_fr24.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"\nroutes_fr24.json written: {len(routes)} routes, {len(used)} airports")
    for a, b, pax, fpd, s, *_ in routes[:8]:
        print(f"  {a}-{b}  {pax/1e6:.2f}M pax/yr  {fpd} flt/day")


if __name__ == "__main__":
    main()
