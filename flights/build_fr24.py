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
import json, os, csv, collections

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# pull/<DAY>/ folders to aggregate, and the window (hours) each one covers.
PULL_DIRS    = [os.path.join(HERE, "pull", "2026-05-16")]
WINDOW_HOURS = 24.0         # 1 = first-hour validation pull; 24 = a full day
LOAD_FACTOR  = 0.80

# observed-flight count -> per-year estimate. 8760 hours/year.
YEAR_FACTOR  = 8760.0 / (WINDOW_HOURS * len(PULL_DIRS))

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


def main():
    airports = load_airports()
    flights = read_flights()
    print(f"flight records        : {len(flights)}")

    pairs = collections.defaultdict(lambda: {"n": 0, "seats": 0.0})
    kept = skip_cancel = skip_ga = skip_apt = 0
    for r in flights:
        o, d, t = r.get("orig_icao"), r.get("dest_icao"), r.get("type")
        if not r.get("datetime_takeoff"):       # cancelled / never flew
            skip_cancel += 1; continue
        if not o or not d or o == d:
            continue
        seats = SEATS.get(t)
        if seats is None:                       # not an airliner -> size filter
            skip_ga += 1; continue
        if o not in airports or d not in airports:
            skip_apt += 1; continue
        rec = pairs[tuple(sorted((o, d)))]
        rec["n"] += 1
        rec["seats"] += seats
        kept += 1

    print(f"  skipped cancelled     : {skip_cancel}")
    print(f"  skipped non-airliner  : {skip_ga}")
    print(f"  skipped unknown apt   : {skip_apt}")
    print(f"  kept                  : {kept}")

    routes, used = [], set()
    for (a, b), rec in pairs.items():
        avg_seats = rec["seats"] / rec["n"]
        annual_flights = rec["n"] * YEAR_FACTOR
        pax = annual_flights * avg_seats * LOAD_FACTOR
        routes.append([airports[a][0], airports[b][0],
                       round(pax), round(annual_flights / 365, 1), round(avg_seats)])
        used.add(a); used.add(b)

    routes.sort(key=lambda r: -r[2])
    out = {
        "meta": {
            "source": f"FR24 pull — {', '.join(os.path.basename(d) for d in PULL_DIRS)} "
                      f"({WINDOW_HOURS:g}h window)",
            "note": f"Observed flights annualised x{YEAR_FACTOR:.0f} "
                    f"({WINDOW_HOURS:g}h x {len(PULL_DIRS)} day(s) -> 8760h/yr). "
                    f"Day-of-week / seasonal mix is whatever was pulled.",
            "n_routes": len(routes),
            "n_airports": len(used),
        },
        "airports": {airports[k][0]: list(airports[k][1:]) for k in used},
        "routes": routes,
    }
    with open(os.path.join(HERE, "routes_fr24.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"\nroutes_fr24.json written: {len(routes)} routes, {len(used)} airports")
    for a, b, pax, fpd, s in routes[:8]:
        print(f"  {a}-{b}  {pax/1e6:.2f}M pax/yr  {fpd} flt/day")


if __name__ == "__main__":
    main()
