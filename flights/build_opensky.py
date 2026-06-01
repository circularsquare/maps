#!/usr/bin/env python3
"""
build_opensky.py  -  Turn an OpenSky monthly flightlist into estimated-passenger routes.

INPUT  : data/flightlist_YYYYMM.csv.gz   (OpenSky, Zenodo record 7923702)
         data/airports.dat               (OpenFlights -- for ICAO->IATA + coords)
OUTPUT : routes.json                     (consumed by index.html)

Unlike build_routes.py (OpenFlights topology, frequency ASSUMED), this uses the
real observed flight count per route. Passenger model:

    pax/yr = observed_flights/month * 12 * avg_seats * LOAD_FACTOR / COVERAGE

Caveats baked into COVERAGE: OpenSky only resolves both airports for ~51% of
flights, and receiver coverage is uneven (good in EU/US/AU, thin in Africa /
parts of Asia). The 1/COVERAGE scale-up is a flat correction -- it slightly
over-corrects well-covered regions. Refine later with per-region coverage.
"""
import csv, gzip, json, collections, os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FLIGHTLIST = os.path.join(DATA, "flightlist_201907.csv.gz")
DAYS_IN_MONTH = 31    # July

# --- assumptions -------------------------------------------------------------
LOAD_FACTOR  = 0.80   # global passenger load factor
COVERAGE     = 1.0    # NO scale-up. A flat 1/0.51 doubled already-complete
                      # counts in EU/US/AU (~2x too hot). Observed counts are
                      # used as-is: accurate where coverage is good, a LOWER
                      # BOUND where it is thin (Africa, parts of Asia).
MIN_FLIGHTS  = 3      # drop routes with fewer observed flights/month (noise/GA)

# Typical seats by ICAO aircraft type code (1-2 class layouts). OpenSky uses
# ICAO typecodes (B738, A320...) -- different from OpenFlights' IATA codes.
SEATS = {
    "A318":110,"A319":140,"A320":165,"A321":200,"A19N":140,"A20N":165,"A21N":210,
    "A306":260,"A30B":260,"A310":220,
    "A332":250,"A333":290,"A338":280,"A339":300,"A337":290,
    "A342":260,"A343":280,"A345":290,"A346":320,
    "A359":300,"A35K":350,"A388":525,
    "B731":110,"B732":120,"B733":140,"B734":150,"B735":110,"B736":110,
    "B737":140,"B738":189,"B739":178,"B37M":160,"B38M":178,"B39M":190,"B3XM":190,
    "B741":410,"B742":410,"B743":410,"B744":416,"B748":410,"B74S":350,
    "B752":200,"B753":230,"B762":210,"B763":245,"B764":245,
    "B772":300,"B77L":300,"B773":350,"B77W":365,
    "B788":242,"B789":290,"B78X":330,
    "B712":110,"B722":130,
    "BCS1":110,"BCS3":140,"BCS5":140,
    "E170":76,"E75S":76,"E75L":82,"E190":100,"E195":120,"E290":110,"E295":130,
    "E135":37,"E145":50,"E45X":50,"ER3":37,"ER4":50,
    "CRJ1":50,"CRJ2":50,"CRJ7":70,"CRJ9":90,"CRJX":100,"CL30":9,"CL60":9,
    "DH8A":37,"DH8B":37,"DH8C":50,"DH8D":78,"DHC6":19,"DH3T":50,
    "AT43":46,"AT45":46,"AT46":46,"AT72":70,"AT73":70,"AT75":70,"AT76":70,"ATP":64,
    "SF34":34,"SB20":50,"SW4":19,"J328":32,"J31":19,"J32":19,"JS41":29,
    "F50":50,"F70":80,"F100":100,"F28":65,"F27":44,
    "B461":80,"B462":85,"B463":100,"RJ1H":100,"RJ70":70,"RJ85":85,
    "MD11":290,"MD81":150,"MD82":150,"MD83":155,"MD87":140,"MD88":150,"MD90":160,
    "DC93":110,"DC95":110,
    "SU95":98,"AN24":48,"AN26":40,"AN28":18,"AN72":68,
    "T134":76,"T154":160,"T204":210,"YK40":32,"YK42":120,
    "IL62":160,"IL76":140,"IL96":300,
    "C208":9,"B190":19,"BE20":9,"E110":18,"L410":19,"SH36":36,"SH33":30,
    "AT4":46,"AT7":70,"D328":32,
}


def load_airports():
    """ICAO code -> (iata, lon, lat, city, country). iata falls back to ICAO."""
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


def main():
    airports = load_airports()
    print(f"airports with ICAO + coords : {len(airports)}")

    pairs = collections.defaultdict(lambda: {"flights": 0, "seats": 0.0})
    n = used_ok = sd = no_ap = no_type = 0
    with gzip.open(FLIGHTLIST, "rt", encoding="utf-8", errors="replace") as f:
        next(f)  # header
        for line in f:
            p = line.split(",")
            if len(p) < 7:
                continue
            n += 1
            typecode, origin, dest = p[4].strip(), p[5].strip(), p[6].strip()
            if not origin or not dest:
                continue
            if origin == dest:                 # training circuits etc.
                sd += 1; continue
            seats = SEATS.get(typecode)
            if seats is None:                  # not a known airliner -> skip GA/biz
                no_type += 1; continue
            if origin not in airports or dest not in airports:
                no_ap += 1; continue
            used_ok += 1
            rec = pairs[tuple(sorted((origin, dest)))]
            rec["flights"] += 1
            rec["seats"]   += seats

    print(f"flight rows scanned         : {n:,}")
    print(f"  skipped same origin/dest  : {sd:,}")
    print(f"  skipped non-airliner type : {no_type:,}")
    print(f"  skipped unknown airport   : {no_ap:,}")
    print(f"  used                      : {used_ok:,}")
    print(f"airport pairs (pre-filter)  : {len(pairs):,}")

    routes = []
    used = set()
    for (a, b), rec in pairs.items():
        if rec["flights"] < MIN_FLIGHTS:
            continue
        avg_seats = rec["seats"] / rec["flights"]
        pax = rec["flights"] * 12 * avg_seats * LOAD_FACTOR / COVERAGE
        fpd = rec["flights"] / DAYS_IN_MONTH / COVERAGE     # est real flights/day
        routes.append([airports[a][0], airports[b][0],
                       round(pax), round(fpd, 1), round(avg_seats)])
        used.add(a); used.add(b)

    routes.sort(key=lambda r: -r[2])
    out = {
        "meta": {
            "source": "OpenSky flightlist_201907 (observed flights, July 2019)",
            "model": f"pax/yr = observed flights/mo * 12 * seats * {LOAD_FACTOR} LF",
            "note": "Observed counts, not scaled up. Lower bound where ADS-B coverage is thin (Africa, parts of Asia). July annualised x12.",
            "n_routes": len(routes),
            "n_airports": len(used),
        },
        "airports": {airports[k][0]: list(airports[k][1:]) for k in used},
        "routes": routes,
    }
    with open(os.path.join(HERE, "routes.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))

    size_mb = os.path.getsize(os.path.join(HERE, "routes.json")) / 1e6
    print(f"\nroutes.json written         : {len(routes)} routes, {len(used)} airports, {size_mb:.1f} MB")
    print("\ntop 12 routes by estimated passengers:")
    ad = {airports[k][0]: airports[k] for k in used}
    for a, b, pax, fpd, s in routes[:12]:
        ca, cb = ad[a][3], ad[b][3]
        print(f"  {a}-{b}  {ca}<->{cb}".ljust(46) +
              f"{pax/1e6:6.2f}M pax/yr  {fpd:5.1f} flt/day")


if __name__ == "__main__":
    main()
