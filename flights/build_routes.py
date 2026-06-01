#!/usr/bin/env python3
"""
build_routes.py  -  Turn OpenFlights data into an estimated-passenger route set.

INPUT  (data/):  airports.dat, routes.dat   (OpenFlights, https://openflights.org/data)
OUTPUT          :  routes.json              (consumed by index.html)

This is the v1 traffic estimate. OpenFlights gives route TOPOLOGY only -- which
airline flies which A->B with which aircraft -- but no flight frequency. So we
assume each airline-on-a-route operates FREQ_PER_DAY flights a day, multiply by
the aircraft's typical seat count and a load factor, and annualise.

When the OpenSky observed-flight data lands, the only thing that changes is that
the per-route flight count becomes a real measured number instead of an
assumption. The seats x load-factor x 365 part stays the same.
"""
import csv, json, collections, os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# --- assumptions (all in one place so they are easy to refine later) ----------
LOAD_FACTOR   = 0.80   # global passenger load factor. Real 2024 avg ~0.83.
FREQ_PER_DAY  = 1.0    # flights/day assumed per (airline, route). OpenSky fixes this.
DAYS          = 365

# Typical seats by IATA aircraft code (1-2 class layouts, rounded).
SEATS = {
    "320":165,"321":210,"319":135,"318":110,"32S":175,"32A":165,"32B":185,
    "32N":180,"32Q":210,"321N":210,"223":140,"221":110,"BCS":135,
    "738":180,"73H":180,"737":140,"73G":140,"73W":140,"733":140,"734":150,
    "735":110,"736":110,"732":120,"73C":140,"73J":180,"7M8":180,"7M9":190,
    "777":310,"77W":365,"772":300,"773":350,"77L":300,"77F":0,
    "787":260,"788":250,"789":290,"781":320,
    "763":245,"762":210,"764":245,"767":230,
    "757":200,"75W":200,"752":200,"753":230,
    "330":280,"333":290,"332":250,"339":300,"338":280,"33X":290,"33E":300,"339N":300,
    "343":280,"342":260,"345":290,"346":320,"340":290,
    "747":410,"744":416,"74E":410,"74H":467,"748":410,"74F":0,
    "380":525,"388":525,
    "717":110,"M80":150,"M81":150,"M82":150,"M83":155,"M87":140,"M88":150,"M90":160,
    "100":100,"146":100,"AR8":90,"AR1":90,"RJ1":100,"RJ8":90,"RJ7":80,"RJ85":100,
    "CRJ":50,"CR1":50,"CR2":50,"CR7":70,"CR9":90,"CRK":100,"CL":50,
    "E70":76,"E75":82,"E90":100,"E95":110,"E190":100,"EMJ":100,"E7W":76,
    "ER3":37,"ER4":50,"ERD":50,"ERJ":50,"E45":50,"E135":37,"E145":50,
    "AT4":46,"AT5":50,"AT7":70,"ATR":68,"ATP":64,
    "DH1":37,"DH2":37,"DH3":50,"DH4":78,"DH8":50,"DHT":37,
    "SF3":34,"SFB":34,"S20":19,"SWM":19,"J31":19,"J32":19,"J41":29,
    "F50":50,"F70":80,"F100":100,"FRJ":100,"F28":65,
    "AN4":52,"AN6":68,"YK2":120,"YK4":32,"SU9":98,"S95":98,"100E":100,
    "IL9":300,"I93":300,"T20":19,"T204":210,"BE1":19,"BEH":19,"BNI":9,
    "JS3":19,"D38":30,"D28":15,"D6":15,"L4T":19,"C208":9,"CNA":8,"CNC":8,
    "PAG":8,"PA2":8,"PL2":8,"DHC":19,"EM2":30,"ND2":26,"WWP":8,"GRJ":34,
    "73P":120,"310":220,"312":220,"313":220,"AB6":250,"ABF":0,
    "L10":300,"D10":270,"D11":290,"D9":110,"D95":110,
}
DEFAULT_SEATS = 150


def seats_for(equipment):
    """Average seats across the (space-separated) equipment codes on a route row."""
    vals = []
    for code in equipment.split():
        s = SEATS.get(code)
        if s is None:
            # strip a trailing 'N' (neo) etc. and retry, else default
            s = SEATS.get(code.rstrip("N"), DEFAULT_SEATS)
        if s > 0:
            vals.append(s)
    return sum(vals) / len(vals) if vals else DEFAULT_SEATS


def load_airports():
    """IATA code -> (lon, lat, city, country). Skips airports with no IATA / coords."""
    out = {}
    with open(os.path.join(DATA, "airports.dat"), encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 8:
                continue
            iata = row[4].strip()
            if not iata or iata == "\\N" or len(iata) != 3:
                continue
            try:
                lat, lon = float(row[6]), float(row[7])
            except ValueError:
                continue
            out[iata] = (round(lon, 4), round(lat, 4), row[2], row[3])
    return out


def main():
    airports = load_airports()
    print(f"airports with IATA + coords : {len(airports)}")

    # directed (src,dst) -> {seats: total, airlines: set}
    directed = collections.defaultdict(lambda: {"seats": 0.0, "airlines": set()})
    skipped_codeshare = skipped_airport = skipped_indirect = total = 0

    with open(os.path.join(DATA, "routes.dat"), encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 9:
                continue
            total += 1
            airline, src, dst, codeshare, stops, equip = (
                row[0], row[2], row[4], row[6], row[7], row[8])
            if codeshare == "Y":               # same plane sold twice -> skip
                skipped_codeshare += 1
                continue
            if stops not in ("0", ""):          # keep nonstop only
                skipped_indirect += 1
                continue
            if src not in airports or dst not in airports:
                skipped_airport += 1
                continue
            rec = directed[(src, dst)]
            rec["seats"] += seats_for(equip)
            rec["airlines"].add(airline)

    print(f"route rows total           : {total}")
    print(f"  skipped (codeshare)      : {skipped_codeshare}")
    print(f"  skipped (1+ stop)        : {skipped_indirect}")
    print(f"  skipped (unknown airport): {skipped_airport}")
    print(f"directed airport pairs     : {len(directed)}")

    # merge into undirected pairs
    undirected = collections.defaultdict(lambda: {"seats": 0.0, "airlines": set()})
    for (src, dst), rec in directed.items():
        key = tuple(sorted((src, dst)))
        u = undirected[key]
        u["seats"] += rec["seats"]
        u["airlines"] |= rec["airlines"]

    routes = []
    used = set()
    for (a, b), rec in undirected.items():
        # estimated passengers/year, both directions combined
        pax = rec["seats"] * FREQ_PER_DAY * DAYS * LOAD_FACTOR
        n_air = len(rec["airlines"])
        avg_seats = rec["seats"] / n_air if n_air else 0
        routes.append([a, b, round(pax), n_air, round(avg_seats)])
        used.add(a); used.add(b)

    routes.sort(key=lambda r: -r[2])
    out = {
        "meta": {
            "source": "OpenFlights routes.dat (route topology, ~2014 vintage)",
            "model": f"pax/yr = seats * {FREQ_PER_DAY} flights/day * {DAYS} * {LOAD_FACTOR} LF, both directions",
            "note": "Flight frequency is ASSUMED, not measured. Replace with OpenSky observed counts.",
            "n_routes": len(routes),
            "n_airports": len(used),
        },
        "airports": {k: airports[k] for k in sorted(used)},
        "routes": routes,
    }
    with open(os.path.join(HERE, "routes.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))

    size_mb = os.path.getsize(os.path.join(HERE, "routes.json")) / 1e6
    print(f"\nroutes.json written        : {len(routes)} routes, {len(used)} airports, {size_mb:.1f} MB")
    print("\ntop 12 estimated routes:")
    for a, b, pax, n, s in routes[:12]:
        ca, cb = airports[a][2], airports[b][2]
        print(f"  {a}-{b}  {ca}<->{cb}".ljust(48) + f"{pax/1e6:6.2f}M pax/yr  {n} airlines")


if __name__ == "__main__":
    main()
