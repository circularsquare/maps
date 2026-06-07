#!/usr/bin/env python3
"""scan_holes.py — one-off: find FR24 coverage holes in the pull.

FR24's flight-summary API has total-blackout airports (0 outbound AND no other
flight reports them as a destination) — Dhaka, Tashkent, Kuwait, Bishkek, etc.
OpenSky (routes.json) shares FR24's ADS-B blind spot over Central/South Asia, so
it can't see these either. OpenFlights routes.dat is SCHEDULE-based (global, no
ADS-B hole), so it's the right reference: an airport with many scheduled routes
but 0 FR24 outbound is a coverage hole. Ranked by scheduled-route degree,
grouped by country. (routes.dat is 2014-vintage -> a handful of since-closed
airports show up; KNOWN_CLOSED filters the well-known ones.)
"""
import json, os, csv, collections

PULL = ["pull/2026-05-15", "pull/2026-05-16"]
MIN_ROUTES = 8                      # scheduled-route degree floor (skip tiny fields)

# airports closed / blacked-out for reasons unrelated to FR24 coverage (2014
# routes.dat still lists them). Real 0s, not holes.
KNOWN_CLOSED = {
    "TXL",            # Berlin Tegel, closed 2020 (-> BER)
    "SDV",            # Tel Aviv Sde Dov, closed 2019
    "THR",            # Tehran Mehrabad still open but mostly domestic; keep? leave in
}

# IATA -> ICAO, ICAO -> (iata, city, country)
iata2icao, meta = {}, {}
with open("data/airports.dat", encoding="utf-8") as f:
    for row in csv.reader(f):
        if len(row) < 8:
            continue
        icao, iata = row[5].strip(), row[4].strip()
        if len(icao) != 4 or icao == "\\N":
            continue
        if len(iata) == 3 and iata != "\\N":
            iata2icao.setdefault(iata, icao)
        meta[icao] = (iata if len(iata) == 3 else icao, row[2], row[3])

# FR24 pull: outbound record count + ok status per ICAO
outbound = collections.defaultdict(int)
status = {}
for d in PULL:
    if not os.path.isdir(d):
        continue
    for fn in os.listdir(d):
        if not fn.endswith(".json") or fn.startswith("_"):
            continue
        icao = fn[:-5]
        doc = json.load(open(os.path.join(d, fn), encoding="utf-8"))
        status[icao] = status.get(icao, True) and bool(doc.get("ok"))
        outbound[icao] += len(doc.get("records", []))

# OpenFlights scheduled-route degree per IATA (distinct directed routes)
deg = collections.Counter()
with open("data/routes.dat", encoding="utf-8") as f:
    for row in csv.reader(f):
        if len(row) < 5:
            continue
        src, dst = row[2].strip(), row[4].strip()
        if len(src) == 3 and len(dst) == 3:
            deg[src] += 1
            deg[dst] += 1

holes = []
for iata, d in deg.items():
    if d < MIN_ROUTES or iata in KNOWN_CLOSED:
        continue
    icao = iata2icao.get(iata)
    if not icao or not status.get(icao):
        continue
    if outbound[icao] == 0:
        c = meta.get(icao, (iata, "?", "?"))
        holes.append((c[2], d, icao, c[0], c[1]))

bycountry = collections.defaultdict(list)
for country, d, icao, iata, city in holes:
    bycountry[country].append((d, icao, iata, city))

print(f"=== FR24 0-outbound holes w/ >= {MIN_ROUTES} scheduled routes (OpenFlights) ===")
print(f"({len(holes)} airports across {len(bycountry)} countries)\n")
for country in sorted(bycountry, key=lambda c: -sum(x[0] for x in bycountry[c])):
    tot = sum(x[0] for x in bycountry[country])
    print(f"{country}  ({tot} scheduled routes, {len(bycountry[country])} airports)")
    for d, icao, iata, city in sorted(bycountry[country], key=lambda x: -x[0]):
        have = "  [dump exists]" if os.path.exists(f"data/airport_{iata.lower()}.txt") else ""
        print(f"   {iata}  {icao}  {d:4d} routes   {city}{have}")
    print()
