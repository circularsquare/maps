#!/usr/bin/env python3
"""
build_backfill.py — turn flightsfrom.com page dumps into data/backfill_routes.csv.

For each FR24 coverage-hole airport, save its flightsfrom.com "Destinations"
page text as data/airport_<iata>.txt (e.g. data/airport_tas.txt for Tashkent).
This script parses every such file, maps codes to ICAO, estimates weekly
frequency + seats, dedupes, and writes the CSV that build_fr24.py reads.

Frequency: flightsfrom shows "N flights per day" or "N-M flights per day" per
route -> weekly = mean(per-day) * 7.
Seats: derived from flight DURATION (a proxy for narrow- vs wide-body), since
flightsfrom doesn't give aircraft type. Tunable via SEATS_BY_DURATION below.

Re-run any time you add/replace an airport_<iata>.txt file. Idempotent.
"""
import os, csv, re, glob, collections

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT  = os.path.join(DATA, "backfill_routes.csv")

# duration threshold (minutes) -> typical seats. The only modelling knob here.
# Under the threshold = narrowbody single-aisle; over = widebody long-haul.
WIDEBODY_MINUTES = 390      # 6h30
NARROW_SEATS, WIDE_SEATS = 180, 280

# flightsfrom occasionally uses non-standard / stale IATA. Map to ICAO directly.
IATA_OVERRIDE = {
    "BSZ": "UAFM",   # "BSZ Bishkek" -> Manas International
    "NQZ": "UACC",   # Astana (renamed; airports.dat still has it under UACC)
    "RMO": "LUKK",   # Chisinau
    "GRV": "URMG",   # Grozny (airports.dat labels URMG Khankala — same city)
    "TFU": "ZUUU",   # Chengdu Tianfu -> Shuangliu (Tianfu ZUTF not in airports.dat)
    "SPX": "HECA",   # Cairo Sphinx -> Cairo Intl (Sphinx HESX not in airports.dat)
    "IKG": "UAFL",   # Karakol (new) -> Issyk-Kul/Tamchy, same lake region
}

ANCHOR  = re.compile(r"^([A-Z]{3}) [A-Za-z0-9]")          # "IST Istanbul" dest line
# All-caps 3-letter AIRLINE names ("ATA Airlines", "ITA Airways", "AIS Airlines")
# look exactly like a dest anchor and get mis-mapped to a same-coded airport
# (ATA -> Anta, Peru). Exclude any anchor whose 2nd word is an airline word.
AIRLINE = re.compile(r"^[A-Z]{3} (Airlines|Airways|Airline|Aviation|Aero|Jet|Express|Connection)\b")
FREQ    = re.compile(r"(\d+)(?:\s*-\s*(\d+))?\s+flights?\s+per\s+day")
DUR     = re.compile(r"^(\d+)h\s+(\d+)m$")


def iata_to_icao():
    m = {}
    with open(os.path.join(DATA, "airports.dat"), encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            iata, icao = row[4].strip(), row[5].strip()
            if len(iata) == 3 and len(icao) == 4 and icao != "\\N":
                m.setdefault(iata, icao)
    return m


def parse_file(path, orig_icao, i2i):
    """Yield (dest_icao, weekly, seats, note) for one airport_<iata>.txt."""
    lines = [ln.rstrip("\n") for ln in open(path, encoding="utf-8")]
    # find the dest-anchor line indices, then slice each block to the next anchor
    anchors = [i for i, ln in enumerate(lines)
               if ANCHOR.match(ln) and not AIRLINE.match(ln)]
    for k, start in enumerate(anchors):
        end = anchors[k + 1] if k + 1 < len(anchors) else len(lines)
        block = lines[start:end]
        iata = ANCHOR.match(lines[start]).group(1)
        city = lines[start][4:].strip()
        # skip not-yet-launched routes ("New airline: <date>")
        if any(ln.strip().startswith("New airline:") for ln in block):
            continue
        per_day = dur_min = None
        freq_txt = dur_txt = ""
        for ln in block:
            mf = FREQ.search(ln)
            if mf and per_day is None:
                lo = int(mf.group(1)); hi = int(mf.group(2)) if mf.group(2) else lo
                # "0-1 flights per day" means sub-daily / intermittent; the
                # arithmetic mean (0.5) overstates it, so use 0.25 (~1.75/wk).
                # All other ranges keep the plain midpoint.
                per_day = 0.25 if (lo == 0 and hi == 1) else (lo + hi) / 2.0
                freq_txt = ln.strip()
            md = DUR.match(ln.strip())
            if md and dur_min is None:
                dur_min = int(md.group(1)) * 60 + int(md.group(2))
                dur_txt = ln.strip()
        if per_day is None or per_day <= 0:
            continue
        dest = IATA_OVERRIDE.get(iata) or i2i.get(iata)
        if not dest:
            yield (None, iata, city)          # unmapped -> reported
            continue
        if dest == orig_icao:
            continue
        seats = WIDE_SEATS if (dur_min and dur_min >= WIDEBODY_MINUTES) else NARROW_SEATS
        weekly = round(per_day * 7.0, 1)
        note = f"{city}; {freq_txt}; {dur_txt}".strip("; ")
        yield (dest, orig_icao, weekly, seats, iata, note)


def main():
    i2i = iata_to_icao()
    files = sorted(glob.glob(os.path.join(DATA, "airport_*.txt")))
    if not files:
        print("no data/airport_*.txt files found"); return

    best = {}            # sorted ICAO pair -> (orig, dest, weekly, seats, note)
    unmapped = collections.Counter()
    per_airport = collections.Counter()
    for path in files:
        iata = os.path.basename(path)[len("airport_"):-len(".txt")].upper()
        orig = IATA_OVERRIDE.get(iata) or i2i.get(iata)
        if not orig:
            print(f"!! origin IATA {iata} ({path}) not in airports.dat — skipping file")
            continue
        for rec in parse_file(path, orig, i2i):
            if rec[0] is None:                 # unmapped dest
                unmapped[rec[1]] += 1
                continue
            dest, o, weekly, seats, diata, note = rec
            key = tuple(sorted((o, dest)))
            # keep the higher-frequency observation if a pair appears twice
            if key not in best or weekly > best[key][2]:
                best[key] = (o, dest, weekly, seats, note)
            per_airport[iata] += 1

    rows = sorted(best.values(), key=lambda r: (-r[2], r[0], r[1]))

    header = (
        "# backfill_routes.csv — GENERATED by build_backfill.py from\n"
        "# data/airport_<iata>.txt flightsfrom.com dumps. Do not hand-edit;\n"
        "# edit the source .txt or build_backfill.py and re-run. Schedule-derived\n"
        "# ESTIMATES for FR24's coverage holes; tagged source=\"sched\" downstream.\n"
        "# seats are a DURATION proxy (<6h30 -> 180, else 280) — tune in build_backfill.py.\n"
    )
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        f.write(header)
        w = csv.writer(f)
        w.writerow(["orig_icao", "dest_icao", "flights_per_week", "aircraft", "seats", "note"])
        for o, d, weekly, seats, note in rows:
            w.writerow([o, d, weekly, "", seats, note])

    print(f"files parsed     : {len(files)}")
    for iata in sorted(per_airport):
        print(f"  airport_{iata.lower()}: {per_airport[iata]} routes")
    print(f"unique routes    : {len(rows)}")
    if unmapped:
        print(f"UNMAPPED dest IATA (add to IATA_OVERRIDE): {dict(unmapped)}")
    print(f"written          : {OUT}")


if __name__ == "__main__":
    main()
