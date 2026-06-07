#!/usr/bin/env python3
"""make_holes_template.py — one-off: emit data/holes_to_grab.txt, the list of
FR24 coverage-hole airports (real, operational, appreciable traffic) with their
flightsfrom.com URLs and a ">>> XXX" marker per airport. Paste each page's text
under its marker into data/holes_pasted.txt, then run split_holes.py.
Excludes airports that already have a dump and the genuinely-closed/war ones.
"""
import os, csv, json, collections

os.chdir(os.path.dirname(os.path.abspath(__file__)))

MIN_ROUTES = 8

# genuinely closed / moved / no-service -> not FR24 holes, skip
SKIP = {
    "UKBB", "UKKK", "UKOO", "UKLL", "UKCC", "UKDD", "UKHH", "UKFF",  # Ukraine, war
    "HSSS",                               # Khartoum, war (closed 2023)
    "URRP", "UUOB", "UUOO",               # Rostov/Belgorod/Voronezh, closed 2022
    "ZBNY",                               # Beijing Nanyuan, closed 2019
    "VDSR",                               # old Siem Reap, closed 2023 (-> new SAI)
    "EDDT",                               # Berlin Tegel, closed 2020 (-> BER)
    "OYSN", "OYTZ", "OYHD", "OYAA",       # Yemen — not listed on flightsfrom at all
    "LLET",                               # Eilat city, closed 2019 (-> Ramon LLER, not in airports.dat)
    "GOOY",                               # Dakar Senghor, closed 2017 (-> Blaise Diagne GOBD, not in airports.dat)
    "ZMUB",                               # Ulaanbaatar Buyant-Ukhaa, closed 2021 (-> UBN ZMCK, not in airports.dat)
}

# no appreciable airliner traffic (heli / bush / sub-30-seat) -> drop entirely
JUNK = {
    "BGUK", "BGUM", "BGJH", "BGBW", "BGNN",   # Greenland heli/STOL
    "CYUT", "PAHL", "FMNR", "NVSP", "WBGM",   # Repulse Bay, Huslia, Maroantsetra, Norsup, Marudi
}

# PARTIAL holes: airports FR24 sees but at a tiny fraction of their real network
# (see scan_partial.py). Curated to FR24's known-weak regions only — European
# low-ratio airports are OpenFlights overcounting seasonal LCC routes, not holes.
# Safe to backfill: merge_backfill keeps observed routes and only adds the
# missing ones. IATA -> short note.
PARTIAL = {
    "TIP": "Tripoli (saw 1 flight)",      "PNH": "Phnom Penh (saw 2)",
    "THR": "Tehran Mehrabad (saw 9)",     "IKA": "Tehran IKA (saw 43)",
    "SYZ": "Shiraz (saw 7)",              "MHD": "Mashhad (saw 33)",
    "LBD": "Khujand (saw 11)",            "DYU": "Dushanbe (saw 49)",
    "PEW": "Peshawar (saw 12)",           "KBL": "Kabul (saw 33)",
    "EBL": "Erbil (saw 58)",              "YKS": "Yakutsk (saw 16)",
    "KHV": "Khabarovsk (saw 42)",         "DME": "Moscow Domodedovo (saw 208)",
    "LAD": "Luanda (saw 25)",             "DLA": "Douala (saw 41)",
    "COO": "Cotonou (saw 35)",            "BKO": "Bamako (saw 28)",
    "FNA": "Freetown (saw 14)",           "BDO": "Bandung (saw 25)",
}

# airports flightsfrom.com has no usable page for -> can't backfill, exclude
_nf = os.path.join("data", "no_flightsfrom.txt")
if os.path.exists(_nf):
    for ln in open(_nf, encoding="utf-8"):
        code = ln.split("#")[0].strip()
        if len(code) == 4:
            SKIP.add(code)

# ICAOs already covered as an ORIGIN in the existing backfill (catches dumps
# named by an override IATA, e.g. bsz->UAFM Bishkek, azn->UTKA Andizhan).
COVERED = set()
_bf = os.path.join("data", "backfill_routes.csv")
if os.path.exists(_bf):
    with open(_bf, encoding="utf-8") as f:
        rows = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    for r in csv.DictReader(rows):
        o = (r.get("orig_icao") or "").strip()
        if o:
            COVERED.add(o)

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

out = collections.defaultdict(int)
status = {}
for d in ["pull/2026-05-15", "pull/2026-05-16"]:
    for fn in os.listdir(d):
        if not fn.endswith(".json") or fn.startswith("_"):
            continue
        ic = fn[:-5]
        doc = json.load(open(os.path.join(d, fn), encoding="utf-8"))
        status[ic] = status.get(ic, True) and bool(doc.get("ok"))
        out[ic] += len(doc.get("records", []))

deg = collections.Counter()
with open("data/routes.dat", encoding="utf-8") as f:
    for row in csv.reader(f):
        if len(row) < 5:
            continue
        s, t = row[2].strip(), row[4].strip()
        if len(s) == 3 and len(t) == 3:
            deg[s] += 1
            deg[t] += 1

holes = []
for iata, dg in deg.items():
    if dg < MIN_ROUTES:
        continue
    ic = iata2icao.get(iata)
    if not ic or ic in SKIP or ic in JUNK or ic in COVERED or not status.get(ic) or out[ic] > 0:
        continue
    if os.path.exists(f"data/airport_{iata.lower()}.txt"):
        continue
    c = meta.get(ic, (iata, "?", "?"))
    holes.append((c[2], dg, ic, c[0], c[1]))

bycountry = collections.defaultdict(list)
for country, dg, ic, iata, city in holes:
    bycountry[country].append((dg, ic, iata, city))

L = []
L.append("# holes_to_grab.txt - flightsfrom.com pages to grab for FR24 coverage holes.")
L.append("# HOW TO USE: for each airport, open the URL, copy the destinations list, and")
L.append('# paste it into data/holes_pasted.txt directly UNDER that airport\'s ">>> XXX"')
L.append("# marker line. Then Claude runs split_holes.py to fan it back out.")
L.append("# Skip (delete the block for) any airport whose page shows no real service.")
L.append("")

# --- PARTIAL holes first (highest value: real hubs FR24 barely sees) ---
L.append("#" + "=" * 70)
L.append("# PARTIAL HOLES — real hubs FR24 sees only a sliver of. Backfill adds the")
L.append("# routes it's missing (observed ones are kept automatically). High value.")
L.append("#" + "=" * 70)
np = 0
for iata in sorted(PARTIAL, key=lambda i: meta.get(iata2icao.get(i, ""), ("", "", "zz"))[2]):
    ic = iata2icao.get(iata)
    note = PARTIAL[iata]
    country = meta.get(ic, (iata, "?", "?"))[2] if ic else "?"
    L.append(f">>> {iata}   https://www.flightsfrom.com/{iata}   # {note} — {country}")
    np += 1
L.append("")

# --- TOTAL holes by country (most scheduled traffic first) ---
L.append("#" + "=" * 70)
L.append(f"# TOTAL HOLES — blank in FR24 both directions. {len(holes)} airports, "
         f"{len(bycountry)} countries.")
L.append("# Tip: skip any page showing only 1-3 destinations to nearby big hubs.")
L.append("#" + "=" * 70)
n = 0
for country in sorted(bycountry, key=lambda c: -sum(x[0] for x in bycountry[c])):
    L.append(f"### {country}  ({len(bycountry[country])} airports)")
    for dg, ic, iata, city in sorted(bycountry[country], key=lambda x: -x[0]):
        L.append(f">>> {iata}   https://www.flightsfrom.com/{iata}   # {city} ({dg} sched routes)")
        n += 1
    L.append("")

with open("data/holes_to_grab.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print(f"wrote data/holes_to_grab.txt: {np} partial + {n} total = {np + n} airports")
