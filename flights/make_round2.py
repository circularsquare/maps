#!/usr/bin/env python3
"""make_round2.py — write data/round2_pasted.txt: the NEXT round of airports to
grab, ranked by rough estimated traffic. Fresh file (round 1's holes_pasted.txt
is left alone). Paste each flightsfrom page under its ">>> XXX" marker, then:
    python split_holes.py round2_pasted.txt && python build_backfill.py && python build_fr24.py

Contents, in priority order:
  1. PARTIAL holes + newly-added airports (DSS/ETM/UBN/YIA) — hand-ranked by
     rough annual pax, since routes.dat can't size post-2014 airports.
  2. Remaining TOTAL holes not yet grabbed — ranked by distinct destinations.
"""
import os, csv, json, collections

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Hand-ranked priority set: IATA -> (rough est. annual pax in millions, note).
# Covers the 20 partial holes + the 4 moved/new airports. Order in the file
# follows this pax estimate (best available; FR24 obs is undercounted and 2014
# routes.dat predates the new airports).
PRIORITY = [
    ("DME", 25, "Moscow Domodedovo — partial (saw 208), many routes missing"),
    ("THR", 16, "Tehran Mehrabad — domestic hub, partial (saw 9)"),
    ("MHD", 8,  "Mashhad — Iran pilgrimage hub, partial (saw 33)"),
    ("IKA", 8,  "Tehran Imam Khomeini — Iran intl gateway, partial (saw 43)"),
    ("KTI", 5,  "Phnom Penh Techo (NEW airport, replaces closed PNH)"),
    ("YIA", 5,  "Yogyakarta (NEW airport, just added) — not yet pulled"),
    ("SYZ", 4,  "Shiraz — partial (saw 7)"),
    ("DSS", 3,  "Dakar Blaise Diagne — real Dakar hub, not pulled"),
    ("NBJ", 3,  "Luanda Agostinho Neto (NEW airport, replaces closed LAD)"),
    ("KBL", 2.5, "Kabul — partial (saw 33)"),
    ("MJI", 2.5, "Tripoli Mitiga (real Tripoli airport, replaces closed TIP)"),
    ("DYU", 2,  "Dushanbe — partial (saw 49)"),
    ("PEW", 2,  "Peshawar — partial (saw 12)"),
    ("KHV", 2,  "Khabarovsk — partial (saw 42)"),
    ("EBL", 2,  "Erbil — partial (saw 58)"),
    ("UBN", 2,  "Ulaanbaatar (NEW airport, just added) — not yet pulled"),
    ("ETM", 2,  "Eilat Ramon — not pulled (replaces closed Eilat city)"),
    ("YKS", 1.5, "Yakutsk — partial (saw 16)"),
    ("DLA", 1.5, "Douala — partial (saw 41)"),
    ("COO", 1.5, "Cotonou — partial (saw 35)"),
    ("BKO", 1.5, "Bamako — partial (saw 28)"),
    ("LBD", 1.5, "Khujand — partial (saw 11)"),
    ("FNA", 0.8, "Freetown — partial (saw 14)"),
    ("BDO", 0.5, "Bandung — partial (saw 25)"),
]

# closed / war / moved / no-appreciable-traffic — same exclusions as make_holes_template.py
SKIP_TOTAL = {
    "WARJ",                                                          # JOG old Yogyakarta (-> YIA)
    "UKBB", "UKKK", "UKOO", "UKLL", "UKCC", "UKDD", "UKHH", "UKFF",  # Ukraine, war
    "HSSS", "URRP", "UUOB", "UUOO", "ZBNY", "VDSR", "EDDT",          # closed
    "OYSN", "OYTZ", "OYHD", "OYAA",                                  # Yemen (not on flightsfrom)
    "LLET", "GOOY", "ZMUB",                                          # moved-old airports
    "BGUK", "BGUM", "BGJH", "BGBW", "BGNN",                          # Greenland heli/STOL
    "CYUT", "PAHL", "FMNR", "NVSP", "WBGM",                          # bush strips
}
# airports flightsfrom.com has no usable page for -> can't backfill, exclude
_nf = os.path.join("data", "no_flightsfrom.txt")
if os.path.exists(_nf):
    for ln in open(_nf, encoding="utf-8"):
        code = ln.split("#")[0].strip()
        if len(code) == 4:
            SKIP_TOTAL.add(code)

iata2icao, meta = {}, {}
for row in csv.reader(open("data/airports.dat", encoding="utf-8")):
    if len(row) < 8:
        continue
    icao, iata = row[5].strip(), row[4].strip()
    if len(icao) != 4 or icao == "\\N":
        continue
    if len(iata) == 3 and iata != "\\N":
        iata2icao.setdefault(iata, icao)
    meta[icao] = (iata if len(iata) == 3 else icao, row[2], row[3])

out, status = collections.defaultdict(int), {}
for d in ["pull/2026-05-15", "pull/2026-05-16"]:
    for fn in os.listdir(d):
        if fn.endswith(".json") and not fn.startswith("_"):
            ic = fn[:-5]
            doc = json.load(open(os.path.join(d, fn), encoding="utf-8"))
            status[ic] = status.get(ic, True) and bool(doc.get("ok"))
            out[ic] += len(doc.get("records", []))

# distinct destinations per IATA (hub-size proxy for the total-hole tail)
dd = collections.defaultdict(set)
for row in csv.reader(open("data/routes.dat", encoding="utf-8")):
    if len(row) >= 5 and len(row[2].strip()) == 3 and len(row[4].strip()) == 3:
        dd[row[2].strip()].add(row[4].strip())
        dd[row[4].strip()].add(row[2].strip())

# already grabbed = a dump file exists OR already an origin in the backfill csv
grabbed = set()
for row in csv.reader(open("data/backfill_routes.csv", encoding="utf-8")):
    if row and row[0].strip() and len(row[0].strip()) == 4 and row[0].strip() != "orig_icao":
        grabbed.add(row[0].strip())

priority_icaos = {iata2icao.get(ia) for ia, _, _ in PRIORITY}

# remaining TOTAL holes: ok, 0 observed, real network, not grabbed/priority/skip
tail = []
for iata, dests in dd.items():
    n = len(dests)
    if n < 4:
        continue
    ic = iata2icao.get(iata)
    if not ic or ic in priority_icaos or ic in SKIP_TOTAL:
        continue
    if not status.get(ic) or out[ic] > 0 or ic in grabbed:
        continue
    if os.path.exists(f"data/airport_{iata.lower()}.txt"):
        continue
    # drop the obvious junk regions handled in make_holes_template
    tail.append((n, iata, meta[ic][1], meta[ic][2]))
tail.sort(reverse=True)

L = ["# round2_pasted.txt — next round of FR24 holes, ranked by rough est. traffic.",
     "# Paste each flightsfrom page UNDER its >>> XXX marker, then:",
     "#   python split_holes.py round2_pasted.txt && python build_backfill.py && python build_fr24.py",
     "# Stop wherever the traffic gets too low to bother.",
     "",
     "#" + "=" * 70,
     "# TIER 1 — partial holes + new airports (hand-ranked by ~annual pax)",
     "#" + "=" * 70]
for iata, pax, note in PRIORITY:
    if iata2icao.get(iata) in SKIP_TOTAL:        # no flightsfrom page -> skip
        continue
    L.append(f">>> {iata}   https://www.flightsfrom.com/{iata}   # ~{pax}M pax — {note}")
L += ["",
      "#" + "=" * 70,
      f"# TIER 2 — remaining total holes ({len(tail)}), ranked by distinct destinations",
      "#" + "=" * 70]
for n, iata, city, country in tail:
    L.append(f">>> {iata}   https://www.flightsfrom.com/{iata}   # {city}, {country} ({n} dests)")
L.append("")

open("data/round2_pasted.txt", "w", encoding="utf-8").write("\n".join(L))
print(f"wrote data/round2_pasted.txt: {len(PRIORITY)} priority + {len(tail)} tail = {len(PRIORITY)+len(tail)} airports")
