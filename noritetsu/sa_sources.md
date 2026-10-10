# Saudi Arabia sources

## Survey (2026-10-08)

Research only; nothing built. Region code `sa`.

### What runs

All three main lines are run by **Saudi Arabia Railways (SAR)**, standard gauge
(passenger figures for Q2 2026 from SAR via Argaam / Madhyamam, 2026).

| line | stations in order | km (published) | service |
|---|---|---|---|
| **East Train** (Dammam - Riyadh railway, the old SRO line) | Riyadh - Hofuf - Abqaiq - Dammam | 449 (en.wikipedia "Dammam–Riyadh railway") | several a day; 389,000 passengers in Q2 2026 |
| **North Train** (the North - South Railway's passenger line) | Riyadh (North) - Majmaah - Qassim (Buraydah) - Hail - Al Jouf - Al Qurayyat | about 1,215-1,250 Riyadh - Qurayyat (en.wikipedia "North–South Railway"; to check) | daily; Qurayyat since Dec 2022; 246,000 passengers in Q2 2026. The luxury "Dream of the Desert" (from late 2026, one- and two-night cruises) is a named train, not a line |
| **Haramain High Speed Railway** (West Train) | Makkah - Jeddah (Al Sulaymaniyah) - King Abdulaziz International Airport - King Abdullah Economic City - Madinah | 453 | many a day; 1.6 million passengers in Q2 2026 |

**Urban**:

- **Riyadh Metro**: 6 lines, 176 km, 85 stations (all open since 5 January 2025): 1 Blue,
  2 Red, 3 Orange, 4 Yellow (KAFD - Airport T1-2, sharing track with 6), 5 Green, 6 Purple.
- **Princess Nourah University APM** (Riyadh, campus people mover, light_rail in OSM):
  leave out, as a people mover.
- **Makkah Al Mashaaer Al Mugaddassah Metro** (Mecca, 18 km): runs only for about a week
  at Hajj. Not weekly service: leave it out (or draw greyed). Note too that it serves only
  pilgrims with Hajj permits.
- **KAFD monorail** (Riyadh): built; whether it carries the public is unclear; leave out.
- Not open: Jeddah metro, Makkah metro, the Riyadh - Qurayyat freight branches, the
  Landbridge (Riyadh - Jeddah), the Al Haditha link to Jordan (freight).

### Sources

1. **OSM** (Overpass, 2026-10-08, `data/raw/sa/survey/osm_routes_ruh.json` for a Riyadh
   bbox; the whole-country queries timed out, see below):
   - Riyadh Metro: all six lines have route relations both ways with `ref` 1-6 and the
     official colours (#00AEE6 Blue, #EF3434 Red, #ED872D Orange, #FFD009 Yellow, #008000
     Green, #7F00FF Purple). Track mostly named per line in Arabic ("المسار الأحمر",
     "المسار الازرق", "المسار البرتقالي", "المسار البنفسجي", "المسارين 4 و 6" for the shared
     stretch) with 55 of 293 subway ways unnamed.
   - SAR: route relations "SAR Line 1: Riyadh <-> Dammam" (8273934, 13880223) and "Al
     Qurayyat <-> Riyadh" (13880224/5); track named "قطار الدمام - الخط الأول" / "الخط الثاني"
     (the two Riyadh - Dammam lines: the old line via Hofuf and the newer direct line) and
     "قطار الشمال" (North Train).
   - Haramain: one route=train relation, 7597735 "Haramain High Speed Line" (operator Saudi
     Arabia Railways), from a whole-country relations-only query
     (`osm_routes_x.json`; the bbox reaches Jordan, Israel, Iraq, Iran and the Gulf, so
     most of its rows are other countries'). Also 2567987 "قطار الشمال الجنوب" (the North -
     South railway, route=railway), 8273965 Riyadh - Az Zabirah (freight to the bauxite
     mine), 15925686 Jubail - Dammam (freight). The way-level query for the Haramain bbox
     timed out four times; the extract will show whether its track is named.
2. **Timetables**: SAR's booking site (sar.com.sa, tickets.sar.com.sa) and Haramain's
   (hhr.sa): no GTFS anywhere (not in the Mobility Database, not in Transitous). Riyadh
   Metro: no open GTFS found (RCRC / Riyadh Bus publish none openly). The three SAR lines
   run daily and Riyadh Metro runs all day, so no feed is needed to say what runs.
3. **Published lengths**: en.wikipedia for each line ("Haramain high-speed railway" 453 km;
   "Riyadh Metro" 176 km and per-line km; SAR lines as above).

### Licence

OSM (ODbL); Wikipedia for the check numbers.

### Geofabrik

`asia/gcc-states-latest.osm.pbf`, 242 MB (shared with ae and qa).

### Recipe

The shared Gulf reader (`qa_sources.md`): **hand lists traced over OSM track by rinf.py**
(nafrica's recipe), stations from OSM route relations' stops.

- SAR register lines: East (Riyadh - Hofuf - Abqaiq - Dammam: the passenger trains use the
  old line through Hofuf; the newer direct track "الخط الثاني" is freight, check which way
  OSM's route relation goes), North (Riyadh - Qurayyat, 6 stations), Haramain (5 stations,
  `highspeed`). Their stations are far apart (Hail - Al Jouf ~350 km): every section is
  station to station, so nothing is dropped as unridden; trace the whole route with
  `--trace` before trusting it, since a long desert line with freight branches (to the
  phosphate and bauxite mines at Al Jalamid and Az Zabirah) invites a wrong turn.
- Riyadh Metro as six register lines from OSM's route relations (stops in order), line 4
  and 6 sharing track (ownership.py decides).
- Expected: 9 register lines, about 2,300 km (449 + ~1,215 + 453 + 176), about 16 + 85
  stations.

### Open

- The North Train's exact passenger km and whether it calls anywhere else (Sudair?).
- Mashaaer metro: leave out (my call) or draw greyed so its track is visible.
- The whole-country way query timed out on 2026-10-08 (the server was busy); the
  extract will answer the rest (`probe_kr_ways.py --region sa`).

## Build (2026-10-08)

Built with `mideast_register.py` (nafrica_register.py's code: a hand station list per line,
traced over OSM track by rinf.py; lists in `mideast_lines.py`), not a reader of its own.

    python tools/slot.py 2 -- python extract.py --region sa --pbf data/raw/gcc-states-latest.osm.pbf --bbox 34.40,16.30,55.70,32.20 --station-areas
    python mideast_register.py --clip sa
    python tools/slot.py 2 -- python build_model.py --region sa --register mideast_register:data/raw/rinf/sa
    python tools/slot.py 2 -- python build_tiles.py --region sa
    python check_model.py --region sa

**Register lines, all running: 3 lines, 2,146 km.**

| line | built km | published | stops |
|---|---|---|---|
| East Train: Riyadh – Dammam | 448.6 | 449 (en.WP Dammam–Riyadh railway) | Riyadh, Hofuf, Abqaiq, Dammam |
| North Train: Riyadh – Al Qurayyat | 1,240.7 | 1,242 (en.WP Riyadh–Qurayyat railway) | Riyadh North, Majmaah, Al Qassim, Hail, Al Jouf, Al Qurayyat |
| Haramain High Speed Railway | 457.0 | 453 (en.WP) | Makkah, Jeddah Sulaymaniyah, KAIA airport, KAEC, Madinah |

The East Train's trace takes the 1981 direct line Riyadh - Hofuf by itself (448.6 against
449), so the old line via Al Kharj and Harad stays freight track. All three are
`listed_only`: their trains call at the listed stations only. The open questions above are
settled: the North Train calls at its six stations only (SAR's booking site, OSM's route).

**OSM lines**: Riyadh Metro's six lines (171 km; 0.96-0.98 of en.WP's line table station to
station, every station), and the Princess Nourah University people mover (10.7 km), kept as
an OSM line as other countries keep their airport and campus movers (Changi, KLIA,
Suvarnabhumi), reversing the survey's "leave out".

**Decisions**
- OSM's SAR route relations (8273934, 13880223-5) and the Haramain's (7597735, no stops) are
  left out in `rules/sa.py` (SKIP_ROUTES): each was the register line again under another
  name, drawn beside it ("SAR Line 1", "القريات - الرياض"; 13880223's ways run 706 km).
- The Mashaaer metro (Mecca, Hajj only) is dropped by `--clip` (mideast_lines.NOT_SERVICE).
- Dream of the Desert: not built (from late 2026; a named train when it runs).
- `--clip` for the Gulf: the bbox takes in Qatar's, Bahrain's and the UAE's coasts, and Dubai
  Marina, the Palm and Lusail's marina lie on reclaimed land outside every simplified outline,
  so nafrica's "inside another country" kept them. mideast_register.abroad cuts what lies
  within 0.15° (~15 km) of another country's outline and outside Saudi Arabia's own.
- Colours (`colours/sa.csv`) are picked: SAR publishes none.
