# United Kingdom sources (surveyed 2026-10-03)

Region code `gb` (ISO and Natural Earth; `borders.py` already files the Channel Tunnel point
eEU00228 under fr/gb). religiondots' outline file calls the UK `uk`, so
`tools/build_regions.py` falls back to Natural Earth's outline for `gb` unless it maps the
two (see "Shared files" below).

## Commands

```
python gb_register.py --fetch        # the small sources below into data/raw/gb (~12 MB, 1 min)
# the extract (managing session): Geofabrik europe/united-kingdom-latest.osm.pbf, ~2 GB
python extract.py --region gb --pbf data/raw/united-kingdom-latest.osm.pbf
python gb_register.py --clip         # France's half of the Channel Tunnel out (after every extract)
python inspect_region.py --region gb
python probe_kr_ways.py --region gb
python gb_register.py --names        # the folded line names, the ELR fill, what is left out
python build_model.py --region gb --register gb_register:data/raw/gb
python build_tiles.py --region gb
python check_model.py --region gb
```

## The build (2026-10-03)

Extract: Geofabrik `united-kingdom-latest.osm.pbf` (managing session), 101,015 track ways,
34,000 stops, 1,259 routes. `inspect_region`: 93.1% of main-line km and 69.9% of branch km
under a route relation; `probe_kr_ways`: 97.6% of main+branch rail km carries a name.

- **464 register lines, 15,291 km** (Network Rail route length is ~15,800 km including
  freight-only; NI Railways ~330 km), 859 lines in all, 4,106 stations. 3 named trains
  (Highland and Lowland Sleepers, the Jacobite).
- OSM passenger sections lie on register track for 43,550 km and on OSM-owned track for
  1,494 km (3.4%); "likely register gaps" 194 km of way length in 46 places.
- **Check** (`check_model.REGISTER["gb"]`, 53 lines against Wikipedia infobox lengths): 32
  within 5%, 41 within 15%. Exact or close: ECML 0.99 (629.0 / 632.7), North Wales Coast 1.00,
  Highland Main Line 1.00, Far North 1.00, Settle-Carlisle 1.01, Cornish Main Line 0.99,
  Aberdeen-Inverness 0.99, Kyle 0.99, Medway Valley 1.00, Breckland 1.00, Lakes Line 1.00. The
  rest are extent differences between OSM's name and the Wikipedia article, each in the
  check's note: OSM's "Heart of Wessex Line" is only Castle Cary - Dorchester (0.38), its
  "Shenfield to Southend Line" only Wickford - Southend (0.55), its "Durham Coast Line" also
  Northallerton - Eaglescliffe and the Metro-shared Sunderland stretch (1.71), and the Great
  Western Main Line (1.32) keeps a few OSM ways named so in Devon (Exeter, Dawlish, Totnes),
  which MLN1 is too.
- Build steps and what each moved (build log `data/logs/build_gb.txt`): 4,248 unnamed ways
  named (882 track-km); 324 stray short pieces folded into the line around them (390 km); 1,989
  junction ends; 59 shortcut sections dropped (1,279 km); junction-ended sections kept 1,115
  (4,018 km) where routes run, dropped 383 (1,119 km). Left out: heritage 422 track-km by
  Wikidata name, usage=tourism 497, the Underground's rail-tagged track 58, no ELR 48, test 50.
- First build for comparison: plain named track (kr_register as is) gave 350 lines and 10,827
  km, and left 3,074 km of way length as "likely register gaps".

## The line unit: what was measured

All figures from Overpass on 2026-10-03 (way lengths summed server-side, so double track
counts twice: "track-km"), or from the extract where it says so.

| candidate | what it is | count | coverage | overlaps | verdict |
|---|---|---|---|---|---|
| **OSM track `name`** (chosen) | Wikipedia-style line names on the ways: "Cotswold Line", "Hope Valley Line", "Settle-Carlisle Railway", "West Coast Main Line" | 1,415 names on rail track, 583 with 1 km or more under a passenger route (26,900 track-km), 406 with 5 km or more | 97.2% of passenger track-km named (probe_kr_ways on the extract: 97.6% of main+branch rail) | none: one name per way | the line a rider names; a partition of the track, so one owner per piece; some names are Network Rail ELR descriptions ("Weaver Junction and Liverpool Line") where mappers used those |
| ELR (`ref` on the ways) | Network Rail's Engineer's Line References: ECM1, MLN1, XTD | 1,029 ELR-shaped refs on main/branch track; 640 with 1 km or more of passenger track | 98.1% of passenger track-km | none | engineering units: the ECML is ECM1-ECM9, an ELR changes at odd places, curves are ELRs; a rider never says "MLN1". Kept as the glue: fills the 2.8% of unnamed passenger track |
| OSM `route=railway` relations | 708 in the UK, 311 with a ref | 708 | not a partition | heavy: the WCML relation (7,323 members) holds its branches too, and other relations repeat them | a mix of Wikipedia lines, ELRs imported for disused railways, Network Rail Strategic Route Sections ("SRS 07.01") and ScotRail's "SC001"-style sections; no |
| Wikipedia / Wikidata named lines | 1,411 UK line items with no ELR; 178 open ones carry an OSM relation, 161 a length | ~1,400 | no geometry of their own | heavy (main lines hold their branches) | the names, but not a register; used for the check table |

Of the 640 ELRs with passenger track, 494 (77%) have 90% or more of it under one track name,
and the median passenger line name covers one ELR: names and ELRs mostly cut the network at the
same places, the names are just what people call them.

The network: ~15,800 route-km of Network Rail (ORR); the NR reference-line file has 1,590 ELRs,
17,751 km including freight and some closed. OSM rail track (not heritage) 31,968 track-km, of
which 27,776 (87%) lies under a passenger route relation.

## Sources

| file (data/raw/gb) | what | licence | size |
|---|---|---|---|
| `wd_elrs.csv` | Wikidata, every item with P10271 (ELR): code, description label, P2043 length (miles, from Railway Codes) | CC0 | 3,363 rows |
| `wd_lines.csv` | Wikidata, UK railway-line items with no ELR: label, length normalised to metres, OSM relation, closed date | CC0 | 1,513 rows |
| `wd_heritage.csv` | Wikidata, UK heritage railways (P31/P279* Q420962): labels, used to leave them out | CC0 | 120 rows |
| `wd_stations.csv` | Wikidata, every item with P4755 (CRS code): label, coordinates | CC0 | 2,643 rows |
| `naptan_910.csv` | NaPTAN access nodes, ATCO area 910 (National Rail): `9100<TIPLOC>`, name, coordinates (no CRS) | OGL v3 | 0.55 MB |
| `nr_gis/NetworkReferenceLines.*` | Network Rail's ELR reference lines: one polyline per ELR (British National Grid) with start and end mileage in decimal miles; from `openraildata/network-rail-gis` (EIR release, archived August 2024) | OGL v3 | 9.4 MB |

Not fetched, and why:

- **Network Rail's track model** (VectorLinks, by ELR and track id): current copy only on the
  Rail Data Marketplace (account). The archived GitHub copy (`network-model/VectorLinks`,
  27 MB shapefile) is there if track-level geometry is ever wanted.
- **geofurlong.com**: CC BY 4.0, built on the same track model; per-ELR spreadsheets at 110-yard
  intervals, one sample (REB2) downloadable without the Marketplace.
- **Railway Codes** (railwaycodes.org.uk): every ELR with mileages, no open licence. Wikidata's
  ELR lengths came from it.
- **GB GTFS**: Aubin's `beta.aubin.app/gtfs/great_britain_gtfs.zip`, ~1 GB, rail plus every bus.
  Not used yet: gtfs_served is not on for the UK (see open threads).
- Wikidata P81 ("connecting line") on stations: 972 of 2,635 stations, the WCML 33. Too thin to
  be the station lists.

## What the reader does with them

See `gb_register.py`'s docstring. In short: kr_register's named-track recipe (adopted the way
ca_register adopts us_register), with the UK's names folded, ELRs used to give stretches one
name, unnamed track named from its neighbours, junction ends as junction stations, and station
lists made from OSM's passenger routes (a station goes on each named line the routes stopping
there run on within 250 m). Only `wd_heritage.csv` and the extract are read by the build; the
other files are check and cross-reference material.

## Open

- Eurostar stops only at St Pancras in GB (Ebbsfleet and Ashford closed in 2020). Until
  2026-10-05 it had no gb line at all, since its one stop here could not run on to the border
  point (see "The Channel Tunnel" below); now each route master is a named train St Pancras -
  border. High Speed 1 credits the ride.
- The London - Aylesbury Line is split where the Metropolitan Line's rail-tagged track (Harrow
  - Amersham) is left to the Underground's OSM line; Chiltern rides there credit the Met.
- Line names follow OSM, so some are Network Rail ELR descriptions ("Weaver Junction and
  Liverpool Line", "Deal Street and Edge Hill Line") and some lines are patchworks (the
  Cumbrian Coast Line is 92 of 138 km; the rest is under other names). A NAME_ALIAS table
  could tidy the worst once Anita has seen them on the map.
- `build_tiles` leaves out 2,044 ways on track no line touches (900 rail): freight and heritage
  mostly; not checked one by one.
- No `colours/gb.csv` yet (243 lines carry an OSM colour). No GTFS check (gtfs_served needs
  the 1 GB GB feed and a `uk`/`gb` outline mapping, below).
- Shared-file notes: `tools/build_regions.py` maps religiondots' `uk` only one way (ISO =
  {"uk": "GB"}); a built `gb` falls back to Natural Earth's outline, which works. gtfs_served's
  outline code looks up `cc == "gb"` in religiondots' file and finds nothing (it is `uk`
  there): fix before turning a UK feed on.

## Lines in pieces (2026-10-04)

Anita, 2026-10-04: "theres also lots of discontinuous register lines in uk". Her example, the
Birmingham to Peterborough Line, was Nuneaton - Wigston and Syston - Oakham with the gap at
Leicester, where the track is the Midland Main Line's. Her principle from the US work: a trip
is entered station to station on a line's strip diagram, so a line that cannot be ridden across
its gap is broken as a line.

**Measured** on the build shipped 2026-10-03 (and the same with today's shared code): **47 of
464 register lines were in more than one piece, 107 pieces, 3,747 km on those lines, 992 km of
it outside each line's biggest piece.** Why, line by line (what the fix did in brackets):

| cause | lines | examples |
|---|---|---|
| the gap is another named line's track: OSM gives every way one name, so where a line runs over another's rails its name stops and starts again | 29 (bridged) | WCML over the Trent Valley Line (Rugby - Stafford, 80 km), Birmingham - Peterborough over the MML (Wigston - Leicester - Syston, 12.7) and the Nuneaton and Water Orton Line (8.8), East Coastway over Keymer Junction - Eastbourne (Lewes - Berwick, 11.8), Crewe - Derby over Stafford - Manchester (Kidsgrove - Stoke, 10.9), Cumbrian Coast over "Carnforth Barrow and Carlisle Line" |
| a gap in the line's own track: build_model dropped a junction-ended section as unridden, or the track at a station is not in the route relations | 4 (bridged over own track), plus parts of SWML and Hallam | South Wales Main Line Newport - Cardiff (14.8 km of four-track where routes use two of the four, so its route share fell under 0.5), Hallam Line Chapeltown - Barnsley - Darton, North Wales Coast Line at Chester, South Eastern Main Line at London Bridge |
| a name on the wrong track | 2 (AREA_NAME) | "Great Western Main Line" and "Great Western Railway" on Devon stretches of MLN1 (Exeter - Dawlish, Totnes - Plymouth, Tiverton Parkway); the Bristol to Exeter Line had pieces for the same reason |
| heritage railways Wikidata's labels missed | 2 (left out) | "Rheilffordd Gwili (Gwili Railway)", "Wirksworth Branch Line" (the Ecclesbourne Valley Railway) |
| really different things under one name, mostly Network Rail ELR descriptions that make patchworks with the Wikipedia names | 9 (split) | Carnforth Barrow and Carlisle Line, Didcot and Chester Line, Tapton Junction (Chesterfield) to Colne, Bethnal Green and King's Lynn Line's Ely stub, West Anglia Main Line, "Bangor Line" on 1.2 km at Folkestone, SWML's 2.1 km at Gloucester, Caldervale's Wakefield stub and Normanton and Colton (their links have no passenger route) |
| a 0.2 km junction-to-junction stub | 1 (left out) | London - Aylesbury at Harlesden |

The 15 biggest: West Coast Main Line 628 km (494.8 + 133.4, Trent Valley), West Highland 292
(Oban Line's track at Crianlarich), South Wales Main Line 269 (four pieces: Newport - Cardiff
dropped, Swansea's reversal, Gloucester), Great Western Main Line 251 (Devon names), Chiltern
Main Line 176 (its own "up Cherwell Valley" track at Leamington), North Wales Coast 169 (Chester),
Sheffield to Lincoln 156 (Barnetby), South Eastern Main Line 134 (London Bridge), Brighton Main
Line 93 (Selhurst - Sydenham over London Bridge to Windmill Bridge Junction), Wessex Main Line 93
(West of England Line at Salisbury), Cumbrian Coast 92, Caldervale 86, Bristol to Exeter 78,
Birmingham to Peterborough 74, Bethnal Green and King's Lynn 72 (WAML and Lea Valley Lines).

**What gb_register does now** (`split_pieces`, the build_model hook us_register introduced,
run after `drop_unridden_sections` so it sees the pieces that ship):

1. `bridge_gaps`: for each line in pieces, from its biggest group of pieces, the cheapest track
   to a station of another piece over the passenger track graph (`track_graph`: every way a
   register name is on, and every other way under a passenger route; heritage and the
   Underground's rail-tagged lines left out; a way no route uses costs 4x, so a four-track
   line's relief pair or a freight curve is the last choice; the plain shortest path is tried
   second). A bridge is taken if it is at most 1.5x the crow-fly plus 5 km and under 100 km,
   half of it under a passenger route (bridges under 2 km exempt: platform roads are often not
   in the relations), and its km on other lines' track is under 15 km or under the smaller side
   it joins (a 2 km stray piece of a name is not the line running on 40 km of another's
   track). The bridge is cut at the stations on it of the lines it runs over (Leicester and
   Syston; a station build_model dropped from its line, Wombwell on the Hallam Line, counts),
   and each part is a section of the line.
2. What is still apart becomes one line per piece, as in the US: the biggest piece keeps the
   id, the others get a hash of the line id and their lowest stop id, name_en "Name (first
   stop – last stop)", aliases.json `pieces` so the app moves saved rides. A piece with under
   two stops shorter than 0.5 km is left out, and so is a piece made only of borrowed track.

**Ownership on bridged track.** A section mostly over another register line's track is listed
in the line's `borrowed` (shipped in lines.json). ownership.py gives each way to the nearest
register line whose sections lie beside it and makes the others there "losers", whose sections
then credit the owner for that stretch; a tie on one way goes to the lowest ref, then the name,
which would have handed the MML's unnamed ways to "Birmingham to Peterborough Line". So
gb_register records the ways beside a borrowed section (state["sec_ways"], which ownership
reads) 2 m further off than the nearest other register line that is a candidate for that way
(more than ownership's EXACT_TIE_M 0.5 m, less than its TIE_M 8 m: never nearest, always a
loser), and gives the section that line's high-speed flag there (ownership prefers a line whose
flag agrees with the way's; a slow flag on the WCML's bridge took 32 Trent Valley slow-line ways
before this). No shared file changed. If ownership's EXACT_TIE_M or TIE_M change, BORROW_PAD_M
must stay between them.

Result, from foot.json: of 263.8 km of borrowed sections, **227.8 km credit the line whose
track it is** (the WCML's 81.8 km credit the Trent Valley Line; Birmingham - Peterborough's
21.4 the MML and Nuneaton and Water Orton). 36.0 km credit the borrowing line itself, in two
kinds of place, both logged: ways whose own OSM name is the borrowing line's, which ownership's
name rule gives to it (the Hallam Line's name on MVN2 at Wakefield, which the ELR rule gives the
Caldervale Line, 7.5 km; Huddersfield Line at Mirfield 3.8; Tees Valley Line at Eaglescliffe
2.3), and ways no other register line's section lies beside (30.6 km of way length; an OSM line
or a station throat had them before: Hallam 13.3, WCML 8.7, Borderlands 5.0). Either way each
way still has one owner. **The country's owned total (build_regions.owned_totals) went 16,125.8
-> 16,106.5 km while register line km went 15,291 -> 15,537**: the borrowed km count once, for
their owners; the 19 km drop is the two heritage lines and the Devon renames.

**Before -> after** (rebuilt 2026-10-04): register lines 464 -> 472, 15,291 -> 15,537 km
(borrowed 264 km among them); lines in pieces 47 -> 0 (36 joined, 9 split into 20 lines, the
two heritage lines gone); stations 4,106 -> 4,084 (the Gwili's and Wirksworth's, the Devon
junction ends; Carmarthen moves to an OSM id, aliased). check_model: within 5% 29 -> 30, within
15% 40 -> 41; Great Western Main Line 1.32 -> 1.01, Chiltern 0.98 -> 1.01, Fen 0.93 -> 0.96,
Cumbrian Coast 0.67 -> 0.77, North Downs 0.93 -> 1.08, West Coast 0.98 -> 1.11 (it has the
Liverpool and Edinburgh branches and now the Trent Valley). "Likely register gaps" 194 -> 137
km. The Birmingham to Peterborough Line reads Oakham, Melton Mowbray, Syston, Leicester, South
Wigston, Narborough, Hinckley, Nuneaton (Oakham - Stamford is OSM's "Peterborough to Manton
Junction Line").

**Still open**: Carmarthen and Swansea are reversal stations on spurs, and neither is on the
South Wales Main Line (SWML's Swansea bridge runs over the Swansea Loop past the station;
Carmarthen was a register station only through the Gwili's name and is now on OSM lines only).
A one-stop split piece is named from its junctions ("Tapton Junction (Chesterfield) to Colne
(Junction near Burley Park – Junction near Burley Park)"). The "Bangor Line" name on 1.2 km at
Folkestone is surely a mapping slip (now its own line). Caldervale's Wakefield Kirkgate -
Ravensthorpe track and Normanton and Colton's have no OSM passenger route, so they stay split.

## Lines vs named trains (`rules/gb.py`)

1,061 route=train relations (2026-10-03). Named trains (28 relations): the Caledonian Sleeper
(10), Eurostar (12, as in fr, be, nl), Le Shuttle (2), LNER's Highland Chieftain and Northern
Lights, West Coast Railways' Jacobite; GWR's Night Riviera and Belmond's trains by name if
mapped. Everything else is a line: the TOC routes (Avanti, LNER, GWR, CrossCountry, Lumo,
Grand Central, TPE, EMR, ScotRail's intercity) are interval products, and the Enterprise
(Belfast - Dublin, hourly-ish). OSM's `service=long_distance` (94 relations) decides nothing.

## Heritage railways

159 names, 538 track-km (Overpass; usage=tourism or a Wikidata heritage name). 38 of them have
3 km or more. Left out of the register; their OSM route relations (Ffestiniog, Welsh Highland,
Ravenglass and Eskdale...) stay lines, as in the USA. Most run weekends and school holidays,
some daily in summer (Ffestiniog, Welsh Highland, Snowdon, the NYMR); none runs weekly all
year as far as checked. Question for Anita: leave them as OSM lines (they count), or mark
heritage relations named trains (they do not)?

Two slipped through Wikidata's labels and were register lines until 2026-10-04
(`HERITAGE_EXTRA`): the Gwili Railway, whose track carries "Rheilffordd Gwili (Gwili
Railway)", and the Ecclesbourne Valley Railway on OSM's "Wirksworth Branch Line" (no National
Rail train runs north of Duffield).

## Borders

- **Channel Tunnel**: RINF point eEU00228 (1.50182 E, 51.01718 N, countries fr and gb) is
  mid-tunnel and already in border_points.json; no `borders.EXTRA` entry needed. gb's High
  Speed 1 (with the tunnel's UK half folded in) ends there (section Ashford International -
  border, 49 km). RINF's point lies ~450 m off OSM's tunnels, whose ways are cut at the border
  at 1.4963 E, 51.0150 N; gb_register accepts a border point within 600 m (BORDER_M). Since
  2026-10-05 borders.MOVE puts the point between the bores' cut nodes, and the French half is
  fr_register's "Tunnel sous la Manche" (next section).

## The Channel Tunnel (2026-10-05)

Anita, 2026-10-05: nothing ran from the UK to France on the map. fr_sources.md "The Channel
Tunnel" has the whole account (causes, the French line, Le Shuttle's call). On the gb side:

- **High Speed 1 did own the tunnel's UK half**, both bores, up to eEU00228 (ways.json; its
  section Ashford International - border, 49 km, credits itself whole). The faint grey came
  from fr's tiles, whose extract held the UK's half with no line on it, and from gb's tiles
  drawing France's half the same way. `gb_register.py --clip` (`clip_channel`) takes France's
  half (two bores and the French undersea crossover) out of data/proc/gb; run it after every
  gb extract, and `fr_register.py --clip` after every fr one.
- **Eurostar** (three route masters: Paris, Brussels, Amsterdam) is now a named train St
  Pancras - border here, 138.8-138.9 km, and **Le Shuttle** Folkestone terminal - border, 29.0
  km, once the border point lay on the bores (borders.MOVE): build_model's border tails find a
  point only within 60 m of a route's track, and RINF's was 450 m off.
- **Eurostar began at a Tube station.** Its London stop nodes are named "London St Pancras",
  no station record's name exactly ("London St. Pancras International"), so build_model took the
  nearest record, the Underground's "King's Cross St Pancras". `rules/gb.py`
  `extra_route_stops` moves a train route's stop node off a metro station it reached by
  proximity alone (the metro record's name lacks a word of the stop's) onto the rail station
  within 1.2 km whose name holds all of the stop's words. A first version without the
  name-mismatch test also moved 23 Elizabeth line, Overground and Thameslink stop nodes that
  build_model had matched by name (Barking to Barking Park, Stratford to Stratford
  International), so it was narrowed.
- **High Speed 1 had no London station.** Its 0.3 km section from the throat junction into St
  Pancras International was dropped as unridden: the routes name only a few of the HS1
  platform roads, so under half of the track beside the section was on a route. HS1 began at
  "Junction near London St. Pancras International". `SERVED_END` keeps that section
  (`served_sections`).
- HS1 loses 0.17 km (148.89 -> 148.72) on its border section, now drawn to the moved point.
- Not fixed: three HS1-named ways east of Ashford, 4.5 km (158243849 east of Ashford
  International, 155331186 and 1376434076 west of Cheriton), lie 40-170 m from the line's
  drawn track (the other track where the two diverge), so no register line owns them; Eurostar
  runs over them, so they draw as its named-train track.
- **Northern Ireland - Ireland**: the Belfast - Dublin line (OSM "Great Northern Railway Main
  Line", ref B) crosses at **-6.374718, 54.064810** (between Newry and Dundalk; OSM ways
  395777146 / 396291515 against both religiondots' and Natural Earth's outlines). Ireland is not
  built; when it is, add to `borders.EXTRA`: `("xNewryDundalk", -6.374718, 54.064810, ["gb",
  "ie"])`, and gb_register picks it up (any border point naming gb on a line's track).
