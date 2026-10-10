# USA register sources (built 2026-10-02)

What `us_register.py` reads, where it came from, what it leaves out and why, and how the build
checks out. The downloads are in `data/raw/us/` (gitignored); this file is the tracked record.
Nothing here needed a login, a key or an account.

## The short answer

- **Lines: FRA's North American Rail Network (NARN), by subdivision.** Each register line is
  one owner's subdivision as NARN names it (BNSF's La Junta Subdivision, NS's Pittsburgh Line,
  Metro-North's Harlem Line). Amtrak routes are named trains or operating patterns over them,
  as commuter lines are (Anita, 2026-09-30 and 2026-10-02).
- **Stations: OpenStreetMap.** NARN has none. A station goes on a line where a passenger train
  route relation that stops there runs along that line (`place_stations`).
- **Geometry: OSM track**, traced inside a corridor round NARN's own line (`OsmTrack.trace`),
  so ownership finds the ways the trains run on. NARN's geometry is very close to OSM's: of
  60,000 sampled points on OSM passenger route track, the distance to NARN is median 1.4 m,
  90% 6.0 m, 95% 8.1 m, 99% 25 m. 2,223 of 2,240 sections were traced; 17 keep NARN's
  geometry (listed in the build log).
- **Lengths: NARN's own KM** is each section's `chain`, so check_model compares every line
  with the register. Outside numbers from Wikipedia (below).

## NARN Rail Lines

FRA's network, distributed by BTS in the National Transportation Atlas Database (NTAD),
updated 2026-07-21. A work of the US government: public domain.

    https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/services/
        NTAD_North_American_Rail_Network_Lines/FeatureServer/0

The ArcGIS Online copy answers; geo.dot.gov and maps.bts.dot.gov did not on 2026-10-02.
`python us_register.py --fetch` pages through it with
`COUNTRY='US' AND PASSNGR IS NOT NULL AND PASSNGR<>''` and writes
`data/raw/us/narn_passenger.geojson` (35 MB): **18,232 segments, 43,570 km**, fetched
2026-10-02. Only passenger-coded segments are fetched, which matters below (holes).

Fields used: `FRAARCID`, `FRFRANODE` / `TOFRANODE` (the topology: segments meet at numbered
nodes, nothing is snapped; checked, every node's position agrees between its segments to
within 5 m), `RROWNER1` (owner, a reporting mark), `SUBDIV`, `BRANCH`, `DIVISION`, `PASSNGR`,
`NET`, `KM`, `STATEAB`, `TRKRGHTS1-9` (kept on the segment, not used).

`PASSNGR`: A Amtrak 32,242 km, B Amtrak + commuter 2,479, C commuter 4,015, T tourist 3,349,
D Alaska Railroad 775, R rapid transit 335, I intercity high speed (Brightline) 268,
E high speed + commuter 108. `NET`: M main 43,029 km; Y yard, O other, I industry, S siding,
T, X out of service, R: 541 km.

### What is left out, and why (the build log's first line)

| left out | km | why |
|---|---|---|
| `PASSNGR` T, tourist or museum | 3,349 | Anita: completion counts service more often than about weekly; these run on a few days a week in season at most. They stay on the map where OSM has them as lines. |
| `PASSNGR` R, rapid transit | 335 | NARN has only the pieces of light rail on old railroad: DART 68, Sacramento 56, the River Line 50, Sprinter 36, TRAX 35, the San Diego Trolley 53, PATH 27, Staten Island 10. OSM has those lines whole, so they stay OSM lines, as metros do in RINF countries. |
| `NET` not M | 218 | yards 109, other 77, industry 14, sidings 10, out of service 7, R 1 |
| C-coded museum lines | 13 | East Troy Electric Railroad (METW, weekend trolleys in season) and the Cuyahoga Valley Scenic Railroad's 2 km coded C (its other 42 km are T) (`TOURIST_OWNERS`) |
| unnamed track touching no named line | 4 | two yard pieces |

The biggest T lines left out (by owner mark): SNC and ADIX on the Adirondack line 275 km,
NYSW's Main Line 131, DL (Pocono and Carbondale mains) 111, GCRX (Grand Canyon Railway) 106,
CTSR (Cumbres & Toltec) 106, WURR 102, AM 92, CONW (Conway Scenic) 88, GMRC's Bellows Falls 83,
INPR's Thunder Mountain 80, DSNG (Durango & Silverton) 74, MC's Cape Main 69, CWR (Skunk
Train) 68, GSM (Great Smoky Mountains) 64. `python us_register.py --narn` lists them all.

Kept: **16,159 segments, 39,655 km.**

### Names

The line key is `SUBDIV`, else `BRANCH` (unless it says only MAIN, MAIN LINE...), else the
`DIVISION` where the branch is generic (NICTD's track is BRANCH "MAIN", DIVISION "SOUTH
SHORE"). 2,417 km have no `SUBDIV`; after the BRANCH/DIVISION fallback, 188.6 km have no name
at all: 184.5 km of it joins the named line it touches (station and terminal track, mostly
Amtrak, MBTA, LIRR, Caltrain), 4.1 km touches none and is left out.

Cleaning (`clean_name`): the owner's mark in brackets goes ("PITTSBURGH LINE (NS)"); NS's
"DANVILLE DISTRICT MONTVIEW TO SALISBURY" is Danville District; a single main track filed
under its own name ("SHAFTER TRACK 1", "MCCOMB - MAIN TRACK #2", "OTTUMWA MAIN 1",
"PITTSBURGH LINE TRACK 1 AND 2 (NS)") is the subdivision; "P & W" / "P AND W" is P&W; the
LIRR's "LIRR MAIN LINE" / "LIRR ML" (BRANCH) are its Main Line.

Written as railroads write them: " Subdivision" after a SUBDIV name that does not already end
in a kind of line (LINE, BRANCH, DISTRICT, CORRIDOR, SECONDARY...), " Branch" after a BRANCH
name; the MBTA's "(MBTA) FITCHBURG" are its commuter lines, "Fitchburg Line". `NAME_FIX` has
the few the register spells as no railroad would (Oregon Electric, FrontRunner, South Shore
Line, Hell Gate Line, Blue Island / South Chicago / University Park Subdivision, East Rail
Line). Names that are no name (the LIRR's "1", "3", "4", "6"; CSX's "VS-2 MAP 49" and "VS-2
MAP 65", valuation maps for the ex-Pan Am line in New Hampshire; the MBTA's "(MBTA)
CONCURRENT" and "CONNECTOR") take the OSM route=railway relation their track lies on, else
"<owner>: <first stop> – <last stop>".

Grouping (`group_lines`): one name and one owner; another owner's piece under 15 km joins the
line it touches (MBTA's 6 km of the Northeast Corridor); pieces of one name and owner more
than 30 km apart are separate lines with the same name and an English name telling them apart
by their end stops (the Northeast Corridor is Amtrak's in two pieces, Washington - New
Rochelle and New Haven - the Massachusetts line, with Metro-North's New Haven Line between;
also BNSF's Creston, Emporia, Glorieta, KO, CSX's Boston, NS's Washington District, Caltrain's
Peninsula). **428 lines** before the build drops any.

`operator` is the owner (as the infrastructure manager is in RINF): `OWNERS` maps the 60
reporting marks to names.

## Stations, sections, geometry (`us_register.py`)

- **Stations**: every OSM rail station a passenger train route relation (route=train, not
  tourism; the excursion routes OSM tags otherwise are filtered by name) stops at, on each
  line along which one of ITS OWN routes runs there: of the route's track within 800 m of the
  stop, at least 300 m lies within 60 m of the line, and the stop is within 600 m of it.
  Placed where it projects onto NARN; every track of the line within 80 m is cut there.
- **Stations no route lists.** OSM's US route relations are not all complete: the MBTA's
  Fitchburg Line lists 4 of its 18 stops, the Newburyport/Rockport Line one, the LIRR's Oyster
  Bay trains none. So a working train station (train=yes, or public_transport=station on a
  railway=station or halt) within 150 m of a passenger route's track is taken as a stop of the
  routes passing it (`unlisted_stations`): 270 on 2026-10-02. Never a metro or light-rail
  station whatever its train tag says (SEPTA's Market-Frankford stations and the MBTA's
  Andrew carry train=yes). Not every railway=station either: old depots kept as museums often
  still are one. With them, 1,743 stations went on a line; 3 lie near a line none of their
  routes runs along (correctly left off).
- **Sections**: an absorbing Dijkstra over the NARN node graph between stations, line ends and
  branch points. A line end is a node with one neighbour or with all its track leaving one way
  (the end of a double track filed as two segments). A branch point is found from the result:
  two sections that share track for 300 m or more part at a node where three ways meet, which
  becomes a junction, and the sections are found again. Without stations (`--narn`), the 428
  lines are 39,371 km of sections over 39,651 km of track: 279 km of second tracks and loops
  in no section.
- **Junctions** (`uj<NARN node>`) are named for the lines that meet there ("Boise City / La
  Junta"), else the nearest OSM station within 5 km ("near Salisbury"), else "end of <line>";
  the three Canadian crossings (Blaine, Rouses Point, Niagara Falls) "Canada – United States
  border".
- **Station overrides** (`NOT_ON`, 2026-10-03; see "Station overrides" below): the few
  stations no rule places right, by hand.
- **Sections between two stops that no OSM passenger route runs over** (under 25% of them
  within 60 m of a route) are left out by the reader: one, on 2026-10-02 (Amtrak's West
  Subdivision, Long Island City - Hunterspoint Avenue, 0.8 km, 22% on a route); none since the
  station overrides (2026-10-03) took both stations off the West Subdivision.
- **The build** (2026-10-02, with the holes file): the reader gives 495 lines, 40,290 km,
  2,640 stations of which 883 are junctions; 2,579 sections drawn on OSM track, 19 on NARN's.
  build_model then drops 160 junction-ended sections (483 km) it finds no route over, leaving
  **448 register lines, 39,807 km**; 863 lines in all with the OSM lines, 5,722 stations;
  32 named trains (43,520 km).

## Checks

`python check_model.py --region us`:

- against NARN's own chainage, 389 lines of 2 km or more: median 0.996, none off by 5%.
- against Wikipedia (`check_model.REGISTER["us"]`):

| line | built | published | ratio | extent |
|---|---|---|---|---|
| Michigan Line | 380.2 | 373.0 | 1.02 | Porter - Dearborn, 232 mi |
| Hartford Line | 96.2 | 100.0 | 0.96 | New Haven - Springfield, 62 mi |
| Keystone Corridor | 162.2 | 168.3 | 0.96 | 30th St - Harrisburg MP 104.6; NARN's line starts at Zoo |
| New Haven Line | 96.5 | 97.4 | 0.99 | Woodlawn MP 11.8 - New Haven MP 72.3 |
| LIRR Main Line | 147.6 | 151.8 | 0.97 | Long Island City - Greenport MP 94.3 |
| Port Jefferson Branch | 51.7 | 52.6 | 0.98 | Hicksville MP 24.8 - Port Jefferson MP 57.5 |
| Montauk Branch | 168.4 | 171.9 | 0.98 | Jamaica MP 9.0 - Montauk MP 115.8 |
| Anchorage Subdivision | 570.8 | 573.0 | 1.00 | Anchorage - Fairbanks, 356 mi |
| Seward Subdivision | 177.6 | 183.0 | 0.97 | Seward - Anchorage, 114 mi |

- **The network as a whole** (`PATH_CHECKS`, in the build log): the shortest path over the
  kept NARN track between two stations against Wikipedia. Washington - Boston 742.8 km
  against the Northeast Corridor's 735 (457 mi), 1.011 (the shortest path takes the Fairmount
  Line into Boston); Philadelphia 30th St - Harrisburg 164.3 / 168.3, 0.976; Los Angeles - San
  Diego 205.5 / 206, 0.998; Chicago - St. Louis 452.1 / 457, 0.989; Porter - Dearborn 372.3 /
  373, 0.998.
- **The whole register against Amtrak's route-miles**: NARN's A + B codes are 34,721 km;
  Amtrak says about 21,400 route-miles (34,440 km).

## What is off

- **Register holes**, before `narn_holes.geojson`: NARN codes some of the track trains run on
  as having no passenger service: UP's paired track across Nevada, the second track where a
  double track splits (Cajon, Donner, the New River gorge), the Sunset Limited's Houston -
  Beaumont (NARN's passenger-coded 126.5 km Beaumont Subdivision has no OSM route near most of
  it and is dropped as unridden), the Texas Eagle through San Antonio. The holes file fills
  them where an OSM passenger route runs (next section): 182 km of way under named trains
  only is left owned by nobody, from 971.
- **Dropped junction sections.** build_model drops a junction-ended section unless OSM routes
  run over half of it. US OSM maps an Amtrak train both ways over ONE track of a double-track
  line, which halved the share when it was measured against all the line's ways (BNSF's
  Emporia Subdivision at Kansas City, under the Southwest Chief, was dropped at 0.45). Since
  2026-10-02 build_model measures it against the section's own length in the US
  (`ROUTE_SHARE_BY_LENGTH`); 160 sections (483 km) are still dropped, short tails into
  junctions and passenger-coded track no OSM route uses.
- **Named trains**: `looks_like_service`'s `us` branch makes Amtrak's long-distance trains and
  its once-a-day trains named trains (32, 43,520 km); corridors, commuter lines and Brightline
  (hourly) are lines.
- Seasonal and weekend-only services NARN does not code (CapeFlyer, Berkshire Flyer, Winter
  Park Express) stay OSM lines.

## Holes: NARN track with no passenger code that trains run on

`python us_register.py --fetch-holes` writes `data/raw/us/narn_holes.geojson` (the main file
stays as fetched). It reads the OSM extract (`data/proc/us`), takes the passenger route track
(route=train, not tourist) lying more than 25 m from every kept NARN segment (3,093 km of
route track points on 2026-10-02, both directions counted), grids it in 0.05-degree cells,
merges the cells of each row, and queries NARN in each envelope for
`COUNTRY='US' AND (PASSNGR IS NULL OR PASSNGR='')`: 858 cells, 494 envelopes. Spatial, not by
subdivision name: where a route leaves the passenger-coded track, the track it runs on can be
under any name. `--holes-dry` prints the envelope count without fetching.

The reader (`accept_holes`, `group_holes`) takes a fetched main-network segment only where an
OSM passenger route runs over it: of its part more than 25 m from passenger NARN (at least
200 m), 60% within 30 m of route track the register lacks. A segment beside a passenger-coded
one (a third track, a siding) is never taken. Taken pieces that meet the passenger line of
their own name and owner only at its ends fill a gap in it; pieces that meet it where it runs
on are its other track ("Elko Subdivision (second track)"), a line of their own so no section
swaps the passenger-coded track for it; the rest are lines under their NARN name. Tested on
pretend holes (Cajon and part of Elko removed from the passenger file): 189.3 of 193.0 km
taken back, Cajon as its own line, the Elko piece as a gap filled.

Only routes of a named network that stop at two stations or more vouch for a hole
(`hole_routes`, 594 of 664): OSM's US route=train relations with no network are excursion
trains almost all, and on the first trial they brought in the St. Croix Valley Railroad (55
km), the Sacramento River Train, the Santa Cruz Beach Train, the West Virginia Line, the
Beesleys Point Secondary and the Salem Branch. `HOLE_SKIP_NETWORKS` also leaves out Brightline
West (being built), VIA Rail and the New Jersey heritage networks.

Fetched 2026-10-02 by the managing session: 14,092 segments in 494 envelopes, 16.0 MB, 188 s;
2,600 of them main network (3,464 km). Taken: **417 segments, 1,018 km**; 62 more (66 km) only
partly ridden (20-60%) and left out, the largest UP Lordsburg 16.0 km at 51% and CPKC
Alliance 7.4 + 4.5 km.

- **Gaps filled** in a passenger line (275 km): UP Elko Subdivision 214.3 (Nevada's paired
  track; Elko is now 437 km), BNSF Lakeside 16.7, CSX Savannah East Route 18.4, Metrolink
  Redlands 14.6, CN Shelby 12.3, CSX Peninsula 10.1, UP Lafayette 6.4, UP Kerrville 2.5.
- **New lines under their own name**: UP Houston Subdivision 108.1 (the Sunset Limited,
  Houston - Beaumont), FWWR and DGNO Carrollton Subdivision 44.5 + 17.7 (TEXRail and DART's
  Silver Line), NS Kansas City District 26.5 (the Southwest Chief), UP Terminal 12.1 and
  Terminal-Passenger 4.7 (Houston), UP Springfield / East St Louis Terminal 8.4, FEC South End
  6.9, UP Salt Lake 6.1, NS Fort Wayne Line 5.6, and short pieces.
- **Second tracks** ("<line> (second track)", 28 lines, 429.3 km after build_model's drops):
  UP Austin 83.3 (San Antonio), UP Shafter 49.1 and Roseville 48.0 + 11.2 + 1.3, BNSF Gallup
  37.7 + 21.7, Seligman 20.2 + 4.9 + 3.9, Ottumwa 19.4 + 16.1, Cajon 13.2, Needles 9.0 + 6.1,
  NS Pittsburgh Line 20.3 + 4.4 + 3.9, UP Provo 19.4, CSX New River 17.5, CSX Keystone 10.3,
  UP Lordsburg 8.5, UP Lynndyl 6.0, CN McComb 4.8, BNSF Front Range 4.3, and short pieces.

| | before holes | with holes |
|---|---|---|
| register lines | 400 | 448 |
| register km | 38,900 | 39,807 |
| passenger-route rail way owned by a register line | 54,719 of 57,218 km | 55,685 |
| way only named trains run over, owned by nobody | 970.6 km | 181.7 |
| junction-ended sections dropped by build_model | 132 (375 km) | 160 (483 km) |

(Both are trial builds with the landed build_model, 2026-10-02.)

**Open (Anita): second tracks.** A second track is a line of its own, so on paired or split
track each direction of a train credits a different line: the westbound California Zephyr
over Shafter's second track, the eastbound over Shafter itself. Completing both means riding
both ways. 28 lines, 429 km.

## Station overrides (`NOT_ON`, 2026-10-03)

Anita said yes (2026-10-03) to a short explicit list for the stations no rule places right.
Each row is a station's OSM name, a point it must lie within 300 m of, and a register line
(owner mark and cleaned key, e.g. `AMTK`, `WEST`) it is not on, or `*` for a station that is no
stop at all. A row whose name no station near its point carries, or whose line no register line
has, stops the build (`resolve_not_on`, `check_not_on_lines`), so a renamed or moved station
fails loudly. A row that takes nothing off is logged as "override not needed". The build log
lists every placement each row took off.

**How the cases were found.** Every register section with an end at an unlisted station (one
no OSM route lists, taken from `unlisted_stations`) was listed with its km, the station's
network tag, the passing routes' networks, and how far the station lies from route track and
from any track (466 sections, 230 stations on the 2026-10-02 build). Most are real stations
on lines whose relations list few stops (the Harlem Line, Fitchburg, the LIRR branches). The
wrong ones:

| station | was on | why it is wrong | row |
|---|---|---|---|
| Long Island City, Hunterspoint Avenue | Amtrak West Subdivision (LIC - Penn Station 3.2 km, Hunterspoint - Harold 3.3) | LIRR stations beside the East River tunnel approach; no route lists them, so every Amtrak and LIRR Penn train passing within 150 m made them its stop | `AMTK WEST` (they stay on the LIRR Main Line) |
| Cedar Park | Austin Western Main Line (Lakeline - Leander cut in two) | the Austin Steam Train Association's depot | `*` |
| Longhorn and Western Train Platform | UP Austin Subdivision (second track) | the Texas Transportation Museum's miniature railway | `*` |
| SMART Central at Wilsonville Station | Oregon Electric | Wilsonville's bus centre; WES's own station is a separate record 20 m off | `*` |
| Tennessee Central Railway Museum | Nashville Subdivision | museum depot | `*` |
| Smiths Creek Depot | Michigan Line (beside Dearborn) | Greenfield Village's depot on its heritage railway | `*` |
| Paradise | Keystone Corridor (Lancaster - Parkesburg cut in two) | the Strasburg Rail Road's stop | `*` |
| Eagle | Keystone Corridor (Strafford - Devon cut in two) | no railway tag, no SEPTA or Amtrak stop | `*` |
| Marceline | Marceline Subdivision | Santa Fe depot; the Southwest Chief does not stop | `*` |
| Clifton (VA) | Washington District | VRE stops only for the yearly Clifton Day | `*` |
| Fridley, Coon Rapids-Riverdale, Anoka, Ramsey, Elk River, Big Lake | BNSF Staples Subdivision (10 sections) | Northstar stations; OSM has no Northstar route, so only the Empire Builder, which passes them, put them on the line | `*` |

Decided 2026-10-03 (Anita: no factual questions to her, decide and record): the event stops
New York State Fair, North Carolina State Fair and Fairplex are kept (trains stop during the
fair only, two to three weeks a year; kept as seasonal, as seasonal lines are drawn as
running). Northstar's six stations stay off. Lexington NC is a `*` row: neither the Piedmont
nor the Carolinian stops there (a station is planned). Newport News is left as built: the new
Transportation Center (12 km north) and the old stop OSM's Amtrak routes still end at are
both on the line, 12.4 km apart over real track; the old stop is a nameless stop-node record,
which NOT_ON (matched by name) cannot reach, and one stale end is not worth a new mechanism.

**Rebuilt 2026-10-03** with the list: still 448 register lines, 39,811 km (from 39,807);
seven change length by under 1 km (West Subdivision 7.30 -> 8.15 km, now New York Penn
Station - the Main Line junction 7.35 km and Penn - the New York Terminal junction 0.80;
Staples Subdivision 10 sections -> 4); the
others change only where a station came off (Keystone Corridor 33 -> 31 sections, Marceline,
Washington District). check_model: chainage median 0.996, none off by 5%; the Wikipedia
table unchanged. The 13 station ids of the `*` stations carry to nothing (no station within
200 m): a saved ride naming one (only plausible for Northstar) would lose that end.

**OSM lines** (`osm_extra_stops`, waiting on a build_model hook, `extra_route_stops` in
rules/us.py; not in the 2026-10-03 rebuild): the same unlisted stations become stops of the OSM routes passing them, so a
line whose relation lists 5 of its 13 stops gets the rest, with `NOT_ON` applied (a `*` station
on no route; a line row on no route running along that register line there) and three more
limits: the route must have a network (no excursion trains), must not be Amtrak's (its
relations list every stop: what would have been added was the fair halts, Lexington, the new
Newport News and Gilroy for the Coast Starlight), and must share the station's network or
operator where the station has one (NJ Transit's Edison is no stop of the Northeast Regional);
a station within 300 m of a stop the route lists is that stop again. Trial 2026-10-03: 204
stations added to 70 routes; 53 OSM lines change (new: LIRR Oyster Bay and Babylon Branches,
SEPTA Chestnut Hill East and Trenton Lines, NJ Transit Princeton Branch). Lines whose
relations have broken runs still get a few long sections across a gap (MBTA Fall River/New
Bedford: Church Street - Freetown 27 km, across the two branches); that is build_model's
tracing of a relation's gaps, which the new stops make show.

## Lines in pieces, and Penn Station's west end (2026-10-04)

**Lines in pieces.** Anita, 2026-10-04: a trip is entered station to station, so a line whose
sections do not connect cannot be ridden across its gap and should be one line per piece.
`group_lines` still joins pieces of one name and owner within `JOIN_KM` (30 km) first: that is
what lets the holes file fill a gap in a passenger line (`group_holes` looks for the gap's
host by name and owner; the Elko Subdivision's 214 km came in that way). Only what is still
apart at the very end is split.

The very end is after build_model has dropped the junction-ended sections no OSM route runs
over (`drop_unridden_sections`), so the split is a hook build_model calls on the register
module, `us_register.split_pieces` (ca_register exposes the same function). Splitting inside
`build()` was tried first and declined: build_model then dropped some pieces whole, so a
line's old id went to a piece that never shipped (Provo Subdivision (second track): the
11.5 km piece took the id and was dropped, the 7.8 km piece that ships came out under a new
id; also KCT and Lynndyl (second track)), and a line's ways, divided between pieces, matched
differently (the MassDOT Framingham Subdivision's Foxboro and Walpole stubs were both dropped).
**The hook needs a three-line change in build_model.py (the managing session's), so until it
lands nothing is split**; the trial below patched it in at run time.

- **Ids.** The biggest piece (km) keeps the line's id; each other piece takes `piece_id`, a
  hash of the line's id and the lowest NARN segment id under the piece (`SECTION_SEGS`), never
  an index. A piece other than the biggest with under two stops and under `PIECE_MIN_KM`
  (0.1 km) is left out: Creston 0.08 km, KCT's High Line 0.07 (and Canada's four crossover
  stubs at Toronto Union and Charny, 0.13 km in all).
- **Names.** The pieces keep the name; an English name gives their end stops ("Michigan Line
  (Battle Creek – New Buffalo)", "Michigan Line (Albion – Dearborn)"), or their end junctions
  where a piece has under two stops, as the lines more than JOIN_KM apart already were.
  check_model's Michigan Line row now has no operator, so it sums the two pieces (380.2
  against 373).
- **Ways and ownership.** reg_ways and register_way_lines' `sec_ways` are moved to the
  pieces' ids, a way going to the pieces whose sections it lies beside (one beside none of
  them stays with the biggest), so clicks and foot.json see the pieces. A second track folded
  into a split line goes to the nearest piece.
- **Saved rides.** A ride on the old id between two stops of the biggest piece is unchanged.
  One between two stops now on another piece keeps a line id that still exists, so no line
  alias moves it: the app's migration (`migrateRide`) only follows `aliases.json` `lines`
  when the old id is gone. Such a ride stops crediting until the app moves it. The build
  records the pieces (`LINE_PIECES`, {line id: [other pieces' ids]}); with build_model writing
  them into aliases.json as `pieces`, `migrateRide` could move a ride whose two stations are
  not both on its line to the piece that has them. A whole-line ride on the old id comes to
  mean the biggest piece only. In the trial, 25 US lines and 12 Canadian ones have stops that
  move to another piece (the Michigan Line's Albion, Jackson, Ann Arbor, Dearborn; the Provo
  Subdivision's FrontRunner stations; the Long Beach Branch's Garden City stretch; Canada's
  West Coast Express stations on the Cascade Subdivision, Hinton and Jasper on the Edson).

Trial 2026-10-04 (the working tree with the hook patched in, against the shipped build):
**47 US lines split into 102** (register lines 448 -> 503) and **16 Canadian lines into 48**
(99 -> 131); every old id still ships; km unchanged but for the stubs left out and Penn
Station below. The biggest, km by piece:

| line | pieces |
|---|---|
| BNSF La Junta Subdivision | 634.1 + 5.3 |
| UP Elko Subdivision | 430.4 + 6.8 |
| UP Shafter Subdivision | 338.1 + 72.1 |
| Amtrak Michigan Line (CN's track through Battle Creek between) | 191.9 + 188.3 |
| BNSF Kootenai River Subdivision | 363.4 + 7.8 |
| NS Danville District | 369.3 + 0.2 |
| UP Coast Subdivision (Santa Clara - Great America apart) | 315.2 + 6.3 |
| BNSF Fort Worth Subdivision (Fort Worth - Gainesville apart) | 204.5 + 103.5 |
| CN McComb Subdivision (Jackson apart) | 293.2 + 2.9 |
| UP Del Rio Subdivision | 282.0 + 0.7 |

Most US pieces are where the subdivision changes hands for a stretch (Battle Creek), where a
junction-ended stretch no route runs over was dropped between two ridden ones, or a second
track's runs. Two lines come apart only at build_model's drop, not in the reader: SFRTA's
Miami Subdivision (Miami Airport 0.2 km apart) and Creston (a 0.08 km stub, left out).

**Penn Station's west end.** NARN files the North River tunnels' New York side (segments
387399, 389167, 1.07 km, Amtrak, BRANCH "HIGHTSTOWN IT") under the New York Terminal
Subdivision; the New Jersey side, same BRANCH, is the Northeast Corridor. So the New York
Terminal Subdivision ran from the state line in the river through node 489969, where the West
Subdivision from Penn Station ends, on to the Empire Connection's 34th Street junction: one
1.45 km junction-to-junction section, NJ trains over one end and Empire Service trains over the
other, neither over half of it, and build_model dropped it. Penn Station reached neither New
Jersey (the Northeast Corridor stopped at its state-line node, 1.07 km short) nor the Hudson
Line (the Empire Connection stopped 314 m short, at "near 34th Street - Hudson Yards").
`SEGMENT_NAME` now reads the two segments as the Northeast Corridor (the tunnel is the
corridor's; the id is unchanged, its lowest segment being 379940 in Maryland), so the corridor,
the West Subdivision and the Empire Connection all end at node 489969, one junction "New York
Terminal / Northeast / West". Northeast Corridor (New Carrollton – Secaucus Junction) 353.30 ->
354.36 km (Secaucus Junction - the junction 7.05 km), New York Terminal Subdivision 16.74 ->
17.13 km (the junction - 34th Street 0.39 km); nothing else moves (ab.py: 2 US lines differ,
Canada identical). A trip Penn Station -> Empire Connection -> Hudson Line, or Penn Station ->
Newark, now has track.

Tried and declined on the way: cutting a line where another ends inside one of its
junction-ended sections (only where the cut moves no other section). It fixed Penn Station
too, but also changed which halves build_model keeps elsewhere: Austin Subdivision +43.0 km
(San Marcos - the second track's junction), Coast +15.6, Spokane +2.7, Oakland +2.4, and
some halves dropped (Buffalo Terminal -2.9). That may well be right (two trains each running
over half of a section), but it is its own change: each needs looking at.

**One-stop subdivisions** (Anita's question, 2026-10-04). In the trial build, 69 US register
lines have one stop and two or more junction ends, 4,930 km (59 with exactly two, the stop
between them), and 133 have no stop at all (2,450 km): long freight subdivisions Amtrak runs
over with one station (Gallup, Winnemucca on the Nevada, Needles, McCook on the Akron), and
second tracks. That is what a subdivision register gives, and the app's strip diagram now
continues past each junction end, so nothing in the build changes for them. What does stop a
rider is a junction end that leads nowhere in the app: no other line's section ends there and
none passes within 80 m (the app's JUNCTION_SNAP_KM). 13 one-stop US lines have one: BNSF
Lakeside at "Lakeside / Spokane" (the Spokane Subdivision's short section there was dropped),
Marceline and East End at their "end of" nodes, Glenwood Springs near Grand Junction, Elko's
ends, Metrolink's River Subdivision near Chinatown, Dallas, Prosper (Fargo), Sanford, Norcross
(Atlanta). They are register gaps of the kind the holes file and the dropped junction
sections leave, each to look at on its own; not changed here.

## Route relations in no order, and New Providence (2026-10-08)

Anita's notes of 2026-10-08: the NJ Transit Morris & Essex Lines' strip diagram had Dover,
Denville, Convent Station and Mount Arlington on side lanes and the Gladstone Branch's stops
interleaved with the Morristown Line's. The diagram was right for the data; the data was wrong.
NJ Transit's old both-ways relations ("Morristown Line: New York <=> Hackettstown", "Gladstone
Branch: New York <=> Gladstone") list their ways in no order, so build_model joined them into
79 and 231 runs and read the stops in run order, and each pair consecutive across two runs
became a section over the shortest track: Dover - Mount Tabor past Denville, Mount Arlington -
Denville past Dover, Denville - Convent Station, East Orange - Hoboken. The line came to 246.8
km for about 145 km of route. 23 US train relations are this broken (4 or more runs holding
stops): most NJ Transit lines, Metro-North's Hudson, Harlem and New Haven, the LIRR's Port
Jefferson, Oyster Bay and Port Washington, several MBTA and SEPTA ones.

`repair_route_runs` (through rules/us.py's `route_runs`, a build_model hook proposed in
handoff_notes/njt_morristown.md, not yet called) rebuilds such a route's runs from its own
track: each track node goes to the nearest stop along the track, stops whose regions touch are
neighbours, the cheapest neighbours joining all stops are the line (a spanning tree), and the
tree is walked as one run out along each branch and back. Where the relation lacks a way (the
Morristown relation stops 450 m short of the Hoboken approach), the two parts are joined at
their nearest loose ends and that pair of stops is left to build_model's gap tracing, as
before; ends under 30 m apart are taken as track.

Trial (2026-10-08, the hook patched in at run time, against the shipped build): 18 OSM lines
change, none of their ids, no station id; register lines unchanged but the Morristown Line
(New Providence, below). Lengths now near the published routes: Morris & Essex 246.8 -> 148.5
km, Montclair-Boonton 180.8 -> 98.9 (Hoboken - Hackettstown 60 mi), North Jersey Coast 181.5 ->
107.0 (66.8 mi), Hudson 252.8 -> 117.2 (74 mi), Harlem 160.5 -> 131.6 (82 mi), New Haven 298.0
-> 178.7, Port Washington 51.7 -> 31.7, Haverhill 95.2 -> 53.3 (33 mi), Providence/Stoughton
156.8 -> 107.3. Commuter OSM lines with a section passing one of their own stops: 15 -> 3 (the
LIRR Hempstead and West Hempstead Branches and SEPTA Manayunk/Norristown, express track, which
the diagram hides as before). Three SEPTA lines grow where their partial relations now reach
Center City across a gap traced over other track with no stop on it (Chestnut Hill West
Suburban Station - Queen Lane 11.6 km, Trenton Suburban Station - Bridesburg 16.1, Cynwyd
Suburban Station - 30th Street 4.5 km where 1.5 km is right: build_model's network trace from
the stop node goes round).

**New Providence** was on the Morristown Line between Chatham and Summit (3.1 + 2.3 km). It is
a Gladstone Branch stop; the Morristown Line passes 560 m off where the two part west of Summit
(inside STATION_M), and the broken Gladstone relation lists a Morristown Line way there. A
NOT_ON row takes it off (NJT MORRISTOWN LINE): Chatham - Summit is one section again, 58.51 km.

## South Station, Tucson, and kinds (2026-10-09)

Anita's notes of 2026-10-09 (handoff_notes/boston_tucson.md has the app half and the numbers):
the East Subdivision's diagram opened with Newmarket, JFK/UMass and the Red Line ahead of
South Station, and the Lordsburg Subdivision's had Tucson twice.

- **South Station's Old Colony approach** (segments 379801, 380038, 374683; 1.1 km, coded C)
  is filed under SUBDIV "EAST". `SEGMENT_NAME` reads it as the Old Colony's: every Old Colony
  train runs over it, and no East Subdivision train does. East is now South Station - Back Bay
  - Attleboro (56.08 -> 54.54 km), and South Station became the Old Colony's own end stop
  (South Station - JFK/UMass 3.71 km; 16.66 -> 18.23 km). The Fairmount Line still ends where
  it leaves, node 495012, now mid-section on the Old Colony.
- **A stop a few metres short of its line's dead end** (`TWIN_STUB_KM`, 50 m): South Station
  stood 5 m before NARN's end of track, which made a junction end "near South Station", and
  the app offered every line near the station from it. Such a stub is left out where no other
  line meets the node and it is no border. 20 in the US (South Station on East and on the Old
  Colony, Rockport, Newburyport, Needham Heights, Gladstone, Elburn, Seward, South Bend
  Airport, Downtown Carrollton, Chestnut Hill West, Greenbush...). The MBTA's 0.19 km "South
  Side Subdivision" at Greenbush, which was little more than such a stub, no longer ships.
- **A stub that is the start of the line's own second track** (`fold_second_track_stubs`):
  between Vail and Benson, UP's second main runs on its own alignment 0.1-1.1 km from the
  first. NARN codes its first 13.1 km as passenger, so it was a Lordsburg section ending 48 m
  from Lordsburg's other main track, and the rest not (the holes file's "Lordsburg Subdivision
  (second track)"). The app found no way on from that end but the Sunset Limited, so it
  offered Tucson there, which the line's other end already reaches on the Gila Subdivision.
  Such a stub (stop-less, from a branch point of the line to a dead end where its own folded
  second track carries on, within COMPANION_KM) now goes into that second track, and
  ownership gives its ways to the line: Lordsburg 506.57 -> 493.45 km (its second track
  13.12 km, owning nothing), Pittsburgh Line 0.30 km, Gallup 0.15, Cajon 0.12. Left alone:
  the Elko Subdivision's paired track, which also has long stubs ending on its own track
  (151.8 km and 55.2 km, the first 3.2 km from the line for 90% of it), offering Winnemucca on
  the California Zephyr. It needs its own look.
- **Kinds** (`REGISTER_KIND_SURE` in rules/us.py; needs the build_model change in the note):
  `register_way_lines` called the Old Colony Line "subway" from the Red Line beside it, so it
  owned Red Line track and the same-kind walk could not reach JFK/UMass on it. NS's Amtrak
  Connection at Cleveland came out light rail. Until build_model reads the flag, the Old
  Colony stays "subway".

## Commands

    python us_register.py --fetch          # NARN, about a minute
    python us_register.py --narn           # the register alone: lines, km, path checks
    python extract.py --region us --pbf data/raw/us-latest.osm.pbf     # managing session
    OMP_NUM_THREADS=2 python build_model.py --region us --register us_register:data/raw/us/narn_passenger.geojson
    OMP_NUM_THREADS=2 python build_tiles.py --region us
    python check_model.py --region us

On 2026-10-02: build_model 7.5 min (the reader is about 5 of it), build_tiles 4 min, 38.7 MB
of tiles (110,401 tiles to z13).
