# New Zealand register sources (built 2026-10-03)

What `nz_register.py` reads, where it came from, what it leaves out and why, and how the build
checks out. The downloads are in `data/raw/nz/` (gitignored); this file is the tracked record.
Nothing here needed a login, a key or an account.

## The short answer

- **Lines: KiwiRail's own line register** (KiwiRail open data, CC BY 4.0): the line names
  ("North Island Main Trunk", "Wairarapa Line", "Melling Branch"), a km post every half or whole
  km of every line, and 920 named locations on the lines. Written into rinf.py's input files
  (as id_register.py and ua_register.py do) and traced over OSM track by rinf.py.
- **Only the stretches with scheduled passenger trains are register lines** (`SCOPE`); the rest
  of KiwiRail's 3,700 km is freight only and is not built.
- **Stations: OpenStreetMap**, matched by name to KiwiRail's Station/Passenger locations, plus
  every OSM station an OSM train route stops at (`osm_stops`).
- **Section lengths: KiwiRail's km posts.** A location's km is where it lies along its line's
  posts, to the metre; a section's `chain` is the difference.
- 15 register lines, 1,455 km; 32 lines in all (17 OSM lines, 5 of them named trains), 157
  stations. Every register line within 2% of its Wikipedia length.

## KiwiRail open data

data-kiwirail.opendata.arcgis.com (also harvested by data.govt.nz), licence CC BY 4.0 ("This
data is made available in good faith but its accuracy or completeness is not guaranteed").
ArcGIS feature services under
`https://services6.arcgis.com/eqX2HMCD3H8MUs6Q/ArcGIS/rest/services/`, fetched 2026-10-03 by
`python nz_register.py --fetch`:

| file | service | what |
|---|---|---|
| `network.geojson` | NZ_Rail_Network/0 | 50 features, one per line ("North Island Main Trunk", code 1), every track drawn, no status; 5,300 km of geometry |
| `kmposts.geojson` | KiwiRail_KM_Posts/0 | 4,209 posts: KM, LINE, LINECODE, LINE_ABBRV (NIMT, NAL, WRAPA...) |
| `locations.geojson` | RailwayLocations/0 | 920 places: NAME, Km_ref (whole km only), LineCode, Status (Station/Passenger 109, Operational 149, Yard 29, Place 232, Historic 401) |
| `metro.geojson` | MetroServices/0 | Auckland and Wellington services per section, summer 2023/24 timetable (not used by the build; read to confirm which KiwiRail lines the metro services use) |

Also there and not used: KiwiRailTrack (every track asset by type), FreightStatus_line, the
level crossings, bridges, tunnels.

**The posts are clean**: one post per km value on every line (787 on the NIMT, at every half km
in the cities), no missing km; two short kms on the NIMT (km 274 - 275 is 0.23 km on the ground,
357 - 358 0.26 km: deviation equations), so KiwiRail's own NIMT chainage reads 1.5 km longer
than the track. The Wairarapa Line's posts start at km 1.8 (Wellington - Distance Junction is
the NIMT's).

**The locations need four fixes** (`REASSIGN`): KiwiRail files Manukau station under the NIMT at
the junction's km (it is the end of the Manukau Branch), the Strand under the NIMT (it is the
end of the Newmarket Branch at Quay Park), Greymouth under the Hokitika Line (it is km 211 of
the Midland Line), and Christchurch station under the Main South Line at km 10, the old station
(the station at Addington lies on the Main North Line's first 400 m). 20 locations lie more
than 400 m off their own line's posts and are left out (the Waitoa Branch's old Thames line
stations, Whakatane on the Taneatua Branch); none is on a line in scope. Two places are a
line's passenger terminus but not Station/Passenger in KiwiRail's list (`STOP_FIX`): the Strand
(a "Place"; Te Huia's and the Northern Explorer's Auckland station) and Dunedin
("Operational"; Dunedin Railways' station).

### LINZ's NZ Railway Centrelines (Topo50), measured and not used

`https://data.linz.govt.nz/layer/50319-nz-railway-centrelines-topo-150k/`, CC BY 4.0, 575
features, last published 2026-09-30, fields `name`, `name_ascii`, `macronated`, `status`,
`track_type`, `rway_use`, `veh_type` (read from the open metadata API,
`/services/api/v1.x/layers/50319/`). It names the lines too, but has no chainage, no stations on
the lines and no line codes, and the geometry needs a LINZ account's API key (WFS or an export).
KiwiRail's data has everything it has plus the km posts and the locations, under the same
licence, with no key, so KiwiRail's is the register.

## The line unit and which stretches

A line is one KiwiRail line, its RINF id KiwiRail's abbreviation, named by KiwiRail's name
(`rinf_countries/nz.py`'s `id_name`). KiwiRail now files the City Rail Link as the NIMT's last
km (Waitematā km 681.9 - Te Waihorotiu - Karanga-a-Hape - Maungawhau km 685), as Wikipedia does
(684.98 km Wellington - Maungawhau), and OSM names those ways "North Island Main Trunk"; the
CRL is open in OSM (AT's East-West, South-City and Onehunga-West lines run through it).

`SCOPE`, with the trains behind each stretch (2026-10 timetables and OSM's route relations):

| line | stretch | trains |
|---|---|---|
| North Island Main Trunk | Wellington - Maungawhau, whole | Kapiti Line, Capital Connection, Northern Explorer, Te Huia, AT's lines |
| North Auckland Line | Westfield junction - Swanson | AT's lines; north of Swanson freight only since 2009 |
| Newmarket Branch | Newmarket - Parnell - the Strand | AT, Te Huia, Northern Explorer |
| Onehunga Branch, Manukau Branch, East Link | whole | AT |
| Wairarapa Line | km 1.8 - Masterton | Hutt Valley Line, Wairarapa Connection; Masterton - Woodville freight |
| Melling Branch, Johnsonville Line | whole | Metlink |
| Main North Line | Christchurch - Picton | the Coastal Pacific (seasonal: drawn as running, Anita's rule) |
| Main South Line | Christchurch - Rolleston; Dunedin - Wingatui | the TranzAlpine; Dunedin Railways' Taieri Gorge train |
| Midland Line | Rolleston - Greymouth | the TranzAlpine |
| Taieri Branch, Taieri Gorge Railway | Wingatui - Taieri - Pukerangi | the Taieri Gorge train |

Left out as freight only: the rest of the NAL (to Otiria), the East Coast Main Trunk and its
branches (Kinleith, Mount Maunganui, Murupara), the Palmerston North - Gisborne Line, the Marton
- New Plymouth Line, the Stratford - Okahukura Line (mothballed; Forgotten World Adventures' rail
carts are no train), the Main South Line south of Rolleston except Dunedin - Wingatui, the
Hokitika, Stillwater - Ngakawau and Rapahoe lines, the Ohai and Bluff lines, Lyttelton -
Christchurch, the Gracefield and Hornby branches.

### Heritage and tourist trains (the project's rule: scheduled more than about weekly)

- **Dunedin Railways' Taieri Gorge train counts**: "The Taieri Gorge departs from the Dunedin
  Railway Station Thursday-Monday, 9:30am", to Pukerangi (dunedinrailways.co.nz/taieri-gorge/,
  read 2026-10-03). So Dunedin - Wingatui (MSL), the Taieri Branch and the Taieri Gorge Railway
  to Pukerangi are register lines, as Australia kept Puffing Billy and the Kuranda line (daily).
  The Taieri Gorge Railway (Dunedin City Council's, no KiwiRail code or posts) is written by
  hand (`HAND`), its km measured along KiwiRail's geometry of it; Pukerangi - Middlemarch (19 km,
  no passenger train now) is left out. OSM calls the station "Puketerangi"; its name stands.
- **Not counted**: the Seasider (Dunedin - Waitati, "returns summer 2026", no days set), the
  Victorian / Seasider Full Day to Oamaru (event days only), the Glenbrook Vintage Railway
  (Sundays in season), Goldfields, Weka Pass, Bay of Islands Vintage, Gisborne City Vintage,
  Driving Creek (a daily ride up a hill and back at a pottery park: an attraction, not a line,
  and no OSM route). Their OSM routes are named trains (no percentage); the build drops most of
  them anyway for having under two stops.
- **Kept as OSM lines** (tram and funicular routes cannot be named trains): the Wellington Cable
  Car (Metlink), Christchurch's City Heritage Tram Loop (daily), MOTAT's Western Springs
  Tramway (daily). **Stale**: "Wynyard Loop" (Auckland's Dockline Tram, closed 2018) is still an
  OSM tram route with four stops and is built as a running line (1.1 km); it needs the hook
  for stale OSM routes HANDOFF lists as open.

### Named trains (`rules/nz.py`'s `looks_like_service`)

Lines: everything on network AT, Metlink or BUSIT (Te Huia, network "AT;BUSIT": two to four
trips a day Hamilton - Auckland, AT HOP fares, a commuter service a rider uses as a line).
Named trains: the rest, i.e. the Northern Explorer, the Coastal Pacific and the TranzAlpine
(KiwiRail Scenic's single long-distance trains), the Capital Connection (one return trip each
weekday, a single train: as Croatia's single "B" trains and the XPTs), Dunedin Railways'
excursions and the heritage routes. Their track counts through the register lines.

## Points, sections, junctions

- Every location in scope is a point on its line: Station/Passenger typed a stop ("10"), the
  rest "80", merged away by rinf.py unless lines meet there. 351 section rows, 351 points.
- A line's ends: the location within 300 m of its first or last post (Wellington, Penrose,
  Maungawhau, Swanson...), else a junction point at the post (`ENDS` overrides: the Midland
  Line starts at Rolleston, 330 m from its km 0). A junction end goes on the line it leaves
  (`JOIN`, by hand: by nearness alone the Wairarapa Line's junction landed on the Johnsonville
  Line, which shares its first 1.5 km of posts with the NIMT, and the Johnsonville Line lost
  Wellington - Crofton Downs). Shared points are listed in names.json "cut" and rinf.py ends
  sections there (`cut_at_junctions`).
- **Stops**: 112 KiwiRail stops all matched an OSM station (3 by distance alone: Mt Albert,
  National Park, Takanini); **Levin** has no OSM station node in the extract and is placed at
  KiwiRail's point (`stop_name`, `AT_KIWIRAIL`). `osm_stops` added Paerātā, Drury, Huntly,
  Rotokauri and Taumarunui (stops OSM's routes list that KiwiRail does not flag).
- **Names**: OSM stop positions are named after their platform ("Maungawhau 3", "Te Waihorotiu
  1", "Hamilton Frankton 2", "Petone Station", "Ngauranga - Platform 1"; Auckland's stations
  are mapped only as stop positions). `rules/nz.py`'s PLATFORM_SUFFIX reads them without it, in
  build_model and (through the new rinf.py hook `plain_name`) in rinf.py.
- `way_line`: OSM's ways carry their line's name (99.5% of main and branch km,
  probe_kr_ways), so a line's pass-2 trace prefers its own ways.
- The two Main South Line pieces are two lines of one name, told apart by an English name with
  their ends: "Main South Line (Christchurch – Rolleston)", "(Dunedin – Wingatui)".

## The build (2026-10-03)

rinf.py: 351 sections traced, none rejected, none traced end to end, none "length off"; 15
lines, 1,455 km traced against 1,463 km of KiwiRail chainage; 126 section ends, 9 not stops.
build_model: 21 junction-ended sections (95 km) kept, none dropped; 32 lines, 157 stations,
3,352 route-km, 5 named trains (1,462 km). Tiles 0.8 MB. About 35 s for the model.

| line | built km | KiwiRail km | ends |
|---|---|---|---|
| North Island Main Trunk | 679.5 | 685.0 | Wellington - Maungawhau |
| Main North Line | 345.9 | 347.4 | Christchurch - Picton |
| Midland Line | 210.2 | 211.2 | Rolleston - Greymouth |
| Wairarapa Line | 88.9 | 89.2 | km 1.8 - Masterton |
| Taieri Gorge Railway | 41.9 | 41.9 | Taieri - Pukerangi |
| North Auckland Line | 32.0 | 32.1 | Westfield junction - Swanson |
| Main South Line (Christchurch – Rolleston) | 19.4 | 19.4 | |
| Main South Line (Dunedin – Wingatui) | 12.6 | 12.0 | |
| Johnsonville Line | 10.3 | 10.5 | Wellington - Johnsonville |
| Onehunga Branch, Taieri Branch, Newmarket Branch, Melling Branch, Manukau Branch, East Link | 3.3, 3.1, 3.0, 2.5, 1.7, 0.4 | 3.3, 3.0, 3.0, 3.0, 1.7, 0.3 | |

Track only named trains run over, owned by nobody: 23.7 km of way length: Pukerangi -
Middlemarch 19.0 (the Inlander's OSM relation runs on to Middlemarch; the train turns at
Pukerangi), 2.2 at National Park (the NIMT is traced through the platform loop, the Northern
Explorer's relation on the main beside it), 1.3 near Mosgiel (the Inlander), 1.2 past
Greymouth station (the TranzAlpine's relation runs on into the yard).

## Checks

`python check_model.py --region nz`:

- against KiwiRail's chainage, 13 lines of 2 km or more: median 0.997; one off by more than 5%,
  the Melling Branch (2.5 of 3.0): OSM's Melling stop position is 450 m short of KiwiRail's
  Melling point (the end of the track).
- against Wikipedia (`check_model.REGISTER["nz"]`), worst 0.02:

| line | built | published | ratio | extent |
|---|---|---|---|---|
| North Island Main Trunk | 679.5 | 684.98 | 0.99 | Wellington - Maungawhau (KiwiRail's posts carry 1.5 km of short kms) |
| Main North Line | 345.9 | 348.04 | 0.99 | Addington - Picton |
| Midland Line | 210.2 | 212 | 0.99 | Rolleston - Greymouth |
| Johnsonville Line | 10.3 | 10.49 | 0.98 | Wellington - Johnsonville |
| Wairarapa Line | 88.9 | 89.16 | 1.00 | Masterton km 90.96 less the 1.8 km shared with the NIMT |
| Taieri Gorge Railway | 41.9 | 42 | 1.00 | Pukerangi 19 km short of Middlemarch on the 60 km line, from Taieri |

## What is off

- **OSM's Kapiti Line relation (1171504) is PTv1** (forward/backward ways, stops under
  "forward:stop" / "backward:stop" or no role), so build_model reads two of its stops and the
  OSM line is "Wellington - Tawa", 21 km. Giving it the others through `extra_route_stops` was
  tried and dropped: its ways assemble into runs out of order and the line came out 108 km
  (Paremata - Paraparaumu 26 km), and the Johnsonville Line and Capital Connection relations
  broke the same way. The NIMT register line carries the track; the relation needs fixing in OSM.
- The Capital Connection's relation lists only Palmerston North, Shannon and Wellington; the
  Coastal Pacific's lists Rangiora twice. Named trains, so only their stop lists suffer.
- Wynyard Loop (closed 2018) is built as a running tram line (see above).
- No `colours/nz.csv`: 12 of 32 lines carry OSM's colour (AT's and Metlink's lines); KiwiRail's
  register lines have none.
- **Timetables**: AT and Metlink publish GTFS, but each covers one city, and gtfs_served would
  judge the NIMT's, the MNL's and the Midland Line's long sections outside them as unserved
  (HANDOFF thread 0's scope problem). Not proposed until gtfs_served judges a feed only within
  its own area; with only passenger stretches in the register there is little for it to find.
- No borders.

## Commands

    python nz_register.py --fetch          # KiwiRail's four layers, a few seconds
    python nz_register.py --report         # the register alone: lines, km, shared points
    python nz_register.py --show NIMT      # one line's pieces with their km
    python extract.py --region nz --pbf data/raw/new-zealand-latest.osm.pbf   # managing session
    python tools/slot.py -- python build_model.py --region nz --register nz_register:data/raw/nz
    python tools/slot.py -- python build_tiles.py --region nz
    python check_model.py --region nz
