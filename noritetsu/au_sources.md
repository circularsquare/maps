# Australia register sources (built 2026-10-02)

What `au_register.py` reads, where it came from, what it leaves out and why, and how the build
checks out. The download is in `data/raw/au/` (gitignored); this file is the tracked record.
Nothing here needed a login, a key or an account.

## The short answer

- **Lines: Geoscience Australia's Foundation Rail Infrastructure, by line name.** Each register
  line is one name in one state's file as GA aggregates them: NSW's "Main Southern Railway",
  Queensland's "North Coast Line", South Australia's "Port Augusta - WA Border", GA's own
  "Trans Australian Railway" for WA. Named trains (the Ghan, the Indian Pacific, the XPTs) and
  the city lines as OSM maps them (Sydney Trains' T1, Translink's T4) run over them.
- **There is no passenger flag**, so **OSM passenger routes decide** which track is passenger
  track: a station goes on a line only where one of its own routes runs along that line, and
  a section is built only where a route runs over it. Freight-only lines (the Pilbara, the
  Queensland coal systems, the grain lines) have no sections and are not built.
- **Stations: OpenStreetMap**, as for the USA (us_register's `place_stations`, with one change:
  a station goes only on the line its route's track lies nearest).
- **Geometry: OSM track**, traced inside a corridor round GA's line (us_register's
  `OsmTrack.trace`). GA's geometry is close to OSM's: of 60,000 sampled points on OSM passenger
  route track, the distance to GA is median 1.8 m, 90% 10.1 m, 95% 23 m, 99% 63 m.
- **Lengths**: GA's `length_km` over a section is its `chain`. It is the length of GA's own
  geometry, so the chainage check mostly checks the tracing; the outside numbers are from
  Wikipedia (below).

## Foundation Rail Infrastructure

Geoscience Australia, a national aggregation of each state's rail centre lines (NSW Spatial
Services, Queensland's Department of Natural Resources, SA's Department of Planning, Transport
and Infrastructure, Victoria's DELWP, Land Tasmania) with GA's own lines for WA and the NT
("National"). CC BY 4.0 ("© Commonwealth of Australia (Geoscience Australia) 2021").

    https://services.ga.gov.au/gis/rest/services/Foundation_Rail_Infrastructure/MapServer
    layer 1 Railway_Lines (polylines), layer 0 Railway_Stations (points, not used)

`python au_register.py --fetch` pages through layer 1 with `featuresubtype IN (90015, 90016)`
(railways and rail sidings, every status; tramlines 90017 are not fetched) and writes
`data/raw/au/ga_rail_lines.geojson`: **36,243 features, 127.9 MB**, fetched 2026-10-02 by the
managing session.

**Fields.** `name` (the state's line name), `operational_status`, `featuresubtype`, `owner`
(Queensland, SA and GA's lines only), `track_gauge`, `tracks`, `length_km`,
`source_jurisdiction` (NSW, QLD, SA, VIC, TAS, National). **There is no ROUTENAME and no
SECTIONNAME** (multi_sources.md said so; the service has neither), no passenger flag, no line
number and no topology: segments are not tied to numbered nodes, so the graph is made from the
geometry (below).

Operational railway by state (km of `length_km`): QLD 9,438, National (WA, NT) 9,514, NSW
9,060, VIC 4,275, SA 4,104, TAS 667. NSW has 1,768 km named UNKNOWN (most of it the extra
tracks of multiple-track lines, yards and sidings).

### What is left out, and why (the build log's first line)

| left out | km | why |
|---|---|---|
| status Dismantled, Abandoned, removed, Disused, Closed | 16,971 | not there, or not in use |
| sidings (90016) | 1,694 | yard and siding track; taken back where a passenger route runs over them (holes, below) |
| heritage | 324 | by name (TOURIST, HERITAGE, STEAM, PICHI RICHI, ZIG ZAG, ...): Anita's rule counts service more often than about weekly. The Puffing Billy Railway (daily) is kept |
| tram or metro | 69 | Glenelg, "Entertainment Centre Tram", Sydney Metro Northwest, and the Epping - Chatswood railway (Sydney Metro since 2019): OSM lines, as metros are everywhere |
| under construction, proposed | 25 | |

Kept: **22,619 segments, 36,665 km** of operational railway.

**Holes.** GA files some main track as a siding or as dismantled: at Springhurst (Victoria) the
North Eastern standard gauge runs on what GA calls an operational siding, its own line there
being "Dismantled". A left-out segment comes back where an OSM passenger route runs over it
away from kept track (of its part more than 25 m from kept track, at least half of it, 60%
within 60 m of route track; `accept_holes`): **449 segments, 96.6 km** (sidings 89.7,
dismantled 5.9). Of all left-out track, OSM routes run over 662 km of sidings (mostly beside
kept track: Queensland files its passing loops as sidings) and 35 km of "Dismantled".

### Topology (`topology`)

Segment ends within 2 m are one node; an end within 3 m of another segment's middle cuts it
there (570 segments cut); a loose end within 60 m of another piece's loose end is joined to it,
nearest pairs first (193 bridges: the state files meet at the border with no shared vertex,
and NSW's Peterborough - Broken Hill stops 1.1 m short of itself at the SA border). Then a loose
end left over joins the nearest end of a piece it is not yet connected to. 24,009 segments,
21,499 nodes.

### The line unit and names

A line is one cleaned `name` in one state (`group`, us_register's `group_lines` with the state
as the owner): two states' "Airport Line" are two lines, and pieces of one name more than 30 km
apart are lines of their own, told apart by an English name with their two furthest stops
("North Coast Line (Cairns – Roma Street)"). Names are written in title case as the register
spells them ("Main Southern Railway", "Port Augusta - WA Border", "Caboolture/Sunshine Coast
Line"); they are English, so `name_en` is empty unless two lines share a name.

**Names of a piece of track, not a line**: South Australia's file names every siding, yard road
and station track ("Adelaide Station Sidetrack 31", "Dry Creek Freight Yard 05", "Keswick
Passenger Terminal 03", "Goodwood - Adelaide 02"); `TRACK_NAME` reads them as unnamed.

**Unnamed track.** NSW files only one pair of tracks of a multiple-track main line under the
line's name and the rest as UNKNOWN, joined by crossovers (the Main Suburban's six tracks out
of Sydney, the Main Northern's four to Hornsby, the Illawarra's): a segment lying a median 40 m
or less from a named line, nearer it than any other, is that line's track (`adopt_beside`:
4,460 segments, 1,349 km). Of the rest, a run whose two ends both touch one line joins it
(39 km); one under OSM routes for 2 km or more is a line of its own, named from the OSM
route=railway relation its track lies on, else by its end stops (4 lines, 23 km: "Epping –
Mernda" is the Mernda extension, opened 2018); 1,290 km is left out (yards, sidings,
connecting curves).

`operator` is GA's `owner` where it has one (Queensland Rail, Aurizon, ARTC, the Rail
Commissioner, Genesee & Wyoming, Westrail, TasRail). NSW and Victoria give none. These are
GA's 2021-era attributions (Westrail is now Arc Infrastructure and the PTA; G&W Australia's
lines are Aurizon's), kept as the register says.

**323 lines**; 193 have track under an OSM passenger route.

## Which track is passenger track

- **Routes that count** (`is_passenger_route`): OSM route=train relations of a network or
  operator that runs scheduled passenger trains (Sydney Trains, NSW TrainLink, Metro Trains
  Melbourne, V/Line, Translink, Queensland Rail Travel, Transperth, Adelaide Metro, Journey
  Beyond, Transwa, ...), or by name (the Ghan, tagged service=tourism in OSM; the Prospector;
  Puffing Billy and the Kuranda Scenic Railway, which run daily). Not counted: the Gulflander
  and the Savannahlander (weekly), the Vintage Rail Journeys tours, Pichi Richi, Hotham Valley,
  and the closed, freight and proposed lines OSM maps as route=train ("Cathkin-Alexandra",
  "Spinifex Flyer", "Worsley to Hamilton", "Hobart - Boyer (Freight)", "Melbourne Metro 2").
  312 routes over 17,791 ways; 978 stations they stop at, and 172 working train stations no
  route lists taken as stops of the routes passing them (us_register's `unlisted_stations`).
  OSM's route members with role "line" count as track (the Puffing Billy relation uses it).
- A line with under 0.3 km of track under a route is not built; dead-end track no route runs
  over is peeled off a line (sidings, freight spurs), unless another line goes on from that
  end (`peel`; the Trans-Australian's last 52 km to the SA border lie 85-95 m off OSM's track
  and would have been peeled).
- **Stations go on the line their route's track lies nearest** (`place_stations`; a line
  within 10 m of the nearest one also gets it, for shared track): GA has no passenger flag, so
  a freight line beside the passenger line would otherwise take the station too. 1,130
  stations placed, 102 placements refused because another line lay nearer.
- **Sections** prefer track a route runs over (`weighted_sections`: uncovered track costs 4
  times its length), so a section between two stops does not take a shorter freight cut-off
  of the same line. Any section under 25% covered by a route is left out (measured again on
  the traced OSM track where GA's own geometry lies further off).
- A line with no station and under 2 km under a route is not built (43: yards and freight
  track beside passenger lines).

## Sections, junctions and second tracks

us_register's method: an absorbing Dijkstra between stops, line ends and branch points
(junction stations "aj<node>"), sections drawn on OSM track. Three things more, because GA
draws every track:

- **Other tracks' sections dropped** (`dedup_sections`): junction-ended sections lying within
  25 m of a section already kept for 70% of their length are another track of it: 2,362
  sections, 1,082 km.
- **Junctions fused** (`fuse_junctions`): a junction only one line uses, between two of its
  sections, is joined through: 245.
- **Companions** (`find_companions`): a line lying within 30 m of a longer line for 85% of it
  is that line's other track and is declared its `companion_of` (Anita: both directions of a
  line one track): South Australia names the two tracks of Adelaide's lines apart (the Gawler
  - Adelaide line beside the Gawler Line, Noarlunga - Adelaide beside the Seaford Line), and
  Queensland's Caboolture/Sunshine Coast Line, Sunshine Coast Line, Ipswich/Rosewood Line and
  Rosewood Line lie along the North Coast Line and the Main Line. Of two about as long, the
  one named as a line ("... Line") is the main one. 26 declared; companions with no stop of
  their own (South Australia's crossing loops, "Coonalpyn Spur Line") and stopless lines under
  3 km are dropped (13 lines, 24.6 km); 11 companions are in the final model, and ownership.py
  folds 6 of them into their lines (Gawler - Adelaide, Caboolture/Sunshine Coast Line,
  Ipswich/Rosewood Line, Rosewood Line, Exhibition Line, Gawler); the other 5 own nothing
  already, ownership's own single-track rule having given their track to the main line.

## The build (2026-10-02)

Built with the two proposed build_model changes patched in (the "au" `looks_like_service`
branch and platform names read without the platform; tried through a scratch runner that
patches them, since build_model is the managing session's): the reader gives **134 register
lines, 17,167 km, 2,128 stations of which 1,003 are junctions**; 2,182 sections drawn on OSM
track (10 only in a 600 m corridor: GA's NT and WA geometry is further off), 10 on GA's own
(the largest Main Southern Junee - Cootamundra, 56 km). build_model drops 136 junction-ended
sections (97 km) no route runs over: **124 register lines, 17,070 km** (11 of them a declared
companion); 251 lines with OSM's, 3,014 stations; **25 named trains, 21,746 km**. Tiles 5.0 MB.

Track only named trains run over, owned by nobody: 819 km, nearly all the Vintage Rail
Journeys tours' (Stockinbingal - Parkes and on 363 km, Dubbo - Geurie and on 282, the Kandos
line 78, 61 near Summit Tank), as intended: a few times a year. The rest is a few km at
junctions (Avon Yard 6.4, Rockhampton 4.9, Albion 4.3, Broken Hill 3.6).

Not in GA (2021-era data), so OSM lines own the track: Perth's Yanchep extension (34 km), the
Thornlie - Cockburn Line (17 km), the Airport Line tunnel (16 km), the Ellenbrook Line,
Melbourne's Metro Tunnel (Cranbourne / Pakenham / Sunbury through Anzac and State Library),
Sydney Metro City & Southwest.

## Checks

`python check_model.py --region au`:

- against GA's own lengths, 119 lines of 2 km or more: median 0.993, one off by more than 5%
  (Inner City Line, Brisbane, 6.0 of 6.5 km).
- against Wikipedia (`check_model.REGISTER["au"]`):

| line | built | published | ratio | extent |
|---|---|---|---|---|
| Main Southern Railway | 617.7 | 617.8 | 1.00 | Cabramatta km 28.43 - Albury km 646.24 |
| Main Northern Railway | 572.0 | 567.0 | 1.01 | Strathfield km 12 - Armidale km 579 |
| North Coast Railway (NSW) | 682.2 | 683.0 | 1.00 | Maitland km 193 - Queensland border km 876 |
| Orange Broken Hill Railway | 802.9 | 801.0 | 1.00 | Orange - Broken Hill |
| Illawarra Railway | 153.0 | 153.0 | 1.00 | Illawarra Junction - Bomaderry |
| Blacktown Richmond Railway | 26.3 | 25.8 | 1.02 | Blacktown km 34.87 - Richmond km 60.68 |
| Perth Kalgoorlie Railway | 653.0 | 653.0 | 1.00 | East Perth - Kalgoorlie (the Prospector) |
| Perth Mandurah Railway | 71.0 | 70.8 | 1.00 | Perth Underground - Mandurah |
| Belair Line | 22.3 | 21.5 | 1.04 | Adelaide - Belair |
| North Coast Line (Qld) | 1,660.8 | 1,681.0 | 0.99 | Roma Street - Cairns; deviations have shortened it |

- **The network as a whole** (`PATH_CHECKS`, in the build log): the shortest path over the
  kept GA track. Sydney - Perth 3,947.0 km against the Indian Pacific's shortest path 3,961
  (0.996); Port Augusta - Kalgoorlie 1,686.8 / 1,691 (0.998); Strathfield - Armidale 566.5 /
  567 (0.999); Maitland - Casino 610.9 / 612.1 (0.998); Lidcombe - Albury 621.6 / 629.6
  (0.987); Brisbane - Cairns 1,648.4 / 1,681 (0.981); Sydney - Melbourne 935.7 / 961 (0.974;
  the shortest path takes cut-offs the XPT does not). The Trans-Australian as built: 726.1 km
  (WA) + 957.6 (SA) = 1,683.7 against 1,691 (0.996).

## What is off

- **Junctions in the strip.** GA's multiple tracks still leave junctions where a second
  track's piece meets the line: 1,003 junction stations, more than the USA's 883 for a network
  under half the size. Displays read through them ("end of North Coast" in the North Coast
  Line's strip), and a line's display is its longest connected piece.
- **Platform-named stations.** OSM in Melbourne, Sydney and Brisbane names stop positions
  after their platform ("Box Hill 3", "Flinders Street 10", "Central, Platform 23", "Albion
  station, platform 1", "Armidale Station, Platform 1": 1,642 rail stop names) and often has
  no plain-named station node beside them, so build_stations made a station of each: 235 such
  records in the model without the proposed change, 2 with it ("Platform 1", "Tarago Platform
  1 north"). The reader also folds a platform-named stop into a plain-named station within
  600 m (`fold_platform_records`; 9 without the change, none needed with it).
- **Named trains** need build_model's `looks_like_service` "au" branch (proposed). Without it
  the tours own 780 km of track no register line covers as OSM lines, and every XPT, Xplorer
  and Queensland Rail Travel train counts as a line.
- **The Ghan is no named train in the app**: OSM's relation lists one stop, and build_model
  drops a route with under two. Its track counts all the same, through the Tarcoola Darwin
  Railway (NT, 1,684.5 km) and the Darwin Line (SA, 559.6 km), which the reader builds from the
  relation's track. The Gulflander, the Savannahlander and Pichi Richi are dropped the same way.
- **Operators** are GA's, 2021-era, and empty for NSW and Victoria.
- **Tourist lines.** Puffing Billy (daily) is a register line; the Kuranda Scenic Railway
  (daily) runs over the Tablelands System, built Cairns - Kuranda. The West Coast Wilderness
  Railway has no OSM route relation and is not built. Twice- or thrice-weekly heritage lines
  (Mary Valley Rattler, Victorian Goldfields, Zig Zag, Walhalla) are left out by name.
- **Weekly long-distance trains** (the Ghan, the Indian Pacific, the Broken Hill Outback
  Xplorer) count as Anita decided: named trains over lines that count. The Gulflander and the
  Savannahlander (weekly, tourist) do not.
- **Colours**: no colours/au.csv yet; 135 of 256 lines carry a colour from OSM.
- **Timetables**: not used (low priority). gtfs_sources.md lists the state feeds.

## Commands

    python au_register.py --fetch          # GA, about a minute (managing session)
    python au_register.py --ga             # the register alone: lines, km, path checks
    python extract.py --region au --pbf data/raw/australia-latest.osm.pbf     # managing session
    OMP_NUM_THREADS=2 python build_model.py --region au --register au_register:data/raw/au/ga_rail_lines.geojson
    OMP_NUM_THREADS=2 python build_tiles.py --region au
    python check_model.py --region au

On 2026-10-02: build_model 10 min (the reader 9.5 of it, most in sections and branch points
over GA's multiple tracks), build_tiles under a minute, 5.0 MB of tiles (14,736 tiles to z13).
