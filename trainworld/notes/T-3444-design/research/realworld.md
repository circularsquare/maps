# T-3444 realworld notes (web reading only; fetched 2026-11-21)

Caveat: WebFetch summaries come from a small model; Wikipedia pages were truncated at 100k chars in places. Counts from taginfo are live as of 2026-11-21.

## Part A: layouts

### Grand Central Terminal
- Wikipedia: https://en.wikipedia.org/wiki/Grand_Central_Terminal : 67 tracks (56 passenger, 11 sidings) in total, 41 in regular passenger use; upper 30 passenger tracks (28 used), lower 26 (13 used). Platform count conflicts between fetches: "44 platforms, 43 island + 1 side" vs "28 platforms carrying 44 platform numbers" (platform numbers = faces/tracks). Six tracks use a Spanish solution. Odd track numbers usually on the east side of a platform, even on the west. Tracks slope down leaving the station north. A c.1909 caption shows balloon loops on both levels.
- subwaynut: https://subwaynut.com/mnr/gct_platforms/index.php : upper level tracks 11-37 are stub tracks with bumpers, island platforms; 2-10 storage; loop tracks 38-42 (built so long-distance trains could reverse: enter on track 1, loop beneath the station to two islands + a side platform for track 38); lower level 101-115, 15 tracks served by 11 platforms, many tracks have platforms on both sides; 51-65 on Waldorf yard, platforms unused.
- Search summary (secondary, conflicting): lower level had two loops (inside and outside); MTA 110th anniversary: 27 upper + 13 lower passenger tracks. https://www.mta.info/press-release/mta-celebrates-110th-anniversary-of-grand-central-terminal
- Model needs: stub tracks in the hundreds, two levels, island platforms shared by two tracks, loop track that returns to a throat, tracks with no platform (storage), track count != platform count.

### Spanish solution
- https://en.wikipedia.org/wiki/Spanish_solution : platform on each side of one track, one for alighting, one for boarding. Examples: Avenida de America (Madrid), Sant Andreu (Barcelona) [Sant Andreu article is a stub, layout unconfirmed: https://en.wikipedia.org/wiki/Sant_Andreu_station_(Barcelona_Metro)], Munich S-Bahn Karlsplatz (Stachus), NOT Marienplatz (Marienplatz not confirmed anywhere I read), Choa Chu Kang LRT, MARTA Five Points, Paris line 6 Charles de Gaulle-Etoile, Mexico City Chabacano line 8, Brussels Rogier. Prague not found.
- Model needs: a track with a platform on both sides, each side tagged with a role (board / alight), door-opening sequence per side.

### Quad track: by direction vs by line
- https://en.wikipedia.org/wiki/Quadruple_track : paired by direction (fast inner, local outer): Malmo-Arlov, East Coast Main Line (King's Cross-Huntingdon), WCML Nuneaton-Colwich, JR West Kusatsu-Kyoto. Slow in the middle: Stockholm mainline. Paired by service: Midland Main Line, Great Western (Paddington-Didcot), Brighton Main Line (speed), JR East Tokyo-Odawara and Hamm-Minden (passenger/freight), Berlin and Copenhagen S-Bahn (distance), Chatham Main Line Shortlands-Swanley (destination). Chicago pairing not stated there.
- NYC 72nd St IRT Broadway-7th: https://en.wikipedia.org/wiki/72nd_Street_station_(IRT_Broadway%E2%80%93Seventh_Avenue_Line) : 4 tracks, 2 island platforms; outer local (1, overnight 2), inner express (2 by day, 3); cross-platform same-direction transfer.
- Chicago Belmont: https://en.wikipedia.org/wiki/Belmont_station_(CTA_North_Side_Main_Line) (via search; fetch of other title 404) and https://chicago-l.org/stations/belmont.html : 4 tracks, 2 islands; Red on the inner tracks, Brown and Purple (Express, peak) share the outer tracks. Pattern is by line: lines Brown/Purple outside, Red inside, with directional pairing still (each island holds one direction of both). Historical note: originally outside = local, inside = express.
- Model needs: track purpose labels per station (local/express/line), per-track direction, platforms joining track pairs, services assigned to tracks by line not just by direction.

### Stacked directions / levels
- 125th St, IRT Lexington: https://en.wikipedia.org/wiki/125th_Street_station_(IRT_Lexington_Avenue_Line) : bilevel, 4 tracks, 2 islands, one per level, split by direction (uptown upper, downtown lower), local and express both on each level; local doors right, express doors left; flying junction south of station.
- Lexington Av/59th St: https://en.wikipedia.org/wiki/Lexington_Avenue/59th_Street_station : IRT is two stacked levels each with two tracks and two SIDE platforms: upper = local, lower = express (stacked by service); BMT Broadway island platform in between in depth (4 levels in all); same-platform transfer listed on BMT platform; paid passageways; 63rd St lines are a separate station with only an out-of-system free transfer.
- Hong Kong Prince Edward: https://en.wikipedia.org/wiki/Prince_Edward_station_(MTR) : 2 islands, 4 tracks, 2 levels (L2, L3); each level holds one direction of each of Tsuen Wan and Kwun Tong lines in opposite... cross-platform for opposite-direction changes; Mong Kok does the same-direction one.
- Lai King: https://en.wikipedia.org/wiki/Lai_King_station : two islands on two levels; Tung Chung line built as new tracks above Tsuen Wan line; same-level cross-platform; extra pair of tracks lets Airport Express pass; 6 tracks in all.
- Model needs: level per track/platform; island spanning two tracks on same level; side platforms at different levels; two lines on one island.

### Tapering / triangular platforms
- Keilbahnhof (wedge station): https://en.wikipedia.org/wiki/Keilbahnhof : tracks split before platforms, platforms converge one end and diverge the other, a Y shape; examples Lewes, Perth, Sutton (UK), Zurich Oerlikon, Arth-Goldau, Pisa San Rossore, Clybourn (US), many others.
- Lewes: https://en.wikipedia.org/wiki/Lewes_railway_station : 5 platforms in two sets in a V; London branch 1+2, Brighton branch 3-5; building between 2 and 3.
- NOT FOUND: a named station with a single island platform that itself tapers to a point (Anita's Shinjuku guess not confirmed; Shinjuku JR is 8 parallel islands, 16 tracks). Platforms in a Keilbahnhof are separate faces that diverge, so width and track gap change along length. Island platform page: https://en.wikipedia.org/wiki/Island_platform (nothing on tapering).
- Model needs: platform width varying along its length; track spacing varying; platform edges follow each track, so edge is derived from track geometry.

### NYC stations and complexes
- MTA open data "MTA Subway Stations and Complexes", https://data.ny.gov/resource/39hk-dx4f (queried via SoQL): 496 rows, 493 distinct Station ID, 445 distinct Complex ID. Columns: GTFS Stop ID, Station ID, Complex ID, Division, Line, Stop Name, Borough, CBD, Daytime Routes, Structure, lat/lon, North/South Direction Label, ADA fields. So one row = a station on one line-division; a complex groups stations.
- Times Sq complex id 611: stations 317 (IRT Broadway-7Av, GTFS 127, 1 2 3), 467 (Flushing, 725, 7), 11 (BMT Broadway, R16, N Q R W), 468 (Shuttle, 902, S) and 163 (IND 8th Av, A27, A C E, the 42 St-Port Authority). Wikipedia says the Times Sq complex has four stations and the IND one is separate but connected: https://en.wikipedia.org/wiki/Times_Square%E2%80%9342nd_Street_station_(New_York_City_Subway) (MTA data puts it in the same complex).
  - IRT shuttle: 1 island, 2 tracks; IRT B'way-7th: 2 islands, 4 tracks; BMT: 2 islands, 4 tracks; Flushing: 1 island, 2 tracks (deepest, 60 ft).
- Atlantic Av-Barclays complex id 617: station 40 (BMT B'way-Brighton, D24, B Q), 27 (BMT 4th Av, R31, D N R), 338 (IRT Eastern Pky, 235, 2 3 4 5). Atlantic Av on the Canarsie line (L24, station 133, complex 133) is its own complex.
- 14 St-Union Sq complex id 602: station 117 (Canarsie, L03, L), 406 (IRT Lexington, 635, 4 5 6), 15 (BMT B'way, R20, N Q R W). https://en.wikipedia.org/wiki/14th_Street%E2%80%93Union_Square_station : Lex has 2 islands/4 tracks on a sharp curve (gap fillers), BMT Broadway 2 islands/4 tracks, Canarsie 1 island/2 tracks crossing beneath both. Lex and Broadway parallel and offset.
- 125th and 59th: above.
- NOT VERIFIED: MTA GTFS stops.txt layout (my memory: a location_type 1 parent such as R16 with child stops R16N/R16S per direction, not per track). Search did not confirm. Check feed at https://catalog.data.gov/dataset/mta-general-transit-feed-specification-gtfs-static-data

### Passing loops, stubs, express stations
- Passing loop: https://en.wikipedia.org/wiki/Passing_loop : double-ended loop off a single track, often at a station; platforms on one or both tracks; "platform road + through road" (through road straight for non-stoppers; common in Russia) vs "up and down" each track with a platform (British); a dynamic loop is long enough to pass at speed; length matters (longer than trains).
- Scissors crossover: https://en.wikipedia.org/wiki/Scissors_crossover : two overlapping crossovers forming an X with a diamond; compact; page does not discuss termini (known use: terminal stub throats; not sourced here).
- Two island platforms, four tracks, express station: 72nd St above; Kent House (London) - the article fetched gave no layout detail; Island platform page says Kent House pairs islands with fast trains through.
- Model needs: crossovers as track connections with their own track purpose, loops with a length, tracks with both a "through" and "stopping" role.

## Part B: data definitions

### OSM
- Public transport overview: https://wiki.openstreetmap.org/wiki/Public_transport (PTv2, approved 2011, never deprecated old tags). Rail: station/halt node `railway=station|halt` (once per station), `public_transport=stop_position` + `train=yes` + `railway=stop`, `public_transport=platform` + `railway=platform`; trams: `railway=tram_stop`. Route relations list stop, platform alternately.
- https://wiki.openstreetmap.org/wiki/Tag:railway=station : "railway facility where trains stop to load and unload"; one railway=station per station; node (unconnected, middle) OR area, not both; halt vs station differs by country: UK removed halt; Belgium station = staffed; DK/HU/IT/SE/DACH halt = no switches and passenger-only.
- https://wiki.openstreetmap.org/wiki/Tag:railway=platform : way or area (area preferred, area=yes); ref = platform number; local_ref = "number or letter of the stop, platform or other feature"; railway=platform_edge exists for detailed mapping: https://wiki.openstreetmap.org/wiki/Tag:railway=platform_edge
- https://wiki.openstreetmap.org/wiki/Key:railway:track_ref : "reference number of a track within a station, yard or other facility"; for ways only, not nodes/areas; examples usage=main + track_ref=2, service=yard + track_ref=57B.
- https://wiki.openstreetmap.org/wiki/Key:service : yard, siding (parallel and connected, for overtaking or boarding where main tracks lack platforms), spur, crossover (very short, joins tracks of opposite directions); usage=* OR service=* on a track, not both; straight-through station tracks get usage=main etc.
- https://wiki.openstreetmap.org/wiki/Tag:public_transport=stop_area : members by role: stop (stop_position), platform, empty role for station building/area, entrance (railway=subway_entrance/train_station_entrance), amenities, signs. Tags name, ref, uic_ref, uic_name, operator, network. "All members are supposed to be interchanged together." Hierarchy of stop areas for larger interchanges.
- https://wiki.openstreetmap.org/wiki/Relation:public_transport : stop_area_group groups stop_areas of one interchange (multi-modal hub). Member roles of group not on page read.
- railway:preferred_direction: no wiki page found in the fetches (Key:service did not cover it); taginfo shows it exists (below).
- Not read: layer/level wiki pages; stop_area_group member roles; JOSM/OSM validators.
- taginfo counts (2026-11-21):
  - public_transport=: platform 4,223,965; stop_position 1,635,647; stop_area 586,670; station 204,929; stop_area_group 7,123 (all modes; buses dominate platform). https://taginfo.openstreetmap.org/keys/public_transport
  - railway=: rail 2,850,054; platform 196,741; stop 165,931; station 104,127; halt 34,734; platform_edge 15,165.
  - railway=platform co-tags: public_transport any 186,302 (95%); ref 65,306 (33%); local_ref 11,184 (6%); name 62,632 (32%); level 16,638 (8%); layer 16,831 (9%).
  - railway:track_ref: 457,519 objects, 455,649 on ways (vs ~2.85M rail ways = ~16% of rail ways at most; value comparison is rough, rail count includes all usage).
  - railway:preferred_direction: 338,132 objects (338,064 ways).
  - service= on ways: yard 687,533; spur 587,408; siding 385,522; crossover 113,262.
  - type=public_transport relations 593,422.
  - Ratio hints only: 104k railway=station + 34.7k halt vs 586k stop_area (all modes), so stop_area is not a rail-station-specific count. No coverage study found (rail stations covered by a stop_area).

### GTFS
- Reference: https://gtfs.org/documentation/schedule/reference/ (403 to fetch); text read from https://raw.githubusercontent.com/google/transit/master/gtfs/spec/en/reference.md
- stops.txt location_type: 0 or empty stop/platform ("where passengers board or disembark"); 1 station ("a physical structure or area that contains one or more platform"); 2 entrance/exit; 3 generic node; 4 boarding area ("a specific location on a platform"). parent_station: for 0/2/3 points to a station (type 1); for 4 points to a platform; forbidden/empty for type 1. platform_code: "just the platform identifier (e.g. G or 3)".
- A trip names its platform by stop_id in stop_times.txt (stop_id may be a platform, with parent_station giving the station); platform = one stop_id, platform_code is the display label. (Trip-naming statement is from standard GTFS, not quoted from a page.)
- pathways.txt: graph with locations as nodes, pathways as edges; pathway_mode 1 walkway, 2 stairs, 3 travelator, 4 escalator, 5 elevator, 6 fare gate, 7 exit gate; is_bidirectional; traversal_time. levels.txt: 0 ground, positive up, negative down.
- No GTFS term defines track; platform is the finest rail unit besides the boarding area.

### NeTEx / Transmodel
- https://developer.entur.no/concepts/NeTEX/stops : StopPlace "a place comprising one or more locations where vehicles may stop"; Quay: platform, stance, quayside; can have sub-quays; BoardingPosition precise spot (usually trains); parent StopPlace allowed, one level only. Tracks not modelled there.
- Also: ScheduledStopPoint (timetable) is separate from StopPlace, linked via PassengerStopAssignment (Austrian profile notes BoardingPositions not used in EPIP): https://mobilitaetsverbuende.atlassian.net/wiki/spaces/NET/pages/182943748/PassengerStopAssignment ; Swiss guide https://opentdatach.github.io/netexRealisationGuideSwitzerland/06_stops.html (read only via search summary).

### Operators
- MTA: station (one row per line structure) vs complex; see NYC section. 496 rows, 445 complexes.
- Japan: Shinjuku https://en.wikipedia.org/wiki/Shinjuku_Station : one name, five operators, each with own station code (JR East SJK, Odakyu OH01, Keio KO-01, Toei S-01); JR alone 8 islands/16 tracks; whole complex 52 platforms, 200+ exits. Operator-by-operator counting is from the Wikipedia article only.
- Germany: EBO s4(2) via https://de.wikipedia.org/wiki/Bahnhof : Bahnhof = installation with at least one switch where trains may start, end, cross or turn; Haltepunkt = no switch; freight and yards also Bahnhof. Boundary = entry signal. Not found: DB station categories / DB Bahnhof numbering (DS100 vs Bahnhofsnummer).

## Not found
- Prague Spanish-solution station; Munich Marienplatz as Spanish solution (Karlsplatz is the one cited).
- A named single tapering island platform (only Keilbahnhof V/Y shapes).
- A study of OSM stop_area / platform ref coverage for rail; only taginfo counts.
- MTA GTFS stop structure verified from the feed; Barcelona platform detail; Highbury; Queens Blvd; Lexington lower levels beyond 59th/125th.
