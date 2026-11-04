# For Anita

## Open now (2026-11-04, maps-c9)

Answer "text acknowledged" per block, or name the lines to change.

### Tutorial (T-2263): your rewording is in build 868
One change to your wording: the Traffic note says "Red borders indicate track at full capacity." (the red
borders in Traffic view mark track full of trains, not crowded trains). The green-bubble note is dropped.

### Track and station bars, mode frame (T-2267 + T-2270; built 2026-11-04 as T-2282 to T-2284 with your changes, T-2285 not built)
Note: notes/T-2267.md. Shots: artifacts/trainworld/review/track-bar.png, mode-colours.png.
Questions: build it as T-2282 to T-2284? the optional cross "Stop (Esc)", the bars' 3 px top edge in the mode
colour, and Shift+1-7 re-levelling (T-2285)? keep the mode colours though they sit only 11-15 dE from the game's
own? (I'd say yes to all.)
- Track type: "Double / Quad / High speed" -> "Regular / High speed"
- Build tab Terrain button and its hover "Show terrain. Steep terrain costs more to build on." -> gone
- New descriptors: "Placing track", "Placing stations", "Deleting", "Selecting track and stations", "Exploring a trip", "New passage from Bedford Av"
- Kept descriptors, gaining a colour square: "New bus line: 3 points", "Route of Bus 3: 3 points", "New ferry line: 2 ports", "Editing stops on Z: 22 stops", "New station building: 4 corners", "Station building outline: 4 corners"
- Optional cross, hover: "Stop (Esc)"
- Level hover (T-2271's): "Viaduct (Shift+3)" / "1.0x cost, $75M per km" / "Upkeep $20k per km per day", 1x on light yellow

### Fares (T-2266, done)
- Optional hover on the fare controls, not built: "Riders pay the old fares until demand has settled". Want it?

### Ferries (T-2273, T-2274, T-2275, done overnight)
The tool's tag no longer flickers; the leg from the last port to the pointer is dark blue (light blue in dark
mode); new ferry lines start at 20 min, 30 min, none. In the Traffic view, buses and ferries are drawn stretch by
stretch by riders, on the same scale as rail (so New York's ferries draw thin). In a ferry's port list, click a
port's dot to remove it, drag a row to reorder; both undo. Shots: artifacts/trainworld/ferries/tool-light.png,
traffic-light.png.
- Hover on a port's dot, new: "Click to remove the stop at X" (the train panel's wording)
- Refusal, new: "A ferry line needs at least two ports."

### Build tab (T-2269, T-2271's level hover, done overnight)
The breakdown sits under the blueprint total, collapsed; parts (track joined by blueprint track) biggest first,
each with its stations and stretches in track order; a checkbox on each row switches it off (grey on the map, out
of the total and Construct, kept in saves); hover highlights it on the map, click moves there. Shots:
artifacts/trainworld/buildtab/hover.png, level-hover.png.
- Level hover: "Viaduct (Shift+3)" / "Upkeep $20k per km per day" -> "Viaduct (Shift+3)" / "1.0x cost, $75M per km" / "Upkeep $20k per km per day"; 1x on light yellow (also on the map's track tag)
- "Blueprint cost breakdown" -> "Cost breakdown"; "No blueprint to construct." -> removed
- New: "By kind"; "Track" (a part with no stations); "X to Y", "X to a junction", "X to a track end", "X to built track", "A junction to a track end"; "n junctions"
- Checkbox hovers: "Disable this stretch" / "Enable this stretch", "Disable this station" / "Enable this station", "Disable all joined track" / "Enable all joined track"
- Stretch hover: "Viaduct (+1)" or "Levels −1 to +1", plus "2 flat crossings, $15.0M"; part heading hover "3 stations, 4.0 km"

### Panels (T-2271's panel items, T-2286, done overnight)
Ho Chi Minh City fits on one line (name cap 140 px); hammer icon on Build (artifacts/trainworld/panels/hammers.png,
the first one); new games start with $10B and new lines with 5 cars; money and region ledgers now turn their days at
the start zone's local midnight. The homes/destinations key is **H**: D is taken (the Select tool's disable). Hours
are written "06:00" to match the top bar clock (you wrote "6:00"; say if you want the zero dropped everywhere).
- History switch: "Days" (hover "Every game day so far") / "Hours" (hover "The last 48 game hours") -> "1d" / "30d" / "All time", no hovers
- Homes button hover: none -> "Show homes (H)"; Destinations: none -> "Show destinations (H)"
- Help panel, new row: "H": "Switch demand bubbles between homes and destinations"
- Line panel level hovers: "High demand: 6-20 on weekdays" -> "High demand: 06:00-20:00 on weekdays"; "Medium demand: 20-24 on weekdays, 6-24 on weekends" -> "Medium demand: 20:00-24:00 on weekdays, 06:00-24:00 on weekends"; "Low demand: 0-6" -> "Low demand: 00:00-06:00"
- Purpose chart axis: "6–10", "10–16" ... -> "06:00-10:00", "10:00-16:00" ...; bar hovers "To destination, 6–10: ..." -> "To destination, 06:00-10:00: ..."

### Map display (T-2271's map items, done)
- Weekend: the High dot is only its weekday rim (those hours run Medium's trains, now counted in the Medium dot).
  Wider dot hover. Editing stops dims bubbles to 35% beneath the network instead of hiding them. A log bar under
  the station tool's "57k trips within 1km" (1,000 to 1M). The bubble hover's mode bar scales with the count (log).
- "lower min demand bubble trip size more": read as the Settings bubble Size slider, minimum 0.5x -> 0.25x. If
  you meant the smallest trip count that gets a bubble at all, say so.
- **Crowding dot**: 100% plus the cube-root mean of each stretch's load past 100%; one crushed stretch in 20
  comes out 118% (yellow), a line full everywhere keeps its load; the hover keeps the peak. Example: BNTNM, 1 of 12
  stretches at 148%: crimson now, yellow (121%) with the mean. PATH, 3 of 5 up to 150%: crimson now, red (138%).
  On since T-2335, the ring the peak. Pictures: artifacts/trainworld/display/t2279-curve.png,
  mean-one-packed.png beside peak-one-packed.png.
- Strings: crowding hover period lines "6-10: 106%" -> "06:00-10:00: 106%"; new weekend High dot hover "Medium
  demand trains run in these hours today", then "On weekdays, peak load 140%" and the weekday strip.
- Fixed on the way: the Lines tab's trains hover showed nothing for most lines since T-2236.

### Demand rules (T-2265, T-2278, T-2276, done overnight)
- **Walking check in real minutes**: a station pair is refused only when walking the whole way is quicker in real
  minutes (no fare, crowding or weights); T-1782's "only a little quicker" fade is gone. Real save 11.72M -> 12.51M
  rail trips a day (+6.7%). It made the demand solve about 10% slower; T-2307 got back a quarter of that, the rest is T-2324.
- **Parking by income**: paid parking times min(1, (GDP per head / $35,000)^0.3), from parking price surveys across
  57 cities (notes/T-2278.md). India 0.46, Vietnam 0.56, China 0.76, Turkey 0.83; 1 from Japan and Korea up. With no
  rail, Ho Chi Minh City's car share 53.7% -> 60.6%, Delhi 59.7% -> 60.9%, rich cities unchanged. Curve:
  artifacts/trainworld/demand/t2278-curve.png.
- **Station buildings**: a change inside counts 1 of the fixed 2 minutes; waits inside count 0.8 (a waiting room).
- Strings: trip explorer no-train line "Walking there is shorter than the walks to, between and from the stations."
  -> "Walking there is quicker than the train.", its new hover "By train at least 34 min, walking 28 min." /
  "Nobody takes a train that takes longer than walking."; removed: "Only a little quicker than walking, so fewer take
  the train." and its hovers; lifted-route hover -> "From these two points walking takes 28 min, less than any
  train. The train trips counted here are worked out for the areas around the two points."; building hover
  "Changing trains inside it feels a little quicker." -> "Changing trains inside it feels 1 minute shorter." plus
  "Waiting in it for a train feels 20% shorter."

### Network diagram (T-2272, T-2289, done overnight; no strings)
Dark mode: black ring, white centre (a thin rim keeps it off the background). Local stops (one line stops, or only
one colour's lines) drawn small. High-speed lines get a centre stripe (as the shinkansen symbol in Japanese atlases).
Stations kept 4x further out. Faint names at far zooms for stations with 3+ lines and no bigger one within 120 px.
Long lonely stretches between cities condensed: all Japan 9,902 x 13,310 -> 8,120 x 10,145, longest stretch 1,111 ->
587. A few more close pairs and overlaps on all Japan and Tokyo came with it (T-2321). Shots:
artifacts/trainworld/diagram/nyc-mid-dark-after.png (before: nyc-mid-dark-before.png), jpn-whole-after.png (before:
jpn-whole-base.png), tyo-out-dark-after.png.

### Waiting room in a station building (T-2308, done overnight)
The toggle is greyed at a building's stations (a building already counts as one), unless one is already bought.
- Hover, at a member: "...Costs $X / Click to build" -> "Waiting room / Waiting here for a train feels 20% shorter. / The station building already counts as a waiting room."
- Same in complex view, with "Waiting at any station in this complex for a train feels 20% shorter."

### New countries overnight
- **Turkey: 74 zones shipped**, on 8800 (build 877): 8 metros (Istanbul 14.6M, Ankara, Izmir, Bursa, Gaziantep,
  Antalya, Sincan, Konya) and 66 zones; people from the 2022 register plus the 3.5M Syrians under temporary
  protection, the earthquake provinces scaled to 2023. Adana is one 2.3M zone (the population grid undercounts it;
  T-2296). Map: artifacts/trainworld/countries/tr-plan.png.
- **Russia (173) and Belarus (8): built and checked, joining after the US** (later this morning). Russia to its
  recognised borders: the boundary data holds no Crimea or occupied Ukraine. 4 lone Siberian towns held. Map:
  artifacts/trainworld/countries/ru-live.png.
- **Taiwan replanned**: 9 zones; "Taitung" (1.29M) wraps the Central Range from Douliu to Hengchun to Hualien (T-2300
  would split it).
- **US and Puerto Rico replan**: 159 new zones built (155 on AWS for $0.74), going in now; Europe (628 zones to 462)
  after.
- Questions: (1) Territory: your October rule (`DE_FACTO`) gives Crimea to the side holding it when Ukraine's units
  are cut, so if Ukraine is ever built Crimea would be in neither country. Keep recognised borders for both, or your
  October rule? (2) Build Ukraine at all, on the population grid only, with a 2001 census and millions displaced?
  The agent leans not yet; so do I.

## From 2026-10-26, still open (checked 2026-11-04)

The rest of that list is built or answered: Canada, Kwai Chung, superholidays, both fare tasks, level of
detail, the slow-computer work, track edits on big networks, the worker crash, related lines, cloud saves,
the terrain tint, the waiting room, real networks sharing track, routed drives (T-279), the base map
palette, the privacy page, and your go on the replan (T-1189, under way tonight).

- Someday tasks you might want back, all still open: T-750 add a city's real network to a sandbox game
  already running; T-685 drawn city stamps at 10% rail share (the stars are done); T-651 unopened zones look
  fully red when zoomed out; T-632 walking feels too strong to play against (you said try later); T-438 a
  giant metro takes most of a smaller neighbour's 80-300 km trips; T-263 stadium events; T-803 more station
  faces by time of day; T-678 day and night tint on the base map; T-784 lighter settings for low-memory
  phones (phones already get one demand worker, T-2122).
