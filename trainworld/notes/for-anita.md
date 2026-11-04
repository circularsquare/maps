# For Anita

## Open now (2026-11-04, maps-c9)

Answer "text acknowledged" per block, or name the lines to change.

### Tutorial (T-2263): your rewording is in build 868
One change to your wording: the Traffic note says "Red borders indicate track at full capacity." (the red
borders in Traffic view mark track full of trains, not crowded trains). The green-bubble note is dropped.

### Track and station bars, mode frame (T-2267 + T-2270, design only)
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

### Map display (T-2271's map items done; T-2279 built, switched off, waiting on you)
- Weekend: the High dot is only its weekday rim (those hours run Medium's trains, now counted in the Medium dot).
  Wider dot hover. Editing stops dims bubbles to 35% beneath the network instead of hiding them. A log bar under
  the station tool's "57k trips within 1km" (1,000 to 1M). The bubble hover's mode bar scales with the count (log).
- "lower min demand bubble trip size more": read as the Settings bubble Size slider, minimum 0.5x -> 0.25x. If
  you meant the smallest trip count that gets a bubble at all, say so.
- **T-2279 crowding dot**: 100% plus the cube-root mean of each stretch's load past 100%; one crushed stretch in 20
  comes out 118% (yellow), a line full everywhere keeps its load; the hover keeps the peak. Example: BNTNM, 1 of 12
  stretches at 148%: crimson now, yellow (121%) with the mean. PATH, 3 of 5 up to 150%: crimson now, red (138%).
  Try it on 8800 with `?crowdMean=1`. Turn it on? Pictures: artifacts/trainworld/display/t2279-curve.png,
  mean-one-packed.png beside peak-one-packed.png.
- Strings: crowding hover period lines "6-10: 106%" -> "06:00-10:00: 106%"; new weekend High dot hover "Medium
  demand trains run in these hours today", then "On weekdays, peak load 140%" and the weekday strip.
- Fixed on the way: the Lines tab's trains hover showed nothing for most lines since T-2236.

### Older asks found open by the audit (2026-11-04)
- T-1920: a zone's rail commuters are dealt over all its work cells by jobs, so cells no station reaches get
  train riders (5 of 71 New York pairs). Deal them only over cells within reach? A demand rule, so yours.
- T-2058: the interlining look (lines side by side looking continuous). Only "turn the default on once you have
  seen it" is left; you have been playing with it (T-2222, T-2223). Make it the default?

## From 2026-10-26, still open (checked 2026-11-04)

The rest of that list is built or answered: Canada, Kwai Chung, superholidays, both fare tasks, level of
detail, the slow-computer work, track edits on big networks, the worker crash, related lines, cloud saves,
the terrain tint, the waiting room, real networks sharing track, routed drives (T-279), the base map
palette, the privacy page, and your go on the replan (T-1189, under way tonight).

- **Mexico**: next country after Turkey, whose build runs tonight. Its census and jobs sources are filed
  (T-2080, T-2081); no build task yet.
- Someday tasks you might want back, all still open: T-750 add a city's real network to a sandbox game
  already running; T-685 drawn city stamps at 10% rail share (the stars are done); T-651 unopened zones look
  fully red when zoomed out; T-632 walking feels too strong to play against (you said try later); T-438 a
  giant metro takes most of a smaller neighbour's 80-300 km trips; T-263 stadium events; T-803 more station
  faces by time of day; T-678 day and night tint on the base map; T-784 lighter settings for low-memory
  phones (phones already get one demand worker, T-2122).
