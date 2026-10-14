# Decisions

Newest first. One entry per decision: what, and why. The spec states the result; this keeps the
reasoning and what it replaced.

## 2026-10-14

- **A check page (`?new=1`, `?synth=1`, `?demandTest=1`) does not autosave, until a copy is saved**
  (T-320, found in T-308: a `?new=1` page closed or hidden wrote its empty game over the `autosave`
  slot, so an agent's check in a profile with Anita's games could wipe one). Chosen over giving
  every `?new=1` game a fresh slot of its own: that would add a "Saved game" card to the list on
  every check, and the page's active slot is also remembered for the next start, so the next normal
  page would have opened the empty check game. Not autosaving is the smaller change and changes
  nothing a player sees: a `?new=1` game has no active slot (`""`) and reads none, the worker
  takes no lease on a slot, so `autosave()` does nothing, and "Save as a copy" still works (it
  makes a slot, takes its lease, and autosave goes on there). `?synth=1` and `?demandTest=1` load
  the last game as before but take no lease either: `?demandTest=1` feeds the real game riders
  from the hand-made network, which changes its cash and would have been saved. The top bar says
  "Autosave is off in this test page. Save a copy to keep the game." The client also no longer
  remembers a test page's slot as the one to open next. New game and Load a file in such a
  page change the page's game only.

- **Zoomed out every station is its own dot, a complex's dots one shape with connectors** (T-313,
  Anita: "stop growing dots to fit all the platform centers within the complex dot, it looks a bit
  silly"; dots "65% bigger radius"; the connector "should look like its made of the same
  material"). Replaces T-217's one dot per complex (centre of the smallest circle, standing for the
  busiest station) and T-225's thin 2 px connector with a 1 px rim. Dots are merged the way boxes
  are (all dark, then all white); within a distance group the connectors are a minimum spanning
  tree over the dots' centres (shortest total, no long joins past a nearer dot), and a player's
  link between two groups joins the two linked stations' own dots, so a link still reads as the
  pair the player chose. The connector has the dots' rim and white 35% of an ordinary dot's white
  radius either side (chosen by eye: thick enough to read as the same material, thin enough that
  dots still read as dots); its white goes under the dots' white so a crowd tint stays whole.
  Overlapping dots pick by nearest centre, so each dot gets its own station; the tag lists that
  station's lines, no longer the complex's. Station names keep clear of a transfer dot.
- **A place gets a zone's name and load only within reach of it** (T-318, found in T-307: a London
  game was named "Boston" and showed a "Loading data" card). `cityNear` and `zoneAt` pick the
  nearest zone at any distance, which is right for a lookup but wrong for a name. The view now
  belongs to a zone when it is inside its boundary or within 40 km of it (about the market's onset,
  25-50 km, and the ring round a city T-300 would model), and before the boundaries have loaded
  within 120 km of its centre (past the farthest edge of New York's metro). Farther is dead space:
  no name, "–" for the split, nothing loads (not even the market) and the clock is not held; the
  focus stays on the last zone. One helper, `placeAt` (`game/zones.ts`), answers it and
  `viewedZone` (`game/cities.ts`) holds it, for T-309's local time as well. Panning inside the
  Northeast does not change (the same zone under the view); only a gap wider than 40 km (upstate
  New York, Maine) now has no name.
- **Closest stations and walking minutes are worked out on the main thread** (T-312, Anita: "list
  3 closest stations ... and their walking distance in minutes"). The walking barrier the demand
  model routes on lives in the demand workers' wasm, which is busy during a solve and would need a
  wasm rebuild and a new query, for a hover. Instead `game/walks.ts` fetches the pack's `walk_row`
  and `walk_x` (about 250 KB) and runs a TypeScript copy of `WalkGrid::field` and `Field::eff`, so
  the walks are the model's own and cross no river that it does not. Minutes are the routed metres
  x the model's 1.3 detour at 4.8 km/h, a plain walk (not the 15 km/h fast walk to stations).
  Ranked by walk, same-complex and same-name stations left out, a name once; a cut-off station
  shows "no walk" only to fill the list when fewer than three can be walked to within 10 km. The
  1 and 2 km rings while placing are an SVG layer beside the ghost, not in the renderer.

- **The station ghost says yes or no, and blueprint track is solid** (T-311, Anita: "theres no
  indicator for whether its okay to put a station here"; "the dashedness looks quite confusing as
  the lines curve"). The ghost is always the full box at its true length (it was a dot below zoom
  15) and is blue where a click would place the station, red where it would be refused, the
  reason in the tag. The test is the click's own: `TrackApi.preview_station` runs `add_station`'s
  code as a trial and rolls it back (cash included), asked of the clock worker once per change of
  spot, one in flight, last answer kept meanwhile; a UI-side guess would drift from the model.
  Colours fixed, not the theme accent (blue #2f6fe0, red #e03c3c). Blueprint track lost its dashes
  for the same colour mixed 45% towards white, opaque: plain 55% alpha would double up where
  segments overlap at every joint and show beads on curves, which the dashes had hidden. Lines
  that do not run keep their 55% see-through. `dist` in the stroke buffers (network.ts) is now
  unused by the shader.

- **Tier 0's jobs proxy weights GHSL non-residential volume by height^1.5** (T-287). By volume
  alone Toronto's centre got a third of its jobs (suburban warehouses outweigh towers); weighted by
  each cell's mean non-residential height (volume over surface) to the power 1.5, jobs within 3 km
  of King and Bay come to 383k against the census pack's 354k, and jobs misplaced between
  municipalities fall from 7.4% to 4.0%. Blend 50% that, 30% OSM job places, 20% population (set
  before looking). The power is in-sample on Toronto; London (T-288) checks it. Population from
  GHSL GHS-POP rather than WorldPop: one product family for people and buildings.
- **Tier 0's curve: prefer a borrowed fitted curve to a length fitted to one published figure**
  (T-287). Fitting only the length to Toronto's 10.5 km median commute (2016) gave
  d^-1 exp(-d/95 km): 17.5% of commutes misplaced between municipalities, against 15.7% with the
  US curve and 11.3% with Toronto's own, and 22% fewer riders on a test line. `tier0.fit_curve`
  stays for cities with nothing to borrow; a distance distribution (Mumbai's census bands) should
  fit both parameters.
- **Panels count today so far, not a whole-day forecast** (T-293, Anita: "lets not even show the
  whole day forecast cuz it shows stuff in the future. lets only show so far today, and call it
  'today'"). Replaces the line panel's Demand forecast, Trains today and Finances (riders, fares
  and running cost a day; the account's whole day of trains), the station and complex boardings a
  day, and the Lines and Stations tabs' figures. Counts come from the boarding account's
  departures since midnight (`TrackApi.today`), so they are the trains and platforms the player
  sees; fares from each line's hourly fare as the cash is paid. Lists sort by today's counts
  re-sorted each game hour, except the complex list, kept in forecast order so rows do not jump
  in the early morning. The purpose split shows shares only. The map's demand views and the
  station's commuters and inter-region trips lists stay demand's whole-day picture (Anita: they
  are demand's picture, not the panels' counts). The line panel's longest wait went: a so-far
  version needs the wait per departure, which the account keeps only per hour.

- **Inter-region trips are no purpose in the filter, and the name replaces "longer trips" on screen**
  (T-305, Anita). They show under All purposes only; a filter value of "longer" falls back to all
  (`asFilter`, `game/purposes.ts`). Code identifiers and docs/demand.md keep "longer trips". The
  top bar's cash has a fixed width with the $ in a fixed place; a station's hover tag lists its
  lines by their chips' symbols (each once across its complex) in place of "one dot for the
  complex's N stations". A line the player gave no symbol is listed by its name without a leading mode or operator word
  ("subway 6X" is "6X", "lirr Port Washington Branch" "Port Washington Branch"); its chip keeps
  showing the name's first letter. Lines are de-duplicated on the text shown.

## 2026-10-13

- **A bubble selection's everyday far end visits only the selection's samples and answers sparse**
  (T-285, Anita: a click on a destination took up to ~3 s to show its homes). Replaces weighing
  every destination sample of the city per click (420 ms a Chrome worker in New York) and a dense
  answer of three numbers per cell per purpose (14.6 MB a worker, 80-360 ms to add up on the main
  thread). Same f32 sums in the same order, checked to the bit; a click now shows its far end in
  80-170 ms. Clicks during a re-solve still wait for its final round (7.5-11 s on the real New
  York save): answering them from a kept copy of the shown result is a redesign of the query
  state (notes/T-285.md, T-122) and goes to Anita for review first.
- **Money is paid as trains pick up riders, not once an hour** (T-284, Anita: "money update live
  so that it feels like people are actually paying fares"). Replaces whole-hour settling. Each
  hour's fare is paid by a curve from the boarding account's departures that hour (riders each
  train takes), its running cost evenly, settled four times a wall second; each hour still pays
  its whole fare and cost, so day totals are unchanged (checked on the real New York save to
  1e-12). Per-train timing was cheap (one pass over the hour's platform events), so no plain
  per-second accrual was needed except for hours without an account. The cash is shown to the
  dollar, since at $10M steps the movement was invisible.
- **"Stations full" counts riders left by the last train, not the hour's total** (T-284). The
  hour's total made the list jump as each hour turned; the last train's left-behind riders follow
  the trains. Same meaning, read now instead of per clock hour.
- **Walks and drives follow real paths** (T-275, Anita: "can people currently walk across the
  hudson? seems like a kinda prereq"). Replaces the straight line x 1.3 for both. Walks: a 50 m
  barrier raster per pack (its water mask, less OSM's walkable bridges), searched from each
  station in the game because stations are the player's (cached per station position, so an
  edit costs what it did; opening a big save pays about 1 ms a station). Drives: per-pack zone
  factors from OSM roads (Dijkstra offline, `pipeline/routes.py`), road length only; speeds stay
  the density model's, as Anita asked. Chosen over a coarse walking graph of H3 cells (cells exist
  only where people or jobs are, so parks would be holes) and over per-cell reach tables (stations
  are unknown offline). notes/T-275.md.
- **The routed road factor is scaled by 1.3 / 1.210** (T-275). Shortest-distance routes come to
  1.17-1.26x the straight line (commute-weighted; New York 1.210), against the 1.3 the density
  speeds were fitted with on New York's ACS car times; unscaled, New York's commute rail fell from
  19.7% to 18.4%, mostly in the boroughs, not by water. One scale for every pack, New York's, so a
  pack's roads still differ from New York's; a refit of the speeds would be the alternative.
- **Walking transfers between stations stay straight x 1.3** (T-275, the T-161 question). Every
  population must build the same graph to share strategy searches (T-119); none of the real New
  York network's 80 station pairs within 300 m has water between them; complexes are the
  player's links.
- **Residential streets are in the drive graph** (T-275): without them rural places joined the
  roads kilometres away (Springfield's near factors had a median of 1.66).
- **Toronto: the first pack outside the US, its own adapters** (T-277). The Greater Toronto and
  Hamilton Area by whole census divisions (cheaper than the CMA, which cuts divisions; whole
  divisions are docs/zones.md's unit), built by `pipeline/canada.py` from Statistics Canada's 2021
  census with the substitutions in docs/adding-a-city.md "Outside the US". Outside the regional
  market for now (`market: -1` in the registry) until T-276 places it.
- **Toronto's gravity curve is its own fit**, d^-1.2 exp(-d/20 km), on the 2021 census flows
  between its 28 municipalities (5.6% of commutes misplaced against 15.1% at the US curve). The
  game now takes each pack's shipped curve at opening (`DemandApi::new`); before, a pack whose
  curve differed from the US default would have been re-solved with the US curve. The 2021
  flows are a lockdown month's, likely somewhat more local than normal; the pre-2020 flows are on
  StatCan's blocked server.
- **Lockdown census jobs put back by income, not by office density** (T-277). The census counts
  at-home workers at home; each employment income band's CMA ratio of at-home to usual-place
  workers puts their jobs back at work tracts. Downtown probably stays under its pre-pandemic
  count, but today's commuting (24.7% at home) is plausible, and an OSM office reweight has
  nothing to check it against, so it was not added.
- **Pearson's connecting share 0.75, by judgement** (T-277). Canada has no national source like
  DB1B; 0.75 is the share of the US hubs DB1B measures (Dulles 0.74, BWI 0.66). Other Canadian
  airports 1. Marked as judgement in docs/demand-worldwide.md.
- **One work-from-home share for the whole Toronto pack**, the Labour Force Survey's May 2024
  figure for the Toronto CMA (24.7%), Hamilton and Oshawa included (they had less at home in
  2021, so they get slightly too few commuters back; noted).
- **OSM commercial and industrial land counts as one job place per 5,000 m²** when spreading a
  census tract's jobs over its areas (80% job places, 20% population). Judgement; only the split
  inside a tract depends on it.
- **OSM for a Canadian pack comes from the Geofabrik extract, not Overpass** (T-277): the public
  servers answered 504 for the area all afternoon; the extract (shared with T-275's routes.py) is
  read once with GDAL and written in Overpass's shape, so the shared enrich scripts run unchanged.

- **Traffic widths show the whole day** (T-274, Anita: "lets show demand for the whole day rather
  than the current hour"). A stretch's width is its riders over the day shown, both directions,
  against the busiest stretch-day of either kind of day, still T-227's straight 0.5 to 6.6 line
  widths. Replaces the riders in the period on the clock (T-227), which made a switch of day look
  far bigger than the panels' daily figures at peak hours (T-267) and made widths change through
  the day. Platform crowds, dot tints and train fill still show the hour.
- **The US border factor stays one value per pack** (T-171, Anita: "i feel like the border
  penalty is weird tbh"). Fitted on LODES OD in five packs, against per-state-pair factors, a
  fade near the line, a factor beyond 6 km only, a factor only between metros (GHSL-style urban
  centres from the pack's cells), a water term, a water-crossing (toll) term, water plus the
  factor, and none: only per-pair factors tie it (and only help Washington, which has its own
  0.45). A metro-only factor cannot reach LODES in New York (half its cross-state commuting is
  inside one urban centre), and water or toll terms cannot replace it (untolled land borders
  show the same cut, and a penalty strong enough to match crushes in-state flows across the same
  rivers). It was not PATH's problem: most of that was realnet's way of counting operators
  (T-269); the rest is mainly the cheap drive across the Hudson (T-270). The fade near the line
  (T-273) is left to Anita.
- **Riders left behind cost a wait in the next crowding round** (T-188, T-164's phase 2, Anita:
  yes). Each round runs the boarding account's rule per period on the period's own trains (even
  headways, the account's arrival shapes, crush loads, separate queues per line) from every
  population's averaged loads, and adds each boarding's average extra wait (capped at 60 minutes)
  at the platform wait's weight. Inside the solve rather than read back from the clock worker's
  account: feeding the account back needs a second and third solve after every solve, a new game
  would open with no account, and per round it settles with the crowding by the same averaging
  with no extra damping. Per period, not per hour (the account's hours are not mapped). Weekday
  only: the weekend keeps the weekday's route and mode choices (T-260) and so its waits. Real New
  York save: rail -0.1%, evening-peak riders left behind 40.6k to 30.4k, LIRR Hempstead -4.2%,
  no busiest line over 0.8% (notes/T-188.md).
- **The demand views show the clock's day by default** (T-266, Anita: "that should actually go in
  the settings ... either weekday, weekend, or show whatever current day is (default)"). The top
  bar's Weekday / Weekend buttons are gone; Settings has "Demand shown for": Today, Weekdays,
  Weekends. It is remembered with the other display settings now that the default follows the
  clock (T-260 left the buttons unremembered so nobody returned to a silently weekend map; with
  Today as the default a remembered choice is one the player made). No other mark of the day
  shown: the clock says the day, and the split's hover says weekdays or weekends.
- **Traffic widths stay as they are** (T-267, Anita asked whether the white border makes widths
  grow faster than riders). It does not: the coloured stroke is linear in the riders an hour
  (half a line at none, T-227) and the casing is a fixed margin drawn under it. What she saw is
  that the width shows the hour on the clock: at 8:00 a Saturday has under 40% of a Monday's
  riders on the busiest lines while the day's riders in the panels fall only to about 77%; at
  midday a Saturday is the busier. The busiest-stretch scale is the same on both days.
- **Weekends are worked out from the weekday's solve** (T-260, from T-170's brainstorm). Two kinds
  of day, weekday and weekend: Anita asked for a weekday/weekend selector, Saturday and Sunday
  share the main change (work down to about 0.3, leisure up about 1.6 in NHTS 2017 and England's
  NTS), and a third kind can come later without new plumbing (T-262). Game day 1, a new game's
  first, is a Monday. A weekend is each purpose's weekday riders per period times a factor from
  NHTS 2017 (TSA for airports), not a second solve: without crowding a trip's choices depend on its
  period's service only, so it costs no solve, no opening time, and weekday results stay
  bit-identical. The price is that weekends keep the weekday's crowded route and mode choices.
  Weekend timing is the model's weekday shares shifted by NHTS's weekend/weekday period ratios,
  not NHTS's own shares, so the weekday's judgement carries over. The same trains run every day
  (Anita, T-170: no separate weekend schedules for now); a weekend service level would need its
  own solve and is open (T-261). The top bar's Weekday / Weekend buttons choose what the demand
  views show and start on Weekday at each page load, like the purpose filter; the trains, platform
  crowds and fares follow the clock's own day through one boarding account per kind of day. Line
  widths are scaled to the busier day's busiest stretch so a weekend reads as quieter. T-170's
  other ideas became T-261 to T-265.
- **Doc tidy after batches 17-19** (T-259). SPEC 8 is a short summary again (it had grown to 90
  lines by appending; docs/interface.md has the detail); SPEC 9 keeps only the latest calibration
  figures (T-198), earlier checkpoints stay in notes/T-090.md and notes/T-162.md. TASKS.md is
  grouped by area instead of by batch, with performance tasks in one parked section. T-142 drops
  from P2 to P3: its nine-city work is done, and its other-country adapters wait for city
  additions, which are paused. Notes of finished tasks carry an "Archived" line.
- **A split inside a curve fits its halves to the node** (T-258). The new PIs either side of the
  node are on the millimetre grid. A few metres from the node their turn angle is measurably
  off, so the arc's radius overran the short leg and the station was refused. Each now takes the
  radius whose curve ends exactly at the node. When that differs from the arc's by more than 1%,
  which only happens within a few metres of the curve's end, the split moves to the curve's end
  instead. A station is never refused for this, and the track keeps its shape to a centimetre.
  The alternative, dropping the grid for these PIs, would have made saves and undo depend on
  float noise.
- **A platform follows its track's own line, sampled at 13 points** (T-254, Anita: "ideally we
  actually fit the track"). T-189's circular arc through the platform's middle and ends cannot
  bend both ways, so on an S-curve (110 St-Malcolm X Plaza, Wall St, Grand Central's Lexington Av
  platforms) the box was a near-flat rectangle with the track snaking across the blue edges.
  Now the clock worker samples the track along each platform (`PlatformShape.pts`, network.ts
  `stationLine`) and every consumer measures position along and across that polyline
  (`polyFrame`, the same arithmetic in GLSL and in TS) instead of the arc. A polyline rather than
  a smoother curve: the distance to it is exact and cheap in a fragment shader, and straight track
  gives exactly the old box. 13 points, not 9: at 9 the outline showed visible facets on
  34 St-Hudson Yards' curve at zoom 16.5 (25 m stretches turning about 10 degrees); 13 points are
  12-17 m stretches. The points ride to the fragment shader as flat varyings (7 vec4) and to the
  selection outline as texels (10 a shape, so at most 204 selected shapes share an outline, down
  from 512; Penn Station is 24). Past the end of an edge shorter than half the platform the line
  carries straight on, as the arc's chord did. Cost on Midtown at zoom 16, clock at speed 2,
  trains on: GPU about 1.55 ms a frame against 1.17 before, CPU 0.46 ms against 0.40, frame gaps
  16.7 ms either way.
- **Quad track is two double-track edges side by side, and Express copy moves stops to twins**
  (T-255, Anita: "easy express line construction / setup... maybe add quad track drawing"). A
  4-track edge kind would have touched every part of the model (capacity per track, platforms per
  track, which track a line is on, picking, drawing); two ordinary double edges reuse all of it,
  and "which pair a line uses" is just its path. The express pair is 8 m to the left of the drawn
  direction (4 m between neighbouring tracks, like double track; wider would read better at
  zoom 15 but nothing real is laid that way), concentric at corners, and starts and ends at the
  stroke's own end nodes after a 150 m swing-out, so lines change pairs at the ends of each quad
  stretch with no extra junction nodes to design. Mid-stretch crossovers are not drawn (the
  player can end one quad stroke and start another to get a crossing point; a crossover between
  tracks 8 m apart cannot be drawn by hand, the curves are too tight). A station on quad track is
  two stations of one name 8 m apart, one per pair (a complex, so riders change across the
  platform on foot), placed together; that keeps the one-node station model rather than a
  four-track station kind. Express copy is the smallest UI for an express line: one button beside
  Copy line, moving each stop to its twin within 12 m, then straight into stop editing so the
  skips are clicks on the map; no naming step (it is "<line> express"). Placing a station within
  a platform's reach of a T-244 joint now joins the two pieces first (it used to be refused as
  "platform too long", a T-244 regression this found).
- **Capacity tags let tools through; tabs keep their own scroll** (T-224, T-215). With any tool
  on, the tags take no pointer events at all (a CSS class from the tool signal) rather than
  forwarding clicks as T-221 did for the trip tool: the track tool and node drags also need the
  moves, presses and double clicks under a tag, and forwarding each kind would copy the map's own
  handling. The tags' hovers are off meanwhile. Each tab remembers its own scroll for the session
  rather than always opening at its top, so going back to a long list (Lines, Stations) returns
  where you were; the first visit opens at the top, and nothing carries over between tabs.

- **Saved games move to their own panel** (T-251, Anita: "i dont like the saves being at the top
  of the settings ... a save button to open a floating center saves panel"). Replaces T-241's
  placement at the top of Settings. The button is a floppy disk at the right end of the dock's tab
  row, beside the gear, rather than in the top bar: the top bar's narrow layouts are a fixed grid
  with no free cell, and the tab row already holds the other "open a pane" buttons. The panel is a
  native modal dialog like the confirmation's (focus kept in it, Esc, a close button, a click on
  the dimmed page closes it); Open, Load a file and Start a new game close it too, so the loading
  card shows. The cards get a 192 x 128 picture (drawn at 384 x 256 for sharp screens) and the
  station count. The note on how saving works is gone (Anita), and Settings' "Back to 100%".
- **Dock tabs are icons** (T-252, Anita): Build a wrench, Lines a route with four stops, Stations
  a platform under its canopy (a house read as homes), Money a bill with a coin; names in the
  game's hover and as the buttons' accessible names. Fixed-width tabs on the left, saves and
  settings on the right, rather than stretched icon buttons.
- **Panels drop one-time explanations** (T-253, Anita: "most of the info we display should be
  stuff that can change"; CLAUDE.md "Panels show what can change"). Notes on how a figure is
  made, what a purpose covers or what a button will do moved to the hover of the label they
  explain; instructions the map's tool hint already gives were dropped; empty states and
  Planned/Broken/Blueprint lines kept to the state. Every sentence moved or dropped is listed in
  notes/T-253.md. The track, delete and trip tool hints are shorter.
- **A click on a station always selects its complex; the Station / Complex switch is gone**
  (T-249, Anita: "in general we should just pick the complex"; T-246's rows already reach a
  station alone). Complex view no longer reads `display.complexView`, which stays only as the
  demand circles' "One circle per station complex". The switch would only have toggled between
  the complex and the one station already reachable from the rows, so it went (Anita wants minimal
  UI); a station opened alone shows "Complex of N stations" as a link back in its subtitle, and any
  new selection object clears the alone state, so clicking the same station on the map again
  shows the complex too. Search, the stations-full button and the Stations tab list all select a
  station and so now show its complex as well.
- **The complex view lists lines, not stations** (T-250, Anita: "one row per line, not one row per
  line-direction"; "say what line it is in the tooltip"). The real New York save gives every
  service pattern its own station and line ("subway 6" twice, to Parkchester and to Pelham Bay
  Park; "mnr Hudson" twice), and every chip reads S, M or L, so the station rows all had the same
  name and near-identical chips. Rows are now one per line name with its patterns' platforms,
  waiting riders and boardings summed, labelled with the line's name, the hover giving the station
  names and where its trains go; a click opens the busiest of its stations alone. The line list
  under the complex map was the same lines a second time and went. The platform strip's hover
  sums per line name too. The note T-246 put under the complex map went (panels show what can
  change).
- **Platforms are blue edge lines and the track runs through the boxes** (T-236, Anita: "show the
  actual platforms as like little blue lines. and maybe the track on top of the station"; T-226's
  grey hairlines "are confusing"). Inside each box, over the crowd fill: the track and line strokes
  drawn a second time, clipped by a stencil to the stations' white (the overlay canvas now asks
  for a stencil), then each platform's two long edges in blue. The stencil, not a plain second
  pass of the strokes, because a second pass everywhere would double the alpha of blueprint,
  dimmed (stop editing) and far-out faint strokes; the white is marked per pass (1 unselected, 2
  selected), so the track is drawn once inside each. The track is capped at 30% of the box's white
  and has no casing: a 6.6 px line (44 px with traffic thickness) would otherwise fill a 6-16 px
  white and hide the crowd, and the casing would cut the outline at the box ends. Blue edges along
  the rim rather than beside the track: the brief said "along its edge", they show each platform
  of a complex where boxes overlap (what the hairlines tried to do), and the crowd fills the space
  between them; they fade in from zoom 15 to 15.5, since at 15 the white is about 6 px and two
  blue lines plus the track read as hatching. Blue #8085f0, a periwinkle: the palette's blue
  (#1f6fd1) and light blue (#5cc3f0) leave no plain blue far from both, and this is 15 CIEDE2000
  from each and 38 from white (any bluer choice was 6-12 from one of them).
- **The complex view lists its stations first, live** (T-246). The member list moves above the
  complex map, with riders waiting now beside boardings a day (header row "Waiting", "A day").
  A click opens that station alone through a per-selection override (`singleStation`) rather than
  flipping the remembered Station / Complex setting, which is shared with the demand circles and
  would make every later complex open as one station. Platform crowds and trains come from the
  renderer's own hour of the boarding account and trip buffer, so the panel and the map always
  agree; they are drawn as a 5 px platform strip after each station's lines in the complex map
  (the map's grey/red fill from the middle out, each direction's half) and train pills on the
  line's band, standing ones solid, ones stopping within 60 game seconds paler and sliding in from
  the arriving end. Inside the platform rather than past its end, because the map is fitted with
  23 px of margin and a train outside the platform left the drawing. Worked out every 10 game
  seconds (a 150 ms real-time check), so at 60x every train is seen standing (dwells are 20-40 s)
  and approaching, and the panel re-renders only when a shown number or train changes.
- **Zoomed out, a linked complex is a dot per distance group and a connector** (T-225, Anita:
  "yes, connector"). Dots group by the complex rule's distances alone (`dotGroups`), so links never
  make a dot larger; each link between two groups is a thin white line with a dark rim between the
  two dots (drawn with the bridge program, 2 px white and 1 px rims, a fixed width on screen), under
  the dots, selected with them and part of the selection's outline and the bubbles' cut-out. Each
  dot stands for its own group's busiest station, not the whole complex's: the line and delete
  tools then pick a station near the dot clicked, not one up to 1 km away at the other end of a
  link; selection still covers the complex in complex view. The tag says "One dot for 16 of the
  complex's 23 stations".
- **A drawn route is one piece of track per corner** (T-244, Anita: "if we delete it it deletes
  the whole curve"). The cut is at the middle of each leg between two consecutive corners, not at
  the corners themselves. A cut at the curve's middle near a click would have replaced the click's
  PI by two half-curve PIs with set radii (Network::split_plan inside an arc), moved the ramps
  between neighbouring vertices, and made dragging that corner a node drag; at the leg middle the
  point is always on the straight, every click stays a PI of its own piece with the same auto
  radius (the whole-leg rule next to an end gives back the half leg it had), and the geometry, cost
  and run times equal the unsplit edge (Rust tests compare them). So a route of n corners is n
  pieces, one edge for none or one; "roughly one per click" as asked. Legs that change level and
  legs under 40 m are not cut (a ramp is centred between corners, a node sits at a whole level).
  A reshape moves the joint next to a dragged corner along with it, keeping its fraction of the
  way between the corners either side, because otherwise the joint's fixed heading would refuse
  almost every drag of a corner that dragged freely before. Pieces are not merged back, and the
  track inspector's Construct now builds one piece; Construct all and per line are unchanged.
- **Stop editing tiers come from the clock worker** (T-237, Anita: stations past the end of the
  line show as unreachable). The JS test put anything off the edges listing the line in the grey
  "changes the route" tier, so a station past a terminal looked like a detour though adding it
  only extends the path. `TrackApi::stop_tiers` now tries every station with `insert_stop`'s own
  search (hops memoised, a turn-valid reachability pass first) and calls it "route" when the new
  path contains the old one: on it, or past either end. The markers and a click cannot disagree,
  which also corrects connected stations the line could only reach by turning back (now "cannot
  be added", where the old test said "reroute"). 0-8 ms a line on the real New York save; the
  graph estimate shows until the answer lands.
- **Saves are cards at the top of Settings** (T-241, Anita: "do autosaves work? how can we load
  from an autosave? isn't obvious to me"). Autosave already worked (every minute if changed, after
  a paid edit, on hiding or leaving the page; checked end to end), so the change is in how the
  saves read. They stay in the Settings pane, now first in it (gear tab hover "Saved games and
  settings"), rather than a sixth tab: the tab row is full at the dock's narrowest and a saves list
  is visited rarely. Each slot is a card: a picture of its network, name, "Autosaved 2 min ago" on
  the game being played (others "Saved ..."), day, cities, lines and cash, the game being played
  marked Playing, an Open button on every other. The picture is a 192 x 128 PNG drawn in the clock
  worker on an OffscreenCanvas each time the slot is written, stored in the IndexedDB record with
  the counts (about 11 KB for the real New York network), not computed from the save bytes on
  demand: opening a save means decoding and loading the network, too slow for a list. Old slots
  show "No picture yet" until next saved (no migration, CLAUDE.md). Cities are the registry city
  nearest each station, most stations first.
- **The loading bar moves with time** (T-243, Anita: "show a bar filling, not a % number"; "we
  seem to freeze at 60% for like the majority of the loading time"). Measured: on the real New
  York save opening the packs is 30% of the load and the first estimate 64%, and both report all
  at once at their end (every worker finishes opening together; every period lands within a few
  hundred ms), so the old 10/50/40 percentage sat at 32% for 2 s and 60% for 3 s. Each step now
  weighs its expected time (packs by size, the first estimate by line stops, scaled by this
  browser's last startup), and inside a step the bar moves with elapsed time against that,
  slowing near the step's end, until the real completion lands; if an expectation grows the
  remaining work is spread over the rest of the bar so it never stands still or goes back.
  Real sub-progress from inside the solve would need the Rust loop to report (a WASM change); the
  estimate was enough. The percentage moved to the bar's hover. notes/T-243.md.
- **A UI scale setting zooms the panels, not the map** (T-247, Anita: "the fonts are a bit big
  tbh right now for me"). Panel and text size in Settings, 80-120% in 5% steps, default 100%,
  remembered per browser with the display settings. Every size in style.css is px, so the panels
  (dock, top bar, map buttons, search, loading card, tool notes and dialogs) get CSS `zoom` from
  one `--ui` variable rather than a rem rewrite of the whole stylesheet; the map's DOM tags only
  scale their font size, since their positions come from the map and a zoomed tag would move off
  its anchor. The dock's drag handles divide by the scale. A stepper, not a slider: the dock
  resizes under the pointer as the scale changes, which would make a slider jump away.
- **A selected train gets a halo and a ring** (T-242, Anita: "highlight the train in an obvious
  way"). The selected train is drawn last, over the others, inside a halo round its whole consist
  as one pill, so its own colour, fill gauge and red packed edge stay as they are: the selected
  station's dark outline (T-189), a white band, a dark ring of the train's shape 6 px out, and a
  soft glow. The station's outline and glow alone were tried first and barely showed: the glow is
  faint over a coloured line, and at zoom 11-13 the whole train is a few px; the ring reads at
  every zoom. The train is not drawn larger: its size encodes nothing, but a bigger pill would cover its neighbours and the station
  it stands at. A click tries the selected train first, since it is drawn on top.
- **The track model names each train's next trip** (T-245, Anita: keep following a train that
  turns back). `hand_over` already decided which departure a finished train takes; the trip now
  carries that departure (`Trip::next`, sent as a fourth number per trip), and the selection moves
  to it once the clock passes the old trip's end. Chosen over matching "a departure of the other
  run at this trip's end time" in the UI because that guess goes wrong exactly when trains are
  removed: a train leaving service can be ready within the hand-over tolerance of a departure
  another train takes. A train leaving service keeps the selection, as before, with "This train
  finished its trip and left service": deselecting would throw away the panel the player was
  reading. There is no camera follow mode to carry along.
- **"Waiting now" changes when a train leaves the platform** (T-238, Anita: update "the moment a
  train pulls in"). The account boards a train at its departure (the hour's events), and the map's
  platform fill already followed it to the game second; the panel re-rendered once a game minute,
  up to a second late at the slowest speed. It now also sets a timer for the station's next
  departure. The drop stays at departure rather than arrival: the dwell is 30 s by default (half a
  real second at the slowest speed), the account has no arrival time per stop, and at a terminal
  a train can stand for minutes before it boards in the account, so moving the drop to arrival
  would take a new per-stop time in every hour's slice for a half-second difference, and the map's
  fill and the panel would have to move together.

- **Station boxes are a fixed, narrower width; traffic lines may exceed them** (T-239, Anita:
  "a bit too fat still ... the lines can get thicker than the stations, that's not necessarily a
  problem"). A box is 75% of what a plain line and its rim cover at zoom 15, for every station
  in every mode, replacing T-204's cover-the-widest-line rule and T-227's growth for traffic
  widths. Dots follow (no growth either), so a dot and a box stay consistent. A wide line
  running under a station now shows beyond its rim. Stop editing markers lose the check, and
  reachable stations are drawn smaller than stops so the line's own stops stand out.

- **No homes in institutions** (T-234, Anita: "why are there homes on rikers island?"). The
  census counts people where they are held or cared for, so Rikers Island had 3,772 residents and
  Homes bubbles. Every pack's block population is now the 2020 census figure less its
  institutional group quarters (PL 94-171 table P5, per block: adult correctional, juvenile,
  nursing/skilled-nursing, other institutional), from each state's summary file on
  www2.census.gov: a national source every state has, no key (the Census API refused keyless
  requests). All four institutional types go: inmates make no trips, and nursing-home residents
  rarely leave and are not commuters, shoppers or pupils (nursing homes are the larger part, 0.5%
  of all people against 0.2% in prisons and jails). Non-institutional group quarters stay: college dorms
  (students travel, and the university purpose's adult homes need them), military quarters,
  shelters and group homes. LODES RAC also puts job holders on these blocks (Rikers: 676 on
  blocks where 27 people live outside the jails; likely address records and LODES's noise), so on
  a block with an institution the job holders are capped at the residents left; other blocks keep
  LODES as it is (blocks with more RAC than people exist everywhere and are LODES's business). The
  packs were changed in place (`pipeline/group_quarters.py`): `pop` and `commute_home` re-spread
  from the block table exactly as `build_city.py` spreads them, school and university homes
  rebuilt, gravity re-solved, 56 cells left with no people and no jobs dropped (as a build drops
  them); every other array byte for byte. The park draw's nearby-residents term was left for
  T-235: it needs a fresh Overpass fetch, and the public servers answered 504 for ten minutes running. Not
  done: prisons' staff are LODES jobs and stay; the census's household population on the same
  blocks (27 on Rikers, DP noise or staff housing) stays. notes/T-234.md.
- **Stop editing mode** (T-233, Anita: "when editing station stops we should display very
  prominently on the map where the current stops are. and hide most other stuff ... each station
  should have a top level bubble that we can click on map to either add it or remove it (toggle)
  ... stations that are reachable on top, and stations that are not reachable behind and like
  slightly greyer"). Stop editing is the line tool with a line (`stopEditing` in game/ui.ts), entered
  as before from Edit stops, New line or Copy line and left by Done, Enter or Escape. While on: the
  demand canvas, trip canvas, capacity tags and longer-trip labels are hidden by a page class
  (style.css); line widths and station circles by riders are not built (demandViews.ts), so lines
  draw at plain width; other lines are drawn at 30% (one renderer uniform, `dimOthers`). Every
  station gets a round marker (map/stopMarkers.ts): DOM over the map with no pointer events, picked
  by the map's own click and mousemove so the wheel and dragging still work under it. Reachable is
  graph-based on the main thread: **route** = the station's node is on an edge the line's path uses
  (full size, on top), **reroute** = joined to the stops by track but off the route (smaller,
  grey), **cut** = on no track network the stops lie on (smallest, greyest). Reroute stays
  clickable because the existing tool already reroutes: the worker's `insertStop` tries every
  slot and takes the shortest valid route, so the tag says "Click to add a stop and change the
  route". A click on a cut marker only shows why, in its own tag for 4 s. A line with fewer than
  two stops has no route to stay on, so what joins its stops counts as route; with none, every
  station does. Removing a stop is `setStops` without it, as the inspector's red crosses do (the
  route between its neighbours is the worker's, normally the same track). A click on a station's
  box with no marker under it toggles too, where it used to say "already a stop". The panel at the
  map's lower left shows "Editing stops on <line>: N stops" and a Done button. The other lines'
  trains fade to the same 30%, fill gauge included (the train shader takes `u_sel_line` and `u_dim`;
  Anita via the orchestrator: "hide most other stuff"); the edited line's trains stay at full strength.
- **Faint internal borders inside a complex's shape** (T-226, Anita: "bring back the internal
  borders of stations as like quite faint gray lines"). After the whites, each platform's own edge
  is drawn as a 1 px hairline along the middle of where its rim was, multiplied into what is
  drawn (white becomes about #d6d6d6). Multiplying rather than painting grey means the same
  hairline on the outer outline only darkens the dark, so the outline needs no special case; crowd
  fills go over it. Platforms only (bridges and dots have none).
- **Traffic thickness linear, 0.5 to 6.6 line widths** (T-227, Anita: "raise the max width by like
  3x ... the scaling should be linear. the most used line should be a fixed width like 3x what it
  is now and the least used lines should be like 0.5x normal width"). Replaces T-084's square
  root from 0.45 to 2.2 at a fixed 30,000 an hour. "Fixed width for the most used" is taken as
  relative: the network's busiest stretch in its busiest hour of the day (every period, the
  purpose shown) is 6.6 wide, so any city or player network gets the full range, and the widths
  still fall off in quieter periods. Station boxes and dots grow to cover the wider lines, as
  before. Picking and the track tag reach half a line's drawn width where that is over the 7 px
  slop, so a click anywhere on a wide line hits it.
- **Crowds on dots** (T-228, Anita: "it doesn't appear for when we are zoomed out and the station
  is a circle though, we should also highlight the circle slightly"). From zoom 11 to 15 the dot's
  white is tinted: towards the crowd grey by its fullest platform side's share of the room (at
  most 45%), or pale red (40% towards the over-full red) while a train at any of its platforms left
  riders behind; a complex's dot takes the whole complex. A tint rather than a fill, as asked, and
  because a dot is 4-15 px across. Its tag gives the riders waiting at it and the fullest platform.

- **After an edit only a final result replaces a final result** (T-232, Anita: "when i edit a
  line minorly the ridership goes up from 19% to 20.5% then back to 19% as it recalcs"). The
  demand client still receives every round, but a re-solve after an edit publishes only its last
  crowding round: until then every figure (top bar split, line and station riders, bubbles and
  their catch-up targets, the boarding account) keeps the last finished forecast, faded with
  "Updating demand 2/4…". Real NYC save, subway 1 cut to 2 trains: the split read 19.0% through
  the whole update (about 11 s, four steps) and then 18.8%, one change. Estimates are still shown
  where there is nothing final to hold, marked rather than held: at startup and after a
  whole-game load (T-172 starts the clock at the first estimate, and holding them would leave
  the panels empty with the clock running for another 7-8 s on New York), and for a city just
  opened (its controls would otherwise wait for the final round). The top bar then says
  "Estimate, refining 2/4…", with a hover that the figures settle when the forecast is finished;
  the other panels are not marked one by one, since they all show the same solve and the top
  bar's status is the game's one place for it. Cost: a query asked during an update (a station's
  catchment, a bubble selection, a trip) waits for the final round, because the workers drop the
  shown solve's state when the new one starts; keeping it would double their memory.
- **"N stations full"** (T-231, Anita: "a global alert in the top left under the demand buttons
  that says like n stations full"). A station complex counts once when, in the game hour now, a
  platform side holds as many riders as fit (`platformRoom`, the T-219 fill's own room, the whole
  platform's where trains use one side) or a full train has left riders behind this hour (the
  figure the station inspector's "Waiting now" shows, so the two agree). Left behind counts the
  whole hour's account, not only up to the minute, for that agreement. Ordered by riders left
  behind, then waiting; a click selects the next complex's busiest member and moves the map
  there at zoom 15 or closer, where platform crowds are drawn. A red-outlined button, not a
  banner: it is a count to click through, and it hides at zero.
- **Station complexes drawn as one shape** (T-217, Anita: "all the white rectangles overlapping
  seamlessly ... and then one continuous black outline surrounding the whole white shape ... and if
  we manually link, this white shape should include a thin bridge"). Every platform's dark shape
  is drawn, then every platform's white inside its rim, so where platforms overlap the rims inside
  vanish; this works for any overlap without knowing complexes (overlapping drawn boxes are always
  within the complex rule's 50 m anyway). A faded blueprint draws only its rim so its see-through
  white is unchanged. The player's links get a bridge: 3 px of white at zoom 15, fixed on the
  ground like the boxes, between the nearest points of the two platforms' centre lines kept 6 px
  inside each end (from the end the rim would cut across it). Only links get bridges: members the
  50 m rule groups but that do not touch stay apart, as Anita asked only for links. The selection's
  outline and glow are one draw over the union (each pixel takes the nearest selected shape's
  distance, shapes in a small texture), since one glow per platform doubled where they overlap.
  Picking stays per platform; bridges are not picked.
- **A complex is one dot zoomed out** (T-217, Anita via the orchestrator: "a complex should always
  be one dot ... picking selects the busiest member", generally on, no setting). Below zoom 15 the
  complex is drawn by its busiest station (boardings plus alightings a day; its first station
  before demand) as one dot at the centre of the smallest circle round its platforms' middles,
  large enough to cover them all, so the lines run into it. The smallest circle rather than the
  mean: the mean leans to the side with more platforms, and a linked pair with 16 platforms at one
  end got a dot nearly twice as wide as needed. A widely linked complex gets a big dot (a
  complex spread over 600 m is a dot 600 m across, 40-80 px at zoom 13-14); accepted, as it is
  what "covering its members" means. The line tool, delete tool and clicks zoomed out pick the
  busiest station too; the dot's tag names it and says "One dot for the complex's N stations".
- **Packed trains' red edge 50% thicker** (T-218, Anita: "not super noticeable"): 3.9 px (was 2.6;
  the white rim is 1.7). Far out, where a pill is under 8 px wide, a packed train is now almost all
  red, which reads as packed.
- **Platform crowds drawn as the platform filling up** (T-219, Anita: "visibly represent crowding
  on platforms as like people filling up the platform, sort of like how darkened space represents
  people filling up trains"). Each direction's half of the box fills with dark warm grey from the
  middle outwards to its riders' share of the room to stand on that side, red while the last
  train there left anyone behind; a platform with trains on one side only fills as one. Room: 2.5
  m usable width a side (half a 6 m island less the edge strip) at 0.3 m² a rider, the roomy end of
  Fruin's queuing level of service E (0.2-0.3 m², people standing in contact; the TCQSM's platform
  scale), about a train's standing density at crush, so a full platform and a full train mean the
  same crush; 1,250 a side for 150 m. From the middle because a platform, unlike a train, has no back
  to fill from, and a fill growing both ways stays centred on the station's label and node. The side bands (T-164, T-199) and "Crowds as dots" are gone: the bands lay over the
  lines beside the platform and needed their own hover reach, and a dot pattern inside a 4-10 px
  half box did not read; Settings has one checkbox, "How full each platform is". A train standing
  in the box hides the fill's middle at zoom 15 (the train is 9 px of a 12.6 px box); from 16 the
  sides show. Only the drawing uses the room: the account still lets a crowd grow past it.

- **The City tab is gone** (T-220, Anita: "now that we have these city bubbles we can get rid of
  the city panel entirely. its kinda redundant"). What it offered and where each went: (1) the
  focused city's name: the top bar already shows it; (2) its intercity total and the top five
  connections with train/other bars: the city overview's bubbles (zoom 8.5 and below) and the
  city inspector, which lists every place with all-mode and train counts; the bars were dropped,
  since the inspector has the numbers and the overview colours each city by its train share;
  (3) City centre: a "Go to <city>" button in the city inspector (the same jump as a city picked
  in search, which also clears the selection so the city's own bubbles show) and search itself;
  (4) Whole region: dropped, zooming out shows every city as one bubble; (5) Explore a trip: a
  button at the bottom of Settings (Anita's suggestion), key 4 unchanged; (6) its "Calculating
  demand" text: the city inspector has its own. Nothing else depended on the tab: no key opened
  it, no selection flow switched to it. A browser that last had it open starts on Lines. The
  notes/*.mjs checks that clicked it (t113, t116, t116-real, t134, t135, t137) go to cities by
  search instead, and t116's Whole region is a fitBounds.
- **A trip's end at a clicked station is the station's own cell** (T-221, Anita: "we should be
  able to explore a trip by clicking two stations"). With the trip tool on, a click on a station
  takes that station as the end: the trip is asked about at the station's position, which the
  model reads as the nearest cell. The model's subzones are the cells of a zone that share their
  nearest station, so the station's cell is in the subzone whose nearest station is this one: its
  walks are the shortest to it and the trip mostly boards there. The busiest cell of its walk
  catchment was the other option; it can lie nearer another station, so the trip would mostly
  start from that one. No worker change: the query already takes points. The model still spreads
  riders over nearby stations (Jay St-MetroTech to Fulton St boards mostly at Borough Hall, 1 min
  away), which the inspector shows as it does for any trip. A station's inspector has "Trip from
  here" (the tool on, the station as the start), since starting from a selected station is the
  natural way in. A capacity tag sitting on a station passes the trip tool's click through to the
  map (it swallowed it before).
- **Longer-trip bubbles are always drawn** (T-222, Anita: "we always want that on"). The Settings
  check "Bubbles outside the selected city" and its remembered value are gone; the bubbles still
  give way under a local purpose filter, since the regional market has no purposes.
- **Five more default line colours** (T-223, Anita: gray, yellow-green, pink, magenta, light
  blue): gray #74797e, yellow-green #9ccb19, pink #f59dc6, magenta #e62aa6, light blue #5cc3f0,
  after the first eight. CIEDE2000: the closest pair in the thirteen is purple and magenta at 19.1
  (the first eight's closest, orange and yellow, is 20.1); each new one against the old eight at
  least 19.1, against each other at least 21.2. Yellow-green, pink and light blue take dark
  letters on their chips (`inkOn`, contrast about 7); gray 4.4 and magenta 4.0 with white, better
  than the orange's 2.5. Against the mode colours at least 17.9 (yellow-green and walk green). Some
  sit near purpose colours, which only appear in panel bars with their names beside them: pink
  4.9 from medical, gray 9.2 from longer trips, yellow-green 14.9 from airport (the teal already
  sat 5.6 from school). Magenta was kept off the purple (a first try, #b5179e, was 10.5 from it)
  and 15.6 from a selected station's catchment colour. Rejected: a deeper pink with white letters
  (#e8559a), which came within 15 of the magenta.

## 2026-10-12

- **The trip explorer draws rides along the track, and opens from a far-end bubble** (T-208,
  Anita: the path should follow "the rail paths rather than being straight lines between
  stations"; "click a home bubble and then click one of the destinations, it should route between
  them"). A ride is the line's own drawn strokes over the stop-to-stop stretches it rides (stop
  indices carried from the solve's route nodes), and falls back to straight lines only when the
  drawn network's ends no longer sit on the stations (an edit since the solve). Walks stay dashed
  and straight: the model's walks are straight-line distances, so a street path would claim
  more than it knows. From a selection, each side is one cell, the busiest: the selection's
  cell with the most legs at its end, the far bubble's cell with the most of the selection's
  trips (a station's far zone: its cell with the most rail legs at that end, since the riders'
  own cells in the zone are not on the main thread). The bubble's drawn centre was the other
  option, but as a weighted mean it can sit in a park or the river with nobody, giving an
  "outside the city" or unrepresentative answer. Longer-trip (regional) far ends do not open it:
  the explorer reads one city's cells. The inspector keeps a Back button to the selection.
- **Complexes are transfer points in demand, as walk edges between members** (T-161, Anita:
  "complexes and manual complexes should allow transfer"). Replaces "complexes stay
  display-only". A rider leaving a train may walk to any other demand station of its complex
  however far, besides the walks within 300 m; the clock worker sends each constructed station's
  complex with the snapshot (`game/complexes.ts` over constructed stations, links included) and
  each demand API takes them with `set_complexes`. The walk costs what every walking transfer
  costs (centre to centre x 1.3 at 1.2 m/s, weight 1.39, then the boarding's wait and 5-minute
  penalty), with no extra fixed penalty, because the 5 minutes already fall on every change and a
  second constant would make a change between two platforms 120 m apart jump in cost the moment
  they became a complex. Members stay separate demand stations joined by edges rather than one
  merged station: merging would put a complex's catchment at one mean point, make a 1 km link a
  free teleport, and lose the platform-to-platform transfer counts the complex map draws (T-176).
  The 100 m merge and the 300 m walks stay as they were. Measured on the real New York save:
  nothing moves (every one of its 855 complexes is one demand station or a few within 300 m of
  each other, since the merge is single linkage), so it matters for platforms more than 300 m
  apart and for the player's links (notes/T-161.md).
- **The player links stations into a complex by hand; only those links can be undone** (T-197,
  Anita: "be able to create interchanges, linking stations into a complex even if they wouldnt
  normally"). From the station inspector ("Link to another station", then a click on the other
  station on the map); up to 1 km centre to centre (real interchanges by passage are a few hundred
  metres; 1 km keeps a complex a place and its transfer walk under 25 perceived minutes). The
  links are part of the network (`Network::links`), saved as an optional trailing block (`LNK1`)
  written only when there are any, so older saves load and a save without links is unchanged.
  They are a setting outside the undo history, like fares, which keeps them off the track
  model's edit operations; removing a station drops its links. Stations the distance rule groups
  cannot be split: their platforms nearly touch, those within 100 m are one station for demand
  anyway, and an "unlink" list of exceptions would be a second thing to keep in step for no
  case anyone has asked for.
- **School and university are one shown purpose; the model keeps two** (T-192, Anita: "we should
  probably merge school and university demand"). The filter, pie, purpose split, timing charts and
  every tooltip show "School and university" (the longer name over "School", so nobody wonders
  where students went), in school's blue-green; `game/purposes.ts` `membersOf` lists which model
  purposes a shown one counts, and the demand client sums them for the map. The model keeps them
  apart because their homes (children against adults), places and timing differ, and merging
  there would change the solve. University's amber is retired: the closest purpose pair is now
  medical and social at CIEDE2000 22 (airport had been 19 from university). The "(metro public)"
  and "(metro IPEDS)" source tags are gone from names; the inspector's note says what is covered.
- **Complex view is the default** (T-192, Anita). Stored settings saved the old default with
  everything else, so a settings version (2) drops a stored complex view once; Anita's browser
  takes the new default without her switching it.
- **The demand bar has no background** (T-192): its background showed as a white strip in the
  6 px between Destinations and the purpose filter. The buttons carry their own; the loading fade
  is now faded text rather than opacity, so the map does not show through a faded button.
- **One timing chart a purpose, outward up and home down** (T-193, Anita). Replaces the two rows
  (To destination, Return home) with a mirrored chart on one scale of departures an hour, which
  halves its height and puts the morning-out, evening-back shape in one picture. Kept per purpose
  rather than one stacked chart, so a small purpose's timing stays visible. Numbers stay in the
  hovers. Weekend days are left for T-170.
- **Network things get the track tag** (T-194). Trains (riders aboard against seats and packed
  full), stations (demand circle count, riders waiting on each side), crowd bands (as the
  station) and demand circles away from stations share the track tag, its 0.1 s rest and pick
  order, rather than a second tag: one place by the pointer for the network, and the same rules
  for when it shows. Circle counts are recomputed from the demand result rather than read from
  the renderer (another agent's file). Station and train counts were in the inspectors only.

- **The trip explorer reads a trip as a commute, with the model's own numbers** (T-195,
  `sim/src/demand/trip.rs`, docs/demand.md 4.4, docs/interface.md). A new worker query
  (`DemandApi.trip`) recomputes the one subzone pair's choice with `mode_choice`'s expressions
  in the same order on the period's graph, so the chances are the solve's f32 numbers (checked by
  adding a whole origin's pairs back up to the solve's rail trips, 1e-6), rather than a JS copy of
  the formulas that would drift. Choices: **(1) a commute** from a home at the first click to work
  at the second: the commute population is the one whose state the query can read without
  touching the everyday code (another agent's), and it is most trips; everyday purposes use the
  same rail and road costs but their walk share and drive distance come from the sampled cells.
  **(2) The last crowding round's costs**, the graph a period keeps; the published loads are the
  rounds' average, so a trip's chance can differ a little from the average that made the loads.
  **(3) Routes, not a path**: rail is a strategy (station pairs by the access logit, departures
  by frequency), so the answer lists every route carrying at least 0.5% of the trip's rail riders
  with its share, and the panel calls the biggest the route and says how much of the riders it
  carries (17% for Park Slope to Midtown, where riders spread over neighbouring stations).
  **(4) All five periods at once**, each from the worker holding it (a few ms each), so the
  panel's period rows switch at once; the clock's period is shown until one is picked. **(5) Key
  4 from any tab, the tab left as it is**, and a button in the City tab, since it reads demand
  rather than builds; clicks alternate between start and end, and the trip is a selection that
  outlives the tool. **(6) A veil of the land colour at 60% over the map** while a trip is drawn:
  the route rides lines drawn in the same colours, and without the veil it vanished into them.
  Rides are drawn straight between the stations called at, not along the track (the renderer's
  geometry is another agent's, and a stop-to-stop line reads well at trip scale).
- **Each train its own load, each platform its own crowd: the boarding account** (T-164,
  `sim/src/track/boarding.rs`, docs/track-and-service.md 6.6; Anita's middle option from the
  brainstorm, notes/T-164.md). Demand's boardings per line, direction, stop and period arrive at
  the platform over the period; each train lets off the line's share alighting there and takes
  whoever fits up to crush; the rest wait for the next train. Choices left open by the
  brainstorm, taken as the simplest: **(1) the in-period shape** is a fixed shape per period
  scaled so the busiest clock hour is the demand model's busiest-hour factor (T-087) times the
  mean, a tent for the peaks and midday, falling for the evening, highest at both ends at night,
  rather than a smooth curve across the day: it keeps each period's total exact and needs no new
  input; the rate jumps at period boundaries, which nobody sees in a crowd. **(2) Common lines
  keep separate queues**: demand already splits a stop's riders between lines by frequency, so a
  shared queue would mostly redo that; a rider left by a full train does not switch lines.
  **(3) A day at a time, not an hour ahead**: the account is worked out for a whole day of
  departures whenever the service or the riders change (from 03:00 from empty platforms; again
  from a day earlier for a direction still queueing at the end), and each hour's slice is read
  from it, so it is the same whenever the player looks and nothing has to be saved. New riders
  are worked out 3 lines at a time between the clock worker's messages (40-60 ms in all on the
  real NYC save, steps of about 5 ms). **(4) Trains are never over full**: what does not fit
  waits, so the red-rimmed over-full train is gone except for a line the account has no riders
  for yet. **(5) The Average / Busiest hour setting (T-087) is removed**: each train has its
  own load. **(6) Crowds are drawn as a band beside the platform**, one side per direction, in
  steps (under 100, 300, 1,000, 3,000, more), train blue, red while the last train left anyone
  behind; dots in the same band are a setting, as the brainstorm asked for options to show
  Anita. **(7) Left-behind waits do not feed back into demand** (phase 2, Anita's call).

- **Airport ground trips and medical visits are everyday purposes five and six** (T-142,
  `everyday.rs`, `enrich_airports.py`, `enrich_medical.py`). Both from sources that exist for
  every US city, and globally where possible. **Airports** keep their own totals: each airport's
  daily outings are its FAA boardings x its DB1B O&D share x its catchment's share inside the pack
  / 366, and the kernel spreads them over residents and visitors (half population, a quarter jobs,
  a quarter leisure jobs, none within 1.5 km of a terminal) by a 20 km decay normalised per
  airport. Not a per-resident rate: that would give Springfield an airport and ignore that
  Newark and Tweed differ forty-fold. Connecting passengers come off (DB1B, national and free:
  Dulles and BWI connect a quarter to a third, Boston almost none); staff are commuters already.
  Terminals from OSM, so a station has to reach the terminal, not the runways. Trips to an
  airport in another pack (Springfield to Bradley, Washington to BWI) are dropped, as for every
  everyday purpose. **Medical** is a rate per resident, 0.04 outings a day (NHTS 2022: 4.02% of
  2.27 trips a person, 0.091 legs; the model's errands and social rates sit similarly under
  NHTS's), to OSM hospitals, clinics, doctors and dentists sized in visits a year (a hospital
  1,000 per bed, a clinic 10,000, a practice 4,000, a dentist 3,000). Doctors and dentists are in
  because hospitals alone would pull every office visit onto a hospital campus; OSM maps them
  unevenly, and where they are missing their visits go to the nearest mapped place. Untagged
  hospitals are sized by their buildings' floor area (footprint x levels), not their site's area:
  site area put sprawling suburban psychiatric and VA campuses above every Manhattan hospital.
  Colours olive and pink, the two hues left that stay about as far from the others as those are
  from each other (green is kept for events). **Two destination samples a home cell** for both,
  not sixteen: with sixteen a city's everyday opening took 70-90% longer (the cost is per
  sample); with two it takes 5-15% longer natively and the aggregates are unchanged (medical
  walk share within 0.3 points, every big airport within 0.4% of its input).

- **Transfers are counted in the assignment, every round, and drawn line to line** (T-176,
  `kernel.rs` `assign_with_transfers`, `ui/inspector/ComplexMap.tsx`). The assignment now notes
  each rider leaving a train at a station that is not their destination and splits them over that
  station's departures like everyone there, keyed by arriving and leaving route node (platform to
  platform); the loads are untouched (bit-identical natively and in the WASM worker). Counted in
  every round and averaged like the loads rather than on the last round only: it adds about 5 ms
  to a 36 ms commute assignment and is lost in the noise of a 1.0-1.4 s stage, while the last
  round alone would be off by that round's change and missing when an uncrowded round is skipped.
  Sent only when asked, for one complex's route nodes (a few kB), never with every result. The
  complex map joins lines, not platform bundles as T-147's notes planned: the real save's Penn
  Station and Grand Central are three bundles each, so bundle to bundle gives at most three
  curves and hides which lines change; the busiest 8 line pairs between different bundles are
  drawn, changes inside one bundle go in each line's hover. Curves sit over the platforms (under
  them they vanished beneath Penn's commuter bundle). docs/demand.md 4.4, docs/interface.md.

- **Towns and neighbourhoods come from a per-city OSM place list, beside the pack** (T-177, T-065,
  `pipeline/places.py`, `game/placeList.ts`). OSM `place=city|town|borough|suburb|quarter|village|
  neighbourhood|hamlet` inside each pack's boundary, one Overpass request a city (cached in
  `data/raw/`), same-named rows within 2 km merged; 18,397 places over the nine cities, about
  730 KB. A separate `<id>.places.json` rather than a block in the `.bin`: it is not demand input,
  it can be rebuilt in seconds without touching the pack, and it loads only when the search box is
  first focused or the station tool first picked, never at startup. No admin boundaries: towns
  are already `place` points and a containment test would add nothing the search uses. Hamlets
  are kept (much of Long Island, New Jersey and Maryland is tagged that way). Search ranks places
  after stations on equal match (bigger kinds, then population, first) and also matches clipped
  nicknames ("south philly"), the only general rule that finds it, since OSM gives South
  Philadelphia no alt_name. A new station is named from the list when the map is below zoom 13,
  where the basemap's tiles hold only the bigger roads (a nearest loaded street there is not the
  nearest street), or when no street is within 400 m; in use, it takes another place at most
  800 m further, else "North Williamsburg", else "Williamsburg 2".

- **The regional market measures zone distances on the ground** (T-128, `demand/intercity.rs`,
  `pipeline/intercity.py`). Its decay and drive times used straight distances in its equirectangular
  New York pack frame (1.2% short east-west at Philadelphia, 2.7% at Boston). Now the pack names
  its frame (`frame`, format 2) and the kernel measures zone centres by great circle, as the world
  frame's lengths are checked against. The sites stay in the pack frame, so positions, selections
  and the app are untouched. Site lon/lat are not stored as T-128 first proposed: the frame is
  affine in lon/lat, so it gives the zone centres' lon/lat exactly, and storing them would add
  about 5 MB to the download. A pack without `frame` keeps plane distances under the old formula
  name (its block stays current, its numbers unchanged). Not re-tuned, under "intercity demand is
  rough on purpose": the day's trips are unchanged (696.1k), corridors move up to about 3%, market
  rail on the real NYC save +0.07%. docs/demand.md 4.4.

- **Capacity tags move off station names; names never yield** (T-184, `map/capacityMarkers.ts`).
  Tags sat at one fixed spot above right of their resource, which at a busy station is where its
  name is ("+9:17" on Times Sq, "+5:39" on Union Sq). Names are the map's orientation, so they
  are placed first and keep their spots; each tag then takes the first of six spots around its
  resource (above right as before, above left, below right, below left, right, left) that clears
  every shown name by 3 px and the tags already placed (with T-102's spacing), and hides only if
  none does. A corner of the tag stays 5-6 px from the resource, so it still reads as that
  resource's. On the real NYC save in Midtown at zooms 12-15 and through a pan, overlaps went from
  1-5 a view to none, with about the same number of tags shown. Not moving the tag along the
  track (the resource is a point; a tag far down the line reads as another station's), not
  hiding names under tags, and not MapLibre collision (both are DOM layers over the overlay
  canvas, T-032). `StationLabels` exposes its shown names' boxes and tells the tags when they move.

- **A loading card in the middle of the map while the clock waits** (Anita, T-180). T-172 moved
  the startup load into the top bar so nothing blocked play; Anita wants it obvious until the
  clock starts. While a loaded game's clock is held a plain card centred over the map says
  "Loading New York, 42%" and the step; it takes no pointer or key (a press on it pans the map),
  has no animation, and goes when the clock starts. Later loads (a pan to another city) stay in
  the top bar, whose status now ends in an ellipsis in a narrow column instead of running over
  the cash. Not a dimmed overlay or a progress bar: the map stays fully visible and usable, and
  the percentage is the motion.

- **Hover tests skip far segments in local units** (T-180, `map/pick.ts`). Every pointer move
  scanned all 117,701 line and track segments of the real NYC save, projecting each end into two
  new arrays (about 5 ms a scan, several a move; hovering kept the main thread 70% busy). The
  pointer is now taken into local units once and a segment whose grown box misses it is skipped
  before the old arithmetic; the answers are identical (4,000 checked). Not a spatial index: the
  box test makes a scan 0.66 ms, and an index would have to follow every edit.

- **The regional market goes to the demand workers as bytes** (T-180). The main thread no longer
  decodes the 8.2 MB pack or clones its text six times while the player can already pan; each
  solving worker decodes its own copy, the aux worker gets none.

- **One rule for which keys a focused control keeps** (T-183, `ui/keys.ts`). main.ts and the
  search box disagreed: search treated a select as a text field and ignored `/` and Ctrl+K,
  while the game's Space, 1-3 and Esc fired from it, so after picking a purpose "/times sq"
  searched nothing and the space paused. Now both ask `keyIsTheirs`: a text field keeps every
  key; a select, slider, checkbox, radio or colour input keeps only the keys it works with
  (arrows, Home, End, Page Up and Down, Enter, Space, and a letter in a select), and everything
  else reaches the game; a button keeps nothing (Space pauses, as before). Before, a slider or
  checkbox swallowed every game key too. A select picked with the mouse also blurs, so Space
  pauses straight after a pick instead of reopening the list; a keyboard user keeps focus and
  the arrows, which is why the blur is mouse only.

- **Same-named search results in one city are told apart by their own lines** (T-183). Two
  complexes more than 50 m apart can share a name (Times Sq-42 St and the 8 Av A/C/E platforms
  in the real NYC save), and every line's badge in that save is "S", so the chips do not help.
  The row's second line lists the lines only that result serves ("subway A, C, E"), which is
  what a player knows their network by; a compass direction from the others is the fallback when
  a result has no lines of its own (blueprint stations). Results in different cities keep the
  city.

- **New York's schools come from NCES like every other city** (Anita, T-181). The NYC DOE
  building input for the five boroughs (T-140/T-150) is removed from the pipeline, under the "no
  city-specific data" rule. New York's pupils rise from 2,694,630 to 2,825,244: NCES includes
  charter schools (141,775 pupils in the boroughs), which the DOE building report did not carry.
  School outings +4.9%. notes/T-181.md.

- **Washington's border factor is its own fit, 0.49** (Anita, T-181), replacing the US 0.23
  there. DC is a city counted as a state, so a factor fitted on states between New York, New
  Jersey and Pennsylvania cut DC's commuting to Maryland and Virginia to a quarter of LODES (3-4%
  of DC's commuters against 13.5% each); with 0.49 it is 10%, and the between-state share
  matches LODES (25.0% against 24.9%). A city config's `border_factor` overrides
  `BORDER_FACTOR_US` in `build_city.py`. This replaces T-162's "one value for every US pack".

- **Borders outside the US: national borders get a factor, first-level regions none by
  default; inside the Schengen area the national factor is weaker** (Anita, T-181). Values are
  fitted when the first non-US pack is built. The US states keep theirs because LODES measured
  it (T-159, T-162); elsewhere admin-1 lines are not assumed to cut commuting (Tokyo's
  prefectures, city-states such as Berlin or Vienna). Settles T-162's open question.

- **Anita's UI answers** (T-182). A search hit for a complex's shared name selects the member
  with the most lines and leaves the Station / Complex setting alone: it is a remembered setting
  that also controls the demand circles, so a search should not change it (replaces T-148's
  "lands on the complex"). The City pane's City centre and Whole region buttons jump like search
  does, the house rule of no animations; those were the only animated camera moves in the code
  (MapLibre's own gesture easing is untouched). The line inspector's Local trips and Longer trips
  rows are gone: the purpose split under Riders a day (T-165) already shows longer trips, and the
  demand result no longer carries the line's regional boardings separately (closes T-175).

- **No new cities until the general design settles** (Anita, T-179). Nine cities are enough for
  now; more packs would make it harder to change things not yet nailed down (the market, the
  purposes, the gravity). T-179's batch was undone; `docs/adding-a-city.md` keeps the method.

- **The regional market's all-pairs figures are derived offline; zone pairs are built only
  between served zones** (T-174). Opening the market visited every zone pair twice in every
  worker (1.2-1.6 s in Chrome, about 80 MB a worker, both growing with the square of the zones:
  30 cities would not have fitted). Chosen over the T-172 suggestion of shipping the pairs
  themselves in a binary pack: those would be 2.3M pairs now and about 20M at 30 cities to
  download and hold in every worker, and would freeze the pair formula's output into the pack.
  The block holds only O(zones) numbers computed by the game's own code, names its formula and
  is ignored when stale (the game derives it at opening, same result), so the market's design,
  which Anita has left open, can change with a one-line rerun (`pipeline/intercity.py
  --derive`). Only served-zone pairs can carry rail, so building just those changes no number.
  Same outputs to the bit natively, 1.2e-7 at most in the browser (libm). notes/T-174.md.

- **One demand worker warms the everyday code up before the others open a city** (T-173). Five
  workers opening New York's everyday layer at once from the browser's baseline code took 7-8 s
  each in most page loads, against 1.3-1.8 s on optimised code. Worker 0 opens a small synthetic
  everyday layer at startup (150 ms) and the other solving workers get city packs once it is
  ready, so they open on the optimised code it leaves (code is shared by a page's workers). New
  game clock 2.7-3.1 s in 10 of 10 loads. Chosen over holding the others until worker 0 had opened
  New York itself (also fixes it, but 4.2-4.9 s: they open after it). Undo or compare with
  `?perfOff=stagger`. notes/T-174.md.

- **Every registered city gets New York's non-work purposes, from national sources only**
  (T-142, under Anita's "no city-specific data"). Everyday outings from LODES CNS sectors (already
  cached), school attendance from NCES EDGE/CCD public schools, university attendance from IPEDS,
  with New York's constants (rates, decays, attendance factors, timing) unchanged and no per-city
  calibration. The runtime opens the everyday layer for any pack that carries its arrays instead
  of New York alone. Per capita the inputs land near New York's (school 0.13-0.15 pupils a
  resident against 0.12; university 0.035-0.078 students against 0.038, highest in the college
  towns of Springfield and New Haven), which is the check, not a target. New York keeps its DOE
  buildings (Anita's open question).
- **A merged college's students are split over its absorbed campuses** (T-142). IPEDS reports an
  institution that absorbed others in the data year at one point: Connecticut State Community
  College put all twelve former Connecticut community colleges (25.6k students) in Hartford. The
  split uses each campus's own IPEDS enrollment the year before, a rule that applies to any
  merger anywhere in the US, not a Connecticut fix. It also added 5,330 students to New York
  (Housatonic and Norwalk in Fairfield County, which New York had silently lost). Multi-campus
  institutions that have always reported as one (Northern Virginia CC) stay at one point:
  nothing national gives their campus split.
- **The everyday pass searches only towards stations its pairs can reach** (T-142), as the commute
  pass does, instead of towards every station. A city opened beside the one being built in, with
  no pairs near the network, then searches nothing. Each destination's search is independent, so
  no number changes (T-149's worker check: every load bit-identical).

- **The complex map draws per-line riders now and leaves transfer flows for later** (T-147).
  The demand assignment does not count transfers at any level (a rider leaving a train joins the
  station's flow and loses which line they came from), and drawing a split nothing measures is
  not done, so the map shows platforms and each line's riders there, and counting transfers in
  the solve is its own task (T-176, notes/T-147.md: where in `assign`, cost, and the strongest 8
  pairs to draw). Width is riders on the line's trains at the platform, not boardings: it says
  how busy the line is there, which a line through a hub with few boardings still is; boardings
  stay in the list under the map and on hover.

- **Platforms on top of each other are drawn as one bundle in the complex map** (T-147). Parallel
  within 15 degrees and centre lines within 12 m (about one island platform's width). The real
  New York save gives every service its own station, so true positions alone stacked Penn's and
  Grand Central's commuter lines into one unreadable pile; side by side, the way the main map
  draws lines sharing track, every line can be seen and clicked. Positions and headings stay true
  for everything else.

- **Complex view enlarges every member's platform box** (T-163), with a per-station flag in the
  renderer rather than a second draw. Picking takes the enlarged boxes, as for the selected one.

- **Search sits at the map's upper right, opposite the demand selector** (T-148). The top bar is
  full and already folds into rows at narrow widths, and a row in the dock would push the
  inspector down for good; the map's upper right was empty (attribution and tool hints are at
  the bottom). It is a plain 1 px input like the selector's controls, with the results in a list
  under it. When the map is narrower than 680 px it moves under the selector rather than overlap
  it. `/` or Ctrl+K focuses it: neither is used by the game, and they are what search boxes on the
  web use.
- **A search pick jumps, it does not fly** (T-148). The camera moves at once, like every other
  state change in the game (no transitions); the City pane's City centre button still flies.
- **A search hit for a name shared by a complex's platforms lands on the complex** (T-148). One
  result for the name; picking it selects the platform with the most lines and turns complex view
  on (the inspector's own Station / Complex setting), because the player searched for the place,
  not for one of its platforms. With a build tool on, a pick only moves the camera, so a half-drawn
  route or a line's stop editing carries on where the player went. Equal matches list cities
  before lines and stations, so "new" offers New York and New Haven before Newark's stations.
- **No geocoder for neighbourhoods or addresses yet** (T-148). Every free geocoder sends what the
  player types to a third party, which is Anita's call; the basemap's tiles hold neighbourhood
  names only in zoom-14 tiles already loaded, and no addresses. The suggested route is a place
  list per pack from OSM built by the pipeline (T-177, notes/T-148.md).

- **Riders by purpose in the line and station inspectors: a thin bar of every purpose, the
  filtered one ringed** (T-165). Under the map's purpose filter the split keeps all six purposes
  rather than switching to the filtered purpose alone, and rings that purpose's segment in the
  bar and bolds its key entry. Why: the inspector's totals stay all-purpose (T-158 kept train
  fill, crowding and these totals counting everyone), so a split of that total is only coherent
  with every part in it; the ring answers "how much of this line is the purpose the map shows"
  at a glance, which is what a thin line on the map makes the player ask. Other purposes are
  not faded, because the key is the whole palette and an authored colour is never faded. A bar
  with a key rather than the pie: the inspector is narrow and the figure sits under a single row;
  counts are on hover, shares in the key. Zero purposes stay listed (muted text) so the key does
  not change shape from line to line.

- **Startup never blocks input; the clock waits for the first estimate, not the last crowding
  round** (T-172, replacing T-149's loading screen). Anita asked for input blocked until the
  viewed city was fully loaded, then found the wait too long (production: 12-20 s new game,
  32-38 s on the real New York save). Now the map, inspectors and building work once the network
  is drawn (0.3-1.4 s); the top bar shows "Loading New York, 42%" and the mode split "–", and
  only the clock is held, until every period's free-flow estimate has landed for every city the
  game needs (new game typically 4.3-5.4 s, NYC save 7.5-8.2 s). Why the first estimate: after
  any edit the game already plays on the newest published result while crowding rounds refine
  it, so startup now follows the same rule; the rounds move a big network's figures by a few
  percent and land 2.5-8 s later. Why still hold the clock: with no demand at all, fares would be
  zero for the first game minutes. The same applies to a file, slot or new game loaded later.
  The failure screen (Try again, Play without demand) stays. Easy to undo: `READY_ROUND` in
  `workers/demandClient.ts` (3 restores waiting for the last round); the screen is
  `ui/LoadingScreen.tsx`. notes/T-172.md.

- **The regional market's free-flow estimate routes with the distance fare** (T-172, fixing
  T-121). Its round 0 used a graph without fares, so it ran a strategy search of its own: 1.8 s
  natively a period on the real New York save, and in five Chrome workers at once about 7.5 s of
  that save's 10 s first estimate. Now it shares the local populations' search, as every crowding
  round already did. Effect on the real New York save after three rounds: market rail trips 2,375
  to 1,886 a day (-21%), market boardings 7,388 to 5,835; New York commute and everyday rail
  change by under 0.005%. T-121 left this to Anita because the market had been tuned against
  the old figure; decided here because Anita has since said intercity demand is rough on purpose
  (CLAUDE.md, 2026-10-12) and the old estimate was the inconsistent one. Two lines to undo
  (`set_crowding` and `uniform` in `IntercityApi::solve`).

- **Work gravity: a state line multiplies the decay by 0.23; the decay length goes from 29 to 40
  km** (T-162, implementing T-159). Fitted on a coarse grid against LODES OD, k set so the share of
  commutes between states matches, scored on trip lengths and county pairs. One value for all US
  packs because New York's own fit (0.235) and Philadelphia's (0.219) agree within 7%, and Boston
  (0.29) and Providence (0.35) are near; Washington's own fit is 0.49 (DC is a city counted as a
  state), but 0.23 still beats no factor there. The decay shape stays d^-1 (a d^-0.9 shape scored
  1.1 points better out of 23, not worth a second change). The factor is a per-pack value and a
  pack without one has 1 (no effect): outside the US an admin-1 factor would be a guess (Tokyo's
  prefectures, city-states like Berlin), so each country sets its own from its own commuting
  data, and which borders get one is Anita's call. Replaces T-072's county-pair K-factors and
  road travel time: one parameter fixes the NJ overshoot, and road time closed only a tenth.
  notes/T-162.md.

- **One map label per name per station complex** (T-169). A complex's stations that share a name
  (the real NYC save has 14 platforms named Times Sq-42 St/42 St-Port Authority Bus Terminal, in
  two complexes) printed the name once each on top of itself. They now share one label at their
  mean position, turned like the member with the most lines and ranked by the lines at any of
  them. The complex, not distance, decides: same-named stations in different complexes (8 Av and
  7 Av at Times Sq) keep one label each, and differently named members of one complex keep theirs.

- **Varying demand: Anita's answers in the T-170 brainstorm.** No random disruptions. No separate
  weekend schedules for now. Superholidays (Asian ones especially) get special treatment.
  Stadium events matter most as points that make bursts of demand on some days; generated,
  realistic-looking game dates are fine. Event notices only for places the player has built in;
  elsewhere events are silent. Delays, if simulated, are tracked per city. notes/T-170.md.

- **The purpose filter keeps per-purpose loads beside the total; it never re-solves** (T-158).
  The populations were already separate per worker (city commutes, NYC everyday, regional
  market); only the everyday layer mixed its four purposes into one assignment. It now assigns
  them purpose by purpose along the same strategies, reusing one station-pair matrix, and sums
  them into the total crowding reads. On the real NYC network that costs about 65 ms per
  period and stage (assignment 98 ms against 32 ms) and no extra matrix; a matrix per purpose
  would have saved part of that time for about 9 MB more WASM memory per worker. Re-solving per
  purpose was never an option: mode choice and crowding must see everyone together.

- **Purpose filter: a dropdown at the end of the None/Homes/Destinations bar, not in settings**
  (T-158). It is a way of looking at the map, chosen while looking, like Homes/Destinations;
  settings hold things set once. Seven choices: All purposes, Work, School, University,
  Shopping and errands, Social and leisure, Longer trips (the regional market loads lines too, so
  it is a segment). A swatch in the pie's purpose colour (T-143) marks a filtered map; longer
  trips take the pie's neutral grey. Not remembered across page loads, so nobody returns to a
  map that silently leaves riders out.

- **What the purpose filter changes, and what it does not** (T-158). It changes line widths,
  station circles (complex view too), the Homes/Destinations bubbles and a selection's bubbles,
  their hover counts and the track tag, which names the purpose. Train fill and crowding count
  everyone aboard: a train is as full as it is. Drawings with no such purpose hide rather than
  show another segment: longer-trip bubbles show only under All or Longer trips, a station's
  commuter catchment only under All or Work. Scales stay absolute, so a small purpose (school,
  longer trips) draws thin and faint against the same full width; that keeps purposes
  comparable. Inspectors keep all-purpose numbers, except a bubble selection's counts and places,
  which match its bubbles and say so; its purpose pie keeps every purpose.

- **Crowds and refused boarding come from a per-train account, not from individual passengers**
  (Anita, T-164). Demand stays counts per period; each line's departures then board what fits up
  to crush and leave the rest on the platform for the next train. She chose this over simulating
  people, and allowed coarser treatment out of view. Signals stay a separate question (notes/T-164.md).

- **Station complexes: platforms within 50 m, derived, not saved** (T-146). Two stations are one
  complex when their platform centre lines (a straight segment of the platform's length along
  the track heading) come within 50 m, or their centres within 100 m (the distance at which the
  demand kernel already merges stations, so every demand station sits inside one complex and a
  complex's catchment never counts one twice); single linkage, levels ignored. 50 m is about
  where the platform boxes touch on screen at the zooms players build at (zoom 14-15), and
  about a street's width between real platforms 10-15 m wide. On New York's real network it
  gives the complexes a rider knows (Times Sq, Grand Central with Metro-North, Herald Sq with
  PATH 33rd St, Atlantic Av with Atlantic Terminal, Jamaica with Sutphin Blvd). Complexes are
  worked out from the stations' geometry on the main thread whenever stations change: no save
  format change, no worker change, and nothing to keep in step when a station moves. Complex
  view is one remembered display setting, shared by the station inspector's Station/Complex
  switch and the demand circles. It aggregates display only; demand transfers keep their
  100 m merge and 300 m walk (T-161).

- **Add NYC public-school attendance using enrollment, not employee jobs** (T-140). Match DOE
  buildings once to reported enrollment, then estimate homes from child-population weights.
  Destination budgets avoid mistaking staff counts for pupil capacity. The initial 90% attendance
  factor and 2.5 km home-distance decay are provisional; morning arrivals and mostly midday
  returns distinguish school from work. Include actual assignment in purpose/timing inspection,
  loads and fares, with explicit public-building coverage and no inferred student OD claim.

- **Lower the everyday-trip exclusion to 150 m** (Anita, T-141). Replaces the initial 300 m
  threshold. New outings below 150 m represented cell-centre distance are excluded from every
  mode and total; exactly 150 m remains eligible. Other demand parameters stay unchanged.

- **NYC everyday outings and a 300 m exclusion** (Anita, T-066). Shopping/errands use retail/
  service jobs; social/leisure use food/recreation jobs and visits to residents. Source-neutral
  weights allow later international inputs. Trial rates and timing generate partial demand,
  never a forced commute-share target. Outward/return routes are priced separately and join
  shared fares, loads and crowding. New outings under 300 m cell-centre distance disappear from
  all modes and totals without replacement. Bubbles now count local trip legs, with purpose and
  directional timing inspection shipped alongside them; station commute queries remain labelled.

- **Separate same-city commuting from the longer-trip overlay; show metropolitan focus**
  (Anita, T-139). The relative longer-trip scale made 26 trips look larger than 583 commuters.
  Same-city longer-trip discs are now hidden at all zooms, with counts retained in the inspector
  and cross-city discs retained. Every far zone carries its exact market ID. Thin solid city
  boundaries replace dashes; a subtle exterior basemap shade and darker outside/lighter inside
  focus stroke distinguish the active pack. NYC's pack includes northern New Jersey.
- **Trip-purpose shares need their own distance/time context** (Anita, T-066).
  Work journeys are generally longer than everyday journeys. Compare counts and length profiles
  separately; a trip-count share is not a passenger-distance, travel-time or transit-mode share.

- **Ship destination purpose/timing inspection alongside non-work demand** (Anita, T-066).
  Each selected destination bubble explains its work, shopping/errands, social/leisure and later
  school/facility trip counts and shares, with a rough daily distribution for each purpose.
  Use the bubble's actual purpose/period results and clearly labelled mode and direction;
  counts must add up. Planned behavior, implementation details in notes/T-138.md.

- **Reopen non-work local trips, starting with NYC where useful** (Anita, T-066/T-138).
  Distinguish trip purposes and daily timing; do not force a reported commute share while the
  model covers only part of non-work travel. T-138 records sources and a proposed bounded first
  slice. Its numerical rates/weights are trial suggestions, not calibration or shipped behavior.

- **Add Northeast cities in coherent batches of at most five** (Anita, T-137).
  Baltimore, Providence, New Haven, Hartford and Springfield fill the corridor between existing
  metros, using disjoint conventional county extents and the established trusted data pipeline.
  Basic pack and gameplay checks replace independent calibration. City/place metadata now drives
  the regional builder through the shared registry; new IDs append without renumbering old places.

- **Traveller totals determine bubble area; omit the selected city's aggregate** (Anita, T-136).
  Remove additive radius offsets and minimum sizes that inflated small flows. Use the normal
  demand mix: blue at full train share, red at zero, blended in between. A Boston selection
  shows grouped totals only for other places, keeping Boston's local dots at their real positions.
- **Selected longer trips stay visible, with city totals when zoomed out** (Anita, T-135).
  Work selections show incoming origins, home selections outgoing destinations. Their smaller
  counts need their own visible scale rather than commuter-density sizing and a small-count
  cutoff. All positive detailed zones remain; regional zooms show one labeled bubble per market
  with its exact daily total. This changes display only, preserving trip assignments and shares.

## 2026-10-10

- **Add cities through the established sources with broad metro extents** (Anita, T-134).
  Boston and Washington, DC reuse the Census/LODES/ACS/water pipeline. Basic format, totals and
  game-loading checks suffice; a new extent study or independent calibration is unnecessary for
  familiar sources. Both get local commute packs and regional endpoints. Original market IDs
  stay stable, while flow headers and corridor tables scale to the pack's market count.
- **Demand controls stay on the map; picking follows drawing order** (Anita, T-131–133).
  None/Homes/Destinations is always visible at the map's upper left, with the existing commute
  endpoints and saved preferences. Exact metropolitan pack footprints show at regional zooms.
  Visible demand bubbles beat underlying trains/stations; same-layer hits follow the last drawn
  object, and selection/catchment discs retain their parent selection. Explicit selected editing
  handles stay usable. Local non-commute trips are deferred until their scope is bounded.
- **Content first; performance work waits for a reported problem** (Anita). Current performance
  is adequate. Choose new content and gameplay features rather than proactive optimization;
  keep the existing responsive architecture. Screenshots and diagnostic plots belong outside
  the project in `maps/artifacts/trainworld/`, ignored by Git; preserve existing evidence there
  and update producers/references instead of cluttering the source directory (T-130).
- **Regional packs parse into numeric rows, and immutable tables release spare capacity**
  (T-123). Generic JSON trees kept large temporary allocations alive during pair construction,
  inflating every worker's WASM memory high-water mark. Typed rows preserve numeric precision
  and pair order while skipping metadata; unused capacity is released once pairs and lookup
  indices are complete. Keep separate worker ownership and the existing trip model. On the
  full NYC/Philadelphia fixture this saves about 46 MB across the pool, and 236 MB on opening,
  with unchanged demand. notes/T-123.md.
- **World coordinates: a Lambert conformal frame for new games, the old frame kept for old saves**
  (T-125, closing T-053; Anita: grow past two cities, intercity track between regions, correct
  lengths, curves and speeds on a line of several hundred km, existing saves identical). The track
  model stays planar; what changes is the plane. New games use a spherical Lambert conformal conic
  tangent at New York with the old origin and metre units: conformal, so arc radii and curve speeds
  are right in any direction, and its scale is `1 + dlat^2/2` (0.24% at 4 degrees of latitude,
  under 0.4% across 35.8 to 45.8 N at any longitude), against the old frame's direction-dependent
  1.2% at Philadelphia and 2.7% at Boston. Straight edges from New York to Philadelphia, Boston,
  Washington and Pittsburgh come out within 0.02% of the great circle (they were off by 0.2 to
  0.8%). A network records its frame: TWT5 is the world frame, TWT4 and older the legacy one, and
  a legacy-frame network still encodes as TWT4 byte for byte. Existing saves are NOT converted:
  they load in the legacy frame and stay there, because moving their track would change lengths,
  quotes and schedules and she asked for identical geometry, money, fares and ledger (the real
  NYC network's derived geometry, costs and service hash identically before and after, and its
  demand results are equal to the last digit). An explicit, tested conversion exists but is not
  offered in the UI (her call, T-129). Rejected: per-region local frames with a region id per node
  (an edge between regions needs a frame of its own, crossing tests need a common one, the format
  grows a field, and the old saves' Philadelphia track still needs a frame that is 1.2% off);
  lon/lat plus a local metric (rewrites the planar geometry); Web Mercator as the model frame
  (conformal but scale 1.32 here, which would scale every limit and cost). Water masks go
  through lon/lat from each pack's own frame (tabulated on a 2 km lattice, under 0.1 m, because
  the exact version made an edit 3.7x slower); all demand conversions go through lon/lat so packs
  keep their own origins; one registry file (`game/registry.json`) names the world, frames, packs
  and cities for the workers, the build, the save header and the UI. The regional market keeps its
  sites in its pack frame, so its own zone distances still carry the old frame's 1.2% to 2.7%
  (T-128). notes/T-125.md.
- **Longer-trip bubbles are drawn like commuter bubbles, one per far zone** (T-120; Anita: "Elsewhere
  in the New York region" is confusing, starts within the region should be visualised normally, and
  the yellow bubble style is out: size by people, red or blue by train usage, everyone not on a
  train red). Replaces T-118's one pale-plus-solid amber disc per place at a fixed 44 px. Now a disc
  per 6 km zone at the zone's own place, the commuter far end's sizing (biggest 0.6 of a zone, scales
  with the map and the size slider), opacity and mode mix, blue for rail and red for the rest.
  Per-zone sums come from Rust so the workers' answers still add up; the station's per-market rows
  became per-zone rows (f32) and the market sums are derived from them. The map's name tags are gone;
  the inspector keeps its by-place lists, and the own region's row is "Other New York region places".
  notes/T-120.md.
- **The demand solve is made faster without changing its numbers** (T-119, closing T-061). It was
  4.5 s to the free-flow estimate and 17 s to round three on the real network, and the bubbles
  waited for round three because a worker could only answer about a round while it had no later
  one under way. Now: strategy searches only towards stations a trip can end at, shared between
  the populations of a worker; each worker sends the bubbles' sums with every round and a worker
  with no periods answers the rest; the pool is up to 5 workers (one per period) plus that one
  (was at most 3). Rejected: splitting mode choice by origin zone over the pool (it only fixes
  the 5-periods-on-3-workers imbalance that five workers remove, at a reduce step per round);
  wasm SIMD on the pair loop (no gain in Chrome). Left alone, flagged in TASKS (T-121): the
  regional market's free-flow estimate routes without the distance fare, 17% more rail than a
  consistent run. Crowding rounds in which no segment is over its seats are skipped, exactly.
  The cost is about 240 MB more WASM memory (5 + 1 workers) on a 16-thread machine.
  notes/T-119.md.
- **One Northeast network with shared crowding** (T-116). Keep the existing NYC metre frame
  and legacy saves; transform city queries and append transformed water masks. Both city
  commute packs and the regional market share one bounded worker pool. Every crowding stage
  reads one frozen combined flow vector, and publications require matching versions/rounds.
  Navigation changes the local commute share and queries, while network loads/income remain
  global. The fixed plane's roughly 1.2% Philadelphia east/west scale error is acceptable for
  this first region; broader world coordinates remain T-053.
- **Longer trips are shown by place on selections and stations** (T-118, 2026-10-10; Anita: a
  Philadelphia bubble selection panned to New York lost its selection and could not show intercity
  demand). One bubble per place (the market's three: New York, Philadelphia, Trenton and Mercer),
  not per zone, because the market is coarse and a player reads places. Selected local cells map
  to the market through the 600 m squares both are built from, with the selected share of a
  square; no new numbers are made, and the numbers are the market's own pair totals. Station
  riders come from the assignment itself (per-station counts by far market), so they equal its
  entries and exits. Rail per pair is kept per period and averaged over crowding rounds like the
  loads. Bubbles use area for trips, a pale all-modes disc and a solid rail disc, in a colour no
  mode uses. The "Longer trips" check hides bubbles only; the numbers stay in the inspectors.
  notes/T-118.md.
- **Intercity begins as a tunable non-commute market** (T-117). Actual population/job locations
  in NYC, Philadelphia and Mercer form coarse zone-pair trips, including long trips within
  metros, with a smooth geographic onset. The common rail graph gives feeder lines actual
  passengers and applies fares/service/shared crowding. Keep attribution separate from commute
  denominators. Judgement trip rates are explicit; measured OD calibration, park-and-ride,
  air/coach and cross-pack commuting remain future work rather than implied implemented features.
- **Conventional centered confirmations** (Anita, T-113). Replace bottom-left questions
  with a modal Are you sure? dialog, cancel focused first, trapped keyboard focus and blocked
  background input. Keep the existing operation-specific warning and action labels.
- **City pane is for navigation** (Anita, T-114). Remove the diagnostic period rail-share
  chart and People/Jobs totals; the top bar already supplies the goal share. Begin with
  New York navigation and extend it with shared-world cities, rather than inventing metrics.
- **Philadelphia is the first Northeast expansion** (Anita, T-115/T-116/T-117). Build the
  standard 11-county metro pack separately from New York, with no duplicate demand counties.
  Preserve New York's track/save frame in the future shared world; transform local demand
  into each city's pack frame and combine shared-line crowding before feedback. City packs
  alone do not generate intercity trips; that separate layer remains T-117. notes/T-116.md.

- **Line management starts in the inspector** (Anita, T-109 to T-112). New line creates an
  empty zero-train draft immediately, enters stop editing and selects its default name for typing.
  Persist creation-order names, custom multi-character badges and independent full-property
  copies. Put list actions above long lists; sort lines by natural name or daily passengers.
  Restrict red stop-removal controls to Edit stops and confirm line deletion. Infill additions
  test all ordered insertion positions, rather than failing after trying only the end.
  Optional save metadata keeps old saves readable.

- **Short turns share the station they reverse in** (T-052). Count their full turnaround on
  the arrival platform and their crossover against crossing through moves or opposite reversals.
  Use per-move conflicts with the own-move queue subtracted, so independent through directions
  stay independent and tiny reversing service introduces no abrupt extra queue. Single track
  retains one shared platform; adding a passing loop would require explicit geometry and cost.
  Only constructed track contributes active station capacity. notes/T-052.md.

- **Blueprint planning leaves running endpoint geometry alone** (T-108). Constructed lines
  use constructed edge ends to identify dead ends; planned lines still preview planned ends.
  Drawing an unused extension must not shift a running train's stop or change its timing.
  notes/T-108.md.

- **Draw the overlay once per presented map frame** (T-075). Adopt the existing queued-frame
  guard after six alternating stress checks reproduce half as many draws and about half the
  drawing work, with unchanged map frames and zero camera lag. This measures overlay work, not
  a claimed halving of total browser CPU. `perfOff=noDouble` retains the old path for comparison.

- **Edit blueprint corners from their inspector** (T-071). Add inserts midway between adjacent
  control points, then the player drags its square; each corner has level and removal controls.
  Reuse checked shape edits and undo rather than adding a map gesture that conflicts with
  selection or drawing. Removing the last corner yields a straight stretch. notes/T-071.md.

- **M1 spec stays current and readable** (T-010). Correct stale schedule, save, demand and
  backlog claims; keep earlier calibration figures explicitly historical. Move oversized demand,
  track/service and interface sections intact into `docs/`, with linked summaries in SPEC and
  their existing section numbers preserved. No gameplay design changes. notes/T-010.md.

- **Independent browser saves with one writer per slot** (T-068). Named slots retain the
  original autosave and copy the current game without replacing it. Hold a browser lock in the
  worker for each writable slot; blocked tabs can play/export and rescue into a new slot.
  Flush before switching, read after acquiring the new lock, and keep save failures visible.
  Different slots can save concurrently. notes/T-068.md.

- **Conservative circle growth at station complexes** (T-094). Platforms within 100 m keep
  individual dots and selection but cap rider growth at 1.6x radius, versus 2.6x for isolated
  stations. This reduces clutter without moving station geometry or hiding distinct platforms.
  A cached spatial grid keeps nearby-platform detection out of the animation loop.
- **Common-line waiting is destination-specific** (T-069). Reverse strategy searches combine
  useful departures and their expected onward cost; boarding splits by frequency. Pooling every
  line at a station would incorrectly shorten waits for the wrong branch. Keep per-line loads
  for crowding, income and displays; fares remain separate from crowding. notes/T-069.md.

- **Top bar follows the map column's width** (T-106). Use container queries, since resizing
  the dock can narrow the map without changing the browser window. Hide secondary labels before
  moving to two or three rows; preserve cash, time, train share and all playback buttons.
  Full mode shares and day/demand remain in hover titles. notes/T-106.md.

- **Earlier car detail with continuous train width** (T-107, Anita). Show separate cars from
  zoom 15 instead of 15.5. Both representations use the same width, including physical scaling
  at very close zooms; remove the detailed view's smaller minimum that made cars suddenly tiny.
  Preserve actual car lengths and route positions. notes/T-107.md.

- **Whole-line trains are the schedule input** (T-104, Anita). Choose an integer allocation
  across both directions for each demand level; derive headway from the capacity-adjusted round
  trip. Keep allocations through route and congestion changes, with zero stopping new departures.
  New lines start 8/4/2. Existing TWT2/3 rates stay intact until the player edits a schedule;
  TWT4 saves the integer inputs. Use continuous capacity delays on resources serving count
  schedules to avoid rounding-induced oscillation. No frame-time scheduling work. notes/T-104.md.
- **Four track display choices** (T-105, Anita). Combine line colour and traffic width into
  Line and Line with traffic thickness (default), alongside Height and Max speed. Speed colours
  show the geometry's per-section limit, not current train motion; curves and straights can
  differ on one edge. Height/speed use uniform-width bare track while trains keep line colours.
  Migrate the previous width checkbox preference and remember the chosen mode. notes/T-105.md.

- **Close-up trains bend car by car; quieter crowding colours** (T-103, Anita). Switch at zoom
  15.5 from a compact capsule to the line's actual 20 m cars. Each has its own route position,
  chord heading and line-bundle offset; keep one GPU instanced draw and the existing analytic
  trip timing. Clamp the full consist onto its route at terminals. Picking follows every car.
  Reduce the empty tint from 60% white to 18%, and darken the filled core from 62% to 48% of
  line RGB. The crowding gauge spans the whole consist rather than repeating in every car.
  notes/T-103.md.

- **Player fares affect riders** (T-067, Anita). Use the saved base-plus-km curve in demand,
  at a US value of time of $20/hour (3 perceived minutes per dollar, chosen for play). Charge
  the base once even with transfers; charge distance on actual track length, so route searches
  see it too. Keep fare outside the ride-time crowding multiplier. Fare-only edits start a new
  debounced solve; identical prices do not. notes/T-067.md.

- **See busiest-hour train crowding as well as averages** (T-087). Default train gauges to the
  busiest hour, matching the crowding model and the line's fullest-train statistic. Settings
  can show the period average; the inspector keeps both counts visible and uses the chosen
  basis for its load bar. This changes display only, not schedules or demand. notes/T-087.md.

- **Delay tags leave room and disappear at city scale** (T-102, Anita). Share station names'
  zoom-12 cutoff and spacing rules; show larger delays first when tags compete, then higher
  utilisation. Keep resource positions, tooltip explanations and junction clicks. notes/T-102.md.

- **Fewer station names, one UI font, no commuter corner summary** (T-101, Anita). Hide names
  below zoom 12 and reserve more space around them, particularly before zoom 14. Keep transfer
  priority and T-074's shared-layer pan. MapLibre's Helvetica container font was overriding the
  project's font on station labels and other map DOM text; explicitly use Zen Maru Gothic there
  and on form controls. Remove the commuter legend and its unused state: the top bar already
  shows mode shares, and hover/inspectors retain local numbers. notes/T-101.md.
- **All construction controls live in Build; Select has no button** (T-100, Anita). The blueprint
  total, undo/redo and Construct stay visible at the bottom of the Build pane, while settings and
  the grouped cost breakdown scroll. Esc and clicking the active tool still return to selection.
  Costs come from the simulation's pricing intervals, with stations, junctions, flyovers and
  crossings separate. Include the actual new fleet charge: starting lines together changes
  delays and train needs, so quote a reversible preview of Construct all, not individual planned
  train estimates. notes/T-100.md.
- **Station names pan as one compositor layer** (T-074). The outer box clips to the viewport;
  the inner box moves from the same camera matrix as the network. Keep a 256 px margin of
  names ready and re-place at 128 px, after movement ends, or when the projection changes.
  Retains DOM text and overlap priorities while reducing browser CPU during the 300-station
  drag benchmark by 34%. `?perfOff=labelPan` restores individual placement for comparison.
  notes/T-074.md.

## 2026-10-09

- **Capacity markers go under the demand views; their tooltip is a styled tag** (T-099, Anita: the
  markers showed over the bubbles and seemed to have no tooltip). The demand canvas moved to z 2 and
  the markers to z 1, so markers sit over the network and under bubbles and a station's catchment
  always (with no demand view there is nothing above them). The old native `title` only appeared
  after a second's still hover; the new tag appears at once, like the bubbles' count tag, in plain
  words: what is at capacity and how long each train waits at the demand level in force.
- **A network with no running line has an answer: nobody by rail** (T-099). New game after the
  real New York save kept the old bubbles because `cell_modes` (and `flows`) answered empty when
  there were no subzones, which the views read as "the workers moved on". Fixed at the source in
  `DemandApi` (no subzones means zero rail), not by special-casing an empty network in the views.
- **Bubble size slider: 0.5 to 2 times the width, default 1** (T-099, Anita). It scales the drawn
  radius on the GPU and the hit test, not the built bubbles, so dragging it costs nothing but a
  redraw. Rejected scaling `BUBBLE_DENSITY` (a rebuild in the worker per slider step).
- **The grey-to-blue colouring switch is gone** (T-099): Anita kept the RGB mix.

- **Anita after playing batch 7**: the RGB mode mix stays as the bubble colour (not grey to blue by
  train share); "Track" is fine as the tool name; trains hiding busy stations from clicks is fine,
  players can zoom in, so T-086 is dropped. Asked for: capacity markers under the demand views
  and a tooltip on them, a fix for stale bubble colours after New game, a bubble size slider
  (T-099).
- **Commuter bubbles replace the commuter dot map** (T-097, views agent; Anita after playing T-084:
  too many dots, and clusters in circles; she wants Subway Builder's view). Bubbles are the city's
  H3 res-9 cells summed into their parents by zoom (res 9 from zoom 13.6, 8 from 11.8, 7 from 10,
  then 6): the cells already carry H3 ids and a parent is the id with its lower digits set, so no
  H3 library and no new data; each step is 7 times the area, about 1.4 zoom levels, which keeps
  the spacing on screen steady. Rejected a square grid (would not line up with the cells) and
  keeping the dots at a coarser value (still dots, still the circles: those came from spreading a
  dense cell's dots over a disc wider than its share).
- **Bubble area is commuters at one fixed density per end, no minimum** (T-097): 25,000 a km² for
  homes, 80,000 for jobs. The same density at every level keeps a place's total area the same as
  the view aggregates. Jobs get their own because at the homes' scale Midtown's bubbles covered
  half of Manhattan. Rejected sizing per view (sizes would change as the player pans) and
  a minimum size (Anita: nobody, no bubble).
- **Bubble colour is the RGB mix of red car, blue train, green walk, scaled to full brightness**
  (T-097, Anita's latest words). Her first words were "how many take the train is the colour (if
  none, gray)": that is one switch away (`BUBBLE_COLOURING = "train"`, grey to blue by train
  share); asked her which. The top bar's split uses the same three colours.
- **Bubbles, a selection's far end and a station's catchment are drawn over the network** (T-097,
  Anita: "in demand seeing mode we want to see this stuff on top"), on the demand canvas moved
  above the network's; station names stay above. Bubbles at 80% so the lines still show through.
  Station catchment: the station's own cells opaque on top, the far end filled at 55% below them
  (her words), both without a minimum size.
- **Clicks: trains and stations first, then a bubble, then the rest of the network** (T-097). The
  bubbles cover much of the map, and a station must stay clickable for its own view. Rejected
  bubbles first (stations unreachable while the view is on).
- **Several bubbles by holding still 0.4 s and dragging a box** (T-097), Shift and click to add one.
  A box is plain to draw and to read; a lasso was the alternative, more work for little gain at
  bubble scale. A plain drag still pans.
- **A selection hides the other bubbles; a station selection hides all of them** (T-097): the
  far end's bubbles share the colours, and with the city's bubbles under them the picture was
  unreadable. A click on any place picks it anew, Escape or an empty click goes back.
- **The switch is "Commuters on the map: Off, Homes, Jobs" at the top of the settings pane**
  (T-097, Anita: it is a viewing setting, and she disliked "demand by home / by job"). Rejected
  "Where commuters live / work" as button labels (too long for three buttons in the pane); those
  words are the legend's title instead.
- **A selection's far end is summed into bubbles in the bubble worker** (T-097): a jobs selection's
  far end is ~78,000 home cells, 15 ms on the main thread; the worker does it in 8-10 ms off it.

- **The bubbles' "where do they go" comes from the gravity plus the kept station pairs, not a
  second mode choice** (T-098, `flows`). Everyone from the gravity's zone trips; rail is carried
  along the solve's station pairs (as `stationRiders` does); walk and drive are the rest. Totals
  are exact and match `cellModes`; far rail per place is a little blurred. Rejected: keeping rail
  per subzone pair from the solve (millions of pairs a period) or running the mode choice again
  for the selection (it needs each round's station table, and would not match the averaged rounds
  the rest of the views show). The answer leaves out the far cells holding the last 1% of the
  commuters and gives them as cut totals: a job selection's homes otherwise run to every cell in
  the region.
- **Hotkeys 1, 2, 3 and Esc; a delete tool** (T-096, gameplay agent; covers T-088). Anita: "lets
  also have hotkeys esc for exit construction mode (and also exit other modes, if there are any),
  1 for start drawing track, 2 for placing station, 3 for delete (for a blueprint, deletes it. for
  a constructed track, ask for confirmation to delete it and then delete it". Digits rather than
  T-088's letters, as she asked.
  - **Esc undoes one thing per press**: a pending question (Keep), a drag, a half-drawn route,
    then the tool, then the selection. With a route half drawn, the first Esc drops the route and
    keeps the tool on, as before, so a slip does not cost the tool and the level set for it; a
    second Esc leaves.
  - **Each tool button shows its key as a small muted digit** (and in its title); four buttons
    with digits did not fit the dock, so "Draw track" reads "Track" (its title keeps the full name).
  - **A number key opens the Build tab** (from any tab): the level and platform settings the tool
    uses, and which tool is on, are shown there; the inspector does not move with tabs (SPEC 8).
    The key of the tool already on does nothing, unlike clicking its button (which goes back to
    Select), so pressing 1 again mid-route cannot drop the route; Esc is the way out.
  - **The delete tool removes track, stations and flyovers**: a junction without a flyover is not
    a thing of its own, so a click there takes the nearest stretch of track; lines stay in the line
    panel. Constructed things ask in the hint area above the bottom bar (Remove / Keep, Esc =
    Keep), not in a dialog: SPEC 8 has no floating windows. The inspectors' Remove track and
    Remove station ask the same way for constructed things (they only had a tooltip).
- **Station reach is soft: the walk to a station weighs 2.0 a minute for 10 minutes and 8.0 after,
  with no edge the player sees** (T-090; Anita: "passengers should consider how much time it would
  actually take to walk to station ... and how much time it would take to drive"). Replaces the
  2.5 km hard edge of T-078. A station now loses riders where the whole trip gets worse than
  driving. The steep second part keeps a station's draw to about 3-3.5 km: with 4.0 after 10
  minutes a lone Manhattan line drew riders from 6 km and more (driving into Manhattan is slow and
  parking dear), and any bound cheap enough to compute cut that tail visibly. Rejected: a convex
  curve (harder to say in a sentence), a longer easy part (12 minutes: more pairs, rail 0.6 points
  higher, no better fit).
- **Two computational bounds instead of a rule**: stations within 4 km straight line, and a pair of
  places offered rail only if its two walks cost at most 80 perceived minutes together (T-090).
  Set where little rail is left (0.1-0.3% of the real network's rail in their last band; a lone
  Manhattan line loses 4% of its riders to them) and so the recompute stays near what the 2.5 km
  edge cost: subzone pairs 13.8M against 13.6M on the real network, a day 15-25% longer. With the
  4 km reach and no pair bound it was 2.3x the pairs.
- **Driving speed by trip length and density** (T-090): 1.3x the straight line; the first and last
  2 km on local streets (40 km/h in open country), the rest on main roads (80 km/h), both falling
  towards 15 km/h where dense (half way at 5,000 commuters living and working per km²). Replaces 28
  km/h everywhere, which made suburban driving 2-3x too slow and suburban rail 1.5-2x ACS. Now Long
  Island, Westchester and Connecticut are at ACS; the four big boroughs keep slow driving.
  Rejected: speed by trip length alone (city driving too quick, the boroughs' rail fell to 29%
  against ACS's 42%); density only at the ends with fast main roads everywhere (32%). No road
  network (T-072).
- **A subzone's walk to a station is the logsum mean of its cells' perceived walks** at the mode
  choice's scale (T-090), not their mean distance or mean cost: with a convex walk cost the plain
  mean put rail 3.3% under one subzone per cell; this is within 0.2%.
- **Line width shows riders an hour in the period in force, both directions together, by the
  square root** (T-084, views agent): 0.45 of a line width when empty to 2.2 at 30,000 an hour.
  The period in force ties the width to the trains moving on screen and to the train fill; per
  hour makes the five periods comparable. Rejected: the whole day (does not move with the clock),
  the busier direction alone (a flow map shows everyone on the stretch), linear width (the few
  busiest stretches swamp the rest). Lines sharing track are laid side by side again at their own
  widths, so a bundle widens rather than overlapping.
- **Station circles show the day's boardings plus alightings** (T-084), up to 2.6 times at 150,000,
  by the square root. Rejected the period in force: the demand result keeps station riders by day
  only, and a station's size changing five times a day reads as noise; the day matches the station
  inspector and the Stations tab.
- **The train fill is a gauge: half full when the seats are taken, full at crush, a red rim when
  over full** (T-084). Rejected: fill in proportion to crush (a train with every seat taken would
  look a quarter full), and colouring trains by load (a second colour scale on top of line colours).
  Same numbers as the train inspector (T-083).
- **The commuter dot map is switched from a button by the mode split in the top bar, its legend in
  the map's top left corner while it is on** (T-084). The dot map shows the goal number on the
  ground, so its switch sits beside it; the legend is docked like the bar at the bottom left, not a
  window. Rejected: the City tab (the legend would go when the player opens another tab with the
  dots still on) and the settings pane (hidden for the main demand view).
- **Train, walking and driving have fixed colours, and the top bar's mode split uses them**
  (T-084), in `game/palette.ts`: train #d81b60, walking #e8a33d, driving #958c80. The split was the
  theme's accent and two tints; keeping that would give the same three modes two colour sets, one in
  the bar and one on the map. Driving is a warm grey dark enough to read: a lighter grey vanished on
  the faint basemap and the map looked mostly train at 26% train.
- **Dots by the spatial carry along a Hilbert curve, coarser zooms as every 4th dot** (T-084;
  Anita's dot rule). 5 commuters a dot from zoom 11.8 in, then 20, 80, 320 every 1.4 zoom levels;
  each level is the same carry at a larger value, so minorities stay right at every zoom (rail at
  5% on Staten Island within 1%). Dots spread over each cell's disc by a generator seeded with the
  cell; a sunflower pattern per cell was tried and drew each cell as a rosette. Rejected: rounding
  per zone (drops a mode wherever it is under half a dot, Anita's minority bug), and quadrupling the
  value every zoom level (the city view went too sparse).
- **The dots have a canvas of their own, drawn only when the camera or their data changes**
  (T-084), with only the 4,096-dot runs in view drawn. Rejected drawing them in the network overlay,
  which redraws every frame while the game plays (0.3-3 ms of GPU a frame for nothing new), and an
  offscreen texture blitted each frame (more code for what a second canvas gets for free).
- **Dots are rebuilt for a solve's first estimate and again after its crowding rounds, not for each
  round** (T-084): the rounds between barely move the modes and each rebuild uploads 18 MB. The
  demand client's `cellModes` cache is now per solve and round (one line in the demand agent's file)
  so the second ask gets the refined answer.
- **A station's far-end places are named by the nearest station within 3 km, else by distance and
  direction** (T-084). Rejected neighbourhood names from the basemap: only loaded tiles have them,
  so a place's name would change as the player pans.
- **T-038 (heat map of rail share) is covered by the commuter dot map** (T-084): it shows the mode
  split by place at cell grain, which is what the heat map was for.

- **Clicks are points of intersection again** (T-092, gameplay agent; reverses T-079's "the track
  passes through the clicked points" below). Anita: "We switched to using a track placement method
  that runs the line through the points we click. I think this actually feels less natural and we
  should revert, sorry. Reverting will help in that it'll make our placement of a track not
  interfere much with the track we previously placed in previous clicks." Each click is a PI with
  an auto radius (T-023's drawing, its lead-in from existing track included), so a click changes
  nothing before the curve at the previous click; through the clicks, every click refitted the
  headings of its neighbours. `geom::through` is gone.
- **The previous click's curve may still tighten on a click** (T-092): the auto radius gives a PI
  next to an end the whole leg to it, so while drawing, the last curve can use all of the leg to
  the cursor, and halves to its share once the next point is clicked. Giving it only half would
  freeze it at the click, but the finished track would then differ from its preview or old saves
  would change shape; the pre-T-079 rule is kept. It happens only when the new point is less than
  half the previous leg away.
- **T-079 track keeps its clicks until it is reshaped** (T-092): edges drawn through the clicks
  (TWT3, tracks | 8) load with their fitted PIs, so they look as they did, and keep the clicks
  (`EdgeData::thru`, written back while the edge is unchanged); the reshaping squares are those
  clicks, and the first reshape or node drag makes them the edge's PIs with auto radius. Dropping
  the clicks on load was the alternative: those edges' fitted PIs have set radii that use their
  whole legs, so any drag would be refused. New saves stay TWT3; new edges never carry clicks.
- **The radius stepper is back in the track inspector** (T-092; T-079 removed it because radii
  followed from the clicks). With PIs a radius is a free choice again: the corner stays where it
  was clicked and the curve can be tighter or gentler.
- **A set radius a reshape leaves no room for goes back to auto** (T-092): a split (a station or a
  branch on blueprint track) sets the radii beside it so the shape does not change, and those
  refused every later drag of a neighbouring corner ("does not fit"). A radius the edit itself
  sets is still refused when it does not fit.
- **Dragging a blueprint node keeps a junction's or station's heading; a track end's follows its
  edge** (T-093, gameplay agent). Anita: "add a feature where we can drag around previously placed
  nodes that are just blueprints." Every edge end at a node must leave along its heading (SPEC
  6.2), so at a node with two or more ends the edges' first PIs (which hold the heading) move with
  it and the rest of the PIs stay; an edge with none to move gets a lead-in PI as drawing does
  (the next drag moves that one, so they do not pile up). Turning the heading to follow the drag
  was the alternative: it would bend every edge at a junction at once and has no natural rule for
  three or more ends. Movable only while every edge there is blueprint and its station is not
  constructed; a press on a constructed node pans the map as usual unless that node is selected,
  where the drag is caught and the hint says it cannot move.
- **No squares while a node is dragged** (T-093): the preview line and its curve labels show the
  new shape; a selected stretch's squares hide during the drag and come back at their new places.
  The moving node shows as a ring.
- **Running cost US$1.50 a car-km, down from 6** (T-091, Anita: "we can lower running costs a
  lot"). At 6, running costs beat fares even at 100x (the real New York save lost $0.65B a game
  day: fares $1.71B, running $2.36B), because both scale by the same factor. Lowering the price
  keeps one economy factor and one note in the Money tab; the alternative, a separate 30x factor
  for running costs, would leave a real-looking $6 on screen that the game does not charge. Now
  that save is about $1.1B a day ahead; a line run far beyond its riders still loses money.
- **Anita's answers on the batch 5 questions**: all the proposed demand views (T-084); driving is
  the alternative for the whole trip and getting to and from stations stays the fast walk only;
  station reach should be soft, riders comparing walk, ride and a simple drive time (T-090);
  drawing goes back to points of intersection (T-092, reversing T-079); blueprint nodes can be
  dragged (T-093). The T-085 click question (trains hide stations) was not answered; T-086 waits.

- **The fast walk: 15 km/h on a path 1.3x the straight line, reach 2.5 km, weight 2.0; New York's
  rail constant 0** (T-078, demand agent, batch 5), replacing the 1.5 km walk at 1.2 m/s (weight
  1.39), the drive/bus other leg and the constant 1.5. Chosen for game feel: a station's reach is
  a circle the player can see, a few stations wide in the centre, fading towards its edge. With the
  old constant every bike-like setting sent a third of New York by rail (the fast walk takes 10-20
  perceived minutes off each end); 0 puts the real network at 22% against ACS's 20%. Transfers
  between platforms stay a walk on foot. The suburbs stay 1.5-3x ACS (mostly the gravity, T-072),
  accepted: the game may be easier than real life. notes/T-078.md.
- **Demand views are asked for, not pushed** (T-078): per-cell modes (4.9 MB in New York) and a
  station's riders are computed by the workers on request for the solve on screen, typed arrays
  only; each solve's results carry only what the UI shows today. A query about a solve the workers
  have moved past answers null.

- **The track passes through the clicked points by keeping PIs underneath** (T-079, gameplay
  agent, batch 5): the clicks are stored beside the PIs (`EdgeData::thru`, save TWT3) and the PIs
  are fitted through them by biarcs with circle-through-neighbours headings, each arc a PI with its
  radius set. Storing the clicks as the geometry's input was the alternative; it would have meant
  a second fit in every geometry path (splits, saves, derive) and new rounding rules, while PIs
  keep every T-040 guarantee and every old save unchanged. Track from before T-079 shows its curve
  middles as squares and is refitted through them on the first drag.
- **No radius stepper on blueprint track any more** (T-079): with the track through the clicks a
  curve's radius follows from where the points are; the inspector lists the curves read only.
  Replaces T-064's per-PI radius stepper.
- **Junction waiting saved = the junction's whole waiting** (T-080): a flyover removes every
  conflict resource at the junction (SPEC 6.2), so the inspector shows the current waiting per train
  through it, at the demand level in force, and the price from a trial edit.
- **Undo, redo, the blueprint's cost and Construct stay on the map** (T-082): they are needed from
  every tab (planning a line, the inspector), so they are not in the Build tab with the tools.
- **Riders on board are an average train of the period** (T-083): period riders on the segment over
  the trains run in the period, as SPEC 2 says; the busiest hour is not exposed per period to the
  main thread.
- **The economy constant scales running costs in the track model and fares in money.ts** (T-081):
  both read `ECONOMY` (params.rs, sent in `money_params`); the fare and per car-km figures the
  player sees stay real.
- **Trains win a click over stations** (T-085, Anita): a train is drawn over its station.

- **Anita's answers after batch 4** (all Anita):
  - Realism is for rough calibration, not a goal; the game may be easier than real life, and
    country differences in transit use are not ours to capture. SPEC 1 priority 3 rewritten.
    Consequence: T-072 (road travel times, county factors) drops to P3; T-034 becomes a game-feel
    tuning of the fast walk (T-078).
  - One access mode: a fast walk at about bicycle reach, both ends; the drive/bus "other leg" is
    removed and buses are out of scope; driving may come back later. Replaces T-020's other leg.
    T-035 (other leg at the work end) is abandoned. Driving stays as the whole-trip alternative.
  - Faster-than-real trains are fine ("nyc train speeds in real life are an abomination").
  - Construct stays final; no free undo of building.
  - Economy at 100x: fares and running costs per day x100, like Subway Builder.
  - Build tools move into the Build tab; track passes through the clicked points; the
    flat/flying junction toggle was unclear (replaced by "Build a flyover" in the inspector);
    trains should show who is on board.
  - She wants to see demand soon and asked which visual schemes to consider; options proposed in
    chat, her pick pending.
  - Dock width and the tab/inspector divider are draggable and remembered (relaxes "nothing
    drags": still no floating windows). The panel's explanatory notes go; the drawing tooltip
    shows length, average cost multiplier and price instead. "Construct all" becomes "Construct
    blueprints". A train drawn over a station wins the click.

- **Idle frames are the default** (T-073): with no trips this hour the game draws no frames and
  the clock wakes once a game minute; an empty game went from 13% of a core to 0.6%.
- **The gravity's next fix tries road travel time before county factors** (T-072 note): OSM road
  times work in every country and also fix the too-slow suburban car times; ACS county flows exist
  only in the US, so K-factors are fitted to what remains.

- **CPU is measured on Chrome pinned to two cores, as medians of repeated windows, with A/B by
  URL flags** (T-045): `?perfOff=` turns a part off and `?perfTry=` turns on a candidate fix
  (`app/src/perfFlags.ts`), read once at load, no effect without the parameter. The machine is
  shared with other agents, so single runs swung 47-79% before. The fixes found are left for the
  owners of those files (T-073 to T-076). notes/T-045.md.
- **T-034 (access and rail constants) paused before changing anything; T-072 (county-to-county
  flows in the gravity) goes first** (demand realism agent). The real-network comparison showed the
  crow-fly gravity sending New Jersey residents to Manhattan at 1.9-4.3x the ACS rate and keeping
  too few in their own county (39% against 52%; LODES, the decay's target, 43%). Access constants
  fitted on top would absorb a gravity error. notes/T-034.md.

- **T-063 done without per-tile messages** (gameplay agent, batch 4): measured on a 2,000 km
  network, the edit cost was station labels rebuilt and measured every edit and stroke arrays
  built on the main thread, not the size of the message. Fixing those and building the render
  buffers in the clock worker took the main thread from 160-193 ms to 3-6 ms an edit; sending
  only what changed waits for 10,000 km networks (T-070).
- **The app runs three crowding rounds, not one; no crush cap** (T-022). After one round the
  segment that takes the overflow is at its most overloaded (the first round moves half of what
  the penalty pushes, all at once); by the third a player-sized network moves under 1%. An
  overloaded line should show: it is the player's signal to add trains. notes/T-022.md.

- **Two WASM modules, one per kind of worker** (T-051), replacing T-026's one module for both:
  with LTO each module keeps only the code its exports reach, so the demand module has no track
  model and the track module no demand kernel or `serde_json`, and nothing downloads twice
  (157 + 79 kB gzipped against 244 for one module, same source). The clock worker compiles a
  third less, each demand worker two thirds less. Demand solve times unchanged. Cargo feature
  `track-api` (default) gates `TrackApi`; profile `wasm` = release with one codegen unit.
- **Both modules stay at opt-level 3** (T-051): opt-level s would save 17 kB gzipped on the
  track module and z 41 kB, for edits and previews 1.2x and 1.8x slower; SPEC 1 ranks CPU and
  never freezing above download size. notes/T-051.md.
- **The capacity pass is incremental** (T-050), replacing "the capacity pass stays global": an
  edit rebuilds the resources around the edges, nodes and lines it touched (whole sections that
  reach them) and re-solves those plus the ones a rescheduled line uses; only lines owning a
  changed resource get their holds and profiles compared. Tested equal to the global pass on
  random edit sequences. 10,000 km: 0.2-1.2 ms instead of 10-13 ms per edit. notes/T-050.md.
- **Touching a station wakes the capacity pass even if no line stops there** (T-050): a line
  passing through needs its platform resource; before, it appeared at the next full pass.

- **The home end of commutes is LODES RAC (jobs by home block) x (1 - tract work-from-home
  share)** (T-054), replacing population x the same share: the home end of the jobs WAC counts at
  work, per block, so the Bronx no longer sends 23% more commuters than ACS counts. **The decay stays
  d^-1 exp(-d/29 km)**: the refit's optimum moved along T-006's ridge by less than its spread.
  notes/T-054.md.
- **Demand has walking transfers up to 300 m, after a ride only** (T-007; SPEC 6.1 promised
  them). A station node takes one route's track, so two lines crossing at different levels could
  not exchange riders at all; New York's real network depends on such changes. Walk edges start at
  route nodes, so a walk between two stations is never priced as a rail trip.
- **Snapshot stations within 100 m are one demand station** (T-007): two lines' platforms built
  side by side would otherwise each take one of a cell's 6 nearest-station slots and need a walk
  to change. Never two neighbours on one line. Results stay per route node, so the app is unchanged.
- **The real-network test is New York's GTFS network solved natively through `DemandApi`, compared
  per county (ACS B08301), per subway complex and link (MTA OD 2024, nycriders) and per operator**
  (T-007), rather than built by hand in the game: GTFS gives real run times and trains per hour
  for all 109 lines, and a change to any constant can be rerun in 10 s. A track-model save of the
  same network exists for looking at in the game. notes/T-007.md.

- **Side-by-side lines are offset in screen pixels in the stroke shader, ordered per edge by a
  network-wide "forward" direction, branch lines on their side** (T-062). Screen-space offsets keep
  the bundle the same width at every zoom, as metro maps do, and cost nothing on the CPU per
  frame; ordering against one agreed direction is what keeps two lines from swapping where an
  edge happens to be drawn the other way.
- **Trains are bought by the car, network-wide, by whatever edit makes the lines need more; short
  of money the edit is refused** (T-028). A per-line fleet would strand cars when a line is cut;
  one train kind means any spare car fits any line. Refusing matches Construct and keeps every
  running line fully served (no partial service to model). Cars are never sold, as track is never
  refunded, so undo and schedule changes can never be used to make money.
- **One fare curve for the network, base plus per km, transfers paying the base once** (T-028),
  instead of the shell's flat fare per line. SPEC 6.4 asks for a curve by distance; one curve is
  one control in the Money tab and one input for demand later. Demand ignores it for now.
- **Running cost per car-km (US$6), cars at US$2.5M, fares $1.50 + $0.10/km to start** (T-028):
  real-world magnitudes (New York R211 cars, NYCT cost per car-km), so the numbers on screen read
  true. Pace of income against construction is left to Anita (TODO.md).
- **Money settles by the game hour in the clock worker; fare income is computed on the main
  thread from demand's riders** (T-028). The worker owns the cash (construct checks it); riders
  live on the main thread. The worker gets the clock as (time, rate, wall time) on every change
  rather than a message per tick.
- **$6B start kept** (T-028 recheck): one real 20 km tunnelled line with its trains is about
  US$4.2B now, leaving room for a short second line.
- **Saves are built, compressed and written to IndexedDB in the clock worker** (T-029): the main
  thread never serialises. Format TWG1 = JSON header (clock, money, fares, ledger) + the track
  model's TWT2 bytes, gzip. Autosave after anything paid for, else once a wall minute when
  something changed, and on page hide; the last game loads on start (`?new=1` skips it).

- **Constructing, and removing or changing anything constructed, is final: it clears the undo
  history** (T-055). Undo would otherwise rebuild removed track or unbuild paid track for free.
  Blueprint edits, lines and schedules stay undoable. Making a constructed junction flying, or a
  constructed station's platform longer, is paid at once like a construct.
- **A line runs only when every edge and stop it uses is constructed** (T-055); before that it is
  "planned": its stop times show, it loads no capacity, has no trains and is not in demand's
  snapshot. A junction's cost is paid when its third edge is constructed, a flat crossing's when
  its second is.
- **The drawing preview runs in the clock worker as a trial edit** (apply, measure, roll back),
  not on the main thread (T-023). One code path gives the preview every check an edit gets
  (crossings, ports, platforms, water) and it costs 0.4 ms a round trip; the main thread would
  have needed its own copy of the geometry and the 1.1 MB water mask. Replaces "previews the one
  alignment being drawn itself" in SPEC 6.5.
- **New track leaving a node gets an automatic PI on the node's heading** (T-023), placed so the
  first arc starts at the node. The player cannot click a point exactly on the heading, and a
  branch whose first stretch runs straight on the main line overlaps it.
- **Station names from the nearest basemap street within 400 m**, shortened New York style, a
  name in use skipped; else a neighbourhood; else "Station N" (T-024). Renamable.
- **Station names beat basemap labels through an invisible symbol layer of the same names** above
  the basemap (T-032): MapLibre's own collision drops the place names underneath, with no per-frame
  work and no filter edits.
- **The clock worker sends the whole network state after every edit** (T-025): simple, 4.7 ms on a
  small network. Per-tile updates (SPEC 6.5) wait for big networks (T-063).
- **New lines: "Line N", the next unused palette colour, 12/8/4 trains an hour, 30 s dwell, 3 min
  turnaround** (T-024). **A new game starts on day 1 at 07:00** with $6B and an empty city.
- **The track save is format TWT2**: constructed flags on stations and edges (T-055). No saves
  existed yet.

- **Demand runs in a pool of up to three workers, the five periods split between them by index**
  (T-026, T-021): half the logical cores less one, capped at 3. Three land the mode split in two
  period-times (0.72-0.83 s in Chrome) and leave the machine to the browser and everything else;
  five would halve that for five busy cores per edit. Mode choice is not split by origin zone: at
  three workers it saves a sixth and needs a reduce step every round. notes/T-026.md.
- **One WASM module carries the track model and demand** (T-026): the app builds with feature
  `demand`. Both kinds of worker load the same file, so two packages would download the track
  model twice; demand adds 74 kB gzipped. The T-005 benchmark export moved behind `demand-bench`.
- **Each demand worker runs a warm-up solve on a small network after opening the city** (T-026).
  V8 optimises a wasm function only for later calls, and the pair loop is one long call, so the
  first real solve took 3x as long (1.6 s to the split instead of 0.74 s).
- **A day's commutes: a trip to work and a trip home per commuter, shared over the five periods
  70/12/6/4/8% and 2/18/58/16/6%; trips home reuse the morning's station pairs reversed** (T-026).
  Judgement until ACS departure times are fitted; reversing keeps one mode-choice pass per period
  and is right for park-and-ride. Replaces the spike's 45% / 40% shares of a 3-hour peak.
- **Of trips not taken by rail, a share walks by zone distance (84% within a zone, 32% next door,
  6% two zones away)** (T-026), giving 5.9% walking with no rail against ACS's ~6%. The bar needs
  a walking figure and the kernel had none.
- **Trains carry 44 seats and 160 at crush per 20 m car** (T-026), a New York subway car, times
  the line's cars. Replaces the hand-made network's per-kind capacities.
- **The game clock's demand level reads the shared period table** (T-026): `demandAt` in
  `game/clock.ts` had its own hours (high 16-19, low 23-6) that disagreed with the schedules'
  16-20 and 0-6.
- **Riders show blank until the first solve lands, and the bar shows the no-rail split** (T-026),
  rather than the demo's made-up numbers.

- **Basemap labels in Zen Maru Gothic Regular, with Noto Sans merged into the same glyph files**
  (T-056). MapLibre fetches one file per font stack and range, so a static host cannot fall back
  from one font to the next in the client; each file holds Zen Maru's glyphs plus OpenFreeMap's
  Noto Sans Regular for codepoints Zen Maru lacks, so Arabic, Devanagari and the rest still draw.
  Regular only: Zen Maru has no italic, and Positron's bold country and capital names stay quieter
  in Regular. No CJK ranges: MapLibre draws kana, kanji and hangul in the browser and never fetches
  them. Made with Stadia Maps' `build_pbf_glyphs` (Rust, installs with cargo on Windows and
  combines font stacks itself); MapLibre's font-maker is not published on npm. notes/T-056.md.
- **The basemap style is rewritten before MapLibre commits it** (`setStyle` with `transformStyle`
  in `map/basemap.ts`) rather than repainted on `style.load`, so fonts and colours never flash.

- **The game is called anitabuilder** wherever a player sees it, lowercase (Anita: "silly name but
  i think apt and definitely not ai-y"). trainworld stays the internal name (folder, crate,
  packages, serve.py, code), so nothing is renamed on disk.

- **Build as a blueprint, pay on Construct** (Anita). Drawing is free and undoable; clicking
  Construct pays and builds. Settles the "should undo refund in full" question from T-040: undo only
  exists for blueprint edits, so nothing is refunded by undo.
- **Removing constructed track refunds nothing** (Anita). Replaces T-008's half refund.
- **Station names and, if feasible, basemap place names in Zen Maru Gothic** (Anita), via our own
  glyph tiles for the basemap (T-056).

- **Themes are called pink, blue and green** (were cream, sky, matcha; Anita). A remembered old
  name is mapped to the new one.
- **Space bar toggles pause and resumes the speed that was running** (Anita, T-046).
- **Nothing selected leaves the inspector blank**, no "Nothing selected" text (Anita, T-048).
- **Trains: rim blended into the core, heading along the chord between the train's ends, quad
  1 px larger** (T-047). The "wiggle" was the hard rim-to-core edge stepping a pixel at a time
  (MSAA does not smooth shading) and, zoomed in, the heading turning in steps at each 50 m
  sample. Core flicker halved, turning jitter at zoom 14 p95 1.45° -> 0.51°, GPU cost unchanged.
  notes/T-047.md.
- **Flat crossings are derived from geometry, not stored as nodes** (T-040). A same-level
  intersection of two edges becomes a crossing record (edge, chainage, edge, chainage) that costs
  money and loads capacity. As nodes, every drag of a PI across a track would split and re-merge
  both edges and rewrite the line paths through them; derived, saves and paths never change.
  Replaces "nodes: end, junction, flat crossing, station" in SPEC 6.2.
- **Keyframes are shared phase tables** (T-040). One table of constant-acceleration phases
  (t, s, v, a) per line, direction and demand level; a trip is (profile, departure). Replaces
  T-002's one (offset, speed, time) keyframe per train, which cannot represent stops and
  acceleration; T-025 integrates it (format in notes/T-040.md). A block simulator can still emit
  per-train phases in the same shape.
- **Train performance: 1.0 m/s² to 60 km/h, 0.5 m/s² above, braking 0.9 m/s²** (T-040). Typical
  EMU starting acceleration with the fall-off of a power limit, as two constant rates so every
  phase stays closed form; 0.9 is normal service braking (1.1-1.3 is the usual maximum). 1 km stop
  to stop: 67.5 s.
- **A hold is a slow approach when shorter than the cost of stopping there, else a stop** (T-040),
  instead of a fixed 30 s threshold: the realised delay then always equals the computed one.
- **Edits are refused only for problems they add** (T-040), so a save that a newer water mask
  makes partly invalid can still be edited and fixed. **Ground level over water** means within
  half a level (4 m) of the ground, so ramp ends count. A station whose track was deleted stays
  (its lines are broken) rather than blocking the delete; a node where two ends leave the same way
  with nothing behind (a junction whose stem was removed) is allowed.
- **The capacity pass stays global** (T-040): 1.5 ms at 2,000 km, 10 ms at 10,000 km and 400 lines,
  skipped when no line, schedule, crossing or junction changed. Incremental resources are a
  follow-up if large networks need them.

- **Gravity decay f(d) = d^-1 x exp(-d / 29 km)** (T-006), replacing exp(-d / 9 km). Fitted to
  LODES OD 2023 for New York on zone-centre distances: no exponential fits (its best is the old
  9-10 km, 15% of trips in the wrong 2 km band, 100x too few trips at 100 km), the power x
  exponential form misplaces 2.9% and lifts the county-pair log correlation from 0.83 to 0.88.
  The pack's `decay` field gains `"pow_exp"` with `decay_pow` (still format 1; readers refuse a
  decay they do not know, and the Rust reader knows both). notes/T-006.md.
- **The gravity balances on commuters, not residents and jobs** (T-042, from Anita's rule in SPEC
  4.1). Home end: population x (1 - the tract's ACS 2019-2023 work-from-home share, B08301). Work
  end: each LODES sector's jobs x (1 - its industry's work-from-home share among the region's
  residents, B08126), a published split rather than a uniform cut, so Midtown loses more commutes
  than the Bronx. Shipped as `commute_home`/`commute_work` per cell; `pop` and `jobs` stay true
  counts. New York commutes 10.18M -> 8.61M a day. ACS from the table-based summary files, since
  the Census API now wants a key. notes/T-042.md.
- **Preact 11 with @preact/signals for the UI** (T-027), as SPEC 7 proposed; nothing argued
  against it. Signals let the clock re-render one text node a game minute; Preact plus signals
  add about 10 kB gzipped. No Vite plugin, TSX through tsconfig.
- **The main thread keeps a `WorldView` snapshot (the save's inputs plus worker results) and edits
  only through functions that will become clock-worker edit ops** (T-027). Fake data lives in
  `game/demo.ts` alone, so swapping in worker snapshots touches no UI code.
- **Settings sit behind a gear at the end of the tab row and show in the tab area** (T-027,
  T-037): a pop-over would float, and a sixth named tab would crowd 304 px of tabs. The
  inspector stays where it is.
- **Station names are DOM labels above the overlay, not a symbol layer** (T-027): under the
  overlay, track was drawn through the names. Costs collision with basemap place names (T-032).
- **Clock speeds 60x, 240x, 960x game seconds per wall second**, provisional (T-027): normal is
  a game minute a second as in the mock.
- **The network is drawn on an overlay canvas, the only mode; the custom-layer mode is gone**
  (T-015). The overlay's one-frame lag while panning came from our animation-frame tick running
  before MapLibre's frame callback; drawing the overlay from the empty camera layer's render
  callback, in MapLibre's frame with its matrix, removed it (0 frames off against every drag
  frame off by one step, by camera probe and by composited screencast frames). Overlay keeps its
  CPU saving: 17-21% of a core playing against 35% for the custom layer. Trains sit above place
  names, which Anita accepted. Replaces "our own WebGL2 custom layer" in SPEC 7. notes/T-015.md.
- **Home care jobs spread over where people live: accepted** (Anita, T-014).
- **People who work at their own home are left out of commute demand** (Anita): LODES counts
  remote workers at the office, so trips scale down by the ACS work-from-home share (T-042).
- **County-government and school-district head-office lumps: a low-priority task** (T-043; Anita).
- **Spatial grain matters most in city centres; suburbs may be coarser** (Anita). 202k cells and a
  12% slower recompute are fine for now; adaptive cells only if performance needs it (T-044).
- **Blocks are spread over their polygon by land area** (T-013): 50 m pixels aligned with the
  water mask, each weighted by its land share, each to the res-9 cell of its centre; blocks too
  small to catch a pixel centre keep their internal point. Replaces "a block's whole population
  and jobs at its centroid". New York goes from 74,920 to 202,337 cells and recompute costs 12%
  more native; Anita: fine, grain matters most in centres, adaptive grain only if ever needed
  (T-044). Polygons from the per-county TIGER2020PL block zips, not the state TABBLOCK20 zips
  (73 MB instead of 510 MB). notes/T-013.md.
- **LODES head-office lumps: three rules** (T-014; the home-care rule accepted by Anita).
  Northwell's 41,521 jobs at North Shore University Hospital: keep 10,000, spread the rest over
  the population of the six counties where it ran hospitals. JetBlue's 10,300 transport jobs at
  its Long Island City office: keep 1,300, the rest to JFK. Blocks with 1,000+ jobs, 80%+ health
  care and at most 35% earning over $3,333 a month are home care agencies: their health care jobs
  go over the county's population (300,188 jobs in 109 blocks). The earnings split between
  hospitals (60-100% high earners) and agencies (35% or less) is clean, which is what makes the
  last rule safe without naming each employer. County-government and school-district lumps are
  left for T-043. notes/T-014.md.

## 2026-10-08

- **+2 is 1.3x and +3 is 1.8x** (were 1.4 and 2.0), so tall viaducts are less outpriced by deep
  tunnels. Updated in SPEC 6.4 and `sim/src/track/params.rs`. (Anita)
- **The stops list shows time from the start of the line, next to each station**, not the time
  between stops. Corrects the reading in the entry below. (Anita)
- **Station access forgiveness stays open until New York is playable**; Anita finds a third driving
  or bussing to stations high but wants to judge by playing.
- **Water mask from TIGER for M1, 25 m runs per row, in pack format 1** (T-030). TIGER AREAWATER
  plus the sea outside the state polygons is one small zip per county; OSM's sea polygons are a
  906 MB global file with no regional fetch, so they wait for the world pack (T-041). Runs per
  row (u16 bounds, a u32 row index) give a binary-search lookup and 1.11 MB for New York; 25 m
  rather than 50 m (0.67 MB) because track is built along shores. Added as `per: "water"` arrays,
  which every reader already skips, so no format bump. notes/T-030.md.
- **Water narrower than ~50 m does not count** (T-030): a railway crosses creeks and ditches on a
  culvert or a short bridge at grade, and requiring a viaduct with two 200 m ramps for each would
  make suburban track absurd. A 2 x 2-pixel opening of the 25 m mask; never adds water.
- **Anita's answers on the T-008 design and the T-009 mock** (all Anita):
  - Double-track routes drawn as one stroke are the default; single track is a setting. Closes the
    T-008 open question.
  - Track cost = US$90M/km x a level multiplier: +3 2.0, +2 1.4, +1 0.8, 0 0.3, -1 1.0, -2 1.1,
    -3 1.2. Replaces T-008's per-level price table. Only the selected level's multiplier is shown.
  - No track at ground level over water; bridges and tunnels over water cost 2x. Replaces T-008's
    "5x at grade, 1.3-2x otherwise".
  - Scheduling rows are high, medium and low demand under a "Demand" header, always all three; the
    "same all day" shortcut is dropped.
  - The stops list shows running time from the previous stop, plus the round trip; replaces a lone
    round-trip figure.
  - Display settings: track colour by line or by height; show/hide trains, labels and the rest.
  - Inspector shows any selection and is empty when nothing is selected; its position must not
    move between tabs.
  - Themes cream, sky and matcha all kept, dark later; Zen Maru Gothic; no decorative heading
    squares.
  - Overlay canvas for trains is preferred (trains above place names are fine) if its lag while
    panning can be removed; otherwise layer mode (T-015).
  - Transit-share heat map later.
- **The city pack ships the solved zone gravity; pack format 1** (T-019). A zone id per cell, grid
  position and two balancing factors per zone, and a `gravity` header block. Opening New York in
  WASM went from 8 s (solve) to ~7 ms (parse and rebuild). Format 0 dropped; nothing else read it.
  notes/T-019.md.
- **The pipeline solves the gravity with the sim crate's own Rust code** (`pack_gravity` bin)
  rather than a numpy copy (T-019): the game rebuilds trips with the same decay table, so the two
  cannot drift, and T-006's refit will be made once. Cost: building a pack needs Rust installed.
- **Access: a logit over access options and egress stations, rail priced at the logsum**
  (T-020), scale 0.2 per perceived minute (nest ratio 0.25 against the mode logit's 0.05).
  Replaces all-or-nothing best station pair, which made station counts lumpy; the logit halves
  their error against the per-cell reference. notes/T-020.md.
- **The other access leg at the home end: 30 perceived minutes + 2.5 x time at 25 km/h x 1.3, to
  the 4 nearest stations within 20 km** (T-020). Picked so walking stays nearly alone within 1 km
  (20 minutes + 2.0 had 14% of people 500 m from a station driving to one) while long trips can
  park and ride; judgement until fitted. Work end walk only for now.
- **No walk bands in access subzones by default** (T-020): with the logit, zone x nearest station
  is as accurate as the old 5-minute bands were (station counts 5.5% off, segments 2.3%), and the
  other leg's extra subzones fit in the old recompute budget only this way (5.7M pairs against
  26.5M). Replaces 5-minute bands.
- **Train keyframe times are relative to an hourly render epoch** (T-016). The float32 time
  uniform stays under 3600 s; each game hour every train's (offset, time) is advanced to the new
  epoch in float64 from its master keyframe and re-uploaded, speed and line buffers untouched.
  With absolute time, positions were off by 1.5 m (median) at day 30 and 20 m at day 365;
  re-based they stay within 2 cm at any day. An hour keeps the uniform small and the re-base rare
  (~1 µs per train). Replaces "reference time uploaded once". notes/T-018.md.
- **Unfocused but visible: 10 fps with the clock running; hidden: no frames, clock frozen**
  (T-017). Stopping on blur would freeze a map on a second monitor; 10 fps costs 3-9% of a core
  against 17-36% at 60. notes/T-018.md.
- **MapLibre 6: our matrix is the custom layer's `defaultProjectionData.mainMatrix`** (mercator to
  clip, float64), and overlay mode gets it from an empty custom layer, since `map.transform` is no
  longer public (T-018). It is also the camera MapLibre actually drew that frame with.
- **Track geometry is straights and circular arcs, drawn as points of intersection** with an auto
  or player-set radius at each. Clothoids change only a second or two per curve and are invisible
  at map scale but cost Fresnel maths and inexact parallel tracks; splines vary the speed limit
  along every curve and make the minimum radius hard to enforce. Replaces "arc + straight or
  clothoid". notes/T-008.md.
- **Curve speed a_lat = 1.1 m/s², minimum radius 100 m, M1 top speed 160 km/h.** 1.1 matches LGV
  design (7,000 m at 320 km/h) and New York's 140 m curve at 25 mph; 100 m lets a Manhattan corner
  be turned at 38 km/h. The track itself has no speed cap.
- **Levels 8 m apart, max grade 4%, so 200 m of ramp per level**, ramps centred between PIs of
  different levels. 8 m clears one track over another with wires; 4% is the usual metro limit.
  Clearance needs a full level of height difference.
- **Topology: double-track routes drawn as one object, single track as an option, flying junction
  as a toggle**, assumed until Anita decides (open in SPEC 12). Every junction then has a known
  layout, so its delay can be explained in one sentence; individual tracks (NIMBY) mostly add ways
  to build tangles.
- **Lines store their exact path** (edges with a direction), not just stops, so new track never
  reroutes a line silently.
- **Capacity from occupancy times and a Kingman-style delay curve**, `W = 0.2 x mix x s x rho/(1-rho)`
  to 90%, straight above, plus an hour's fluid queue past 100%. Track sections `90 s + v/0.6`
  (33 trains/h at 40 km/h to 15 at 320, matching real metros, commuter lines and the Shinkansen);
  junction crossings and flat crossings 90 s; one line alone has no queueing delay. Delay is a hold
  at the resource's entry in the run-time profile, so trains visibly queue. Computed in the clock
  worker in one pass (frequencies are inputs, so no iteration), not with the demand as SPEC 2.1
  said before.
- **Costs: one worldwide price list per route-km**, at grade 20, viaduct 60-100, tunnel 90-170
  US$M, stations separate; water 5x at grade but 1.3-2x for viaducts and tunnels (a reading of
  "water ~5x" that Anita should confirm, open in SPEC 12). Calibrated to the Transit Costs
  Project's non-US averages. **Starting money $6B**: one real first line.
- **Edits are operations owned by the clock worker** and dirty only touched tiles, crossings,
  resources and lines, plus one step to lines sharing a resource; demand wakes only for service
  changes (stops, frequency, or a station-to-station time moving 5 s or more). Saves hold PIs, not
  geometry: ~170 KB compressed for 10,000 route-km.
- **Line colours are a normal saturated transit palette, player-settable; only the UI is styled.**
  Replaces "pastel line colours" from the earlier look decision. (Anita: "the only things we should
  be making stylistic decisions for are ui".)
- **The 10k-train target is a simulation target, not a rendering one.** Anita: most trains are
  local to a city and not drawn when zoomed out. Spec priority 2 now measures demand recompute per
  city instead of frame time.
- **Move to MapLibre 6** (Anita agreed); T-018, done early in M1.
- **Demand on two levels (from T-005)**: doubly constrained gravity between 2 km zones, by
  distance only, solved once per city; mode choice between access subzones (zone x nearest
  station x 5-minute walk band), ~11 per station. Replaces the spec's cell-to-cell pairs: 5.6-23
  billion pairs, distance truncation still leaves billions, and even served cells only are
  90-180M pairs growing with network coverage. Subzones cost 15-21M pairs, under a second per
  period in WASM, and stay within 3% of exact per-cell access on segment loads. Trip
  distribution does not react to the network (no logsum feedback), so an edit never reruns it.
  notes/T-005.md.
- **Train drawing layout (from T-002)**: each line resampled every 50 m into a float texture,
  one 16-byte keyframe per train (line, offset, speed, reference time) uploaded once, position
  computed in the vertex shader; coordinates relative to a city origin, with the matrix composed
  in float64 on the CPU. Our per-frame JS stays ~0.05 ms at any train count, so train count is
  a GPU question only. Custom layer vs overlay canvas is left open (T-015). notes/T-002.md.
- **New York's boundary is the 2023 MSA plus Fairfield CT, Dutchess and Orange NY** (25 counties,
  21.7M people). The added three each end a commuter rail line a player would expect; Mercer NJ
  and south stay out for Philadelphia. Details in notes/T-003.md.
- **Pack cells take a block's whole population and jobs at its centroid** for now; spreading big
  blocks over their area is T-013.
- **Accounts and shared saves on shonei-server**, as the plan for now (M4). (Anita)
- **Anita installs only rustup + the wasm32 target**; agents install the wasm build tools
  (wasm-pack or wasm-bindgen-cli) themselves in T-001 via cargo.
- **Project docs: CLAUDE.md + SPEC.md + TASKS.md + ARCHIVE.md + DECISIONS.md.** Anita expects the
  project to run long and does not want beads; she asked whether agents can keep docs current and
  archive finished work. Spec is rewritten in place, history lives here.
- **No signals for now; tangles discouraged by junction and segment capacity.** Keeps every train
  analytic, which is what makes 10k trains cheap. Capacity overload adds delay to the lines through
  it, computed in the background. Signals must stay possible later: track is a directed segment
  graph and the renderer/demand consume keyframes, not `position = f(t)`. (Anita)
- **Tiers 0 and 1 only; tier-2 OD becomes a check.** Subway Builder turned out to assign workplaces
  by gravity too, so tier 1 is already at its level. (Anita)
- **Long-distance demand between zones, not city centroids.** Two neighbouring cities must exchange
  far more short trips across their shared edge than long ones between far ends. The local and
  long-distance layers share one distance curve across the city boundary. (Anita)
- **Several long-distance stations per city are normal** (Tokyo, Kansai). The network is one network;
  each trip picks its best station pair. (Anita)
- **One train kind.** A later kind would differ by speed profile and cost (HSR, light rail), never by
  urban vs intercity. Scheduling is per period for every line, with a "same all day" shortcut.
  Replaces the first draft's urban/intercity kinds. (Anita)
- **Towns can become full cities mid-game** once their pack exists, mainly for development. (Anita)
- **Rust → WebAssembly for the simulation and demand core**, TypeScript + MapLibre for the app.
  (Anita agreed)
- **Start in the Northeast US, New York first.** Familiar to Anita; LODES, NHTS long-distance OD and
  nycriders give checks. (Anita)
- **Mode split bar as the goal number; one bubble per city in the world view.** Subway Builder's
  city-wide "80% drive, 16% train" readout is what Anita likes in it. (Anita)
- **Look: few colour tokens, light and soft; no ink-on-paper styling.** Anita found the paper idea too
  statementy; fixed dock, square buttons and pastel line colours stay.
- **Name: trainworld for now.** Anita dislikes it; a better one is welcome.
- **Flows, not agents; trains as functions of time.** Subway Builder's own changelog blames per-group
  passenger work for its lag and freezes; NIMBY Rails simulates individual riders and slows when
  queues grow. See `research/`.

- 2026-10-11 (T-143, Anita feedback): replace the all-city navigation list with the viewed city and its five largest external corridors by all-mode demand. Label the existing corridor totals as both directions. Preserve demand selection while its first result loads; use stable purpose colours in a share pie and departure plots. T-149 separately tracks startup gating and city scheduling.

- 2026-10-11 (T-150, Anita broader school-data request): preserve NYC DOE building totals inside the five boroughs and use official NCES EDGE/CCD 2023-24 public-school locations/enrollment elsewhere in the NYC metro. Spatially exclude the DOE area to avoid duplicates; expand estimated pupil homes across all metro counties. Keep university inputs separate from child-origin/school-day assumptions. National source reuse is implemented; wider city rollout and university attendance remain follow-ups.

- 2026-10-11 (T-153, Anita short-trip feedback): reserve the distance-based walking share before rail and road compete. Attractive rail may win the non-walking remainder, but it must not erase trips that the simple model already considers walkable. Keep the existing 2 km commute-zone grain as an explicit precision limit; everyday purposes continue to use sampled cell distance.

- 2026-10-11 (T-154, Anita compact intercity feedback): halve both detailed and regional longer-trip bubble radii while preserving area proportionality. Use 5 px connection bars and show train/all trips in the city list; keep percentages and expanded counts in tooltips.

- 2026-10-11 (T-144, Anita station-rendering feedback): ordinary stations are track-aligned platform boxes. Rider magnitude is a separate, off-by-default analytical layer of large translucent circles behind the boxes; those circles do not change picking.

- 2026-10-11 (T-145, Anita track-placement feedback): draw the live track placement ghost as a solid, dark-cased stroke. Its body follows Height or Max speed display colours and is white in Line/Traffic modes; invalid geometry stays red. The placed blueprint retains its normal dashed, faded construction state. Preview speed colours come from the track model's fitted geometry rather than main-thread inference.

- 2026-10-12 (T-152): model university attendance as a separate adult-origin purpose. For the first US layer, use one IPEDS 2023 budget per active degree-granting reporting UNITID, count degree-seeking undergraduates and graduates, subtract exclusively online students, and never replicate a system total across child campuses. Treat attendance, adult homes, decay and timing as explicit provisional assumptions.

- 2026-10-12 (T-155): label demand-derived riders, boardings and loads as steady-state forecasts. Animated train movement selects the current segment but does not observe passengers. Any adoption ramp must affect assignment, mode share and revenue together rather than scale only the display.

- 2026-10-12 (Anita): avoid city-specific data. Use sources that cover every city or every country; a one-city source only when it is very easy to get and use (NYC DOE building enrollment probably was not worth it), and never one for a minor city. Recorded in CLAUDE.md "Working here".

- 2026-10-12 (T-157, Anita track tooltip request): resting the pointer on track for 0.4 s shows what the track display draws there, in the demand views' tag. The traffic view states the hovered line's daily forecast on that stop-to-stop stretch with both directions summed, matching how its width is built, rather than a per-direction pair or the width's current-period hourly rate (Anita asked for daily traffic). Counts round to the nearest 100 above 1,000, since they are forecasts. Height states the level the colour shows, by the Build tab's level names. Select tool only: drawing, station and delete tools have their own tags. Bubbles, capacity tags, editing handles, trains, stations and junctions above the track keep it hidden, following the pick order.

- 2026-10-12 (T-149, Anita's viewed-city-first request): the demand pool loads only the packs a game needs: the viewed city (`activeCity`, nearest centre to the camera; the camera starts on New York and saves hold none, so the initial view is unchanged), every city with a built station inside its footprint or within 4.5 km of it (the walk's 4 km bound plus a margin), and the regional market always (the City panel's corridors need it). Any other city has no rail access, so it adds nothing to loads, fares or crowding; a worker check found every line and New York figure bit-identical with one pack or nine. Blueprints do not count, since demand never sees them. Opened packs stay open.
- 2026-10-12 (T-149): readiness is the loaded game's network solved to the last crowding round for every city it needs, not the free-flow estimate. Constructed cities other than the viewed one are part of it, because shared crowding couples them; a solve waits for every needed pack rather than publish without one. At startup a loading screen takes all input and the clock is held until ready; a later file, slot or new game holds the clock only. A pan to an unloaded city (necessarily one without construction) loads it without blocking or holding the clock, since it changes no number elsewhere, then solves again to give it a view.
- 2026-10-12 (T-149): the loading percentage is work done, weighted 10 download / 30 opening / 60 solving (judged from measured startups), never decreasing and capped at 99 until the result is published; it stalls rather than invent progress. On failure at startup the screen says why and offers Try again or Play without demand.

- 2026-10-12 (Anita): intercity demand need not be closely calibrated; its design is early and undecided. Recorded in CLAUDE.md "Working here".

- 2026-10-12 (Anita): track tooltips show after 100 ms of rest, not 400 ms, so they feel responsive (`DWELL_MS` in map/trackHover.ts).

- 2026-10-12 (Anita, answers to the day's questions): no street-address geocoder for now, neighbourhoods and cities are enough (T-177 stays OSM places only); a station complex stays display-only in demand for now (T-161 not yet); the single US border factor's side effects are accepted (Center City to Camden 0.39x, Fairfield to Manhattan 0.25x, LIRR 1.16x); the playable-while-loading gate stays, with a centre-stage loading indicator until the clock starts (T-180); no new cities until the general design settles.

- 2026-10-12 (Anita): only she has saves, so save compatibility is not a constraint until she says otherwise; T-129 (offer converting old saves) abandoned. Opening ramp redesigned (T-156): no ramp in demand, riders use a new line at once; only the commuter bubbles' mode colours catch up over about one game day, as if each person takes one trip a day. This supersedes the T-155 rule that a ramp must move assignment, mode share and revenue together, because there is no ramp in the model now, only in what the bubbles show. Zoomed far out, every city condenses to one longer-trips bubble, the focused city included (T-186).

- 2026-10-12 (T-156): the bubbles' catch-up blends trip legs per cell, old times (1 - f) plus new times f, so a bubble's size moves with its colour and both agree with its hover tag; f is the share of a day's trips departed since the result landed, by the commute's period profile (36/15/32/10/7%, docs/demand.md 4.2), on the game clock. The commute profile stands for all purposes since it is the one the model states for every city; a flat 24 h ramp would move the bubbles as much at 3 am as at 8 am. Hover tags read what the bubbles show; the inspectors, a bubble selection, its far end and the top bar stay the current forecast, since they report the model and blending them would contradict line riders and fares. Results landing while the clock is held, the first result, and purpose or city changes show at once. Redrawn in the bubble worker every 0.01 of f, at most four times a second, and only while catching up. The city overview's bubbles catch up the same way; a selection's longer-trip bubbles do not, matching its inspector.
- 2026-10-12 (T-186): the city overview starts at zoom 8.5 and below, where the metropolitan outlines are fully drawn and neighbouring cities are on screen; at 9.5 (the selection's per-city threshold) New York still fills the map, too soon to lose its own bubbles. Sizes come from the market's corridors (both directions; the City pane's figures), so Homes and Destinations look the same there. The overview's largest bubble is 28 px before the size slider, larger than T-154's 19 px city circles, because it has no local bubbles to stay under. A selected city is drawn at 28 px with its destinations on the same scale, so a destination is never larger than the city it is part of. A new `city` selection with its own inspector; clicking another city's bubble selects that one. Trenton has no registered centre, so it has no bubble but counts in totals and the inspector. With longer-trip bubbles hidden or a local purpose filtered, there is nothing to condense to and the focused city's own bubbles stay. T-139's rule (a selection's longer-trip bubbles omit its own market) is unchanged; T-186 revises only what is drawn with nothing selected.

- 2026-10-12 (Anita): station complexes, including ones the player links by hand (T-197), are transfer points in demand (T-161), replacing "complexes stay display-only". Park demand is weighted by area and by population nearby, not by size alone (T-196).

- 2026-10-12 (Anita): new standing list docs/demand-worldwide.md of demand assumptions to recheck for cities around the world; every new or changed demand rule adds or updates an entry. The state-line border factor stays for the US (Anita unsure state lines matter much; it is on the list). The flatter distance curve (T-198) goes ahead for the US packs. Left-behind riders feeding demand (T-188): yes, later.

- 2026-10-12 (T-189, T-190, T-191, Anita's TODO): stations are dots below zoom 13 and their platforms from 13, where a 150-200 m platform is 10-14 px, about the dot's size, so the switch hardly changes what is seen; one zoom further out platforms would be shorter than the dot. A platform is drawn as the circular arc through its middle and both ends along its track, with flat ends, at its true length; at a dead end it runs from the node along the track, matching where `service.rs` stops trains. Its width is the larger of the true 12 m and the lines through it plus the rim: true width alone would hide the box under the line until about zoom 17. Selection never resizes a station (replacing T-144's 1.35 times and T-163's enlarged complex): a 2 px dark outline and a 6 px soft dark glow, the selected stations drawn after the others; above the demand bubbles they are shown by clearing the bubble canvas over the station and its outline, fading out over the glow, rather than drawing the station a second time on top, so a train standing in it and its crowds stay visible. A click or hover there goes to the station. Platform crowds draw only where platforms do (from 13, was 12). Train pills bend along the circle through their two ends and middle, one quad each (cheaper than segments, the same draw call); close-up cars stay straight. Lines sharing track are drawn over each other, the highest line id on top (T-190, Anita: "a hard problem"), so stations are no longer double width; T-062's slots and offsets are gone from the network buffers, demand layers and picking. Lines are 60% of their former width at every zoom, bare track, casing and the selected extra in step (T-191); trains keep their size so a pill still reads over a thinner line; picking keeps its 7 px reach, which stays wider than any drawn half line.

- 2026-10-12 (T-199, follow-up to T-189/T-191): trains are 75% of their former width, 3.75/6/9 px at zoom 9/12/15 (was 5/8/12), about 1.4 times a plain line from zoom 12 instead of twice; a pill keeps its former minimum length (2.2 times the old width) so it stays as easy to spot and its fill gauge, which runs along the length, keeps its resolution. The white rim stays as it was. Platform crowd bands are lighter: drawn from zoom 14 (platforms still from 13), 1 px plus 1 px a step of crowd, so 2-6 px (was 4-12), 2 px clear of the platform or a standing train (was 3), the train blue at 40% opacity (was 70%). The left-behind red is 90% opacity (was 85%) and one step thicker than the blue would be, so against the softer blue a platform leaving riders behind is the one that stands out. The band hover (T-194) reaches 10 px beside a platform (was 18) and only where bands are drawn (the renderer's `crowdsShown`, from zoom 14).
- 2026-10-12 (T-198, from T-185, Anita "i think so"): the work gravity's curve is d^-0.8 exp(-d/30 km) in every US pack (was d^-1 exp(-d/40 km), T-162), one value for all nine as T-185 recommended rather than a length per pack. It lengthens job-centre residents' commutes by 0.6-1.3 km and moves centre-to-suburb commuting from 0.73-0.87x LODES to 0.88-0.97x in all nine, and lowers trips misplaced in eight; New York is about neutral (16.2% to 16.7%), the one hint that a very centre-heavy city might keep a steeper curve (docs/demand-worldwide.md). The US border factor stays 0.23 (refitted 0.228 New York, 0.207 Philadelphia); Washington's own factor follows its refit at the new curve, 0.49 to 0.45. The curve lives in each pack, but the kernel's `Params` default changes with it, since `city_gravity` uses a pack's gravity only when the two agree (otherwise every pack would be re-solved at opening on the old curve). notes/T-198.md.
- 2026-10-12 (T-209, Anita: "a complex of like 10 stations renders as 2 station bubbles"): no rule changed. The circles already group exactly as `complexes.ts` does, links included (checked on all 855 complexes of the real New York save: one circle each). What looks like one complex drawn twice is two complexes sharing a name: New York's Times Sq-42 St (10 stations) and its 8 Av platforms (4, 290 m away), two Grand Central complexes (16 and 8 stations, 335 m apart), 14 St/6 Av, Bleecker St, Canal St. Merging same-named stations was rejected because a player's stations take nearby place names and two unrelated stations can share one; linking them by hand (T-197) is the way, and makes one circle and one label. The name labels now take the player's links too (they grouped by geometry alone).
- 2026-10-12 (T-210, Anita): every hover outside the map canvas is the game's own tag (`ui/tip.ts`, from `data-tip`), shown at once and following the pointer, never a browser `title` or SVG `<title>`; a stray `title` from markup the game does not write is turned into the tag under the pointer. notes/T-210.md.
- 2026-10-12 (T-211, Anita): search is a magnifying-glass button until opened (click, `/`, Ctrl+K or Ctrl+F); Escape, a pick or leaving the box empty closes it, while text left in it keeps it open. Ctrl+F is taken from the browser's find only when focus is in the game (not another text field), so typing in a slot name still gets the browser's find.
- 2026-10-12 (T-212, Anita): the loading card and the top bar say "Loading data… N%" rather than listing cities; the step line stays, and the top bar's hover names the cities.
- 2026-10-12 (T-213, Anita): "Station demand circles" and "One circle per station complex" move from Settings to the top of the Stations tab, above its list, with their note. The complex setting goes with the circles since it is the circles' grouping (and the inspector's Station / Complex switch, which stays where it is).
- 2026-10-12 (T-206, Anita: "we should only add or remove trains at the ends of lines"): an edit that changes a running line's departures takes effect at the game time it is made. Departures before it keep the old pattern (an era, kept in the clock worker's service only until no train of it can be drawn, never saved), so running trains carry on where they were; a train whose profile changed is drawn from where it was. At each terminal a train that has finished its run and turnaround takes the next departure within that departure's headway (an hour at most) and stands until then; a departure with no train waiting is a train entering service there; a surplus train leaves at the end of its turnaround; a train that had left service never comes back. The same hand-over runs at every change of period, and at any change of pattern a departure closer than half a headway after the old pattern's last is left out, so trains never leave in pairs. Chosen over per-train state stepped by the clock: trips stay a function of the service, its eras and the hour asked for, cheap and alike from any window. A reload re-spaces once (eras are not saved: saves are Anita's only). The boarding account's day starts six hours before a line's latest change while it has eras, so its trains are the ones drawn. notes/T-206.md.
- 2026-10-12 (T-207): trains do fill where a line is short of trains (a player's Manhattan trunk at 14 trains is packed on every peak train); on the real New York network 3% of morning-peak and 8% of evening-peak trains reach crush, because demand's crowding penalty moves riders away once seats run out and demand carries fewer subway riders than the real city. Not changed: loads stay demand's. Changed: a train packed to crush is drawn full with a red edge and its tag says "packed full", since a full train had looked exactly like one at 85% or one with no load data. notes/T-207.md.
- 2026-10-12 (T-204, Anita: station boxes "change proportions as we zoom in and out ... they end up looking stubby. lets have them only ever be drawn at a single scale", and turn into dots "around ~4x more zoomed in"): a platform box's width is fixed on the ground like its length, so both scale with the map and its shape is the same at every zoom. The width is what covers the widest line through it plus the rim at zoom 15 (12.6 px for a plain line, about 23 m at New York's latitude), not the true 12 m, which at zoom 15 is narrower than the line itself; since lines stop widening at zoom 15 the box covers them at every zoom from there. Boxes now start at zoom 15 (was 13), dots below; crowd bands from 15 (were 14). Replaces T-189's "true width, never narrower than the lines on screen", which kept the box a constant 8-13 px wide while its length doubled each zoom. Close in (zoom 17 on) boxes are much wider than the lines and neighbouring platforms overlap more; accepted as the cost of one shape.
- 2026-10-12 (T-205, Anita: "hide trains and stations when zoomed out, lower min line thickness like 40%"): below zoom 11 neither stations nor trains are drawn or picked; at 11 the screen holds about 30 km, a city such as Manhattan with its neighbours, and further out the dots and pills crowd the lines. A selected station or complex is still drawn and clickable, and demand circles stay (they are their own setting). Line widths go to 0.85 px at zoom 9 (40% of T-191's 2.1), linear to the unchanged 3.9 at 12, with bare track, casing and the selected extra scaled the same; a stroke under a device pixel is drawn one pixel wide and fainter in proportion, so far-out lines do not break up. Pick reach stays 7 px.
- 2026-10-12 (T-196, Anita: parks "by size and by population nearby"): OSM parks are 8% of the social/leisure destination draw, from ATUS 2023 (7.6% of US leisure outings end outdoors away from home; NHTS has no place type), taken evenly from the population and leisure-jobs draws so T-066's 50/50 between them stays; parks move outings and add none. A place draws sqrt(area) x D / (D + 2,000), D the residents per km² within 2 km of its edge: the square root is Anita's diminishing return, and a density rather than a count because a count grows with a park's edge (Harriman State Park outdrew Central Park five times). The density term saturates in built-up places so it does not double what the gravity already does for parks with many homes nearby. Nature reserves count as parks only when named as one ("... Park", "Recreation Area"), other named ones at a quarter, unnamed none (New England's land-trust and watershed parcels); playgrounds, dog parks and pitches are left out (inside parks, and mostly walks under the 150 m cut). No season. The exponent, the 2 km and the 2,000/km² are judgement. notes/T-196.md.
- 2026-10-13 (T-229, Anita: changing a line's trains "lags"): the stepper shows the asked-for count at once and clicks during an edit are folded into one edit sent when it lands; a train count sends a service-only state (the renderer keeps its buffers and takes the new run profiles), since rebuilding the whole network's buffers was half the edit's 85 ms on New York. notes/T-229.md.
- 2026-10-13 (T-230, Anita: "we should fix the demand forecast's average train to instead actually show something closer to the per train account"): every train-fullness figure the player sees comes from the boarding account. The line panel drops demand's busiest-hour average train ("Fullest train at high demand") for "Trains today": fullest train, trains leaving full and when, riders left on the platform, longest wait. A train with no account yet has no gauge, rather than the forecast's average in its place; the train inspector's typical train is this hour's trains from the account. The forecast's figure disagreed because the account's riders arrive fastest mid-peak (1.8 times the mean against demand's 1.4): 28 of New York's 93 lines under 100% in the forecast leave riders behind. notes/T-230.md.

- 2026-10-13 (Anita, answers): long-linked complexes get one dot per nearby group plus a thin connector (T-225); traffic width stays relative to the network's busiest stretch (full range for every network, T-227); nursing homes stay out of homes with prisons (T-234); the 1 km manual link limit stays ("a bit long but fine"); the 46/46/8 social and leisure split stays for now (Central Park possibly undercounted, T-240). Lines may be wider than station boxes in traffic mode.
- 2026-10-13 (Anita, after trying T-255): express lines and quad track are on hold, with design decisions still to make; no further work on them (T-257) until she says. An express line keeps its local line's colour. She is not sure Express copy differs enough from Copy line to need its own button (it moves stops onto the twin stations of the express pair); left in place for now.
- 2026-10-13 (Anita): the same trains run every day for now (T-261 closed). The demand views' day moves to Settings with three choices, the clock's current day by default (T-266).
- 2026-10-13 (Anita): traffic widths show the day's riders, not the hour's (T-274). Intercity: option 2 of research/intercity-review.md (retune the market and add commutes between cities), inside a world divided into zones: city zones, non-city zones between cities, unmodelled space (T-276); Trenton/Mercer probably part of New York. Routed walking and driving before tolls (T-275; T-270 tolls later). Toronto is the first non-US city, an exception to the pause on new cities (T-277). Waits stay as they are (half the headway, the first 5 minutes at 1.37 and the rest at 0.4, at boardings and transfers alike); Anita asked whether infrequent service is penalised, and this rule is the answer. notes/ belongs to the agents.
- 2026-10-13 (T-276 agent, docs/zones.md 6): the focus waits until every zone's boundary has loaded (falling back to the nearest centre until then), so it never flips as boundaries arrive one by one and opens a pack for nothing; where two boundaries share a point the registry's first zone wins. Nearby zones open only while Homes or Destinations is on (they exist for their bubbles), never during the startup load or a game load's clock hold, show no loading bar, and a solve never waits for them; once open they stay open like a viewed city's, at about 60-90 MB of WASM memory per solving worker each (Philadelphia 90, New Haven and Springfield together 57). A nearby zone's bubbles skip the catch-up and show the current forecast. A click on a nearby bubble focuses its zone and selects the bubble in one click: the focused zone takes over the nearby zone's worker and bubbles, and the old focus's become a nearby zone's.
- 2026-10-13 (T-276, docs/zones.md): the world is zones. A non-city zone is a pack like a city's (same pipeline, cells, purposes and views, `kind: "zone"` in the registry), one county-sized statistical unit each (US county, Canadian census division, NUTS-3), modelled when it touches two city zones (Worcester, Windham, New London, Litchfield); a coarse second model was rejected because every view, query and loading path would need a twin. How far modelled space reaches around cities is Anita's (open). Mercer County joins New York's pack (Anita's lean): in the New York CSA, LODES 175k jobs both ways with New York's counties against 88k with Philadelphia's; market id 2 keeps its name with no sites.
- 2026-10-13 (T-276): commutes are one doubly constrained gravity over every zone pack (`joint_gravity`), within-pack pairs with the pack's own curve and factor (so the runtime rebuilds them exactly), cross-pack pairs on the great circle cut at 150 km with a factor per region pair taken from the pack holding both regions, else the country's admin-1 default (US 0.23, as T-171 asked: not one global value), national borders 0.05 (US-Canada, judgement). Each pack's homes are scaled to its own jobs, not the union's: scaled to the union the solve forced net commuting through neighbours and barely converged. The cross-pack part (1.12M commuters, 5.9%) travels in the regional layer at commute timing, the local commute's mode choice and the work purpose's weekend factor; pairs under one commuter a day between 6 km zones are dropped and their commuters given to the home zone's kept pairs (172k pairs among the 13 zones, the market pack 8.2 to 13.9 MB).
- 2026-10-13 (T-276): the market's longer trips: 0.107 weekday departures a resident (was 0.015), onset 25-50 km (was 40-80), `max(d, 25)^-2.37` (was `exp(-d/180)/max(d,40)^1.2`), fitted together on NHTS 2017 Northeast weekday ground trips of 50-1,000 km, with an outside world (Kontur people in no zone) in every normaliser so 17% of departures leave the map. The car pays 15 cents a road km and the local parking, both over two occupants. Local outings keep their 40-80 km taper (the overlap from 25 km is under 2% of social outings; airport trips keep their reach). The market's trips between adjacent packs now include short regional trips across their seam, so they far exceed NEC FUTURE's metro-to-metro figures (New York - Philadelphia 380k against 57k); long pairs are right (New York - Boston 53k against 50k).
- 2026-10-14 (Anita): one save for the whole world, at least as the player feels it (T-304); London first; the sampler's extents as proposed (Greater London, Ile-de-France, Berlin and its ring, Shanghai's 16 districts, the Mumbai metropolitan region); non-city zones between cities only (T-300); line and station panels show only counts so far today, labelled "Today", never whole-day forecasts (T-293); T-122's frozen copies rejected (memory grows with open cities), instead compute what is on screen or likely to be looked at first (T-303); pack backups deleted.
- 2026-10-14 (Anita): the one-world design (docs/world.md) is approved. Local time per region as NIMBY Rails does it: the clock shows the local time where the player is looking, and each region's traffic runs on its own local time (T-309). Old saves convert to the world frame on load, correcting anything that moves substantially where that is easy (T-308).
- 2026-10-14 (Anita, T-310): the Money tab's "A day at this schedule" is gone (a whole-day forecast, as T-293 dropped elsewhere), with the code only it used (`income`, the per-day fare and running figures). The top bar cash keeps its fixed width but the digits sit right after the $. A reload does not restore the exact view: the URL hash only picks the nearest city or zone, and the map starts at that registry centre and zoom (nearest centre, not boundary, since it must be known before the boundaries load). The Lines and Stations lists sort by clicking column headers (a second click reverses, an arrow marks the column), not a dropdown; the sort stays a per-page-load setting as before. The grey veil's flash at the end of a pan was `cityBoundaries.ts` emptying the shade source on every change of focused zone before the new shape arrived; it now swaps in one setData from footprints made ready at startup.
- 2026-10-14 (T-307): the draw anchor moves on MapLibre's moveend only (not during a pan), so a long drag rebuilds the worker's buffers once; at 300 km from the anchor Float32 is still centimetre-exact. The grid is 1 degree, so a fresh anchor is at most about 80 km from the view, inside the 100 km reach. New York keeps (-73.98, 40.73) as the anchor until the view leaves it, so it draws and picks exactly as before. The main thread switches its anchor when the state built at it arrives, not when it asks, so nothing is ever converted with one anchor and drawn with another; previews crossing a switch are shifted (a translation in Mercator). The demand views stay on the home anchor for now (T-317): rebuilding the bubble workers on an anchor change is its own piece of work, and their discs are coarse enough that it matters only far from New York.
- 2026-10-14 (T-306): new games are in the Mercator frame of the whole Earth (TWT6, registry `frame.kind` "merc"). Lengths are measured on the ground: each piece of track keeps its ground length (straights as rhumb lines in closed form, arcs by Simpson's rule on 1/k) and chainage is ground metres everywhere, so profiles, capacity, service, boarding and costs kept their arithmetic; metre figures entering the plane are multiplied by k where the arc is (its middle, not the PI: up to a kilometre apart on wide turns, 0.02% of k a km at London). The conic and legacy frames keep k exactly 1, so their results are bit for bit as before. A plane circle is a ground circle only up to the rhumb line's own curvature (a 1,500 m curve at London measures 1,500.4 m on the ground); left as it is, being sub-metre. TWT6 keeps the game's home origin in its origin fields (rather than zero, as docs/world.md first said) so a conversion back to the conic has one. Track is refused within 5 degrees of the poles and across the antimeridian (`OutsideWorld`). `joint_gravity` takes its conic from the market's frame, since the registry's game frame is no longer one. App code comparing world metres with a metre figure goes through `groundScale`/`groundDist` (station complexes, links, nearest station). docs/world.md.
- 2026-10-14 (T-317): the demand views follow the draw anchor, each disc set carrying the anchor it was built at. The overlay draws every set with its own anchor's matrix and the hover finders get the pointer as longitude and latitude and turn it into each set's units (`discAt(discs, pointer, anchor)`), so while an anchor switch is still rebuilding the bubbles, the old ones stay exactly where they were instead of jumping. The bubble workers keep each cell's longitude and latitude, are told the anchor with their cells and again on every change (`sync`, before anything is asked of them), and answer with the anchor their positions are relative to. Disc sizes use k at the disc (`kAtY` of the disc's local y; `MERC_K` stays only for the stress network). So bubbles in New York are as before to a fraction of a pixel (k there varies by 0.3% over the metro, where the constant was New York's centre), and bubbles at Toronto are 4.7% larger on the ground than before (the old constant was 1.319 where k is 1.384); London's will be 18% larger than the constant would have made them.
- 2026-10-14 (T-308): old saves convert to Mercator when they load, in the clock worker and nowhere else: any network that is not TWT6 goes through `TrackApi.convert_save_frame` once, from the autosave, a slot or a file, and the slot is written again in TWT6 straight away. The old bytes are kept under a hidden IndexedDB key (`old-frame-<slot>`, not listed as a save) since a conversion cannot be undone from the new file. Money, fleet, fares and the ledger are copied; the player is told nothing. Nothing was corrected: on the real NYC save no line's run time moves over 0.3%, no line needs another train, no curve is newly refused (the save's 586 geometry refusals are the same set before and after), so a set radius, already read as ground metres, needed no special case. Where a legacy save moves more (Philadelphia track about 1.2%) the new length is the true one. Riders on the real save fall 0.4% (8.84 M to 8.81 M; the demand model re-solves on the new lengths and times). docs/world.md.
- 2026-10-14 (T-309): local time per region. One game clock keeps world time in UTC-5 (registry `clockUtcOffset`), New York's standard time, so saves made before it need no conversion and the Northeast and Toronto run bit-identically (checked on the real New York save: a week of trips, boarding slices and today's counts). Each registry entry has a `utcOffset`, whole hours of standard time with no daylight saving (a seasonal hour would move every period twice a year for nothing the game models). Half hours round down: India's +5:30 is +5, putting the game's 6-10 morning peak on Mumbai's real 6:30-10:30, nearer its late office peak than +6 would. A place is in the nearest registered centre within 120 km, else it takes `round(lon / 15)` until its region is registered (London 0; Paris also 0 until its entry gives +1). A line runs on its first stop's region's local time (simple, and a timetable has one clock); that carries its service periods, its boarding account's hours and kind of day, and its fares and running costs. Demand needs no change: a solve has no clock, so its period q is each region's own. The top bar, the demand views' period and their "Today" follow the local time at the view's centre, switching as the map moves; the line inspector and Lines tab use each line's own. Left for T-321: ledger days, the bubbles' catch-up, market trips across offsets. docs/world.md "Local time".
