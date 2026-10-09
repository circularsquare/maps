# Decisions

Newest first. One entry per decision: what, and why. The spec states the result; this keeps the
reasoning and what it replaced.

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
