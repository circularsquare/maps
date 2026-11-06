# Decisions

Newest first. One entry per decision: what, and why. The spec states the result; this keeps the
reasoning and what it replaced.

## 2026-11-05

- **Desert rivers run on water carried in from outside the desert** (T-079, notes/T-079.md), for
  the canoe and the land hazard's drinking water alike (`tools/hotdesert.py`). T-042's test (a
  reach's own driest month >= 0.1 m³/s) kept Wadi ad-Dawasir, which WaterGAP gives 292 m³/s and 98
  in its driest month. Outside hot desert seasonal rivers stay (the paddler picks the season), so
  Cooper Creek and the Diamantina, fed from Queensland, still paddle; a source on flow permanence
  (Messager et al. 2021) would be the way to drop them.
- **Small corridor nodes keep the search's parent** (T-077, notes/T-077.md): a node with fewer
  spread trips than 2% of the drawing threshold neither spreads nor re-chooses, which skips the
  share work for 82% of nodes in an L3 desert window; lines change by a few per level. Slack is in
  the search's weight (time + K x hazard) but the 3% and 1% budgets stay shares of time, so a canoe
  start's sea hazard does not widen them on land.
- **Walking starts run the general search** (T-067, notes/T-067.md): with land hazard they need
  time and hazard arrays; the general loop took 1.1-1.5x the walking loop's time at L10, so no
  hazard-carrying copy of the walking loop was written. `solveWalk` stays for hazard off.
- **Survival shows in the hover from 0.5% risk** (T-067), on both starts; it showed whenever the
  path had any hazard. With land hazard 95% of the cells reached from Cluj have some, nearly all
  tiny, and "survival 99%" (rounded down) on a walk to Vienna would overstate the risk fiftyfold.
- **Corridors count a re-chosen tree over land** (T-056, notes/T-056.md): a land node takes,
  among near-equal land parents, the one most trips pass through when each trip is spread over
  routes within 3% (time), keeping every tree route within 1% of the fastest. The search's own
  tree bunched sectors into grid-direction spokes and parallel strands whose place is set by
  speed noise. Costs about 2-3x the old counting time; the search and time map are unchanged.
- **Land danger as designed** (T-042, docs/hazard.md section 8; Anita): OSM wells, springs and
  villages count as water on today's map ("we can do well/spring/village as water"); each place
  takes the kinder of January and July ("more intuitive"); a path stops under 1% survival on land
  as at sea; the traveller as proposed (no polar gear, no disease or violence counted). She also
  offered marking explicit desert areas "if it's easier and has similar effect"; distance to water
  is kept since it is built and also covers dry gaps outside named deserts.
- **Carrying the canoe stays at twice the walking time** (Anita: "twice walking seems
  reasonable"; T-071).

- **River water off the river line is a dead end, not still water** (T-036; was: no canoe node, unpainted). Each such cell is entered from the shore, a canoe node beside it, or a cell nearer its patch's edge, at 5 km/h, and left only to the shore. Full still-water edges would let a route paddle along the line's own water against no current (up the Amazon at 5 km/h against 3) or up an unrunnable river's gorge; dead ends paint every cell and allow only crossings of river water one cell wide, at both transfers. Carrying gains those crossings on canoe starts (L1 windows in Laos and the Amazon: 15-25% of nodes over 5% earlier); the K trips are unchanged. Revisit if river water should carry the river's current. notes/T-036.md.

- **Upstream is max(p - 0.4 v, 2.5 km/h), carrying stays x2** (T-071; were 0.6 and 1.5). Recorded upstream crews made 2.5-3.5 km/h per moving hour on big rivers (Lewis and Clark, Mackenzie, HBC brigades, a 2015 canoe up the Mississippi), and the bank current on a big river is ~0.25-0.4 of the mean; single-carry portages run 1.5-2 x walking time. Revisit with a small-canoe upstream log on a huge river, or if small rivers (4.3 km/h up at Q = 1) prove too fast. notes/T-071.md.

- **Globe on MapLibre 5.24.0, the time layers unchanged** (T-059). 5.24.0 is the last 5.x (2026-04-23); 6.x (6.16.0 now) was not tried. MapLibre 5 draws canvas sources on the projection's subdivided tile mesh, linear in Mercator within the tile, so the Mercator-y canvases bend onto the sphere as they are: no plate carrée repaint, raster tiles or custom layer. The globe is always on (no switch) and turns flat at zoom 11-12 by MapLibre's own rule. Revisit if a later MapLibre stops reprojecting canvas sources, or for 6.x. notes/T-059.md.

- **Corridor speed is along the river and smoothed** (T-064). A river step counts the river's own length (so a reach reads its model speed, not its lower over-ground speed), averaged over 6 edges each way of the same mode; lines are cut for speed only 0.4 km/h past a class edge, and coloured by their own mean km/h. Revisit the window (`SMOOTH`) if real changes, like a stretch of forest at L1, read too blurred.

- **Window edge strip fixed by more canvas rows, not by moving the hole** (T-037). `paint` makes up to 4 canvas rows per cell row (capped at 20 M pixels, so L10 gets 2), because rounding the hole outwards opens a gap and inwards doubles the colour; a hairline of about 1 px remains. Revisit if memory on phones matters.

- **100 m goes where T-047's rule puts it** (Anita: "2.8gb seems reasonable. that rule seems
  fine"): towns at 2% urban by quarter-degree block, mountains at 70% rugged, water country with
  people. Not taken: mountains from 50% rugged (~3 GB more), towns from 5% urban (~0.8 GB less).
- **10,000 BC moves to later** (Anita: "not really that necessary"); roads and 100 m come first.
- **Cells stay crisp at zoom 9-10**, no smoothed band edges (Anita: "not sure if smoothing is the
  right way forward... maybe rounding / diagonals makes more sense. or a hex map would also look
  nice"); diagonals and a hex look are tested as painting only (T-060).
- **Hex is the default cell look** (2026-11-05, Anita: "the hex looks nicer somehow. more
  omnidirectional"); `?look=crisp` and `?look=diag` stay (T-060).
- **Corridor colour is speed along the stretch**, one ramp for every mode, and width is how many
  trips share it (Anita: "lets just have color be directly reflecting speed, and not mode";
  T-062). This replaces T-052's yellow-to-dark-red by share and its paddled blue. A globe view
  (T-059) and instant hovers of our own (T-061) asked for the same day.
- **River lines go faint** (Anita: "they look like routes"); the routes a canoe takes show as
  corridors over them (T-052).

## 2026-10-26

- **L0.1 windows sit on 0.05 degree steps**, not the 0.25 of L3 and L1: at zoom 12 the 0.25 steps
  made windows 2-4 times the needed size (0.54 M cells and 0.6 s in Cluj, now 0.12 M and 0.1 s).
  **Ways tagged `foot=no` count as not walkable**, like motorways, so parkways and fenced
  expressways are not walked. A canoe start stops at L1 until T-050 (T-038).
- **River lines are the canoe's own river steps**, not the basemap's waterways (T-045): the
  basemap draws only some rivers when zoomed out and has no size, and ours show exactly what the
  canoe can use. Cost: 4.2 MB for the world view and 100-300 ms per rebuild on the page (T-048).
  **Hill shading sits under the basemap's water and the time colours**: above them at low
  opacity it also shaded the sea floor (Terrain Tiles include ETOPO1 bathymetry).
- **Canoe speed rows show only on canoe starts** in the speeds panel (T-044); walking rows always.
  A 1 m/s current stands for "strong" (the top 0.2% of the drifter grid; Gulf Stream ~1.3).

- **K stays 1000 h** after the check trips (T-006): it is the lowest tried that sends Brest to
  A Coruña round the coast; 3000 and up send Bergen to Iceland round by Denmark and England.
- **Exposure x3 counts every water term past 20 km**, supplies included, since rates are per real
  hour (T-006). **Getting into a canoe node on a land cell costs the whole step carried**, not
  half paddled: half a step paddled let paths hop from river cell to river cell overland faster
  than carrying (Constanta to Izmir went over Thrace that way; T-006).
- **Roads come from OSM** (Anita: "yes can use osm"), for the 100 m level and for the 1 km roads
  layer (T-010) alike; published road tiles fall under ODbL. GRIP4 dropped: no bridges or paths.
- **No 100 m world at 15-30 GB** (Anita: "lets stop before we get there"). The fine level goes
  only where it adds something (T-047); test regions first (T-038).

- **8 moves at sea past 20 km from shore is on** (T-022): with 16 everywhere a canoe click at L10
  took 1.0-1.1 s on real data, over the design's 1 s limit; 8 gives ~0.8 s, with open-sea bands
  up to 8% long on diagonals. Back to 16 if the search gets faster (T-035).
- **Canoe L3 windows are capped at 1.5 M cells** (T-022): at the walking cap of 3 M a canoe L3
  window took ~2 s, since its water is solved too. The L1 cap stays 2 M (~1.2-1.3 s).
- **Time shares the weight array while hazard is off** (T-022): with every hazard 0 they are
  equal; keeping them apart cost ~15% speed for nothing until T-006 sets `cfg.hazard`.

- **River gradient north of 60N comes from GMTED, not RiverATLAS** (T-020): `sgr_dk_rav` there
  (HYDRO1k-based) sums to a 13 km drop down the lower Yukon and closed most Alaskan main stems;
  GMTED's drop over 10 km gives the Yukon 109 m from Fort Yukon (about 130 real). South of 60N
  `sgr_dk_rav` is kept as designed.
- **OSM rapids close a step on rivers of 1000 m³/s or more** (T-020): the DEM gradient misses the
  Livingstone Falls (the Congo reads 0-1 m/km below Kinshasa), and a named rapids on a river that
  big is not run with a laden canoe. Smaller rivers' OSM rapids are only the rapids class.
- **A dam or weir outranks the gradient on its step** (T-020): the DEM drop at a dam is the dam
  itself, so with the gradient first the Iron Gates stayed closed in 10,000 BC.
- **Sea ice from the least-ice month** (T-033, orchestrator's call under "the season a paddler
  would choose"): August north of the equator, February south. The June-August mean closed
  Hudson Bay (37%) and the whole Arctic coast. Waves, currents and sea temperature keep the
  3-month summer mean.
- **Straits MAP walks over get a canoe node** (T-021): MAP's 1 km surface is land along the
  Bosporus and across the Dardanelles, so an L1 land pixel touched by the NE 10 m ocean polygon
  keeps its land node and gets a sea canoe node. Costs a ~1 km canoe fringe on most coasts;
  the alternative, editing the walking land mask, would change walking times.
- **Our surface's land cover table follows MODIS where no class matches** (T-009): shrubs 4.2,
  open broadleaf forest 4.86, moss and lichen 4.86, the speeds MAP mostly gives those pixels,
  replacing the draft guesses in research/friction.md 3. Wetland stays 2.0 and Greenland keeps
  the terrain factors although MAP differs in both, since ours follows MAP's stated recipe.
- **Our surface stores uint16 km/h x 1000** (T-009), not the page's uint8 x 40: steep high
  ground goes under 0.1 km/h, where x 40 steps are 25% errors. 1 km pixels with half or more
  water are water; GMTED30" mean for elevation until T-011.
- **The basemap is quieted in the page, by layer type** (T-029): road lines (casings hidden) stay dark but at 0.3 opacity and half width, labels (not water) shrink to 0.85x in grey. Dark faint roads read better under the colours than light ones would; the numbers are `ROAD` and `LABEL` in `index.html`.
- **The canoe is always carried overland**, at twice the walking time, never left (Anita, T-005).
- **The search counts time and danger**: cost = time + K x hazard, a linear combination, on every
  mode (Anita: "count both time and danger, can use linear combination"). Replaces 2026-10-14's
  fastest path with survival only reported; a time-only search took deadly sea shortcuts.
- **The orchestrator commits `isochrone/` paths as it goes** (Anita); pushing still needs her word.
- **Small water counts as 1 km/h land in the L3 and L10 means too** (T-013), so L10 times from
  Cluj rise 3-8% (ponds were dropped from the mean before). Kept for one rule at every level;
  to undo, leave converted pixels out of the L3/L10 means only. notes/T-013.md.
- **Canoe design choices** (T-005, docs/canoe.md): portage is not its own edge (unrunnable river
  steps are dropped and the search carries round); wide rivers MAP marks as water get no grid
  edges, only river edges, so they never become a current-free channel; 16 moves at sea, as the
  128-day limit keeps L10 sea work to ~1 M cells (8 moves is the fallback); RiverATLAS over
  HydroRIVERS plus a DEM, for its reach gradient; the sea is painted on a canoe start; K starts
  at 1000 h, tuned on check trips in notes/T-005.md.

- **Orchestrator setup** (Anita asked to work on it "more seriously", like trainworld): CLAUDE.md
  brought up to trainworld's current rules, milestones in TASKS.md, TODO.md hers only, open
  questions in notes/for-anita.md. Answers left open for 12 days were decided below, to show after.
- **Walking becomes directed** (downhill faster, directed Tobler) once our own elevation lands
  (T-011); research/friction.md recommended it, and the canoe needs a directed graph anyway.
- **Small water under 5 km² is walkable** at MAP's 1 km/h (T-013): paths barely change; it removes
  the dark specks at L1 and the false "pond blocks walking". Easy to revisit.
- **10,000 BC land cover from Beyer 2020** (era-correct, 0.5 degree) over Hengl PNV (sharp, but
  today's climate puts forest over Younger Dryas Europe). Hengl stays the pick for a "today
  without clearing" toggle.
- **Season: annual on land, local summer at sea** (research/canoe.md 3 and friction.md 6 each
  recommended one; this joins them). A switch later.
- **The page stays build-free; worker.js splits into ES modules** (T-019) before the canoe graph
  lands, rather than moving to Vite and TypeScript: 8800 keeps serving the working tree live, no
  snapshot build as in trainworld. Revisit if the toggle panel needs components.
- **Big downloads run detached by an agent** (stage C's 100-130 GB of GLO-90, T-011), overnight,
  per Anita's standing "run long jobs yourself".

## 2026-10-14

- **View refine thresholds and sizes** (T-002): L3 from zoom 5, L1 from zoom 7; windows are the
  view plus a quarter of its size each side (the L3 under an L1 window, its full size each side,
  so it seeds L1 well); caps 3M cells (L3), 2M (L1). L1 seeds from the L3 window, not L10:
  seeds are better and the L3 window is cheap (~30-100 ms at zoom 7+). Measured times in
  docs/resolution.md.
- **Overlays are canvas sources painted in the worker** (T-002), replacing a PNG data URL made
  on the main thread. Canvas sources let the page cut a hole in the coarser layer where a finer
  window is (otherwise coarse colour shows through the window's water) without re-encoding the
  world image; the worker does all the per-pixel work.
- **5 degree tiles at both levels** (600 cells at L1, 200 at L3): divides both grids exactly and
  lines L3 and L1 tiles up. `data/tiles/` (135 MB) is gitignored and rebuilt in 45 s, like the
  raw tif.

- **Docs and tasks follow trainworld's layout** (Anita): CLAUDE.md, SPEC, DECISIONS, TASKS,
  ARCHIVE, TODO, notes, research, docs. Compartmentalised work goes to subagents.
- **Fastest path, survival reported afterwards** (Anita): "the pathfinding should still just take
  fastest path i think and not care about survival". Hazard is a per-cell rate; survival along the
  fastest path is exp(-sum hazard x time).
- **Canoe is the next mode** (Anita: "the most interesting mode of transport to add next").
- **Dynamic resolution: coarse first, refine the view** (Anita: "compute it coarsely at first,
  then wherever we're looking at we refine"). Design in docs/resolution.md. Her suggestion of a
  coarser grid in uniform regions and a finer one in mountains is kept as a later stage: the view
  refine comes first because it also gives the close-up look she asked for.
- **Modes as layers, toggles remove layers or edges**, so each toggle combination does not need
  its own prepared data (orchestrator, so toggles stay cheap as modes are added).
- **The MAP friction surface is the walking base for now**; CC BY 4.0 allows it with credit. Our
  own surface from primary layers is needed for 10,000 BC and finer terrain (research T-004).

## 2026-10-13

- **Walking first, whole world, click anywhere** (Ashley). Static page with the search in a worker,
  no server, so it can be published like the other maps.
- **10 km cells for the first version**: 7.5M cells solve in ~0.3 s in the browser. A cell's speed
  is the mean of its 1 km pixels' friction (time per metre), so crossing time is kept; a cell is
  land when a quarter of its pixels are.
- **A day is 8 hours of walking** in the legend and hovers.
