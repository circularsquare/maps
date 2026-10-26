# Decisions

Newest first. One entry per decision: what, and why. The spec states the result; this keeps the
reasoning and what it replaced.

## 2026-10-26

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
