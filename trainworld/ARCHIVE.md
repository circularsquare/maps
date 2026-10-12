# Archive

- T-154 [P1] Compact intercity demand display. Done 2026-10-11: halved detailed and regional bubble radii, preserved area proportionality, reduced city connection bars to 5 px and replaced the visible percentage row with train/all trip counts; tooltip and ARIA detail retained. Production build and served browser review pass. notes/T-154.md.
- T-153 [P1] Short-trip walking correction. Done 2026-10-11: reserve walking before rail and driving compete, including exact-distance everyday disaggregation; focused native demand suite, production WASM, nine-city worker and school-purpose checks pass. Commute walking remains limited to 2 km gravity-zone distance. notes/T-153.md.
- T-144 [P2] Station rendering. Done 2026-10-11: track-aligned platform boxes and placement ghosts, separate off-by-default translucent demand circles, and box-based select/delete/line-stop picking. Typecheck, geometry, picking, demand-layer regression and production build pass. notes/T-144.md.
- T-151 [P2] Batch 10 integration and documentation tidy. Done 2026-10-11: feedback filed with original TODO intact, UI/data reviewed, served build and 9-city worker regression pass; browser review verified charts, loading selection and compact city panel. notes/T-151.md.
- T-150 [P1] Broader school inputs. Done 2026-10-11: NCES adds 3,199 public schools / 1,849,674 pupils across NYC metro outside DOE area; preserved DOE and other arrays, expanded child homes; source/pack/WASM checks pass. University and international sources researched; university implementation T-152. notes/T-150.md.
- T-143 [P1] Focused-city panel, demand readiness and purpose charts. Done 2026-10-11: top-five external train/other bars, preserved dimmed initial selection, purpose pie and matching timing colours; typecheck/build/browser review pass. notes/T-143.md.

Finished and abandoned tasks, newest first. Each: id, date, one-line outcome (or why abandoned).
Never delete entries.

- T-140 done 2026-10-11. NYC public-school attendance from 1,213 matched DOE buildings and
  child-population origins joins purpose/timing inspection, fares, loads and crowding. 150 m
  exclusion retained; native, production WASM, nine-city worker checks and build pass.
  Live visual QA unavailable; broader facilities/cities remain T-142. notes/T-140.md.

- T-141 done 2026-10-11. Lowered everyday-trip exclusion from 300 m to 150 m; exactly 150 m
  is eligible. Boundary tests, production WASM/worker checks and build pass. NYC totals are
  unchanged for the existing sampled cell endpoints. notes/T-066.md.

- T-066 done 2026-10-11 (NYC first slice). Sector/resident shopping/errands and social/leisure
  outings exclude below-300 m trips, route both directions separately and join shared fares/
  loads/crowding. Local trip-leg bubbles include purpose counts/shares and departure charts.
  Native, production WASM, nine-city worker checks and build pass; live visual QA unavailable.
  Further purpose/city rollout is T-140. notes/T-066.md.

- T-139 done 2026-10-11. Same-city longer-trip circles hidden at every zoom, exact zone market
  IDs retained through queries, commuter hover clarified. Exterior basemap shade and thin solid
  boundaries with darker outer/lighter inner focus edges. Native demand tests, production WASM
  contract checks, nine exact geometry masks, type check and build pass; live visual QA unavailable.
  notes/T-139.md.

- T-138 done 2026-10-11. Researched NYC non-work purposes, timing and survey denominators;
  bounded a NYC-first everyday-trip proposal using cached sector jobs/population, with directional
  schedules and explicit later facility layers. Research/design only, runtime unchanged. notes/T-138.md.

- T-137 done 2026-10-11. Added Baltimore, Providence, New Haven, Hartford and Springfield local
  packs, exact boundaries and regional endpoints; nine playable cities and ten stable market IDs.
  Independent pack/disjointness checks, direct WASM load, 26 browser checks including local riders
  and existing NYC saves, type check and production build pass. notes/T-137.md.
- T-136 done 2026-10-11. Longer-trip circle areas are proportional to total travellers without
  size floors/offsets; normal red-to-blue train-share colours retained. The selected city's
  grouped circle is omitted. Rendered-buffer area ratios, Boston exclusion and browser
  interactions pass; type check and production build pass. notes/T-136.md.
- T-135 done 2026-10-11. Selected longer trips use a visible relative scale; all positive zones
  remain up close and regional views aggregate exact totals to labeled city/town bubbles. Boston
  work-selection direction, all five places, zoom restoration and hide/show pass browser checks;
  type check and production build pass. notes/T-135.md.
- T-134 done 2026-10-10. Boston and Washington, DC are playable with local commute packs,
  water/boundary assets, city navigation and five-place regional demand. Reused established US
  sources and broad metro extents. Pack checks, 101 native tests, WASM/production build and
  browser city/line/query/old-save checks pass. notes/T-134.md.
- T-133 done 2026-10-10. Map picks follow visible layer and instance order; bubbles beat
  underlying network and delay tags, selection discs retain parents, and hidden layers cannot
  intercept clicks. Focused stacking/marker tests and real NYC browser interactions pass.
  notes/T-133.md.
- T-132 done 2026-10-10. None/Homes/Destinations is always visible at the map's upper left;
  choice persists and fits narrow maps. Removed duplicate Settings selector. Browser checks,
  type check and production build pass. notes/T-132.md.
- T-131 done 2026-10-10. Exact NYC/Philadelphia metropolitan demand-pack footprints outlined
  at regional zooms, fading away up close and surviving style reloads. Source geometry,
  production shipping and browser checks pass. notes/T-131.md.
- T-130 done 2026-10-10. Moved 103 diagnostic images to gitignored maps/artifacts/trainworld/
  with matching hashes; updated references, capture/plot producers and artifact rules.
  Content/gameplay now takes priority over proactive performance work. notes/T-130.md.
- T-123 done 2026-10-10. Compact regional-pack parsing and immutable-table capacity trimming
  save 236 MB on opening and about 46 MB after the full-network solve across five solving
  workers plus the auxiliary. Demand results match; 100 native tests, full fixture comparison,
  browser demand/memory checks and production build pass. notes/T-123.md.
- T-125 done 2026-10-10 (closes T-053). World coordinates: new games are in a Lambert conformal
  frame tangent at New York (scale 1 + dlat^2/2, under 0.4% from 35.8 to 45.8 N; New York to
  Philadelphia, Boston, Washington and Pittsburgh within 0.02% of the great circle, were off 0.2
  to 0.8%); saves made before it load in the old frame with identical geometry, costs, riders and
  money (TWT4 bytes unchanged, TWT5 for the new frame); one city registry
  (`game/registry.json`) drives water, packs, header, workers and build; demand and edit speed
  unchanged (edits +0.1 ms). Follow-ups T-126 to T-129. notes/T-125.md.
- T-053 done 2026-10-10 as T-125.
- T-120 done 2026-10-10. Longer-trip bubbles drawn like commuter bubbles: one disc per far 6 km zone
  at its own place (so no "Elsewhere in the region" blob), area by trips, blue train and red the
  rest, commuter opacity and size slider, no amber or name tags; per-zone sums from Rust keep the
  workers' answers additive; the inspector's own-region row is "Other New York region places".
  Zones under 0.5 trips a day dropped, at most 400 drawn. notes/T-120.md.
- T-119 done 2026-10-10 (with T-061). Demand solve on the real combined network: 1.7 s to the
  free-flow estimate (was 4.5 s), 6.1 s to round three (was 17.2 s), bubbles at 1.9 s (were
  17.4 s, after the last round), results unchanged to 2e-16. Strategy searches only towards
  stations a trip can end at and shared between populations, bubbles' sums sent with each round,
  a worker with no periods for questions, pool up to 5, step shown in the top bar. Found: the
  market's free flow leaves the fare out (T-121). notes/T-119.md.
- T-118 done 2026-10-10. Longer (intercity) trips shown by place for a bubble selection and for a
  station: inspector "Longer trips" sections, bubbles with labels, a settings check; native
  conservation tests and a real-app browser run pass. Also fixed the demand client serving the
  old city's cells to a view that reacts to a focus change. notes/T-118.md.
- T-117 done 2026-10-10. Separate non-commute intercity market (NYC, Philadelphia, Mercer
  County) with local access, shared crowding and per-pair attribution; constants are judgement
  values, not fitted OD. Native tests pass. notes/T-117.md.
- T-116 done 2026-10-10. New York and Philadelphia share one buildable save (NYC frame kept,
  both water masks, city navigation, regional demand coordinator, legacy saves load unchanged).
  Native tests (82) and type check pass. notes/T-116.md.
- T-115 done 2026-10-10. Philadelphia-Camden-Wilmington pack built: 110,578 cells, 6.245M
  residents, 2.981M jobs; gravity/water/source checks pass and New York stays unchanged.
  City-isolated ACS cache fixed; shared-world activation remains T-116. notes/T-115.md.
- T-114 done 2026-10-10. City tab diagnostic totals/period share chart replaced by city
  navigation; current New York button tested. Philadelphia navigation waits for T-116.
  notes/T-114.md.
- T-113 done 2026-10-10. Centered Are you sure? modal replaces bottom-left questions, with
  cancel-first/trapped/restored focus and blocked background input. Browser checks pass at
  900/400 px, including Escape and actual deletion. notes/T-113.md.

- T-112 done 2026-10-10. Edit stops above list exposes red removal controls down to zero;
  line deletion requires confirmation. Map additions test all slots, including infill and ends,
  with stale-query protection. Actual-click browser and native checks pass. notes/T-112.md.
- T-111 done 2026-10-10. Inspector badge edits colour and saved multi-character symbols,
  widens horizontally and raises text; redundant schedule text removed and headings clarified.
  Save/copy and narrow-layout browser checks pass. notes/T-111.md.
- T-110 done 2026-10-10. New line immediately opens an empty zero-train draft with its unique
  creation-order name selected for typing; independent Copy line preserves every setting.
  Native save/undo compatibility and browser interactions pass. notes/T-110.md.
- T-109 done 2026-10-10. New line and natural-name/daily-passenger sort above the line list;
  real New York's 109-line sorting and narrow layout pass. notes/T-109.md.

- T-108 done 2026-10-10. Unused blueprint extensions preserve constructed endpoint stops,
  profiles, fleet and capacity; planned lines retain full geometry previews. Native exact
  invariance and undo/redo checks pass. notes/T-108.md.

- T-052 done 2026-10-10. Short turns occupy shared platforms and conflict with crossing
  through trains or reversals; independent through directions remain independent. Constructed
  track alone sets active platform capacity. Native and whole-count browser checks pass.
  notes/T-052.md.

- T-075 done 2026-10-10. Clock loop skips duplicate draws when MapLibre queues a frame; 500-line/
  10,000-train pan check falls from ~119 to ~60 draws/s, halves draw work, with zero alignment lag.
  notes/T-075.md.

- T-071 done 2026-10-10. Blueprint inspector adds/removes corners and sets individual levels;
  new squares drag into curves, undo/redo restores exact shapes, constructed track stays guarded.
  Narrow inspector and browser checks pass. notes/T-071.md.

- T-010 done 2026-10-10. M1 spec and backlog match current schedules, fares, common lines,
  saves and map/UI behavior; oversized sections moved intact to linked detailed docs. Historical
  calibration stays dated, reference and section checks pass. notes/T-010.md.

- T-077 done 2026-10-10. Removed unused debug refusal export from production WASM, retaining
  native test diagnostics and UI issue kinds; track module is 627 bytes smaller. notes/T-077.md.
- T-068 done 2026-10-10. Named browser slots preserve the original save and protect against
  simultaneous writers; switching/rescue, two-tab, quota-failure and narrow UI checks pass. notes/T-068.md.

- T-094 done 2026-10-10. Nearby station platforms cap rider growth at 1.6x radius while isolated
  stations retain 2.6x; cached spatial lookup and real NYC GPU-buffer checks pass. notes/T-094.md.
- T-069 done 2026-10-10. Common useful lines combine waits and split riders by frequency for each
  destination; branch/fare/crowding/conservation tests and real NYC browser solve pass. notes/T-069.md.

- T-106 done 2026-10-10. Responsive top bar follows map width and keeps key values/playback
  controls clear; browser layout checks pass from 196 to 1248 px and after dock resizing.
  notes/T-106.md.

- T-107 done 2026-10-10. Separate cars appear from zoom 15; shared train width removes the
  sudden shrink. Browser pixel measurements agree across the transition; car picking, type
  check and production build pass. notes/T-107.md.

- T-105 done 2026-10-10. Track display offers Line, traffic thickness (default), Height and Max
  speed with geometry speed colours; old width preferences migrate. Narrow UI, actual width
  uploads, curve limits and persistence checks pass. notes/T-105.md.
- T-104 done 2026-10-10. Schedules allocate whole trains across the entire line; congestion-aware
  headways derive from counts. TWT4 persists them, older saves retain rates until edited. Native
  42 tests and browser schedule/save/quote checks pass. notes/T-104.md.

- T-103 done 2026-10-10. Close-up trains have actual car counts, cars following curves, windows
  and front cab; any car is pickable. Crowding colours use less white and a darker filled side.
  Real-save counts, curve fixture, zoom, direction, gauge and WebGL checks pass. notes/T-103.md.

- T-067 done 2026-10-10. Fares affect route/mode choice and fare-only edits recompute demand;
  base charged once, distance fare uses track length and stays separate from crowding. Native
  demand 16 tests pass, real-save fare/recompute checks pass. notes/T-067.md.

- T-087 done 2026-10-10. Train gauges choose period average or busiest hour (default); inspector
  lists both rider counts. Real-save gauge math, inspector and setting persistence pass.
  notes/T-087.md.

- T-102 done 2026-10-10. Delay indicators share station names' zoom cutoff and spacing; worst
  delays win collisions. Real-save pan/zoom and fixture priority, tooltip, inspector and display
  toggle checks pass. notes/T-102.md.

- T-101 done 2026-10-10. Station names hidden below zoom 12 and spaced more generously above;
  restored Zen Maru Gothic in map UI and form controls; commuter corner legend removed. Real-save
  zoom, spacing, font and visibility checks pass. notes/T-101.md.
- T-100 done 2026-10-10. Blueprint quote, undo/redo and Construct moved into Build; Select button
  removed; itemised track conditions, stations, junctions, flyovers, crossings and actual new
  fleet costs. Narrow UI and quote-to-charge checks pass. notes/T-100.md.
- T-074 done 2026-10-10. Station names move together during flat pans, with a padded viewport
  and periodic placement for incoming names; 300-station drag benchmark browser CPU 59.7% ->
  39.4% of one core. Real-save animated camera checks pass. notes/T-074.md.
- T-099 done 2026-10-09. Capacity markers sit under the demand bubbles and catchment and get a
  plain tooltip ("Junction at 104% of capacity", "Each train waits 1m03s here at high demand");
  a new game (or any network with no running line) now gives bubbles with no rail instead of
  keeping the last network's (`cell_modes` and `flows` answered empty with no subzones; fixed in
  sim/src/demand/api.rs with test `views_with_no_network`, needs the demand wasm rebuilt); a
  "Bubble size" slider in the settings pane; the grey-to-blue colour switch dropped (Anita kept
  the RGB mix). notes/T-097.md "T-099".
- T-086 abandoned 2026-10-09. Trains hiding busy stations from clicks is fine (Anita: players can
  zoom in).
- T-097 done 2026-10-09. Commuter bubbles replace the dot map: homes or jobs summed into H3 parents
  by zoom, area = commuters (no minimum), colour = RGB mix of red car, blue train, green walk;
  hover for counts; click, Shift-click or hold and drag a box to see where those people work (or
  live) as bubbles and in the inspector; switch in the settings pane; bubbles and a station's
  catchment drawn over the network, its far end filled and see-through. Nothing drawn per frame
  while playing; 0.01-0.2 ms GPU a camera frame. notes/T-097.md.
- T-095 covered 2026-10-09 by T-097: a station's catchment is now drawn over the network.
- T-098 done 2026-10-09. `flows(end, cells)` in game/demand.ts: for any set of home (or job)
  cells, all their commuters by mode spread over the far-end cells, totals equal to `cellModes`,
  the smallest far cells holding 1% given as cut totals; 10-20 ms in the workers for one cell or
  thousands, 250 kB-1.2 MB an answer. notes/T-098.md.
- T-096 done 2026-10-09. Keys 1, 2, 3 pick draw track, station and a new delete tool from any tab;
  Esc backs out one step at a time (question, drag, route, tool, selection); delete removes
  blueprint at once and asks above the bottom bar for anything constructed, as the inspectors'
  Remove buttons now do. notes/T-096.md.
- T-088 covered 2026-10-09 by T-096 (number keys, as Anita asked, instead of letters).
- T-090 done 2026-10-09. Soft station reach: the walk to a station weighs 2.0 a minute for 10
  minutes and 8.0 after, so a station's draw fades out by about 3-3.5 km with no edge (bounds of
  4 km and 80 perceived minutes for both walks are computational only); driving is 1.3x the
  straight line, local streets for 2 km at each end, main roads between, 40/80 km/h in open
  country falling to 15 where dense. Real New York network: rail 20.5% against ACS 20.3% (was
  22.1%), county error 3.5 points (4.3), Long Island and Westchester at ACS; lone Manhattan line
  477k riders, 93% (406k, 79%); a day 15-25% slower, pairs held. notes/T-090.md.
- T-089 covered 2026-10-09 by T-090: the faster suburban drive brought Long Island, Westchester,
  Connecticut and the outer counties to their ACS rail shares; inner New Jersey's 1.7x is the
  gravity (T-072).
- T-084 done 2026-10-09. Demand views: line width by riders an hour in the period in force (lines
  side by side at their own widths), station circles by riders a day, trains filled like a gauge
  (half = seats taken, full = crush, red rim over full); a commuter dot map by home or job coloured
  train/walking/driving (Hilbert-order carry, 1.7M dots for New York built in a worker in 0.25-0.5 s,
  0.2-0.65 ms GPU a camera frame, nothing per frame while playing), toggled from the top bar with
  a legend in the map's corner; a selected station's catchment and far-end places on the map and in
  its inspector. notes/T-084.md.
- T-038 covered 2026-10-09 by T-084: the commuter dot map coloured by mode shows rail's share by
  place, at cell grain, so a separate heat map would repeat it.
- T-093 done 2026-10-09. Blueprint track ends, junctions and stations drag with the select tool,
  previewed, one undoable edit, Escape cancels; a junction or station keeps its heading (each
  edge's first PI moves with it), a track end's follows its edge; constructed ones stay put.
  notes/T-093.md.
- T-092 done 2026-10-09. Clicks are points of intersection again (T-079 reverted, `geom::through`
  gone): a click changes nothing before the previous click's curve, which can only tighten;
  corner dragging and the radius stepper back; T-079 track loads unchanged and reshapes from its
  clicks. notes/T-092.md.
- T-091 done 2026-10-09. Running cost US$6 -> US$1.50 a car-km (`RUN_COST_CAR_KM`, params.rs),
  so fares beat running costs at 100x: the real New York save goes from -$0.65B to about +$1.1B a
  game day. The Money tab reads the figure from the track model, so it needs no text change.
- T-078 done 2026-10-09. One access mode, a fast walk at both ends (15 km/h, 2.5 km reach, weight 2,
  rail constant 0); the other leg gone. Real network rail 26.1% -> 22.1% (ACS 20.3%), county error
  6.2 -> 4.3 pts, fullest AM segment 481% -> 144%; lone Manhattan line 779k riders/162% -> 406k/79%;
  recompute 15-28% faster (pairs -32 to -37%). Per-cell modes and station riders on request in
  game/demand.ts. notes/T-078.md.
- T-085 done 2026-10-09. Dock and drawing polish: dock width and the tab/inspector divider
  draggable and remembered; Build tab notes gone; "Construct blueprints" on one line; the drawing
  tag reads "5.1 km, cost 1.10x, $507M"; trains win a click over stations. notes/T-085.md.
- T-079 done 2026-10-09. Track through the clicked points: biarcs through the clicks, stored as
  PIs with set radii plus the clicks (save TWT3, TWT2 loads; the real New York save loads
  unchanged); drawing and reshaping squares are the clicks; 4 new Rust tests. notes/T-079.md.
- T-080 done 2026-10-09. No flat/flying setting; new junctions flat; the junction inspector explains
  in plain words and offers "Build a flyover" with its price and the waiting it saves; junction
  capacity markers open it. notes/T-080.md.
- T-083 done 2026-10-09. The train inspector shows riders on board (an average train of the period
  on its segment) against seats and crush, with a load bar. notes/T-083.md.
- T-082 done 2026-10-09. Build tools (tool, level with its price, single track, platform) in the
  Build tab; undo, redo, blueprint cost and Construct stay on the map. notes/T-082.md.
- T-081 done 2026-10-09. Economy at 100x: one constant `ECONOMY` (params.rs) scales running costs
  and fares; prices shown stay real; Money tab says so. notes/T-081.md.

- T-034 superseded 2026-10-09. Fitting the drive/bus access leg and the rail constant to real
  shares: Anita dropped that leg (one fast-walk access mode) and made realism rough calibration
  only; the game-feel tuning is T-078. Its measurements stay in notes/T-034.md.
- T-035 abandoned 2026-10-09. The other access leg at the work end: no other leg exists any more
  (Anita, one access mode).
- T-073 done 2026-10-09. Idle frame loop on by default (`overlay.ts`, `?perfOff=idle` reverts):
  headless check, empty new game 2 overlay draws in 2 s (was ~120); New York's real network
  (109 lines, 1,756 stations, T-007 save loaded as the autosave) keeps drawing (10 fps unfocused).
- T-045 done 2026-10-09. CPU on two pinned cores, median of 5: new game playing 13%, dragging
  30%, unfocused 4%, paused 0%; with three lines 13% / 38.5% / 5.5% / 0%. Station labels cost 8.5
  points while dragging; an empty game draws 60 fps for the clock (`?perfTry=idle` takes it to
  0.6%, T-073); clock, symbol layer, overlay draw and markers cost nothing measurable. Measurement
  flags `?perfOff=` and `?perfTry=` left in `app/src/perfFlags.ts`. notes/T-045.md.
- T-051 done 2026-10-09. Two wasm modules, one per kind of worker (feature `track-api`, profile
  `wasm` with one codegen unit): the clock worker loads 157 kB gzipped and each demand worker 79
  instead of one 244 kB module (678 kB -> 412 + 194 kB raw); demand solve times unchanged (split
  886-942 ms against 914-981 on T-005's network). Both stay at opt-level 3: s or z made edits
  1.2-1.8x slower for 17-41 kB. notes/T-051.md.
- T-050 done 2026-10-09. Incremental capacity pass: an edit rebuilds only the resources around
  what it touched; equal to the global pass on random edit sequences. Capacity per edit in WASM
  at 10,000 km 0.2-1.2 ms instead of 13; whole edits there 1.5-2.4 ms (schedule, 1-line edge)
  instead of 12-13, the busiest edge 19 instead of 34-38 ms. notes/T-050.md.
- T-063 done 2026-10-09 (not per tile). At 2,000 km an edit cost 160-193 ms on the main thread,
  mostly station labels rebuilt and measured every edit; now 3-6 ms (round trip 26-36 ms): labels
  kept when stations are unchanged, render buffers built in the clock worker with per-edge and
  per-line caches. 10,000 km left for T-070. notes/T-063.md.
- T-022 done 2026-10-09. Eight MSA rounds logged on the real network and the single line: one
  round leaves the overflow segment at its worst (963% of crush against 310% free flow), a
  player-sized network settles in 2-3 rounds; the app now runs three; no crush cap (frequent lines
  stay under ~120% converged, the extremes are 0.4-train patterns, T-069). notes/T-022.md.

- T-064 done 2026-10-09. Blueprint track reshaped: drag a PI on the map with a live preview,
  step a curve's radius and set the stretch's level in the track inspector; one undoable edit
  each; constructed track refused. notes/T-064.md.
- T-054 done 2026-10-09. Home end of commutes = LODES RAC jobs by home block x (1 - tract
  work-from-home share): county totals within 10% of ACS commuters (the Bronx was 23% over); decay
  refit lands on T-006's ridge, so d^-1 exp(-d/29 km) stays; own county still 39% against 43%.
  notes/T-054.md.

- T-031 done 2026-10-09. Capacity display finished: junction inspector (diagram, moves with
  trains an hour, wait per train, which cross, busiest crossing) and the line panel's round trip
  split into running, dwell, turning and waiting, with the worst places named; queries
  `junction_info` and `line_delays` in TrackApi. notes/T-031.md.
- T-007 done 2026-10-09. New York's real rail network (109 lines, 867 stations, from GTFS) solved
  through `DemandApi` and compared with ACS, MTA OD and operator counts: rail 26.8% of commutes
  against ACS 20.3%; the boroughs within 2 points, every suburban county 2-10x over (no bus
  without parking, other leg free of parking); subway AM link loads top 20 493k against 505k
  (log r 0.92). Walking transfers (300 m) and 100 m station merging added to demand. Track-model
  save `T-007-real-nyc.save`. notes/T-007.md.

- T-062 done 2026-10-09. Lines sharing track drawn side by side in the stroke shader, in an order
  that holds along shared stretches, branch lines on their side; trains ride their own stroke;
  station dots span the bundle. notes/T-062.md, `../artifacts/trainworld/T-batch4-lines.png`.
- T-028 done 2026-10-09. Money: cars bought by whatever edit makes the lines need them (refused
  if short), running cost per car-km and fares (base + per km, from demand's riders) settled each
  game hour, Money tab with yesterday and today and the fare curve; $6B start kept.
  notes/T-028.md.
- T-029 done 2026-10-09. Local saves: the clock worker autosaves to IndexedDB (after anything
  paid for, else once a minute, and on page hide), the last game loads on start, save to and load
  from a file, new game; reload and import round-trip exactly. notes/T-029.md.
- T-032 done 2026-10-09 (with T-024). Station names stay DOM labels in Zen Maru Gothic; an
  invisible symbol layer with the same names above the basemap makes MapLibre drop colliding
  place names. notes/T-025.md.
- T-024 done 2026-10-09. Station tool (on track or a node, named after the nearest basemap
  street), line tool (click stations in order; add and remove stops; delete), saturated palette
  plus any colour, stops list with time from the start and the round trip. notes/T-025.md.
- T-023 done 2026-10-09. Track drawing tool: PIs at the selected level, snapping to nodes and
  track, automatic turnout lead-in at nodes, radius and km/h per curve, cost and refusal reasons
  by the cursor, single/double and flat/flying settings, colour by line or height.
  notes/T-025.md.
- T-055 done 2026-10-09. Blueprint and construct in the track model: built flags on edges and
  stations, built cost, lines run only on constructed track, construct (all, a line's needs, an
  edge, a station) paid and final, no refunds, undo for blueprint edits only. notes/T-025.md.
- T-025 done 2026-10-09. The clock worker owns the network (TrackApi with the pack's water mask),
  applies edits, sends states, keyframes and each hour's trips; the renderer positions trains
  from shared phase tables on the GPU (within 1.1 mm of float64); the demo world is gone; network
  snapshots feed demand. Edit round trip 4.7 ms. notes/T-025.md.
- T-026 done 2026-10-09 (with T-021). Commute demand runs live in the app: a pool of up to 3
  demand workers (Rust `DemandApi`, one WASM module with the track model) solves the five periods
  of `track::params` on every network snapshot; the top bar's split and the line and station
  inspectors' riders read `game/demand.ts`. Chrome, 3 workers, hand-made 434-station network:
  split 0.69-0.83 s after the snapshot, refined 1.34-1.52 s; 23 MB per worker; newer networks
  supersede running solves; nothing on the main thread. notes/T-026.md.
- T-021 done 2026-10-09 as part of T-026: periods side by side in the pool; mode choice not split
  by origin zone (a sixth faster at 3 workers, not worth a reduce step yet; T-061).

- T-056 done 2026-10-09. Basemap labels in Zen Maru Gothic Regular from our own glyph tiles
  (`app/public/fonts/`, 52 ranges, 3.8 MB, Noto Sans merged in for other scripts) made by
  `pipeline/glyphs.py`; style setup moved to `map/basemap.ts`. A New York session fetches one
  81 kB file (29 kB gzipped). `../artifacts/trainworld/T-056-before.png`, `../artifacts/trainworld/T-056-after.png`, notes/T-056.md.
- T-040 done 2026-10-09 (started 2026-10-08). Track model in `sim/src/track/`, in the default
  build: PI arcs, levels/ramps, validity incl. the pack's water mask (a ground-level Hudson
  crossing is refused), costs, undoable edits with dirty sets, exact run-time profiles (1 km stop
  to stop 67.5 s), capacity with holds (SPEC's 27 s / 81 s junction), keyframes as shared phase
  tables, saves, `TrackApi` for the clock worker; 28 tests. 2,000 km / 60 lines: an edit with its
  profiles 1.1-3.8 ms WASM, 1.3-4.3 native; unused track 0.03 ms; save ~100 KB per 10,000 km
  compressed. notes/T-040.md.
- T-006 done 2026-10-09. Gravity decay fitted to LODES OD 2023: d^-1 exp(-d/29 km) replaces
  exp(-d/9 km). Trips in the wrong 2 km band 15.0% -> 2.9%, county-pair log r 0.828 -> 0.881,
  mean 15.3 -> 18.4 km (LODES 19.6), within 10 km 36.3% -> 40.0% (LODES 39.1%). Pack `decay`
  `"pow_exp"`, still format 1. SPEC 9 tier-2 line; `../artifacts/trainworld/T-006.png`; notes/T-006.md.
- T-042 done 2026-10-09. Work from home left out of commutes: ACS 2019-2023 B08301 by tract (home
  end) and B08126 by industry mapped to LODES sectors (work end); `commute_home`/`commute_work`
  per cell, the gravity balances on them. New York 10.18M -> 8.61M commutes a day (Manhattan's
  work end -19%, the Bronx's home end -9%). notes/T-042.md.
- T-014 done 2026-10-09. LODES head-office lumps: 341k jobs (3.3%) moved by three rules.
  Northwell at North Shore University Hospital 41,521 -> 10,000 (rest over six counties'
  population); JetBlue's LIC office 9,012 transport jobs to JFK; 109 home care agency blocks
  (300,188 jobs, 172k in Brooklyn) over their county's population. County-government and
  school-district lumps listed, left for T-043. `pipeline/lumps.py`; notes/T-014.md.
- T-013 done 2026-10-09. Blocks spread over their polygon by land area (50 m pixels, water
  excluded, per-county TIGER2020PL polygons). New York cells with people 72,460 -> 198,594, with
  jobs 55,910 -> 190,612, cells 74,920 -> 202,337; pack 2.2 -> 6.9 MB raw (3.8 MB gz with the
  water mask); recompute 452 -> 508 ms native, rail share unchanged. `../artifacts/trainworld/T-004.png`; notes/T-013.md.
- T-049 done 2026-10-09. Themes renamed pink, blue, green in the UI, CSS and SPEC 8; a
  remembered old name maps to the new one.
- T-048 done 2026-10-09. Empty inspector is blank (no text); its top stays at 360 px.
- T-047 done 2026-10-09. Train wiggle was the hard rim-to-core edge stepping a pixel at a time,
  plus heading steps at each 50 m sample when zoomed in; fixed in the shader (blended rim,
  heading from the train's two ends, padded quad), core flicker halved, GPU cost unchanged.
  notes/T-047.md.
- T-046 done 2026-10-09. Space bar toggles pause and resumes the previous speed; ignored while
  typing in a field.
- T-037 done 2026-10-09. Settings behind a gear in the tab row: track colour by line or height,
  show trains, station names, capacity markers (stub until T-031), map labels, and the theme;
  remembered per browser. `../artifacts/trainworld/T-037.png`. notes/T-027.md.
- T-027 done 2026-10-09. The game shell replaces the spike at `localhost:8800/trainworld/`:
  Preact + signals, app split into game/ map/ ui/ workers/, dock with fixed-height tab area and
  an inspector for every selection kind (empty by default), top bar, build toolbar, three
  themes, demo world in one swappable file, clock worker and protocol stubs, `?debug=1` readout.
  `../artifacts/trainworld/T-027.png`, `../artifacts/trainworld/T-027-line.png`. notes/T-027.md.
- T-015 done 2026-10-09. Overlay canvas drawn inside MapLibre's frame with its camera matrix; the
  one-frame panning lag is gone (every drag frame off by one step before, none after, by camera
  probe and composited frames), CPU playing 17-21% against 35% for the custom layer; overlay is
  now the only mode. notes/T-015.md.
- T-030 done 2026-10-08. Water mask in the city pack (still format 1): 25 m raster over the
  boundary's box as runs per row, from TIGER 2020 AREAWATER (77 counties) plus the sea outside the
  TIGER states; water under ~50 m wide dropped. New York 11,410 x 11,647 pixels, 266,401 runs,
  1.11 MB raw (~0.6 MB gz), water 40.2%; 18 named places checked; layout and lookup in
  notes/T-004.md, `../artifacts/trainworld/T-030.png`. OSM route is T-041. notes/T-030.md.
- T-020 done 2026-10-08. Access logit over access options and egress stations (logsum, 0.2 per
  perceived minute) and the other access leg at the home end (30 min + 2.5 x time at 25 km/h,
  4 nearest stations within 20 km); walk bands dropped. New York: commutes with rail access 33% ->
  51%, other leg 31% of rail, 24% of rail from homes beyond walking distance; vs per-cell
  reference station entries/exits 5.5% (was 5.8/6.5%), segments 2.3% (was 3.0%); recompute
  0.45 s native, 0.55-0.72 s WASM (was 0.7 / 0.78-1.04). notes/T-020.md.
- T-019 done 2026-10-08. Pack format 1 ships the 2 km zone gravity (zone id per cell, two
  factors per zone), solved by the sim crate's own code via `pack_gravity` from build_city.py;
  New York city open in WASM 8.0 s -> ~7 ms; pack 0.87 -> 0.97 MB gzipped. notes/T-019.md.
- T-017 done 2026-10-08. 10 fps when the window is visible but unfocused (clock keeps running),
  nothing when hidden or paused; CPU focused 30-36% layer / 17-21% overlay, unfocused 3-9%,
  hidden and paused under 1%. notes/T-018.md.
- T-016 done 2026-10-08. Keyframe times relative to an hourly render epoch, re-based in float64
  and re-uploaded as (offset, time) only; GPU vs float64 reference within 2 cm at day 0, 30 and
  365 (absolute time: 1.5 m median at day 30, 20 m at day 365); plus a clamp that stopped
  km-long one-frame jumps. notes/T-018.md.
- T-018 done 2026-10-08. app/ on MapLibre 6.13, npm audit clean; matrix from
  `defaultProjectionData.mainMatrix`, worker URL via `setWorkerUrl`, both `?mode=` work; globe
  projection does not draw our layer (T-033). notes/T-018.md.
- T-009 done 2026-10-08. Static UI mock at `localhost:8800/trainworld-mock/` (`mock/`): left
  dock with tabs and a line inspector, top bar with mode split and clock, build toolbar; three UI
  token sets (cream, sky, matcha) and two fonts to compare, one saturated line palette;
  `../artifacts/trainworld/T-009-a/b/c.png`. Awaits Anita's pick. notes/T-009.md.
- T-008 done 2026-10-08. Track model and capacity design: straights and arcs drawn as PIs,
  a_lat 1.1 m/s², min radius 100 m, levels 8 m with 4% grades (200 m ramps), double-track routes
  with a single-track option (topology open for Anita), Kingman-style delay from occupancy times
  baked into profiles as holds, cost table and $6B start, edit dirtiness rules; save ~170 KB for
  10,000 route-km. SPEC 6 rewritten. notes/T-008.md.
- T-005 done 2026-10-08. Demand kernel spike in `sim/src/demand/` (feature `demand`): gravity
  between 2 km zones, mode choice between access subzones. Real New York pack: recompute (free
  flow + one crowding round) 0.5-0.7 s native, 0.8-0.9 s WASM; synthetic 153k cells 0.7-1.0 s
  native, 1.0-1.3 s WASM; edit the same; peak 22-26 MB; gravity once per city 2.5-8 s WASM.
  notes/T-005.md.
- T-002 done 2026-10-08. Render spike at `localhost:8800/trainworld/`: trains positioned on the
  GPU from keyframes; our JS ~0.05 ms/frame at any count, GPU 0.13 ms at 10k, 0.36 ms at 100k,
  60 fps up to 3M (headless used the real RX 6600); the basemap repaint each frame (~2 ms, ~26%
  of a core) is the real cost; paused ~1%. notes/T-002.md.
- T-004 done 2026-10-08. New York pack format 0 in `data/packs/nyc.{json,bin}` from
  `pipeline/build_city.py`: 74,920 res-9 cells, 21,736,575 people, 10,176,424 LODES 2023 jobs,
  1.80 MB (0.87 MB gzipped), 15 s build; checked by `pipeline/read_pack.py`. notes/T-004.md.
- T-003 done 2026-10-08. New York = 2023 NY-Newark-Jersey City MSA plus Fairfield CT, Dutchess
  and Orange NY (25 counties); `data/packs/nyc.boundary.geojson`. notes/T-003.md.
- T-001 done 2026-10-08. Toolchain skeleton: `sim/` Rust crate → wasm-pack → loaded in a module
  Web Worker in the Vite `app/`, result sent back as a transferred buffer; registered as
  `trainworld` on serve.py. Headless check: 10,000 train offsets in 0.6 ms, WASM loaded in
  5 ms, 18.65 kB .wasm.
