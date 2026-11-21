# T-3444 research: the code side of track, platforms and stations (agent "code", 2026-11-21)

Read-only survey. No builds, no runs. Paths are relative to `C:\Users\anita\projects\maps\trainworld` unless absolute.
Line numbers are from the working tree on 2026-11-21 (master e778ce68 plus uncommitted edits by others).
Helper scripts and their raw output sit beside this file: `archive_counts.py`, `archive_wide.py` (+ `archive_wide.txt`,
313 entries), `gitlog.py` (+ `gitlog_summary.txt`), `sizes.py` (+ `sizes.txt`).

---------------------------------------------------------------------------------------------------------------------

## 1. Representation

### Storage: flat arrays by stable id (`sim/src/track/net.rs`, 959 lines; methods in `net/*.rs`)

- **Network** (`net.rs:597-808`): struct of arrays. Nodes `node_x/y/level/flying`, plus station fields living *on the
  node*: `node_platform` (length, 0 = no station), `node_name`, `node_named`, `node_built`, `node_off`,
  `node_upgrades`, `node_one`, `node_fare`, `node_own`, `node_berth`; derived `node_ports`, `node_moves`,
  `node_heading`. Edges `edge_a/b/tracks/built/off/pis/thru/crossover/kind/sep/oneway/join/limit/ramp`, derived
  `edge_infer` (the inferred way), `edge_len`, `edge_pieces`, `edge_vert`, `edge_cost`, `edge_time`, etc. Lines
  `line_stops` / `line_path` as spans into pools, plus ~40 per-line arrays.
- **NodeData** (`net.rs:134-140`): x, y, level, flying.
- **EdgeData** (`net.rs:142-187`): `a`, `b`, `tracks: u8` (2 double default, 1 single; comment at 146), `pis` (interior
  PIs), `built`, `off`, `thru` (legacy clicked points), `crossover` (one diagonal of a crossover, 161-162), `kind`
  (TrackKind: regular, HSR, tram, street tram), `sep` (imported real-network grade separation), `day`, `oneway`
  (0 not set / 1 a->b / 2 b->a / 3 pinned both), `join: [u8;2]` (per end: 0 none, 1 left, 2 right track of the
  double it meets), `limit`, `ramp`.
- **Ports** (`net.rs:96-105`): per node up to `MAX_PORTS` edge ends `edge<<1|end`, each with `side` (0 against the
  node heading, 1 along) and `rank` (right to left on that side). Every edge end meets the node at its centre.
- **StationData** (`net.rs:217-244`): `platform: u16`, `name`, `named`, `built`, `off`, `upgrades: u8` (bits),
  `berth` (ferry port), `day`, `one: u8` (T-3332 one-track platform, 0 both, 1/2 a way along a fixed axis),
  `fare`, `own`. **A station is a node**; there is no station object apart from the node, no platform object, and
  no record of which tracks have a platform beyond `one`.
- **LineData** (`net.rs:246-310`): `stops: Vec<u32>` (station node ids), `path: Vec<u32>` (`edge<<1|dir`, dir 0 = a to
  b), `hand` (order set by hand, T-2895), `ones: Vec<(u32,u8)>` (player's one-way stops, T-3711), `one_way`,
  `twin: Twin` (T-3333), and schedule fields.

### Lanes and the move table (`sim/src/track/lanes.rs`, 544 lines; T-3137)

- `Network::lanes(e)` (`lanes.rs:82-95`): derived, never stored. Single = one lane on the centre line with the
  edge's way (`edge_way`); double = two lanes at `±LANE_HALF_M` (2 m), right lane carries a->b with right-hand
  running. `Lanes` holds **at most two lanes** (`lanes.rs:35-38`, `lane: [Lane; 2]`).
- A path entry `edge<<1|dir` names the lane on double track (direction = track). `lane_of(p)` (`lanes.rs:98-101`).
- `Moves` (`lanes.rs:47-61`), `derive_moves` (`104-129`): per node, for each arriving end, a bitmask of ends it may
  leave by: **any end on the other side** with a lane running each way the move needs. Positions across the node:
  `rank * PORT_PITCH_M` (8 m) plus the lane offset. Two moves cross when `(a1-a2)(b1-b2) < 0` (`Moves::at`, 64-70).
  A plain two-ended node never changes lanes (no wrong-line running; DECISIONS T-3137 C1).
- **The way of single track**: `edge_way` (`189-196`): player's `oneway` else `edge_infer`, which `derive_ways`
  (`335`) works out per *run* of single pieces through 2-end nodes, by end votes (`end_vote`, `259`): a join gives the
  way of the double track it joins, a bare end says nothing, a station at a bare end / a junction with no join / double
  end-to-end says both. Runs meeting where only single track meets take their way together (`run_group` 424,
  `group_ways` 458, T-3672). Pre-0.4 saves pinned (`pin_station_runs`, 274). Joins count only where `join_applies`
  (`247`): node of 3+ ends or a double's loose end, with double on the end's other side. `join_away` (`202`).
- Readers of the table: routing (`net/paths.rs:198-252` `find_path_side`), `check_line`, the flat-junction badge
  (`capacity.rs:328` `junction_crosses`), junction capacity (`capacity.rs:551` `build_nodes` moves), the train guard
  (`conflict_nodes`, `capacity.rs:783`).

### One-way single track (T-3094)

Edge flag `oneway` (save flag bits 11-12). `dir_ok` skips entries against the way. Capacity: `directed()` (`capacity.rs:339`)
treats it like one direction of double; two-way single shares one key (`key_of`, `344-351`). An edit that changes a
way reroutes lines through the same stops (`TrackWorld::way_reroutes`, `world.rs:1995`).

### Quad: two doubles 8 m apart, nothing joins them

- `QUAD_SPACING = 8.0` (`wasm.rs:1682`); `express_verts` (`wasm.rs:1703`) lays the express pair concentric, swinging
  out over 150 m; `quad_beside` (`wasm.rs:1798`) finds the other pair's loose end (along within 2 m*k, across
  `0.5..1.5 x QUAD_SPACING` = 4-12 m, heading within 0.03 rad, `wasm.rs:1818`).
- Doubling up: `double_up` / `double_up_op` (`wasm/crossings.rs:12, 50`), refuses `Alongside` when a pair is beside.
- **Twin found by geometry, several times over, with different thresholds** (the core of the quad problem):

| Rule | Where | Distance | Parallel | Scope | Cost |
|---|---|---|---|---|---|
| `twin_at` (station naming beside, free station, and doubling up's `Alongside` refusal at `wasm/crossings.rs:104-106`) | `sim/src/track/wasm/draw.rs:1305-1337` | 4 m (0.5 x 8) to 15 m (1.25 x 12) | sin <= 0.03 | **every edge** (`for f in 0..edge_count`) | O(E) per call |
| `station_beside` / `on_track_beside` (two pairs, per-direction stops) | `sim/src/track/twin.rs:96-142` | 4-16 m (`BESIDE_MIN_M`, `BESIDE_MAX_M`, 53-58) | sin <= 0.1 | 8 pieces each way (`BESIDE_HOPS`), only stations in `complex_of` | complex_of is O(stations) |
| `quad_beside` (carry both pairs on) | `sim/src/track/wasm.rs:1798-1840` | 4-12 m across, 2 m along | 0.03 rad | node's neighbourhood | |
| `add_crossover` (figure from docs 6.1, body not read) | `sim/src/track/wasm/crossings.rs:202` | 6-15 m | parallel | nearest | |
| drawing beside (a laying distance, not a finder) | `sim/src/track/wasm/beside.rs:9` `BESIDE_SPACING = 12` | 12 m | | | |
| map ribbons (`besideSamples`) | `app/src/map/corridors.ts:10-12,126` | 20 m (`CORRIDOR_M`), samples every 25 m | | edges with lines | 60-350 ms on midtown (T-2768) |
| quad hover / "Quad" tag | `app/src/map/quadPairs.ts:8-11` | 8 ± 3 m (`SLACK_M`) | cos >= 0.95 | 50 m grid | |
| complex | `sim/src/track/twin.rs:147-225` `near_enough`/`complex_of` AND `app/src/game/complexes.ts:28-56` | centres 100 m, platforms 50 m | | **all stations per call** | O(N) per call, called per stop |

  The sim copy's constants carry the comment "game/complexes.ts PLATFORM_GAP_M" (`twin.rs:64-67`): kept in step by hand.
  So: at least six separate finders checked in code (twin_at, station_beside/on_track_beside, quad_beside,
  besideSamples, quadPairs, the complex rule twice), plus add_crossover's 6-15 m from the docs.
  (`stationSnap.ts:56` `orderOverlapping` only ranks candidate tracks under the pointer; it finds no pairs.)
- **Hidden twin line** (T-3333): `LineData::twin`, `Twin::{Parent{split}, Of(l)}` (`twin.rs:13-22`); `twin_plan`
  (`229-260`) builds the twin's stops from `station_beside` of every stop; `TrackWorld::twin_ops` (`world.rs:1850`)
  keeps it in step in every edit. Capacity, timetables, boarding, money and demand see two lines; the app folds them
  (`listedLines`, `foldTwins`). Open follow-up T-3335 (panel parts that still show one pair).

### Crossovers (T-338, T-1688)

A crossover is two `Edge`s with `crossover: true` (save flag 32), an S between two parallel edges 6-15 m apart, split 60 m
either side; no `Op` of its own (the edit is `Op::Batch` of Split/Edge ops). `TrackApi::add_crossover` /
`preview_crossover` (`wasm/crossings.rs:202, 216`); the worker op `crossover` exists, no UI sends it (ui-building.md
"Crossovers", line 760-765). Within one double there is no crossover: changing track inside a double is not
representable; turning back at a station is abstract (capacity's turn places) and drawn by the app
(`turnbackStrokes`). Single-lead links are a move-table entry, drawn by `singleLeadStrokes`.

### Station upgrades drawn as track (`sim/src/track/upgrades.rs`, 97 lines)

Bits on `StationData::upgrades` (`upgrades.rs:8-17`): `TERMINAL` 1, `PARK_RIDE` 2, `WAITING_ROOM` 4, `PASSING` 8 (save
node-flag bits 4-7). Neither the terminal's sidings nor passing tracks are edges: capacity reads the bits
(`turn_places`, `capacity.rs:534-538`; `overtake::passing_at`, `overtake.rs:35-38`, requires `active_node_tracks > 1`),
the app draws them (`passingTracks`, `app/src/map/network.ts:1113`; terminal end bars, `terminalBays.ts`).

### One-track stations (T-3332) and the single-track station split (T-3212)

- `StationData::one`; queries in `net/stations.rs`: `one_track` (20), `one_platform` (25), `no_passing` (30),
  `one_side` (40-45; only at a 2-port node on double), `one_for_side` (47), `run_passes` (54-63, a run that passes on
  the other track: `Run::pass`). Placing: `one_at` (`wasm/draw.rs:1253-1273`), switch: `wasm/onetrack.rs` (134 lines).
  Stored as a way along a fixed axis (`one_for_travel`, `net.rs:196-202`) so renumbering can't flip it (but T-3354,
  "one-track platform jumps sides when dragged north-south", is still open, not reproduced).
- **What the sim models at a plain station on two-way single track**: `build_sections` (`capacity.rs:446`) cuts the
  single-track section there (a passing point) unless `no_passing` (a one-track station); its platform is one shared
  track (`capacity.rs:588`, `track = 0` when `active_node_tracks == 1`), and it pays one-platform upkeep
  (`one_platform`, `UPKEEP_ONE_PLATFORM_SHARE` 0.6, `wasm.rs:326`). **What the app draws**: the track splitting into two
  tracks with a platform each side (`singleSplitEnds`, `app/src/map/terminalBays.ts:51-105`; ui-map.md "Lines and
  track" T-3212 and "Stations" T-3564). So the passing loop is free, has no geometry, and is drawn by the app from its
  own rule over nodes and edges. (Docs: track-and-service.md 6.1 line 248 says "a single-track station has one shared
  platform track and gets no passing loop"; 6.2 line 864 says sections split "at every station but a one-track one".
  Both match the code if "no passing loop" means "no PASSING upgrade"; it reads as a contradiction.)

### Per-direction stops (T-3348 stage 1, `sim/src/track/perdir.rs`, 440 lines)

A line stores one stop node. A run whose way the stop's platform does not serve stops at the nearest station of the
stop's complex, else any within `WAY_WALK_M = 300` m (`perdir.rs:13`), whose platform its trains pass on their own track
within `WAY_COS = 0.5` (60 degrees) (`way_group` 45-62, `way_station` 64-70; `way_group` also scans every node, 55).
Nothing is saved: `way_stops` (357), `back_run_of` (404), `back_run` (437). Lines with such stops are rebuilt on **every**
edit (`Network::back_lines`, `net/paths.rs:149-166`, which also scans all edges and lines each edit). One-way stations
(T-3565): `leg_stop` (100), `route_legs` (215), `take_legs` (291), and `wasm/legs.rs` `flatten` (27) puts their place in
the stop list from the track, not the clicks.

### Platforms

Zero width in the sim. Length 60-400 m (rail) in 20 m steps; centred on the node at a through station, running from the node
at a dead end (`platform_reach`, `net/stations.rs:128-136`: plat/2 through, plat at a 1-port node). `check_station`
(`138-194`): no more than 2 ports (`StationJunction`), reach + the other end's station reach must fit on each edge
(`PlatformStation`, `PlatformJoint`, `PlatformJunction`, `PlatformTooLong`), not on a ramp, no flat crossing within, not at
ground over water. The app draws the box at a fixed ground width (1.125 x a line, ~24 m at NYC), `stationGeometry.ts`
(711 lines), samples 13 points along the track (`PLATFORM_POINTS`).

### Station complexes, buildings, passages

- Complex: derived, never saved. App `game/complexes.ts` (173 lines; 100 m centres, 50 m platforms, links, buildings);
  sim `twin.rs:147-225` the same rule written again. Demand gets `st_group` (`demand/network.rs:44-47`).
- Passage: `Network::links` (`net.rs:798`), `LNK1`; priced `upgrades::passage_price`.
- Building: `building.rs` (402 lines), outline of 3-24 corners; members are stations whose platform centre line is inside
  or within 8 m (`TOUCH_M`); saved `BLD1`; not in undo history.

### Edits and undo (`sim/src/track/world.rs`, 2,244 lines)

`Op` (`world.rs:20-44`): `Node`, `Edge`, `Station`, `Line`, schedule ops, `Split`, `DeleteEdge`, `Batch`. Undo is record
replacement, so new kinds of record need new Ops. Every checked edit runs `way_reroutes`, `twin_ops`, `check`
(`world.rs:1850-2036`).

---------------------------------------------------------------------------------------------------------------------

## 2. Stops and stop picking

- **Storage**: `LineData::stops` are station node ids; `path` is the exact `edge<<1|dir` list from first to last stop
  (`net.rs:251-254`). The save writes stops as node-id deltas and path as edge-id deltas with the dir bit
  (`save.rs:280-294`). A stop *is* a platform choice only in that each pair of quad, each leg of a split, and each
  one-track platform is its own node.
- **Which track at a station**: on double the direction picks the lane, so there is no choice. On quad the stop node
  says which pair. Per-direction: the return run picks another node by `way_station`. One-way legs: `route_legs`.
- **"Pass"** has three meanings in code:
  1. `Run::pass` (`service.rs:34`): a stop kept in the per-stop arrays where the train goes by without stopping (a
     one-track station's other track, T-3332; the other run of a one-way station or one-way stop, T-3565/T-3711), no
     dwell, no riders (`span` -1).
  2. passing tracks (`PASSING` upgrade): a station where a faster train overtakes (`overtake.rs`).
  3. passing points on single track (any station but a one-track one, `capacity.rs:446`).
  Plus "stations a line passes" (on its path, not a stop): `stop_passed`, the stop list's "N stations passed" rows.
- **Click a stop** (`wasm/lines.rs`): `add_stop` (346) / `insert_stop` (340) fit the station where `best_slot`
  (`wasm.rs:1327`) puts it (every insertion position, route length + reversal penalty `order_cost`, at least 5 km per
  reversal, T-1899); `straighten` (`wasm.rs:1379`); `edited_stops` (`lines.rs:1117`); `distinct_stops` (1105);
  `settle_ones`. `set_stops` (30), `order_stops` (45, T-2895: sets `hand`), `sort_stops` (62). Stop tiers for the
  markers: `stop_tiers` (`lines.rs:452`), one query per station, tiers on/route/reroute/cut/tram/port/reverse/close/open/
  turn/unturn (`app/src/map/stopMarkers.ts:34-56`).
- **Exact stop ordering**: today only T-2895 "stop order by hand" (drag rows in the line panel; `LineData::hand`, saved in
  `LHO1`; ui-building.md 1054-1066). The older manual mode that listed every visit (T-3023) was dropped by T-3109;
  saves' `LMN1` fold into open lines (`save::fold_manual`, `save.rs:1538`). The `?exact` switch is gone (grep finds
  nothing in app/src). T-3079 (open, P3) is "exact stop ordering back" with waypoints (T-3384 folded in), to be styled as
  Anita's numbered circles per track in notes/T-3444.md.
- **Markers**: one DOM marker per station node (`stopMarkers.ts`, 579 lines), sized by zoom (9-24 px), no pointer events;
  the map's pick decides. Quad pairs' nodes 8 m apart overlap at most zooms; there is no "push aside" layout today.
- **Routing per hop**: `find_path_side` allocates `dist`/`prev` of `2 * edge_count` per call (`net/paths.rs:200-202`): a
  stop insertion test is O(E log E) per hop, memoised per edit by `HopMemo`.

---------------------------------------------------------------------------------------------------------------------

## 3. The app side

- **Track tool** `app/src/map/tools.ts` (1,359 lines; split in T-2374): `snap` (334), `joinPoint` (437-470: with single
  picked, the pointer's side of a double's centre line picks a track; within `DEAD_PX` it merges both), route requests carry
  `single`/`quad` (927-969). Parts: `stationTool.ts` (450), `toolTouch.ts`, `toolDelete.ts`, `toolMarks.ts`,
  `toolParts.ts`, `stationSnap.ts` (266; `orderOverlapping` 56, `roomNear`, `likelyFirst`), `realSnap.ts`, `edgeEdit.ts`,
  `nodeDrag.ts`, `blueprintBox.ts` (Select box, 574).
- **Track bar** `ui/ToolBar.tsx` (219): Single/Double/Quad icons (quad 116); station bar `PlatformTracks` (189, T-3332
  one or both tracks). Select bar `ui/Blueprint.tsx` (497): 1/2 icons, Double up button (216-232) via
  `game/blueprintPick.ts` `doubleUpPieces` (177), `doubleUpPicked` (204) -> worker `doubleUp` (`workers/clockEdits.ts:166`).
- **Drawn network** `app/src/map/network.ts` (2,295 lines), built in the clock worker: `networkSteps` (1550-~2100), a
  generator since T-3803. Per edge: polyline from the sim (`edgeParts`), then app-side changes:
  - `lanePoints` (819) and the double-to-single taper (`TAPER_M` 40, `taperAt` 1576),
  - `legShifts` (615) / `shiftedEnds` (717): single ends drawn on the joined track (reads `e.join`), easing back,
  - `splitPoints` (850) with `singleSplitEnds` (`terminalBays.ts:51`): two-way single split into two at stations, 60 m
    leads (`SINGLE_SPLIT_LEAD_M`, terminalBays.ts:17), flag `F_SPLIT`,
  - `turnbackStrokes` (896): assumed crossovers at a line's first and last double-track stop, 50 m,
  - `singleLeadStrokes` (983): the single-lead link of a joined two-way single,
  - `passingTracks` (1113), `linePassing` (1152), `passLead` (1175), `innerStops` (1184): outer tracks bowing out 100 m,
    shortened before junctions (T-2248).
  Per line: `lineBands` (`lineBands.ts`, 1,582) for lines side by side on shared track, using `besideSamples`
  (`corridors.ts:126`, 399 lines) to find parallel edges within 20 m; `bandPoints` (1229); train samples per line
  (`runSamples`, 1757-1808), including the return run's own samples (T-3250). Cache: `NetCache` (1398) keeps edge,
  line, station, speed and sample entries; an edge entry is kept when its polyline, ends, flags, lanes, split key and
  shift key match (1598). Then `facilityPlace.ts` (564, waiting rooms and car parks), `buildingNet.ts`,
  `oneWayMarks.ts`, `complexPassages` (373).
- **How a station on track redraws neighbours** (T-2767/T-2768): placing a station splits an edge (`Op::Split { edge, s,
  node, new_edge }`: one half keeps the id, the other is a new edge); both halves' polylines and ends change, so their
  cache entries miss, the lines over the split edge (8-15) are rebuilt, and `besideSamples`
  resamples the neighbours of the changed edges (60-350 ms in dense midtown). T-2768's fix made `lineBands` redo only
  lines reading a changed pair rule (109 -> 14 lines on a delete). Not fixed: "a split that leaves the polyline's shape
  unchanged" is still a change to the neighbours, since band records do not name the pieces they read (notes/T-2768.md).
- **The rebuild is whole-network** (T-2404, T-2701): `networkFromState` walks every edge, line, train sample and station
  to find what changed and compares posted arrays (notes/T-2404.md table, lines 48-60). T-3803 option 1 made it a
  generator of ~0.1 s pieces with edits going ahead between them; option 3 kept facilities per complex (~35% cheaper).
- **Renderer** `renderer.ts` (1,155), `rendererDraw.ts`, `rendererStrokes.ts`, `rendererShaders.ts` (1,307; lanes in the
  shader: `u_lane_mode`, `a_lane`, `LANE_NARROW`), `strokeCache.ts`. Trains: `trainGuard.ts` (1,117) reads
  `NetworkBuffers.conflicts` (= `guard_conflicts`) and `singles` (= `guard_singles`) and the drawn single split's two tracks.
- **App rules that copy sim rules** (kept in step by hand): complexes (`game/complexes.ts` vs `twin.rs`), the station
  ghost's first-frame guess (`stationSnap.ts` `likelyFirst`, `layoutIssues` vs `check_station`; T-3893 had to make the
  ghost's join guess compare ramp, limit and kind "as the worker does"), the quad pairing (`quadPairs.ts` vs `twin_at`).
  Possibly also the passing station's run-out (`passWay` in network.ts 1653 and `PASSING_RUN_OUT` in
  `wasm/rebuild.rs:427`); I did not compare them.

---------------------------------------------------------------------------------------------------------------------

## 4. Save format (`sim/src/track/save.rs`, 1,801 lines; docs/architecture.md section 11, lines 469-575)

- File: `TWG1` = JSON header + network bytes, gzipped (`workers/saveFile.ts`). Network bytes: magic `TWT6` (Mercator;
  TWT4/TWT5 still read and converted), origin, `right_hand`, then nodes, edges, lines, then trailing blocks.
- **Nodes** (`save.rs:218-237`): mm deltas, level, one varint of flags: bit 0 flying, 1 station, 2 built, 3 off, 4-7
  upgrades (terminal, park and ride, waiting room, passing), 8 named, 9-10 `one`; then platform length and name.
- **Edges** (`save.rs:238-266`): a and b as deltas of dense node ids, one flags varint: bits 0-1 tracks, 2 built, 3 has
  thru, 4 off, 5 crossover, 6 HSR, 7 sep, 8-10 kind (tram), and only on single track bits 11-12 oneway, 13-14 join a,
  15-16 join b; then PIs (mm deltas, radius, level), thru points.
- **Lines** (`save.rs:267-295`): name, colour, tph or trains, dwell, turnaround, cars, stops (node-id deltas), path
  (edge-id deltas << 1 | dir; a deleted edge written as none, T-2820).
- **Trailing blocks** (`save.rs:107`, 38 of them, in order): LUI1, LNK1 (links), LAU1, LHS1, LBS1, LBS2, LRT1, LGR1, LGL1,
  LOR1, LPH1, LSP1, LFR1-3 (ports), BLD1 (buildings), LBD1 (build days), LBP1, LHO1 (hand order), LMN1 (old manual),
  LBH1, LOW1, LSL1, LPC1, LTW1 (two pairs / twins), LPS1 (piece speed limits), LWE1, LOS1 (one-way stops), LPD1, LHR1,
  LFZ1, LSN1, LFD1, LGN1, LRA1 (ramp per level), LWP1, LFB1, LBR1. Track-related: LNK1, BLD1, LTW1, LPS1, LRA1, LOS1,
  LHO1, LBD1.
- **Save levels**: `LEVEL = 34` (`save.rs:112`), raised by one per new block, new header field meaning or new meaning of
  bytes; a Rust test pins `(LEVEL, BLOCKS.len()) == (34, 38)` (`save.rs:1704`). Named old levels: `OLD_STATION_WAYS` 18
  (single track at a station made its run two-way before; loads pinned), `OLD_BUILDING_NAMES` 23, `OLD_BOUND_NAMES` 26,
  `OLD_BUS_ROWS` 28. A save above this build's level, or with unknown blocks or flag bits, plays read-only
  (architecture.md 534-551).
- Sizes (T-3137 notes, 11-10): the real NYC save has 1,647 edges, all double, 8,293 PIs; network 128 KB raw, 75 KB
  gzipped with lines.

---------------------------------------------------------------------------------------------------------------------

## 5. Bug history and why

### Method

`archive_wide.py` selects ARCHIVE.md + ARCHIVE-to-2026-10-19.md entries mentioning track kinds, platforms, stations,
crossovers, joins, flyovers, legs (313 entries, 2026-10-09 to 11-21). I read them and put task ids in classes by cause,
in `classes.py` (counted by script into `classes.txt`; unit: task id, each id in one class). Two kinds per class:
**fix/patch** (a bug fix, a patch over a gap in the model, a refusal or limit lifted) and **feature** (something built
to cover the gap). **The classification is mine**, from one-line archive entries; keyword counts from
`archive_counts.py` are not causes (for example "lane" also matches demand's SIMD lanes).

### Counts (from classes.txt)

| Class | fix/patch ids | of them dated 11-10..11-18 | feature ids |
|---|---|---|---|
| A single and double as kinds of one edge; joins and inferred ways at nodes | 26 | 26 | 2 (T-3135, T-3137) |
| B pairing found by geometry (quad, twins, ribbons, complexes) | 13 | 6 | 9 |
| C track drawn but not modelled | 11 | 8 | 3 |
| D stops as nodes, return run derived | 11 | 11 | 4 |
| E stations as nodes that split track | 15 | 3 | 1 |
| F same rule in sim and app | 3 | 2 | 0 |
| total | 79 | 56 | 19 |

**Confound**: single track came back to the Build tab on 2026-11-10 (T-3093), one-way track the same day (T-3094), and
lanes and joins landed 11-11 (T-3137). Class A (all 26 in nine days) and D (all 11 in nine days) are the bug tail of new
features, not a long-run rate. E is spread over October, B over both months.

### Classes, with the ids (descriptions below include some feature ids for context; the table's split is classes.py's)

**A. Single and double as two kinds of one edge, with joins and inferred ways at nodes.**
T-3135, T-3137 (the C1-C4 remodel itself), T-3134 (ghost drew every segment double), T-3162/3163 (branch from one track),
T-3164/3161/3189 (unusable one-way badge, single-lead link, false flat badge), T-3171 (double off single drawn wrong),
T-3190/3191 (single from a double's loose end), T-3204-3206/3209 (leg end shifted off its line, `legShifts`), T-3262
(count lost across type switch), T-3344 (double drawn at single ends: trains crossed lanes with no crossover), T-3379
(kink where a short single joins a double), T-3380 (delete preview two tracks wide), T-3386 (inferred way lost at a
station), T-3647 (T-3386's rule broke tram lines in 0.4), T-3537/3538 (street snap merged a double's split), T-3609
(double splitting into two singles showed a flat junction), T-3618 (same, drawn-first case, closed unfixed "the rework
covers it"), T-3709 (stale split check), T-3533, T-3534, T-3580 (drawing by count: widths, stripes, traffic twice thick),
T-3904 (t3137 checks). Open: T-3353 (second single from a double's end refused), T-3354.

**B. Pairing found by geometry (quad, twins, beside, complexes).** (T-3206: the quad hover wording.)
T-255, T-423, T-442 (quad express chord refusals), T-708 ("curve does not fit" artefact), T-711, T-2009 (quad price and
multiplier), T-727f (express pair became one 5,400 km edge), T-733 -> T-1688 (auto crossovers on quad added, then removed:
design churn), T-1678, T-1681 (doubling up refusals), T-1687, T-1689 (which pair gets the station), T-1846 (ribbons
flickered where side-by-side tracks were cut at different places, `corridors.ts`), T-1859 (complex rule moved to drawn
platform lines), T-3261 (quad merged to two at each snapped click), T-3339 (quad going single per pair), T-3356 (two-pair
switch didn't show: her stations clicked a little off level, beside rule widened), T-3363 (station on both pairs in one
click, folded), T-3605 (one-way leg stations paired within 50 m, pairing removed). Open: T-3335, T-3348.

**C. Track drawn but not in the model (passing loops, sidings, splits, turnback crossovers).**
T-2233, T-2248, T-2249 (passing tracks drawn, shortened before junctions, island crowd), T-1527, T-3053 (terminal drawn,
trains stopped halfway into the platform), T-3202, T-1967 (opposing trains on single track drawn through each other),
T-3212 and its follow-up c97eb4ba (split thickening zoomed out), T-3403 (profit view on single-track splits), T-3540,
T-3564 (platforms on single-track stations drawn one side, then both again), T-3606, T-3615 (ghost and box widths for
one-sided stations), T-3569 (t3212 check fixture). Open: T-3354, T-3264 (trains drawn through each other).

**D. Stops as nodes, the return run derived (per-direction, one-way stations, legs).**
T-3094 (way back), T-3102, T-3250 (return trains on their own way drawn wrong), T-3422 (return legs on blueprint),
T-3332 (pass), T-3348 stage 1, T-3411, T-3429, T-3541, T-3562, T-3565, T-3595, T-3605, T-3651 (train took the passing
tracks instead of the side platforms), T-3672 (side platforms still bugged), T-3700 (stale check), T-3711 (shift-click
one-way stop, a workaround for the Caracas zigzag). Most came from player saves (Caracas, Nagoya) 11-16 to 11-18.

**E. Stations as nodes that split track, platform centred on the node, reach checks.**
T-258, T-622, T-624, T-760, T-860, T-1155, T-1161, T-1299, T-1667, T-1699, T-1708, T-1964 (58 refusals at joints under a
platform), T-2412, T-3286, T-3629, T-3648; also T-3690 (git f06dbf47, not counted).

**F. The same rule in sim and app, kept in step by hand.**
T-3893, T-3078 (state copied raw edge kind, app read it as tram), T-3139 (sim heading y north vs map y south); also of
this kind but counted under A or B: T-3134, T-3171 (sim right, drawing wrong), T-1859, the complex rule (two copies).

79 fix/patch ids and 19 feature ids, against about 2,880 archive entries in all.

### Git

`gitlog.py`: 641 commits touched `sim/src/track`, `app/src/map` and the track panels from 2026-10-09 (a75adfd0) to 11-21. Commit
subjects bundle tasks, so subject keywords are weak evidence; file churn is better. Most-touched files: `wasm.rs` 148
commits, `tools.ts` 138, `renderer.ts` 114, `net.rs` 104, `network.ts` 91, `params.rs` 81, `world.rs` 67, `save.rs` 61,
`service.rs` 60. In fix-like subjects: `tools.ts` 31, `wasm.rs` 28, `net.rs` 21, `renderer.ts` 21, `network.ts` 19,
`world.rs` 15, `draw.rs` 11.

### Why: which representational choices make the bugs

1. **Track count is a property of one alignment** (`EdgeData.tracks`, lanes derived from it, at most two). Everything
   that is "which track" happens at nodes, where an edge end meets the node at its centre; joins (`EdgeData.join`) and
   inferred ways (`derive_ways` over runs, votes, groups) patch it in. Every new join case (double's loose end, double
   drawn at single ends, double splitting into two singles, street snap) needed its own fix, and the drawing
   (`legShifts`, taper, split) re-derives the same answer from the same flags. T-3137 noted that mid-edge the model is
   fine and everything that breaks is at nodes; the fixes since are still at nodes.
2. **Groups of parallel tracks are not stored**; at least six finders work them out from geometry, each with its own
   thresholds (table in 1). A quad station, two pairs, the Quad tag, the ribbons and the station naming can each disagree about
   whether two tracks are a pair (T-3356 is that case). The hidden twin line is the cost of having no quad object.
3. **Station-only track is drawn, not modelled**: a passing loop at a plain single-track station, passing tracks, terminal
   sidings, turnback and single-lead crossovers have no geometry in the sim; the app draws them by its own rules
   (`singleSplitEnds`, `passingTracks`, `turnbackStrokes`, `singleLeadStrokes`) and trains are placed on them by the
   guard and `terminalBays`. Each drawing rule has to agree with a capacity rule it cannot see.
4. **A stop is a node, the return run is derived**: where the two directions use different platforms, the sim works out
   per run which node to stop at (`way_station`, legs, `back_run_of`, `back_lines` every edit), and the stop list order
   for one-way stations is computed from the track. The Caracas and Nagoya reports all landed here.
5. (smaller) **A station is a node that splits track**: the reach checks, joins under a platform, trims and kept pieces
   (`keep_under_stations`, `station_trim_ops`, `unjoin_near`) exist because the platform is not its own object.

---------------------------------------------------------------------------------------------------------------------

## 6. Replace versus keep, and the seams

### Earlier options on the same question (all estimates pessimistic by CLAUDE.md's own note: T-3137 C1 was guessed at days and took about 45 minutes)

- T-3137 (b) "two objects: each double is two directed single tracks" with a grouping record: rejected 2026-11-10,
  estimated "about 20 agent-days", "high risk of quiet errors in capacity and demand for weeks"; save +10 KB raw (shared
  alignment + group id) or +75 KB raw (alignment per track) on NYC; junction nodes and moves 2-4x, crossing pairs 4x
  (notes/T-3137.md 87-111). Anita's new notes ("treat double track as a pair of single tracks ... saved joined") are
  close to this, with a shared saved alignment.
- T-3137 (c) lanes inside the piece: built (C1-C4), 7-8 agent-days estimated.
- T-3348 (b) "a first-class four-lane piece" (`edge<<2|lane<<1|dir`): rejected, "about 40-60 h", and "points the wrong
  way for her station editor" (notes/T-3348.md 129-135). T-3348 (a), recommended: 27-39 h, stage 1 built.

### Keep (sound, and other code leans on them)

- **Alignment geometry**: PIs with arcs, `geom.rs` (914), ramps and levels (`geom::vertical`, `Ramps`, `LRA1`), curve and
  ramp speed limits (`geom::limits`), `express_verts` as the offset-curve maker, `trace.rs`, `cross.rs` (crossings on a
  1 km grid), `bridge.rs`, `roads.rs`, `frame.rs` (Mercator, ground metres). A parallel track is an offset of one
  alignment; that maths stays whatever owns the offset.
- **Costs** `cost.rs` (1,098) and `net/costs.rs`: price by level, water, terrain, kind and track count.
- **Run-time profiles** `profile.rs` (974), keyframes per `(edge, dir)` in `service.rs`, `timing.rs`, `spacing.rs`.
- **Capacity model** (6.2) `capacity.rs` (1,543): resources keyed by directed track `key_of` (`edge*2+dir` or `edge*2`),
  platform tracks keyed `(node, side_in)` (`capacity.rs:588`), junction moves from the move table, turn places,
  following losses and passing waits (`overtake.rs`). The formulas stay; the keys change if tracks change.
- **Demand's view**: station points (merged within `station_merge_m` 100 m, `demand/api.rs:195`, `demand/kernel.rs:401`),
  lines as ordered stops with hop minutes and headways, complexes `st_group`, `stop_x/y` (platform positions), and
  `back_stops` (`demand/network.rs:23-80`); walking changes up to `xfer_walk_m` 300 m (`demand/kernel.rs:397`).
  Demand never sees track; anything the rework does reaches it only through these arrays.
- **Services and boarding**: `service.rs` (2,053), `boarding.rs` (1,723), `span.rs`, the hidden-twin trick is replaceable
  but the services themselves are per line and run.
- **Undo** by record replacement (`Op`, `world.rs:20-44`): keep the mechanism, add record kinds.

### Replace (what the notes ask to change)

- `EdgeData.tracks`, `join`, inferred ways (`derive_ways`, `run_group`, `group_ways`, `pin_station_runs`), `lanes.rs`'s
  two-lane cap and `Moves`' port-centre slots: the node model.
- Quad by geometry: `twin_at`, `quad_beside`, `station_beside`, `twin.rs` and the hidden twin, `quadPairs.ts`,
  doubling up's `Alongside`.
- Stations as nodes: `StationData` on the node, `one`, `platform_reach`, `check_station` reach rules,
  `upgrades::PASSING`/`TERMINAL` as stand-ins for track, `perdir.rs` way stations, `back_lines`.
- App drawing that stands in for missing model: `legShifts`, taper, `singleSplitEnds`, `passingTracks`, `turnbackStrokes`,
  `singleLeadStrokes`, `corridors.ts`/`besideSamples` for finding parallels (the ribbon layout itself can stay), the
  station box at a fixed width, `stationSide.ts`.

### The seams (where new and old meet)

1. **The path entry `edge<<1|dir`, and the return run as the path reversed**: lines (`LineData::path`), saves, service
   keyframes, capacity keys, boarding, the app's line strokes and train samples, demand's route nodes through
   `packService`. A line stores only its forward path; its return run is that path backwards, which is what lets
   `lines_on_edge` (the edge-to-lines index) and every "which lines does this edit touch" test stay local. The
   exception is a line whose way back differs (`back_run_of`, one-way track, per-direction stops): those are listed by
   `Network::back_lines` (`net/paths.rs:149-166`) and rebuilt on every edit, because their way back is not on
   `lines_on_edge`. **My reading**: if double track becomes two one-way single tracks as edges, every double-track line's
   return run is on other edges, so the exception becomes every line unless the return path is stored (or indexed) too.
   If tracks stay lanes inside a piece holding N tracks, the entry needs a lane (T-3348 (b)'s cost). This is the
   constraint that most separates the designs.
2. **Demand's network arrays** (above): stations as points, stops, hop times, `back_stops`, `st_group`.
3. **The snapshot to the app** (`wasm/snapshot.rs` 283, `workers/stateRows.ts` 635, `ClockEdge`/`ClockNode` rows with
   `tracks`, `join`, `oneway`) and `edgeParts` polylines: the drawn network's input.
4. **Capacity's keys**: `key_of`, platform `(node, side)`, `turn_places`, `passing_at`.
5. **The Op set and the save blocks**.

---------------------------------------------------------------------------------------------------------------------

## 7. Performance facts to beat (with task, date, save and machine)

| What | Figure | Source |
|---|---|---|
| Station placed on built track, click to result | 1.9-3.7 s; worker buffers 2.1 s | T-2763, 2026-11-07, real NYC save, Chrome on 2 cores |
| Same, worker `buffers` only | 200-900 ms, of it `lineBands` 130-750, `besideSamples` 60-350, 8-15 lines redone 50-380 | T-2768, 2026-11-17, real NYC save, Chrome on 4 cores, machine busy |
| Same, as "shows up" on the big save (current: the note says T-3803 left this kind unchanged) | 1,925 ms median (2,732 ms with edits 1 s apart), n=3-6 | T-3803, 2026-11-19/20, 2,647-station save (221 lines), 9-13 of 14 cores taken by others |
| ui-building.md's statement | "0.3 to 3 s on the real save" | ui-building.md 690-692 (T-2763) |
| Deleting a built station | 650-2,200 ms -> about 0.5 s less after T-2768 (109 -> 14 lines redone) | T-2768, 2026-11-17 |
| T-2767's old title | "2-4 s" (stale; review says 0.2-0.9 s) | review-edit-speed-2026-11-20 |
| One full state on the big save | network 301 ms median, rows 94, netForPost 29, post 22 (60/20/5/5%) | T-3803, after option 3 |
| networkFromState own time in a profiled run | 1.2 s of 3.8 s; `legShifts` 0.7 s, `runSamples` 0.3, `buildingsDrawn` 0.29, `lineBands` 0.1 | T-3803 notes 201-204 |
| Follow-up at rest, North America | worker 0.4-1.1 s (`networkFromState` 150-740 ms, `buildState` 30-110) + 35-100 ms main thread | T-2404, 2026-11-06 |
| Click freeze, NYC / North America | 271-322 -> 35-42 ms p50 / 492-595 -> 37-45 ms p50 (follow-up deferred to rest) | T-2404, 2026-11-06, cores 12-15 pinned, Anita playing |
| A placement that cut a line's edge | 1,723 ms -> 369-446 ms (`lineBands` whole pass 700 ms, 90 lines' strokes 750 ms) | T-2238, 2026-11-03, real save |
| `lineBands` per rebuild | Tokyo 110 ms, real NYC 260 ms, all Japan 190 ms; first build adds 280 / 500 / ~900 ms | T-1799/T-1848, 2026-10-29/30 |
| Ribbon order | 8-10 ms a build on Japan, 7-8 ms NYC | T-1799 |
| `preview_route` own time | 2-3 ms in WASM on all Japan; from-node previews 2.5 -> 1.6 ms | T-2700, 2026-11-07 |
| Ghost after branching | 2.5 s mean, 4.9 s max -> 0 (trials skip services) | T-1779, 2026-10-29 |
| Long-piece ghost | 100-300 ms (stale: before T-1779 and T-2700, never re-measured) | T-1040, closed 2026-11-20 |
| Penn Station to Emeryville snapped preview | 3 s (stale, from 2026-10-19; kept as a check case for the rework) | T-763 |
| NY-Emeryville quad edit | 3.3 s -> 0.15 s natively (5,688 corners traced in 0.8 s) | T-727f / T-727c, 2026-10-19 |
| Track derive on all Japan | 4.9 s (stale: from T-737, 10-19; T-1039 and T-3234 since) | T-758, closed 2026-11-20 |
| Mask re-derive | 100-230 ms a mask -> 7 masks in 7.5 ms (box only) | T-1039, 2026-10-23 |
| Incremental capacity | 0.2-1.2 ms per edit at 10,000 km (was 13); whole edits 1.5-2.4 ms; busiest edge 19 ms | T-050, 2026-10-09, synthetic |
| Next edit's wait after an edit | 612/1,270 -> 158/712 ms median/p90 | T-3803 option 1, 2026-11-20 |
| Rename's own follow-up | 610-643 -> 107-111 ms | T-3803 |

Requirements Anita accepted on 11-20 (TASKS.md:364): a station placed on track that leaves its shape alone redraws no
neighbours (T-2767); the drawn network rebuilds only near a change (T-2701); the Penn Station to Emeryville snapped preview
as a check case (T-763).

Whole-network passes still on every edit (non-local costs to design out): `networkFromState`'s walk over all edges, lines,
samples, stations (T-2404); `Network::back_lines` scanning every edge and line (`net/paths.rs:149-166`) and rebuilding
every line with a derived return run; `twin_at` scanning every edge per station placement (`draw.rs:1310`);
`complex_of` scanning every station per call (`twin.rs:174`), called per stop by `station_beside` and `way_group`;
`way_group` scanning every node (`perdir.rs:55`); `find_path_side` allocating 2E per hop (`net/paths.rs:200-202`).

---------------------------------------------------------------------------------------------------------------------

## 8. Sizes (lines in git-tracked files, 2026-11-21; `sizes.py`)

`sim/src/track` all: 61,812 lines in 129 files, 22,590 of them in test files (code-tour.md says about 54,000 on 11-18; the
gap is not explained). Track-kind tests (double, quad, quad_edit, onetrack, twin, perdir, lanes, oneleg, oneway, loop_way,
beside, flyover, stop_pass, stop_edit, stop_order, lane_preview, branch, upgrade, platform, station, stroke): 6,822 lines.
Track-kind browser checks: 24 files, 4,229 lines.

| Group | Lines |
|---|---|
| sim core model (net.rs + net/, lanes, perdir, twin, upgrades, building) | 6,176 |
| sim edit and draw API (wasm.rs, wasm/draw, beside, flyover, crossings, links, rebuild, edits, onetrack, oneway, twin, legs, lines, upgrades, snapshot, flip, gap, buildings, world.rs) | 13,286 |
| sim save | 1,801 |
| sim consumers to adapt (service, capacity, overtake, timing, boarding, span, spacing, related, wasm/service, wasm/money) | 9,114 |
| sim to keep (geom, cost, profile, cross, bridge, roads, trace, frame, params, streets, bus, ferry) | 7,483 |
| app drawn network (network, lineBands, corridors, terminalBays, stationGeometry, stationSide, facilityPlace, oneWayMarks, buildingNet) | 6,403 |
| app tools (tools, stationTool, stationSnap, toolTouch/Delete/Marks/Parts, blueprintBox, stopMarkers, edgeEdit, nodeDrag, dragHandles, realSnap, pick, blueprintPick, rebuild, remove, complexes) | 6,398 |
| app renderer and trains (renderer, rendererDraw/Strokes/Shaders/Parts, strokeCache, trainGuard, trackHover, networkTips) | 5,867 |
| app panels (TrackInspector, StationInspector, StationSides, StationUpgrades, ToolBar, Blueprint, widgets) | 2,022 |
| worker glue (clock.worker, clockClient, clockEdits, clockState, clockPost, stateRows, platformLines, saveFile) | 6,216 |
| **total** | **64,766** |

Biggest single files: `wasm.rs` 2,411; `network.ts` 2,295; `world.rs` 2,244; `service.rs` 2,053; `save.rs` 1,801;
`boarding.rs` 1,723; `wasm/draw.rs` 1,696; `clockClient.ts` 1,671; `lineBands.ts` 1,582; `capacity.rs` 1,543;
`wasm/lines.rs` 1,383; `tools.ts` 1,359; `rendererShaders.ts` 1,307; `clock.worker.ts` 1,296; `wasm/rebuild.rs` 1,217.

Counts of track-count checks: `edge_tracks[` in non-test Rust, 39 uses in 18 files; `tracks === 1/2` and the like in the
app, 50 in 19 files (network.ts 19, terminalBays 5).

---------------------------------------------------------------------------------------------------------------------

## Unsure

- The bug classes are my reading of one-line archive entries, not of each task's notes; counts are about +-20%.
- Not opened: ui-building.md "Drawing track" (228-379) and the body of "The Select tool" (801-958); the bodies of
  `lineBands.ts`, most of `perdir.rs`, `wasm/rebuild.rs`, `add_crossover`. The 8 m port pitch (`PORT_PITCH_M`) and 2 m lane
  offset (`LANE_HALF_M`) are quoted from docs and comments, not read from params.rs.
- `sim/src/track` is 61,812 lines by my count against code-tour.md's "about 54,000" (dated 11-18); I can't explain the
  gap (both say tests included).
- The return-run reading in seam 1 is mine, not tested.
- Whether docs 6.1's "no passing loop" for single-track stations is stale or means the PASSING upgrade (code: a passing
  point with one shared platform track).
- `legShifts` at 0.7 s in T-3803's profile: one profiled run on a loaded machine. Checked: it reads `ClockEdge.join`
  from the state rows, then works out each end's direction and the tracks to keep clear of from the polylines, building
  a per-node table over every edge on every build (`network.ts:615-700`), which is why it is whole-network.
- Line numbers drift: others had uncommitted edits in the tree (git status at start).
