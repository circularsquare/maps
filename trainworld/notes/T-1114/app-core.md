# T-1114 review: app/src outside ui/

Read-only review of `app/src/game/`, `map/`, `workers/`, `main.ts` and `perfFlags.ts`, 2026-10-24.
`wasm/` is generated and left out. Every "unused" claim was checked with a search over `app/src`
(ui/ included) and `checks/`. Line numbers are as of commit 54005ec plus the working tree. Several
files are moving under other agents right now (`main.ts`, `game/world.ts`, `map/zoneFocus.ts`,
`game/fareCurve.ts`, `game/days.ts`), so re-find a line by its text before editing it.

## Summary

- **56 cleanup items and 9 bugs**, in 11 batches. Batches 1 to 8 are cleanups an agent can do in one
  sitting each. Batch 9 (perf switches) needs a yes from the orchestrator. Batches 10 and 11 are
  file splits.
- **Lines that could go:** about 850 to 950 net, roughly:
  - demand client and workers: 220
  - renderer and map views: 200
  - build tools: 150
  - clock worker: 110
  - tags and labels: 90
  - game/: 100
  - perf switches: 40

  The splits in batches 10 and 11 move about 3,000 lines without cutting them (renderer shaders
  about 850, `demandClient.ts` about 1,000, `clock.worker.ts` about 600, the station tool about 500).
- **The five best:**
  1. Bug B1: each region's running cost is charged at ordinary rates for high-speed lines, so the
     region ledgers and the T-835 tail are wrong for every line with fast trains
     (`clock.worker.ts:819-821`).
  2. Bug B2: the demand client keeps the full results of the last 50 solves on the main thread. That
     is a few hundred MB after 50 edits on a big network (`demandClient.ts:1966-1968`).
  3. Dead GPU work. A per-stroke buffer is uploaded on every network change for a shader input
     nobody reads, and one uniform is set every frame for nothing. The station shapes also carry an
     arc curvature that is always 0. About 35 lines go (item 1).
  4. One small WebGL helper module. Shader compiling is copied 4 times, zoom interpolation 3 times,
     hex to float colours about 20 times (some parsed on every frame), the GPU timer twice and
     canvas sizing 4 times (item 2).
  5. Geometry constants and helpers. The Earth radius is defined 5 times, metres per degree about
     20 times, "pixels per local unit" 9 times, and point-in-ring 7 times. The two tests against the
     coarse zone outlines differ: realms uses a slow full scan, trackZones a banded one (items 33 to 35).

Comment slop is not a sweep of its own. CLAUDE.md says "trim a function's comments when you edit it;
no sweeping rewrites", so each batch lists the history and stale comments in its own files, and
they go as those files are touched.

Checks reach into private state. Renaming any of these breaks a check:
- `window.tw.tools` fields: `ghost`, `stationSpot`, `alts`, `lastPreview`, `tag`, `real.spot`,
  `real.last`, `real.covers`, `real.nearestDrawn`, `real.ring` (t671, t727, t1020)
- `renderer.gpuMs` (t714, t658)
- `drawnShapeOf`
- `stationSize[1]` and `stationSize[3]` (t144-check.mjs:227-229)
- `CapacityMarkers`, which t133 builds without `labels`
- `isNortheastSave` (t116)

Grep `checks/` for any name you rename.

---

## Bugs (batch 0: fix first, each on its own)

**B1. High-speed lines' running cost per region uses the ordinary rates.**
`workers/clock.worker.ts:814-821` (`regionLines`) prices every running line at
`moneyParams[3] + moneyParams[1] * cars`, the ordinary train-km and car-km rates.
`TrackApi.running_cost` (`sim/src/track/wasm.rs:546-559`) uses `HS_RUN_COST_TRAIN_KM` (5) and
`HS_RUN_COST_CAR_KM` (3) for fast lines. The effects:
- The region ledgers (Cities, the Money tab by region) are wrong for lines with fast trains.
- The regions no longer add up to the network's ledger.
- The T-835 tail (`runTail` through `localGroups`, in `settle` and `unsettleAfter`) is charged at
  the wrong rate.

Fix: pick the high-speed parameters when `a.line_fast(id)`. This goes well with item 20, which
names the `moneyParams` indices. Risk: low (money figures change for fast lines, toward what the
network ledger already says). Effort: small.
*Fixed* (appbugs): `regionLines` takes `moneyParams[6]`/`[5]` for a fast line. A new game with a
6 km high-speed line and an ordinary one in New York (10 cars, 6/4/3 trains): the region's day
24.31 -> 29.88 US$M, the network's 29.88 both times (no period here is followed by no service, so
the T-835 tail did not move it).

**B2. The last 50 solves keep their full results on the main thread.**
`workers/demandClient.ts:1966-1968` pushes each solve onto `this.log` (kept at 50). Each solve keeps
`results`, the last round of `byRound` (`:1906` frees only earlier rounds) and `clusterRes`: every
period's segment, boarding, alighting and crush loads, plus the purpose parts. Nothing reads an old
solve's results. The views copy what they need, and checks t124 and t326 read only
`log.at(-1).results`. At about 5,000 stops that is roughly 6 MB a solve. Fix: when a newer solve
publishes its final, set `results = []` and `byRound = []` and drop `clusterRes` on older entries.
Keep the timing fields the debug readout uses. Risk: low. Effort: small.
*Fixed* (appbugs): `publish`'s final empties every older log entry's `results`, `byRound`, `clusterRes`.
Real save, 20 train-count edits each solved to its final (CDP heap after GC): the log's 22 solves
held 39.4 MB of typed arrays before, 1.9 MB after (the current solve's); array buffers 149.4 ->
111.9 MB, JS heap 122.7 -> 120.3 MB. About 1.9 MB a solve, so up to ~95 MB at the 50 cap before.

**B3. The `otherCells` cache never evicts.** `demandClient.ts:2091` caches each nearby zone's cells
(xy, H3 as BigUint64, zone) per frame epoch. It drops an entry only on failure, so panning across
many zones grows it without bound. Fix: drop the entry when the zone closes
(`closeFull`/`closeBare`) or leaves `nearbyZones`. Risk: low. Effort: small.
*Fixed* (appbugs): `dropCells` on `closeBare`, `closeFull` and on a new frame epoch's miss.
A 16-place pan from Washington to Boston with Homes on (real save) left 4 entries, 1.6 MB, before
and after: every one a zone still open (three the save reaches, one bare spare), so nothing closed
to drop; the cache is now bounded by the open zones.

**B4. A preview that crosses an anchor change keeps its marks at the old anchor.**
`workers/clockClient.ts:608-610` moves the preview's `verts`, `pts` and `arc` to the current
anchor, but not `marks` (local units, stride 3). `crossoverCheck`'s `parts` (`:626-628`) is not
moved either. The crossing and junction marks of that one preview draw offset. Fix: pass `marks`
(and `parts`) through `toCurrentAnchor`. Risk: none. Effort: small.
*Fixed* (appbugs): `marks` moved; `crossoverCheck` now carries `anchor` and its `parts` are moved.
(A preview's `parts` are indices, not positions.)

**B5. Backspace in the track tool leaves the real-network snap starting from the undone point.**
`map/tools.ts:462-476`: `takeBackSegment` puts `start` back but never calls `real.begin` again, so
the snap's start stays at the end of the undone segment (set at `:452`). `requestPreview` (`:491`)
also does nothing while `real.active`, so no ordinary preview shows either. Separately,
`:463-466`: Backspace while a segment's edit is in flight pops the chain and ends the run instead
of undoing one segment. This was found by reading the code; check it in the game. Fix: keep the
ring point in `Placed` and call `begin` with it. Risk: low. Effort: small.
*Fixed* (appbugs): `Placed.ring`, `begin(prev.ring)` on Backspace; Backspace while a segment is
being placed is ignored (it is not on the undo stack yet).

**B6. `probe()` leaks a GPU buffer per call.** `map/renderer.ts:2393` adds a buffer to `this.res`
on each call, and the buffer is freed only at the next `setNetwork`. Only debug and check code calls
it (`checkPrecision`, t-checks in a loop). Fix: delete the buffer after the readback.
Risk: none. Effort: small.
*Fixed* (appbugs): made outside `this.res`, deleted after the readback.

**B7. A train-load setting is stuck if an old browser stored it.** `game/ui.ts:47,107`
(`trainLoadBasis`): the comment says it is "no longer a setting (T-164)". Nothing in the UI sets it,
but the stored display settings are spread over the defaults (`:170`). A browser that stored
`"average"` keeps it for good, and nothing can change it back. `map/demandLayers.ts:93-106` still
honours it. Fix: delete the field and the `basis` parameter, and always use the busiest hour
(`demandViews.ts:85,88,114`, `demandLayers.ts:25,93,106,136`). Risk: low (for Anita, if her
browser holds "average", line widths would change to the default). Effort: small. About 8 lines.
*Fixed* (appbugs): field and parameter gone, dropped from storage on load. (It set the train
gauges' busiest-hour factor, not line widths.)

**B8 (minor). Line widths can keep a stale day scale.** `map/demandViews.ts:86-89`: `on.days`
feeds the busiest-day width scale, but neither the change key `now` nor the effect follows
`demandByDay`. When another day's view lands while the shown day's object is unchanged, the widths
keep the old scale until something else changes. This is plausible from the code but was not seen
in the game. Risk: low. Effort: small.
*Not a bug* (appbugs): every writer of `demandByDay` (`publish`, `publishNewLines`,
`publishCluster`, the cache load) replaces every day's view at once, and `addBareViews` keeps
every view object; the shown day's view always changes with the others, so `demandView` reruns.

**B9 (minor). Two tag leftovers.**
- `map/reversalMarks.ts:68-71`: turning a tool on hides the marks but not an open tip, unlike
  `capacityMarkers.ts:65`.
- `map/realNetwork.ts:129`: `ensure()` waits on `map.isStyleLoaded()`, which waits on every tile
  (as `:160` itself says). After a pan during tile loading, each `moveend` adds another
  `once("idle")` listener, and new regions' layers appear only at the next idle.

Risk: none. Effort: small.
*Fixed* (appbugs): the tip hides when a tool is picked; `ensure` waits on `getStyle()` and
`style.load`, as `showReach` does.

---

## Batch 1: WebGL helpers and dead GPU inputs

Owns: `map/renderer.ts`, `map/demandOverlay.ts`, `map/gradeMarks.ts`, `map/shadeLayer.ts`,
`map/stationGeometry.ts`, `map/overlay.ts`, `map/tripView.ts`, a new `map/gl.ts`,
`game/palette.ts` (adding `rgbf` only), and `checks/t144-check.mjs` (index change only).
Do this before batch 10, which moves the shaders.

**1. Remove dead GPU inputs.** (rank 3)
- `renderer.ts:144,171,1406,1778`: the stroke shader declares `u_px_per_unit` and `a_dist` but
  never reads them. Line 1406 uploads a buffer of `s.dist` for every stroke set on each
  `setNetwork` (about 0.5 MB for New York's track and lines), and line 1778 sets the uniform every
  frame. Drop all four. Keep `Strokes.dist`, because `network.ts` `turnbackStrokes` reads it. Its
  doc "(dashes)" is stale.
- `renderer.ts:359,361,985-989`: `a_sz.x` and `a_sel.y` are never read. `plainStationSize` fills
  only `x`, `demandLayers.ts:186` leaves it 0, and `NetworkLayers.stationSize`'s doc ("the widest
  line through it") is wrong. Make both single floats, and replace `plainStationSize` with zeros.
  Update `checks/t144-check.mjs:227-229`, which reads `stationSize[1]` and `[3]`.
- `stationGeometry.ts:285,297-301,365,385-388,505,544-553,597-600`: `StationShape.k` is always 0.
  The only places a StationShape is made are `stationShape` (`:365`) and `bridgeShape` (`:505`).
  That makes these dead: the arc branch of `stationCoords`, the arc sampling in `shapePoints`, the
  arc path in `stationPath`, `CUT_FS`'s `u_size.z` (marked "(unused)", filled at
  `demandOverlay.ts:639`), and `stationSdf` in `SHAPE_GLSL`, which no shader calls. Keep `arc`,
  because `TRAIN_FS` uses it.
- `renderer.ts:1105,2333`: `crowdMs` is written and never read.

Risk: low (only the bridge outline changes code path, and it was already straight). Effort: small.
Mechanical: yes. About 35 lines.

**2. Add one `map/gl.ts`.** (rank 4)
- `compile(gl, vs, fs, feedback?)`. Copies: `renderer.ts:950-963` (with transform feedback),
  `demandOverlay.ts:223-234`, `gradeMarks.ts:148-160`, `shadeLayer.ts:69-80`.
- `uniforms(gl, prog, names)`: `renderer.ts:965-969`, and the loops at `gradeMarks.ts:184`,
  `demandOverlay.ts:332,476` and `shadeLayer.ts:82`.
- `interp(z, stops)`. Copies: `renderer.ts:971-982`, `gradeMarks.ts:36-46`, and
  `stationGeometry.ts:220` `interpolate`, which gives the same results.
- A `GpuTimer` (begin, poll, ms). Copies: `renderer.ts:1749-1754,2137,2413-2422` and
  `demandOverlay.ts:557-562,600,646-652`. Keep the `renderer.gpuMs` field, which checks read.
- Canvas sizing to the map's pixel ratio. Copies: `overlay.ts:138-143`, `demandOverlay.ts:350-355`,
  `tripView.ts:66-72` and `faces.ts:161`.
- Clip-to-CSS projection. Copies: `renderer.ts:2143-2147` `projector`, and `overlay.ts:225-226`
  and `237-239`.

Risk: low (the error text prefixes change; nothing matches on them). Effort: small. Mechanical: yes.
About 60 lines.

**3. Add `rgbf(hex)` to `game/palette.ts`, cached per hex.** About 20 copies of
`rgb(x).map((v) => v / 255)`:
- `renderer.ts:28-34,81-82,110,1295-1296`
- per frame: `renderer.ts:1786-1806` and `2120-2121`
- `demandOverlay.ts:65-66` and `576` (per frame)
- `gradeMarks.ts:168` (`rgbf`, called four times a frame at `:243-246`)
- `shadeLayer.ts:61` (its own parse)
- `palette.ts:179` `inkOn`

The per-frame ones parse hex strings on every draw. A small `Map<string, Float32Array>` cache makes
them free. `gradeMarks.ts:246`'s `"#d23c3c"` is also `renderer.ts:29` `BAD_RGB`. Risk: none.
Effort: small. Mechanical: yes. About 20 lines.

**4. Small renderer simplifications.**
- `renderer.ts:1949-1959` and `1966-1979` set the same eight bridge-program uniforms, for bridges
  and for connectors. Make one helper taking the half width.
- `bridgeProg` and `selProg` are typed `| null`, but `init` always sets them, which is why the `!`
  at `:1557-1559` is needed.
- Building a `PlatformShape` from the buffers is copied at `renderer.ts:1179-1180,1244-1245,1531`
  and at `network.ts:617` (`shapeAt`). Export one `platformShapeAt(n, i)` from `network.ts`.
- The `stationSizes` field could be a local.

Risk: low. Effort: small. Mechanical: yes. About 20 lines.

**5. Cheap per-frame fixes.**
- `shadeLayer.ts:117` allocates a `Float32Array(16)` on every MapLibre frame; keep one as a field.
- `tripView.ts:112` calls `pts.slice(1)` twice per part per frame.
- `renderer.draw` builds 3 projector closures and runs filter/join over `selStations` and
  `hotStations` every frame.
- `main.ts:~146-167`: the overlay's options closure runs `n.stationIds.indexOf(...)` for each
  selected, hot and picked station every frame. With a station-index map on `NetworkBuffers`, or
  one built in `setNetwork`, each lookup is O(1).

Risk: low. Effort: small. Mechanical: yes.

History and stale comments in these files go as they are touched:
- `renderer.ts:36-38, 41-46, 197, 359, 361, 1762-1763, 2088`
- `stationGeometry.ts:19-22, 27-31, 195-196, 202-204`
- `StationStyle.linePx` (`stationGeometry.ts:234`) says it is 0 when lines are not drawn, but
  `renderer.ts:1900` always sets it
- `plainStationSize`'s doc
- packed lines `renderer.ts:1524` and `1536`

## Batch 2: demand client and workers, small helpers and dead code

Owns: `workers/demandClient.ts`, `demand.worker.ts`, `demandProtocol.ts`, `demandClusters.ts`,
`demandCache.ts`, `cityReach.ts`, `commutersProtocol.ts`, `selectionMerge.ts`, and `game/trip.ts`
(the market layout constants only). Fixes B2 and B3 too, if batch 0 has not.

**6. `cityReach.ts` `reaches` (`:92-112`) can call `distanceTo`.** The early-out on the box plus
`distanceTo(f, lon, lat) <= REACH_M` is the same test. `citiesReached` runs on the main thread for
every new service (every edit), walking each station against whole outlines, and `distanceTo`
already has the T-889 edge grid. The other two copies of the segment-distance loop,
`gridDistance` (`:211-215`) and `walkDistance` (`:244-248`), then share one
`nearestOnEdge(ax, ay, bx, by)`.

There is a second cheap win. `reach` and `nearZones` (`demandClient.ts:609-612`) are WeakMaps keyed
on the `Service` object, and `serviceFromSnapshot` makes a new one each snapshot, so a
timetable-only edit redoes the reach. Key them on a string of the station positions instead.

Risk: low (rounding exactly at 4.5 km). Effort: small. Mechanical: mostly. About 20 lines.

**7. Delete `Cluster.nodeLocal`.** `demandClusters.ts:53,340,348,371` write it and nothing reads
it: the worker builds its own in `demand.worker.ts:105` `nodeLocal()`. It costs an
`Int32Array(2 x stops)` per cluster per solve in every worker. Risk: none. Effort: small.
Mechanical: yes. 4 lines.

**8. Join the four mode-query routines in `demandClient.ts`.** `cellModes`, `purposeModes` and
`bareModes` repeat:
- the six-array `CellModes` build (`:2137-2138,2164-2165,2197-2199,2223-2225`) → `modesOf(data,
  version, day)`
- the "still shown" test (`:2135,2192,2220`)
- the plain `Target` literal (`:2106,2159,2376,2439`) → `shownTarget(src?)`
- the element-wise add (`:2124,2131,2183,2186,2218,2468,2514`) → `addInto(dst, src)`

Risk: low. Effort: small. Mechanical: yes. About 30 lines.

**9. Two transfers queries in one.** `demandClient.ts:2306-2321` (`transfersAt`) and `2341-2353`
(`platformTransfersAt`) share the ask, `byMask`, the all-periods test and the "still" check. Make
one `transferTriples(t, nodes, each)`. Line `2337-2339` becomes
`Uint32Array.from({ length }, (_, i) => i)`. Risk: low. Effort: small. Mechanical: yes.
About 14 lines.

**10. `demandClusters.ts` helpers.**
- `reachView` (`:264-277`) and `subNetwork` (`:321-335`) build the same local station arrays →
  `stationsOf(n, stations)` returning `{ stLocal, stXY, complex }`.
- `lineOff` is built twice (`:100-101`, `249-250`) → `lineOffsets(n)`.
- `splitClusters`' city-box test (`:145-148`) repeats `cityReach` (`:240-243`).

Risk: low (cluster signatures unchanged; t326 and t756 compare the figures). Effort: small.
Mechanical: yes. About 22 lines.

**11. One demand day helper.** Reweighting by purpose appears in `bareView` (`demandClient.ts:2799-2808`)
and `dayResult` (`2866-2873`). Trips, rail, walk, share by period and split are worked out three
times (`2797-2814`, `2893-2901` with `2936-2942`, and `2954-2960`). Make `reweighted(...)` and
`figuresOf(rows)`. Risk: low. Effort: small. About 25 lines.

**12. `demand.worker.ts` repeats.**
- `tripNodesGlobal` and `marketTripNodesGlobal` (`:866-891`) are one routine with two layouts. Add
  `MARKET_TRIP_LAYOUT` beside `TRIP_LAYOUT` in `demandProtocol.ts`, and use it in the worker
  (`:657`, `:881`) and in `game/trip.ts:281` (`MT_HEAD`, `MT_PAIR_LEN`, its own copy now).
- The sums keep-and-send block is copied at `:1186-1190` and `1237-1244` → `keepSums`.
- The city figures are copied between `solveBare` (`:269-275`) and `figures` (`:1139,1147-1153`).
- The period mask `mine.reduce(...)` is recomputed at `:595,662,1063` (set it once at init), and
  the literal `31` at `:601,662` is the all-periods mask.

Risk: low (t325 compares bare against solved). Effort: small. About 30 lines.

**13. `demandClient.ts` one-liners.**
- "End the load in hand" is written out 5 times (`:762-765,1421,1477-1480,1735-1738,1774-1776`) →
  `endJob()`.
- The stop count is written 4 times (`:1394,1463,1468,1514`).
- `this.status.startsWith("failed")` appears 12 times → `get failed()`.
- `lastLoad` plus a stamp is written 4 times (`:961,980,1274,1285`).
- `routeNodes(layout)` is the same at `:2383-2385` and `2418-2420`; the `bareSquares` fetch at
  `:2413-2417` and `2568-2572`; the member indices at `:2236` and `2619`.
- `:2636` rebuilds `intercityZones("icStation", ...)` by hand.
- The query-kind test at `:1785` repeats the `QueryKind` list at `:153`. Derive it as
  `Extract<FromDemand, { id: number }>["kind"]`, and declare `qStart?`/`qMs?` in the protocol so
  the cast at `:1788` goes.

Risk: none. Effort: small. Mechanical: yes. About 35 lines.

**14. Dead in the demand files.**
- `FromDemand` `ready.info` (`demandProtocol.ts:183-185`, worker `:418`): never read.
- The constructor's `_city` (`demandClient.ts:640`) and `startDemand`'s `city` parameter (`:3027`)
  are unused. Removing them also changes `main.ts:117` and `:468`.
- `marketXY` (`:166`) is an alias of `marketToWorld`.
- `EVERYDAY_PURPOSES` (worker `:41`) is an alias of `EVERYDAY_NAMES`.
- The `"set_fare_curve" in p` guard (worker `:1057-1058`): every generated API has the method.
- Exports used only in their own file: `periodsOfWorker`, `poolSize`, `CITY_REACH_M`,
  `MARKET_REACH_M`, `JOIN_M`, `footprint`.

Risk: none. Effort: small. About 12 lines.

Comments to fix in these files as they are touched:
- stale: `demand.worker.ts:1-3` ("owns New York local commutes, Philadelphia..."), and
  `commutersProtocol.ts:1` (names `demandViews.ts`; the user is `commuterView.ts`)
- wrong: `commutersProtocol.ts:15-22` says 25,000/80,000 a km², but the code has 50,000/160,000
- stacked JSDoc blocks where only the last attaches: worker `:94-102`, protocol `:117-118`,
  client `:616-618`, and the orphaned `poolSize` doc at client `:109-116`
- history: client `:1628-1629, 1664, 2193-2195`, `demandCache.ts:25`, `selectionMerge.ts:4-7`
- naming: `cacheWrites_` (`:564`) beside the public `cacheWrites` reads as a typo; rename it to
  `cacheWriteWaits`. `LEVEL` is defined twice (client `:125`, clusters `:13`).

## Batch 3: clock worker, client and protocol

Owns: `workers/clock.worker.ts`, `clockClient.ts`, `protocol.ts`, `saveFile.ts`,
`realRoute.worker.ts`, and `game/types.ts` (removing the WorldView `city` and old `Save` fields
only). Fixes B1 and B4 too.

**15. Remove casts the WASM declarations made unneeded.** `clock.worker.ts:2249-2252,2283` cast
`api` to add `preview_render_speeds` and `preview_marks`. `sim.d.ts:503,519` declare both, so the
casts and the comment at `:2249` ("...before the track WASM declarations regenerate") are stale.
Risk: none. Effort: small. Mechanical: yes. 3 lines.

**16. Remove the WorldView `city` stub and the unread `Save` fields.**
- `clockClient.ts:104,164,168,234`: `private city = { population: 0, jobs: 0 }` is never assigned.
- `WorldView.city` (a `CityStats` stub, "New York", drive 100%) is read nowhere in app/src or
  checks. The same goes for `save.version`, `save.city` and `save.clock`.
- Remove them with `CityStats` and the three `Save` fields (`game/types.ts:108-112,185-192,225`).
- The init message's `city` field (`protocol.ts:106`, sent at `clockClient.ts:313`) is never read
  by the worker. `let city = HOME_CITY` (`clock.worker.ts:46`) never changes, and `:1897-1898`
  assigns it the same value again. Use `HOME_CITY`.

Risk: low (typecheck catches any reader). Effort: small. Mechanical: yes. About 16 lines.

**17. One route request shape.** The route request's fields are written four times:
`protocol.ts:49` (the edit op), `protocol.ts:114` (the preview), `clock.worker.ts:1994`
(`RouteAsk`) and `clockClient.ts:656`. The worker also repeats its setup three times
(`set_route_tangent`, `set_design_speed(op.design ?? 160)`, `set_route_hsr`, and the
ends/PIs/track-code arguments at `:1748-1751,2046,2227`). Export `RouteAsk` from the protocol, and
add `routeArgs(api, r)` in the worker. Then the default of 160 and the track code (`single ? 1 :
quad ? 4 : 2`) live in one place, and `routeKey` (`:1995`) agrees with them. Risk: low (a missed
flag would make an edit differ from its preview). Effort: small. About 12 lines.

**18. Opening a slot gunzips the save twice.** `clock.worker.ts:1403-1406` decodes the save and
checks `isNortheastSave` before `loadGame`, which decodes it and makes the same check
(`:1451-1456`) before it changes anything. Drop the first. Risk: low (the error text is the same).
Effort: small. 2 lines.

**19. Ledger entry helpers.** Empty day and hour entries are spelled out 7 times
(`clock.worker.ts:784,789,797,798,807,919,920`) and once more at `game/regionMoney.ts:141`. The
field-by-field add appears at `:873` and `914`. Put `newDay(d)`, `newHour(h)` and
`addEntry(into, e, sign)` in `saveFile.ts`, which both sides already import. Line `914` leaves out
trains and build on purpose (the carry handles them): keep that difference explicit. Then a new
ledger field is added in one place. Risk: low. Effort: small. About 14 lines.

**20. Name the money parameters.** `moneyParams` is read by index (0 to 10) at `:815-820` and
`950-961`. Its start value `[2.5, 2, 100, 9]` (`:249`) is a stale placeholder with 4 entries where
11 are read. Decode `money_params()` once into a named object, in the order of `sim.d.ts:420`'s
doc, and spread it into `moneyState`. B1's fix is then one readable line. Risk: low. Effort: small.
About 8 lines.

**21. `lineDaysAt` has one caller and three fixed arguments.** `clock.worker.ts:1089-1127`: the only
caller (`:1094`) passes `closed=false`, `shift=undefined`, `past=true`. Drop the two parameters,
the `a.today(now)` branch and the local-midnight `h0`, and rewrite the doc for the past-24-hours
case. `TrackApi.today` then has no caller in the app; the sim reviewer can drop it. Risk: low.
Effort: small. About 8 lines.

**22. Duplicates in the worker.**
- Turning edge and line polylines into local units is copied: `buildState` (`:626-637`, `697-703`)
  and `slotSummary` (`:1235-1240`, `1249-1254`) → `edgeLocal`, `lineLocal`. Careful: `slotSummary`
  must not fill `edgeDrawn` without `edgeSpeedDrawn`, or `buildState`'s `?? []` (`:645`) gives
  zero speeds quietly.
- `failAnswer` (`:2002`) repeats `isPreview`'s four kinds (`:478`) and `sendStale`'s preview
  literal (`:480`) → `emptyPreview(seq, extra)`.
- `zonesOfPoints`' `stride` and `start` parameters (`:447-468`): every caller uses the defaults.
- The model copy in `keep()` (`:289`) and `rebuildModel` (`:329`) is the same.
- `api.set_now(gameNow())` at `:2093` repeats `:2056`.

Risk: low. Effort: small. About 25 lines.

**23. Fallbacks on fields that are always present, and fields nobody reads.**
- Always present: `msg.upkeep ?? []` (`:2135`), `msg.day ?? 0` (`:2169-2170`), `q.day ?? 0`
  (`:2192`); in the client, `l.upkeep ?? 0`, `l.reversals ?? []`, `l.endTurns ?? []`,
  `s.fullTrack ?? []`, `s.links ?? []` (`:137-164`).
- `ClockRender.edgeSpeed?` (`protocol.ts:348`) is always set. Make it required, and drop the
  conditionals at worker `:1582,1595`.
- Nobody reads FromClock `saved`'s `bytes`, `encodeMs` and `totalMs` (`protocol.ts:480`). Only `at`
  is read.
- Nobody reads realRoute's `fit.mean` (`realRoute.worker.ts:43,637-639`).
- `networkSnapshot` (`clockClient.ts:36,76,635`) is an exported signal with no reader outside the
  file. Make it a plain `let`, and fix the mention at `protocol.ts:497`.

Risk: none. Effort: small. About 15 lines.

**24. Types derived from the protocol.**
- `clockClient.ts:52-60` restates the FromClock `preview` fields. Use
  `Omit<Extract<FromClock, { kind: "preview" }>, "kind" | "seq">`.
- `saveSlotsState` (`:28-29`) and `manageSaveSlot`'s action union repeat `protocol.ts:182-183`.
- `saveFile.ts:30-35` and `45-50` repeat the ledger fields. Make one `LedgerEntry`, which both
  entry types extend.

Risk: none. Effort: small. About 15 lines.

**25. One "newest waits" queue in the client.** `clockClient.ts:225-230,618-633,677-697`: the station
check and the crossover check use the same "one in flight, the newest waits" code (Busy/Next/Cb
trios). Write one small generic helper. Leave the preview queue alone, because it has the stale
logic. Risk: low to moderate (callback order). Effort: small. Mechanical: no. About 15 lines.

**26. One regions.json, read once.** `realRoute.worker.ts:78-100` and `map/realNetwork.ts:92-114`
each fetch and parse noritetsu's `regions.json` (about 1.5 MB), the second on the main thread. Each
works out ring boxes by its own copy of the loop, and `realSnap.ts:38-42` (`ringBox`) has a third.
The base URLs are written twice (`realNetwork.ts:22-23`, the worker's `:80`).
- Step one (low risk): share `ringBox`, `inRing` and the base list through a small module both can
  import (see item 34).
- Step two (moderate): the worker already sends the page a `cover` message with the rings. It
  could send the boxes the page needs too, so the main thread neither fetches nor parses the 1.5 MB
  file.

Effort: small then medium. About 20 lines.

Comments to fix in these files as they are touched:
- stale: `clock.worker.ts:8` ("per-tile updates are follow-up work"); `saveFile.ts:102-106`
  ("TWT5 world frame"; TWT6 Mercator is the world frame, as `:118-126` says)
- history: `saveFile.ts:170` ("before T-894")
- misplaced: `slotSummary`'s doc (`:1195-1196`) is cut off by the THUMB constants
- stacked JSDoc blocks in `protocol.ts:46-48, 143-147` (the first is an older duplicate),
  `413-414, 460-463`
- spacing: `:1545`, `:1893`

## Batch 4: build tools

Owns: `map/tools.ts`, `edgeEdit.ts`, `nodeDrag.ts`, `endMarkers.ts`, `blueprintBox.ts`, `pick.ts`,
`stopMarkers.ts`, `realSnap.ts`, `curveCircle.ts`, `trackTagRows.ts`, `trackHover.ts` (topSeg
callers only) and `network.ts` (`previewStrokes` only). Fixes B5 too. Mind the private fields
checks read (see the list at the top).

**27. `pick.ts` `topSeg`'s `offset` and `slot` are never used.** `:97,109,130,137-147`: all 9
callers (pick.ts 5, `trackHover.ts` 3, `blueprintBox.ts:309`) pass `undefined, 0`. Drop both
parameters and the sideways shift. Risk: low (typecheck). Effort: small. Mechanical: yes.
About 12 lines.

**28. One curve-label placer, in `trackTagRows.ts`.** There are three copies: `tools.ts:1196-1211`,
`edgeEdit.ts:219-225` and `nodeDrag.ts:191-204`. Only the tools copy skips labels that are off
screen or overlapping and caps them at 300, which `docs/ui-building.md:84` says all three do. Move
`curveLabels`, `setCurveLabel` and one `placeLabels(pool, box, labels, toScreen)` there, so
`edgeEdit` and `nodeDrag` stop importing the 1,400-line `tools.ts` for two helpers. Also share:
- the box setup (`edgeEdit.ts:70-72`, `nodeDrag.ts:46-49`, `tools.ts:137-139`,
  `endMarkers.ts:31-34`)
- the drag teardown (`edgeEdit.ts:173-174`, `nodeDrag.ts:166-168`)

Risk: low (the drag labels get the culling the docs describe). Effort: small to medium.
About 35 lines.

**29. `tools.ts` repeats.**
- Clearing the preview is written 5 times (`:158-160,369-375,453-456,472-473,501-506`) →
  `clearPreview()`. `:472` does not clear the drawn preview today, and would.
- The 7 tag branches in `place()` (`:1227-1315`) each set display, class, content and transform →
  `showTag(content, bad, x, y)`, as `stopMarkers.ts:327` does. Keep the field name `tag`.
- The line tool's `toggle` and `addStop` (`:782-820`): `addStop`'s only caller is `toggle`, which
  has already made `addStop`'s checks (`:804-811`). The "already a stop" message can never show,
  and the refresh after the edit (`:797-800`, `816-819`) is copied. Merge them into one.
- The lazy SVG creation is written 3 times (`:874-880,1006-1016,1172-1179`).
- `stationRings` (`:1017-1020`) recomputes what `circleOnScreen([h.lx, h.ly, unitMetres()])`
  returns.

Risk: low. Effort: small. Mechanical: yes. About 50 lines.

**30. Pointer-move work in the tools.**
- `tools.ts:621-658` `tripsNear` walks every demand cell on each `place()`, which runs after every
  map frame and every pointer move while the station tag is up. Remember the last
  (key, lng, lat, unit) and its answer.
- `blueprintBox.ts:90-111` `grabAt` makes two passes over `w.edges` with `p.edges.includes` inside,
  on every pointer move. Cache the movable nodes per (`bpPicked` value, world version).
- `tools.ts:1200` reads the container's `clientWidth` and `clientHeight` straight after writing a
  transform (`:1195`), which forces a layout every frame while drawing. Use `overlay.cssW` and
  `overlay.cssH`. The same goes for `stopMarkers.ts:144-146,167-168,222-226`, which keep their own
  copies and a resize listener.
- `tools.ts:709-718`: hovering a plain node with the delete tool projects every point of every
  edge. Use `edgesNear` and `pointerLocal` as `snap()` does.
- `lastStationTrack` (`:260-286`) scans every node on each move over overlapping track. Cache it
  per `lastPlaced` and world version.

Risk: low. Effort: small. Mechanical: mostly.

**31. Dead in the tools.**
- `Tools.drawing` (`:185-187`), `EdgeEditor.dragging` (`edgeEdit.ts:113-115`) and
  `BlueprintBox.dragging` (`:84-86`)
- `RealSnap.reachLast` (`:98-99,339`), written and never read
- `CurveLabel.radius` (`:1350,1360`), never read, and its comment is wrong
- `tools.ts:699` `if (hit) return hit;`: `hit` is always null there
- `altsKey`'s default parameter (`:1132`)
- `edgeEdit.setPis` (`:109-111`) is a one-line wrapper with one caller
  (`ui/inspector/others.tsx:574`)
- exports used only in their own file: `cellIndex` `CELL_SIDES`/`CellIndex`, `snapIndex`
  `EdgeNear`, `stopMarkers` `stopTiers`, `blueprintBox` `segmentTouches`, `stationSnap`
  `SnapEdge`/`SnapNode`/`edgesByNode`

Risk: none. Effort: small. About 20 lines.

**32. Small moves.**
- `previewStrokes` (`nodeDrag.ts:208-236`) is a stroke merger that `blueprintBox` also uses. It
  belongs in `network.ts` beside `strokesFromPreview`.
- The pure tag helpers at `tools.ts:1335-1432` (`placementTag`, `placedLevels`, `TOP_KMH`,
  `HSR_KMH`, `COST_YELLOW`) belong in `trackTagRows.ts`.
- The 9 px hit radius is written three times (`nodeDrag.ts:25`, `endMarkers.ts:21`,
  `blueprintBox.ts:28`).
- `blueprintBox.ts:306,309`'s bare 7 is `SLOP_PX`.
- The 5000 m radius cutoff appears at `tools.ts:1348` and `edgeEdit.ts:217`.

Risk: none. Effort: small. About 15 lines.

Comments to fix in these files as they are touched:
- stale: `edgeEdit.ts:3` and `nodeDrag.ts:5` ("radius and speed"; the labels show speed only
  since T-530)
- history: `tools.ts:5, 1365`, `realSnap.ts:8`, `endMarkers.ts:5`, `stopMarkers.ts:9`
- written as an instruction: `pick.ts:329-331`
- `tools.ts:49` is a second import from `../game/ui`
- the unformatted dense blocks at `tools.ts:697-698, 763-764, 839-894, 980-998, 1043-1162`
- `pick.pointerLocal` and `demandHover.pointerLocal` share a name but mean different things (see
  item 41)

## Batch 5: geometry constants and shared shape tests

Owns: `game/frame.ts`, `map/geo.ts`, `map/coarseOutlines.ts`, `game/realms.ts`,
`game/trackZones.ts`, `game/listScope.ts`, `game/walks.ts`, `game/zones.ts`, `game/cities.ts`,
`game/placeList.ts`, `map/hoverRest.ts`, `map/stationNames.ts`, `map/airportMarks.ts`,
`map/curveCircle.ts`, and the constant lines only in `map/tools.ts`, `map/realSnap.ts`,
`workers/realRoute.worker.ts`, `workers/cityReach.ts` and `workers/commuters.worker.ts`.
Do it after batches 2 to 4, which edit some of the same files, or do it before them, but not at the
same time.

**33. One set of earth constants.** (rank 5)
- `EARTH_R = 6371008.8` is defined in `game/frame.ts:29`, `map/geo.ts:21`, `map/tools.ts:77` and
  `map/curveCircle.ts:3`, and written inline at `tools.ts:645,1019` and
  `realRoute.worker.ts:52`. Keep `frame.ts`'s and import it.
- Metres per degree `111_320` (and its partner `110540`) is written about 20 times:
  - `cities.ts:213,220,227,230`, `zones.ts:15,68`, `listScope.ts:54`, `placeList.ts:55,86,146`
  - `stationNames.ts:180`, `airportMarks.ts:58`
  - `cityReach.ts:18`, `commuters.worker.ts:95`
  - `demandClusters.ts`, `clock.worker.ts` (2 each)

  Add one `M_PER_DEG` beside `EARTH_R`.
- "Pixels per local unit at a zoom", `512 * 2 ** zoom / EARTH_C`, is written 9 times:
  `renderer.ts:71,1747,1936`, `demandOverlay.ts:571`, `intercityView.ts:211,299,374`,
  `realSnap.ts:169,389`, `demandHover.ts:77`, `stationGeometry.ts:39`. Add `pxPerUnit(zoom)` to
  `geo.ts`.
- Three ground-distance functions do one job: `walks.ts:159` `metresBetween` (equirectangular),
  `listScope.ts:53` `distFromCentre` (the same formula with its own constant), and `geo.ts:75`
  `groundM` (haversine). Keep two at most: an exact one and a fast one.

Risk: low (some figures change in the last digit where 111,320 meets 6371008.8 × π/180, which
is 111,195; use the same constant each file uses now unless the change is intended). Effort: small.
Mechanical: yes, constant by constant. About 30 lines.

**34. One point-in-ring test.** There are 7 copies of the even-odd crossing test:
- `realSnap.ts:44-51` and `realRoute.worker.ts:102-109` (identical)
- `hoverRest.ts:53`, `realms.ts:84-93`, `trackZones.ts:120-129` (banded)
- `cityReach.ts:108,250` (a local-metre form)

Put `inRing`, `ringBox` and `nearRing` in one module with no DOM imports (say `game/rings.ts`), so
the worker can import it too. Risk: low. Effort: small. Mechanical: yes. About 25 lines.

**35. One test against the coarse zone outlines.** `realms.ts:84-93` `inRings` scans every ring of
a zone's coarse outline on each call. `trackZones.ts:92-129` built a banded index for the same
outlines because the full scan "took 0.6 s". Both read `coarseOutlines()`, and each keeps its own
copy of the map (`realms.zoneOutlines`, `trackZones.outlines`). Move `bandsOf` and `inside` into
`map/coarseOutlines.ts` as `inOutline(id, lon, lat)`, and use it in `realmOfPlace` and
`hiddenRealmAt`. `realmOfPlace` runs for every station of each world snapshot whose position
changed, and the click test too. Risk: low (the same even-odd rule). Effort: small. About 15 lines.

**36. Widening a zone box is written twice.** `zones.ts:64-72` (`reachBoxesOf`) repeats the
margin math of `cities.ts:228-231` (`zonesTouching`), as its own comment says. Make one
`widenBox(box, marginM)` in `cities.ts`. Risk: none. Effort: small. About 5 lines.

## Batch 6: tags, labels and small map files

Owns: `map/capacityMarkers.ts`, `reversalMarks.ts`, `stationLabels.ts`, `labelLayout.ts`,
`networkTips.ts`, `trackHover.ts`, `demandHover.ts`, `hoverRest.ts`, `airportMarks.ts`,
`stationSound.ts`, `stationLife.ts`, `gradeMarks.ts` (the colour cache only, if batch 1 has not
done it), `mapView.ts`, `basemap.ts`, `cityBoundaries.ts`, `zoneLoading.ts`, `realmOverlay.ts`,
`realNetwork.ts`, `game/format.ts` and `main.ts` (`tripClick`, the basemap import). Fixes B9 too.

**37. Delete the dead trip-tool click on capacity tags.** `capacityMarkers.ts:36` (`tripClick`),
`:127-134` and `main.ts:~286`. The rule `.cap-marks.pass .cap-mark { pointer-events: none }`
(`style.css:687`) applies whenever the tool is not Select, and the effect at `:62` sets the class
synchronously, so the branch never runs. Its own comment calls it a fallback. Keep the tool guard
at `:135`, because t133 calls `onclick` on a mock DOM. Risk: low. Effort: small. Mechanical: yes.
About 9 lines.

**38. Shared tag pieces.**
- A `BoxGrid` (`hits(r)`, `add(r)`) in `labelLayout.ts` for the two copies of the 64 px bin grid:
  `stationLabels.ts:239-246,268,272-276` and `capacityMarkers.ts:193-200,234-247`.
- `tagLines(el, TipLine[])`: the "one div per line" fill at `capacityMarkers.ts:81`,
  `reversalMarks.ts:109`, `trackHover.ts:186`, `tools.ts:1259,1313` and `stopMarkers.ts:329`.
- `tipBeside(tip, el, container, side)`: `capacityMarkers.ts:80-85` and `reversalMarks.ts:108-113`
  differ only in the side.
- The 14 px pointer-tag offset (`trackHover.ts:192-193`, `demandHover.ts:133`,
  `hoverRest.ts:121-132`, whose "flush with it" depends on the other two) → one `TAG_OFFSET_PX`.
- `firstSymbolId(map)` in `basemap.ts` for the lookup at `cityBoundaries.ts:177`,
  `realmOverlay.ts:105`, `zoneLoading.ts:128` and `realNetwork.ts:137,172`.
- The projected-box clamp copied at `realmOverlay.ts:70-75` and `zoneLoading.ts:70-75`.

Risk: low. Effort: small. Mechanical: yes. About 45 lines.

**39. One home for the train capacity figures.** Seats and crush per car (44 and 160) are written
5 times:
- `renderer.ts:40`, `demandLayers.ts:63`, `networkTips.ts:33`
- `ui/inspector/others.tsx:478-479`, `LineInspector.tsx:306`

The gauge formula is copied at `demandLayers.ts:69-74`, `networkTips.ts:89`, `others.tsx:521`,
and in the shader at `renderer.ts:846-850`. Export the constants and `gauge` once (from
`game/boarding.ts` beside `tripLoad`), and interpolate them into the shader string. The ui reviewer
lists the inspector half; one agent should do both. Risk: low. Effort: small. Mechanical: yes.
About 10 lines.

**40. Formatting helpers into `game/format.ts`.**
- `trackHover.ts:31-35` `riders` and `networkTips.ts:36-37` `forecast` round the same way, and
  `:36` says so.
- `networkTips.ts:35` `n0` is a local formatter.
- `game/points.ts:31-41` `pointsShort` repeats `moneyShort`'s `sig` (`format.ts:40-43`) and its
  999.5 thresholds. Export one `sig3` and write `pointsShort` with it.

Risk: none. Effort: small. About 15 lines.

**41. Small dead and needless pieces.**
- `mapView.ts:78` `export { maplibregl }` is imported by nobody. `:13` re-exports basemap functions
  only for `main.ts:33`, which can import `basemap.ts` itself.
- `stationSound.ts:88-91`: the `ready` getter is never read, and nobody reads the boolean `play()`
  returns (`stationLife.ts:214`).
- `demandHover.ts:81-83` `pointerLocal` is a one-line wrapper of `toLocal`, and its name clashes
  with `pick.ts:26`. Inline it at `commuterView.ts:439,568`.
- `networkTips.ts:51-67` `circleOf` repeats `circleList`'s member filter (`:196-203`) and builds
  a Set of every built station on each station hover.
- `airportMarks.ts:127-145`, every map frame: reads the container's size (use `overlay.cssW/H`),
  writes every mark's display and transform without checking for a change, and does not return
  early when nothing is shown.
- Exports used only in their own file: `reversalsOf`, `stationNames` `STREET_NAME_ZOOM`,
  `Suggestions`, `endsInStation`, `suggestNames`, `placeName`, `stationLife` `TIRED_WAIT_S`,
  `CROWDED_LEFT`, `boardedAt`, `stationSound` `MAX_POPS`, `Arrival`, `zoomGain`, `trackHover`
  `dailyRiders`, `trackTip`, `demandHover` `MIN_PICK_PX`, `Finder`, `faces` `MAX_PER_PLATFORM`,
  `sprite`, `geo` `ANCHOR_*`, `gradeMarks` `markRadii`. Dropping `export` is optional and of
  little value. Do it only where a file is being edited anyway.

Risk: none. Effort: small. About 20 lines.

Comments to fix in these files as they are touched:
- stale: `networkTips.ts:3` (the tag shows "after the same rest as the track's", but it shows at
  once); `stationSound.ts:29` ("the red face's test"; faces are not red since T-802)
- history: `networkTips.ts:10-11, 150-151`, `geo.ts:43`, `trainFollow.ts:6`
- future-ideas notes, not a why: `airportMarks.ts:3-6`
- detached from its line: `faces.ts:185`
- `zoneFocus.ts:47` is a redundant first `viewCentre` write; leave it to the agent editing that
  file

## Batch 7: render-side map views

Owns: `map/network.ts`, `commuterView.ts`, `intercityView.ts`, `demandViews.ts`, `bubbles.ts`,
`boundaryGeometry.ts`, `coarseOutlines.ts` (decode only), and `game/trip.ts` (the far-end helper
only). Do it after batch 1. Fixes B8 too.

**42. `Strokes.info` is 4 floats a segment, but only the first is read.** `network.ts:44-47,124,150,524`:
slots 1 to 3 are always the constants 0, 1, 1, kept from before T-190. Every reader takes
`info[i*4]` only: `renderer.ts:1509`, `demandLayers.ts:156,180`, `tripView.ts:447`,
`trackHover.ts:62`. One float a segment saves 12 bytes per line segment, in memory and in the
worker transfer. Risk: moderate (the stride changes in 6 files at once, and a missed reader reads
the wrong stretch). Effort: medium. Mechanical: yes, with care. About 5 lines.

**43. One helper for a trip to a far end.** The trip is built three times, each with the
`sel.end === "home" ? { from: near, to: at, far } : {...}` swap and the same `tripOpenedFrom`
literal: `intercityView.ts:494-497,544-546` and `commuterView.ts:559-562`. Add one helper in
`game/trip.ts`. Separately, `commuterView`'s `nearPoint` reaches `intercityView` two ways (a
`click()` argument from `demandViews.ts:179`, and `setNearPoint` at `intercityView.ts:555-556`).
Keep one. Risk: low. Effort: small. About 15 lines.

**44. Repeats inside `commuterView.ts`.**
- The `[mt, mw, md]` end swap appears 4 times (`:485,711,767,857`) → `endModes(m, end)`.
- The "shown is current" test is copied (`:254`, `318`).
- The box filter is copied: `groupsIn` (`:436-446`, with a literal 8 for `DISC`) and `farIn`
  (`:566-575`).
- `ZoneShown` (`:192`) is the type of `shown` (`:172`).
- `Place` is declared and exported identically at `commuterView.ts:38` and `demandViews.ts:36`,
  and nobody imports either.
- The experiment switches `SPLIT_BUBBLES` (`intercityView.ts:86`) and `LINES_TO_SELECTION`
  (`commuterView.ts:58`) are fixed to true. Fold them in.
- `intercityView.ts:333` clears the labels on every zoom event, and `placeLabels` then re-creates
  and measures up to 8 (each `offsetWidth` forces a layout). `placeLabels` already reuses
  elements: drop the clear, and trim on the early returns.

Risk: low. Effort: small. About 35 lines.

**45. `network.ts` and `boundaryGeometry.ts` small items.**
- `network.ts`: `edgeOf` (`:580`) duplicates `edgeEnds` (`:502`). `lngLatLocal` (`:665-667`) only
  wraps `toLocal`. The sample index math is copied in `lineAt` and `laneAt` (`:688-708`).
  `rightOf`'s `x` parameter is unused.
- `boundaryGeometry.ts`: the "cells by whole degree" loop is written 4 times (`:347,514,523,624`)
  and the box union twice (`:362,371`) → `eachCell(box, fn)`. Its header describes only
  `cityGeometry`, not the shade cut or `OutlineEdges`.
- `coarseOutlines.ts:43-48` decodes deltas inline though `decodeRing` (`:26-31`) does it.

Risk: low. Effort: small. About 30 lines.

Comments to fix as touched: history at `network.ts:45-46, 192-193, 523`,
`intercityView.ts:22-23, 69-71, 77, 344` and `demandLayers.ts:185`, and the typo "asked for,." at
`commuterView.ts:43`.

## Batch 8: game/

Owns: `game/world.ts` (after the agent now editing it has landed), `trip.ts`, `demand.ts`,
`regionMoney.ts`, `boarding.ts`, `complexLive.ts`, `ui.ts`, `cities.ts`, `lineActions.ts`,
`account.ts`, `format.ts`, `palette.ts` and `main.ts` (the comments only). Fixes B7 too.

**46. Delete exports nothing references.** Each was searched over app/src (ui/ included) and
checks/, and none is mentioned anywhere outside its own line:
- `world.ts:236-239` `setLineColour`, `:245` `stationById`
- `demand.ts:84-89` `stationRidersFor`
- `format.ts:147-150` `perKmFrom`
- `lineActions.ts:64` `editingStopsOf`
- `regionMoney.ts:131` `RegionLedger` (or use it as the base of `regionLedger`'s return type at
  `:135`)
- `trip.ts:623` `PERIOD_NAMES`
- `account.ts:75` `signedIn`

Risk: none. Effort: small. Mechanical: yes. About 20 lines.

**47. Merge routes once in `trip.ts`.** `mergeMarketPeriods` (`:334-349`) and `averageTrips`
(`:483-502`) both do the same steps: merge routes by label with a weight, keep the five biggest,
renormalise, keep only the kept routes' pairs and re-index them. Make one
`mergeRoutes(sources, label)` returning `{ pairs, routes, cut }`. The station-name closure `st` is
also written three times (`:195,217,286`). Risk: low (the trip explorer's routes; t586 compares
trips). Effort: small. Mechanical: yes. About 25 lines.

**48. One "fold edits while one is in flight" helper in `world.ts`.** `setTrainCounts` (`:95-121`)
and `queueHeadway` (`:169-194`) are the same machine: a pending signal, a per-line job, a send
loop, the refusal and the cleanup. They differ only in the op, the undefined/null sentinel and the
`noteAnswer` guard. One small generic helper replaces both. `stepTrainCount` and `setTrainCount`
(`:58-75`) share their first four lines too. Risk: low to moderate (the stepper's feel when
clicking fast; check that rapid clicks still fold). Effort: small. Mechanical: no. About 25 lines.

**49. Look up stations and lines by id once per world state.** `world.value?.save.stations.find(...)`
and `save.lines.find(...)` are written about 30 times. Some run on hover paths:
`hoverRest.ts:92,98`, `networkTips.ts:71,134,147`, `trackHover.ts:55`, `tools.ts:175,785-817`,
`stopMarkers.ts:190`. Others: `trip.ts:632,639`, `holidays.ts:42`, `remove.ts:27,47`,
`lineActions.ts:43`, `demandViews.ts:198`, `nodeDrag.ts:79`, and the ui inspectors. They look up
by `id`, by `num`, and by `Number(id)`. Build `stationsById`, `stationsByNum`, `linesById` and
`linesByNum` maps once per `WorldView`, in `clockClient.ts` where the view is made or as lazy
getters, and give `lineById` and `stationById` in `world.ts` those maps. Risk: low. Effort: medium
(many call sites, each trivial). Mechanical: yes. About 15 lines net, and O(1) hovers.

**50. Two exported `PlatformNow` types.** `boarding.ts:125` (used by `ui/inspector/others.tsx`)
and `complexLive.ts:55` export different shapes under one name. Rename the complex one
`ComplexPlatform`. Also `boarding.ts:99-100` sets `lo` and `hi` before the empty test. Risk: none.
Effort: small.

**51. `ui.ts` tidy.**
- The orphan doc "Platform length for new stations, m." at `:203` has drifted above
  `purposeFilter`; it belongs on `buildPlatform` (`:207`).
- The browser-settings migrations (`:149-166`, `OLD_THEME` `:21-22`, the `lineLoad` and dropped-key
  handling, the version list at `:136-138`) serve browsers from before 2026-10-09 to 2026-10-23.
  Only Anita and the checks' fresh profiles use the game, so they can go to a plain "unknown keys
  are dropped" step. This is her call (settings, not saves), and of low value.
- `:13-15`: the TABS comment narrates the City tab's removal.
- `palette.ts:73` `modeScheme`'s type repeats `BubbleScheme` (`:103`), and the grey
  `[163, 155, 143]` is written twice (`:86`, `:129`).

Risk: none, except low for the migrations. Effort: small.

Comments to fix in these files as they are touched:
- `cities.ts:14-25`: five stacked JSDoc blocks over `City`, where only the last attaches. Merge
  them into one.
- `demand.ts:207-210`: `CityCells` says "New York 202,337" and "the fixed NYC world frame". The
  cells are in the loaded game's world frame (T-125, T-308).
- `main.ts:1-13`: the header's URL list duplicates code-map's "URL switches" and is out of date
  (only six `perfOff` parts, no `perfTry`, `?acct`, `?backupMs` or `?demandWorkers`). Point at
  code-map instead.
- `main.ts:~383`: "1, 2, 3: draw track, place stations, delete" leaves out 5 (the blueprint tool,
  `TOOL_KEYS`).
- `main.ts:~379-412`: the keydown handler writes `nodeDrag.cancel(); edgeEditor.cancel();` three
  times. A `cancelDrags()` would do (2 lines).
- `main.ts:~473-586`: the `?debug=1` probes (about 115 lines) could move to a `debugProbes.ts`
  loaded with `import()` only under `DEBUG`. That keeps them out of the main bundle.
  `checkPrecision` reaches into `(renderer as any).tripsBy`, so give the renderer a small accessor.
  Risk low, effort small. Checks call `window.tw.*`, so keep every name.

## Batch 9: perf switches (needs the orchestrator's yes)

Owns: `perfFlags.ts`, `map/overlay.ts`, `map/stationLabels.ts`, `map/capacityMarkers.ts`,
`main.ts`, `workers/demandClient.ts`, `demandProtocol.ts`, `demand.worker.ts`, `game/zones.ts`,
`map/cityBoundaries.ts`, `ui/TopBar.tsx` and `ui/SkyBanner.tsx`. Do it after batches 2 and 6.

**52. Prune `perfFlags.ts` to the switches that still earn their keep.** There are 28 names in
three groups.
- **Keep: the A/B oracles checks use.** `clusters` (t326), `split` (t409), `reach` and `chunks`
  (t756), `periods` (t124), `early` (t303) and `zoneZoom` (t871). Each compares exact figures
  against the old path.
- **Keep, or drop, as Anita likes: the part switches.** `labels`, `shadow`, `hover`, `clock`,
  `overlay`, `markers` and `marks` turn a part off to measure its CPU. Each costs about one line at
  its use, and no check uses them.
- **Drop: the "before T-n" switches no check uses.** They keep an old, worse path alive for an A/B
  that was settled long ago:
  - `idle` (T-073), `noDouble` (T-075), `labelPan` (T-074)
  - `stagger` (T-173), `bareStart` (T-351), `ahead` (T-327)
  - `spread` (T-124), `close` (T-301), `span` (T-842)
  - `openAhead` and `batchOpen` (T-889), `track` (T-1047)

  `batchOpen` also takes the protocol's `oneByOne` and the worker's one-by-one path
  (`demand.worker.ts:360-370,406`).
- **Drop the `perfTry` switches.** `labelLayers` was never adopted (`stationLabels.ts:27,161`,
  `capacityMarkers.ts:101`, which calls `perfTry` once per tag in the map loop). `hoverCursor`
  (`main.ts:~97,~364`) is cheaper as the default: write the cursor only when it changes, with no
  visible difference. `idle` and `noDouble` are in its type but never called.
- Rewrite the file's header (`perfFlags.ts:1-46`) to list only what is left, and update notes/T-045.md
  and code-map's URL switches.

Why: each dropped switch removes a branch on a hot path (demand loading, the frame loop, label
placement) that nobody runs. The header is also history ("before T-..."), against the comment rule.
Risk: low (nothing a player sees changes with the switches off). The cost is losing the ability to
A/B those old paths, which notes/T-045.md records. Effort: small. Mechanical: no (each branch is
read to keep the live side). About 40 lines.

## Batch 10: split `renderer.ts` (after batches 1 and 7)

**53. Move the shaders.** Move the shader strings (`renderer.ts:89-948`) and the module constants
they read into `map/rendererShaders.ts`. The constants are `TEX_W`, `LINES_W`, `QV`/`qVars`/
`qPack`/`qZero`/`qUnpack`, `SEL_STRIDE`, `HOT_*`, `SPEED_BLEND` and `SEATS`/`CRUSH`, or the shared
constants from item 39. The shaders read only module constants, never class state, so the seam is
real. `renderer.ts` goes from about 2,400 to about 1,550 lines. A second, smaller seam:
`updateCrowds`, `placeCrowds`, `queueShare` and `dotCrowd` (`:2257-2355`) are pure CPU work over
`BoardHour` and could become a crowds module. Risk: low (a pure move; a missed constant fails the
typecheck). Effort: medium. Mechanical: yes.

## Batch 11: split the two big workers and the station tool (each its own sitting)

**54. Split `demandClient.ts` (3,048 lines).** The pure parts have real seams:
- `demandService.ts`: `Service`, `serviceFromSnapshot`, `packService`, `purposeDemand`
  (`:141-149,438-539`). `demandCache.ts:56-61` `Svc` then imports the type instead of copying it.
- `loadEstimate.ts` (`:291-436`, `OPEN_MS_PER_MB` through `solveEstimate`). The module state
  `solveWaves`, set in the constructor (`:643`), becomes a parameter.
- `demandAnswers.ts`: `intercityZones`, `addIntercity`, `corridorOf`, `corridorSplit`,
  `signature`, `byMask`, `bareParts`, `concat`, `topOf`, `periodsOfWorker`, `poolSize`
  (`:117-237`).
- `demandDays.ts` (medium): `dayResult`, `dayDemand`, `dayCorridors`, `bareView`, `factor`, as
  functions over the day and period tables.

Loading, closing and the queries stay in the class, because they share too much state. About 1,000
lines leave the file. Risk: low for the first three, moderate for the days. Effort: medium.

**55. Split `clock.worker.ts` (2,285 lines).** Four parts have seams:
- `masks.ts` (`:380-568`, about 190 lines): `api`, `frame` and `emptyTrack` are passed in, and the
  `masksIn` set is exported for `rebuildModel`.
- `thumbnail.ts` (`slotSummary` and the THUMB constants, `:1195-1330`).
- `edits.ts` (`apply`, `endArr`, `pisArr`, `freeEnds`, `doubleUpAnswer` and the preview decoders,
  about 200 lines). It returns `fares` and `newLineCars` instead of writing them.
- `ledger.ts`: the pure ledger helpers with item 19's.

Leave settling, slots and autosave, and panic recovery in the worker: they share `clk`,
`settledT`, `dirty`, `activeSlot` and the lease, and splitting them would need a shared state
object, which costs more than it saves. The worker ends near 1,700 lines. Risk: low as a pure
move. Effort: medium.

**56. Split the station tool out of `tools.ts` (1,433 lines).** The station tool (roughly
`:550-1160`, about 500 lines) shares only `snap()`, `screen()`, `pack()`, the box, ring and tag
elements, and `place()` with the track tool. `place()` mixes both tools' branches and must become
one place function per tool, and the checks read private fields (see the top). The line tool
(about 70 lines) is separate as well. Do items 28 to 32 first. Risk: moderate. Effort: large.
Mechanical: no. This one is worth doing only if the tools keep growing.

---

## Not worth doing

- A shared helper for re-adding MapLibre layers on `style.load`. Each site differs
  (`stationLabels.ts:94-95`, `realNetwork.ts:161,195`, `realmOverlay.ts:111/152`,
  `zoneLoading.ts:136/163`).
- A shared `addTo(map, key, v)` for the 17 `m.set(k, (m.get(k) ?? 0) + v)` lines. It is idiomatic.
  Only `points.ts:45` and `regionMoney.ts:30` define the same helper, and those two could share it.
- Dropping `export` from the ~150 exports used only in their own file. That is churn for no gain,
  except where a file is edited anyway.
- Layering. `game/` imports from `map/` and `workers/` in 12 places, and `workers/` holds
  main-thread modules (`clockClient`, `demandClient`, `cityReach`, `demandClusters`). code-map.md
  documents it, and moving files would touch every import for little gain.
- The demand query wrappers at `demand.ts:426-457` (eight `queries?.x(...) ?? Promise.resolve(null)`)
  read fine as they are.

Sub-reviews: the demand workers, clock worker, renderer and map views, build tools, and tags were
read by one subagent each. I spot-checked their main claims: `u_px_per_unit` and `a_dist`,
`StationShape.k`, `Cluster.nodeLocal`, B1 against `wasm.rs`, the 50-solve log and `WorldView.city`.
The `game/`, `main.ts`, `perfFlags.ts` and cross-file items are my own.
