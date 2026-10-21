# Performance survey (2026-10-15): where the costs are, what grows, what to do first

Design only. Anita, 2026-10-15: "quantify impact, prioritise things that are high impact and/or allow
scaling". Target machine: 4 cores, 8 GB (notes/T-361.md), which now gets 2 solving workers (T-381).
New measurements here are native and static (another agent was timing workers in a browser): a
scratch bench linking the sim crate with its counting allocator (`sim/src/demand/alloc.rs`), on
a shared box (Anita working), so treat times as +-30%; native is roughly 1.5-2x faster than WASM.
Scripts: the scratchpad's `perf/` (cellbench, cellmerge.py, samples.py, packbytes.py, market.py);
not kept, the numbers are below.

## 1. Where the costs are now (real New York save unless said)

| What the player feels | Now | Source |
|---|---|---|
| Load to first figures / final (card goes at final) | 3 workers 11.7-12.2 s / 26.8-27.1 s; 2 workers (target) 13.3 s / 33.5-34.5 s; slower laptop 1.3-2x | T-361 |
| Edit to final | New York edit 21-27 s at 3 workers, 28 s at 2; Toronto edit 4.5 s (clusters) | T-361, T-326 |
| Click | idle 0.33-0.43 s; during a solve 5-10 s until T-392 lands | T-303 |
| Memory | about 400 MB fixed per solving worker; demand WASM 1,460 MB at 3 workers, 1,050 at 2; renderer 1.86 GB at 3; aux worker 32 MB, 477 MB after a long pan | T-361, T-325 |
| Download at startup | about 43 MB uncompressed: market 14, New York 18.3, two reached zones 3.2, every zone's water 4.9 plus headers 0.34, every boundary 0.76, two WASM modules 1.3 | here, T-303 |
| Frames | 13-15% of a core while playing, our draw 0.1-0.2 ms; long tasks while loading are the main thread descheduled (3-7 at 3 workers) | T-076, T-348 |
| Main thread spikes | pack copy per worker 45-70 ms, final publish 70-110 ms, long pan 50 ms, edit round trip 130-214 ms on 10,000 km | T-354, T-356, T-070 |

**Panning (T-714, 2026-10-19, `checks/t714-pan.mjs`, 1400x900, RX 6600, demand final, clock at 60x).**
Main thread a frame over drags at zoom 12/14/16 in Manhattan and Brooklyn and a wheel zoom 11-16:
de76ad3 (before 10-18) 5.6 ms, 9e466a4 5.3, bfe93f9 5.7, d931dcf 5.2, today's tree 4.7; 60 fps
throughout, overlay draw 0.4-0.5 ms CPU and 1.3-1.4 ms GPU, camera at rest 8-12% of a core in all.
With the main thread throttled 4x: 20.5-23.8 ms, 3-6% of frames over 50 ms, no trend by build. A
5K framebuffer, the real network, bubbles with station circles, 960x and the track tool all left
today within old's numbers; the station tool alone stalled on hover (T-724, below). Of the pan's main
thread MapLibre's basemap is about 70%, our overlay plus per-frame placers about 20%. So no
regression in our drawing; lag felt on 2026-10-19 coincided with the machine at 60% CPU from other
work (three rustc and a pipeline python). `cityNear` (run on every map move) now scans instead of
sorting every zone.

**Pointer moves (T-724, T-689, 2026-10-19, `checks/t724-hover.mjs`, 1400x900, demand final).**
A move's handlers, window capture to bubble, over a zigzag in Manhattan z12/z14/z16 and Brooklyn z14.
Select tool: p50 17-19 ms, p95 24-27 (z12 6/9) before; 1.4-2.1 and 2.4-3.8 after; the resting list's
timer 18-25 ms to 2-4. Station tool: p50 14-27, p95 21-37 before; 1.1-2, 3-6 after; its snap
(every node and edge point projected) p95 8-11 ms to 0.1-0.8; the name looked up after a 150 ms
rest 10-17 ms, 36-58 on fresh tiles, to under 2 (7 in the first view). `pick` 2.6-5.3 ms a call
to 0.4. The costs were: every station's shape worked out per pick, every train placed, every
segment visited, the snap projecting 1,756 nodes and every edge point, `querySourceFeatures`
decoding every tile in view the first time (12-23 ms) and `getStyle`. Now: grids (`cellIndex.ts`,
`snapIndex.ts`) built in idle time after each new network (5-20 ms each), a station skipped when its
node is further than its platform reaches, a train only placed when its line's stretch near the
pointer could hold it, the name read only from the tiles within 400 m (each tile's features kept
until its data changes, and the view's tiles read ahead one per idle moment while the station tool
is on). Same answers: 30,000 picks, 6,900 snaps and track directions, 2,300 name suggestions
against the old code, identical. Left: the first one or two moves after loading or an edit scan as
before (21-35 ms once) until the grids are built. Wheel zoom: the per-frame placers' total 196 ms
to 90-120 over a 40 s scroll run (names and capacity tags write only what changed, collisions by a
64 px grid); every placement under 2 ms but the first time names show (zoom 12), when all 1,756
are measured once, 17-25 ms; frame callbacks max 9-11 ms; the move before a scroll 30 ms (T-675)
to 2-3. `artifacts/trainworld/t724/`.

**Trains/headway switch (T-747, 2026-10-19).** Anita's world save (143 lines, 103 still on
legacy rates, 40 on counts), subway M express to headways: 2.9-3.1 s in the clock worker's
`apply`, all of it `settle_counts` running its 80-round cap twice (about 12 ms a round on that save, two
`update_scoped` passes over every counted line). M express's high-level track (resource 4828)
is over capacity from the legacy-rate lines alone (rho 1.07), so the fill keeps its rate at 0,
and the `held` test (`|rho - 1| <= 1 - FULL_RHO`) never counted an over-full track as full
(T-752 changed that): every round repeated the last. The loop now stops at the first round that changes no rate or round
trip; final state identical to the bit (digest of every line's rates, counts and round trips
over six switches), native switch 3.1 s to 0.25 s the first time and 2.1 s to 25 ms after.
The panel also shows the new mode at once (`pendingHeadway`, null for back to counts). In
headless Chrome, demand final: click to the panel in the new mode 2.9-3.1 s before (the first
switch; later ones 0.1-0.2 s since the counts no longer moved), 1.5-5 ms after; the worker's
busy time for the first switch 2,970 ms to about 320 ms, later edits 60-65 ms in all. The real
New York save (mnr Hudson) after: 2-4 ms to the panel, edits 40-100 ms. Not the save's size: a
round costs more with more counted lines, but the seconds came from the cap.
`artifacts/trainworld/t747-*.json` and the traces beside them.

**Adding a stop (T-648, 2026-10-19, `checks/t648-stop-edit.mjs`, 1400x900, demand final, paused).**
The real save's lines stop at every station they pass, so each click first builds (and constructs)
a station on a long straight stretch of one line, then adds it with a click on its marker; 12
lines. The flash was the line panel's stop list: the stations passed (T-646) came from a query
asked after each new world, so for one to three frames the new stops showed with the old gaps, a
"+ 1 station passed" row beside the new stop that pushed every row below down a line and back.
Also found with the frame test (screencast, pixels the same before and after that some frame
between draws otherwise): for a frame or two the old trips drawn on the new run profiles (trains
jumping, or staying on a line that had just lost its service), and the past-24-hour counts a row
off for up to 250 ms. The toggle markers never changed twice (the brief's guess: the estimate and
the worker agree on this save; after an edit the worker's last tiers now stand anyway). Fixes: the
passed stations ride in the clock state; the client holds a state until its trips (posted right
after it, so world, network and trips are taken in one task); the counts follow their stations to
the new stop order. Latency, medians of 12 clicks in interleaved runs against HEAD (2a46781) on
its own Vite: click to world 97, 95, 94.5 ms before, 76, 70, 76 after; click to the frame drawn
with the new network 142, 139, 132 before, 116, 112, 130 after. With the station left a blueprint:
world 114 to 101, drawn 167 to 145.
Where it went: one worker message instead of an `insertStop` query and `setStops` (10-15 ms);
turn-back strokes searched every stroke of a line per call, now a binary search (incremental buffer
build 30 to 15 ms on the main thread, same buffers to the byte); edges and nodes left out of the
state when unchanged (transfer 12 to 6 ms); the shadow name layer re-tiled only when the names
move. Left, after: worker apply 5, state 6-9, buffers 15-20, transfer 6, the first boarding hour
about 10 (now before the state is taken), main thread 25 to take the state (world effects 15:
labels, money charts, places) and about 40 more to the drawn frame (panel renders, layers, the
frame). A station built as a blueprint (the line turns planned) costs 5-20 ms more in
`blueprint_cost_items`, a trial construction of the whole blueprint on every state. The counts go
blank for the edited line once the worker drops its riders, until demand returns (the model's
state, not drawing). `artifacts/trainworld/t648-*.json`, strips `t648-before-strip.png` and
`t648-after-strip.png` (same click, panel and map).

**The freeze after a stop edit (T-789, 2026-10-20, notes/T-789.md).** With Europe covered the market's
corridors were every place pair (300,700 at d8592be, 501,501 with 1,001 places), summed on the main
thread per period, day and publish: tasks of 3.4-6 s, 7-10 s frozen after each edit in any city. Now
only pairs with trips, the rest of a day's sums kept by key, the cache written in a worker: longest
task 0-56 ms in Paris and Grevena. Still growing with zones squared: each result message (T-790).

**Station circles zoomed far out (T-630, 2026-10-19).** Anita's world save (world-map-day73: 143
lines, 2,367 stations, Toronto to Tokyo) and the real New York save, circles on, clock at 240x,
zooms 5-10 over New York, Washington and Japan, 1400x900 and 2400x1300 at 1.5x. Drawing was never
the cost: 59-60 fps on the world save before and after, main thread 3-5 ms a frame panning and 1-2
still; overlay GPU 4-6 ms at z5-6 over New York (1.5-2 over Japan) and the same with the circles
off, so the track strokes, not 2,400 circle quads; circles rebuilt only when the world, solve,
period or a display setting changes (4-7 ms, at most once in 5 s), never per tick. The lag was the
circle's hover tag. `circleTip` runs on every pointer move that hits no train, station or track
(most of the map far out) and again each game minute while its tag is up (4 a second at 240x); it
worked out every station's complex and, for each, a set of every built station: 98-116 ms a call on
the world save, 38 on New York's. Now the circle list (centre, size, riders, busiest name) is kept
until the network, world, solve, mode, complexes or split change: 0.03-0.14 ms. Pick far out
(blueprints in the world save): a whole city's segments sorted per query, every width scanned per
call, the blueprint passes walking every candidate. Now the candidates are ordered by marks when
there are many, each set's last query is reused within a move, the widest width kept per array, and
the blueprint passes look at the blueprint segments only: 5.7 to 2.4 ms a pick at z5, 2.3 to 0.4
at z6, 5.5 to 1.8 at z8, 3.8 to 1.5 at z10 over New York; a pointer move's handlers p50/p95 8.7/24
ms to 5.9/9.9 at z5 and 8.1/21.8 to 5.5/9.2 at z6 (z10 2.8/4 to 2.7/3.5). New York's save has no
blueprints, so its picks are unchanged (1-1.4 ms) and moves 3/5 ms. Same answers: 50,000 picks
(trains on and off, each asked twice) and 5,800 circle tags at zooms 5-14 on both saves against
HEAD's copies, identical. Circles were not merged or culled far out, since they cost nothing
measurable. Scripts were scratch; outputs in `artifacts/trainworld/t630/`.

**Per worker, New York, native** (cellbench): commute engine opens in 50 ms (+12.5 MB); the everyday
layer in 1.2 s (+69 MB, peak +83), mostly destination sampling; its `set_network` 0.43-0.73 s
(+88 MB: `edge_pair` 56 MB and the pair list); the commute engine's `set_network` 0.45 s. One
period's free flow: commutes 1.5 s, everyday 0.5-0.8 s, each +46-50 MB held per period. So the
seconds are in the per-period rounds (commute pair loop first), and the fixed memory is in the
everyday arrays and set-up peaks. Packs are 83% per-cell arrays (16 x f32 plus h3 u64, zone u32,
region u8: 73 B a cell); routes, water and walk are the rest. Packs deflate 2.1-2.6x.

## 2. What grows with registered cities, built in or not

| Item | Now (18 zones) | At 200 zones | Why |
|---|---|---|---|
| Water masks, clock worker (T-302) | 36 requests, 5.2 MB, before the clock is ready | about 400 requests, 45-55 MB, held in clock memory | `clock.worker.ts` init loads every pack's water before "ready"; the track cannot draw and no solve can start until it ends |
| Boundaries (T-302) | 18 requests, 0.76 MB; 18 GeoJSON sources and line layers; a footprint and outside polygon per zone | 200 requests, 15-20 MB (tier 0 outlines average about 100 KB; London's 268 KB); 200 MapLibre sources | `cityBoundaries.ts` one source per zone, `zones.ts`, `cityReach.ts` |
| Place lists | 1.3 MB on first search or station tool | 15-30 MB | `placeList.ts` loads every zone's |
| Regional market | 14 MB download, +45 MB in every solving worker, even for a London-only save | today's format grows with covered zones (258k sites, 172k commute pairs for 13 zones); T-362 moves it into packs | `demandClient.ts` opens it always |
| Packs never closing (T-301) | aux 32 to 477 MB over a long pan | a world tour opens dozens: unbounded | WASM memory never shrinks |
| Registry, frames, time zones | 185 B a zone | 37 KB | fine |
| Offline joint commute gravity (T-297) | 12 min over 23k zones | grows with every zone in one gravity | only matters if the world's commutes stay one solve; keep it per region of touching zones |

Without T-302 the first item alone puts about 9 s at 50 Mbit/s and 400 requests in front of every
load at 200 zones; it is the one thing that hurts every player whatever they build.

## 3. Anita's idea: merge sparse cells into their H3 parent

**What scales with cell count.** Pack bytes (73 B a cell). In each worker: the commute engine's
cell arrays (about 60 B a cell, and twice, since the everyday layer opens its own engine), the
everyday layer's destination samples and `edge_pair` (68 slots a cell, u32 each: 544 B a cell,
113 MB for New York), station access per network (6 nearest stations, 49 B a cell per engine,
transient), and the opening and `set_network` loops over those slots. **What does not**: the
commute gravity is zone to zone on 2 km zones (6,407 for New York, shipped solved); mode choice is
over subzone pairs (cells grouped by their access within a zone: 4,733 on the real network), and
the per-period passes run over pairs, not cells. So merging cuts memory and opening time, not
round time.

**Counts.** Parent = res 8 (7 children); "trip ends" = a weekday's trip ends from the pack's
weights (commutes both ends, the everyday rates of `everyday.rs` at both ends); a parent merges
when its present children's ends total under the threshold. Packs already hold no empty cells.

| Pack | cells | under 100 people | merged at 1k / 2.5k / 5k / 10k ends | their share of all trip ends |
|---|---|---|---|---|
| New York | 207,766 | 74% | 49% / 61% / 70% / 78% | 4.4% / 10.4% / 21% / 37% |
| Toronto | 75,442 | 73% | 57% / 62% / 67% / 75% | 3.3% / 6.1% / 12% / 31% |
| Mumbai | 46,978 | 66% | 35% / 49% / 55% / 60% | 1.1% / 2.6% / 4.0% / 6.0% |
| London | 88,286 | 67% | 42% / 50% / 56% / 63% | 2.7% / 5.5% / 9.9% / 20% |
| Philadelphia | 110,560 | 79% | 53% / 66% / 75% / 82% | 8.0% / 20% / 39% / 64% |
| Windham | 12,688 | 98% | 80% / 83% / 84% / 85% | 49% / 68% / 81% / 91% |

**Savings, New York at 5k** (70% of cells): pack 18.3 to about 7.7 MB raw; per worker about
-100 to -110 MB (everyday -79, engines -17, access -10 transient) of about 450-500; opening and
`set_network` about 1-1.5 s less native per worker (in parallel, so about that off the load);
rounds unchanged. At 1k: about half of that.

**But most of the memory is available without changing the grain:**

- **48% of the everyday slots are dead today** (cells with no budget for that purpose: school and
  university slots of cells with no school, every slot of a cell with no residents), still
  allocated and still visited. Storing slots only for cells with a budget (an offset per purpose)
  gives identical results: New York 113 to 59 MB a worker; the nine Northeast cities about -200 MB
  of T-178's 496 MiB. Half a day to a day. (T-395)
  **Done 2026-10-15**, measured natively with the counting allocator (`t395_probe`, since removed):
  samples + `edge_pair` New York 107.8 to 56.6 MB, the nine Northeast cities 396 to 207 MB; heap
  after opening plus `edge_pair`, nine cities, 448 to 276 MB a worker; New York after
  `set_network` 160.7 to 113.5 MB. New York's opening (1.20 / 1.17 s), `set_network` (0.75 s) and
  20 passes (17.2 / 16.3 s) unchanged within noise. Every output dumped (loads, parts, kind
  totals, cell modes, selections, five periods and three crowding rounds on the real track)
  identical to the byte.
- **Samples scaled to a cell's trips** (2 to 16 instead of always 16, one per trip or so): slots
  fall to 23-33% of today's (New York 113 to about 32 MB), with a little more sampling noise in
  sparse cells and the totals kept. (T-397)
- Bytes are better won by compression on publish (2.1-2.6x, already planned with T-060) and, if
  still needed, u16 weights.

**What merging costs.** Results change exactly where it merges: at 5k, a fifth of New York's trip
ends and two fifths of Philadelphia's sit in merged cells. Suburban station catchments go from
0.1 km² to 0.74 km² grain (a centroid moves 300-400 m, about 4 minutes' walk), which is where
walk-or-ride (T-371) is decided. Trips inside a merged cell fall under the 150 m exclusion and
vanish unless an intra-cell distance is added. Mixed resolutions touch every h3 user: 17 pipeline
files, the bubbles' grouping by masking digits (`commuters.worker.ts`), the everyday seeds. About
3-5 days, moderate risk, and every tier 0 city's checks re-run.

**Plain answer**: it would help memory and pack bytes, not the seconds a solve takes. Do T-395
(identical results) now; T-397 if memory still binds; keep T-044 parked. If it is ever done, 1k
ends (half the cells, 1-8% of trip ends outside Windham-like zones) is the threshold to look at,
not 5k.

## 4. Ranked work

Impact on the target machine (2 workers) where it differs. "Scales" = what it lets grow.

**Do now**

1. **T-302 zone files on demand** (with T-400 for outlines and places): removes 36 requests and
   5.2 MB from before the clock is ready now, 400 requests and about 50 MB at 200 zones, plus
   15-20 MB of outlines and 200 MapLibre sources. Scales: registered zones. 1-2 days, low risk.
2. **T-396 the market by cluster and opened only when reached** (a requirement on T-377): without
   it a worldwide market makes the world one cluster and every edit re-solves everything (Toronto
   back from 4.5 s to 28.5 s, worse with each served city); and today every save pays 14 MB and
   +45 MB a worker for the Northeast market. Scales: served cities. Part of T-377; section 5.
3. **T-395 everyday slots only where there is a budget**: -54 MB a worker for New York (-108 MB at
   2 workers), about -200 MB a worker for a nine-city save; identical results. Scales: cities
   built in. Half a day to a day, low risk.
4. **T-398 lower a worker's set-up peak** (T-361 item 2): 80-150 MB a worker (estimate), so 2
   workers fit T-361's 1 GB budget after an edit. Scales: memory per city. About 1.5 days, low to
   moderate risk, bit checks.
5. **T-124 reuse the periods an edit did not touch**: since T-381, 2-3 workers each run 2-3
   periods in turn, so this now saves time, not only CPU. A train-count edit on a HIGH line
   re-solves periods 0 and 2 only (`PERIOD_LEVEL`); with today's fixed ownership (worker 0 holds
   0, 2, 4) that is 2 periods' rounds instead of 3, about 28 s to 19 s; if the dirty periods could
   go to idle workers, about 10 s. 1-2 days, moderate risk.

**Before about 50 cities**

6. **T-301 close packs and cap the aux worker**: bounds the aux at a few hundred MB on long pans
   (477 MB today on the Northeast alone). 1 day, low risk.
7. **T-399 one engine per city** (T-361 item 3): done 2026-10-15, -10 MB a New York worker, -33 MB
   with nine cities; the subzones stay per layer (different weights). notes/T-399.md.
8. **T-401 profile one period's round by layer natively**, the basis for T-036: no breakdown
   exists beyond cellbench's free flow (commutes 1.5 s, everyday 0.5-0.8 s). Half a day.
9. **T-281 walk search on opening**: about -0.5 s of load in every worker. 1 day, low risk.
10. **T-397 samples scaled to trips**: a further -60 to -80 MB a worker for New York; opening about
    half as long. 1 day; changes figures slightly.
11. Pack compression on publish (T-060): 43 MB of startup download to about 18-20 MB.

Big networks (the whole of Japan, T-737) have their own ranking in section 6; its first item, T-756,
belongs beside items 1-5 once a player's network spans several zones.

**Later**

12. T-036 faster pair loop (the round time itself; large, high effort), T-354, T-356, T-359,
    T-346 (each 50-110 ms of main thread or about 1 s on a bare start), T-070 (big networks),
    T-076 (a smoothness call for Anita), T-344 (rare), T-297 (pipeline only).
13. T-044 / the cell merge: only if bytes still bind after compression and T-395, T-397.
14. T-178 is covered by T-395 and T-397. T-168 looks covered by T-325 (unreached zones open bare)
    and T-326 (only touched clusters re-solve); worth checking and archiving.

## 5. T-362 and the cluster split

Inputs since the brief (Anita): near pairs below **250 km** (not 300); long trips must not merge
whole cities into one solve.

**What breaks today's decoupling.** `demandClusters.ts` joins one MARKET node to every station
within 200 km (`MARKET_REACH_M`) of any market zone centre. With every zone in a world market that
is every station: one cluster, and T-326's gain is gone. Also the market is one population
(`IntercityApi`) solved over all its pairs at once, so it can only live in one cluster as built.
The design itself is otherwise safe: normalisers are fixed against the world field (adding a zone
changes no other zone), car and air have no capacity, far pairs between served zones only.

**What the code offers for loose coupling.**

- A cluster's round already takes its crowding from one summed load vector per route node
  (`demand.worker.ts` `step`: `total` into `crowd_with_external`). Adding a fixed background vector
  (another cluster's or the market's last loads) is a change in the worker, not the kernel.
- Crowding is exactly zero below seats (`crowd_factors`: 1 + alpha x(1+x)/2 with x from load over
  seats, waits only on full trains), so background on a line under its seats changes no figure.
  And the everyday pass already skips a re-pass when the factors are unchanged (`uniform`). So
  "re-solve the neighbour only past a few percent" can be judged on the factors, per route node:
  re-solve only if some factor moves by more than a tolerance. Exact for uncrowded lines.
- Anita's point fits: long-trip riders crowd one or two intercity lines, and only segments over
  seats couple. But a population cannot be solved on some lines only: each pass routes all of its
  pairs over the whole cluster graph. So the workable form is per population, not per line:
  the market's loads as background on city clusters, and the market re-solved when an intercity
  line's factors move; a city cluster re-solved when the market's background moves its factors.
- Needed in `splitClusters`: the market splits by network component (its pairs grouped by the
  components their end stations reach; a pair with no rail path is constant, solved once), and
  the cluster `sig` includes the background quantised to the tolerance, so small changes keep
  results. With near pairs to 250 km, MARKET_REACH_M should follow that cutoff.
- Risk: the market's rounds and a city's rounds stop being averaged together (MSA per round), so
  results differ slightly from a joint solve and depend on order. A bit check against the joint
  solve on the Northeast (market loads are a few percent of New York's) bounds it.

## 6. Big networks: all of Japan (T-737, 2026-10-19)

`checks/t737-japan.mjs`, notes/T-737.md. Japan's whole N02 network (709 lines, 10,320 nodes) and its
seven cities alone (491 lines, 5,992 nodes), 2 solving workers. Playing at 240x and 960x does not
lag: 55-60 fps, 4.6-5.5 ms of main thread a frame over the pans, 0.3% of frames over 50 ms. Load 7.7 s
(10.3 s before T-737 fetched the water masks before the one load). Demand is the wall: the whole
country never solves (each worker out of WASM memory: station x station tables over one cluster of
10,320 stations, 426 MB each; Tokyo's pack alone natively 243 s and 3.28 GB peak); the seven cities
take 680-735 s to their final with 2.5 GB a worker and 6 GB in the renderer.

T-761 (2026-10-19, notes/T-761.md): the whole country now solves, in the browser on 2 workers its
final at 1,158 s with 3.2-3.4 GB a worker; the market's pass is about 85 s a period in WASM and most
of that time (4 passes x 3 periods a worker), the logsums over 12.4 million zone pairs its largest
part (T-767).

Ranked (T-756, T-757 and T-760 done 2026-10-19, notes/T-756.md):

1. **T-756 each population on the stations within its reach**: Tokyo's pack on the whole network
   natively 243 s / 3.28 GB to 107 s / 0.50 GB cut to its box + 20 km, same figures (operator
   boardings to 2 in 11 million). Makes whole-country demand possible at all, and cuts every
   multi-zone cluster's rounds (the seven cities' Tokyo cluster holds several Kanto zones, each
   searching the whole cluster). Scales: network size. 1-2 days, moderate risk (per-population node
   maps through the load sums, crowding, results).
2. **T-757 a failed demand worker says so**: today the error is only in the console and demand never
   finishes. Small.
3. **T-759 the clock state on a big network**: 360-500 ms of main thread on load, 130-200 ms when
   resent mid-play. Scales: network size. With T-499, T-070.
4. **T-758 the track derive with water masks**: about 4.9 s in WASM for 709 lines (0.42 s natively
   without masks). Load only.
5. **T-760 negative stop-to-stop distance** for stops closer than their platforms (guarded since
   T-737). Correctness, not speed.

## 7. The snapped route's fit (T-754, T-727f, 2026-10-19)

notes/T-727.md "T-754". New York-Emeryville's fit (`trace.rs`, 5,690 corners) natively 779 to 136 ms, in WASM 2.1-2.4 s to
0.27-0.36 s:
the sample sweep was 146 ms a pass, three passes (bounding circles now skip most pieces), and easing
tight curves refitted the whole route 16 times a curve (now 25 corners around it). It runs in steps
of at most 10 ms natively and the router worker drops it for a newer request. The quad edit over the
same route 3.3 s to 0.15 s: one express curve left not fitting kept the express pair from being cut,
so it went in as one 5,400 km edge.
