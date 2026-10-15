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
7. **T-399 one engine per city** (T-361 item 3): 15-30 MB a worker for New York (cellbench: the
   second engine's 12.5 MB plus its access and subzones), the second `set_network` (0.45 s native).
   1-2 days, moderate risk.
8. **T-401 profile one period's round by layer natively**, the basis for T-036: no breakdown
   exists beyond cellbench's free flow (commutes 1.5 s, everyday 0.5-0.8 s). Half a day.
9. **T-281 walk search on opening**: about -0.5 s of load in every worker. 1 day, low risk.
10. **T-397 samples scaled to trips**: a further -60 to -80 MB a worker for New York; opening about
    half as long. 1 day; changes figures slightly.
11. Pack compression on publish (T-060): 43 MB of startup download to about 18-20 MB.

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
