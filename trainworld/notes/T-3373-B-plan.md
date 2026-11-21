# T-3373 stage B: zones and the market split between solving workers (design, for Anita)

Agent splitplan, 2026-11-21. Design only: nothing in app/ or sim/ changed, nothing measured in a browser. Every
figure below comes from measurements already in notes (named where used) or from the code; "est." marks an
estimate, with its method in "Method".

## For Anita, in plain words

- Today each worker copies every city and the longer-trips market, so an extra worker costs 0.4 GB on New York and
  1.3-1.5 GB on all of China and the 2,000-station saves. That is why stage A left those on one worker.
- B gives each city to one worker and the market to one worker. After each crowding round they swap line loads.
- On those big saves a second worker would then cost about 0.2 GB. Measured in stage 4 on all-China: a load on 2
  workers takes 100 s against 147 s on 1 (32% less), for 0.07-0.18 GB more; on 3 workers 77 s. Stage 1 had predicted
  40-50% on all-China and 27-38% on North America; North America is not measured yet.
- New York gains little: one city is three quarters of its work, so it stays copied. B lets it run 3 workers in
  today's memory limit (load about 50 s, from 60-70).
- Yours: A's limit (1.4 GB for all workers on an 8 GB machine) is below one worker on these saves, so it would
  have to rise to about 2 GB. And every trip figure moves once: typically by a ten-millionth or less, at most by
  about a third of a rider on a busy station (measured on New York, stage 2).
- 24-40 hours in 9 stages; the first (2-3 h) can stop the plan cheaply. C (one shared memory) would replace B only
  in browsers that support it, and only B lets a save past 4 GB load. Build B, decide on C after.

## What is copied today, and what B changes

Today (docs/demand-runtime.md "Warm-up and the worker pool", "Clusters"): every solving worker opens every reached
zone's packs, every market pack, composes the market, sets every population on every network, and solves the
periods `periodsOfWorker` gives it (two workers: 0, 3, 4 and 1, 2). A population is one city's commutes
(`DemandApi`), its everyday layer (`EverydayApi`, opened beside the commutes and sharing their cells, so the two go
together), or a cluster's market part (`IntercityApi`, all sharing one composition). Periods share no rider, which
is why workers today never talk during an edit's solve. The one exception is T-1607: with 3 workers a load moves the
evening's market to another worker, and the two sides trade loads through the page every round.

**The unit of dealing is a (population, period) pair.** Today's pool gives every worker every population for some
periods; pure B gives every worker some populations for all periods. The general form is a *holding plan*: each
population is held by one or more workers, and each of its five periods is solved by exactly one of its holders.
Memory per worker is what it holds; time per worker is the sum of its (population, period) costs. Today's dealing is
one such plan (every population held everywhere), and A's is another (one worker). So a planner that picks the best
plan under a memory limit never does worse than today, and it handles the two hard cases with one rule:

- **A dominant city** (New York is 75-78% of its cluster's passes, notes/T-2624.md and T-3512.md): held by two or
  three workers with its periods split, as today, because whole on one worker it would be slower than today's split.
- **The market** cannot be split inside one population and stay exact (notes/T-1607.md), and it is 30% of the passes
  on all of China (T-1511), about 60% of a round on North America (T-2606, inferred), 80% on all of Japan (T-1607).
  It is held once where it is under about 1/n of the work, else copied to a second worker with its periods split
  (a second copy of its composition, about 0.45 GB on the big saves).

The planner: start from one holder per population, dealt longest first by cost to the least loaded worker within
each worker's room; then while the busiest worker is more than 15% over the even share and room allows, add a copy
of its costliest population on the least loaded worker and split that population's periods by `PERIOD_COST`. Costs
are each population's pass time per period from its last solve (the worker already times every pass: `passMs`,
`addPass`), kept in the save header beside `demandFine` so the next load plans from them; before any are known,
the zone's `localTrips` (the stand-in `memoryCheck.ts` already uses) and the market at 40% of a period. Memory per
population: the measured WASM grown by its open (each `opened` message carries `memBytes`), else
`GB_PER_MTRIPS` per million trips and the market at `WASM_BASE_GB`.

### What each worker holds under B

- Its populations' packs (city commutes and everyday layer), set up on the network (`set_network` on the
  population's reach view), and their solved state for the periods it solves.
- If it holds the market: the market packs, the composition (sites, pair rules, commutes between zones, the 34 MB
  corridor figures) and every cluster's market part.
- The network: the whole snapshot (small), the clusters, the graph per period it solves, its own search cache
  (`with_skims`). Fixed per worker as today: about 33 MB WASM floor, up to 32 MB grown ahead (T-1594), the JS
  isolate (`MEM_WORKER` 0.15 GB, T-3472 measured 110-150 MB).
- The clusters must come out the same in every worker, but a worker no longer holds every city whose box joins
  stations (`splitClusters(n, cities, marketZones)`). So the page sends every open population's box and the market's
  zone boxes (from the market holder's `zone_boxes`, a few KB) with the plan, and each worker splits from those.
- Not changed: the aux worker (a commute copy of every open zone for the bubbles, 225-385 MB on big saves,
  T-3917), the focus pool and the zone-in-view aside (one worker holding one zone already). Trimming the aux's
  copies of reached zones, since their holders could answer, is a later idea, not in this plan.

### What crosses between workers, and how big

Per cluster and period, after every round, each holder sends the others its populations' loads summed on the whole
network's route nodes: segments, boardings and alightings, 3 x N values (as T-1607's `loads` message). The final
round's loads also feed the capacity fit, with the riders coming to the platforms (free flow and rounds boardings,
2 x N more). Workers send them to each other directly over `MessageChannel` ports the page sets up, so the main
thread (T-3917's stalls) never handles them.

N, route nodes (2 per line stop, `jobNodes`): New York real save 3,512 (T-842); North America about 5,900 (est.:
day 383 had 4,270 for 1,613 stations, 2.65 a station, times 2,240); asianita about 7,000 (est., same rate); all-China
about 5,200-6,300 (est., 2.2-2.65 a station); all of Japan 20,640 (T-2642). As exact sums (below) each value is 8
bytes.

| | per exchange | per period, each sender (4 crowding rounds + the final + the riders coming) | per solve, 2 workers | per solve, 3 workers |
|---|---|---|---|---|
| New York | 84 KB | 0.5 MB | 5 MB | 14 MB |
| North America, asianita, all-China | 125-170 KB | 0.7-0.95 MB | 7-10 MB | 21-29 MB |
| all of Japan | 495 KB | 2.8 MB | 28 MB | 84 MB |

Plus once per solve a few bytes per population and period (which periods each kept, below), and the results: each
worker sends one result part per period for its populations (today one worker sends the whole), about 41 x N floats
(1 MB on North America), which the page adds. Against solves of 60-450 s this is nothing; the cost that matters is
waiting, below.

### Exact sums: the main design decision

Today one worker adds every population's loads in the cluster's fixed order (`sumAll`, f32), so a network's figures
are the same to the bit however it was loaded (T-392) and at any worker count. T-1607 stays exact only because the
market is one population added last. Once a cluster's cities are spread over workers, each worker's partial sum added
to the others' is a different float order, and the figures would depend on the plan: they would move when the pool
grows or shrinks, when a zone opens elsewhere, or between two players' machines.

**Recommended: add across populations in whole numbers.** Every cross-population sum (the round loads, the
capacity fit's input, the riders coming, the cap ratios, the result's per-node arrays, and the page's `addParts`)
adds each value rounded to a whole number of steps, held in a Float64Array. The step is set per array so its
largest total stays well under 2^53 steps: 2^-20 of a rider for loads and boardings, a finer one for the fare
arrays (`fareX`, `baseX`, in dollars). Whole numbers below 2^53 add
exactly in any order, so the total is the same whatever the order, plan or worker count, then rounded once to f32
for the kernel. Per-city totals (the summary, fare sensitivity) need no re-adding on the page: their terms are
stepped too, so they add the same in any order. Effect, measured in stage 2 on New York only: every figure moves
once. The typical figure moves by 1e-8 to 1e-7 of itself; the largest by 0.375 riders on a station of 14,276
(2.6e-5), and 2.8e-4 on one line figure. Taking today's sums in reverse order moves the same figures by about the
same. After that, the same network gives the same bits at 1, 2 or 3 workers, under any plan. It also makes "the others' loads" exact as total minus mine, which the edit stages use.

Alternative with no figure change at all: a chained fold. The plan deals each cluster's populations in runs of the
registry's order, and every exchange passes the running f32 sum from worker to worker in that order (w0's sum to
w1, which adds its own, and so on), then sends the total back. Bit-identical to today, but it constrains the plan,
puts W-1 hops in every exchange and every result part, and every holder has to take part in each local solve. About
3-5 h more than the recommended way.

## Expected memory and time

### Method

- Solving workers' memory = the sum over workers of (WASM + 0.15 GB JS). WASM = heap held x 1.13 (all-China's
  measured WASM over heap, 1,360 against about 1,200) + 33 MB floor + up to 32 MB ahead.
- Parts of a worker: New York from T-3071 (WASM 378 MB with the market, 193 without: market 185, cities and set-up
  160, of which New York city about 130 by trips). All-China from T-1169 stage 1's heap by part (measured 2026-10-27,
  rail 41.7M a day then, 34.5M now, so used as shares only) less T-1558 and T-1559's measured cuts: market about 40%
  (composition about 0.36 GB, its parts and periods 0.12), cities 60% over 171 zones (Shanghai 30 MB, the top ten
  157 MB). North America and asianita have no breakdown; assumed like all-China's, since the market's composition
  grows with the market packs open (about 2.2 per zone) and they have 160-163 zones. Stage 1 measures them.
- Time: pass shares as above (New York city about 65-75%, its market a third of the passes in T-1511, before T-2652
  made the market's searches about 1.5 times faster, so 25-33%; all-China market 30%; North America market about
  60%, inferred in T-2606). Stage 1 measures all of them. Dealt with `PERIOD_COST`; then +5-15% for waiting at the round exchanges and +5% for stations searched
  twice where two workers' cities reach the same stations (T-1543: a part of a period costs "a little over half").
  Loads also gain because pack opening and set-up are no longer repeated in every worker (all-China: 35 s of a 97 s
  load was opening packs, T-1594; composing China's 370 market packs holds a worker for a minute, T-1533).
- Today's figures: notes/T-3373.md (A), T-3917.md (North America, asianita), T-3868.md (edit stage 200-210 s).

### Table (solving workers' memory in GB; times to the demand final)

The design's estimates, from older notes. Stage 1's predictions, made from measured shares, are in "Stage 1 results"
below and take precedence.

| Save | Today, A's pool | Today, 2 workers (periods) | B, 2 workers | B, 3 workers |
|---|---|---|---|---|
| New York real save (one city 3/4 of the work) | 2 workers: 1.06 GB; load 60-72 s | (that is A's) | 0.90 GB est.; load about the same | 1.25 GB est., inside 1.4 GB; load about 47-54 s (today's 3-worker time) |
| all-China (t1156) | 1 worker: 1.51 GB; load 124-142 s | 2.9 GB; 91 s | 1.75-1.8 GB est.; load 70-85 s est. | 1.9-2.0 GB est.; 55-70 s est. |
| North America (na-day-491) | 1 worker: about 1.5 GB; load 333 s, edit 296 s (T-3868: 236-245, 200-210) | +1.3 GB; load 226 s, edit 188 s | about 1.7 GB est.; load and edit about today's 2-worker (200-230 s, 180-200 s) | about 2.3-2.4 GB est. (a second market copy); load and edit about 30% under B's 2 workers |
| asianita (t3600) | 1 worker: about 1.64 GB; load 455 s | +1.4 GB; 248 s (indicative) | about 1.8 GB est.; about 250-280 s est. | about 2.5 GB est. |

The tab's other memory (map, GPU, page, clock and aux workers: 2.6-2.8 GB on the big saves) is unchanged. So the
big saves would settle at about 4.4-4.5 GB with 2 workers, against 4.2-4.3 at 1 and 5.5-5.7 at 2 today.

What decides North America: if its market is near 60% of the passes, B's second worker is the market alone plus a
few cities, and the speed is today's 2-worker speed; only a third worker (with a second market copy) goes further.
If stage 1 finds it nearer all-China's 30%, 2 workers come out about 15% faster than today's 2.

## Stages (hours of agent time; each lands on its own)

| # | Stage | Hours | Lands as | Check, and stop rule |
|---|---|---|---|---|
| 1 | **Measure and simulate.** Per population: WASM grown at its open (`opened` memBytes deltas, one worker, `?lod=fine` off), pass time per period (`passMs`), the market's share, on New York, all-China, North America and asianita. The unsplit build loaded twice on all-China and New York, to see whether its hashes repeat (T-1354, T-1594: they may not). The planner as a pure function with unit tests, run offline on those numbers for 2 and 3 workers. | 2-3 | the planner module, unused; numbers in notes | Stop and report if all-China and North America at 2 workers aren't predicted at least 25% faster than 1 worker for at most 0.4 GB more. If the baseline's hashes don't repeat, later "equal to the bit" checks compare the sections that do (rail, the market) and name the rest. |
| 2 | **Exact sums.** Whole-number sums in every cross-population sum in the worker and in `addParts`; per-city totals re-added on the page. | 2-4 | off by default (`?exactSum=1`) until stage 4 passes and she has OK'd the rounding change | Same bits at 1, 2 and 3 workers and with `?perfOff=share`; against master, the largest change of any figure reported (expected about 1e-6). |
| 3 | **Who opens what, behind `?split=1`.** The plan on the page; every rule that now means "open in every worker" made to mean "open in its holders": a solve sent once every city is open (the per-item opened sets, about line 889 and 1299 of `demandClient.ts`), the `open` posts (about 1811), the first-station prefetch (`prefetchViewed`), the warm-up hold (`held`), the reopen at another level (`releveling`), closes (`sweep`, `closeStrays`), the pack hashes, the loading card's progress by worker acks, a replacement worker's `reopen`, the watchdog's replace-all (T-3781), `shrink`. Boxes for the clusters sent with the plan. | 3-5 | off by default | With `?split=1` and the plan forced to "everything everywhere", everything as today (figures equal); with a real plan, each pack opened on its holders only (load log), the load reaches its solve. |
| 4 | **B for whole solves.** Done (splitb4, below): the rough solve on the zone in view's holder, `capAhead` and `capPool` counting the plan at a 25% share, so all-China gets 2 workers and New York 3 without `demandWorkers`. The ports were measured and not built: the page's relay cost 4 ms of waiting in a 97 s load. | 2-4 | under the flag | Passed: figures equal to the bit on New York and all-China; all-China at 2 workers loads 32% faster than 1 worker for 0.07 GB more. |
| 5 | **Questions.** 20 query kinds and `rows`, sorted into four groups (provisional: this plan read about half the handlers, the stage reads each). A zone's own questions go to its holders (subSums, cellModes, purposes, flowRail, flows, purposeModes, trip, tripLengths, cells). The station-wide ones go to every worker, whose answers add (station, transfers). The market's go to its holder (icStation, icTrip, airLegs, regLegs, rows, zoneSquares), and icSel gets the zone's squares from the zone's holder, as bare zones do now (`bareSquares`). The aux worker's (bare, bareFlows, bareSquares) stay as they are. Answers record which populations they cover; today they record a period mask. | 3-5 | under the flag | Each kind answered equal (to rounding) split against not, on New York and North America: a station, a bubble selection, a trip, transfers. |
| 6 | **Edits.** Kept periods: holders agree a cluster's period is kept (every population kept it). The local solve runs on the holders of the touched populations only, crowding on the last whole solve's total minus their own (exact), so other workers stay free. The market window runs on the market's holder. Then the local refine and T-3925's restart, and T-2624's per-period results arriving as one part per holder. | 4-6 | under the flag | North America Line 6 +1 and a station: local result within today's local's spread, final equal to the bit; paired runs on and off for timing. |
| 7 | **Pool size and memory rules.** `workersFor`, `capAhead`, `capPool` and `growTo` count the plan's per-worker needs. Memory restarts (T-1205) and running out of memory (T-3072) deal the plan again. A grown worker opens only its share. The detail budget (`demandLod`) stays a total for all zones; the 3.5 GB refusal applies to the largest worker; the saves card and `memoryNeed` follow the plan. The pool is decided before packs open, since WASM never shrinks and a late re-deal leaves the old high water. | 2-4 | under the flag | all-China and North America on a reported 8 GB pick 2 workers at the new limit, New York 3. A forced out-of-memory still finishes with the same figures. |
| 8 | **Copies and re-dealing.** A dominant city held by several workers with its periods split; the market copied when over 1/n of the work. A zone opened later goes to the least loaded worker. A re-deal happens only between solves, when the busiest worker is 25% over the even share. Per-population costs are saved in the save header. | 2-4 | under the flag | New York at 2 and 3 workers no slower than `?split=0` in paired runs; a save reloaded plans from its saved costs. |
| 9 | **On by default**, exact sums with it. `?perfOff=split` keeps the old pool for a release; smoke's expected figures updated once for the rounding. Docs: demand-runtime.md's pool, "A market set apart" (now one case of the plan) and "One period in parts" (B gives every worker its own populations' free flow, so parts stay only for a population with several holders). DECISIONS, a changelog line. | 2-3 | default on | smoke; the batch checker on master. |

Total 24-40 h. Checked against past work: T-3925 was guessed at 2-4 h and went from filed to landed in about 2 h
(2026-11-21); stage A was guessed at 6-10 h and landed the day she chose it (2026-11-16); T-1607, the nearest relative
(a worker-to-worker load exchange, result parts, `forget_period`), landed in a day alongside T-1609 and T-1733. So
the low end is likelier, about 24-30 h. That fits the 25-40 h first given for B. The likeliest overrun is stage 3:
the page's "open in every worker" assumption runs through about twenty places in `demandClient.ts`.

Order: after T-2624 stage 2. Both change the worker's round loop (`step`), `figures`, the page's `receive` and
`addParts`. Stage 2 here (exact sums) touches only the sums and could go before T-2624 if the tree is free.

## Stage 1 results (agent splitb1, 2026-11-21)

**Verdict: go on, uncalibrated.** The stop rule was at least 25% faster than 1 worker for at most 0.4 GB more,
at 2 workers with the market held once.

- All-China: loads 41-48% faster, edits 40-50% faster, about 0.2 GB more.
- North America is the thin one. Loads come out 27-29% faster with the plan's planner, and 36-38% once the planner
  also counts opening time (a step added after that first figure). Its opening figures come from a 2-core run.
  Edits come out 32-34% faster, or about 26-28% if the market alone on a worker runs 10% slower. About 0.2 GB more.

What the terms mean:
- **Load**: from the import to the whole solve's last result.
- **Edit**: that solve without the opening. A train edit on these saves re-solves every period of the one big cluster.
  No edit was run.
- **Memory**: the solving workers' WASM plus 0.15 GB of JS each.

Every prediction assumes two workers run side by side as fast as one alone. The one 2-worker load this session (on a
fully loaded machine) ran each worker 1.7-1.8 times slower. So the figures are checked only against the gains
earlier sessions measured for today's 2-worker pool, and stage 4's measured stop rule is the real test.

**How measured.** My Vite (6150), headless Chrome pinned to 2-5 cores claimed from the ledger, `?new=1&paused=1`,
explicit `demandWorkers`, the save imported. The machine was busy throughout: the ledger at 8-14 of 14 claimed and
80-100% CPU, so wall times are slow and noisy, and shares within one run are what the prediction uses. Each solving
worker's `opened` and `done` messages were recorded (open ms, live heap) through a listener the init script added. A
dev-only scratch hook in `demand.worker.ts` gave each population's pass ms per period and round (`passByRound`) and
its cluster, read over CDP after the final. The hook is reverted; it is kept in the scratchpad as
`splitb1/probe-hook.ts.txt` for stages 4 and 8. Figures were hashed by section with `t328-cache-check`'s `__sig`.
Scripts: scratchpad `splitb1/measure.mjs`, `sim.mjs` (the prediction), `summary.mjs`, `sigs.mjs`; per-run JSON
`run-<name>.json`.

**Measured on 1 worker** (load s; the solve is set-up + passes + the rest; heap MB at the end):

| Save | Load | Opening on the worker | Solve | Market's share of passes | Heap: cities / market | WASM | Above live heap at the peak |
|---|---|---|---|---|---|---|---|
| all-China, 2 runs | 229, 230 | 71-73 s, market composing 13-15 s of it | 143-146 = 18-21 + 96-102 + 26 | 26-28% (Shanghai 15%) | 674 / 510 | 1,299 | 108 with the market, 12 cities alone |
| North America (2 cores) | 464 | 120 s (cities alone on 4 cores: 63 s), market 14.5 | 331 = 34 + 278 + 19 | 65% (New York 16%) | 670 / 474 | 1,325 | 163 with the market, 18 cities alone |
| New York, 2 runs (2 cores) | 177, 208 | 16-18 s, market 7-10 | 160-188 = 3-4 + 149-169 + 7-16 | 23% (New York city 76-77%) | 117 / 151 | 365 | 82 |
| asianita | 441 | 60 s, market 11 | 374 = 18 + 319 + 38 | 61% (Tokyo 14%) | 572 / 662 (split by pass share, no run without the market) | 1,489 | 247 |

- Opening, not downloading, holds a load up: on all-China every pack was downloaded by 44 s and the worker opened
  until 87 s. Each worker opens every pack today, so 2 workers today wait just as long. Under B each worker opens its
  own share.
- Each save is one cluster: all-China 170 of its 171 zones with the market, North America 124 of 129, New York all
  4, asianita 120 of 123. So every round is a meeting point of every worker.
- 2 workers in this session were slower than 1: all-China took 317 s against 230 on 5 cores, each worker's opening
  and passes about 1.7-1.8 times slower, with the machine at full load. So no calibration run was possible.
  Instead, the model's figures for today's 2-worker pool, made from the 1-worker shares, were checked against the
  earlier sessions' measured gains. All-China: 23-28% predicted, 25-36% measured (T-3373 A). North America: 29-31%
  predicted, 32% measured (T-3917). asianita: 33-37% predicted, 45% indicative. Memory came out at 2.84 GB predicted
  against 2.74 GB measured in this session's 2-worker all-China run.

**Predicted** (`sim.mjs`). The planner's plan, with each worker's opening as its units' open ms. Passes use the
load's round barrier (each round as long as its slowest worker) for the high end and perfect interleaving for the
low end. The rest of the solve is unchanged (high end) or shrinks with the busiest worker's share (low end). The
market is held once.

| Save | 1 worker | Today, 2 workers | B, 2 workers | B, 3 workers |
|---|---|---|---|---|
| all-China | 230 s; 1.42 GB | 164-176 s; 2.84 GB | load 118-136 s (41-48% faster), edit 40-50% faster; 1.59 GB | load 85-105 s; 1.75 GB |
| North America | 464 s; 1.44 GB | 322-330 s; 2.89 GB | load 289-296 s (36-38%), edit 32-34%; 1.63 GB | held once: 264-270 s, 1.81 GB; market copied on all 3: 193-205 s, 3.02 GB |
| New York | 177-208 s; 0.51 GB | 104-108 s; 1.01 GB | the planner keeps today's pool (faster than any split) | load 77-100 s (today's 3: 95-117 s); 1.20 GB, inside 1.4 GB |
| asianita | 441 s; 1.60 GB | 277-294 s; 3.21 GB | load 267-281 s (36-39%), edit 35-38%; 2.00 GB est. (1.77 if a cities-only worker grows as little past its live heap as China's and North America's) | held once: 257-271 s, 2.40 GB est. |

**Hashes repeat.** New York's two 1-worker loads match to the bit in every section. All-China's two match in every
section but the per-zone figures. There, one load had one more zone shown (176 against 175, 10 more numbers);
`t328-cache-check` already leaves that section out, and the other zones were not compared one by one. The 2-worker
all-China load matched the rest too. Both imports composed the market once from all 370 packs, so pack order never
varied, and T-1354 is untested here, not ruled out. Stage 4's "equal to the bit" can compare every section but the
per-zone one, with per-zone hashes added to tell a zone shown once from a zone that differs.

**What changes for the later stages**

- The planner (`app/src/workers/demandPlan.ts`, unused; tests `checks/t3373-plan.mjs`, 23 pass) also weighs each
  unit's opening, since a load waits for the slowest worker's opening. Unit by unit, it moves zones held once where
  that shortens the slowest opening plus the busiest solve. On North America that takes the load from 27-29% to
  36-38% faster: the market's worker takes the small zones whose opening outweighs their passes. Stage 3 sends the
  open ms (the `opened` message has it) and stage 8 keeps it in the save header beside the pass costs.
- Memory: a cities-only worker grows only 12-18 MB past its live heap, and the market's worker carries the 108-163
  MB peak. `sim.mjs` gives B's second worker 0.17-0.19 GB on all-China and North America. That leaves out a worker's
  own floor: an idle focus worker holds 33 MB of WASM, and the no-market worker sat 31 MB over its live heap where
  the model adds 19. Call it about 0.2 GB, close to the plan's 0.25.
- North America and asianita are market-heavy (61-65% of passes). With the market held once, 2 workers can take at
  most about 35% off their passes, and edits come out 32-38% faster. The stage-1 data cannot show the risk to this:
  the market alone on a worker might be slower than its share on one worker. A 10% loss would put North America's
  edit at about 26-28% faster. Stage 4 measures it. A second market copy (+0.6 GB) is worth it only at 3 workers.
- Not tested here:
  - the plan's +5% for stations searched twice
  - the time to exchange loads between rounds
  - whether opening in parallel splits evenly. In the 2-worker run each worker's open ms went from 71 s to 118 s,
    and T-173 describes the same kind of effect. The load figures depend on this; the edit figures don't.
- Hours are unchanged: 24-40 for the 9 stages. This stage took about 1.5 h.

**Stage A's memory limit: left as it is.** The raise is one line (`SOLVING_SHARE` in `demandClient.ts`, 0.175 to
0.25), and it was not made. Raising A's share to 25% (2 GB on a reported 8 GB) before B splits anything changes
only New York-sized saves. They go back to 3 workers (0.53 GB each, 1.6 GB together, +0.5 GB), and
all-China, North America and asianita stay at 1 (1.45-1.64 GB a worker). The raise belongs in stage 7, where the
limit counts the plan's per-worker needs.

## Stage 2 results (agent splitb1, tree splitb2, 2026-11-21)

**Built, off by default.** `?exactSum=1` turns it on for a tab (`exactSum.ts` `SUM_FLAGS`, sent in every solving
and focus worker's `init`). It covers every sum across populations in the worker:
- the round loads and the T-1607 sides' exchange (`loads` carries f64 whole steps; `bothSides`)
- the riders coming (`comeRaw`) and the capacity fit's inputs
- a result's per-node arrays (`seg`, `board`, `alight`, `loadOfCrush`, `fareX`, `baseX`, `xferIn`), its purpose
  blocks (`purposeParts`) and the `extra` and `delta` added to them
- the summary's slots 1, 2, 3 and 5, and `fareRaw`'s terms
- the local solve's frozen loads (`frozenAt`, the cap ratios, the window's `less`)
- the stored partner loads (`inbox`, `sharedMkt`, `sharedTrail`, `remapKept`)

On the page, `addParts` adds parts from their unrounded sums (`exact`, f64) and rounds once. Steps:
- 2^-20 of a rider for loads, boardings, transfers and summaries
- 2^-24 of a dollar for the fare extras
- 2^-32 for `loadOfCrush` (a share of crush capacity)

Every term is rounded to its step and added in an f64, which is exact while a total stays under 2^53 steps. That
is 8.6 billion riders, 537 million dollars, or 2 million units of crush at a route node. Per-city totals did not
need re-adding on the page: stepped terms add the same in any order.

Left as they were:
- `marketFigures` (the market's parts, one unit held whole)
- the question answers (stage 5)

With the flag off the figures are unchanged, but the code is not. The summing loops call `termOf` per element, and
the local solve's lines changed. No timing was taken either way.

**Checks** (New York real save, imported, `?new=1`, my Vite on 6150; the machine at full load, runs on 1-3 cores):

| Run | Workers | Hash | Note |
|---|---|---|---|
| flag off | 1 | 5657f014 | same as stage 1's on the earlier master: the flag-off path is unchanged |
| flag off | 3 | 5657f014 | today's T-1607 sharing (8 `loads` messages, 8 part results) still matches 1 worker |
| flag off, populations in reverse (`?sumOrder=reverse`, dev) | 1 | e5dbc9e0 | the control: today's sums depend on the order |
| `exactSum=1` | 1, 2 and 3 | 7c35251f | 3 workers shared period 3: 8 `loads` messages, all f64, and 8 part results, all with `exact` |
| `exactSum=1&perfOff=share` | 3 | 7c35251f | no sharing |
| `exactSum=1`, in reverse | 1 | 7c35251f | the order no longer matters |

- **Every array of the final period results**, which the hash leaves out, was dumped too: `seg`, `board`,
  `alight`, `arrive`, `loadOfCrush`, `fareX`, `baseX`, `xferIn`, the purpose blocks, `lean`, the summaries and
  `fareSense`. With `exactSum=1` all 13 match to the bit at 1 worker, in reverse and at 3 workers sharing. Against
  flag off they move by at most 0.375 riders (`xferIn` 0.374 at 372, 1.7e-3 relative), `loadOfCrush` by 1e-5,
  `fareX` by 4e-6 dollars, `fareSense` by 1e-8 relative. `baseX` is all zero on this save. The fare-sensitivity
  terms run to about 4.9 million, far inside the step's room.
- `checks/t3373-sums.mjs` (Node, 6 pass): 300 random populations give the same bits in 20 shuffled orders and split
  over 1-5 workers. The plain f32 control differs between orders.
- **How far figures move**, flag on against off at 1 worker:
  - The day's rail trips: 13,252,827.509 against 13,252,827.507.
  - Every other number: median 2e-8 to 8e-8 relative among values of 10 or more.
  - Largest: 0.375 riders on a station of 14,276 (2.6e-5 relative), and 2.8e-4 on one line figure of 1,340.
  - Today's own sums taken in reverse order move the same figures by the same amounts (0.374 riders, 2.8e-4).
    The reversed run and the exact run agree with each other within 0.012 riders on every line and 0.009 at that
    station. So both small changes take the same jump there; what makes it was not looked into. The jump is
    larger than the plan's "about 1e-6" for the worst figure; the typical figure is well under it. New York only:
    no big save was compared.
- **Not covered by these checks:**
  - T-1543's preview parts: no run produced one with `exact`
  - the local solve after an edit and T-1609's refine: no edit was run; their result is an interim one that the
    edit's whole solve replaces
  - the bubbles' subzone sums the page adds across workers: per period, not across populations; stage 5's
  - the questions (stations, transfers): stage 5's
- No Rust change, so no cargo suites. Typecheck clean; smoke 30/30 with the flag off.
- Left for stage 9: the docs (`demand-runtime.md`'s sums) and a DECISIONS line for the step sizes.

## Stage 3 results (agent splitb3, tree splitb3, 2026-11-21)

**Built, off by default, and further than the row asked.** `?split=1` gives each zone and the market one solving
worker, and those workers now trade each round's loads, so a split load already reaches the same figures. Stage 3
was meant to stop at "opens on holders only"; the exchange at the end of a round was part of stage 4. With exact sums
the general form was small, so it went in here. `?split=all` runs the same holder code with every pack on every
worker (today's pool).

**On the page** (`demandClient.ts`; `Holding` in `demandPlan.ts`):
- One rule, `holdersOf(item)`: an item's solving holder plus the aux worker. Flag off or `split=all`, it is every
  worker, so nothing else changed there. Read by:
  - the `open` posts and the T-173 warm-up hold
  - acks and "open everywhere": a solve is sent once every holder has acked
  - the loading card's progress by worker acks (`packAcks` over the item's holders)
  - a replacement's `reopen` (only what it holds), so the watchdog's replace-all and `restartForSave` reopen by holding too
  - `shrink`'s ack recount
- `prefetchViewed`, the reopen at another level (`lodApply`), `closeStrays` and the sweep need no change of their
  own: they go through `fetchCity` and `closeFull`.
  - Closes still go to every worker. A worker not holding a pack ignores its close.
  - The pack hashes are unchanged: they are of the bytes, whoever opens them.
- **Dealing** (`Holding`): a zone is placed when it is first asked for (`fetchCity`, so in the order the page asks
  for zones, not the order downloads land) and keeps its holder for good.
  - The market goes on the last worker.
  - A zone goes to the worker it adds least to, by its `localTrips`.
  - The market counts as 40% of the work from the start (stage 1 measured 23-65%). The first run counted it only
    once its packs arrived, and its worker got 93 of China's 171 zones on top of the market.
- **Shrink** (`memoryShort`, `capAhead`, `capPool`, `restartForSave`): units the stopped workers held are placed
  again on the survivors and downloaded again (the browser's cache). The next solve waits for them. `growTo`
  never grows under the split (stage 7 deals to new workers).
- **Every solve under the split is a barrier solve, whole.** No local stage, no kept periods or clusters, no rough
  solve, no T-1543 parts, no T-1607 market sharing. Every solving worker gets every period; the aux worker one group.
- **The solve message** carries the plan (`split`): each zone the solve covers, in registry order, with its cells'
  box and its holder; the market's holder; the workers holding anything; and the worker's own index.
  - Boxes and each zone's everyday purpose count come with each `opened` (`boxes`, `kinds`).
  - The market's 6 km zone boxes come from its holder at each solve's start (`zones`, relayed by the page), since
    a deferred close can change the composition after the page last heard.

**In the worker** (`demand.worker.ts`):
- A worker solves only the zones the plan gives it (`by === me`), and the market only if it is its holder.
- Clusters are joined from every zone of the plan and the holder's market zones, so all workers split alike.
- A cluster's period goes round by round as before (side `"held"`). After each round a worker sends its own loads to
  the cluster's other holders (`loads` with `from`, `to`, relayed through the page). It then crowds on the exact sum
  of its own and theirs.
- The capacity fit and the riders coming (`come`) are added up the same way.
- Each holder sends a `result` part for every period and round. The page adds them (`addParts`, exact).
- `alignParts` (`demandSums.ts`) lays each part's purpose blocks out on the union of their keys: a worker holding no
  market has no longer-trips block. Each worker also makes as many everyday blocks as the clusters' zones have on
  any worker (`planKinds`), so the market's outings across a zone line go to their purposes as with one worker. A
  first 3-worker run, whose market worker held no zone, put them in the longer trips: the purposes section differed
  (every other section matched).

**Checks** (my Vite 6221, save imported into `?new=1`, `demandWorkers=2` unless the row says 3; `measure.mjs`, `fields.mjs` and
`opens.mjs` in scratchpad `splitb3/`). Each run has a fresh profile, so every hash is from a fresh final, not a
cached result.

| Run | Hash | Final | WASM per solving worker (aux) |
|---|---|---|---|
| NYC flag off, master | 5657f014 | 139 s | 333, 333 (65) |
| NYC flag off, this tree | 5657f014 | 119 s | 333, 333 (65) |
| NYC `exactSum=1`, master | 7c35251f | 161 s | 333, 333 (65) |
| NYC `exactSum=1&split=all` | 7c35251f | 146 s | 333, 333 (65) |
| NYC `split=1` | 7c35251f | 121 s | 161 (New York City), 237 (the market, New Haven, Hartford, Long Island) (65) |
| NYC `split=1`, after the purposes fix | 7c35251f | 121 s | 161, 237 (65) |
| NYC `split=1`, 3 workers | 7c35251f | 124 s | 161 (New York City), 33 (the 3 small zones), 237 (the market alone) (65) |
| NYC `split=1`, worker 1 out of memory 15 s into the load | 7c35251f | 159 s, on 1 worker | 365 (65) |
| NYC flag off, worker 1 out of memory 15 s into the load | 5657f014 | 198 s, on 1 worker | 365 (65) |
| all-China `exactSum=1`, master | 90310524 | 170 s | 1,267, 1,235 (289) |
| all-China `split=1`, first dealing (market counted late) | 90310524 | 127 s | 353, 979 (289) |
| all-China `split=1` | 90310524 | 112 s | 577 (136 zones), 755 (the market's 370 packs and 35 zones) (289) |
| all-China `split=1`, 3 workers, after the purposes fix | 90310524 | 118 s | 353, 353, 627 (the market alone) (289) |

- **Every array of the final results matches to the bit** on both saves: `seg`, `board`, `alight`, `arrive`,
  `loadOfCrush`, `fareX`, `baseX`, `xferIn`, the purpose blocks, `lean`, the summaries and `fareSense`.
  The hash covers every section, per-zone figures included.
- **Each pack opened on its holders only**, by the workers' own `opened` messages:
  - NYC: worker 0 held New York City alone; worker 1 the other 3 zones and the 9 market packs.
  - all-China: worker 0 held 136 zones and no market pack; worker 1 held 35 zones and 370 market packs.
  - The aux worker still holds every zone, as planned.
- **Memory**, solving workers' WASM at 2 workers: NYC 666 MB to 398 MB; all-China 2,502 MB to 1,332 MB. One worker
  alone holds 365 MB on NYC and about 1,300 MB on all-China, so a second worker now adds about 33 MB of WASM on
  each (plus its JS heap, about 0.15 GB), where it added 300 MB and 1,200 MB. Load times are from a machine at 3-10 of 14 cores
  claimed, so only roughly comparable; stage 4 times them properly.
- **Edits** (play.mjs, new NYC game, `demandWorkers=2`, two three-station lines one after the other): under
  `split=1` each edit solve reaches its final. The second line's rail trips equal the unsplit `exactSum=1` game's to
  every digit (483,720.38901925).
- **T-3933 with the flag on** (oomretry's `inpage.mjs` with `split=1`, real save through Saved games):
  - the failed grow in worker 1 halved the pool (2 to 1, the units moved and reopened)
  - worker 0's two failures saved the game and reloaded it into the retry on 1 worker, with the note
  - the final came with 12,886,431 rail trips a day, the T-3933 figure
- `npm run check` clean. `checks/t3373-plan.mjs` 26 pass, 3 of them new for `Holding`. No Rust change. Smoke with
  the flag off: 29 of 30. The one failure is the wiki figures check (`wiki/read/brt.md` out of date), in a tree with
  no wiki change.

**Not done, for the next stages:**
- **Questions (stage 5).** Station, transfers, trips and the market's questions still go out as if periods were
  dealt by worker. Under the split each worker answers for its own zones over all five periods, so a panel may
  show one worker's part or count a period twice. Not checked.
- **Edits (stage 6).** Every edit re-solves every cluster whole, with no local result. That is slower than today
  after an edit, and fine behind a flag.
- **Pool rules (stage 7).** `capAhead` still counts a whole save per worker, so all-China and North America get 1
  worker under the split unless `demandWorkers` is set. `growTo` is off under the split.
- **Ports (stage 4's other half).** Loads still go through the page.
  - NYC per round: 84 KB per period per worker.
  - All of Japan: about 0.5 MB.
  - The T-3917 stalls on the main thread are what ports would remove; not measured.
- **Load timing (stage 4).** The stop rule (at least 20% faster than 1 worker, at most 0.4 GB more) needs paired
  runs on a quiet machine. Memory already passes: all-China's second worker adds about 0.05 GB of WASM against one
  worker (1,332 against about 1,300), plus the JS heap of 0.15 GB.
- Copies and measured costs (stage 8).

## Stage 4 results (agent splitb4, tree splitb4, 2026-11-21)

**Stop rule passed: go on to stage 5.** All-China at 2 workers under `?split=1` loads 31-32% faster than 1 worker
(the rule asked at least 20%; stage 1 predicted 41-48%) and holds 0.07 GB more as measured, 0.18 GB more by the
plan's own count (at most 0.4 GB allowed). Every run below reached the same figures as the unsplit build with exact
sums.

**Built, all under `?split=1`** (`demandClient.ts`, `memoryCheck.ts`):
- **The pool counts the plan.** Under the split a further solving worker adds `SPLIT_EXTRA` (0.2 GB) to what one
  worker takes alone, where today it adds a whole copy. The solving workers may then take `SPLIT_SHARE` of the budget
  (25%, 2 GB on a reported 8 GB) instead of 17.5%: Anita's "the memory limit rises with it" (11-21).
  - `poolBytes` and `poolIn` hold the rule. `workersFor`, `capAhead` (with "Use more cores" too) and `memoryNeed`'s
    total use it.
  - `capPool` adds the workers' needs under the split: a cut moves the stopped worker's units to the others, so it
    saves only that worker's JS heap.
  - Flag off, every count is unchanged: the same products as before.
  - With no `demandWorkers`, all-China's 171 zones come out at about 1,733 MB a worker by `memoryNeed`. That gives
    1.93 GB for 2 workers and 2.13 GB for 3, so the pool of 3 (16 threads) is cut to 2 before the packs open. No later
    cut (`capPool`) came. New York keeps 3. Flag off, all-China is still cut to 1.
- **The rough solve runs on the zone in view's holder.** The solve goes to every worker with the plan, as whole
  solves do. The holder solves the zone alone, with no market, and sends whole results. The others drop what they were
  solving and stay idle until the whole solve supersedes the rough one. Its zone is never open on them, so
  `roughReady` stays false. `Solve.lead` says which worker's `plan` names what is being solved; before, it was always
  worker 0's.
- **Ports: not built.** Measured through the page on all-China at 2 workers, with a scratch probe (kept as
  scratchpad `splitb4/probe.diff`):
  - 80 relays of 8.7 MB in all took 3 ms of the main thread in a 97 s load.
  - Of 120 waits for the other holder's loads, 10 blocked, for 1.5 s together. Only 4 ms of that came after the
    partner had sent; the rest was the partner still solving.
  - A port would save at most those 4 ms. A message sitting in the receiving worker's queue while it is in a pass
    is the same with a port.
  - Not measured in play, where T-3917 found 6-11 main-thread stalls a minute over 200 ms. Stage 6 measures the
    edits, and builds the ports only if the relay shows there.

**Checks** (my Vite 6181; save imported into `?new=1`; Chrome pinned to 6 cores from the ledger, other agents holding
0-8 of 14 cores; each run a fresh profile. Scripts: scratchpad `splitb4/measure.mjs`, `batch.ps1`, `timing.mjs`,
`zones.mjs`; per-run JSON `run-<name>.json`.)

| Save | Run | Load to the final | Solving WASM + demand JS | Hash |
|---|---|---|---|---|
| all-China | 1 worker, flag off (as players have it) | 148, 146 s | 1,365 MB | dcbcd9d0, f44ff9fd |
| all-China | 1 worker, `exactSum=1` | 143, 148 s (and 176 s, the session's second run, left out) | 1,364 MB | 4f6b3013, 90310524 |
| all-China | 2 workers, `split=1` | 98, 105, 96 s, mean 100 (and 97 with the probe) | 1,432 MB (WASM 577 + 755) | 90310524 (3 runs), 4f6b3013 |
| all-China | 3 workers, `split=1`, one run | 77 s | 1,461 MB (WASM 353 + 353 + 627) | 90310524 |
| all-China | 5 workers, `split=1`, one run | 78 s | 1,367 MB of WASM (161, 193, 193, 193, 627) | 90310524 |
| all-China | no `demandWorkers`, `split=1` | 108 s, on 2 (3 cut to 2 before the packs opened) | 1,442 MB | 90310524 |
| all-China | no `demandWorkers`, flag off | 146 s, on 1 (as on master) | 1,366 MB | f44ff9fd |
| all-China | 2 workers, `split=1&perfOff=focus` | 104 s | 1,332 MB of WASM | 90310524 |
| New York | 2 and 3 workers, `split=1` | 95 s each | WASM 161 + 237; 161 + 33 + 237 | 7c35251f |
| New York | no `demandWorkers`, `split=1` | 98 s, on 3 | WASM 431 MB | 7c35251f |
| New York | 2 workers, flag off | 95 s | WASM 333 + 333 | 5657f014 |
| New York | 2 workers, `split=1&perfOff=focus`, market packs held 40 s | 126 s | WASM 161 + 237 | 7c35251f |

- **The stop rule's figures.** 2 workers against 1 with the flag off, the build players have: 100 s against 147 s,
  32% less. Against 1 worker with exact sums (145 s), 31% less. The worst pairing, the slowest split run (105 s)
  against the fastest 1-worker run (143 s), is 27% less, and the run with no `demandWorkers` (108 s) 25%.
- **Memory, two ways.** Measured: the solving workers' WASM plus the demand workers' V8 heaps over CDP
  (`Runtime.getHeapUsage`, 29-36 MB a worker), 1,432 MB against 1,365 MB, 67 MB more. That heap leaves out each
  worker's compiled WASM code and isolate, so by the plan's own count (WASM plus 0.15 GB of JS a worker) it is
  1,332 + 300 against 1,299 + 150 MB, 0.18 GB more. Both are under the rule's 0.4 GB. Which figure a worker's JS
  really costs is T-3472's measure (110-150 MB, taken another way), not settled here.
- **3 workers loaded 23% faster than 2** (77 s against 100 s, one run) for 0.03 GB more WASM and a third worker's JS.
  The rule gives all-China 2 because `memoryNeed`'s estimate (1,733 MB a worker) is about 280 MB above what one
  worker takes (1,299 MB of WASM and its JS). Stage 7 sets the pool's real rules and should weigh this.
- **Figures.** Every section matches the unsplit build: view, lines, stations, purposes, the market and the corridor
  tables. The per-zone figures of all 174 zones shown in every run match zone by zone (`zones.mjs`). Two zones
  (`bxx`, `yas`) were shown in some runs and not others, at 1 worker and split alike, flag on and off. That is
  stage 1's "one more zone shown", and it changes the whole hash (90310524 against 4f6b3013 with exact sums,
  f44ff9fd against dcbcd9d0 with the flag off). So a 1-worker run and a split run match to the bit both ways: with
  the two zones shown (4f6b3013, 1 worker and 2 split) and without (90310524). New York matched to the bit in every run.
- **The rough solve ran.** New York, `perfOff=focus` (no focus pool), with every market pack's download held 40 s so
  the whole solve had to wait. Worker 0, New York City's holder, sent round 0 of all five periods at 9.7 s and round 1
  by 33 s. Worker 1 sent nothing for it. The whole solve went at 44 s and reached 7c35251f. Without the delay, the
  whole solve superseded the rough one 2.6 s after it was sent, before its round 0. On all-China with
  `perfOff=focus` no rough solve was sent at all.
- `npm run check` clean; `checks/t3373-plan.mjs` 26 pass. Smoke, flag off: 30 of 30.

**Not done, for the later stages:**
- The saves card's memory figure follows `memoryNeed`'s total, now counted by the plan under the split. Its
  per-worker refusal (3.5 GB) still judges one worker holding everything (stage 7).
- `growTo` stays off under the split (stage 7).
- `memoryNeed`'s estimate is above what was measured, which keeps all-China at 2 where 3 ran faster; see
  "3 workers" above (stage 7).

## Risks

- **Checks that can't be read.** T-1354 (open: the market's corridor figures depend on the order its packs
  arrive) and T-1594 (an all-China load's signature differed between runs of one build) mean today's figures may
  not repeat to the bit. B changes which worker opens which pack, and when, so a mismatch could be either. Stage 1
  runs the unsplit build twice first; if T-1354's order effect shows up, fixing it comes before stage 4.
- **Results changing.** Once, by rounding, when exact sums go on (yours to OK, or ask for the chained fold). After that, the
  check that matters is that figures don't depend on the plan: a dev `?splitSeed=` that deals at random, with the
  hash the same across seeds. A mis-routed question (stage 5) is the quiet failure: a panel showing one worker's
  part of a station's riders. Each kind gets a split-against-not comparison.
- **Stage A's worker count.** The 17.5% share (1.4 GB on a reported 8 GB) is below one worker on every big save
  (1.44-1.64 GB). B makes a second worker cost about 0.25 GB instead of 1.3-1.5, but the rule would still give 1.
  Proposed: raise the share to 25% (2.0 GB on 8 GB) when the plan splits, and count the plan's real per-worker
  needs. That gives 2 workers on all-China, North America and asianita (1.7-1.8 GB est.) and 3 on New York
  (1.25 GB). Yours: it trades about 0.3-0.4 GB of the tab for about a third off loads and edits. A machine that
  reports less than 8 GB scales down as now; phones stay at 1 worker, where B does nothing.
- **The periods work (T-2624).** Its stage 2 solves the edited cluster period after period and sends each period's
  final as it lands. Under B a period's final is one part from each holder, and the page shows it once all are in.
  Each round of a period now waits for every holder, so a worker with less of that period idles; the worker loop
  fills the wait with the next period's round. That can put the first period's final a little later than with one
  worker holding the whole period. Stage 6 measures it against T-2624's own target (lines move within 12 s on North
  America).
- **Waiting and repeated searches.** Every round of a cluster is a meeting point for its holders: 5-15% idle (est.),
  more at the end of a solve. Stations reached by cities on two workers are searched twice (about 5%). The market's
  shared station table (`with_skims`) is rebuilt on the market's holder.
- **Market-heavy saves.** All of Japan (market 80% of a period) gains only with the market copied, so about today's
  memory there. B does not split the market inside itself.
- **The high-water mark.** WASM never shrinks, so moving a population off a worker leaves its space empty there.
  That is why re-deals are rare and the pool is fixed before packs open.

## Would C make B unnecessary?

C is one engine on one shared memory with a thread pool (60-100 h, pr-survey 2026-11-16).

- **Memory**: yes, where it runs. Every copy past the first goes, including the market copy B keeps on market-heavy
  saves, and a thread costs its scratch, not a JS isolate per worker.
- **Speed**: better than B where one city dominates. Threads can split New York's own pass; B cannot (exact, whole
  populations).
- **But**: a shared wasm32 memory is one 4 GB space for everything. Today's largest save fits (all 723 Chinese zones
  est. 2.6-2.9 GB held, T-1169), with no room for continent-wide saves until Memory64 is in every browser. B spreads a
  game over several 4 GB spaces: T-1560's "route to the whole world". C also needs cross-origin isolation:
  a Cloudflare header rule (your word), CORS or CORP on data.anita.garden and the map tiles, and Safari has no
  credentialless mode. And it keeps a fallback pool for browsers that aren't isolated: two paths to maintain.
- **Shared work**: stage 2 (exact sums) is needed by C as well, since a thread pool adds in whatever order threads
  finish. The rest of B (the plan, the exchange, the question routing) would be obsolete in browsers running C, but
  would remain their fallback.

So C would make B unnecessary only on isolated desktop browsers, for saves under 4 GB. My order stays: B now. After
it, C only if the measured gap justifies 60-100 h and a hosting change: New York-type saves where one city is most
of the work, or a large share of players on browsers B serves poorly.

## Related tasks

- T-354 (pack copies per worker) closes with B: each pack opens on its holders only. T-361 (one demand engine per
  city) is unchanged.
- T-2757 (questions wait after a memory restart) gets easier: a restarted worker holds certain zones, so questions
  about the others need not wait. It still needs its own design.
- T-1560 is this design's ancestor (notes/T-1169.md section 5, option H). It can close when B is filed as built.
