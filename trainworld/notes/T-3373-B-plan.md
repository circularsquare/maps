# T-3373 stage B: zones and the market split between solving workers (design, for Anita)

Agent splitplan, 2026-11-21. Design only: nothing in app/ or sim/ changed, nothing measured in a browser. Every
figure below comes from measurements already in notes (named where used) or from the code; "est." marks an
estimate, with its method in "Method".

## For Anita, in plain words

- Today each worker copies every city and the longer-trips market, so an extra worker costs 0.4 GB on New York and
  1.3-1.5 GB on all of China and the 2,000-station saves. That is why stage A left those on one worker.
- B gives each city to one worker and the market to one worker. After each crowding round they swap line loads.
- On those big saves a second worker would then cost about 0.25 GB, and loads and edits would be about a third
  faster, as two workers are today (est.: the first stage checks the split it rests on).
- New York gains little: one city is three quarters of its work, so it stays copied. B lets it run 3 workers in
  today's memory limit (load about 50 s, from 60-70).
- Yours: A's limit (1.4 GB for all workers on an 8 GB machine) is below one worker on these saves, so it would
  have to rise to about 2 GB. And every trip figure moves once, by about a millionth.
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
for the kernel. Per-city totals (the summary, fare sensitivity) are re-added on the page from the per-city figures
each result already carries, in the registry's order. Effect: every figure moves once, at the rounding level (T-1545
moved loads by about 1e-6, the worst element 1.4e-5; this should be smaller, since the rounding is 2^-20 riders and
an f64 total is more exact than today's f32 running sum). After that, the same network gives the same bits at 1, 2
or 3 workers, under any plan. It also makes "the others' loads" exact as total minus mine, which the edit stages use.

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
| 4 | **B for whole solves.** The per-round exchange over ports, the capacity fit's input, result parts added on the page, the rough solve on the zone in view's holder. Every solve is whole: no local stage, no kept periods, no refine, no T-1543 parts. | 4-6 | under the flag | Figures equal to the bit to the unsplit build with exact sums, on New York and all-China. All-China at 2 workers: memory and load time measured. Stop rule: under 20% faster than 1 worker, or over 0.4 GB more, stops the plan here. |
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
