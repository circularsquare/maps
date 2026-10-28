# Performance overview: time, memory, scaling, what next (2026-10-28)

A reading of the notes as of T-1611 and T-1639; nothing new measured. "M" is measured (source in
brackets), "I" is inferred from the notes. Times are cold loads to the weekday final unless said.

## Where time goes

**Big loads** (3 solving workers unless said)
- All of Japan: 954 s (M, T-1594); 1,311 s on 2 workers after T-782's exact change, was 1,579 s
  (M, T-782). The market is about 75% of every crowding round: 105-120 s a daytime period in the
  browser, the 88 city populations together 30-40 s (M, T-782). A market pass is 4 a period (free
  flow + 3 crowded); natively about 50 s, two-thirds of it the ~8,200 destination searches over the
  whole graph (M, T-782, T-1545). WASM runs those searches about twice as slow as native.
- Five periods dealt over 3 workers go 2, 2, 1: the third worker idles about half of each round,
  and the worker with two daytime periods sets the time (M, T-782).
- All-China, 217 zones: 97 s, after the memory was grown 32 MB ahead (was 253-293 s, M, T-1594).
  All 723 zones: 292 s on 1 solving worker, the pool cut to one by the memory budget (M, T-1611).
- Pack opening is the floor: about 7 s a worker for Japan's 45 (T-898); all-China opening 30-35 s
  (M, T-1594). An all-China load closes and reopens about 40 zones for nothing (T-1634).
- Clock worker: 5.7 s on the China save (`TrackApi.load` 4.3, masks 2.5, overlapping; M, T-1156).

**While building**
- New York, a new line: first figures 9-11 s (the split free-flow preview, M, T-1543), the final
  about 54 s on a busy machine (M, T-1530).
- China (169 zones): an edit's solve 29-31 s to the final (M, T-1156, before T-1545 and T-1594, so
  likely less now, I). On Japan and China high speed joins every zone into one cluster, so an edit
  re-solves nearly everything; only unchanged periods are kept (a train-count edit re-solves 1 or 3
  periods of 5, a new station all 5).
- Moving the view to a coarse zone: fine and shown 59 s later (217 zones, 2 workers) or 76-96 s
  (723 zones, 1 worker): a whole solve of the cluster (M, T-1223, T-1611).
- Drawing: with the stroke cache and simplifying, Japan at 1x with the map still is 9% GPU (was 77%),
  panning 34% (was 78%), MapLibre alone 22% of that (M, T-1638, T-1639).

## Where memory goes

- **A solving worker, all of Japan**: WASM 1.12-1.32 GB, heap peak 1.10-1.28 GB (M, T-1611). The
  market pass's peak is about 0.3 GB of window (was 1.9 GB of station x station tables, T-1545).
  Per zone median 6.1 MB, Tokyo 99, Osaka 56 (M, T-1169).
- **A solving worker, all 723 Chinese zones**: 2.65 GB held, 2.97 GB heap at a refine's peak, WASM
  2.77-2.99 GB (M, T-1611), against the 4 GB ceiling. Held before today's fixes, of 3.35 GB: city
  zones' commute packs 1.17 GB, their everyday layers 0.47, the market's composition of 726 packs
  0.70, the market's network 0.42 (now about 0.12), solved market periods 0.20 (now about 0.04),
  zones set on the network and their periods 0.37. Coarse zones hold almost all of the city parts
  (1.98 of 2.01 GB). A refine adds 250 MB of skims (two station x station tables over all China).
- **New York**: heap 185 held / 256 peak, WASM about 350 MB (M, T-1169 stage 2b).
- **Aux worker**: Japan 131 MB, all-China 1,053 MB, since it opens every zone fine (M, T-1611).
- **Each worker's JS heap** outside WASM: about 0.7 GB, counted in the budget (T-1205).
- **Main page**: little measured lately. The budget sets aside 0.6 GB; New York's renderer was
  1.86 GB at 3 workers on 2026-10-15 (M, perf-survey). A line edit on Japan can post 35 MB of
  strokes whole (T-1509).
- The page budget is 8 GB: before each solve the pool drops workers until the sum fits (T-1205),
  and never adds them back (T-1366). So memory turns into time: all-China fit 3 workers instead of 2
  after T-1559 and loaded in 276 s instead of 354 (M, T-1169).

## How it scales

**With zones (cities) opened**
- Memory per worker: linear in zones open, each at its own size (3 MB median Chinese zone, 6 MB
  Japanese, 30-100 MB for a metro), about 25-35% less for a coarse one (M, T-1206 stage 0). The
  fine set is capped by a 1.5 GB budget (by design).
- **Every solving worker holds every zone**, so the page's memory is zones x workers, plus the aux
  copy. The 8 GB budget caps that product by dropping workers: past a point, more zones means fewer
  workers and a slower load, not more memory (by design, M on all-China).
- The market's composition grows with the market packs open (726 packs 0.7 GB): linear (M).
- Load time: downloads and opening linear in zones (I); a round's city solves linear in zones,
  each on its own reach (by design, T-756).

**With stations**
- City populations: each on the stations within 30 km of its box, so linear in stations overall (by
  design, T-756).
- The market's pass, time: worse than linear. One destination search per station, each over the
  whole cluster's graph, so roughly stations squared (I from T-782: 8,200 searches are two-thirds
  of a pass). This is now the main thing that grows faster than the map.
- The market's pass, memory: no longer stations squared. It holds columns for the stations in a
  sweeping band: on Japan 197 stations' route costs and 1,954 stations' trips at once (M, T-1545),
  so about stations x band width (I: between linear and squared, depending on how deep commuting
  reaches; Tokyo's 100 km commutes make the band wide).
- Still squared: the skims cache on a cluster solved from the cities' shared table (250 MB at an
  all-China refine, M). Small clusters only by design, but a fine zone on a big cluster pays it.

**With lines and trains**
- Demand barely notices: lines add graph edges and trains are frequencies, so neither changes the
  number of searches or zone pairs (I). A train-count edit re-solves only that level's periods
  (by design).
- Drawing: stroke segments grow with track and line length (Japan 860k each). After T-1638/T-1639
  the GPU cost while the camera is still no longer grows with them; panning grows with the segments
  on screen after simplifying (M). Trains themselves are under 1-2 GPU points (M, T-1618).
- Clock worker: load linear in edges and lines (`TrackApi.load`, `buildState`, I).

## Areas of improvement (ranked by gain per effort within each group)

**Compute and load time**
1. T-1608 (abandoned 2026-10-28): stopping a search once its origins settle saves only 6% of
   nodes on Japan, since every destination has a far longer-trip start (notes/T-1608.md).
   T-1647 profiles the search per node instead.
2. T-1607 share the slow worker's load: split a period's crowding rounds in parts over workers (as
   T-1543 does for the preview). Japan about -25%. Exact. Little memory (loads exchanged a round).
3. T-1634 keep reached zones open while the reach is worked out: saves ~40 reopenings an all-China
   load and steadies the load history. Small, exact.
4. T-436 start an edit's crowding rounds from the previous solve: fewer rounds after an edit;
   figures depend on edit history (her call). Not measured. No memory if kept in place (I).
5. T-921 commute pairs under 1 person a day dealt in chunks: New York -14%, stations within 0.8%.
   Approximation. No memory.
6. T-875 the evening period from midday's choices: up to 13% of a solve. Approximation; measure first.
7. T-898 pack opening (7 s a worker for Japan): measure natively first. Exact.
8. T-1640 redraw only as fast as trains visibly move: most of the ~8% GPU left. Live-feel check.
- Avoid: raising caches for speed (`WINDOW_STRATEGY_BYTES` was +41 MB a worker for 3 s a pass).

**Memory**
1. T-1367 free closed zones at the close: all-China at rest 2,041 MB against 1,502 with the same
   zones open. Exact. Also lets T-1205 restart sooner.
2. Aux worker opening zones coarse (no task): it holds 1,053 MB on all-China with every zone fine.
   Its questions (cells, a bubble's modes) may not need fine samples (I). Medium.
3. T-876 fold the market's tiny far-zone rows (94% of entries under 0.001 riders): 12 MB on New York,
   growing with zones served. Approximation nobody can see.
4. Sparse commute trip columns in the market window (no task): part of Japan's 145 MB of trip slots.
5. T-361 one demand engine per city, not two; T-354 pack copies per worker. Read before sizing.
6. A leaner coarse level (no task): coarse zones are 1.6 GB of all-China's 2.65 and keep the
   cell-to-cell walks a fine zone has. Dropping parts changes their figures; needs T-1629's bias check.
7. T-1560 zones dealt between workers instead of periods: city parts divided by the pool (all-China
   2.8 to 1.4 GB a worker with 2, est.). Very large redesign; the route to the whole world.

**Decoupling** (already in: clusters, kept periods, each city on its reach, zone in view first,
one period in parts, coarse/fine, the stroke cache)
1. T-1609 local refine: a zone going fine solved alone against its cluster's loads frozen, so a
   refine takes 3-10 s instead of 59-96 s. Approximation (neighbours see the old crowding until the
   next solve). No memory.
2. The same idea for edits (no task, I): on Japan and China one cluster holds every zone, so cluster
   decoupling saves nothing there. Re-solve first only the populations whose reach the edit touches,
   others frozen; show that; let the whole solve follow. Builds on T-1609.
3. T-1607 (above): splits work across workers within a period.
4. T-1560 (above): decouples memory from the pool size, so more workers stop costing memory, and
   the budget stops cutting workers on big saves. That is a speed gain too.
5. T-1366 grow the pool back when memory frees. Small.
6. Main thread from workers: T-1509 (post only moved stroke runs on a line edit, 35 MB on Japan),
   T-1505, T-1172 (load bookkeeping 5-7% of the main thread), T-1211 (DOM size).

## What I'd do next

1. T-1647 (T-1608's early stop came to 6%), exact, aimed at the one cost that grows faster than the map.
2. T-1607, exact, uses the core that idles half of every round; no memory bought.
3. T-1609, then the edit version of it: the biggest wait a player feels on a big network, an
   approximation they can't see.
4. T-1367 and T-1634: memory and opening wasted today, nothing traded.
5. Measure T-436 (warm-started crowding rounds) and put the history question to Anita with numbers.
