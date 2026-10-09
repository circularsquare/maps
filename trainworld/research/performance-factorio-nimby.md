# Performance: Factorio and NIMBY Rails

Research report, 2026-10-08. Sources are Wube's Friday Facts (FFF) and Carlos Carrasco's
NIMBY Rails devblog. Each item: what it is, then what it means for trainworld.

## Sleeping, deferred and bucketed updates (the main idea)

- **Entity sleeping.** The governing rule is "do less". Idle entities leave the update list
  and cost zero CPU until an event wakes them. Solar panels and accumulators are computed as
  one group. https://factorio.com/blog/post/fff-148
  - Stations with nobody waiting and trains between stops cost nothing per tick. Wake with
    timed events, never polling.
- **Deferred robot updates (2.0).** A moving robot is updated every 20 ticks, a parked one every
  60; the renderer draws a predicted position in between. 10-25% overall.
  https://factorio.com/blog/post/fff-421
  - The most directly useful one. Trains on fixed-headway lines are a function of (line,
    departure time); simulate stop events only, let the renderer interpolate.
- **Rotating buckets and refcounts (2.0).** Radars update one bucket per tick; chunks hold a
  "kept revealed" counter instead of rechecking. Idle roboports sleep (1 ms to 0.025 ms per
  tick). Same FFF-421.
  - Spread catchment and demand bookkeeping over buckets; refcount coverage.
- **Animation from the tick counter.** Crafting animations became `offset + tick % length`,
  evaluated only at render time (2%). https://factorio.com/blog/post/fff-204
  - Anything cosmetic is a pure function of time on the render side.

## Merging many things into one (segments)

- **Belts.** Consecutive belts merge into one line holding gaps between items, not positions.
  A flowing line moves by changing two end-gap integers. Item movement 50-100x faster, belts
  5-10x. https://factorio.com/blog/post/fff-148, shipped https://factorio.com/blog/post/fff-176
  - Riders on a train or a platform are counts per (destination, line) bucket, not agents.
- **Fluids 0.17.** Fluid state moved out of entities into contiguous memory, grouped into
  independent systems updated in parallel; pipes 3-6.5x faster.
  https://factorio.com/blog/post/fff-271
- **Fluids 2.0.** Pipes merge into segments broken only by machines and pumps; each segment is
  one container solved like an electric network. https://factorio.com/blog/post/fff-416
  - Each line, or each transfer-connected network, is one solved unit; independent networks
    (cities) solve in parallel with no locks.
- **Electric network.** Regrouping connector data into contiguous arrays made transfers 2x
  faster; 0.16 was 2.4x faster than 0.15 mostly from layout, not algorithms.
  https://factorio.com/blog/post/fff-209

## Memory layout

- Prefetching the next entity by hand gave 5-13%; shrinking entities did not help, because
  latency, not bandwidth, was the limit. https://factorio.com/blog/post/fff-204
  - WASM has no prefetch instruction; flat structure-of-arrays from day one is the only lever.

## Multithreading: what worked and what did not

- **0.16.** Trains, electric networks and belts on separate threads were *slower* than in
  sequence: threads writing nearby memory kept invalidating each other's caches. The
  read-only render "prepare" pass on up to 8 threads did work.
  https://factorio.com/blog/post/fff-215
- **2.0.** Read-only circuit logic parallelised (9.5% on a real save). Parallel electric network
  was rolled back: CPU 0.5% to 15% for 0.5 ms to 0.39 ms. Same FFF-421.
  - Parallelise only independent, mostly-read work (per-city demand, path queries), each
    worker owning its own memory.

## Determinism and the fixed tick

- Fixed 60 UPS deterministic lockstep; only inputs cross the network
  (https://factorio.com/blog/post/fff-76, latency hiding
  https://factorio.com/blog/post/fff-83, desync reports
  https://factorio.com/blog/post/fff-188).
  - Even single-player: deterministic sim gives replays, bug repro, and save = inputs. Floats
    in WASM are deterministic if no fast-math.

## Trains and rail pathfinding

- Graph nodes are rail segments (runs without junctions, signals or stops). Bidirectional A*;
  a binary heap gave ~20%, A* over Dijkstra ~2x. https://factorio.com/blog/post/fff-331,
  https://wiki.factorio.com/Train_path_finder
- Repath only on events (signal change, failed reservation, stop added).
  - Fixed lines barely need train pathfinding. Spend effort on the passenger graph; repath on
    network edits only.

## Rendering

- Sprite atlases per draw layer, one draw call per batch; 4 vertices / 80 B per sprite.
  https://factorio.com/blog/post/fff-227, https://factorio.com/blog/post/fff-264,
  https://factorio.com/blog/post/fff-251
  - All trains in one instanced WebGL2 draw.

## Saving

- Snapshot with `fork()` and write while the game continues; ~85% memory copy, 10%
  compression, 5% disk. https://forums.factorio.com/viewtopic.php?p=300079 (FFF-201 thread;
  FFF itself not fetched)
  - Copy flat arrays, compress in a worker; never save state that can be regenerated.

## NIMBY Rails

- **Scale and threading.** 600 trains and 70k riders at 5 sim frames per 6 ms frame. Train and
  passenger AI dominate, both run in parallel; trains about to cross a signal move to a
  final serial pass. Accounting made async (it was 2/3 of train-AI time). Unloading capped at 5
  destinations per train per frame. https://carloscarrasco.com/page/28/ (2023-04)
- **Sim and UI decoupled.** Triple-buffered sim state; train motion copied each frame (~1 MB
  per 1,000 trains), heavy passenger data only on demand.
  https://carloscarrasco.com/page/21/ (2023-11)
- **Passenger paths.** Async helper warms the path cache on workers before trains arrive (an
  order of magnitude, https://carloscarrasco.com/page/53/, 2021-03). Only transfer stations are
  intermediate nodes (https://carloscarrasco.com/page/19/, 2024-01). Per-tile path caches had
  poor hit rates in dense saves (https://carloscarrasco.com/page/16/, 2024-04).
  - NIMBY simulates individual riders and still tops out in the hundreds of trains and tens
    of thousands of riders. trainworld's flow model avoids that ceiling; keep NIMBY's
    transfer-station graph and the "serial pass only for conflicts" idea.

Unverified: that Factorio rails and signals are inactive entities whose state changes only on
block occupancy change (likely, not sourced).
