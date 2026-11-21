# T-3444 design, part 8: performance

Agent tdperf, 2026-11-21. Design draft; nothing is built or measured here. Figures are from research/code.md section 7
and the notes it cites. Each is tagged **measured** (task, date, conditions), **stale** (from before fixes that
changed the path), or **proposed** (a budget). Most measurements were taken on a loaded machine, so every budget has
two parts: an absolute number, and a same-run ratio against master, old and new alternating in one loaded game
(T-3803's method, 20 arms a side).

Anita's accepted requirements (TASKS.md, 11-20): a station placed on track that leaves its shape alone redraws no
neighbours (T-2767); the drawn network rebuilds only near a change (T-2701); the Penn Station to Emeryville snapped
preview is a check case (T-763).

## Requirements

| # | What | Today | Budget (proposed) | Check |
|---|---|---|---|---|
| R1 | Station on built track, shape unchanged (T-2767) | worker buffers 200-900 ms, of it `lineBands` 130-750 and `besideSamples` 60-350; 8-15 lines redone (measured, T-2768, real NYC, 4 cores, busy). Shows up 1,925 ms median (measured, T-3803 before, 2,647-station save, 9-13 of 14 cores taken) | Drawn records rebuilt: 0 track chunks, 0 line strokes, 0 band records; only the station, its platforms and its complex's passages and facilities. Worker drawn-network time 30 ms or less. Shows up 400 ms median or less on the 2,647-station save, and at most a third of master's in the same run. The sim edit no slower than today's split, 2.6-3.0 ms natively on real NYC (measured, T-3137 C1) | `t2763-edit-lag.mjs` plus rebuild counters (below); T-2768's `edit-compare.mjs` fingerprints equal apart from the new station |
| R2 | Rebuild only near a change (T-2701) | `networkFromState` walks every edge, line, sample and station: 150-740 ms of a 0.4-1.1 s follow-up on North America (measured, T-2404, 11-06, cores 12-15 pinned); one full state on the 2,647-station save 301 ms network, 94 rows, 29 netForPost, 22 post (measured, T-3803 after option 3) | Drawn-network work for a one-bundle edit independent of network size: 30 ms or less in the worker on North America day 383, and within 2x of the same edit on a 10-station save. Items visited at most 4x items changed plus a constant | Counters; one segment added on NA and on a small save |
| R3 | Penn Station to Emeryville snapped preview (T-763) | 3 s (stale, 2026-10-19, before T-1779 and T-2700); the quad edit 3.3 s to 0.15 s natively (measured, T-727c) | No slower than master in the same run, and 1 s or less in the browser on real NYC, double and quad | Measured first in stage 1; `t2403-follow.mjs` style timing of the one ghost |
| R4 | Delete | Built station delete 650-2,200 ms, about 0.5 s less after T-2768 (measured, 11-17, real NYC) | A station delete: R1's numbers. A bundle delete: drawn work of its own chunks and the lines on it only. Delete one track: that bundle's chunks and its lines | `t2763-edit-lag.mjs` delete arms |
| R5 | Track edit, click to follow-up at rest | NYC worker 40-60 ms of own work; NA 0.4-1.1 s (measured, T-2404) | NYC 60 ms or less; NA 100 ms or less in the worker; main thread 30 ms or less on NA (today 35-100 ms) | `t2763-edit-lag.mjs`, segment add and reshape |
| R6 | Previews, per pointer move | Ghost 35-37 ms p50 pointer to drawn on NYC (measured, T-2404); `preview_route` 5-8 ms in the worker on NYC, 2-3 ms WASM own time on all Japan (measured, T-2700) | Track ghost no slower than master. Station ghost 5 ms or less in the worker (it compiles the bundle under the pointer only, never the world) | `t2403-follow.mjs`; a station-ghost arm |
| R7 | Load | Native: NYC 88-110 ms, Northeast 1,271-1,559 ms (measured, T-3137 C1-C2); shipped London start 0.6 s settled (measured, T-3233). Browser: not measured on master | Native 1.2x master or less on NYC and Northeast; London start 1 s or less; browser 1.1x master or less on the 2,647-station save. An old save's one-time conversion adds at most one load's time, plus a settle only for lines the conversion changed | Stage 1 measurement; the parity bin's timing |
| R8 | Compile | (new) | A whole compile at load 30 ms or less natively on real NYC. A platform edit 2 ms or less natively, also on a 5,400 km bundle | Native bench beside the parity bin |
| R9 | Memory | Not measured | WASM memory after load and the clock worker's drawn-network cache, each 1.2x master or less on the 2,647-station save and NA | Stage 1 measurement |
| R10 | No quiet change | `t3137-lanes-parity.mjs` (bin `lanes_parity`): 44 sections equal (measured, C1-C2) | At stages 2, 3 and 4, every section equal on real NYC, the Northeast save, London's start and the small games, or each difference reviewed and recorded | The parity check extended to platforms and stations |

Wording for player text and the changelog: the map catching up is "shows up". The numbers updating afterwards are
"calculating trips", which is not part of any budget here.

**What stages 2-4 can and cannot meet.** R1 and R2 need the drawn network keyed by records (stage 5). Before that,
stage 3 doubles the edges going through today's whole-network build: 1,647 double edges become 3,294 track edges
on real NYC. So stages 2-4 get a no-regress guard instead: worker follow-up and shows-up within 1.2x of master in the
same run, plus R7 and R10. Stage 4 does meet R1's sim half: no `back_lines`, `twin_at`, `complex_of` or
`way_group` scans.

## Whole-network passes to design out

| Pass (research/code.md 7) | Cost | Replaced by | Stage |
|---|---|---|---|
| `networkFromState` walks every edge, line, sample, station to find what changed (T-2404) | 150-740 ms NA | the edit's touched set, carried from the sim with the state (T-2404's "medium work") | 5 |
| Posted arrays compared with the last ones | part of the above | patches by range (`NetCache.last`, `NetPatch`) | 5 |
| `back_lines` scans every edge and line each edit and rebuilds every line with a derived return | every edit | return paths stored with the line (01) | 4 |
| `twin_at` scans every edge per station placement | O(E) | bundle slots | 2 |
| `complex_of` scans every station per call, called per stop | O(N) per stop | complex kept per station, updated near the edit through a grid | 4 |
| `way_group` scans every node | O(N) | the simple-stops rule (membership) | 4 |
| `find_path_side` allocates `2 * edge_count` per hop | O(E) per hop | routing on authored tracks (smaller than the derived graph: no stop splits) with scratch arrays kept and stamped by generation | 3 |
| `derive_ways` over runs | per edit | ways stored | 3 |
| `legShifts` builds a per-node table over every edge on every build | 0.7 s of a profiled 3.8 s (T-3803) | each track drawn at its offset | 5 |
| `besideSamples` resamples neighbours of changed edges | 60-350 ms midtown | gone: lines on a track are known from paths | 5 |
| Complex rule in `game/complexes.ts` as well as `twin.rs` | per state | one copy in the sim, sent to the app | 4 |

## Keys: why stable ids make caching work

Every drawn entry is keyed by records that an unrelated edit never changes:

- **Track chunks.** A bundle can be long now: with no splits at stations, real NYC is about one bundle a line, and a
  snapped cross-country route is one bundle of 5,688 corners. So the unit is a **chunk**: a run of the alignment's
  PIs, cut where a PI's own content hash says so (content-defined, about 8 PIs a chunk on average). A cut never
  depends on chainage or on platforms, so moving one PI changes its chunk and at most its two neighbours, and
  placing a platform changes none. Key: (track id, hash of the chunk's PIs, the offset profile within it, kind,
  built state).
- **What stays keyed by bundle and chainage**: platforms, stop points, speed-limit ranges, spread zones and train
  samples are positions (bundle or track, chainage); only the stroke chunks are cut by PI content. The plain
  alternative is fixed chainage windows (every 1 km from the bundle's start): simpler, but a PI moved or added near
  the start shifts every window after it, so a reshape at one end of a 400 km bundle redraws all of it. Content cuts
  avoid that at the cost of a hash per PI.
- **Platforms**: (platform id, shape hash). **Stations**: (station id, member list). Passages and facilities per
  complex, as T-3803 option 3 keeps them.
- **Line ribbons**: (line id, track id, chunk key). The lines on a track come from paths, so a station placement,
  which changes no path (a span covers a whole bundle track), touches no ribbon. Order rules (`rel`) are per track,
  and T-2768's dependency records (`rels`, `relDeps`) now name tracks.
- **Train samples**: by (line, run, track span, chunk) in chainage, never by derived edge, so a stop split leaves
  them alone unless the line's timing changed (a new passing point on single track does change it, rightly).

What each edit rebuilds:

| Edit | Rebuilt |
|---|---|
| Station on built track | station, platforms, complex passages and facilities |
| Station delete | the same |
| Segment added | its bundle's chunks, end nodes' drawing, lines on it |
| Reshape (one PI) | 1-3 chunks of that bundle, lines on those chunks; platforms the edit re-projects |
| Delete one track | that bundle's chunks, lines on it |
| Train count or headway | that line's samples only |

## The compile, per bundle

Steps, with the cost each must stay under (proposed, native):
1. **Geometry per chunk** (pieces, limits, vertical profile): today's derive, cached by chunk key; only changed
   chunks are derived again.
2. **Tracks.** At default offsets the sim needs no per-track geometry: lengths are the alignment's (01's 0.1% lie),
   lateral positions at nodes are slot offsets. Offset polylines are made for drawing only, per chunk. Spread zones
   add a per-zone S, also per chunk.
3. **Stop splits.** The bundle's stop points per track, sorted by chainage, cut the derived edges: O(platforms on
   the bundle), tens of microseconds.
4. **Derived ids stay put.** Each bundle keeps its derived edges per track. A new stop point splits one span: the
   first half keeps that span's id, the second half gets a new one, and every other span keeps its id. Removing a
   stop point joins two spans under the first one's id. Otherwise every edit inside a long bundle would invalidate
   keyframes, capacity keys and boarding for every line on it.
5. **Moves** at the bundle's end nodes: small (ends x tracks).
6. **Downstream**: lines on the touched spans get their derived paths spliced from their spans, then profiles,
   capacity and services as today's incremental passes (0.2-1.2 ms capacity per edit, measured T-050, synthetic;
   T-1863 settles only lines sharing capacity).

A platform edit therefore adds one derived edge per served track. Budget: steps 1-5 for it 0.5 ms or less natively
whatever the bundle's length; the whole edit under
R1's 2.6-3.0 ms. A platform that no line stops at yet still splits derived edges (passing lines' keyframes and
capacity keys change), so it is budgeted the same as one with stops.

## How the app's drawn network changes (stage 5)

- **network.ts**: entries per edge become entries per (track, chunk); `networkSteps` stays a generator but walks
  only the touched set. Out go `lanePoints`' lane split, the taper (`taperAt`), `legShifts`/`shiftedEnds`,
  `splitPoints`/`singleSplitEnds`, and at their stages `passingTracks`, `turnbackStrokes`, `singleLeadStrokes`
  (01's table). Input: per-track chunk polylines and the touched set from `snapshot.rs`/`stateRows.ts`, whose rows
  become bundles, tracks, platforms and stations.
- **lineBands.ts**: bands per track chunk, lines from paths; `besideSamples` no longer an input; per-line memo kept.
- **corridors.ts**: deleted. Lines on two separate parallel bundles (two doubles 12 m apart, not one bundle) are no
  longer ordered as one corridor; each bundle orders its own. Worth one look at midtown in stage 5.
- **Main thread**: GPU buffers already keep unchanged runs (T-1028); `fullAttr`, station buffers and textures move
  to the same per-run patches, for R5's main-thread number.

## Budgets per stage

| Stage | Must meet |
|---|---|
| 1 | Measurements below, on master |
| 2 | No regress (1.2x); R10 |
| 3 | No regress; R7, R8 (load compile), R10; routing without the 2E allocation |
| 4 | No regress; R1's sim half; R7; R10 with stations and platforms |
| 5 | R1, R2, R4, R5, R6, R9 |
| 6-11 | No regress against stage 5; R3 again at 6 (bundles of 4) and 8 (crossovers) |
| 12 | R7 on the rebuilt shipped starts |

## Measurements to take first on master (stage 1, when cores are free)

Claim cores in the ledger; alternate master and a copy in one loaded game where two builds are compared.
1. `t2763-edit-lag.mjs` on real NYC and the 2,647-station save (`artifacts/trainworld/t3600-asianita-1852.save`):
   station on built track, station delete, segment add, reshape; click to shows up, with the worker split
   (`networkFromState`, rows, netForPost, post) and main thread. n of 6 or more.
2. `t2403-follow.mjs` on NYC and North America day 383: steady and just after a click.
3. Penn Station to Emeryville snapped preview on real NYC, double and quad (R3).
4. Native, T-3137 C1's set: NYC and Northeast load, a station split into the busiest edge, reroute all lines, 600
   station-pair searches.
5. Browser load to playable: real NYC, the 2,647-station save, London's start.
6. Memory: WASM memory after load and clock worker heap, 2,647-station save and NA.
7. Save bytes per section (part 7).
About 3-4 h of agent time, most of it waiting on runs.

No calls for her.
