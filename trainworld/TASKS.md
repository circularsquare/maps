# Tasks

Open agent work. Format and rules in `CLAUDE.md`. P1 = now or blocking, P2 = this milestone,
P3 = later. Finished or abandoned tasks move to `ARCHIVE.md`.

**Where things stand (2026-10-09, end of batch 7).** M1's New York sandbox is playable end to end:
drawing and editing track (corners, node drags, hotkeys, delete tool), stations, lines, schedules,
money, saves, demand with soft station reach, and the demand views (riders on the network,
commuter bubbles with click-through). Nothing is in progress. The batch headings below are
history; fare-sensitive demand (T-067), busiest-hour train views (T-087) and annotation cleanup
(T-100 to T-102) are done as of 2026-10-10. The next open P2s are T-069 (common lines) and T-010 (the M1
doc tidy), then M2 (Northeast, long distance). The whole `trainworld/` folder was still untracked in git at
this point; Anita commits.

- T-104 [P1] Schedule whole trains on a line with derived headways and persistent allocations. taking, 2026-10-10, train_counts.

## M1, New York sandbox

- T-105 [P1] Four track display modes: Line, traffic thickness (default), Height and Max speed. Taking, 2026-10-10. notes/T-105.md.



### Batch 2 (started 2026-10-08, paused overnight, resumed 2026-10-09): foundations, one agent per area

Performance follow-ups (T-045, T-050, T-051 done, ARCHIVE; measurements in notes/T-045.md):
- T-075 [P3] The overlay draws twice per frame while dragging and playing (122 draws a second):
  skipping the frame loop's draw when MapLibre has a frame queued (`?perfTry=noDouble`) saved
  about 2 points, inside the noise. Adopt only with T-074 if it is cheap to keep.
- T-076 [P3] Playing costs 13-15% of a core on any network, nearly all of it Chrome presenting 60
  frames a second (GPU process 9, renderer 4; our draw 0.1-0.2 ms). A lower frame rate when the
  fastest train on screen moves under about half a pixel a frame would roughly halve it. A
  smoothness judgement for Anita; measure the visible step first.
- T-077 [P3] Drop `TrackApi.issues()` (Debug text of every issue; the app reads `issue_kinds`):
  a few kB of the track wasm module. notes/T-051.md.

Track model (`sim/src/track/`): T-040 is done (ARCHIVE). Its follow-ups, none blocking:
- T-052 [P3] Capacity gaps: a short-turn reversing at a through station does not yet conflict with
  the other direction there; a single-track station has one platform track for both directions.
  notes/T-040.md "Capacity".
- T-053 [P3] Track across several cities (M2): positions are planar metres in one city's frame;
  intercity track needs a frame per region or lon/lat in the save.

Data (`pipeline/`, demand constants): batch 2's four data tasks are done (ARCHIVE).

### Data and model quality, later

- T-041 [P3] Water mask from OSM instead of TIGER, for cities outside the US (M3) and the world
  pack's intercity track (M2): coastline-derived sea polygons plus inland water, reproducing the
  T-030 checks in `read_pack.py`. The osmdata.openstreetmap.de sea file is one 906 MB global zip,
  so this wants a once-per-world preprocessing step. notes/T-030.md.
- T-043 [P3] County-government and school-district head-office lumps in LODES (listed but left
  alone in notes/T-014.md): check which are real, spread the proven ones as T-014 did.
- T-044 [P3] Adaptive cell grain, only if performance needs it: keep res 9 (or finer) in dense
  centres, merge sparse suburban cells (SPEC 3).

### Batch 3 (started 2026-10-09): the game itself

Gameplay agent: T-025, T-055, T-023, T-024 and T-032 are done (ARCHIVE), T-031 finished in
batch 4. Their follow-ups, notes/T-025.md:
- T-070 [P3] Edits on 10,000 km networks: 23-36 ms of main thread an edit plus deserialising the
  whole state (round trip 130-214 ms). Send what an edit changed (edges, nodes, lines by id)
  instead of the whole state, keep stroke buffers per edge in the worker, upload sub-ranges.
  notes/T-063.md.
- T-071 [P3] More blueprint editing: add or remove a PI on placed blueprint track, set one PI's
  level (T-064 does PIs, radii and the whole stretch's level; dragging track ends, junctions and
  stations is T-093). notes/T-064.md.
- T-065 [P3] Station names when zoomed out: street names come from the basemap's loaded tiles,
  which carry them only from zoom 13; further out a new station gets a neighbourhood or a number.

Demand agent (`app/src/workers/demand*`, `app/src/game/demand.ts`, `sim/src/demand/`): T-026 and
T-021 are done (ARCHIVE). Their follow-ups, none blocking:
- T-058 [P3] Fit the commute timing and walking: the period shares of trips to work from ACS
  B08302 (time leaving home, by tract) and of trips home from NHTS; the walk curve from ACS
  B08301 walked by county. Both are judgement today (SPEC 4.2, 4.3). notes/T-026.md.
- T-059 [P3] Trips home chosen from the work end (the return direction's crowding in the choice) instead of the morning's station pairs reversed.
  notes/T-026.md.
- T-060 [P3] City packs on R2: versioned path, compressed, cached for good, one base-URL constant
  in `workers/demandClient.ts` (and the clock worker's pack URL); parse the header in JS so
  `serde_json` leaves the WASM. notes/T-026.md "Serving the pack from R2 later".
- T-061 [P3] Demand on big networks: split mode choice by origin zone over the pool and cancel
  inside a period, once one period's solve takes well over a second (today 0.3-0.45 s on 434
  stations). notes/T-026.md.

Basemap agent (the basemap style module in `app/src/map/` and `app/public/fonts/`): T-056 is done
(ARCHIVE). Its follow-up, for cities in Japan (M3):
- T-057 [P3] Kana and kanji on the basemap in Zen Maru Gothic: MapLibre draws them in the browser's
  system font today (they never come from glyph tiles). MapLibre 6's style `font-faces` can pin a
  self-hosted Zen Maru file to those codepoints, fetched only when such a label is drawn; check its
  size and the main-thread cost of drawing them. notes/T-056.md.

Later in M1:
- T-068 [P3] Several save slots in the browser and a guard against two tabs sharing the one
  autosave (T-029 has one slot). notes/T-029.md.

### Data and model quality (not yet scheduled)

- T-069 [P2] Common lines in the demand kernel: a rider at a stop served by several lines that
  reach the same next stops boards the first train of any of them, so the wait is half the
  combined headway; today each line's own headway counts, so a trunk shared by lines or a
  branching service looks rarer than it is (on the real network the 7 local's two GTFS patterns
  halved its riders until folded together). notes/T-007.md.
- T-072 [P3] (was P1; realism is now rough calibration only, and Anita: real road travel times
  are not a priority) County-to-county flows in the zone gravity: the crow-fly gravity sends Hudson,
  Bergen, Essex, Passaic and Union residents to Manhattan at 1.9-4.3x the ACS rate (it cannot see
  the Hudson) and keeps 39% of commuters in their own county against 52% in ACS (LODES OD, the
  decay's fit target, has 43%: head-office job placement). Fit county-pair factors (K-factors) to
  the ACS 2016-2020 county flows less work from home, ship a region per zone and the factor table
  in the pack, refit the decay on lengths within pairs; then T-034. notes/T-034.md section 3.
  **Before fitting K-factors** (orchestrator, 2026-10-09): try the general fix first. Replace
  crow-fly distance in the gravity's decay with zone-to-zone road travel time from OSM (offline,
  ~5k zone centroids; this sees the Hudson's few crossings and also fixes T-034's "suburban roads
  2-3x too slow" for the car alternative). It works in every country; ACS county flows exist only
  in the US. Then fit county K-factors only to the residual, and report how much each step closes
  the 1.9-4.3x NJ overshoot and the 39% vs 52% own-county gap. Order still T-072 before T-034.
- T-036 [P3] Faster pair loop: ~35 ns per subzone pair native against ~21 before the logsum. A
  packed per-destination layout (done in T-090) or wasm SIMD over egress stations could make
  10-minute walk bands (2.4x the pairs, every error a fifth smaller) fit the recompute budget.
  Since T-090 a third more origin subzones each build a row over all stations and visit the
  destination zones though few pairs survive there: about 10% of a stage. notes/T-020.md,
  notes/T-090.md.

### Batch 4 (started 2026-10-09, planned while Anita was away)

Gameplay agent (`app/` except demand files; `sim/src/track/` except the capacity pass while the
performance agent has T-050), in order: T-029 saves, T-028 money, T-062 side-by-side lines,
T-031 rest, T-064 edit blueprint track, then T-063 if time. All six done (ARCHIVE); follow-ups
T-067, T-068, T-070, T-071.

Demand realism agent (`sim/src/demand/`, `app/src/workers/demand*`, `app/src/game/demand.ts`,
`pipeline/`, `data/`), in order: T-007 real New York network and ridership targets, T-054,
T-022, T-034, T-066, T-058. The first useful output is the real-network comparison. T-007,
T-054, T-022 done (ARCHIVE); T-034 paused for T-072 (the gravity), and T-066 and T-058 not
started (the other-trips layer and the walk curve sit on the same distance model; T-058's period
shares from ACS B08302 do not and could go any time). Next: T-072, then T-034, T-066, T-058.
- T-066 [P3] Non-commute trips (SPEC 4.1 "Other trips"): shopping, school, errands, leisure as a
  second trip layer, population to attractors (jobs as the stand-in; schools and retail from open
  data if cheap), its size relative to commutes and its time profile from NHTS/ACS-era travel
  surveys for New York; the mode-split bar then covers all trips, not commutes only.

Performance agent (`sim/src/track/` capacity pass for T-050, `sim/Cargo.toml` and the wasm build
for T-051, measurements in app/ without behaviour changes for T-045): T-045, T-050, T-051 done
(ARCHIVE); follow-ups T-073 to T-077 above.

### Batch 5 (started 2026-10-09): Anita's feedback after playing

Demand agent (`sim/src/demand/`, `app/src/workers/demand*`, `app/src/game/demand.ts`,
`pipeline/`, `data/`):
T-078 done (ARCHIVE). Not started: T-066 non-commute trips, T-058 timing (both P3 now; realism
is rough). T-089 covered by T-090 (ARCHIVE).

Gameplay agent (rest of `app/`, `sim/src/track/`): T-081, T-082, T-083, T-080, T-079 and T-085
done (ARCHIVE). Follow-ups:
  (T-088 tool keys: covered by T-096, ARCHIVE.)

### Batch 6 (started 2026-10-09): Anita's answers on demand views, reach, money, drawing

- Views agent: T-084 done (ARCHIVE). Follow-ups, notes/T-084.md:
- T-094 [P3] Station circles at complexes: the real New York save has several GTFS stations at one
  place (34 St-Penn Station, Grand Central), and sized by riders (up to 2.6x) their circles pile
  up. Size a group of stations within 100 m (one demand station, SPEC 4.4) as one circle, or cap
  the growth lower.
- Demand agent: T-090 done (ARCHIVE).
- Gameplay agent: T-092 and T-093 done (ARCHIVE).

### Batch 7 (started 2026-10-09): Anita's feedback on the demand views

- Gameplay agent: T-096 done (ARCHIVE).
- Views agent: T-097 and T-099 done (ARCHIVE).

### End of M1

- T-010 [P2] Doc tidy at the end of M1 (see CLAUDE.md).

## Later

- T-011 [P3] Northeast long-distance data: NextGen NHTS OD zones, Amtrak NEC ridership by station
  pair if published, air routes from `flights`.
- T-012 [P3] GHSL non-residential layer as the tier-0 jobs proxy; compare with LODES WAC in New
  York to see how good a proxy it is before relying on it abroad.
- T-033 [P3] Draw track and trains under MapLibre's globe projection (world view): our shaders
  assume mercator, so under globe the network collapses to a speck at zoom 3 and vanishes at 6-10.
  Use `shaderData.vertexShaderPrelude` (`projectTile`) with per-tile projection data, keeping
  centimetre precision at street zoom. notes/T-018.md.
- T-039 [P3] Dark theme (SPEC 8).
