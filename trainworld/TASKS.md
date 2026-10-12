# Tasks

Open agent work. Format and rules in `CLAUDE.md`. P1 = now or blocking, P2 = this milestone,
P3 = later. Finished or abandoned tasks move to `ARCHIVE.md`.

**Current direction (Anita, 2026-10-10): content and gameplay first.** Performance is adequate.
Keep optimization tasks parked until Anita reports a concrete performance problem; do not
choose them proactively. Focus next work on new places, travel content and useful game features.


**Where things stand (2026-10-10).** M1's New York sandbox is playable end to end:
drawing and editing track (corners, node drags, hotkeys, delete tool), stations, lines, schedules,
money, saves, demand with soft station reach, and the demand views (riders on the network,
commuter bubbles with drawing-order picking). Fare-sensitive demand, common-line waiting, busiest-hour
crowding, whole-line train allocations, four track display modes, browser save slots and annotation/
train cleanup and the M1 documentation tidy are complete. Remaining items are optional P3
improvements. The batch headings below retain project history. M2 (Northeast and
long distance) is the next larger milestone. See ARCHIVE.md for completed work.


## M1, New York sandbox




### Batch 2 (started 2026-10-08, paused overnight, resumed 2026-10-09): foundations, one agent per area

Performance follow-ups (T-045, T-050, T-051 done, ARCHIVE; measurements in notes/T-045.md):
- T-076 [P3] Playing costs 13-15% of a core on any network, nearly all of it Chrome presenting 60
  frames a second (GPU process 9, renderer 4; our draw 0.1-0.2 ms). A lower frame rate when the
  fastest train on screen moves under about half a pixel a frame would roughly halve it. A
  smoothness judgement for Anita; measure the visible step first.

Track model (`sim/src/track/`): T-040 is done (ARCHIVE). Its follow-ups, none blocking:
- T-126 [P3] A world frame for games far from New York: the conic is good to 0.4% within 5 degrees
  of latitude of its origin (35.8 to 45.8 N). Another region group is a registry entry with its own
  origin (the origin is in the save's network bytes); the New York default stays. notes/T-125.md.
- T-127 [P3] The renderer's single mercator origin (-73.98, 40.73): Float32 positions are
  centimetre-exact within about 300 km, 12 cm at 1,000 km (visible at the deepest zoom). Re-anchor
  per view if a game's track spreads further. notes/T-125.md.
- T-128 [P3] The regional market's own zone distances and drive times are computed in its pack's
  equirectangular New York frame (1.2% off at Philadelphia, 2.7% at Boston). Store site lon/lat in
  `pipeline/intercity.py` and compute on the ground; moves numbers the market was tuned against, so
  Anita decides. notes/T-125.md.
- T-129 [P3] Offer converting an old save to the world frame (`TrackApi.convert_save_frame` exists,
  tested): lengths and quotes change (Philadelphia track 1.2% longer), money already spent does not.
  Anita decides whether and where it appears. notes/T-125.md.

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
- T-121 [P2] The regional market's free-flow estimate routes without the distance fare
  (`IntercityApi::solve` never calls `set_crowding`; the local populations do): its rail boardings
  are about 17% higher than a consistent run gives. Fixing it also lets round 0 share New York's
  strategy tables (about 0.25 s of the 1.7 s to free flow). Anita decides: it moves numbers the
  market was tuned against. notes/T-119.md.
- T-122 [P3] Station catchments, a selection's far end and longer-trip queries wait for the solve
  to finish (a worker answers only when all its periods are at the round asked). Keep per-round
  copies of what they read (`sub`, the market's pair and station rows) so they answer from the
  published round, as the bubbles' sums do since T-119. notes/T-119.md.
- T-124 [P3] Reuse the periods an edit did not touch: a change to one demand level's trains leaves
  the other levels' periods identical (their inputs are the same), so the workers could keep
  their results. Saves CPU, not time, while each period has a worker; helps the 1-3 worker pools
  of smaller machines. notes/T-119.md.

Basemap agent (the basemap style module in `app/src/map/` and `app/public/fonts/`): T-056 is done
(ARCHIVE). Its follow-up, for cities in Japan (M3):
- T-057 [P3] Kana and kanji on the basemap in Zen Maru Gothic: MapLibre draws them in the browser's
  system font today (they never come from glyph tiles). MapLibre 6's style `font-faces` can pin a
  self-hosted Zen Maru file to those codepoints, fetched only when such a label is drawn; check its
  size and the main-thread cost of drawing them. notes/T-056.md.

Later in M1:

### Data and model quality (not yet scheduled)

- T-072 [P3] (was P1; realism is now rough calibration only, and Anita: real road travel times
  are not a priority) County-to-county flows in the zone gravity: the crow-fly gravity sends Hudson,
  Bergen, Essex, Passaic and Union residents to Manhattan at 1.9-4.3x the ACS rate (it cannot see
  the Hudson) and keeps 39% of commuters in their own county against 52% in ACS (LODES OD, the
  decay's fit target, has 43%: head-office job placement). Fit county-pair factors (K-factors) to
  the ACS 2016-2020 county flows less work from home, ship a region per zone and the factor table
  in the pack, refit the decay on lengths within pairs. Historical diagnosis: notes/T-034.md section 3
  (T-034 is superseded; T-090 improved the driving alternative).
  **Before fitting K-factors** (orchestrator, 2026-10-09): try the general fix first. Replace
  crow-fly distance in the gravity's decay with zone-to-zone road travel time from OSM (offline,
  ~5k zone centroids; this sees the Hudson's few crossings). It works in every country; ACS county flows exist only
  in the US. Then fit county K-factors only to the residual, and report how much each step closes
  the historical 1.9-4.3x NJ overshoot and the 39% vs 52% own-county gap. This remains optional P3.
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
T-054, T-022 done (ARCHIVE); T-034 was superseded and T-090 improved walking/driving calibration.
Optional follow-ups remain T-058 timing, T-059 return choices and T-072 gravity geography.
- T-142 [P2] Continue non-work demand: airport/medical facility visits, school coverage beyond NYC public buildings, and further cities/countries. Find suitable attendance/visit inputs and retain purpose/timing inspection. notes/T-140.md and T-138.md.

Performance agent (`sim/src/track/` capacity pass for T-050, `sim/Cargo.toml` and the wasm build
for T-051, measurements in app/ without behaviour changes for T-045): T-045, T-050, T-051 done
(ARCHIVE); follow-ups T-073 to T-077 above.

### Batch 5 (started 2026-10-09): Anita's feedback after playing

Demand agent (`sim/src/demand/`, `app/src/workers/demand*`, `app/src/game/demand.ts`,
`pipeline/`, `data/`):
T-078, T-066's first non-work slice and T-140 school attendance are done (ARCHIVE); T-142 continues them above. T-058 commute
timing remains P3. T-089 covered by T-090 (ARCHIVE).

Gameplay agent (rest of `app/`, `sim/src/track/`): T-081, T-082, T-083, T-080, T-079 and T-085
done (ARCHIVE). Follow-ups:
  (T-088 tool keys: covered by T-096, ARCHIVE.)

### Batch 6 (started 2026-10-09): Anita's answers on demand views, reach, money, drawing

- Views agent: T-084 done (ARCHIVE). Follow-ups, notes/T-084.md:
- Demand agent: T-090 done (ARCHIVE).
- Gameplay agent: T-092 and T-093 done (ARCHIVE).

### Batch 7 (started 2026-10-09): Anita's feedback on the demand views

- Gameplay agent: T-096 done (ARCHIVE).
- Views agent: T-097 and T-099 done (ARCHIVE).

### End of M1

### Batch 8: line management feedback (2026-10-10)

T-109 to T-112 complete: sorting, immediate empty lines/copies, editable badges and safe infill
stop editing. Browser interaction and native save/model checks pass; see ARCHIVE.md.

## Later

### Batch 9: dialogs and Northeast expansion (2026-10-10)

T-116, T-117, T-118, T-120 and T-131–136 done: see ARCHIVE.md.

- T-011 [P3] Northeast long-distance data: NextGen NHTS OD zones, Amtrak NEC ridership by station
  pair if published, air routes from `flights`.
- T-012 [P3] GHSL non-residential layer as the tier-0 jobs proxy; compare with LODES WAC in New
  York to see how good a proxy it is before relying on it abroad.
- T-033 [P3] Draw track and trains under MapLibre's globe projection (world view): our shaders
  assume mercator, so under globe the network collapses to a speck at zoom 3 and vanishes at 6-10.
  Use `shaderData.vertexShaderPrelude` (`projectTile`) with per-tile projection data, keeping
  centimetre precision at street zoom. notes/T-018.md.
- T-039 [P3] Dark theme (SPEC 8).

### Batch 10: UI feedback and education demand (2026-10-11)

- T-145 [P2] Solid track placement ghost using the selected height/speed colour mode, or a white body by default. Taking, 2026-10-11, UI subagent; notes/T-145.md.
- T-146 [P2] Station complexes from nearly touching platforms, station/complex inspector toggle and aggregated map bubbles. Open; notes/T-146.md.
- T-147 [P3] Station-complex inspector mini-map with per-line traffic and strongest platform transfer flows, capped for large hubs. Open; depends on T-146.
- T-148 [P3] Search stations, lines, cities and eventually neighborhoods/addresses. Open.
- T-149 [P2] Load/solve the viewed city first; startup loading percentage and interaction/clock gate until it is ready; avoid loading unbuilt cities except the viewed city. Open; notes/T-149.md.

- T-152 [P2] University attendance as a separate purpose: IPEDS directory/fall enrollment first, excluding exclusively online students; adult origins/timing, campus/system deduplication, and qualified Wikidata fallback for broader countries. Taking, 2026-10-11, demand-data subagent; notes/T-152.md; research/education-data.md and notes/T-150.md.

- T-155 [P2] Clarify that line riders and peak load are steady-state forecasts, not observed animated-boardings counters; consider an optional service-opening ramp-up if instant adoption remains confusing. Taking, 2026-10-11, intercity-display subagent; notes/T-155.md.
- T-156 [P3] Service-opening adoption ramp. On first service, store a line opening time and add a 20-minute perceived-time penalty that halves every 7 game days, with daily demand recomputes until day 28; existing saves start mature and later route/schedule edits do not restart it. Validate and tune before adoption. Open; notes/T-155.md.
