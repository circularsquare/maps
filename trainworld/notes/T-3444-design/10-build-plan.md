# T-3444 design, part 10: staged build plan

Agent trackdesign, 2026-11-21, after her answers (`answers-2026-11-21.md`). Revised 2026-11-28 by agent trackplan
after Gemini's two reviews (section "Review 11-28" at the end) and her 11-27 notes: the track rework comes before new
modes (DECISIONS 2026-11-27), and "we'll have to be careful with edit speed and crashes ... we can do [crash testing]
all at once closer to release". Vocabulary as in `01-model.md`.

## Where it stands, 11-28

Built on tw/track, 11-21 to 11-23, each with gates A and B byte identical and smoke 30/30 (ARCHIVE.md has each):
stages 1 (T-3948, on master), 2a (T-3949), 2b (T-3950), 3a1 (T-3951), 3a2 (T-3952), 3b (T-3953), 3c (T-3954), 3d
(T-3955), 3e (T-3956), 3f (T-3957), 3g (T-3958's re-saved starts, shipped at the merge), 4a1 (T-3959), 4a2 (T-3960),
4c (T-3962), 4e's platform picture (T-3964 in part), Add a track (T-4024, then T-4085: in the blueprint, up to 16),
T-4030's stages a to f (T-4031 to T-4036: tracks as records, runs on derived edges, paths name tracks, trains and
ribbons per track, platforms on tracks, routing to stop points), flip one track (T-4038), and T-4037's g1, g2 and g3
(her (b), 11-23: paths written on the track as stations cut it; stations as anchors in the sim). g3b stopped part way:
with the station tool switched to anchors 79 tests fail, in six groups (notes/T-4037.md on tw/track). The station tool
still cuts the track (kind 2), so nothing of g3 reaches a player yet.

Paused 11-23 to 11-27 (her word, other work first). Master moved 575 commits meanwhile (135 code files changed on both
sides since the 11-23 sync: network.ts 2,483 lines on master against 702 on tw/track, world.rs 1,202 against 238,
save.rs 177 against 1,161, stateRows.ts 1,117 against 46). That sync is running now (agent tracksync, under T-3444).

## Where it lands: the tw/track branch

The rework lands on a long-lived branch, **tw/track**, in its own worktree (`.tw-trees/_track`). Master keeps
shipping everything else and takes hotfixes. Rules:
- Master is merged into tw/track after each landed master batch that touches `sim/src/track`, `app/src/map`,
  `app/src/workers`, stations, lines or saves, so each sync stays about an hour. This lapsed during the pause (one
  sync on 11-23, then none for 575 commits); from the 11-28 sync on it is the orchestrator's standing step after a
  landing. A hotfix on master to code the rework replaced is ported by hand.
- Each sync renumbers tw/track's save level past master's (07-saves "Save level"), runs both cargo suites, gates A
  and B against master's figures on the saves that use nothing new, and smoke; one `edit_bench` NYC run when the
  sync touched the edit path (world.rs, wasm/edits.rs, workers, map/network.ts).
- tw/track merges into master at each release point, after that release's gates.
- Agents on the rework work in trees cut from tw/track; their landings are the orchestrator's commits to tw/track.
- Crash hunting waits for the crash round before the release (her 11-27). Crashes she or players report on a tw/track
  build are fixed as they come.

## The 11-28 sync: what it must keep

From the reviews and the check of 11-28 (master 25bfd66a5, tw/track 269568c20). Each item is a quiet wrong answer or a
quiet slowdown if missed, so each gets a test or a check line in the sync's report.

1. **Save level 44** holding all 47 blocks: master's 46 in order, then `LBN1` last; the app's `SAVE_LEVEL` with it.
   Her saves written by the tw/track build on 8800 (f382ee54, level 42: `... LBO1, LBE1, LBN1`) open (the blocks
   between are absent, read as none).
2. **`capacity/places.rs` (T-4444)**: `spots` lists a section under its piece by `k / 2`. On tw/track a capacity key
   is a derived edge (anchors' extra parts come from a free list, a 3+ track piece's keys are not `edge * 2 + dir`),
   so it must use the piece of the derived edge (`net.d_piece(k)`, which tw/track's capacity.rs already uses, or the
   piece stored with the key). Test: an update by touched records equals the whole update on a save with an anchor and
   on 4 tracks.
3. **`complexes.rs` learns `apart` (T-4048)**: master added "unless kept apart" to `twin.rs complex_of`; tw/track moved
   `complex_of` and demand's partition into `complexes.rs` (T-3962) before that, so demand would join pairs the player
   separated. Test: a separated pair stays two complexes in `Network::complexes` and `complex_of`.
4. **`sets_rates` (T-4272, world.rs)** strips the path-like fields of `LineData` before comparing; tw/track adds
   `picks`, which a station placement can change (`pick_ops`). Without stripping it, every station on running track
   goes the whole way (`Local::Whole`) and loses T-4272's answer at the click.
5. **`map/network.ts`**: master's T-4153 stages 1 to 10 (redraw by cut stretch and changed records) against tw/track's
   per-track rows (T-3955, T-4034). Master's by-record passes win; the per-track offsets ride in them. `?patch=check`
   clean on a 4-track fixture.
6. **Nudges (T-3742, T-4510, `nudge.rs`, `LOF1`)** merge as master's code: `LOF1` is by line index and slot, so nothing
   to remap; `nudge.rs`'s `is_twin` guard still holds while twins exist.
7. Merges clean by the code read, to confirm by the suites: `capacity/limits.rs` (T-4309, by line and resource only),
   `timing.rs`'s plan memo (T-4449, keyed by line, run index, the free run's duration, length and phase count, and the
   holds, never by path entries; tw/track has not touched timing.rs since 11-23).

## Release 1: what is left

In order; hours of real agent time, calibrated below. "After" is what it waits for.

| Stage | Task | What | Check | Hours | After |
|---|---|---|---|---|---|
| sync | T-3444 (tracksync) | Master's 575 commits into tw/track, the list above | cargo both suites; gates A, B against master's figures; smoke; 1 `edit_bench` NYC | 3-6 | running |
| g3b | T-4037 | The six groups left with anchors switched on (drawing and branching at a station, rebuild under an anchor, one-track, one-way legs, quad pairs, tests asserting split topology), with the branch review's anchor findings 4 to 10 (rebuild's `tram_after`, `passing_reach`, `culprits`; busway and ground-over-water checks; `lengthen_op`, `can_one_stop`; `anchor_at` reading track ways; a turn at an anchor only onto a track with a platform there; `delete_tracks`' sketch with `anchor_follow_ops`; the split's chainage from `split_plan`). Then the station tool places anchors (kind 3: `clockEdits.ts`, `previewStation`, `PlatformPicture.tsx`), `TOOL_ANCHORS` goes | cargo with the switch on, then off and removed; gates A, B; smoke; grep checks/INDEX.md for checks counting pieces | 3-5 | sync; T-4518's answer before the switch |
| T-4518 | (her call) | If yes: a click picks the stretch between the stations either side, and an edit on it cuts the track at their platform ends in the same edit | play: make the stretch between two stations single, then undo | 1.5-2.5 | g3b |
| g4 | T-4037 | Anchors drawn with their piece's heading and track count, ribbons clipped to their windows, a drag slides a station along its track; on master's T-4153 network.ts | screenshots (curve, 4 tracks, a line ending and one turning at an anchor); worker no slower than 1.2x | 1.5-2.5 | g3b |
| T-4520 | branch fixes | The branch review's findings 1 to 3, 11, 12: a flip's charge and its undo against the per-track price; Add a track on built two-way single keeps its lines running until Construct; a line left on a removed track saves readably; anchors in a frame conversion; a platform on a removed piece | a cargo test each | 2-4 | sync |
| T-4067 | connect an added track | A join naming one track across a node; draw a single from an added track's end to the track beyond | play: a double made 4, both new tracks joined, a line on them runs | 2-3 | sync |
| 4b | T-3961 | Old station nodes become anchors: paths renumbered into parts (exact under (b)), the 1 mm tolerance, and every node-keyed field moved with its station (07-saves step 6) | the reference saves, figures equal within the tolerance; a test listing each moved field | 2-3 | T-4037 |
| 4f | T-3965 | The starts rebuilt with stations as records | each start loads settled; rides within 2% | 1-2 | T-3961 |
| ids | T-4043 | Station ids apart from node ids (plan in notes/archive/trees/stationids-untracked/T-4043.md), its audit adding `LBK1`'s connections, `LAP1` and `apart`, bus keeps | its listed cargo test; gates A, B; the reference saves' figures | 3-5 | T-3961 |
| plat ids | T-4517 | Platform ids apart from node ids (T-4035 departure 1: today at most two per station, ids `2s`, `2s + 1`), so an island serves two tracks and a station on 4 tracks has four platform edges | cargo: a station of two islands on 4 tracks, saved, loaded, undone | 1.5-3 | T-4043 |
| 4d | T-3963 | Station tool templates (Side platforms, Island at zero width, Single platform), pointer and count pick the tracks, the preview of covered tracks and ways; placed as anchors | play on single, double and 4 tracks; strings | 3-4 | T-4517 |
| 4e | T-3964 | Join to station and Make a separate station on the platform picture; joining a pair the player separated (`LAP1`) clears the separation | play check; strings | 1.5-2 | T-4043 |
| 5a | T-3966 | Re-scoped (T-4153 did the rest on master, DECISIONS 2026-11-23): keep master's by-record drawing true with anchors and parts, ribbons keyed by a line's tracks with parts merged, so a station placed redraws no other bundle and no other line's ribbon | a station on built track on the real NYC save and the 2,647-station save no slower than master's figure for the same click; `?patch=check` clean | 1.5-3 | g4 |
| 5b | T-3967 | Line ribbons per track from paths; `besideSamples` replaced for the ribbons (lineBands.ts) and the network diagram (clockPost.ts, diagram/graph.ts), which also reads it | midtown ribbons as today; the diagram unchanged on real NYC; the Penn Station to Emeryville preview timed | 2-4 | T-3966 |
| syncs | (orchestrator) | Master into tw/track after each batch that touches it (about six through release 1) | as the branch rules | 4-8 in all | each master landing |

Side by side: T-4520 and T-4067 beside g3b (T-4067 and g3b both edit wasm/draw.rs: expect a small merge); T-4043
beside g4; T-3964 beside T-4517; T-3967 beside T-3963. The longest chain is sync, g3b, g4, T-3961, T-4043, T-4517,
T-3963, then the gates: about 26-43 h of it in a row.

Every stage keeps T-4267's rule (no step of an edit costs time that grows with the save, notes/T-4267.md section 3):
new rows and indexes are patched by record (anchors in `nodes_info`, platform rows, station ids), and a stage on the
edit path runs its budget row's `edit_bench` NYC. The Asia save is the gate's, below.

## Gates before the release 1 merge

| Gate | Task | What | Hours |
|---|---|---|---|
| Crash round | T-4519 | Her 11-27: crash testing all at once, near the release. The edit fuzzer (`sim/src/bin/edit_fuzz.rs`, from T-4116, with its `stale_runs` and `bad_paths` invariants and a save and reload every 10 ops) on the branch's final state: Amsterdam test, Vienna, real NYC, North America 383 and her Asia save (the big two are slow: over 5 minutes an episode of 60 ops), anchors on; her own saves played on a tw/track build (her 8800 save from the T-4116 panic first, if she still has it); a Gemini adversarial read of the whole branch diff against master. Every panic, refused reload or broken invariant gets a test and a fix; T-4116 closes in it | 3-5, the fixes in the fix tail |
| Edit speed | T-4521 | `edit_bench` on her Asia save (`asianita-day568`, Cores on Auto): the click run on the Tokaido trunk, a station built, Delete line, a stop click, against master's numbers from the same week; `perf_bench` on real NYC. Worse than master by more than 10% on any number goes to her as a trade-off with the exact figures (CLAUDE.md); the release run on all three saves settles it | in T-4521 |
| Outside the game | T-4026 | shonei-server's save check walks TWT7 (every body level) at the final header level, building on T-4435's level 43 (not deployed); `pipeline/readsave.py` reads TWT7 with parts. One server deploy on her yes, before release 1 ships | 2-3 |
| Starts | T-3958 | `realnet.py resave` in an empty `REALNET_OUT` (41 lines, exit 0), then `realnet.py ship` | 0.5-1 |
| Merge | T-4521 | The reference saves (Amsterdam test, Vienna, North America 383) and real NYC load with figures equal to master's where nothing new is used; smoke; her play on 8800 with the merged build; then the merge commit | 3-5 |

## Hours

**Release 1 from here: about 40-70 h**, plus the fix tail of 10-20 h. That is the stages above (about 33-57 h
with the syncs) and the gates (about 8-14 h). Calibration: T-4030's stages a to f were planned at 4-6 h each and took
about 1.1-1.5 h each from start to landing on 11-22; T-4038 took 50 minutes against 2-3 h. But g3, planned at 1.5 h,
met 98 failing tests and stopped part way, and the syncs are a kind of work with no history yet, so these figures
keep about half of each first guess, not a third. The 11-21 plan put all of release 1 at 73-102 h; about half of it
is built.

Releases 2 to 4 are unchanged in content and hours (below), each after its own crash round and Asia bench.

## Reference saves (decision 25)

Run in this order, lightest first: her Amsterdam "test" save (`saves/amsterdam-test-day-61.save`, odd layouts), the
community Vienna save (share 3znpvqx7, rev 2, 759 stations, day 337, tram single-track oddities; read-only, in
`saves/`), and North America day 383 (the big one). The real NYC save stays the developer parity save (part 5's gate
A); the 2,647-station save stays the big-save timing case of part 8; her Asia save (`asianita-day568`) is the edit
speed gate's, since her waits are on it.

## The stages as planned 11-21

Kept for the stage ids and for releases 2 to 4, which are unchanged. Release 1's rows 1 to 4c, 4e's platform picture and
Add a track are built (listed in "Where it stands" above); what is left of release 1 is the table "Release 1: what is
left", which replaces these rows.

| # | Stage | What lands | Check | Hours |
|---|---|---|---|---|
| 1 | Prep, on master. **Done 2026-11-21** (T-3948, notes/T-3948.md: the guard, the baselines, the saves) | T-1183 (the fixed-rate line path out of 69 tests); the save guard for the new magic; part 8's seven baseline timings, when cores are free; the Vienna save fetched | cargo suites; timings in notes | 4-6 |
| 2a | Bundles as saved records | A bundle id and slot on today's edges (`LBN1`, level 35), set by drawing quad and doubling up, found once on old saves by today's twin rule; `twin_at`, `quad_beside`, `quadPairs.ts`, `Alongside` read it and go | real save figures unchanged; t3333 and t3348 checks | 3-4 |
| 2b | The three cases in the tools. **Done 2026-11-21** (T-3950); her 11-21 call changes two parts: the second click to one double becomes one track (3e), the Select bar's "4 tracks" becomes Add a track (T-4024) | Track bar 1, 2 and More (4 for now); 4 tracks picked whole, a second click picks one double; Select bar's 4 replaces Double up; one shared pointer pick for every tool; titles "Single track", "Double track", "4 tracks" | play check on 4 tracks and double | 3-4 |
| 3a1 | Tracks as their own edges: the compile | Authored bundles with tracks, nodes with joins; the compile to the derived graph, one track per edge, stable ids, the node pitch, capacity's keys; lanes' cap and the inferred-way code out. Stations still nodes here | sim tests on the small saves | 4-6 |
| 3a2 | Partners, turns and parity | Partners stored (the loader pairs each old double), turn moves at stop nodes and bare ends, routing and capacity on derived keys checked whole | **parity gates A and B** (part 5) | 3-4 |
| 3b | The new save and old saves | TWT7 (level 36) reader and writer; old saves converted on load, old bytes kept | the three reference saves load to parity, Amsterdam first | 3-5 |
| 3c | Paths in track spans | Paths as (track, direction) spans; return by partner; routed returns stored; routing on authored tracks; `back_lines` out | parity A and B; perdir and t3565 checks | 4-6 |
| 3d | Drawing each track | Per-track rows to the app; each track at its own offset; `legShifts`, the taper and `singleLeadStrokes` out; wiki "Tracks" page | screenshots: joins, branches, flyovers, both running sides | 5-6 |
| 3e | Tools on tracks | A single from a double's end extends the track under the pointer, no automatic merge; joins stored from the pointer's pick; second click straight to one track on any count (2b's one-double step goes); the track picture; delete or flip one track; the Delete tool takes the whole run | play check: extend each track of a double, merge two singles manually, delete one side, one track of 4 picked in one click | 4-5 |
| 3f | Long selections and ways | A track's way spreads to the nearest branch points (T-3662); double click traces one track to the next branch points, Shift+click extends; set a selection's way in one edit | play check on a long one-way leg | 3-4 |
| 3g | Starts on the new format | `realsave` writes bundles; the sandbox starts rebuilt, loading settled as today | each start loads settled | 2-3 |
| 4a1 | Platforms and stations: sim | Platform and station records (level 37); stations stop splitting track; stop points compiled; the station-as-node fields out | parity where nothing new is used | 4-6 |
| 4+ | Add a track (T-4024, after 4a1) | The Select bar's Add a track and the track picture's + replace 2b's "4 tracks" (Double up's op goes); the new track's way, partner, ends joined at the run's nodes, no crossovers along it; the per-track price; the refusal reasons; up to 16 | play check: a double to 4 tracks in two clicks runs a line on the new pair; a refusal names its reason | 3-5 |
| 4a2 | Smart allocation | Each way's stop point picked at the click and stored, re-picked only when it goes; the "trains can't pass here" flag; `perdir` way stations and `station_beside` out | her split legs get a two-way stop in one click; T-3605 and T-3711 cases unchanged | 3-4 |
| 4b | Old stations converted | Station nodes to stations and platforms; each stop's way picks stored as today's rule makes them | the reference saves, figures equal | 3-4 |
| 4c | Demand and the stop list | Demand snapshot rows from stop points (part 6); the one complex rule in the sim; the stop list's track chip | packed arrays byte for byte on NYC, then rail trips a day | 4-6 |
| 4d | Station tool with templates | Side platforms, Island (zero width), Single platform; pointer and count pick the tracks; joining by overlap; the preview of covered tracks and their ways | play check on single, double and 4 tracks; strings | 5-6 |
| 4e | Station panel | The platform picture (add, remove); Join to station, Make a separate station | play check; strings | 4-5 |
| 4f | Starts with stations | The starts rebuilt with stations as records, keeping today's stations | each start loads settled, rides within 2% | 2-3 |
| 5a | Drawn network per bundle | Drawn records keyed by bundle chunk and chainage; rebuild only what an edit touched | **T-2767**: a station on built track redraws no other bundle, worker 30 ms or less, shows up in 400 ms median or less on the 2,647-station save (1,925 ms today); **T-2701** | 4-6 |
| 5b | Line ribbons per track | Lines side by side on one track laid out from paths, `besideSamples` out; **T-763** timed | ribbons on midtown; the Penn Station to Emeryville preview | 3-4 |
| 6a | More tracks | The More button's count stepper 3 to 16 and its two layouts (presets of ways and pairs), drawn in one stroke | play check: 6 tracks drawn by direction | 3-4 |
| 6b | By direction | Drawing by direction, peel-off and the flying split (T-3348's stages 2-3); AA/BB and ABBA themselves are only ways, settable from 3f | locals and expresses on AA/BB, no flat crossing at the split | 5-6 |
| 6c | One line on several tracks | The hidden twin line out; a line runs as patterns over its tracks (part 5); panels read one line | the t3333 save gives the same trains and rides | 3-5 |
| 7a | Single-track stations as real loops | The passing loop at a single-track station becomes real track | parity B with the reviewed diffs of part 5 | 3-5 |
| 7b | Passing tracks as real track | The passing tracks button builds them, with the passing purpose; overtakes read the tracks; the `PASSING` bit out (the terminal bit stays) | T-2231's line: express and local times | 4-6 |
| 7c | Track purpose | Any line, Passing track, Only for {lines}, on a track or a long selection; routing, overtakes and capacity read it | play check: an express pair kept to expresses | 4-6 |
| 8a | Crossovers as single track | A single drawn between two tracks of a bundle lays the crossover's S; scissors as two; old T-338 crossovers as singles; turnback crossovers laid with the stop edit; turn moves on double out, `turnbackStrokes` and `add_crossover` out; the sandbox starts rebuilt with their crossovers | a line turning at a terminus: figures equal; delete a crossover, lines refused or rerouted | 4-6 |
| 8b | Build crossovers? | When a line can't reach a track for lack of crossovers, the refusal offers to build them: the placer picks spots, a blueprint with its cost, one click | play check: the fifth track's line, a quad stop on the far pair | 4-6 |
| 9a | Island platforms with spread | The island template spreads the tracks for a 6 m island, S from target max speed (flyover speed where room is short) | island on double at 80 and 160 km/h: S lengths, speed never below target | 5-6 |
| 9b | Platform faces | Getting on and getting off sides (the Spanish solution); no effect on figures | panel and drawing check | 2-4 |
| 9c | The island change minute | A change between the two edges of one island counts 1 minute instead of 2 (decision 23) | demand kernel test; ceiling tests unchanged | 2-3 |
| 10 | Tapering and drawn platforms | Platforms between two bundles' tracks; the drawn platform (T-3859) | wedge station at a Y | 4-6 |
| 11a | Exact stops: storage and switching | Stop points in running order; switching both ways with the notice; sim and demand read them | a line listed exactly as its simple stops gives the same runs and arrays | 5-6 |
| 11b | Exact stops: the circles | Circles per stop point, pushed apart along their platform, tinted by level, numbered; clicks, waypoints, touch | play check on 4 tracks and a Spanish station | 5-6 |
| 12a | A track and platform extract | OSM platforms, stop areas, `railway:track_ref` and preferred direction; one region measured first | the region's counts against taginfo | 4-6 |
| 12b | Importing them | Platform shapes and track order for the sandbox starts and the 34 extract cities | the starts build; rides within 2% | 4-6 |
| 12c | One station per side-by-side place | In the starts, a local and express stop become one station with four platforms (decision 26, low priority) | each start loads settled | 2-3 |
| 13 | Quick-place and copy layouts | Place a station on all of a bundle's tracks in one click (the islands of AA/BB too); copy a station's layout to another place | play check | 4-6 |

## Releases (each a merge of tw/track into master)

1. **Stages 1-5 and Add a track, the foundation**: about half built; about 40-70 h left with the gates (table "Release 1:
   what is left"), plus 10-20 h of fixes. Players see: single, double and more tracks; one track pickable, deletable,
   flippable; Add a track, up to 16, and a way to connect it; long selections; stations that sit on the track without
   cutting it; the station tool with three templates and a preview; Join to station and Make a separate station.
   The sandbox starts are rebuilt inside it. The causes of the single/double joining bugs (research/code.md classes A
   to E) are gone, so those reports should thin out once the fix tail is done.
2. **Stages 6-8, more tracks** (about 30-44 h): any count drawn in one stroke, the Build crossovers? offer, drawing
   by direction, real passing loops and passing tracks, track purpose, crossovers as single track.
3. **Stages 9-11 and 13, stations** (about 27-37 h): islands with spread, the Spanish solution, the island change
   minute, tapering and drawn platforms, exact stops, quick-place and copied layouts. Stage 11 needs only stage 4, so
   it can move up.
4. **Stage 12, real stations from OSM** (about 10-15 h), low priority, any time after stage 4.

## What later master changes mean for each stage

| Master change | Stages | What it means |
|---|---|---|
| Save levels 35 to 43 (`LBT1` to `LOF1`) | sync, T-3961, T-4043, T-4026 | Level 44 at the sync; 4b moves the node-keyed fields with their station; T-4043's audit takes `LBK1` and `LAP1`; the server check learns the final level (07-saves) |
| T-4444 capacity by touched records (`places.rs`) | sync, 7a, 7b, 8a | Sections listed under their piece by the derived edge, not `k / 2`; passing loops, passing tracks and crossover singles add sections through the same index |
| T-4309 track limits per line (`limits.rs`) | 6c | Keys by line and resource; a line's patterns put their resources under the one line |
| T-4449 retime's plan memo (`timing.rs`) | 6c, 7a, 7b | Keyed by line and run index: at 6c a pattern needs its own run index in the key, or two patterns share a plan |
| T-4504 overtake plans (running, planbest) | 7a, 7b | Overtakes on real passing track go through the same plan cache |
| T-4272 click confirm, T-4442 owed together, T-4445, T-4446, T-4464 | sync, g3b, T-4067, T-3963, T-3964 | Every new edit (anchor placement with its `Op::Line` per line, a join, a template's batch, Join to station) stays answered at the click by `commit_local`; `sets_rates` strips `picks` |
| T-4153 drawn network by cut stretch, T-4429, T-4262 and T-4274 (GPU slots) | sync, g4, T-3966, T-3967, 8a | 5a shrinks to keeping that true with anchors and parts; 5b writes per-track ribbons into the per-line slots (trainSlots.ts, strokeSlots.ts, strokePost.ts; strokeCache.ts is the framebuffer cache and not one of them); 8a's `turnbackStrokes` goes through the same passes |
| T-4291, T-4299, T-4300, T-4360, T-4284 (rows, post, complexes, page by record) | g4, T-4043, T-4517, T-3963, T-3964 | New rows patched by record; `ComplexBuilder` learns one station with several stop nodes at T-4043 |
| T-4048 separated stations (`apart`, `LAP1`) | sync, T-3961, T-4043, T-3964 | `complexes.rs` honours it; moved with the station in 4b; by station id at T-4043; Join to station clears a separation |
| T-4080 kept bus stops (`LBK1`) | T-3961, T-4043 | Their connections name station nodes: moved in 4b, by station id at T-4043 |
| T-3742, T-4510 nudges (`LOF1`); T-4511 single trains (P3) | sync, 6c, 7b | Nothing at the sync (by line and slot). 6c: the twin guard in `nudgeable` goes, and a row's offset moves every pattern of the line; `follow_pairs` and the nudge's `anchor` and `group` read lines, not patterns. 7b: real passing tracks keep marking `FOLLOW_PASSING`, so lines timed together still nudge together. T-4511's per-train offsets, if built first, need a rule for a line on several tracks at 6c |
| T-4514 waits from the real gaps (running, realgaps) | a sync, 7a | Comes in at a sync; 7a's reviewed differences against parity B are taken again after it |
| T-4267 D2, the demand snapshot by record (T-4461) | 4c (built) | tw/track's demand rows take complexes from the sim's partition, which walks every station per snapshot; D2 patches it by record in the sim |
| T-4435 server check at level 43 (not deployed) | T-4026 | One server change and one deploy for both |
| Monorail and cable cars (T-4479 to T-4498, after the rework, her 11-27) | release 2 on | Built on bundles and tracks: a new kind on a bundle, as the busway is |

## Order and what each stage unblocks among the paused tasks

| Task | What it is | Where it goes |
|---|---|---|
| T-1183 | Remove the fixed-rate line path | Stage 1, on master, before the branch |
| T-3365 | What single, double and quad mean | Answered by stages 2-3 (the three cases, second click, the wiki page); close with stage 3 |
| T-3348 | Quad platforms (stages 2-5 paused) | Stage 1 stays (built). Its stages 2-3 are stage 6b; its stage 4 (quad stations) is stages 4d-4e and 13. Close with stage 6 |
| T-3335 | Line panel on both quad pairs | Moot once the twin line is gone (stage 6c); close then |
| T-340 | Quad end-node stations and express side | End-node stations are a platform at a bundle end (stage 4); the express side is a layout's ways (stage 6b). Close with stage 6 |
| T-3859 | Custom drawn platforms | Stage 10 |
| T-3662 | Switching one kerbside piece breaks lines | Stage 3f (a way spreads to the branch points); close then |
| T-3353 | Second single from a double's end refused | Joins by track id (3a) and extending one track (3e); close with 3e |
| T-1948 | Node drag lead-in capped too short | Free to build after 3e |
| T-3036 | Debug overlay for street gaps | Freed now (decision 27) |
| T-3354 | One-track platform jumps sides when dragged | A platform names its track; close with stage 4 |
| T-3264 | Trains drawn through each other | Stage 3d removes the cross-track case; her hunch (a local and an express on one track) stays its own task |
| T-3079 | Exact stop ordering back | Stage 11 |
| T-2767, T-2701, T-763 | Edit-speed requirements | Stage 5's checks; close T-2767 and T-2701 at 5a, T-763 at 5b |

## Review 11-28

Gemini read the plan against master (`artifacts/trainworld/gemini/T-3444-plan-review.md`) and the branch's code
(`artifacts/trainworld/gemini/T-3444-branch-review.md`). Each finding checked against master 25bfd66a5 and tw/track
269568c20 by agent trackplan.

**The plan review**

| # | Finding | Verdict | What changed |
|---|---|---|---|
| 1 | Master at level 43 against the plan's 36-41; the guard inverted | Partly. 07-saves's table was stale (tw/track is at 42 since the 11-23 sync, not 36-41), and tw/track refuses master's 43 saves until the sync. No misread either way: the header level is only the guard, blocks read by name in order, TWT7 by its body level, and master refuses any TWT7 by its magic (T-3948) | 07-saves "Save level" rewritten; sync item 1 (level 44, 47 blocks) |
| 2 | `LBT1`, `LBK1`, `LAP1`, `LBO1` missing from 07-saves | Partly. `LBT1`, `LBK1`'s connections and `LAP1` are by station node and `LBK1` does name a station (07-saves said bus stops never did). They stay right through T-4037 and T-3961 because anchors are nodes; 4b's converter must move each with its station, and T-4043's audit must list them. `LBO1` and `LOS1` are by stop place in the file, by node only in memory | 07-saves blocks table and old-save step 6; T-3961 and T-4043 name them |
| 3 | `places.rs`, `limits.rs`, the plan memo regress or break | Partly. `places.rs`'s `k / 2` stands: wrong pieces on tw/track's keys. `limits.rs` keys by line and resource: wrong. The plan memo keys by line, run index and the free run, never by entries, and tw/track has not touched timing.rs since the base: wrong | Sync items 2 and 7; per-stage table |
| 4 | 5a written against the old network.ts, the wrong file, under-sized | Partly. T-4153 did most of 5a's goal on master (DECISIONS 2026-11-23); strokeCache.ts is the framebuffer cache (right). What is left is smaller, not bigger: keep T-4153's passes true with anchors and parts | 5a re-scoped to 1.5-3 h; T-3966's files fixed; the network.ts conflict is the sync's (item 5) |
| 5 | No hours for merging master | Stands. 575 commits, 135 code files on both sides | The sync (3-6 h) and later syncs (4-8 h) in the plan; the branch rule made the orchestrator's standing step |
| 6 | T-4026 tagged after the release it must precede | Stands | T-4026 is a gate before the merge, after T-3963 (the last format change) |
| 7 | Nudges missing: release 1 drops them, 6c and 7b break timed pairs | Partly. Release 1: wrong once synced (`LOF1` by line and slot, `nudge.rs` reads phases, not track). 6c and 7b: stand | Per-stage table rows for 6c and 7b; T-3970, T-3972 notes |
| 8 | T-4116's panic not a gate | Stands | The crash round (T-4519), her 11-27 note; T-4116 closes in it |
| 9 | Complexes in the sim regress `ComplexBuilder` and drop `apart` | Partly. 4c is built and feeds demand only; the map keeps `ComplexBuilder` (T-3962 departure 2), so no edit-speed loss there. `apart` missing from `complexes.rs` stands; the sim partition per snapshot is whole-save (D2's) | Sync item 3; per-stage table (D2) |
| 10 | Add a track ships with no way to connect | Stands | T-4067 in release 1 |
| 11 | T-3963 lacks platform ids; T-4037's (a)/(b) unresolved | First half stands (T-4035 departure 1; T-4043 keeps platform ids node-local by plan). Second half wrong: her (b) on 11-23 (DECISIONS), g1 and g2 built on it | New stage T-4517 before T-3963 |
| 12 | Deleting `corridors.ts` breaks the network diagram | Stands: the diagram (11-03) calls `besideSamples` (clockPost.ts, diagram/graph.ts), which the design missed | 5b covers the diagram |
| 13 | T-3973's goal keeps the Passing track purpose she dropped | Stands | T-3973's Goal rewritten |
| 14 | 3b and 4b quoted lower here than in 07-saves | Partly. 3b is built. 4b under (b) is a renumbering plus the field moves | 4b at 2-3 h |

**The branch review** (all on tw/track only; 4 to 10 reach players only once the station tool places anchors)

| # | Finding | Verdict | Goes to |
|---|---|---|---|
| 1 | Flipping one track of a built double charges 20% of its price, undo gives none back | Partly. The charge is the rise in built value and the per-track price (T-4024) prices a same-way pair above an A B pair, so a flip to A A costs as drawing A A would; whether undo gives the charge back was not traced | T-4520 |
| 2 | Add a track on built two-way single stops its lines and charges an anchor's platform | Stands by the code read: `segment_ops` makes the built track one-way and the planned track the other way, so a two-way line's return lies on planned track | T-4520 |
| 3 | A line left on a removed track saves bytes its own build refuses | Stands by the code read: `rekeyed` gives None, `span_of` no slot, `path_out` writes slot 0, `path_in` counts it unread, so the save opens as "newer" | T-4520; the crash round's reload check |
| 4 | Rebuild ignores anchors | Stands; most of it is g3b's group 2, plus `tram_after`, `passing_reach`, `culprits` | g3b |
| 5 | Anchors skip the busway and ground-over-water checks | Stands: `check_station` returns after `check_anchor`, before the water test; `check_anchor` reads no busway | g3b |
| 6 | `branch_end`, `lengthen_op`, `can_one_stop` need ports | Stands; `branch_end` is g3b's group 1, the other two new | g3b |
| 7 | `anchor_at` ignores the tracks' ways | Stands: the running side's rule only, so after a flip a one-track anchor's platform lands on the wrong track | g3b |
| 8 | A turn at an anchor onto a track with no platform, on 3+ tracks | Partly: on two tracks it is the design's station rule; on 3+ tracks with platforms on the outer two it lets a train turn onto a bare track. Rule: a turn at an anchor goes onto a track with a platform there | g3b, before T-3963 |
| 9 | `delete_tracks`' sketch skips `anchor_follow_ops` | Likely, not traced | g3b |
| 10 | `Op::Split` projects again instead of taking the snapped chainage | Likely on a hairpin, not traced | g3b |
| 11 | Frame conversion leaves an anchor's middle unscaled | Likely (TWT7 keeps a game's old frame), not traced | T-4520 |
| 12 | A platform on a removed piece saves as a slot platform | Likely, not traced | T-4520 |

**Found in the check, in neither review**: `sets_rates` and `picks` (sync item 4); g3b's question about picking
between stations was never put to her (T-4518); her 8800 saves at tw/track's level 42 (sync item 1); the README's
"three open calls" were answered 11-21 and it still said nothing was built.
