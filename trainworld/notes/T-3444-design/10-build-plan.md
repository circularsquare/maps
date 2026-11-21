# T-3444 design, part 10: staged build plan

Agent trackdesign, 2026-11-21, after her answers (`answers-2026-11-21.md`). Vocabulary as in `01-model.md`. Every
stage is one landable piece of 2-6 hours of real agent time, checked on its own, with the docs, wiki and strings of
what it changes. The tasks to file are in `tasks-to-file.md`.

## Where it lands: the tw/track branch

The rework lands on a long-lived branch, **tw/track**, in its own worktree (orchestrator's call on her decision 4,
maps-98: she wants hotfixes to keep shipping without the half-finished rework). Master keeps shipping everything else
and takes hotfixes. Rules:
- Stage 1 lands on **master**, before the branch is cut, and ships in a release before the first merge back: its save
  guard lets today's game say "saved by a newer version" for a tw/track save instead of failing.
- Master is merged into tw/track after each landed batch that touches track, station, line or save files, so the
  merge back stays small; a hotfix on master to code the rework replaced gets ported by hand into tw/track.
- tw/track merges into master at each release point (after stages 5, 8, 11 and 12 below), each after a smoke and a
  run on the reference saves.
- Agents on the rework work in trees cut from tw/track; their landings are the orchestrator's commits to tw/track.

## Reference saves (decision 25)

Run in this order, lightest first: her Amsterdam "test" save (`saves/amsterdam-test-day-61.save`, odd layouts), the
community Vienna save (share 3znpvqx7, rev 2, 759 stations, day 337, tram single-track oddities; read-only; a fetch on
2026-11-21 got "rate limited", so stage 1 fetches it into `saves/`), and North America day 383 (the big one). The real
NYC save stays the developer parity save (part 5's gate A); the 2,647-station save stays the big-save timing case of
part 8.

## Calibration

Past track work against its estimates: T-3137 (double track as lanes, four stages) was estimated at 7-8 agent-days and
landed in about a day (C1-C2 committed 11-11 00:27, C3-C4 02:29); T-3333 and T-3332 were each estimated 11-16 h and
each landed within a day of being scoped; T-3348 stage 1 was estimated 5-8 h and its sim was built the same day. So
first guesses run about twice the build time. But T-3137's day of work was followed by 26 single-track fixes over the
next nine days (research/code.md class A), and per-direction stops by 11 more (class D), so the plan adds a **fix
tail of 10-20 h** after stages 3 and 4.

The stages come to **about 140-200 hours, plus the fix tail**. That is more than the first draft's 110-160: her
answers added long selections, track purpose, the crossover offer, station templates and two later stages.

## The stages

| # | Stage | What lands | Check | Hours |
|---|---|---|---|---|
| 1 | Prep, on master | T-1183 (the fixed-rate line path out of 69 tests); the save guard for the new magic; part 8's seven baseline timings, when cores are free; the Vienna save fetched | cargo suites; timings in notes | 4-6 |
| 2a | Bundles as saved records | A bundle id and slot on today's edges (`LBN1`, level 35), set by drawing quad and doubling up, found once on old saves by today's twin rule; `twin_at`, `quad_beside`, `quadPairs.ts`, `Alongside` read it and go | real save figures unchanged; t3333 and t3348 checks | 3-4 |
| 2b | The three cases in the tools | Track bar 1, 2 and More (4 for now); 4 tracks picked whole, a second click picks one double; Select bar's 4 replaces Double up; one shared pointer pick for every tool; titles "Single track", "Double track", "4 tracks" | play check on 4 tracks and double | 3-4 |
| 3a1 | Tracks as their own edges: the compile | Authored bundles with tracks, nodes with joins; the compile to the derived graph, one track per edge, stable ids, the node pitch, capacity's keys; lanes' cap and the inferred-way code out. Stations still nodes here | sim tests on the small saves | 4-6 |
| 3a2 | Partners, turns and parity | Partners stored (the loader pairs each old double), turn moves at stop nodes and bare ends, routing and capacity on derived keys checked whole | **parity gates A and B** (part 5) | 3-4 |
| 3b | The new save and old saves | TWT7 (level 36) reader and writer; old saves converted on load, old bytes kept | the three reference saves load to parity, Amsterdam first | 3-5 |
| 3c | Paths in track spans | Paths as (track, direction) spans; return by partner; routed returns stored; routing on authored tracks; `back_lines` out | parity A and B; perdir and t3565 checks | 4-6 |
| 3d | Drawing each track | Per-track rows to the app; each track at its own offset; `legShifts`, the taper and `singleLeadStrokes` out; wiki "Tracks" page | screenshots: joins, branches, flyovers, both running sides | 5-6 |
| 3e | Tools on tracks | A single from a double's end extends the track under the pointer, no automatic merge; joins stored from the pointer's pick; second click to one track; delete or flip one track | play check: extend each track of a double, merge two singles manually, delete one side | 4-5 |
| 3f | Long selections and ways | A track's way spreads to the nearest branch points (T-3662); double click traces one track to the next branch points, Shift+click extends; set a selection's way in one edit | play check on a long one-way leg | 3-4 |
| 3g | Starts on the new format | `realsave` writes bundles; the sandbox starts rebuilt, loading settled as today | each start loads settled | 2-3 |
| 4a1 | Platforms and stations: sim | Platform and station records (level 37); stations stop splitting track; stop points compiled; the station-as-node fields out | parity where nothing new is used | 4-6 |
| 4a2 | Smart allocation | Each way's stop point picked at the click and stored, re-picked only when it goes; the "trains can't pass here" flag; `perdir` way stations and `station_beside` out | her split legs get a two-way stop in one click; T-3605 and T-3711 cases unchanged | 3-4 |
| 4b | Old stations converted | Station nodes to stations and platforms; each stop's way picks stored as today's rule makes them | the reference saves, figures equal | 3-4 |
| 4c | Demand and the stop list | Demand snapshot rows from stop points (part 6); the one complex rule in the sim; the stop list's track chip | packed arrays byte for byte on NYC, then rail trips a day | 4-6 |
| 4d | Station tool with templates | Side platforms, Island (zero width), Single platform; pointer and count pick the tracks; joining by overlap; the preview of covered tracks and their ways | play check on single, double and 4 tracks; strings | 5-6 |
| 4e | Station panel | The platform picture (add, remove); Join to station, Make a separate station | play check; strings | 4-5 |
| 4f | Starts with stations | The starts rebuilt with stations as records, keeping today's stations | each start loads settled, rides within 2% | 2-3 |
| 5a | Drawn network per bundle | Drawn records keyed by bundle chunk and chainage; rebuild only what an edit touched | **T-2767**: a station on built track redraws no other bundle, worker 30 ms or less, shows up in 400 ms median or less on the 2,647-station save (1,925 ms today); **T-2701** | 4-6 |
| 5b | Line ribbons per track | Lines side by side on one track laid out from paths, `besideSamples` out; **T-763** timed | ribbons on midtown; the Penn Station to Emeryville preview | 3-4 |
| 6a | More tracks | The More button's count up to 16 and layout; adding tracks to a selection, unconnected; removing one; the price per track | play check: add a fifth track, a line refused until crossovers | 4-6 |
| 6b | By direction | AA/BB and ABBA; drawing by direction, peel-off and the flying split (T-3348's stages 2-3) | locals and expresses on AA/BB, no flat crossing at the split | 5-6 |
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

1. **Stages 1-5, the foundation** (about 70-97 h). Players see: single, double and more tracks; one track pickable,
   deletable, flippable; a single extends one track of a double; long selections; the station tool with three
   templates and a preview; stations placed on built track showing up in under half a second. The sandbox starts are
   rebuilt inside it. The causes of the single/double joining bugs (research/code.md classes A to E) are gone, so
   those reports should thin out once the fix tail is done.
2. **Stages 6-8, more tracks** (about 31-46 h): any count, added tracks with the Build crossovers? offer, AA/BB and
   ABBA, real passing loops and passing tracks, track purpose, crossovers as single track.
3. **Stages 9-11 and 13, stations** (about 27-37 h): islands with spread, the Spanish solution, the island change
   minute, tapering and drawn platforms, exact stops, quick-place and copied layouts. Stage 11 needs only stage 4, so
   it can move up.
4. **Stage 12, real stations from OSM** (about 10-15 h), low priority, any time after stage 4.

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

## Side by side

Stages 2 to 5 mostly run in order, since each changes the records the next reads, but within them: after 3a2, the
pieces 3b (save), 3c (paths) and 3d (drawing) own different files and can run at once; after 4a2, 4b, 4c, 4d and 4e
can. From stage 6 on: 6c beside 6a and 6b; 9b and 9c beside 9a; 11 beside 7 and 8 (not beside 6c, which also edits
the line panel); 12a any time (pipeline only). `tasks-to-file.md` lists each stage's files.
