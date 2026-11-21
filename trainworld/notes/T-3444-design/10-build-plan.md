# T-3444 design, part 10: staged build plan

Agent trackdesign, 2026-11-21. Vocabulary as in `01-model.md`. Every stage is one landable piece of 2-6 hours of real
agent time, checked on its own, with the docs, wiki and strings of what it changes. Nothing starts before she has
agreed the design.

## Calibration

Past track work against its estimates: T-3137 (double track as lanes, four stages) was estimated at 7-8 agent-days
and landed in about a day (C1-C2 committed 11-11 00:27, C3-C4 02:29); T-3333 (a line on both quad pairs) and T-3332
(one-track stations) were each estimated 11-16 h and each landed within a day of being scoped; T-3348 stage 1 was
estimated 5-8 h and its sim was built the same day. So our first guesses run about twice the build time.

But a build is not the whole cost. T-3137's day of work was followed by 26 single-track fixes over the next nine days
(11-10 to 11-18, research/code.md class A), and per-direction stops by 11 more (class D). The rework is bigger than
either, so the plan adds a **fix tail of 10-20 h** after stages 3 and 4, spread over player reports.

The parts' own first guesses add up to about 175-255 h (tools 03, stops 04, sim 05, demand 06, saves 07, import 09).
Cut toward past build times, with stages 3 and 4 kept at a margin (they touch every consumer of the track model), the
stages below come to **about 110-160 hours, plus the 10-20 h fix tail**.

## The stages

| # | Stage | What lands | Check | Hours |
|---|---|---|---|---|
| 1 | Prep | T-1183 (the fixed-rate line path out of 69 tests); part 8's seven baseline timings on master, when cores are free; today's build told that the new save magic means "saved by a newer version" (part 7), shipped at least one release before stage 3 | cargo suites; timings in notes | 4-6 |
| 2a | Bundles as saved records | A bundle id and slot on today's edges (`LBN1`, level 35), set by drawing quad and doubling up, found once on old saves by today's twin rule; `twin_at`, `quad_beside`, `quadPairs.ts`, `Alongside` read it and go | real save figures unchanged; t3333 and t3348 checks | 3-4 |
| 2b | Quad as one thing in the tools | Quad picked whole, named "Quad track"; a second click picks one pair; Select bar 4 replaces Double up; one shared pointer pick for every tool | play check on quad and double | 3-4 |
| 3a | Tracks as their own edges: the compile | Authored bundles with tracks and partners, nodes with joins; the compile to the derived graph, one track per edge, stable ids, the node pitch, compile-made turn moves; lanes' cap and the inferred-way code out. Stations are still nodes in this stage | **parity gates A and B** (part 5): the real NYC save and the small saves give the same paths, capacity and rides as master | 4-6 |
| 3b | The new save and old saves | TWT7 (level 36) reader and writer; old saves converted on load with the old bytes kept; round trips | the three test saves (call 25) load to parity | 3-5 |
| 3c | Paths in track spans | Lines store paths as (track, direction) spans; the return by partner lookup; routed returns stored; routing on authored tracks; `back_lines` out | parity A and B; perdir and t3565 checks | 4-6 |
| 3d | Drawing each track | Per-track rows to the app; each track drawn at its own offset; `legShifts`, the taper and `singleLeadStrokes` out; wiki "Tracks" page | screenshots: joins, branches, flyovers, both running sides | 5-6 |
| 3e | Tools on tracks | Joins stored by the drawing tool from the track the pointer picked; second click picks one track; delete or flip one track (the whole track through its continues, T-3662) | play check: branch off each track, delete one side, lines rerouted or refused | 4-5 |
| 3f | Shipped starts on the new format | `realsave` writes bundles; the sandbox starts rebuilt so they still load settled (London 0.6 s natively, not 43 s; part 9) | each start loads settled, a screenshot each | 2-3 |
| 4a | Platforms and stations: sim | Platform and station records (level 37); stations stop splitting track; stop points compiled; the simple-stops rule with preferred stop points; the "trains can't pass here" flag for one-track stations on single track; `perdir` way stations, `station_beside`, the station-as-node fields out | parity where nothing new is used; her split legs as one station | 4-6 |
| 4b | Old stations converted | Station nodes to stations and platforms, with frozen copies of today's pairing rules, so the pairs today's 300 m rule pairs become one station | the three test saves, figures equal; one-track stations as before | 3-4 |
| 4c | Demand and the stop list | Demand snapshot rows from stop points (part 6); the one complex rule in the sim; the stop list's track chip | packed arrays byte for byte on NYC, then rail trips a day (12,581,681) | 4-6 |
| 4d | Station tool | Count icons, the pointer picks tracks, joining by overlap with the ghost saying so | play check on double, quad, single; strings | 4-5 |
| 4e | Station panel | The platform picture (add and remove a platform); Join to station, Make a separate station | play check; strings | 4-5 |
| 4f | Shipped starts with stations | One station per place where tracks run side by side (call 26); pieces rejoined across stations | each start loads settled; rides within 2% | 2-3 |
| 5a | Drawn network per bundle | Drawn records keyed by bundle chunk and chainage; rebuild only what an edit touched | **T-2767**: a station on built track redraws no other bundle, worker 30 ms or less, shows up in 400 ms median or less on the 2,647-station save (1,925 ms today); **T-2701**: a far edit costs the same on any network size | 4-6 |
| 5b | Line ribbons per track | Lines side by side on one track laid out from paths, `besideSamples` out; **T-763** timed | ribbons screenshot on midtown; the Penn Station to Emeryville preview | 3-4 |
| 6a | Any number of tracks | Track counts in the Advanced row up to 16; add a track at a chosen side, remove one; the price rule per track | play check: a fifth track | 4-6 |
| 6b | Quad by direction | AA/BB and ABBA presets; drawing by direction, peel-off and the flying split (T-3348's stages 2-3) | locals and expresses on AA/BB, no flat crossing at the split | 5-6 |
| 6c | One line on several tracks | The hidden twin line out; a line runs as patterns over its tracks (part 5); panels read one line | the t3333 save gives the same trains and rides | 3-5 |
| 7a | Single-track stations as real loops | The passing loop at a single-track station becomes real track | parity B with the reviewed diffs of part 5 | 3-5 |
| 7b | Passing tracks as real track | The passing tracks button builds the preset; overtakes read the tracks; the `PASSING` bit out (the terminal bit stays); starts rebuilt | T-2231's line: express and local times | 4-6 |
| 8 | Crossovers | The crossover tool; crossover records compiled to short diagonal edges (today's shape); old pair-to-pair crossovers converted; turn moves drawn from the sim's answer, `turnbackStrokes` out | crossover on double and quad; a line turning partway | 4-6 |
| 9a | Island platforms | The island tool: spread zones with the S from target speed | island on double at 80 and 160 km/h: S lengths, speed never below target | 5-6 |
| 9b | Platform faces | Getting on and getting off sides (the Spanish solution); no effect on figures | panel and drawing check | 2-4 |
| 10 | Tapering and drawn platforms | Platforms between two bundles' tracks; the drawn platform (T-3859) | wedge station at a Y | 4-6 |
| 11a | Exact stops: storage and switching | A line's stops as stop points in running order; switching between simple and exact both ways with the notice; sim and demand read them | a line listed exactly as its simple stops gives the same runs and arrays | 5-6 |
| 11b | Exact stops: the circles | Circles per stop point, pushed apart along their platform, tinted by level, numbered; clicks, waypoints, touch | play check on quad and a Spanish station | 5-6 |
| 12a | A track and platform extract | A trainworld-owned extract (or noritetsu's, if it keeps them) with OSM platforms, stop areas, `railway:track_ref` and preferred direction; one region measured first | the region's counts against taginfo | 4-6 |
| 12b | Importing them | Platform shapes (island or side) and track order (AA/BB against side by side) for the sandbox starts and the 34 extract cities | the starts build; rides within 2% | 4-6 |

Docs, wiki and strings are inside each stage.

## Releases

Stages group into four releases a player would notice, each shippable on its own:

1. **Stages 1-5, the foundation** (about 60-84 h). Players see: quad is one thing; one track of a double can be
   picked, deleted or flipped; the station tool picks tracks; stations placed on built track show up in under half a
   second. The causes of the single/double joining bugs (research/code.md classes A to E) are gone, so those reports
   should thin out once the release's own fix tail is done. Nothing new to build yet. The sandbox starts are rebuilt
   inside it (3f, 4f).
2. **Stages 6-8, more tracks** (about 23-34 h): any count, a fifth track, AA/BB and ABBA, real passing loops and
   passing tracks, the crossover tool.
3. **Stages 9-11, stations** (about 21-28 h): island platforms, the Spanish solution, tapering and drawn platforms,
   exact stops with numbered circles. Stage 11 needs only stage 4, so it can move up if she wants exact stops sooner.
4. **Stage 12, real stations from OSM** (8-12 h), any time after stage 4.

## Order and what each stage unblocks among the paused tasks

| Task | What it is | Where it goes |
|---|---|---|
| T-1183 | Remove the fixed-rate line path | Stage 1, before any build stage (review-cleanup 11-20: "best before the track rework's build stages") |
| T-3365 | What single, double and quad mean | Answered by stages 2-3 (quad as one thing, Select bar 4, second click, the wiki page); close with stage 3 |
| T-3348 | Quad platforms (stages 2-5 paused) | Stage 1 stays (built). Its stages 2-3 are stage 6b; its stage 4 (quad stations) is stages 4d-4e. Close with stage 6 |
| T-3335 | Line panel on both quad pairs | Moot once the twin line is gone (stage 6c); close then, don't build it before |
| T-340 | Quad end-node stations and express side | End-node stations are a platform at a bundle end (stage 4); the express side is a preset's ways (stage 6b). Close with stage 6 |
| T-3859 | Custom drawn platforms | Stage 10 |
| T-3662 | Switching one kerbside piece breaks lines | A track's way belongs to the whole track through its continues, so the switch takes the whole leg (stage 3e). Close with stage 3 |
| T-3353 | Second single from a double's end refused | Joins by track id with real lateral positions (stage 3a) remove the cause in the model; the drawing's cone rule is a small piece with 3e |
| T-1948 | Node drag lead-in capped too short | The drag acts on the bundle's alignment after stage 3; free to build after 3e |
| T-3036 | Debug overlay for street gaps | Touches the street graph only, not the track model. Recommended: free it now |
| T-3354 | One-track platform jumps sides when dragged | A platform names its track, so it cannot jump; close with stage 4 |
| T-3264 | Trains drawn through each other | Each track drawn at its own offset (stage 3d) removes the cross-track case; her hunch (a local and an express on one track) stays its own task |
| T-3079 | Exact stop ordering back | Stage 11 |
| T-2767, T-2701, T-763 | Edit-speed requirements | Stage 5's checks; close T-2767 and T-2701 at 5a, T-763 at 5b |

## Trees side by side

Stages 2 to 5 run one after another, since each changes the records the next reads; within a stage, 3d with 3c, and
4c, 4d and 4e after 4a, can run side by side. From stage 6 on, two trees can run at once where files differ: 6a/6b
with 11, 7 with 9, 12 beside any of them (pipeline only).

## To file when she agrees (no ids given here)

- One task per release with its stages inside, on the lines track-kinds, quad-track, single-track, passing-tracks
  and stops as fits; T-3444 stays the parent and closes with stage 12.
- A DECISIONS line for each lie the compile makes when it is built: alignment length used in spreads; crossovers and
  turn moves at a fixed 50 km/h.
