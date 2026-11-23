# T-3444: tasks to file, one per stage

For the orchestrator, 2026-11-21. One entry per stage of `10-build-plan.md`, in the TASKS.md format, ids left as
`T-????`; `{after:stage N}` names the stage whose task it waits for (swap in its id). T-3444 stays the parent and
closes with the last of them.

**A new line** for LINES.md, in the track-stations group:
`- track-rework | Track rework | Tracks, platforms and stations rebuilt as records: every track its own, stations that
don't cut track, any count of tracks, on the tw/track branch (T-3444). | update:track-stations`
Stages go on it except where an existing line fits better: passing-tracks (7a, 7b), stops (11a, 11b), route-choice
(9c), real-networks (12a-12c).

**Branch**: every stage but 1 lands on tw/track, its own worktree (part 10, "Where it lands"). Stage 1 lands on
master.

**Starts now**: stage 1 (all of it on master; its timings wait for free cores). 12a could start now too (pipeline
only, nothing in the game), but it is P4.

**Side by side** (different files): 3b, 3c and 3d after 3a2; 4b, 4c, 4d and 4e after 4a2; 6c beside 6a/6b; 9b and 9c
beside 9a; 11a/11b beside 7 and 8 (not beside 6c); 12a beside anything. Everything else runs in the order below.

## Release 1: the foundation (stages 1-5)

- T-???? [P2] {L:track-rework} Track rework prep on master
  - Context: Before the tw/track branch is cut: T-1183 out (the fixed-rate path its tests still use), a save guard so today's game reads a tw/track save as "saved by a newer version" (it must ship in a release before the first merge back), part 8's seven baseline timings on master, and the community Vienna save (share 3znpvqx7 rev 2) fetched read-only into notes/T-3444-design/saves/ (rate limited on 2026-11-21).
  - Goal: T-1183 done, the guard shipped, the seven timings written down, the three reference saves in saves/.
  - Other: Starts now. Files: save.rs decoder top, the 13 test files of T-1183. 4-6 h. notes/T-3444-design/10-build-plan.md stage 1
- T-???? [P2] {L:track-rework} {after:stage 1} Bundles as saved records
  - Context: Quad is two doubles found again by geometry in six places with six distances. A bundle id and slot on today's edges, saved, set by drawing and doubling up, found once on old saves by today's twin rule.
  - Goal: The geometry pairing rules (`twin_at`, `quad_beside`, `quadPairs.ts`, `Alongside`) are gone and the real save's figures are unchanged.
  - Other: Files: save.rs (LBN1), wasm.rs quad, wasm/draw.rs, wasm/crossings.rs, twin.rs, quadPairs.ts. 3-4 h. stage 2a
- T-???? [P2] {L:track-rework} {after:stage 2a} Single, double and more in the tools
  - Context: Your three cases: the track bar's 1, 2 and More (4 for now); 4 tracks picked whole, a second click picks one double; the Select bar's 4 replaces Double up; titles "Single track", "Double track", "4 tracks".
  - Goal: Every tool picks by one shared pointer rule; no "Quad track" or Double up left.
  - Other: Files: tools.ts, ToolBar.tsx, Blueprint.tsx, TrackInspector.tsx, blueprintPick.ts. 3-4 h. stage 2b
- T-???? [P2] {L:track-rework} {after:stage 2b} Every track its own edge
  - Context: The compile from bundles with tracks to the derived graph, one track per edge, with stable ids, the node pitch and capacity's keys; the two-lane cap and the inferred-way code go. Stations are still nodes in this stage.
  - Goal: The sim tests pass on the small saves.
  - Other: Files: net/*.rs, lanes.rs, capacity.rs keys, a new compile module. 4-6 h. stage 3a1
- T-???? [P2] {L:track-rework} {after:stage 3a1} Track partners, turns and parity
  - Context: Each track's partner stored (the loader pairs each old double); turn moves at stop nodes and bare ends; routing and capacity on the new keys checked whole.
  - Goal: Parity gates A (real NYC) and B (the small saves and the Amsterdam "test" save) equal to master: paths, capacity and rides.
  - Other: Files: the compile module, net/paths.rs, capacity.rs. 3-4 h. stage 3a2
- T-???? [P2] {L:track-rework} {after:stage 3a2} New save format and old saves
  - Context: TWT7 at level 36; old saves converted on load with the old bytes kept.
  - Goal: Amsterdam "test", Vienna and North America day 383 load to parity, Amsterdam first.
  - Other: Side by side with 3c and 3d. Files: save.rs, saveFile.ts. 3-5 h. stage 3b
- T-???? [P2] {L:track-rework} {after:stage 3a2} Line paths in track spans
  - Context: Lines store paths as (track, direction) spans; the return run by partner lookup; routed returns stored, so `back_lines` stops rebuilding lines every edit; routing searches authored tracks.
  - Goal: Parity A and B; the perdir and t3565 checks pass.
  - Other: Side by side with 3b and 3d. Files: net/paths.rs, perdir.rs, wasm/lines.rs, LineData. 4-6 h. stage 3c
- T-???? [P2] {L:track-rework} {after:stage 3a2} Draw each track at its own offset
  - Context: Per-track rows to the app; `legShifts`, the double-to-single taper and `singleLeadStrokes` go; the wiki gets a "Tracks" page.
  - Goal: Joins, branches and flyovers look right on both running sides.
  - Other: Side by side with 3b and 3c. Files: wasm/snapshot.rs, stateRows.ts, network.ts, renderer and shaders. 5-6 h. stage 3d
- T-???? [P2] {L:track-rework} {after:stage 3d} Extend and edit one track
  - Context: A single drawn from a double's end extends the track under the pointer, with no automatic merge (your note: merging is manual); joins come from the pointer's pick; the second click reaches one track, which can be deleted or flipped.
  - Goal: Extending each track of a double, merging two singles manually and deleting one side all work; lines are rerouted or refused by name.
  - Other: Closes T-3353 with it. Files: tools.ts, wasm/draw.rs, wasm/edits.rs, toolDelete.ts. 4-5 h. stage 3e
- T-???? [P2] {L:track-rework} {after:stage 3e} Long track selections and ways
  - Context: Your decision 10: a way set on a track spreads to the nearest branch points; a double click traces one track to the next branch points and Shift+click extends it; a selection's way is set in one edit.
  - Goal: A long one-way leg set in one edit; T-3662's kerbside case keeps its lines.
  - Other: Closes T-3662. Files: blueprintBox.ts, pick.ts, wasm/oneway.rs, TrackInspector.tsx. 3-4 h. stage 3f
- T-???? [P2] {L:track-rework} {after:stage 3b} Sandbox starts on the new format
  - Context: `realsave` builds today's edges directly, so it changes with the format; the starts must keep loading settled (London 0.6 s, not 43 s).
  - Goal: Every start rebuilt and loading settled.
  - Other: Files: sim/src/bin/realsave.rs, pipeline realshared.py. 2-3 h. stage 3g
- T-???? [P2] {L:track-rework} {after:stage 3f} Stations as platform records
  - Context: Platforms and stations as records (level 37); stations stop cutting track; stop points compiled; the station fields on nodes go.
  - Goal: Parity where nothing new is used.
  - Other: Files: net/stations.rs, service.rs, the compile module. 4-6 h. stage 4a1
- T-???? [P2] {L:track-rework} {after:stage 4a1} Smart stop allocation
  - Context: Your decision 3: one click makes a two-way stop, each way's stop point picked at the click and stored, re-picked only when it goes; the "trains can't pass here" flag; `perdir` way stations and `station_beside` go.
  - Goal: Your split legs get a two-way stop in one click; the T-3605 and T-3711 cases are unchanged.
  - Other: Files: perdir.rs, twin.rs, wasm/lines.rs. 3-4 h. stage 4a2
- T-???? [P2] {L:track-rework} {after:stage 4a2} Old stations converted
  - Context: Each old station node becomes a station with its platforms; each stop keeps the way picks today's rule makes.
  - Goal: The three reference saves' figures equal to master.
  - Other: Side by side with 4c, 4d, 4e. Files: save.rs converter. 3-4 h. stage 4b
- T-???? [P2] {L:track-rework} {after:stage 4a2} Demand from stop points and the track chip
  - Context: Demand's snapshot rows come from stop points; one complex rule, in the sim; the stop list gets its track chip.
  - Goal: Packed demand arrays byte for byte on NYC, then the same rail trips a day.
  - Other: Side by side with 4b, 4d, 4e. Files: demandClient.ts, demand/network.rs, complexes.ts, the line panel's stop list. 4-6 h. stage 4c
- T-???? [P2] {L:track-rework} {after:stage 4a2} Station tool templates and preview
  - Context: Your decision 2: Side platforms, Island (zero width for now) and Single platform; the pointer and count pick the tracks; a platform placed overlapping a station joins it; the ghost shows the covered tracks and their ways.
  - Goal: All three templates on single, double and 4 tracks; strings listed.
  - Other: Side by side with 4b, 4c, 4e. Closes T-3354. Files: stationTool.ts, stationSnap.ts, ToolBar.tsx. 5-6 h. stage 4d
- T-???? [P2] {L:track-rework} {after:stage 4a2} Station panel platform picture
  - Context: A picture of the station's tracks (one way red, the other blue, two-way grey) with platforms to add and remove; Join to station and Make a separate station.
  - Goal: Platforms added and removed from the panel; lines that need a platform are named when refused.
  - Other: Side by side with 4b, 4c, 4d. Files: StationInspector.tsx, StationSides.tsx. 4-5 h. stage 4e
- T-???? [P2] {L:track-rework} {after:stage 4b} Sandbox starts with station records
  - Context: The starts rebuilt with stations as records, keeping today's stations.
  - Goal: Every start loads settled; rides within 2% of today's.
  - Other: Files: realsave.rs, realshared.py. 2-3 h. stage 4f
- T-???? [P2] {L:track-rework} {after:stage 4e} Drawn network rebuilt per bundle
  - Context: Drawn records keyed by bundle chunk and chainage, so an edit redraws only what it touched (T-2767, T-2701).
  - Goal: A station on built track redraws no other bundle: worker 30 ms or less, shows up in 400 ms median or less on the 2,647-station save (1,925 ms today).
  - Other: Closes T-2767 and T-2701. Files: network.ts, strokeCache.ts, clockPost.ts. 4-6 h. stage 5a
- T-???? [P2] {L:track-rework} {after:stage 5a} Line ribbons per track
  - Context: Lines side by side on one track laid out from their paths; `besideSamples` goes.
  - Goal: Midtown's ribbons look as today; the Penn Station to Emeryville preview timed (3 s on 10-19).
  - Other: Closes T-763. Files: lineBands.ts, corridors.ts. 3-4 h. stage 5b

Release 1 merges into master after 5b and its fix tail (10-20 h of player reports, filed as they come).

## Release 2: more tracks (stages 6-8)

- T-???? [P3] {L:track-rework} {after:stage 5b} More tracks up to sixteen
  - Context: The More button's count and layout; tracks added to a selection run unconnected until crossovers are built (your follow-up on decision 8); one track removed; the price per track (half a double beside an other-way track, else 0.6).
  - Goal: A fifth track added and priced; a line set on it is refused until crossovers are built.
  - Other: Files: ToolBar.tsx, Blueprint.tsx, wasm/draw.rs, cost.rs. 4-6 h. stage 6a
- T-???? [P3] {L:quad-track} {after:stage 6a} Tracks by direction
  - Context: AA/BB and ABBA layouts; drawing by direction, the peel-off and the flying split (T-3348's stages 2-3); crossing flat with ramps off.
  - Goal: Locals and expresses on AA/BB with no flat crossing at the split.
  - Other: Closes T-3348 and T-340 with 6c. Files: wasm/draw.rs, flyover.rs, tools.ts. 5-6 h. stage 6b
- T-???? [P3] {L:quad-track} {after:stage 5b} One line over several tracks
  - Context: The hidden twin line goes; a line runs equal shares of its trains on each track it uses; panels read one line.
  - Goal: The t3333 save gives the same trains and rides.
  - Other: Side by side with 6a and 6b. Closes T-3335. Files: twin.rs, service.rs, world.rs twin_ops, the line panel. 3-5 h. stage 6c
- T-???? [P3] {L:passing-tracks} {after:stage 6c} Real passing loops on single track
  - Context: The loop drawn at a single-track station becomes real track; a little less waiting where lines share one, and a terminal there turns 30 an hour (decision 20).
  - Goal: Parity B except the reviewed differences part 5 lists.
  - Other: Files: capacity.rs, overtake.rs, terminalBays.ts, save.rs converter. 3-5 h. stage 7a
- T-???? [P3] {L:passing-tracks} {after:stage 7a} Passing tracks as real track
  - Context: The passing tracks button builds real tracks with the passing purpose; overtakes read the tracks; each line stops on one platform track (decision 19); the terminal stays an upgrade.
  - Goal: T-2231's line keeps its express and local times.
  - Other: Files: upgrades.rs, overtake.rs, network.ts passing drawing, the station panel. 4-6 h. stage 7b
- T-???? [P3] {L:track-rework} {after:stage 7b} Track purpose
  - Context: Each track or long selection can be Any line, Passing track or Only for chosen lines; routing, overtakes and capacity read it.
  - Goal: An express pair kept to expresses; a passing track used only for overtakes.
  - Other: Files: net/paths.rs, overtake.rs, TrackInspector.tsx. 4-6 h. stage 7c
- T-???? [P3] {L:track-rework} {after:stage 7c} Crossovers drawn as single track
  - Context: Your note: a crossover is a single track drawn between two tracks (the tool lays the S); a scissors is two; old crossovers load as singles; turnback crossovers are laid with the stop edit, priced, deletable; figures at termini unchanged.
  - Goal: A line turning at a terminus has today's figures; deleting its crossover refuses or reroutes it by name.
  - Other: An old double-track diagonal loads as a double, so the derived graph stays as today; the sandbox starts are rebuilt with their crossovers. Files: tools.ts, wasm/draw.rs, wasm/crossings.rs, capacity.rs turns, network.ts. 4-6 h. stage 8a
- T-???? [P3] {L:track-rework} {after:stage 8a} Build crossovers offer
  - Context: Your follow-up on decision 8: when a line can't reach a track for lack of crossovers, the refusal offers to build them in sensible spots, as a blueprint with its cost, in one click.
  - Goal: The fifth track's line and a stop on a quad's far pair each work in one click.
  - Other: Files: a crossover placer in the sim, issues.ts, the stop and chip UI. 4-6 h. stage 8b

Release 2 merges into master after 8b.

## Release 3: stations (stages 9-11 and 13)

- T-???? [P3] {L:track-rework} {after:stage 8b} Island platforms that spread the tracks
  - Context: The Island template spreads the tracks for a 6 m island, the S laid for the target max speed (flyover speed where room is short); built track gets a blueprint reconstruct.
  - Goal: At 80 and 160 km/h the S is about 70 and 140 m each end and trains never slow below the target.
  - Other: Files: geom.rs offsets, stationTool.ts, stationGeometry.ts. 5-6 h. stage 9a
- T-???? [P3] {L:track-rework} {after:stage 8b} Getting on and off sides
  - Context: Faces on platform edges for the Spanish solution; no effect on the figures for now (decision 22).
  - Other: Side by side with 9a. Files: the platform record, StationSides.tsx, stationGeometry.ts. 2-4 h. stage 9b
- T-???? [P3] {L:route-choice} {after:stage 4c} Island change counts one minute
  - Context: Your decision 23: a change between the two edges of one island counts 1 minute instead of 2, as flavour.
  - Goal: The kernel test passes; ceiling tests unchanged.
  - Other: Side by side with anything. Files: demand kernel change cost, demand-worldwide.md. 2-3 h. stage 9c
- T-???? [P3] {L:track-rework} {after:stage 9a} Tapering and drawn platforms
  - Context: Platforms between two bundles' tracks where they split; the drawn platform from T-3859.
  - Goal: A wedge station at a Y works.
  - Other: Closes T-3859. 4-6 h. stage 10
- T-???? [P3] {L:stops} {after:stage 4c} Exact stops in running order
  - Context: A line's stops as stop points in running order, out and back; switching between simple and exact both ways with a notice.
  - Goal: A line listed exactly as its simple stops gives the same runs and demand arrays.
  - Other: Side by side with 7 and 8, not with 6c. Files: wasm/lines.rs, LineData, demandClient.ts. 5-6 h. stage 11a
- T-???? [P3] {L:stops} {after:stage 11a} Numbered stop circles per track
  - Context: Your notes: a circle per stop point, pushed apart along its platform, tinted by level, the chosen ones in the line colour with the stop number; clicks, waypoints, touch.
  - Goal: Works on 4 tracks and a Spanish station; closes T-3079.
  - Other: Files: stopMarkers.ts, the line panel. 5-6 h. stage 11b
- T-???? [P3] {L:track-rework} {after:stage 6b} Quick-place and copy station layouts
  - Context: Your decision 2: place a station on all of a bundle's tracks in one click (AA/BB's islands too) and copy a station's layout to another place.
  - Other: Files: stationTool.ts, ToolBar.tsx. 4-6 h. stage 13

## Release 4: real stations from OSM (stage 12, low priority)

- T-???? [P4] {L:real-networks} Track and platform extract from OSM
  - Context: OSM platforms, stop areas, `railway:track_ref` and preferred direction, from a trainworld-owned extract (or noritetsu's if it keeps them); one region measured first.
  - Goal: The region's counts match taginfo's.
  - Other: Could start now (pipeline only). 4-6 h. stage 12a
- T-???? [P4] {L:real-networks} {after:stage 12a} Real platform shapes and track order
  - Context: Island or side platforms and track order (AA/BB against side by side) for the sandbox starts and the 34 extract cities.
  - Goal: The starts build; rides within 2%.
  - Other: Also after stage 4f. 4-6 h. stage 12b
- T-???? [P4] {L:real-networks} {after:stage 4f} One station per real station in starts
  - Context: Your decision 26 (low priority): a stop where local and express tracks run side by side becomes one station with four platforms; separate lines under one name stay separate stations in one complex.
  - Goal: Every start loads settled; rides within 2%.
  - Other: 2-3 h. stage 12c
