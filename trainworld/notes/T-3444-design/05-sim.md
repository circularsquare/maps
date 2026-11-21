# T-3444 design, part 5: the simulation on the new model

Agent tdsim, 2026-11-21. Design for the rework; nothing is built. Vocabulary as in `01-model.md` (track, bundle, slot,
offset, node, connection, platform, stop point, station, complex, simple stops, exact stops, preset, derived graph).
Players see single track, double track and "more" (3 to 16 tracks); "bundle" stays our word. Stage numbers are
`10-build-plan.md`'s. Today's code: `research/code.md` sections 1, 2, 5-7. Her answers: `answers-2026-11-21.md`.

## What does not change

No signals (SPEC 2.1): trains never block each other; delay comes from schedules through capacity. Platform
assignment is static per line and stop (per pattern, section 7); there is no choice of a free platform while running.
Every formula in track-and-service.md 6.2 stays as written: section occupancy `70 s + v / 0.6`, single track's run
time + 240 s, platform dwell + 60 s, the pooled turn `max(turnaround + 60 s, 240 s)` over its places, junction and
crossing moves 120 s (45 s tram), Kingman's delay, following losses, timetabled passing waits, progressive filling,
the settle. What changes is **what each resource is keyed by**: today's keys name an edge's direction or a station
node; the new keys name a track (or a span of one) and a stop point. Decided rules kept: a plain node never changes
tracks (T-3137 C1); a terminal turns 30 trains an hour and draws no sidings (T-3107); two-way single track queues as
one stream (T-3526); one-way stations are stops of their own (T-3605); locals' waits at passing stations are worked
out train by train (T-3346).

## 1. The derived graph's contract

The compile turns bundles, nodes, connections and platforms into single-track derived edges (one track, one way or
both) between derived nodes (bundle nodes plus stop nodes). There is no crossover record: a crossover is an ordinary
single-track bundle (section 5). Path entries stay `edge << 1 | dir`. This is what each consumer gets from it.

| Consumer | Gets from the compile | Today from |
|---|---|---|
| Routing (`find_path_side`, `check_line`, `reach_from`) | derived edges with way, purpose and run time; per derived node a move table from the connections, lateral positions in metres; the turn moves (section 3); per station its stop points by way | `lanes()`, `node_moves`, station nodes |
| Line paths | the map (track id, dir) span <-> derived entries, both ways, so authored paths compile to runs and the app's answers map back | `LineData::path` directly |
| Capacity (`capacity.rs`) | per derived edge its track id, bundle id, way and purpose; two-way chains for single-track sections; stop points with station id, platform length and served track; turn places per stop node; move positions for crossing tests; flat crossings per pair of bundles (a scissors' two singles included) | `key_of`, `(node, side_in)`, `turn_places`, `track_pos`/moves |
| Profiles and service (`profile.rs`, `service.rs`) | derived edge length (the alignment's, 01's < 0.1% lie), every speed cap (kind, piece limit, curves, an island's spread S), levels and ramps; keyframes per (derived edge, dir) | edge geometry, `v_cap` |
| Boarding and money | each run's stop points by stop index; the station and platform of each | `stop_node`, `stop_idx` |
| Demand snapshot | stop nodes with position, station, complex, level, platform ids (part 6) | station nodes |
| The app | (track id, chainage) for every train sample and dot; the turn moves lines use; junction moves named by track | `edgeParts`, app rules |

**Ids.** Incremental capacity (T-050) rebuilds only resources near an edit, and an edit elsewhere leaves every rate,
round trip and phase "to the bit" (T-1863). So the compile **keeps the derived ids of every bundle it does not
recompile**, and gives the same ids for the same saved records on every load (a load settles from the saved rates and
changes nothing). Free slots for a recompiled bundle come from a free list, as edge ids do today. Resource keys are
built from derived ids, so they stay put too.

**Node positions.** Bundle ends on one side of a node sit in rank order, as `derive_node` does today. The pitch
between neighbouring bundle ends is half the sum of their widths plus 4 m, so a 4-track end (±6 m) beside a double
end never interleaves and makes false crossings. For two doubles that is 8 m, so positions come out as T-3137 C1's
`4 x old - 2` and every crossing test agrees with today.

**Size.** The real NYC save has 1,647 double edges, so 3,294 derived edges plus stop splits: the same count of route
search states as today's (edge, dir) pairs. Nothing here grows the search.

## 2. Capacity on the new keys

| Resource | Key today | Key on the new model | Stage |
|---|---|---|---|
| Track section (per direction) | `edge * 2 + dir`, cut where the set of lines changes | the derived edge (one track carries one way, or both) | 3a |
| Single-track section | `edge * 2` on two-way single, between passing points (every station but a one-track one) | a chain of two-way derived edges, between passing places (section 4) | 3a, 7a |
| Platform track | `(station node, side_in)`; one shared on single track | the stop point (stop node, track) | 3a (stations still nodes), 4a |
| Turning | station node; places `ceil(tracks / 2)`, all with a terminal | the stop node; places `ceil(p / 2)`, all `p` with a terminal, `p` = platform tracks at that stop node | 4a |
| Junction move | node and move from rank slots | derived node and move from connections, positions in metres | 3a |
| Flat crossing | one per direction of each route | one per track of each bundle, loaded by all the other bundle's trains (the same as today on double) | 3a |
| Following loss | stretch of same (edge, dir) | stretch on the same track, joined across a split that rejoins (section 4) | 3a, 7b |
| Passing wait | (faster run, station node with `PASSING`) | (faster run, the slower run's stop node on a passing track) | 7b |

What falls out of one track per derived edge, with no special case:
- A two-way track's platform carries both directions (today's "a single-track platform shares both directions").
- A one-track station's other track holds nothing at the platform (T-3332).
- Lines on different tracks of one bundle share nothing: a by-direction layout's express and local, or two doubles of
  4 tracks, follow nobody and queue together only where their paths meet (as quad's pairs do today, 6.2).
- Mix (Kingman's `1 - sum share²`) keeps its streams as (line, direction), and a line on two-way track is one stream
  (T-3526).

**Junction moves from connections.** Continue, join and turnout connections become moves; crossing pairs come from
positions across the node, so the flat-junction badge, junction capacity and the train guard agree by construction
(as since T-3137). A node with only continues never changes tracks (C1). Merges and diverges cost nothing, as today.

## 3. Turnbacks and terminals

Today a turn is abstract: a train arriving at a station it stops at, or at a track end, may leave back the way it
came, holding one turn place and crossing nothing (6.2, T-797, T-3107). On one-way tracks a turn has to move from the
arriving track to the track the line leaves on.

**Stages 3a to 7: internal turn moves.** The compile makes a turn move at every stop node and bare end, never stored,
from each served track to the track the return run leaves on there (its partner, section 6, or itself on two-way
track). Only a path that turns back may use it (T-797: at a stop or a track end). It is priced exactly as today: the
pooled turn place and nothing else, no platform hold, no crossing. Every terminus figure stays where it is.

**From stage 8: real turnback crossovers.** Where a line turns back on one-way tracks and no crossover joins its
arriving track to its leaving track beside the stop, the stop edit lays one: a crossover single (section 5), in the
blueprint with the line's edit, priced as single track; old saves get theirs free when they are converted. "Beside the
stop" means its nearer turnout lies within 200 m of the platform's end with no other junction or station between.
- **Where it goes**: on the side the trains arrive from, 20 m clear of the platform's end (`BRANCH_CLEAR`, so its
  turnout is never inside a platform); if a junction, station or track end is in the way, on the far side; if neither
  fits, no crossover is laid and the line turns by the internal move as before stage 8 (still open for her, below).
- **What it licenses**: the turn move from that arriving track to that leaving track at the stop node. The run's path
  keeps the turn move (arrive, stand the turnaround, leave on the other track); its train is drawn as today.
- **Capacity**: a turning line's moves over a crossover beside its turning stop are booked as today's turn place and
  nothing else, so no figure moves. A through line whose path runs over the same crossover pays its junction moves as
  at any junction.
- **Deleting it** refuses the edit, or reroutes the lines that turn over it where another crossover beside the stop
  serves, naming the lines, as a way change does today (`way_reroutes`).
- Two-way track needs none (the train leaves on the track it came in on), nor does a bare end of single track.

This replaces `turnbackStrokes` (an app rule) with track the sim knows about: the class C fix.

**The terminal** stays an upgrade on the station, read by the turn-place count (30 an hour a pair at the default
turnaround), drawn as today with buffer bars and no sidings. A player who wants a real stub terminal builds one (stub
tracks ending at buffer stops, a scissors of two crossover singles in front), and it runs as built: turning at a track
end, through moves over the scissors priced as junction moves, the turning line's moves over it as its turn place.

## 4. Passing and overtakes

Today a line on a shared stretch follows the slower line ahead (following loss), unless the slower one stops at a
station with the `PASSING` bit inside the stretch, where the timetable has it wait (`overtake.rs`). With passing
tracks drawn, not modelled, both lines stay on one edge through the station. With real passing tracks (stage 7b)
the paths part there. These rules keep the model whole:
1. **A shared stretch joins across a split that rejoins.** Two lines' runs on the same track, parted at node X and
   back on one track at node Y going the same way with no turn between, are one stretch. Following losses are
   counted only on the common track.
2. **A passing place** is such a split where the slower run stops between X and Y on a track whose purpose is Passing
   track (section 8). The passing-tracks preset and the old-save loader mark their stopping tracks so. With several
   such stops inside one split, the passing place is the last before Y; before that the two run on separate tracks
   and lose nothing to each other.
3. **Which track a line takes through a station it doesn't stop at.** A Passing track is avoided (section 8). Between
   two Any line tracks of equal time, ties go to the continue connection (the same slot), so a through line keeps to
   the through track (the class D bug T-3651 made impossible).
4. **The rejoin node Y is not a merge** (6.2's kind 12) for lines of one joined stretch: no merge resource, no
   `pace_merges`, no `MERGE_FIRST` weight in the spacing. Otherwise the 3% pacing and merge-first placement would
   fight the overtake plan's placement; today there is no merge there (both lines on one edge).

The overtake plan itself (`overtake::plan`, `train_waits`, T-3346's train-by-train waits, T-3238's caps, T-3929's
lead-in) is untouched; only its points and keys come from these rules.

**Old station-only track becomes real track** (stages 7a and 7b). Three figures move on old saves that have it:

| Old save holds | Today | Real track | Figure |
|---|---|---|---|
| Passing tracks (`PASSING`) | a stopping train holds one of two pooled platform tracks, (dwell + 60 s) / 2 | each line stops on its own platform track (the preset puts islands so inner and outer tracks both have one) | platform waits rise a little at busy passing stations where two stopping lines shared the pool |
| Single-track station (loop drawn, not modelled) | one shared platform track; section cut at the station | a 2-track loop bundle, each track one way by the running side, a platform each | with one line nothing moves (one stream, no queue); with two or more a little less waiting |
| Single-track station with a terminal | one turn place, 15 an hour | two platform tracks, 30 an hour | more trains can turn there |

Held fixed by rule:
- The single-track section's run time is measured from passing place to passing place as today (from the loop's stop
  point), not from the loop's end nodes, so the 240 s margin gives today's figures.
- The loop's end nodes are merges and diverges of one-way tracks: no junction resource.
- A track section is cut where its set of lines changes, so a stopping line leaving for the outer track shortens the
  main track's sections. The occupancy per train should be the same; to confirm in parity B against `build_sections`.

**Stages 4a to 7a.** Platforms no longer split track but loops are not yet real. A platform on two-way single track
keeps a "trains can't pass here" flag (today's `no_passing`): without it, the stop node cuts the single-track section
and its platform is one shared track, as today. Stage 7a drops the flag: the loader builds a loop where it was off.

## 5. Crossovers are single track

A crossover is an ordinary single-track bundle with a turnout at each end. The player draws it with the track tool
(Single, the pointer picking a track at each end); where both ends lie on neighbouring parallel tracks the tool lays
the S itself. It is priced as single track, picked and deleted like any track. A scissors is two of them crossing: a
flat crossing inside the bundle, allowed. For the sim there is nothing new: routing takes one when it is the fastest
way to the next stop, its curves cap its speed, its turnouts are junction moves, a scissors' crossing is a flat
crossing. A crossover never turns a train; turning stays at stops and track ends (section 3).

**Old T-338 crossovers** load as ordinary track of their own track count (a diagonal between two doubles of 4 tracks
loads as a double, joined at both ends), keeping their 50 km/h as a piece speed limit (T-3405's `limit`). So the
derived graph and every figure are today's.

**Tracks added to a bundle stay unconnected** to the others (an island of track) until crossovers are built (her
follow-up on decision 8). The sim never joins them by itself.

### The crossover placer (stage 8)

When a line is set to use a platform or track it can't reach (a stop click, the track chip, exact stops, a route
edit), the sim says whether crossovers would fix it and, if so, where.

**Finding that only crossovers are missing.** The ordinary search fails first. Then the same search runs once more
on a relaxed graph: inside every bundle, a virtual move from each track to each other track of the bundle whose way
allows it, available anywhere along the bundle, costing the S's run time plus 600 s (so it is used only where nothing
real serves, and as few times as possible). If the relaxed search finds a route, the stop is unreachable only for lack
of crossovers, and each virtual move on that route is a need: (bundle, from track, to track, direction, the stretch
of the bundle where the route made it). If the relaxed search fails too, the refusal is the plain one, with no offer.
Both runs are searched, so the return can add needs of its own. The extra search runs only after a failure: one
Dijkstra per failing hop.

**Where each crossover goes.** For each need, walking back along the run's path from the place the route needs to be
on the new track (the platform's near end, or where the track it must reach begins):
- the first spot whose whole S fits on that bundle, both turnouts 20 m clear of any platform end (`BRANCH_CLEAR`),
  other nodes, flat crossings and ramps (no split on a ramp), and inside the stretch where the run is on both tracks;
- if none fits on that bundle, the same on the bundles before it on the path that carry both tracks, back to the run's
  previous stop;
- if still none: no offer for that stop, and the refusal says "No room for crossovers between {A} and {B}" (a draft
  for her review).

**Shape and direction.** One crossover single per need, from the track the run is on to the track it needs, crossing
any tracks between it flat (a single piece the player sees and deletes whole, as an old pair-to-pair crossover). Its
turnouts face the run's direction of travel, and its way is that direction; if the other run needs the same move the
other way at the same place, it is two-way. Its S is laid for the flyover target max speed (80 km/h by default), as the
tool's own S is: about 80 m between tracks 4 m apart.

**Cost and confirmation.** Each is priced as single track at its level (single track's per-km price times the S's
length). The offer, "Build crossovers?", shows them as blueprint on the map with their total; one click accepts, and
the line edit goes through with them in one undoable edit. Declined, nothing changes.

**Turnback crossovers** (section 3) use the same shape, room rules and cost, with section 3's spot (arrival side
first, then the far side), and come with the stop edit and its cost without the offer (still open for her, below).

## 6. The return run: partner lookup or stored route

The return run is the forward path mapped track by track to each track's **partner** in the same bundle; where that
fails it is routed and stored with the line.

**The partner is stored per track** (one small field on the bundle's track). "The nearest track running the other way"
does not work: on default 4-track offsets (-6, -2, +2, +6) with ways A B A B, the B track at -2 has A tracks at -6
and +2, both 4 m off; on the real NYC save, where imported lines sit 6 m apart (`realsave.rs`), two doubles bundled
together have tracks at -5, -1, +1, +5, and the -1 track's nearest A track is the other double's; by direction
(A A B B) both A tracks pick slot 2. So the partner is set by whatever lays the tracks: a double stroke pairs its two;
presets pair by their pattern (ABAB 0-1, 2-3; AABB 0-3, 1-2; ABBA 0-1, 2-3); the loader pairs each old double's two
tracks; a two-way track is its own partner; partners never cross bundles. A track added alone takes, when its way is
set, the nearest other-way track without a partner (ties to the lower slot).

The lookup holds where every forward span's partner exists, each pair of consecutive partner spans has a connection
at their node going the return's way, and the result passes the stop points stored for the return (section 8).
Where it fails anywhere (a one-way leg, a crossover on the forward path, a turnout whose partners don't join, a
stored back stop on other track), the whole return is routed through the stored back stops and stored with the line.

**When a stored return is routed again**: when an edit recompiles a bundle it runs over, changes a platform at one of
its stops, or edits the line. New track elsewhere never reroutes it, as for forward paths today. Lines with no way
back (T-3835's loops one way by their track, or a broken return) go on a short list retried after every track edit,
so building the missing track still gives them their return; the whole-network scan of `back_lines` goes.

## 7. A line on several tracks of one bundle (replacing the hidden twin, T-3333)

Today a line on both quad pairs keeps a hidden second line (`LineData::twin`, `twin_ops`, `foldTwins`,
`pin_twin_phases`); every consumer sees two lines, each with half the trains, the twin half a headway behind.

**Patterns, derived, not stored as lines.** A line set to use all tracks (today's two-pairs switch) gets P patterns,
P = the most tracks of its way on any bundle of its path (2 on ABAB). Pattern i takes, on every bundle, the
(i mod n)-th of its n tracks that way in slot order (so all patterns share a double), and stops at the stop point of
each station on that track; it is routed once and stored as the return is, with the same re-route rules. Where a
pattern can't reach a station's stop point, the switch stays on and the line runs on one pattern, the panel saying
why (as `line_pairs_why` does now). A track whose purpose keeps the line off it is skipped.

To every consumer each pattern is one service line with its own two runs, carrying 1/P of the line's trains (counts
round up so all patterns are equal, as the twin's do), its headway P times the line's, departures 1/P of a headway
apart. Capacity, the timetable, boarding, money and demand keep their shape, and a save with two pairs gives today's
figures. What goes: the twin `Op` traffic in every edit, `LTW1` (loaded as the switch), the app's folding (the line's
figures are summed in the sim).

## 8. Stops, track purpose, timetables

**Smart allocation (simple stops).** One click on a station makes a two-way stop. At the click the sim picks a stop
point for each run and stores both on the stop: a stop point of the clicked station the run reaches going its way
(fewest crossing moves, then nearest the pointer); if the station has none for that way, the stop point of the nearest
station within 300 m (its complex first) that the run passes on its own track going that way; if none, the run passes
(`Run::pass`). Stored, never re-derived. An edit re-picks a stop only when:
- its stored stop point is removed (that stop only);
- the stop list changes so the other run's stored point is another listed stop. Then that run passes, so two one-way
  stations listed as separate stops stay separate (T-3605);
- the line is reversed: the two stored points swap runs.
Shift click still makes a one-way stop (T-3711), and its way is still weighed at each stop edit, as now. Her split legs
150 m apart work in one click without joining the stations; joining is only for naming. The stop list's track chip
shows and changes either stored point. This replaces `perdir.rs`'s way stations, `way_group`'s every-node scan and
`back_lines`.

**Track purpose** (stage 7c), per track or long selection:

| Purpose | Routing | Stops | Overtakes and capacity |
|---|---|---|---|
| Any line (default) | as usual | any line may stop | a stop on it is no passing place |
| Passing track | lines that don't stop on it treat it as absent unless it is their only way | stopping lines may be allocated to it | a stopping line waits there for faster lines (section 4) |
| Only for {lines} | other lines treat it as absent | only those lines are allocated to it | none of its own |

Setting a purpose reroutes the lines it keeps off the track through their same stops, or refuses the edit naming them,
as a way change does. A way set on a piece spreads along its track through continues to the nearest junction nodes
(the T-3662 fix); lines are rerouted over the whole spread in the same edit.

**Timetables.** A run's stops are stop points, one per stop index (`Run::stop_nodes` today). `Run::pass` keeps its
meaning: a stop the run goes by on another track. Departure boards, the chart and boarding read the run's stop points,
so a stop can show its platform track. Exact stops (stages 11a, 11b): the cycle's stop points are listed in running
order, out and back, and runs read them directly; `flatten`, `way_stops` and `back_run_of` are not used for such a
line. Per-train stops stay T-3346's (later). Island and Spanish solution platform faces change no figure (no shorter
dwell).

## 9. Stages, checks and hours (sim side)

| Stage | Sim work | Replaces | Check | Hours |
|---|---|---|---|---|
| 3a | compile (stations still bundle nodes), stable ids, node pitch, routing and capacity on derived keys, internal turn moves | `lanes.rs`, `key_of`, `track_pos` leftovers | A, B | 10-14 |
| 3b | loader: partners from old doubles, old offsets kept, old crossovers as track with 50 km/h | | A, B | 1-2 |
| 3c | path spans, return by partner lookup, stored routed returns, the retry list | `back_lines`' scan | A, B | 4-6 |
| 3f | way spreading with reroutes | `derive_ways`, `run_group`, `group_ways` | B | 1-2 |
| 4a | stop points from platforms, smart allocation stored both ways, re-picks, turn places per stop node, the "can't pass" flag | `perdir.rs` way stations, `station_beside` for stops | A, B; split legs on both running sides | 8-12 |
| 4b | old stops' picks stored as today derives them (`way_station`, legs), so nothing moves | | A, B | 2-3 |
| 6c | patterns | `twin.rs`, `twin_ops`, `LTW1` | B; a two-pairs game equal | 5-8 |
| 7a | real loops, sections from passing places | the station loop rule | B except section 4's rows | 3-5 |
| 7b | passing places, stretch joining, Y not a merge | `PASSING` bit, `passing_at` | B except section 4's rows | 5-7 |
| 7c | track purpose in routing, allocation, overtakes | | B; new purpose tests | 4-6 |
| 8 | the crossover placer (relaxed search, spots, shape, cost, the offer), turnback crossovers laid by stop edits and old-save conversion, turn booking, delete refusal | `turnbackStrokes` | A, B: no figure moves; placer tests on added tracks | 8-12 |
| 9a | spread zones' speed caps | | an island's limit never below its target | 1-2 |
| 11a, 11b | exact stops read by runs | | a line listed exactly as its simple stops gives the same runs | 3-5 |

**Gate A, the real NYC save** (`T-007-real-nyc.save`, nothing new used): every line's path mapped back to old
(edge, dir), run times, round trips, fleets, rates per level, every resource's rho and delay by mapped key, phases;
demand's packed arrays and rail trips a day (12,581,681 on 2026-11-16, part 6). Necessary but weak: all double, no
3-end node, as far as I know no upgrades.

**Parity B, the reference saves**, in this order: her Amsterdam "test" save (`saves/amsterdam-test-day-61.save`,
lightest, first), the community Vienna save (tram single-track oddities; fetched in stage 1), North America day 383;
plus T-3137's parity games on both running sides (flat and flying Y, flyover, single branches, single Y, quad with a
crossover, circle, flat crossing) for the cases the saves may lack. `checks/t3137-lanes-parity.mjs` and its bin are
the start; each stage keeps them equal except where the table says.

## Still open for her

1. Where no turnback crossover fits beside a stop (a junction or station too close on both sides), the line still
   turns there as it does today, with no crossover drawn, rather than the stop being refused? (yes: nothing that works
   today should stop working)
2. Passing waits happen only at stops on tracks marked Passing track; a split where the slower line stops on an Any
   line track gives no overtake wait (the two simply run on their own tracks)? (yes: the player says where trains step
   aside, and by-direction layouts get no surprise waits)
3. Turnback crossovers at a line's ends come with the stop edit and its cost, without the "Build crossovers?" offer?
   (yes: every double-track line needs them, and asking each time adds a click to the simplest case)
