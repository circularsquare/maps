# T-3444 design, part 5: the simulation on the new model

Agent tdsim, 2026-11-21. Design draft for discussion with Anita; nothing is built. Vocabulary as in `01-model.md`
(track, bundle, slot, offset, node, connection, platform, stop point, station, complex, simple stops, exact stops,
preset, derived graph). Stage numbers are `10-build-plan.md`'s. Today's code: `research/code.md` sections 1, 2, 5-7.

## What does not change

No signals (SPEC 2.1): trains never block each other; delay comes from schedules through capacity. Platform
assignment is static per line and stop (per pattern, below); there is no choice of a free platform while running.
Every formula in track-and-service.md 6.2 stays as written: section occupancy `70 s + v / 0.6`, single track's run
time + 240 s, platform dwell + 60 s, the pooled turn `max(turnaround + 60 s, 240 s)` over its places, junction and
crossing moves 120 s (45 s tram), Kingman's delay, following losses, timetabled passing waits, progressive filling,
the settle. What changes is **what each resource is keyed by**: today's keys name an edge's direction or a station
node; the new keys name a track (or a span of one) and a stop point. Decided rules kept: a plain node never changes
tracks (T-3137 C1); a terminal turns 30 trains an hour and draws no sidings (T-3107); two-way single track queues as
one stream (T-3526); one-way stations are stops of their own (T-3605); locals' waits at passing stations are worked
out train by train (T-3346).

## 1. The derived graph's contract

The compile turns bundles, nodes, connections, platforms and crossovers into single-track derived edges (one track,
one way or both) between derived nodes (bundle nodes plus stop nodes). Path entries stay `edge << 1 | dir`. This is
what each consumer must get from it.

| Consumer | Gets from the compile | Today from |
|---|---|---|
| Routing (`find_path_side`, `check_line`, `reach_from`) | derived edges with way and run time; per derived node a move table from the connections, lateral positions in metres; the turn moves (section 3); per station its stop points by way | `lanes()`, `node_moves`, station nodes |
| Line paths | the map (track id, dir) span <-> derived entries, both ways, so authored paths compile to runs and the app's answers map back | `LineData::path` directly |
| Capacity (`capacity.rs`) | per derived edge its track id, bundle id and way; two-way chains for single-track sections; stop points with station id, platform length and served track; turn places per stop node; move positions for crossing tests; flat crossings per bundle pair | `key_of`, `(node, side_in)`, `turn_places`, `track_pos`/moves |
| Profiles and service (`profile.rs`, `service.rs`) | derived edge length (the alignment's, 01's < 0.1% lie), every speed cap (kind, piece limit, curves, an island's spread S, a crossover diagonal's 50 km/h), levels and ramps; keyframes per (derived edge, dir) | edge geometry, `v_cap` |
| Boarding and money | each run's stop points by stop index; the station and platform of each | `stop_node`, `stop_idx` |
| Demand snapshot | stop nodes with position, station, complex, level (part 6) | station nodes |
| The app | (track id, chainage) for every train sample and dot; the turn moves lines use (to draw crossovers); junction moves named by track | `edgeParts`, app rules |

**Ids.** 01 says derived ids "may change on any edit inside a bundle". The sim needs more: incremental capacity
(T-050) rebuilds only resources near an edit, and an edit elsewhere must leave every rate, round trip and phase "to
the bit" (T-1863). So the compile must **keep the derived ids of every bundle it does not recompile**, and give the
same ids for the same saved records on every load (a load settles from the saved rates and changes nothing). Free
slots for a recompiled bundle come from a free list, as edge ids do today. Resource keys are built from derived ids,
so they stay put too.

**Node positions.** Bundle ends on one side of a node sit in rank order, as `derive_node` does today. Today's 8 m
pitch fits a double (±2 m) but a quad end (±6 m) beside a double end would interleave and make false crossings. The
pitch between neighbouring bundle ends is half the sum of their widths plus 4 m. For two doubles that is 8 m, so
positions come out as T-3137 C1's `4 x old - 2` and every crossing test agrees with today.

**Size.** The real NYC save has 1,647 double edges, so 3,294 derived edges plus stop splits: the same count of route
search states as today's (edge, dir) pairs. Nothing here grows the search.

## 2. Capacity on the new keys

| Resource | Key today | Key on the new model | Stage |
|---|---|---|---|
| Track section (per direction) | `edge * 2 + dir`, cut where the set of lines changes | the derived edge (one track carries one way, or both) | 3 |
| Single-track section | `edge * 2` on two-way single, between passing points (every station but a one-track one) | a chain of two-way derived edges, between passing places (below) | 3, 7 |
| Platform track | `(station node, side_in)`; one shared on single track | the stop point (stop node, track) | 3 (stations still nodes), 4 |
| Turning | station node; places `ceil(tracks / 2)`, all with a terminal | the stop node; places `ceil(p / 2)`, all `p` with a terminal, `p` = platform tracks at that stop node | 4 |
| Junction move | node and move from rank slots | derived node and move from connections, positions in metres | 3 |
| Flat crossing | one per direction of each route | one per track of each bundle, loaded by all the other bundle's trains (the same as today on double) | 3 |
| Following loss | stretch of same (edge, dir) | stretch on the same track, joined across a split that rejoins (section 4) | 3, 7 |
| Passing wait | (faster run, station node with `PASSING`) | (faster run, the slower run's stop node at a passing place) | 7 |

What falls out of one track per derived edge, with no special case:
- A two-way track's platform carries both directions (today's "a single-track platform shares both directions").
- A one-track station's other track holds nothing at the platform (T-3332).
- Lines on different tracks of one bundle share nothing: AA/BB's express and local, or quad's two pairs, follow
  nobody and queue together only where their paths meet (as quad's pairs do today, 6.2).
- Mix (Kingman's `1 - sum share²`) keeps its streams as (line, direction), and a line on two-way track is one stream
  (T-3526).

**Junction moves from connections.** Continue, join and turnout connections become moves; crossing pairs come from
positions across the node, so the flat-junction badge, junction capacity and the train guard agree by construction
(as since T-3137). A node with only continues never changes tracks (C1). Merges and diverges cost nothing, as today.

## 3. Turnbacks and terminals

Today a turn is abstract: a train arriving at a station it stops at, or at a track end, may leave back the way it
came, holding one turn place and crossing nothing (6.2, T-797, T-3107). On one-way tracks a turn has to move from the
arriving track to the track the line leaves on, which no connection in 01 provides (a stop node has only continues).

**Proposed: a turn move at every stop node and bare end, made by the compile, never stored.** It goes from each
served track to the track the return run leaves on there (its partner, section 6, or itself on two-way track). Only
a path that turns back may use it (T-797's rule: at a stop or a track end). It is priced exactly as today: the pooled
turn place and nothing else, no platform hold, no crossing. This keeps every terminus figure where it is.

The game "places" the automatic crossovers only as drawing: the sim reports each turn move a line uses (stop node,
from track, to track, which side of the platform the train comes from) and the app draws the diagonal there. That
replaces `turnbackStrokes` (an app rule over nodes) with a sim answer, which is the class C fix. Pricing the drawn
diagonal as a crossing move would be more realistic (a terminus crossover blocks the other track) but moves every
line end's figures; that would be her call, and I would not.

**The terminal.** I recommend it stays an upgrade on the station, read by the turn-place count above (30 an hour a
pair at the default turnaround), drawn as today with buffer bars and no sidings. 01's test 2 table has the
`TERMINAL` bit replaced in stage 7 by "presets that build real track (the 30 an hour rule kept)". That conflicts:
with the bit gone, capacity would have to tell a terminal from its geometry (a scissors near a stub), which is the
kind of geometry rule this rework removes. A player who wants a real stub terminal can build one with the new model
(stub tracks ending at buffer stops, a scissors crossover in front), and it runs as built: turning at a track end, the
crossover's moves priced as junction moves. The upgrade stays the simple way to 30 an hour.

## 4. Passing and overtakes

Today a line on a shared stretch follows the slower line ahead (following loss), unless the slower one stops at a
station with the `PASSING` bit inside the stretch, where the timetable has it wait (overtake.rs). A shared stretch is
a run of path entries both lines take; with passing tracks drawn, not modelled, both lines stay on one edge through
the station.

With real passing tracks (stage 7) the paths part at the station: the stopping line takes the outer track, the
through line the inner one. Three rules keep the model whole.
1. **A shared stretch joins across a split that rejoins.** Two lines' runs on the same track, parted at node X and
   back on one track at node Y going the same way with no turn between, are one stretch. Following losses are
   counted only on the common track.
2. **A passing place** is such a split where the slower run stops between X and Y. It replaces `passing_at` and the
   bit: any layout that lets one train stand aside works, a preset or one drawn by hand. With several stops of the
   slower run inside one split (a long AA/BB stretch between two merges), the passing place is its last stop before
   Y; before that the two run on separate tracks and lose nothing to each other.
3. **Which track a line that does not stop takes.** Routing ties between the inner and outer track (same length; the
   preset's S is laid for the target speed). Ties go to the continue connection (the same slot), so a through line
   keeps to the through track. This is the class D bug T-3651 ("train took the passing tracks") made impossible.

The overtake plan itself (`overtake::plan`, `train_waits`, T-3346's train-by-train waits, T-3238's caps, T-3929's
lead-in) is untouched; only its points and keys come from rules 1-2.

**Two figures that move on old saves, stage 7.** Both come from old station-only track becoming real track.

| Old save holds | Today | Real track | Figure |
|---|---|---|---|
| Passing tracks (`PASSING`) | a stopping train holds one of two pooled platform tracks, (dwell + 60 s) / 2 | static assignment: each line stops on its own platform track (the preset puts islands so inner and outer both have one) | platform waits rise at busy passing stations where two stopping lines shared the pool; a line can be set to the inner track to spread them |
| Single-track station (loop drawn, not modelled) | one shared platform track; section cut at the station | a 2-track loop bundle, each track one way by the running side, a platform each | platform holds halve; with one line through it nothing moves (one stream, no queue); with two or more lines a little less waiting |

A third, for a single-track station with a terminal: today it has one turn place (one track), 15 an hour; a real
loop has two platform tracks, so the terminal gives 2 places, 30 an hour (her call 4).

Held fixed by rule:
- The single-track section's run time is measured from passing place to passing place as today (from the loop's stop
  point), not from the loop's end nodes, so the 240 s margin gives today's figures.
- The node where a passing place rejoins (Y) is not a merge (6.2's kind 12) for lines of one joined stretch: no merge
  resource, no `pace_merges`, no `MERGE_FIRST` weight in the spacing. Otherwise the 3% pacing and the merge-first
  placement would fight the overtake plan's placement, and today there is no merge there (both lines on one edge).
  The loop's end nodes are merges and diverges of one-way tracks, so they add no junction resource either.
- A track section is cut where its set of lines changes, so the stopping line leaving for the outer track shortens
  the main track's sections. I expect the same occupancy per train; to confirm in parity B against `build_sections`'
  section joins.

**Between stages 4 and 7**, platforms no longer split track but loops are not yet real. A platform record on two-way
single track keeps a "trains can't pass here" flag (today's one-track station on single track, `no_passing`): without
it, the stop node cuts the single-track section and its platform is one shared track, exactly as today. Stage 7 drops
the flag: the loader builds a loop where it was off, and a platform on single track with no loop is then no passing
place.

## 5. Crossovers

**Player-placed** (stage 8, the tool back, her 11-12 answer): a crossover record on a bundle node between two
neighbouring tracks, one diagonal or both (scissors). The compile makes each diagonal a short derived edge from track
A some metres before the node to track B some metres after, with a 50 km/h limit; a scissors' two diagonals cross as a
flat crossing. That is today's T-338 shape (two diagonals, four end nodes, 60 m either side), so routing, profiles
and capacity need nothing new. A line uses one when it is the fastest way to its next stop (as today). A crossover never
turns a train: turning stays at stops and track ends.

**Old T-338 crossovers** do not all fit a record: 01 allows a crossover between neighbouring tracks only, but an old
one between quad's two pairs joins a pair to a pair (on ABAB, slots 0 to 2 and 1 to 3, across the track between), and
its diagonals are double edges. In stages 3-7 an old crossover stays what it is today: its diagonals are a short
bundle of kind crossover from one node of the quad bundle to another, joined to the tracks it meets, so the derived
graph is today's and parity B's "quad with a crossover" game stays equal. In stage 8 the loader turns those that join
neighbours into records and leaves the rest as such bundles; any change there is reviewed in parity B.

**Automatic**: only the turn moves of section 3, drawn and never priced as crossings. Trains also change tracks by
themselves wherever connections allow (joins and turnouts at junctions, quad meeting double), as today; no other
crossover is ever placed by the game.

## 6. The return run: partner lookup or stored route

01: the return run is the forward path mapped track by track to each track's partner in the same bundle; only where
there is no partner is it routed, and then stored with the line.

**The partner cannot be "the nearest track running the other way".** On default quad offsets (-6, -2, +2, +6) with
ways A B A B, the B track at -2 has A tracks at -6 and +2, both 4 m off: a tie. On the real NYC save, where imported
lines sit 6 m apart (`realsave.rs`), two doubles bundled together have tracks at -5, -1, +1, +5, and the -1 track's
nearest A track is the other double's +1: the wrong pair. On AA/BB (A A B B) both A tracks pick slot 2.

**Proposed: the partner is stored per track** (one small field on the bundle's track), set by whatever lays the tracks:
a double stroke pairs its two; presets pair by their pattern (ABAB: 0-1, 2-3; AABB: 0-3, 1-2; ABBA: 0-1, 2-3); the
old-save loader pairs each old double's two tracks; a two-way track is its own partner; a track added alone has none
until the player sets a way that gives it one (the nearest other-way track without a partner, ties to the lower slot).

The lookup then holds where every forward span's partner exists and each pair of consecutive partner spans has a
connection at their node going the return's way. Where it fails anywhere (a one-way leg, a crossover on the forward
path, a turnout whose partners don't join), the whole return is routed through the line's stops and stored with it.

**When a stored return is routed again**: when an edit recompiles a bundle the stored return runs over, or changes a
platform at one of its stops; or when the line itself is edited. New track elsewhere never reroutes it, the same as the
forward path today ("new track never reroutes a line by itself"). Lines with no way back (T-3835's loops that run one
way by their track, or a broken return) go on a short list retried after every track edit, so building the missing
track still gives them their return, as `back_lines` does today; the list holds only those lines, so the whole-network
scan of `back_lines` goes (the code report's deciding constraint).

## 7. A line on several tracks of one bundle (replacing the hidden twin, T-3333)

Today a line on both quad pairs keeps a hidden second line (`LineData::twin`, `twin_ops`, `foldTwins`,
`pin_twin_phases`). Capacity, timetables, boarding, money and demand see two lines, each at half the trains, the
twin's departures half a headway after the line's.

**Proposed: patterns, derived, not stored as lines.** A line set to "use all tracks" (today's two-pairs switch) gets
P patterns, P = the most tracks of its way on any bundle of its path (2 on ABAB quad). Pattern i takes, on every
bundle, the (i mod n)-th of its n tracks that way in slot order (so all patterns share a double), and stops at the stop point of each station on that
track; it is routed once and stored as the return is (section 6), with the same re-route rules. Where a pattern can't
reach a station's stop point (no platform on its track), the switch stays on and the line runs on one pattern, the
panel saying why, as `line_pairs_why` does now.

To every consumer each pattern is one service line with its own two runs, carrying 1/P of the line's trains (counts
round up so all patterns are equal, as the twin's do), its headway P times the line's, departures 1/P of a headway
apart. So capacity, the timetable, boarding, money and demand need no change in shape, and a save with two pairs gives
the same figures as today. What goes: the twin `Op` traffic in every edit, `LTW1` (loaded as the switch), the app's
folding (the line's figures are summed in the sim).

## 8. Timetables and services with per-track stops

- A run's stops are stop points, one per stop index (`Run::stop_nodes` today), picked by the simple-stops rule (01)
  for simple stops or listed for exact stops (stage 11). Routing to a station targets its stop points going the run's
  way, cost then fewest crossing moves then the clicked track (01's rule as a search target).
- `Run::pass` keeps its meaning: a stop the run goes by on another track (one-track stations, one-way stations,
  one-way stops). One-way stations (T-3565, T-3605) stay stops of their own; nothing pairs them.
- Departure boards, the chart and boarding read the run's stop points, so a stop can show its platform track.
- Exact stops (stage 11): the cycle's stop points are listed in running order, out and back. Runs read them directly;
  `flatten`, `way_stops` and `back_run_of` are not used for such a line. Per train stops stay T-3346's (later).

## 9. What each stage does in the sim, and how it is checked

| Stage | Sim work | Replaces | Check | Hours |
|---|---|---|---|---|
| 3 | the compile (stations still bundle nodes), stable ids, node pitch, routing and capacity on derived keys, path spans; partners stored (the loader pairs each old double) with the return by partner lookup and turn moves, since a line's end now arrives on one derived edge and leaves on another | `lanes.rs`, `key_of`, `track_pos` leftovers, `derive_ways` | parity A and B below | 14-20 |
| 4 | stop points from platforms, the simple-stops rule as a search target, stored routed returns, turn places per stop node, the single-track "can't pass" flag | `perdir.rs` way stations, `back_lines`, `station_beside` for stops | A and B; her split legs on both running sides | 8-12 |
| 6 | patterns | `twin.rs`, `twin_ops`, `LTW1` | B: two-pairs save equal to today | 5-8 |
| 7 | passing places, real loops and passing tracks, stretch joining, section run time from passing places | `PASSING` bit, `passing_at`, the station loop rule | A; B except the three changes of section 4, whose diffs are reviewed | 8-12 |
| 8 | crossover records to diagonal edges, turn moves out to the app | edge crossovers' own code, `turnbackStrokes` | B: converted T-338 crossovers reviewed, the rest equal | 3-5 |
| 9 | spread zones' speed caps; faces stored, no effect on figures | | a spread island's limit never below its target | 1-2 |
| 11 | exact stops read by runs | | a line listed exactly as its simple stops would run gives the same runs | 3-5 |

**Parity A, the real NYC save** (`T-007-real-nyc.save`, nothing new used): every line's path mapped back to old
(edge, dir), run times, round trips, fleets, rates per level, every resource's rho and delay by mapped key, phases;
demand's packed arrays and rail trips a day (12,581,681 on 2026-11-16, part 6). It is necessary but weak: the save is
all double with no 3-end node and, as far as I know, no upgrades.

**Parity B, the small saves** that carry what NYC lacks, both running sides: T-3137's parity games (flat and flying Y,
flyover, one-way and two-way single branches, single Y, quad with a crossover, circle, flat crossing), plus a terminal,
passing tracks, single-track stations, two pairs with a twin, her split legs, a one-way street pair, a one-track
station. `checks/t3137-lanes-parity.mjs` and its bin are the starting point; each stage keeps them equal except
where the table says.

## Her calls

1. Turning at a station stays as today: holds a turn place, crosses nothing, the crossover only drawn? (yes; pricing
   it as a crossing move would cut what termini carry)
2. The terminal stays an upgrade giving 30 an hour, with no sidings or real stub track behind it; real stub terminals
   are something a player can build manually? (yes)
3. Old passing tracks become real track in stage 7, each line stopping on one platform track instead of sharing two:
   busy passing stations with two stopping lines wait a little more? (yes, it is what static platforms mean)
4. Old single-track stations become real passing loops, a platform on each track: a little less waiting where two or
   more lines share one, and one with a terminal turns 30 an hour instead of 15? (yes)
5. A line on several tracks runs as equal patterns, 1/P of its trains each, as two pairs does now, extended to any
   number of tracks its way? (yes)
6. Spanish solution faces change nothing in the figures for now (no shorter dwell)? (yes, flavour first)
