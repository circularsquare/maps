# T-3444 design, part 1: the model (tracks, bundles, nodes, platforms, stations)

Agent trackdesign, 2026-11-21. Design draft for discussion with Anita; nothing is built. Research behind it:
`research/code.md` (today's code, bug classes, performance), `research/players.md`, `research/games.md`,
`research/realworld.md`. Her notes: `../T-3444.md`. Earlier designs this builds on: `../T-3137.md` (lanes, and the
rejected "two objects" option (b)), `../T-3365.md` (single/double/quad, her 11-12 answers), `../T-3348.md` (AA/BB and
quad stations, stage 1 per-direction stops built).

## Vocabulary (fixed for every part of this design)

| Word | Means | Player sees |
|---|---|---|
| **track** | One rail track: the thing a train runs on. Has a way (one way, or both ways). | "track", "one track of a double" |
| **bundle** | Tracks laid together on one shared alignment, from node to node. Saved as one record. Single track is a bundle of one, double a bundle of two, quad of four, any count after. | "single track", "double track", "quad track", "6 tracks" |
| **slot** | A track's place across its bundle, counted 0, 1, 2 ... from the left looking from the bundle's start to its end. | never |
| **offset** | How far a track's centre line sits left (+) or right (-) of the bundle's alignment, in metres. Varies along the bundle where tracks spread. | never |
| **node** | A cross-section where bundles end and meet: a junction, a track end, a place where the track count changes. Every track of a bundle ends at the same node. | "junction", "track end" |
| **connection** | One move a train may make through a node, from one track end to another. Kinds: continue (the same track carries on), join, turnout, crossover. | crossovers are drawn; the rest is just "track" |
| **platform** | A strip beside a track, or between two tracks, over a length. It serves the tracks along its edges. Zero width (today's look) or an island with real width. | "platform", "island platform" |
| **stop point** | Where a train stops on one track at one station, named (station, track): the middle of the station's platform edges on that track (its end at a buffer stop). Derived, never placed. | the numbered circles of exact stops |
| **station** | A named group of platforms. Part 2 defines it. | "station" |
| **complex** | Stations a rider changes between on foot (as today). | "station complex" |
| **simple stops** / **exact stops** | A line's stops as stations (today's editing), or as stop points in running order. Part 4. | stop editing, and the exact option beside it |
| **preset** | A named layout the tools build in one go: double track, quad, AA/BB, an island station, passing tracks, a terminal. | the icons in the bars |
| **derived graph** | The graph the simulation runs on, compiled from the saved model. Internal. | never |

## The idea in one paragraph

What the player builds and the game saves is a small set of **authored records**: bundles (with their tracks),
nodes (with their connections), platforms, stations and lines. Every track is its own single track with its own way,
so a double is two one-way tracks that happen to share an alignment, exactly as she asked ("after they are drawn they
should mostly just be 2 single tracks in slightly different positions"), and saved joined ("in terms of how we save
it we can potentially keep them joined"). The simulation does not read these records directly. They are compiled
into a **derived graph** of single-track edges with stop nodes, in the shape today's capacity, profiles, services,
boarding and demand already read (one track per edge, path entries `edge << 1 | dir`). The compile is per bundle, so
an edit recompiles only the bundles it touched. Drawing reads the authored records, keyed by ids a platform or a
station never changes.

## The records

### Bundle

- **Alignment**: today's edge geometry: PIs with radii, levels and ramps, the start node and end node, kind (regular,
  high speed, tram, street tram), speed limit. All of `geom.rs` keeps working on it.
- **Tracks** (1 to 16, ordered by slot): each has a stable **track id**, a **way** (a to b, b to a, both), an
  **offset profile** (its base offset plus any spread zones, below), its own **built/blueprint** state and build day
  (an added fifth track, or an island's reconstruct, is blueprint while the rest is built), and its **partner**: the
  track a return run uses (below). Default offsets: 4 m between neighbouring tracks (today's double is ±2 m, quad's
  pairs 8 m apart centre to centre, so quad's default slots are -6, -2, +2, +6). Offsets are stored per track, so an
  imported real network keeps its own spacing (NYC's pairs 6 m apart) and figures do not move on load.
- **Partner** (stored, set by the tool or preset that made the track): on a double, the other track; on quad side by
  side, the other track of its pair (A B | A B); on AA/BB, the outer tracks partner each other and the inner tracks
  each other (A A B B: 0 with 3, 1 with 2); on ABBA, 0 with 3 and 1 with 2. A one-way track with no partner (a lone
  leg) has its return routed. A geometric "nearest other-way track" rule was rejected: it ties on default quad and
  picks the express track for a local on AA/BB.
- **No stations inside**: a platform lies on a bundle without splitting it. A bundle runs from junction or end to
  junction or end. The only reasons for a node mid-way are a junction, a change in track count or kind, a level
  change point the player made a node, and a crossover (below).

A **double drawn in one stroke is one bundle with two tracks**, way by the running side. A single is a bundle of one
track, two-way unless set. Today's two pairs of quad 8 m apart become **one bundle of four**: this is "quad as one
remembered thing" (her 11-12 answer) by construction, not by a geometry rule.

### Lateral offset: one rule (her island case and tapers)

A track's offset can **vary along its bundle**; tracks never leave a bundle part way. Leaving happens only at a node,
where a track ends and the next bundle's track carries on (a turnout).

- An **offset profile** is the base offset plus zero or more **spread zones**: (start chainage, end chainage, extra
  offset). Between the base and the spread the track runs a reverse curve (S) whose length comes from a target speed:
  for a sideways shift `d` with curve radius `R`, the S is `2 * sqrt(R * d)` long (two arcs). `R` is the radius the
  game already uses for that target speed. Example: an island 6 m wide spreads each track 3 m; at 80 km/h the S is
  about 70 m each end, at 160 km/h about 140 m. The speed is the bundle's target max speed, or the flyover target
  speed if she prefers the shorter S (question for her, part 3).
- Drawing, train positions and the lateral positions at nodes all come from `alignment + offset(chainage)`. Lengths
  differ from the alignment's by a few centimetres in a spread; the compiled edge takes the alignment's length (a lie
  of under 0.1%, recorded in DECISIONS when built).
- A track in a spread runs at the speed the S allows. Since the S length is chosen from the target speed, the limit is
  never below the bundle's target.

Why not "a track leaves the bundle": that is what a turnout is, and it needs a node, a new bundle, and the island
would no longer be one piece the player picks. The offset profile keeps the island station one bundle.

### Node

A node is a point (x, y, level) where bundles end. Every track of a bundle ends at its bundle's end node; there are
no per-track nodes. A node's **track ends** each have a lateral position across the node (the bundle's slot offset
there, plus the bundle end's own position across the node, as `derive_node` works it out today). Positions are metres,
replacing `capacity.rs`'s rank arithmetic.

**Connections** are what a node stores beyond geometry. There are four kinds:

1. **Continue** (never stored): where a bundle's pieces meet end to end with the same track count, slot i carries on
   as slot i. A node that only joins two pieces of one stroke has nothing but these. Changing tracks there is never
   allowed (today's rule: no wrong-line running at a plain node, DECISIONS T-3137 C1).
2. **Join** (stored per track end, as today's `EdgeData.join` generalised from "left or right of a double" to "this
   track id"): where track counts differ or bundles of different strokes meet, which track end each continues into.
   Set by the drawing tool when it builds the meeting (it knows which track the pointer picked, T-3137 C3), so it is
   never guessed later. A track end with no join at a junction connects to every track end on the other side whose
   way matches (today's rule for an unjoined single; old saves rely on it).
3. **Turnout**: the default at a junction where a bundle branches off another: each track of the branch connects to
   the track of the main bundle with the same way and nearest lateral position. Built by the tool and stored as joins,
   so later edits don't re-derive it.
4. **Crossover**: an explicit extra move between two neighbouring tracks of the same bundle, at a node placed for it
   (one direction: a diagonal; both: a scissors). It is a stored record on the node, not track the player edits: the
   compile turns each diagonal into a short derived edge from one track before the node to the other after it, at
   50 km/h, a scissors' two diagonals crossing flat (today's T-338 shape, so routing, profiles and capacity treat it as
   they treat today's crossovers; part 5). Priced as today's crossover. Today's T-338 crossover edges between two pairs cross a track in between, so they load as they are (a
   small crossover bundle) and become records in stage 8.
5. **Turn** (never stored, made by the compile): at every stop node and bare track end, a move from a track to its
   partner, so a line can turn back there. It holds a turn place as today and crosses nothing (part 5); the drawing
   shows the assumed crossover as today.

Track ends of neighbouring bundles at one node are spaced half the two bundles' widths plus 4 m apart across the node
(today's fixed 8 m pitch would let a quad's ends interleave with a neighbour's and invent crossings).

What edits do to connections:
- **Split** a bundle (a new node part way, e.g. for a crossover or a branch): the new node gets continue for every
  slot. Nothing stored.
- **Join** two bundle ends end to end: same count, continue; different count, the tool stores joins by nearest
  lateral position and shows them in the ghost before the click.
- **Delete one track** of a bundle (second click, her 11-12 answer): its slot goes, joins and crossovers naming it go,
  the other tracks keep their offsets (no shift), lines on it are rerouted through the same stops or the edit is
  refused naming the line (as a way change does today, `way_reroutes`).
- **Add a track** (the fifth track): a new slot at the chosen side, offset 4 m past the outermost, continue at both
  ends where the neighbouring bundle has a track at that place, else a loose end. Part 3.

### How this answers T-3137's reasons for rejecting "two objects"

T-3137 option (b) (two single-track edges per double, each with its own nodes) was turned down on 2026-11-10 for four
named reasons. This model differs from (b) in exactly the places those reasons point at.

| Reason (b) was rejected | Here |
|---|---|
| Junction nodes and moves grow 2-4x (a double Y becomes several nodes and internal diamonds) | Tracks share their bundle's nodes, so a double Y is still one node. The moves are the same set today's lane move table holds (one per track pair that may follow), so the count does not grow. |
| Shallow crossings 4 m apart at every turnout trip the bad-crossing rule | Tracks of one bundle are never tested against each other for crossings (parallel by construction). Track ends meet at nodes, so a branch never crosses its parent part way. Crossing moves inside a node come from lateral positions, as today. |
| Stations span two track nodes | Stations are not nodes. A platform lies on a bundle; its stop points are compiled onto each served track at one chainage. |
| The grouping layer must keep every edit whole | The bundle is the saved record. Select, delete, price, reshape and drag act on it whole; one track is reached only by a second click (her 11-12 answer), and "delete one track" is one edit on the bundle record. |

What (b) offered that is kept: flyovers, crossovers, single-lead junctions, scissors and spreads round an island are
all ordinary records here, not drawing patches.

### Platform

A platform is: a **station id**; where it lies: a **start track**, the **chainage** on it where the platform starts,
and a **length**, running on along that track's continues (so it may span a plain joint or a crossover node, never a
junction; the save stores exactly this, part 7); its **edge tracks**: the track on
its left edge and the track on its right edge (either may be empty: a side platform has one); a **width** (0 = today's
zero-width look, drawn over the gap between tracks; above 0 = an island or side platform with real width, which puts a
spread zone on the tracks beside it); a **level** (the bundle's at that chainage). Optional **faces**: board, alight
or both on each edge (the Spanish solution; default both).

- Edge tracks are named by track id, so they may belong to **two different bundles** where those diverge from a node:
  that is a **tapering platform** between a Y's legs. Its width follows the gap between the two tracks along its
  length, and its start chainage is measured on one of them. (Research found wedge stations, Lewes and Zürich
  Oerlikon, but no real single island that tapers; the model allows both.)
- A platform **serves** each edge track. Train length on a track is capped by the platform's length, as today.
- A platform never splits anything. It is valid when its length lies on track with no junction node inside it (a
  crossover node or a plain joint inside is fine), not on a ramp, no flat crossing within (today's rules, kept).
- **Zero width overlaps freely**: today's look and ease stay ("we wanna maintain platforms that take up no space /
  can overlap, for ease"). Two zero-width platforms may share a gap.
- A **drawn platform** (T-3859, the last stage) is a polygon the player draws; it serves every track whose centre line
  runs along one of its edges within 2.5 m for at least 60 m. It compiles to the same record (a range per served
  track), so nothing downstream knows it was drawn.

### Stop point (derived)

Each (station, track) pair with a platform edge on that track gives one stop point, named by those two ids, so moving
or reshaping a platform keeps it, and a Spanish-solution track with a platform on each side has one stop point, not
two. It sits at the middle of the platform edges on that track (at a buffer stop, the train's stopping place as
today). The compile splits that track's derived edge there and makes a
**derived stop node**. All tracks of one bundle served by platforms with the same middle share one derived cross-section
node (so an ordinary double-track station is one node, as today); offset platforms make separate nodes.

### Station and complex

Part 2. In short: a station is a named group of platforms: those placed together, those placed overlapping it side by
side at any level, and those the player joins to it. The complex stays as today above it.

### Line

- **Path**: stored in authored terms, as a list of (track id, direction) spans from first stop to last. A span
  covers a whole bundle track from node to node, so placing or removing a platform changes no line's path.
- **Return run**: by default the forward path mapped track by track to its stored **partner** (on a double, the
  other track). That is a lookup, not a search, so a line on
  ordinary double, quad or AA/BB still costs nothing to rebuild on an unrelated edit. Only where a track has no
  partner (one-way legs drawn as separate singles) is the return routed, and the result is stored with
  the line, so `back_lines` stops rebuilding every such line on every edit (the code report's deciding constraint).
- **Stops**: simple stops are station ids plus, per stop, a **preferred stop point**: when the stop is added, the one
  the line reaches from the stop before with the fewest crossing moves, on a tie the one whose track is nearest the
  pointer (simple stop markers are one per station, so a click does not land on a track); the track chip changes it.
  Exact stops are stop points in running order for the whole cycle, out and back. Part 4 has the details.

## Simple stops: the one rule (class D bugs)

Today each run's platform is derived by distance (`way_station`, the complex then 300 m, `back_lines` every edit).
That derivation is where the Caracas and Nagoya reports landed (11 fixes in a week). The new rule removes the
distance search; membership in a station does the work instead.

> **A run stops at a station on the stop point of that station its path reaches going its way.** If its path can reach
> more than one, it takes the one on the track it is already on (no crossing move); if still more than one, the
> stop's preferred stop point. A run that can reach no stop point of the station going its way passes it (as a
> one-track station's other way does today).

- **Ambiguous** only where a run could reach two platform tracks of one station going its way with the same number of
  crossing moves: quad by direction with crossovers both ways before the station, or a station with a platform on both
  sides of a two-way single. Then the preferred stop point wins, and the stop list shows the track chip (part 4) so the player
  sees which.
- **Her split legs 150 m apart** (T-3348 stage 1, players 2026-11-13): two one-way legs with a station each. Under this
  rule the two platforms must be **one station**. Placed as one click on both legs (the station tool's count picks
  "both tracks", part 3), or joined later ("Join to station" in the station panel), they are; then each run stops on
  its own leg with no distance rule. Old saves: the loader joins exactly the pairs today's `way_station` would pair.
- **T-3605 "no pairing"** (2026-11-17: a one-way station on one-way single track is a stop of its own, ordered by the
  run that passes it): kept. Two one-way stations that the player has not joined are two stations; a line lists each
  as its own stop, served by the run going its way. Joining is always the player's act, never a distance guess.
- **Exact stops store** the stop points themselves in running order (out and back), with waypoints; nothing is
  derived. Part 4.

## The derived graph (what the sim reads)

- One **derived edge per track span** between derived nodes (bundle nodes plus stop nodes), one lane each, with the
  track's way. Path entries keep `edge << 1 | dir`, so `service.rs` keyframes, `capacity.rs` keys (`key_of`),
  `profile.rs`, `boarding.rs` and demand's `packService` keep their shape; only how the ids are made changes.
- **Moves** at a derived node come from the connections (continue, joins, turnouts, crossovers) with lateral positions
  in metres. A stop node has only continues (and any crossover placed there).
- **Ids**: derived ids are internal; nothing saved or drawn holds them. They change only inside the bundles an edit
  recompiles; untouched bundles keep theirs (incremental capacity and figures that must not move by a bit depend on
  it), and a load numbers them the same way every time. The compile keeps a map from (track id, chainage) to (derived
  edge, position) for the sim's answers to the app.
- **Compile scope**: an edit recompiles the bundles it touched and the nodes at their ends. A platform edit recompiles
  only the bundles under that platform. A long bundle (a 5,400 km snapped route is one stroke) is compiled, cached and
  drawn in **chunks** cut by PI content, so an edit near one end does not redo the whole bundle (part 8).
- **Routing** (stop insertion, the ghost's line reroutes) searches the authored tracks between nodes, which are far
  fewer than derived edges; stop points are positions on them.
- **Node size**: the cap on ends at a node (`MAX_PORTS`, 8) counts bundle ends, not track ends, so a quad meeting a
  quad is 2 ends, not 8.

### Platforms across nodes, and reshapes

- A platform's place is its start track, start chainage and length (above), running on along continues.
- Reshaping or dragging an alignment changes chainages downstream. The edit moves each platform on it to the point
  nearest its old middle and keeps its length, and rewrites speed-limit and spread ranges the same way; a platform
  that no longer fits is refused with the edit, as today's reach rules refuse.

## Test 1: the layouts she named

| Case | How it is stored | How a player builds it | Stage |
|---|---|---|---|
| Plain double, side platforms (today's default) | Bundle of 2 tracks; one station with two zero-width platforms (one outside each track) | Draw double; station tool click (as today) | 3 |
| Single track, station with passing loop (today's "single-track station split") | Single bundle with a node each side of the station; between them a 2-track bundle (spread from 0 to ±2 m over the S); a platform outside each track | Station on single track builds the loop preset (as it is drawn today, now real) | 7 |
| One-track station (T-3332) | One platform on one track | Station tool with 1 picked, pointer on the track | 4 |
| Quad side by side (two pairs) | Bundle of 4: ways A B A B by running side | Draw quad (one stroke), or Select bar 4 on a double | 3 |
| AA/BB (by direction, locals outside) | Bundle of 4: ways A A B B; two islands between slots 0-1 and 2-3, or side platforms | Quad by direction preset; station with 4 picked | 6 |
| ABBA | Bundle of 4: ways A B B A; one wide island between slots 1-2 (or two side platforms outside) | Preset variant, or Select a quad and flip two tracks' ways | 6 |
| Island platform on double | Bundle of 2 with a spread zone at the station (each track 3 m out) and one platform of width 6 between them | Station tool's island choice; S length from target speed | 9 |
| Spanish solution | One track with a platform each side; faces board / alight | Platforms on both sides, then set faces in the station panel | 9 |
| AA above BB (stacked, 125th St Lexington) | Two bundles at different levels on the same plan line (overlap allowed across levels); one station whose platforms are at both levels; circles tinted by level | Draw two doubles at two levels (or one quad split by level, later preset); one station tool click per level, joined as one station by overlap | 4 (stations), 6 (preset) |
| Lexington Av/59th St (two levels plus a crossing line) | Two stations (IRT and BMT), each its platforms; one complex | As built, a passage or overlap makes the complex | 4 |
| Offset platforms (NYC, longitudinally offset) | Platforms on different tracks at different chainages; one station if the player joins them, else two stations in one complex | Two clicks; "Join to station" | 4 |
| Tapering platform between a Y's legs | Platform whose left edge is a track of bundle 1 and right edge a track of bundle 2, from the node outward | Drawn platform tool, or station tool in the wedge | 10 |
| Grand Central stubs | Many bundles of 1-2 tracks fanning from a throat of nodes; each stub ends at a buffer stop; islands between pairs | Draw tracks one by one (second click picks one track), platforms per gap | 3-9 |
| Grand Central loop tracks | A single-track bundle looping from the throat back to the throat; no platform needed | Draw it; nothing special | 3 |
| Platform-less tracks (storage, through tracks at a station) | Tracks with no platform on either edge; a line through them passes | Nothing extra; the station tool skips them | 4 |
| Passing tracks at a station (today's upgrade) | Bundle widened from 2 to 4 tracks over the station (nodes either side, turnouts), islands so every track has a platform (close to today's two platforms each way) | Passing tracks button builds the preset; or draw it by hand | 7 |
| Terminal (today's upgrade) | Stays the upgrade bit: 30 an hour, no stub track drawn (T-3107) | Station panel button, as today | kept |
| Scissors terminal built by hand | Stub bundle of 2 ending at buffer stops; a node short of the platforms with both diagonals (a scissors crossover) | Crossover tool | 8 |
| Quad tram | Bundle of 4 of kind tram, off street (as today's quad tram) | Draw quad with Tram | 3 |
| Hex and more | Bundle of 6, 8 ... (cap 16) | Track count in the track bar's Advanced row, or add tracks one by one | 6 |

Where the table is thin: stacking (AA above BB) relies on two bundles at different levels overlapping in plan, which
today's drawing allows across levels; a one-stroke "quad by level" preset would need a bundle whose slots carry
different levels, which this model does not have (each bundle has one level profile). Recommended: stacked layouts
stay two bundles, joined as one station by overlap. Grand Central's throat is many small bundles and nodes; it is
buildable but slow to build by hand, which is what the station editor (later) is for.

## Test 2: what each stand-in is replaced by, and the stage that deletes it

Every rule today that finds a pair or draws missing track, with its replacement and the stage that removes the old
code (so the two models never live side by side for long). Stage numbers are part 10's.

| Today (file) | What it does | Replaced by | Deleted in stage |
|---|---|---|---|
| `twin_at` (wasm/draw.rs) | finds the pair beside, 4-15 m, scans every edge | bundle slots | 2 |
| `station_beside` / `on_track_beside` (twin.rs) | station on the other pair, 4-16 m | stations hold all their platforms | 4 |
| `quad_beside` (wasm.rs) | the other pair's loose end, 4-12 m | the bundle's end node | 2 |
| `add_crossover`'s 6-15 m search (wasm/crossings.rs) | finds the parallel edge | crossover record on a bundle node | 8 |
| `quadPairs.ts` | "Quad" tag, 8 ± 3 m | bundle track count | 2 |
| `orderOverlapping` (stationSnap.ts) | which pair gets the station | pointer picks tracks of one bundle | 4 |
| doubling up's `Alongside` (wasm/crossings.rs) | refuses a pair beside | Select bar 4 adds slots to the bundle | 2 |
| `besideSamples` / corridors (corridors.ts) | parallel edges within 20 m for line ribbons | lines on a track are known from paths; ribbons per track | 5 |
| the complex rule written twice (twin.rs, game/complexes.ts) | complexes | one copy, in the sim, sent to the app | 4 |
| `legShifts` / `shiftedEnds` (network.ts) | draws a single end on the joined track | each track drawn at its own offset | 3 |
| double-to-single taper (`taperAt`, network.ts) | 40 m merge drawing | the join's real geometry (a turnout) | 3 |
| `singleSplitEnds` (terminalBays.ts) | draws a passing loop at single-track stations | the loop is real track (preset) | 7 |
| `passingTracks`, `linePassing`, `passLead` (network.ts) | draws passing tracks | real tracks | 7 |
| `turnbackStrokes` (network.ts) | assumed crossovers at line ends | the sim reports each turn move a line uses and the app draws it there (part 5) | 8 |
| `singleLeadStrokes` (network.ts) | the single-lead link | a join plus a crossing move, drawn from the record | 3 |
| hidden twin line (`LineData::twin`, twin.rs, `foldTwins`) | a line on both quad pairs | one line whose runs use any track of the bundle, assigned per run (part 5) | 6 |
| `perdir.rs` way stations, `back_lines` | each run's platform by distance; rebuilt every edit | the simple-stops rule above; stored return paths | 4 |
| `EdgeData.tracks`, `join` (2 values), `derive_ways` / `run_group` / `group_ways` | track count on an edge, inferred ways | tracks with their own way; joins by track id | 3 |
| `lanes.rs` two-lane cap | lanes inside an edge | one track per derived edge | 3 |
| station fields on the node (`StationData`, `one`, `platform_reach`, `check_station` reach rules, `keep_under_stations`, `station_trim_ops`) | a station as a node splitting track | platform records | 4 |
| `upgrades::PASSING` bit | passing tracks drawn but not modelled | a preset that builds real track | 7 |
| `upgrades::TERMINAL` bit | turnback sidings, not drawn | kept as it is (30 an hour, part 5); without it capacity would need a geometry rule to spot a terminal | not deleted |

## What stays exactly as it is

PI and arc geometry, levels, ramps, flyover planning (now laying tracks of bundles), costs by level and kind, run
profiles, keyframes, the capacity formulas (section 6.2), services, boarding, money, undo by record replacement (new
`Op` kinds: Bundle, Platform, Station, Crossover), demand's network arrays, station buildings and passages, buses
and ferries.
