# T-3444 design, part 7: saves

Agent tdperf, 2026-11-21; levels, blocks and old-save steps revised 2026-11-28 by agent trackplan after Gemini's review
(10-build-plan "Review 11-28"). TWT7 is built on tw/track (body levels below). Vocabulary as in `01-model.md`; stage
numbers are `10-build-plan.md`'s. Today's format: `docs/architecture.md` section 11, `sim/src/track/save.rs` (top
comment, `BLOCKS`, `LEVEL` 34), research/code.md section 4. Her answers: `answers-2026-11-21.md`.

## What is saved

The authored records and nothing derived, as today ("inputs only, ids renumbered densely"):

| Record | Saved | Not saved (compiled on load) |
|---|---|---|
| Node | x, y, level, flying; its stored joins (track end to track end) | continues, turnouts' lateral positions, moves |
| Bundle | its two nodes, alignment (PIs, radii, levels, thru points), kind, built/off, sep; per track (any count, 1 to 16, no pairing assumed) its way, partner (a slot or none), purpose and offset profile (base offset, spread zones) | pieces, lengths, limits, ramps' shapes, derived edges |
| Platform | its station, the track at its start and the chainage there, length, edge tracks (left, right, either may be none), width, faces | stop points, derived stop nodes |
| Station | name, named, built/off, upgrades, fare zone, own name, berth | complex (derived as today) |
| Line | as today, but the path as track spans; per stop its station and two stop points, out and back (either may be none for a one-way stop); the return path only where it is not the partner mapping | derived path, keyframes |

- **No crossover record.** A crossover is an ordinary single-track bundle with a turnout (a join) at each end; a
  scissors is two such singles crossing. The save holds nothing else for it.
- **Stop points are stored, never re-derived**: the smart allocation picks both at the stop click (out and back), so
  the save carries both. Each is written as its index among the stop's station's platforms plus the edge (left or
  right), so it costs about a byte.
- **Track purpose** ("Any line", "Passing track", "Only for {lines}") is a field on each track from the start: 2 bits,
  plus for "Only for" an index into a table of line sets written once per save. Absent (Any line) until stage 7c.
- Ids in the file stay dense indexes. A track is written as its bundle's index plus its slot, so no track id is
  saved; stable in-session ids are given fresh on load. The header fields I read (`lineDays`, `notes`, `shapes`,
  `hidden`, `schematic`, `regions`) key by line or zone, not node or edge. To check at build time: `achievements`'
  `of` (what earned a step, a string) never names a node.
- **A platform across a node.** A platform may run over a plain joint, never a junction (01). It is saved as its
  start track, the chainage there and a length; the length runs on along continues past any plain joint.
- **Chainage after a reshape.** Moving a PI shifts every chainage after it, so a reshape edit rewrites that
  bundle's platforms and speed-limit ranges in the same edit, keeping them where they were on the ground.

## Format: a new network magic, TWT7

The network body changes meaning (edges become bundles of tracks, path entries become track spans, stations leave
the nodes), so blocks on TWT6 cannot carry it. **TWT7** is a new magic with its own body:

```
TWT7, body level (varint), origin lon, lat (f64), right_hand
nodes      count; per node: x, y (mm deltas), level, flags (flying, has joins);
           then, if flagged, joins: track end -> track end (each as bundle end at this node + slot)
line sets  count; per set its line indexes (for "Only for {lines}")
bundles    count; per bundle: a, b (node deltas), flags (built, off, kind, HSR, sep, has thru, all default offsets),
           track count; per track: way (2 bits), partner (slot + 1, 0 none; a two-way track its own), purpose
           (2 bits, + set index), and unless default: base offset
           (cm), spread zones (start and end chainage in mm, extra offset in cm, target speed); PIs and thru points
           as today's edges
stations   count; per station: flags (built, off, named, upgrades), name
platforms  count, in station order; per platform: station delta, start track (bundle delta, slot), start chainage
           (mm), length (20 m steps), left and right edge track (slot, or none, or another bundle for a taper),
           width (cm), faces (2 bits an edge)
lines      as today, but: per stop its station delta and the out and back stop points (platform within the station
           and edge, 0 for none); path as spans (bundle delta, slot; direction from the track's way, a bit only on
           two-way track); a flag, then the return spans only where the return is not the partner mapping
```

- **Reserve the end model's fields now** (purpose, offset profiles, width, faces, tapering edges), written as absent
  until a stage uses them. A stage that starts writing one raises the level, and a build that reads a nonzero
  reserved field it cannot use counts it as unread (as unknown flag bits are today, T-1826).
- **The trailing blocks keep their names**; under TWT7 the ones that name a node or edge take the new meanings
  below. The decoder picks meanings by magic. No old build reads TWT7, so nothing misreads a block.
- Chainages in mm, like every other position, so derived geometry stays bit-identical on load (save.rs top comment).

### Blocks and flags that change meaning

| Today | Holds | Under TWT7 |
|---|---|---|
| node flags: station, built, off, upgrades 4-7, named 8, `one` 9-10; platform length, name | station on the node | station record; `one` becomes a single platform on that side |
| edge flags: tracks, oneway, join a/b, crossover, kind, HSR, sep | per edge | bundle flags, track count and ways; joins become node joins; a crossover edge becomes a single-track bundle |
| `LNK1` | node pairs | station pairs |
| `LOR1` | line, then node indexes | line, then station indexes |
| `LFR3` | ports by node index, berth from the node | ports by station index, berth from the station's first platform middle |
| `LFZ1`, `LSN1` | by node index | by station index |
| `LBD1` | days of built stations, edges, lines | stations, bundles, lines |
| `LPS1` | speed limit per edge | per bundle with a chainage range (a limit between two stations no longer has its own piece) |
| `LRA1` | ramp per edge | per bundle |
| `LOS1` | per line, stop place and side (one-way stops, T-3711) | read on load into the stop's out and back stop points; no longer written |
| `LTW1` | quad lines and hidden twins | read until stage 6c, then folded on load and no longer written |
| `BLD1` | outlines | unchanged (membership derived by geometry) |
| `LBT1` (level 35) | BRT stations by node index | by stop node while stations are nodes, anchors included; by station from T-4043 |
| `LBK1` (level 40) | kept bus stops; a connection to a station names its node index | the same; moved with its station in 4b; by station from T-4043 |
| `LAP1` (level 41) | separated station pairs, node pairs as `LNK1` | station pairs, as `LNK1` |
| `LBO1` (level 38) | per line, a stop's place in its stops and its kind (in memory `LineData::only`, by node) | unchanged in the file; 4b moves the in-memory nodes, as for `ones` (`LOS1`) and `picks` |
| edge flags kind 4 (level 37) | a busway | a bundle kind (in TWT7's bundle flags since the 11-23 sync) |
| `LBN1` (tw/track, level 39) | bundles over pieces | read and written until one-piece bundles (6a); written last |

All the line, bus and ferry blocks (`LUI1`, `LRT1`, `LGR1`, `LPH1`, `LPD1`, `LWE1`, `LBS2`, `LBQ1`, `LBE1`, `LAF1`,
`LOF1` and the rest) are by line index and do not change. Bus stops do reach station nodes, through `LBK1`'s
connections (T-4080), so `LBK1` is in the table above.

## Save level, as built and from here (revised 11-28)

The header's `level` is only the guard (T-4014): a build refuses a save above its own level. No read depends on it:
the trailing blocks are read by name in the order they are written (a block absent is read as none), and TWT7's own
layout goes by its body level. So a level only has to be above every level the other line of builds has written.

| Line of builds | Levels |
|---|---|
| master | 34 on 11-21; then 35 `LBT1`, 36 `LBQ1`, 37 the busway (edge kind 4), 38 `LBO1`, 39 `LBE1`, 40 `LBK1`, 41 `LAP1`, 42 `LAF1`, 43 `LOF1` (11-27, 46 blocks) |
| tw/track | 39 `LBN1`, 40 TWT7, 41 TWT7 body level 3 (11-21, 11-22); 42 at the 11-23 sync, holding master's `LBE1` and `LBN1` (43 blocks) |
| the 11-28 sync | 44: master's 46 blocks in order, then `LBN1` last (47) |
| from then | each sync takes one above master's; each release 1 stage that adds a block or a body layout raises it |

Her saves from the tw/track build on 8800 (f382ee54, level 42: `... LBO1, LBE1, LBN1`) open in the synced build;
master refuses them by the TWT7 magic. The sync's check opens one.

**TWT7 body levels built**: 1 (T-3953) the body; 2 (T-3954) a path entry names its track's slot, and a line keeps a
routed return run; 3 (T-3959) platforms as records after the stations, stop points per stop; 4 (T-4031) each track's
joins; 5 (T-4085) a planned track on a constructed bundle. A path entry's part (T-4037 g1) rides in the slot varint
(`slot + 1 + 17 * part`), byte identical where there is none, so no level.

**Left in release 1**: T-4043's stations with several stop nodes (a trailing block in its plan, `LSJ1`, or a body
level), T-4517's platform ids, T-4067's join naming one track across a node, and T-3963's platforms on the tracks a
template picks. Each raises the header level; one that changes the body layout raises the body level too. T-4026
(shonei-server, readsave.py) learns the final one before the merge. Releases 2 on: as before, one raise per stage that
writes a reserved field, a new block or a new line field (run track choice, loops and passing tracks, purpose, spreads,
faces, tapers, exact stops); stage 8 needs none (its crossovers are plain single-track bundles).

**The body level.** The header's `level` reaches the network decoder (`decode_at`), but a build would still parse a
newer body with its own layout and fail or misread. So TWT7 carries its level right after the magic, and a decoder
meeting a body level whose layout it does not know stops there and reports a newer save, parsing nothing. Additions
that keep the layout (trailing blocks, reserved fields) are refused the same way since T-4014 (below).

## How old saves load (a nicety, per CLAUDE.md "Saves are hers")

Best effort, the "Older frames" pattern from architecture.md section 11: **convert on load, write the slot in the
new format, keep the old bytes under a hidden key `old-model-<slot>`** (as `old-frame-<slot>`), so a converter
fixed later can redo it from the original. Nothing is refused: what cannot be mapped is dropped and logged to the
console, the rest plays. The converter lives in a load-only module (`track/legacy/`) holding today's record shapes
and **frozen copies of `twin_at` and `way_station`**; stages 2 and 4 delete those rules from live code, and only the
converter keeps them, with no other callers. Steps:

1. **Edges to bundles** (3b). A double edge becomes a bundle of two, ways by running side, each the other's partner
   (old quad's pairs thus load as two doubles, A B A B paired 0-1 and 2-3, step 3). A single becomes a bundle
   of one; its way is `edge_way` at load (the inferred way included, after the `OLD_STATION_WAYS` pinning), since the
   new model infers nothing.
2. **Crossovers** (3b). A T-338 crossover edge becomes a single-track bundle with a join to the track it meets at
   each end.
3. **Quad** (2b). Pairs that frozen `twin_at` calls twins, with ends at the same nodes' neighbourhoods, become one
   bundle of four (slots -6, -2, +2, +6). Where the pairs are not concentric end to end (one pair split where the
   other is not), they stay **two bundles of two**, and their stations become one station with platforms on both.
4. **Joins** (3b). Today's left/right join becomes a join to that track of the double. An unjoined end stays
   unjoined and connects to every track whose way matches (01's rule for old saves), so old merged ends load as they
   are; new drawing never merges.
5. **Stations** (4b). A station node becomes a station: a double gets two zero-width side platforms, one outside each
   track; a one-track station (`one`) one platform on its side; a single-track station platforms as today's model
   holds them (the loop becomes real track at 7a). Length from the node's platform, centred at a through node, from
   the node at a dead end. Twins named alike merge into one station; pairs frozen `way_station` pairs merge into one
   station (02 "Old saves").
6. **Merging at station nodes** (4b). Once its station has moved to platforms, a two-end node is a plain joint, and
   the converter joins its two edges with the rules `unjoin_near` already uses (wasm/edits.rs, T-1964/T-244: same
   built and off, track count, kind, way, speed limit and ramp; no crossover, no thru points; radii kept). That join
   straightens a joint up to `JOINT_KINK_M` off the straight, which changes lengths slightly, so the converter keeps
   the merge **only where the joined bundle's compiled length is within 1 mm of the two edges'** (T-3961: to the bit
   merged only 22 of 4,581 reference stations, within 1 mm 2,790; 1 mm is invisible); elsewhere (a station
   mid-curve, a kink) the node stays as a plain joint and the platform runs across it. The earlier build day wins, as
   `joining_pieces_keeps_the_earliest_day` already tests. The station becomes an anchor on the joined piece (T-4037).
   **Every field keyed by the old station node moves with it** to the anchor's node, in memory, in the same pass:
   the station record (fare zone `LFZ1`, own name `LSN1`, BRT road `LBT1`, build day `LBD1`, upgrades, berth `LFR3`),
   links `LNK1`, separated pairs `LAP1`, `LOR1`'s node lists, kept bus stops' connections `LBK1`, and each line's
   stops, `picks`, `ones` and `only`. One missed is a quiet wrong answer (a fare zone or a separation lost), so 4b's
   test lists each.
7. **Lines** (3b, 4b). Each path entry `edge<<1|dir` gives a track (the direction picks the track on double). Under
   her (b) (T-4037, 11-23) merged edges become parts of one piece, exactly: `(P1, part j)` and `(P2, part j)` become
   `(P, part j)` and `(P, part c + j)`, `c` being P1's parts, and a turn at a merged station stays a turn there. Stops map node to station. Each stop's out and back stop points are what the run
   already stops at today: the platform its path passes going each way, the `way_station` partner where today's rule
   sends it there, one way only where `LOS1` says so. So every line stops where it did.
8. **Settled state** (`LRT1`, `LPC1`, `LWP1`, the `ERA1` tail). Kept, since lines keep their indexes, merges are
   length-exact and the parity gate at stages 3 and 4 (part 8) requires the converted network to give today's paths,
   profiles and capacity exactly. A line the conversion changed (a twin folded at 6c, a turnback crossover added at
   8) loses its rates and settles on load, as an edit does. If parity cannot be held at some stage, that stage's
   converter drops the settled blocks and the save settles in full once: about 43 s natively for a London-sized
   save (T-3233), about three times that in the browser.
9. **Upgrades.** Waiting room and park and ride move to the station; terminal kept (part 5 keeps its rule);
   passing tracks become real track at 7b by running the preset in the loader, dropped where it is refused.
10. **Turnback crossovers** (8). Where a line turns back on double track with no crossover there, the converter lays
    a real crossover single, free, as the stop edit would lay one.

A player's game started from a shipped real network is an old save like any other and takes this converter.

**Other readers of the network bytes**, each to learn TWT7: `trackFrameKind` and `isRetiredTrackFormat`
(workers/saveFile.ts), the frame mapping in `game/frame.ts`, the shared-save check in `game/sharedSave.ts` (its
`/^TWT[4-6]$/` already refuses TWT7 as "version" in old builds, which is right; new builds widen it),
`TrackApi.save_stations` (the demand pool opens zones ahead from the bytes), `save_unread`, `convert_save_frame`
(an anchor's middle scales with its piece, branch review finding 11), `pipeline/readsave.py`, and shonei-server's save
check (T-4435 taught it level 43, not deployed; T-4026 teaches it TWT7 and the final level, one deploy before
release 1 ships). A TWT4 or TWT5 save converts its frame first (to TWT6 bytes, as today), then its model.

**Reference saves for the conversion**, in this order: her Amsterdam "test" save
(`notes/T-3444-design/saves/amsterdam-test-day-61.save`, the lightest, odd layouts), the community Vienna save
(share id 3znpvqx7 rev 2, 759 stations, day 337, tram single-track oddities; fetched read-only in stage 1, as the
fetch on 11-21 was rate limited) and North America day 383. Each must convert cleanly at 3b and 4b; every other save
is best effort. The real NYC save stays the developer parity save (part 8, R10).

## What plays read-only

Nothing, since T-4014 (2026-11-21): a save above the build's level, with an unknown block or header field, or with flag
bits or reserved fields it does not know, is refused at open with "This game was saved on a newer game version. Reload
the page to update.", its slot left as it is (docs/architecture.md "Saves from a newer build"), and an older build
cannot upload over a newer account copy. tw/track's levels must stay above master's for this to hold (each sync
renumbers them; master also refuses any TWT7 by its magic, T-3948). The read-only
mode below was T-1826's.

- **In a new build**: a converted old save plays and saves as TWT7.
- **In an old build**: today a TWT7 save would fail outright. `trackFrameKind` (workers/saveFile.ts) treats any
  magic other than TWT6/TWT5 as the equirectangular frame, the conversion returns nothing and the decoder says "not
  a trainworld track save". It cannot play read-only, since it cannot decode. **Stage 1 fix on master (0.5-1 h):** a
  magic `TWT7` or later is reported with the existing string "This game was saved by a newer version. Reload the
  page to update." (reused word for word), and the slot is left alone. It ships before the first merge of `tw/track`,
  and hotfix builds from master carry it.
- **A stage 3 build opening a stage 4 save**: the body level stops the decoder (above), same message, slot left
  alone. Not playable read-only, since the station section's layout differs; the only such step in the plan.

## Size on the real NYC save

Real NYC (`T-007-real-nyc.save`): 109 lines, 1,756 stations, 1,647 edges (all double, every edge station to station:
1,756 - 109 = 1,647), 8,293 PIs, network 128 KB raw, 75 KB gzipped. The split by section is **my estimate from byte
counts per field**, to be replaced by the stage 1 measurement.

| Section | Today | TWT7, converted with merging | TWT7, no merging |
|---|---|---|---|
| Nodes | ~39 KB (1,756 with names and platforms) | ~2 KB (~220 track ends) | ~16 KB (1,756, no names) |
| Edges / bundles (head) | ~5 KB | ~1 KB (~110 bundles) | ~8 KB |
| PIs | ~80 KB | up to ~94 KB (+1 PI per merged station node, at most 1,538) | ~80 KB |
| Stations | (in nodes) | ~26 KB (name, flags) | ~26 KB |
| Platforms | (in nodes) | ~7 KB (3,512, chainage deltas) | ~7 KB |
| Lines | ~6 KB | ~8 KB (about one span a line, two stop points a stop) | ~10 KB |
| **Raw** | **128 KB** | **~138 KB** | **~147 KB** |

About +8% raw with merging, +15% without; gzipped about 76-84 KB; in line with T-3137's +10 KB raw for a shared
alignment plus a group id. Spreads (~8 bytes a zone), purposes (2 bits a track) and joins cost only where used.
Stage 1 measures it: a scratch bin encoding each section alone on real NYC, the Amsterdam test save, North America
day 383 and London's start (1 h).

## Undo

Undo stays record replacement (`Op`, world.rs). New kinds: **Bundle** (id, old record or none), **Node** (with its
joins), **Platform**, **Station**; **Line** as today with spans and stop points. A crossover is a Bundle op like any
track. Split and DeleteEdge become Bundle and Node ops; Batch stays.

- Undo restores the same stable ids, so the drawn network's cache and the stations' panels find their records again;
  ids are never reused while a history entry names them.
- Derived ids are not restored; the touched bundles are compiled again after an undo, as after any edit.
- A way or purpose set on a long selection is one Batch of the traced bundles' records.
- Memory per step is the replaced records. Deleting one track replaces its whole bundle (PIs included): a few KB
  usually, up to ~180 KB for a 5,400 km bundle of 5,688 corners, which today's long edges already cost.
- Buildings stay outside undo, as today.

## The real networks' shipped saves

**Rebuilt by the pipeline in the new format at 3g and 4f, not converted** (part 9), keeping today's stations:
- A start must load settled (London 0.6 s natively against 42.8 s unsettled, T-3233); `realsave` settles and writes
  `LRT1`, while a start converted at load would settle on every new game.
- `realsave.rs` builds `EdgeData` and `StationData` directly, so it stops compiling at stage 3 anyway.
- `realshared.py` knows each piece's pair count, so it writes bundles of four with no geometry guess.

The 41 files in `data/packs/real/` are written new (data/frozen/ links the old ones, T-731).

**Work, hours of agent time**: stage 1 magic guard 0.5-1; 2b `LBN1` and quad grouping 2-3; 3b TWT7 reader, writer,
converter (edges, crossovers, joins, spans), round-trip tests 6-10; 4b stations, platforms, stop points, merging,
frozen `way_station` 5-8 (3b built 11-21; 4b under (b) is a renumbering plus the field moves, 2-3 h in 10-build-plan); `readsave.py` 1-2; each later stage's fields 1-2.

Nothing open for her.
