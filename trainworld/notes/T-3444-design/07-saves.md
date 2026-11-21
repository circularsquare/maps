# T-3444 design, part 7: saves

Agent tdperf, 2026-11-21. Design draft for discussion; nothing is built. Vocabulary as in `01-model.md`; stage
numbers are `10-build-plan.md`'s. Today's format: `docs/architecture.md` section 11, `sim/src/track/save.rs` (top
comment, `BLOCKS`, `LEVEL` 34), research/code.md section 4.

## What is saved

The authored records and nothing derived, as today ("inputs only, ids renumbered densely"):

| Record | Saved | Not saved (compiled on load) |
|---|---|---|
| Node | x, y, level, flying; its stored connections (joins by track, crossovers) | continues, turnouts' lateral positions, moves |
| Bundle | its two nodes, alignment (PIs, radii, levels, thru points), kind, built/off, sep; per track its way and offset profile (base offset, spread zones) | pieces, lengths, limits, ramps' shapes, derived edges |
| Platform | its station, the track at its start and the chainage there, length, edge tracks (left, right, either may be none), width, faces | stop points, derived stop nodes |
| Station | name, named, built/off, upgrades, fare zone, own name, berth | complex (derived as today) |
| Line | as today, but the path as track spans, stops as station ids plus the clicked stop point where one was clicked, the return path only where it is not the partner mapping | derived path, keyframes |

Ids in the file stay dense indexes. A track is written as its bundle's index plus its slot, so no track id is
saved; stable in-session ids are given fresh on load. The header fields I read (`lineDays`, `notes`, `shapes`,
`hidden`, `schematic`, `regions`) key by line or zone, not node or edge. To check at build time: `achievements`'
`of` (what earned a step, a string) never names a node.

## Format: a new network magic, TWT7

The network body changes meaning (edges become bundles of tracks, path entries become track spans, stations leave
the nodes), so blocks on TWT6 cannot carry it. **TWT7** is a new magic with its own body:

```
TWT7, body level (varint), origin lon, lat (f64), right_hand
nodes      count; per node: x, y (mm deltas), level, flags (flying, has stored connections)
           then, if flagged: joins (track end -> track end, each as bundle end at this node + slot),
           crossovers (a bundle end, two neighbouring slots, one way or both)
bundles    count; per bundle: a, b (node deltas), flags (built, off, kind, HSR, sep, has thru, all default offsets),
           track count; per track: way (2 bits), and unless default: base offset (cm), spread zones
           (start and end chainage in mm, extra offset in cm, target speed); PIs and thru points as today's edges
stations   count; per station: flags (built, off, named, upgrades), name
platforms  count, in station order; per platform: station delta, start track (bundle delta, slot), start chainage
           (mm), length (20 m steps), left and right edge track (slot, or none, or another bundle for a taper),
           width (cm), faces (2 bits an edge)
lines      as today, but: stops as station deltas, each with a stop point byte (0 none clicked, else the platform
           edge); path as spans (bundle delta, slot; direction from the track's way, a bit only on two-way track);
           a flag, then the return spans only where the return is not the partner mapping (01 "Return run")
```

- **Reserve the end model's fields now** (offset profiles, width, faces, crossovers, tapering edges), written as
  absent until a stage uses them. Each stage that starts writing one raises the level, and a build that reads a
  nonzero reserved field it cannot use counts it as unread (as unknown flag bits are today, T-1826).
- **The trailing blocks keep their names**; under TWT7 the ones that name a node or edge take the new meanings
  below. The decoder picks meanings by magic. No old build reads TWT7, so nothing misreads a block.
- Chainages in mm, like every other position, so derived geometry stays bit-identical on load (save.rs top comment).

### Blocks and flags that change meaning

| Today | Holds | Under TWT7 |
|---|---|---|
| node flags: station, built, off, upgrades 4-7, named 8, `one` 9-10; platform length, name | station on the node | station record; `one` becomes a single platform on that side |
| edge flags: tracks, oneway, join a/b, crossover, kind, HSR, sep | per edge | bundle flags, track count and ways; joins become node connections; crossover edges become crossover records (stage 8) |
| `LNK1` | node pairs | station pairs |
| `LOR1` | line, then node indexes | line, then station indexes |
| `LFR3` | ports by node index, berth from the node | ports by station index, berth from the station's first platform middle |
| `LFZ1`, `LSN1` | by node index | by station index |
| `LBD1` | days of built stations, edges, lines | stations, bundles, lines |
| `LPS1` | speed limit per edge | per bundle with a chainage range (a limit between two stations no longer has its own piece) |
| `LRA1` | ramp per edge | per bundle |
| `LOS1` | per line, stop place and side | stop place kept; the side is read as the run's way (part 4 may fold it into the stop point byte) |
| `LTW1` | quad lines and hidden twins | read until stage 6, then folded on load and no longer written |
| `BLD1` | outlines | unchanged (membership derived by geometry) |

All the line, bus and ferry blocks (`LUI1`, `LRT1`, `LGR1`, `LPH1`, `LPD1`, `LWE1`, `LBS2` and the rest) are by
line index and do not change. To confirm at build time: that `LBS2`/`LBH1` bus stops never name a station node.

## Save level, stage by stage

| Stage | Format | Level |
|---|---|---|
| 1 | TWT6. The decoder learns that any `TWT7` or later magic is a newer save (below) | 34 |
| 2 | TWT6 plus a trailing `LBN1` block: bundles over today's edges (count; per bundle its edges in slot order) | 35 |
| 3 | TWT7 body level 36; stations still on nodes, as a station section that names its node | 36 |
| 4 | TWT7 body level 37; stations and platforms as above (the body's one change of layout) | 37 |
| 6-11 | TWT7; one raise per stage that writes a reserved field, a new trailing block or a new line field (run track choice, presets, crossovers, spreads, tapers, exact stops) | 38 onward |

A stage that ships writes its level; every later build reads every earlier level. The stage 3 station section is
converted by stage 4's loader with the same function as TWT6 stations, so that conversion is written once.

**The body level.** The header's `level` reaches the network decoder (`decode_at`), but a build would still parse a
newer body with its own layout and fail or misread. So TWT7 carries its level right after the magic, and a decoder
meeting a body level whose layout it does not know stops there and reports a newer save, parsing nothing. Additions
that keep the layout (trailing blocks, reserved fields) still play read-only in an older build, as today. So the
layout itself should change only once, at stage 4; everything later goes in reserved fields or blocks.

## How old saves load (a nicety, per CLAUDE.md "Saves are hers")

Best effort, the "Older frames" pattern from architecture.md section 11: **convert on load, write the slot in the
new format, keep the old bytes under a hidden key `old-model-<slot>`** (as `old-frame-<slot>`), so a converter
fixed later can redo it from the original. Nothing is refused: what cannot be mapped is dropped and logged to the
console, the rest plays.

The converter lives in a load-only module (`track/legacy/`) holding today's record shapes and **frozen copies of
`twin_at` and `way_station`**. Stages 2 and 4 delete those rules from live code; only the converter keeps them,
with no other callers. Steps:

1. **Edges to bundles.** A double edge becomes a bundle of two, ways by running side. A single becomes a bundle of
   one; its way is `edge_way` at load (the inferred way included, after the `OLD_STATION_WAYS` pinning), since the
   new model infers nothing.
2. **Quad.** Pairs that frozen `twin_at` calls twins, with ends that meet the same nodes' neighbourhoods, become one
   bundle of four (slots -6, -2, +2, +6). Where the pairs are not concentric end to end (one pair split where the
   other is not), they stay **two bundles of two**, and their stations become one station with platforms on both.
3. **Joins.** Today's left/right join becomes a join to that track of the double. An unjoined end stays unjoined
   (connects to every track whose way matches, 01's rule for old saves).
4. **Stations (stage 4).** A station node becomes a station: a double gets two zero-width side platforms, one
   outside each track; a one-track station (`one`) one platform on its side; a single-track station platforms as
   today's model holds them (the loop becomes real track at stage 7). Length from the node's platform, centred at a
   through node, from the node at a dead end. Twins named alike merge into one station; pairs frozen `way_station`
   pairs merge into one station (02 "Old saves").
5. **Merging at station nodes.** Once its station has moved to platforms, a two-end node is a plain joint, and the
   converter joins its two edges with the rules `unjoin_near` already uses (wasm/edits.rs, T-1964/T-244: same built
   and off, track count, kind, way, speed limit and ramp; no crossover, no thru points; radii kept). That join
   straightens a joint up to `JOINT_KINK_M` off the straight, which changes lengths slightly, so the converter keeps
   the merge **only where the joined bundle's compiled length equals the two edges' to the bit**; anywhere else
   (a station mid-curve, a kink) the node stays as a plain joint and the platform runs across it (the rule in "For
   the lead" below). The earlier build day wins, as `joining_pieces_keeps_the_earliest_day` already tests.
6. **Lines.** Each path entry `edge<<1|dir` gives a track (the direction picks the track on double); merged edges
   fold into one span. Stops map node to station. A stop point byte is written only where the run could reach two
   platform tracks of the station (01's ambiguous case).
7. **Settled state** (`LRT1`, `LPC1`, `LWP1`, the `ERA1` tail). Kept, since lines keep their indexes, merges are
   length-exact (step 5), and the parity gate at stages 3 and 4 (part 8) requires the converted network to give
   today's paths, profiles and capacity exactly. A line the conversion changed (a twin folded, a path rerouted)
   loses its rates and settles on load, as an edit does. If parity cannot be held at some stage, that stage's
   converter drops the settled blocks and the save settles in full once: about 43 s natively for a London-sized
   save (T-3233), about three times that in the browser.
8. **Upgrades.** Waiting room and park and ride move to the station; terminal kept (part 5 keeps its rule);
   passing tracks become real track at stage 7 by running the preset in the loader, dropped where it is refused.

A player's game started from a shipped real network is an old save like any other and takes this converter.

**Other readers of the network bytes**, each to learn TWT7: `trackFrameKind` and `isRetiredTrackFormat`
(workers/saveFile.ts), the frame mapping in `game/frame.ts`, the shared-save check in `game/sharedSave.ts` (its
`/^TWT[4-6]$/` already refuses TWT7 as "version" in old builds, which is right; new builds widen it),
`TrackApi.save_stations` (the demand pool opens zones ahead from the bytes), `save_unread`, `convert_save_frame`,
and `pipeline/readsave.py`. A TWT4 or TWT5 save converts its frame first (to TWT6 bytes, as today), then its model.

## What plays read-only

- **In a new build**: as today, a save above the build's level, with an unknown block, or with flag bits or reserved
  fields it does not know (T-1826). A converted old save is not read-only: it plays and saves as TWT7.
- **In an old build**: today a TWT7 save would fail outright. `trackFrameKind` (workers/saveFile.ts) treats any
  magic other than TWT6/TWT5 as the equirectangular frame, the conversion returns nothing and the decoder says "not
  a trainworld track save". It cannot play read-only, since it cannot decode. **Stage 1 fix (0.5-1 h):** a magic
  `TWT7` or later is reported with the existing string "This game was saved by a newer version. Reload the page to
  update." (reused word for word), and the slot is left alone. It should ship at least one release before stage 3.
- **A stage 3 build opening a stage 4 save**: the body level stops the decoder (above), same message, slot left
  alone. Not playable read-only, since the station section's layout differs; the only such step in the plan.

## Size on the real NYC save

Real NYC (`T-007-real-nyc.save`): 109 lines, 1,756 stations, 1,647 edges (all double, every edge station to station:
1,756 - 109 = 1,647), 8,293 PIs, network 128 KB raw, 75 KB gzipped. The split by section below is **my estimate
from byte counts per field**, to be replaced by a stage 1 measurement.

| Section | Today | TWT7, converted with merging | TWT7, no merging |
|---|---|---|---|
| Nodes | ~39 KB (1,756 with names and platforms) | ~2 KB (~220 track ends) | ~16 KB (1,756, no names) |
| Edges / bundles (head) | ~5 KB | ~1 KB (~110 bundles) | ~8 KB |
| PIs | ~80 KB | up to ~94 KB (+1 PI per merged station node, at most 1,538) | ~80 KB |
| Stations | (in nodes) | ~26 KB (name, flags) | ~26 KB |
| Platforms | (in nodes) | ~7 KB (3,512, chainage deltas) | ~7 KB |
| Lines | ~6 KB | ~4 KB (about one span a line) | ~6 KB |
| **Raw** | **128 KB** | **~134 KB** | **~143 KB** |

About +5% raw with merging, +12% without; gzipped about 75-82 KB; in line with T-3137's +10 KB raw for a shared
alignment plus a group id. Spreads (~8 bytes a zone), crossovers (~3) and joins cost only where used. Stage 1
measures it: a scratch bin encoding each section alone on real NYC, the 2,647-station save and London's start (1 h).

## Undo

Undo stays record replacement (`Op`, world.rs). New kinds: **Bundle** (id, old record or none), **Node** (with its
connections, so crossovers and joins undo with it), **Platform**, **Station**; **Line** as today with spans. A
separate Crossover op only if crossovers get ids of their own (01 lists one; either works for saves). Split and
DeleteEdge become Bundle and Node ops; Batch stays.

- Undo restores the same stable ids, so the drawn network's cache and the stations' panels find their records again;
  ids are never reused while a history entry names them.
- Derived ids are not restored; the touched bundles are compiled again after an undo, as after any edit.
- Memory per step is the replaced records. Deleting one track replaces its whole bundle (PIs included): a few KB
  usually, up to ~180 KB for a 5,400 km bundle of 5,688 corners, which today's long edges already cost.
- Buildings stay outside undo, as today.

## The real networks' shipped saves

**Rebuilt by the pipeline at each stage that changes the format (3, 4, later 7), not converted** (part 9):
- A start must load settled (London 0.6 s natively against 42.8 s unsettled, T-3233); `realsave` settles and writes
  `LRT1`, while a start converted at load would settle on every new game.
- `realsave.rs` builds `EdgeData` and `StationData` directly, so it stops compiling at stage 3 anyway.
- `realshared.py` knows each piece's pair count, so it writes bundles of four with no geometry guess.

The 41 files in `data/packs/real/` are written new (data/frozen/ links the old ones, T-731).

**Work, hours of agent time**: stage 1 magic guard 0.5-1; stage 2 `LBN1` and quad grouping 2-3; stage 3 TWT7 reader,
writer, converter (edges, joins, spans), round-trip tests 6-10; stage 4 stations, platforms, merging, frozen
`way_station` 5-8; `readsave.py` 1-2; each later stage's fields 1-2.

## Her calls

- Which of your own saves must come through the conversion cleanly? The rest are best effort, old bytes kept so a
  later fix can redo them. [The 2,647-station save, North America day 383 and the real NYC save as the converter's
  test saves; everything else best effort.]
