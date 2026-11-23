# T-4037 Stations placed without splitting track: design (stage g of T-4030)

Agent trackg, 2026-11-23. Design only, nothing built. Read on tw/track at 07950df5 (`.tw-trees/_track`, stages a-f,
T-4038, 4c, 4e and the levels work landed), the addtrack and stationids trees' notes, and the stage notes T-4031 to
T-4036 and T-4038. No advisor in this session, so none was called.

## For Anita

- Players see no difference: a station goes on the track without breaking it, dragging slides it along its track,
  and a line can still turn back at a middle stop.
- (a) A line's route names whole stretches from junction to junction, plus a note "turn back here" at a middle stop.
- (b) Your idea: a line's route is written on the track as the game cuts it at stations, as before the rework. A
  turn at a station is an ordinary turn at a stretch's end. The track you draw stays one piece.
- I'd pick (b). The rest of the game already reads routes that way, so most of last stage's special cases go, and
  a mistake shows as a broken line rather than a train quietly turning in the wrong place.
- (b) is about 6 hours of agent work, (a) about 5; (b) saves some of that back when old saves are converted.
- Old saves load the same either way; with (b) saves stay about today's size.
- Neither way touches the trip calculation; finding a line's route stays as fast as in the released game.
- The one question: go with (b)?

## 1. Where things stand after stages e and f

- A station off the track is an **anchor**: a node with no ports at its platforms' middle; its platforms name a
  piece's tracks at a chainage (`PlatformAt::On`). The compile cuts every track of the piece there into derived
  edges, and runs (stage b) are built on derived edges.
- **Line paths name whole pieces** (`TAG | piece << 5 | key << 1 | dir`, stage c). Routing reaches anchors with
  special cases (stage f: seeds from an anchor, a goal ranked by the time at the anchor, carry straight on after
  arriving). Every stop matcher reads `stop_nodes_for`, which puts anchors in among a path's nodes and trims a path's
  end pieces at an anchor terminus; where no stop is an anchor it takes the old code (two code paths: about 34 call
  sites, about 70 lines mentioning anchors in paths.rs, stops.rs, service.rs, perdir.rs, related.rs, lines.rs,
  validate.rs).
- Not done, and g's job whichever way is picked: a tool that places an anchor (today every station placed splits the
  piece: `station_ops` -> `Op::Split` + `Op::Station`); a split of an anchor's piece is refused (the anchor would be
  left off its piece); reshape keeping platforms by chainage; drag sliding; the app drawing an anchor (east-west, no
  track count, network.ts ~1883-1956, `platformLines.ts`); a line ending at an anchor drawn to its piece's end
  (`line_path` exports whole pieces); a turn back at an anchor between a line's ends (T-4036 departure 1: `[P, P
  back]` reads as a turn at the piece's end, so a route that turns at such a stop goes on to the next place it can
  turn, or has none: a regression against master as soon as g makes every new station an anchor); a loop closing at
  an anchor (T-4036 departure 2).

## 2. The two ways

**(a) Whole pieces plus a turn marker.** Paths stay as stage c made them. A line turning at an anchor between its
ends keeps a marker per turn (path index, anchor), in `LineData` or as a tagged entry in the path. The search emits
it, `StopNodes` places it, `build_run` cuts the run there, and every reversal reader (`stop_turns`, `path_turns`,
`turn_count`, `loop_kind`, `carry_worth`, `way_runs`, `stop_ways`, related lines, the loop tools) asks for it. The
sim works out windows for the app from stops and markers.

**(b) Paths on the compiled track (her idea).** A path entry names a **part** of a track: (piece, key, part, dir),
part k being the window between the k-th and (k+1)-th cut along the piece. Since an anchor cuts every track of its
piece, part k is the same window on every track of it, so rekeying a line to another track keeps its part. Every
station is then a node on the path, as on master, and a turn at a station is `[part k, part k back]` at a node. The
pieces the player draws, picks, prices and deletes stay whole; only line paths and runs see the cuts. Names are made
from the records (the piece, the record key, the anchors' order on the piece), never from compile ids, so undo and
saves stay exact (compile ids move when a piece's track block is replaced, as Add a track can do).

| | (a) whole pieces + turn marker | (b) paths on the compiled track |
|---|---|---|
| A turn at a middle stop | a marker every path producer must emit and every reader must ask for | an ordinary reversal at a node, as on master |
| A line ending at a station | the whole end piece, trimmed by `build_run` and `stop_nodes_ends` (built) | ends at the station's node; no trim |
| f's anchor special cases | stay, and grow with the marker | mostly go: anchors become plain nodes of the path; one code path for old and new stations |
| Placing, removing a station | changes no path (01-model's reason for whole pieces) | rewrites the paths over that piece in the same edit, as master's `Op::Split` does today |
| When a path goes stale | a dropped marker gives a plausible wrong route (a train runs on to the junction and turns there): quiet | a part that no longer connects breaks the line, and `check_line` says so: loud |
| Flip one track (T-4038) | reroutes must keep markers | rekey keeps the part (same window on every track); reroutes come from the search in parts; nothing else |
| Add a track (T-4024) | nothing new | nothing new for anchors (keys kept, the new track is cut at the same anchors so its parts line up); the addtrack tree rebases over the wider entry type (`partner_back`, `way_lines`, rekey), mechanical |
| Station ids (T-4043) | a marker naming the anchor node rides T-4043's stop-node rule | parts are counted on the piece, not named by node, so station ids never touch paths |
| Saves | a new per-line block for markers; paths one entry shorter per station passed than master's | the part packed into the per-entry slot varint (`slot + 1 + 17 * part`): byte identical where no part, which is every old save; an older tw/track build sees a slot past the piece and refuses, the usual unread rule; paths the length of master's (about 2 bytes per station a line passes) |
| Old saves | load as now | load as now (no anchors in them: every part 0) |
| Drawing lines | the sim works windows out from stops and markers | each entry is a window; the app clips a ribbon to it, a line ending or turning at a station drawn to the station |
| Trains | on runs (derived entries) already | the same |
| Undo | markers ride `Op::Line` | station edits carry an `Op::Line` per line over the piece (as master's split); names from records, so undo is exact by record replacement |
| Route search (stop click, reroute, 600 station pairs) | pieces plus anchor checks and a turn-at-anchor state; stage f measured +10% on Vienna's pairs from the anchor checks alone, with no anchor in the save | parts: the shape master searches today, where stations split pieces; the anchor checks go |
| Demand solve (this week's speed work) | never reads paths: stops and stop times only (`packService`); unchanged | unchanged |
| Load | as now | `build_run` maps entries 1:1 to derived entries, no trims; might give back some of stage b's +6% on North America, unmeasured |
| Old stations converted (4b) | merge `[P1, P2]` into `[P]`, add a marker wherever an old line turned at the merged station, trims at ends | `(P1, part j)` and `(P2, part j)` become `(P, part j)` and `(P, part c + j)`; turns at merged stations kept as they are, nothing added |
| Cost now | about 4.5-5 h | about 5.5-6.5 h |

## 3. Recommendation: (b)

1. It is the model the rest of the game was written for. Every path reader on master treats a station as a node; (b)
   keeps that true, so f's second code path (anchors or not) closes, and the anchor branches that only the twin
   tests run go away. (a) adds a third thing every reader must know.
2. Its failure is loud. A stale part breaks the path's chain and `check_line` flags it; a lost marker in (a) gives a
   route that runs and looks right.
3. The edits that rewrite paths in (b) are where master already rewrites them: a station placed (master: split), a
   station removed (master: join), a split or join of a piece. 01-model's reason for whole pieces (a platform edit
   changes no path) mattered for redrawing; 5a keeps it by keying ribbons on a line's tracks with parts merged (below).
4. 4b and 5b get simpler: the converter is a renumbering, and a ribbon's end or turn is a part's end.
5. Departure to record (DECISIONS at landing): 01-model's "a span covers a whole bundle track from node to node" no
   longer holds for line paths; spans cover a part. 01-model "Line" and 08's "Line ribbons" key are rewritten then.

Call I'd make and tell her after: a drag slides a station along its track between its neighbours (the next station
on that piece, a junction, the piece's end); it does not slide past another station. Sliding within those bounds
changes no part number, so no path changes; past one would reorder the stops, which is a stop edit, not a drag.

## 4. Build steps for (b)

Real time, checked against stages a-f: planned at 4-6 h each, they took about 1.1-1.5 h each from start to landing
on 11-22 (a 02:12-03:18, b 02:12-03:41, c to 05:10, d to 06:34, e to 08:01, f to 09:37; T-4038, planned 2-3 h, about
50 minutes). Each step below is about one of those stages in size. Each lands on tw/track alone, gates A and B byte
identical where the step says so.

| | Step | What | Check | Hours |
|---|---|---|---|---|
| g1 | Paths name parts | A `PathEntry` newtype over u64 (piece, key, part, dir; tag kept), so the compiler finds every site (63 path signatures in 15 files, 85 `path_edge` calls); `path_part`; a part index on each derived `Part`; entry <-> derived entry both ways in O(1); `build_run` maps 1:1; `line_path` exports a window per entry (the app still gets `edge << 1 \| dir`); save packs the part into the slot varint; `pipeline/readsave.py` reads it. Paths still whole pieces in practice (no step makes parts yet besides e/f's twin tests, which build them by routing) | A, B byte identical three ways; re-saved bytes equal; cargo round trip of a path over parts; e/f twins still pass | 1.5 |
| g2 | Routing and matching on parts | The compile gives a cut node its moves: each part on into the next on the same track, and a turn to the partner's part (01's stop-node turn, at the split station's 600 s search cost); the search's states by derived slot; f's anchor seeds, goal and carry-on go; `stop_nodes_for` becomes the path's nodes (kept as a name, then inlined where it reads well); `build_run`'s and `stop_nodes_ends`' trims go; related lines' `PART_BASE` goes; `d_turns_back` true at an anchor where a run reverses; a loop closing at an anchor | A, B identical; e/f twins pass; new twins: a line turning back at a middle anchor stop equal to the split station's (run figures, capacity, turn place), a loop closing at an anchor; Vienna 600 pairs and stop click timed base / branch in one run | 1.5-2 |
| g3 | Stations placed without a split | `station_ops` places an anchor node, the station and its platforms `On` (piece, key, middle) in one batch, plus an `Op::Line` per line over the piece with its parts renumbered (single, double, one-track, quad's drawn pair, free station); delete merges; a split or join of a piece holding anchors moves their platforms and parts; reshape keeps platforms by their old middle; no junction inside a platform; a station at a track end or a junction stays on that node; the slide op (bounded as above). Issue text of its own for a platform off its track (T-4035 departure 5) | play: a station on built double makes no track node, figures equal to master's split for the same click, undo; a line turned at it; cargo: place, slide, split beside, delete, undo to the bit; sim edit within R1's 2.6-3.0 ms | 1.5 |
| g4 | App | Anchors drawn with their piece's heading and track count (`station_tracks` or a `nodes_info` field), `platformLines.ts` the same; line ribbons clipped to their windows; dragging an anchor or a platform end slides it (`nodeDrag.ts`, `dragHandles.ts`) | screenshots: a station on a curve and on 4 tracks, a line ending at one, a line turning at one; a drag; worker no slower than 1.2x | 1-1.5 |

About 5.5-6.5 h. g1 then g2 in order (g2 needs the type); g3 after g2; g4 can run beside g3 on a fixture made with raw
edits, its play check after g3 lands. Smoke after g1 (saves) and g3 (edits).

Not in g: Add a track. If it lands before g1, g1's audit covers its sites; if after, its rebase meets the new type in
`partner_back`, `way_lines` and rekeying (about 20 minutes either way).

## 5. What changes in later stages because of (b)

- **4b (T-3961, old station nodes merged into anchors)**: the converter renumbers path entries into parts, exactly,
  turns at old stations included; g3 lifts the two things it waited on (a station or a branch on an anchor's piece).
  The 1 mm length tolerance stands. One-way legs as anchors (T-4036 departure 4) stay 4b's. Once old saves convert,
  the stations-by-axis code (`station_platforms`, `along`, `axis_slot` for station nodes, T-4038's `StationWays`
  pre-check) can go: a follow-up.
- **4d (T-3963, station tool templates)**: builds on g3's placement op; templates choose the keys of the `On`
  platforms. More than two platforms per station needs platform ids apart from nodes (T-4035 departure 1), as before.
  Nothing about paths. Still after g (the same station tool files).
- **4f (T-3965, sandbox starts with station records)**: the starts' lines get their paths by routing their stops,
  not written by hand, so parts come out right; or through 4b's converter.
- **5a (T-3966, drawn network per bundle)**: a station placed rewrites the paths over its piece, so ribbons must be
  keyed by the line's tracks with parts merged (one helper) plus its end windows, never by raw entries; then 08's
  "a station placement touches no ribbon" holds. Its goal is unchanged.
- **5b (T-3967, line ribbons per track)**: lays lines out from those merged tracks; a line ending or turning at a
  station is a window's end, no special case.
- **T-4038 (flip)**: nothing more. Tool-placed stations become anchors naming their tracks, so the flip's station
  refusal no longer meets them; it still meets stations on nodes (old saves, track ends) until 4b.
- **T-4043 (station ids)**: nothing; its plan (stops stay stop nodes) holds.

## 6. Follow-ups to file (no ids this round)

- The stations-by-axis code goes once 4b converts old saves (above).
- T-4036 departure 5: a deleted piece's own reroute left lines broken in the reroute twin; cause not looked into.
- 01-model "Line" and 08-performance "Line ribbons" rewritten at g's landing, with the DECISIONS line.
- `readsave.py` users and T-4026 (shonei-server's save check) learn the part in the slot varint.
- Strings g3 and g4 may add: the platform-off-its-track issue, any drag refusal; listed when built.
