# Review of the Code and docs cleanup line (2026-11-20)

Agent cleanreview, read only. Nine open tasks on `{L:cleanup}`, each checked against the files as they are today,
git history, ARCHIVE, DECISIONS and later tasks, and against the track, platform and station rework (T-3444). Hours are
real agent time. Calibration: T-3734's last stage moved 700 lines out of the clock worker in about 25 minutes, but its
four stages with their checks took about 3 hours (18:47 to 21:38), so a move is cheap and the checks around it are not;
T-2381 split 11,500 lines of Rust and landed in one commit on 11-05 (its start time is not recorded).

Verdicts: keep 1 (T-3841), rewrite 6 (T-3458, T-3460, T-2843, T-2374, T-2375, T-1183), low value 2 (T-2382, T-2383).
None is fully done by later work, but T-2375 and T-2843 are each about half done.

Who holds what right now (wait for these to land before touching the files): riverfix holds the diagram's
`sheetLines.ts`; bigload and savefail hold `demandClient.ts`, and savefail also `clock.worker.ts`, `main.ts`,
`slots.ts`, `clockClient.ts`, `saveFile.ts`; rampspeed holds `net.rs`, `wasm.rs`, `service.rs`, `save.rs` and 23 more
sim files, plus `tools.ts` and `clock.worker.ts`; islands1 holds `demand.worker.ts` and the intercity files; gapfill
is in `pipeline/`.

Size that fits one read: an agent's Read takes about 25,000 tokens, which is about 1,500 lines of this code (47-60
bytes a line). That is the yardstick used below for "a part an agent can read whole".

---

## T-3458 Fewer code comments

**Verdict: rewrite (narrow it).**

- Comments are 17% of app lines (22,000 of 128,000), 15% of sim, 14% of the pipeline: ordinary for a codebase.
- A seeded sample of 80 comment blocks from clock.worker.ts, net.rs, renderer.ts and world_plan.py: about 10 restate
  the code (`/// The network's frame (origin and kind).` over `fn frame`), 2 or 3 tell history, the rest give a
  unit, a reason or what a field means. Phrases like "used to", "no longer", "as before" turn up about 430 times
  across the project, and most of those are not history on reading.
- The biggest single pattern is task ids: about 7,200 app comment lines and 2,900 sim ones name a T-id.
- Rework: no overlap beyond the order below.

Value: removing what restates code or tells history cuts about 1 comment in 8, roughly 2% of what an agent reads. As a
sweep that is 6-10 hours for little. Done on each file right after it is split (as its own change, so the move stays a
pure move), it is 15-20 minutes a file and lands on the files agents read most.

New text:

    - T-3458 [P3] {L:cleanup} {after:T-2374} Trim comments in files just split
      - Context: Your 11-16 ask after the code tour. A sample of 80 comment blocks found about 1 in 8 restating the code and 1 in 30 telling history; most give a reason or a unit.
      - Goal: Each file split by T-2374 or the track model split gets one pass after its move: comments that restate the code or tell history go, no code change.

For Anita: task ids in comments (in about a third of all comment lines). Keep them as pointers to the notes (my
recommendation: they are how an agent finds the reason), or strip them?

## T-3460 Code written twice

**Verdict: rewrite (one small fix; the other two pairs have a reason to stay).**

- Car street speeds: `track/streets.rs` constants and demand's `Params` defaults (`kernel.rs` 329-332), kept equal by a
  test. The track module is always compiled (`lib.rs`), so `Params` can take its defaults from `streets.rs`: one copy,
  and the test can go. About 15 minutes.
- The diagram's carry solve: `sheetLines.ts` keeps its JavaScript solver (about 230 lines, `cornersOf` to `pcg9`).
  The game uses it only if the diagram worker cannot load the track module's WASM (which would also stop the game) or
  with `?perfOff=carryWasm`. But three standing checks run the sheet in Node without the WASM and so run this solver
  (`t2580-sheet-water`, `t3358-river-banks`, `t3364-off-sheet-water`). Removing it means teaching them to load the
  WASM in Node first: about 1 hour more for 230 lines out of the diagram worker. Keep it, and say so.
- The real-rail router: `realRoute.worker.ts` (710 lines) and `pipeline/realroute.py` (230) share the rules and
  constants (60 degree turns, other family at 3x), but the game's has grown track kinds (T-3069) and trams, while the
  pipeline's is rail only and runs offline in `realnet.py route`. Sharing one copy would mean running Node from the
  pipeline. Keep both, and say so.
- Rework: no overlap.

Value: small. About 30 minutes. Waits on nothing for the speeds; the code-tour line is a doc edit.

New text:

    - T-3460 [P3] {L:cleanup} Street speeds written once
      - Context: From the code tour. Car street speeds are written in streets.rs and in demand's Params, kept equal by a test. The other two pairs stay: the diagram's JavaScript carry solver is what three Node checks run, and the pipeline's real-rail router is a Python, rail-only copy of the game's.
      - Goal: Demand's street speed defaults read streets.rs, the test that kept them equal is gone, and code-tour's "Odd things" says why the other two pairs stay.

## T-2843 State machine B for demand health

**Verdict: rewrite (smaller than it reads, and it can take in four untasked bugs from the same review).**

- Already fixed: findings 1 and 2 (T-2848, T-2849, 11-09), 3's main case (T-2849: a real failure replaces the graphics
  notice), 4 (T-3734 stage 2, DECISIONS 2026-11-18).
- Still reachable by reading the code (not run): `demandStopped` still holds "graphics" beside "memory" and "error"
  (`loading.ts` 155, 212) and `lineFigures.ts` treats any value as stopped (53, 122), so a graphics notice lifts the
  4x clock cap and settles every pending line while demand is fine. `fail()` in demandClient.ts never clears
  `memoryResolve`, so the "freed memory" hover line can stay after a failure. The family has grown since 11-09
  (`graphicsGone`, `gameFailed`, `noWebgl2`, `noSimd`, `figuresStopped`, T-3781's restart).
- The same review's findings 5, 7, 8 and 9 have no task and are still in the code: Backspace or undo mid-run leaves
  stale held state (`takeBackSegment` in tools.ts clears neither `held` nor `earlyNode`); a node drag outlives a lost
  mouse button (`nodeDrag.ts` `onMove` never checks it); a file from the start menu that fails to open falls back to
  the oldest slot (`fileChoice` still writes `from: ""`); `old-frame-*` copies (`slots.ts` 720) are never deleted.
  savefail (T-3869) is editing slots.ts and the save code now, so re-check 8 and 9 after it lands.
- Rework: finding 5 is in the track tool, which the rework may rewrite; it is 3 lines either way.

Value: low-severity edge cases. About 1.5 hours for the demand-health part (a separate graphics signal) and 1.5 hours
for the four bugs; two pieces, since they touch different files.

New text:

    - T-2843 [P3] {L:cleanup} Leftover fixes from the 11-09 code review
      - Context: Two wrong states are still reachable: a graphics notice lifts the 4x cap and settles lines while demand runs, and a failure leaves the "freed memory" hover line. Four small bugs from the same review have no task: Backspace or undo mid-run leaves stale held state, a drag outlives a lost mouse button, a failed file from the start menu opens the oldest game, old-frame save copies are never deleted.
      - Goal: Graphics loss is its own signal apart from demand stopping, and each of the four is fixed with a scratch proof.
      - Other: notes/T-2843.md machine B and findings 3, 5, 7, 8, 9.

## T-3841 Agent report inbox takes in non-reports

**Verdict: keep; move it to the checks line ("agent tool fixes" fits it better than cleanup).**

- Still true: `tools/reports.py` `from_stop` falls back to the last assistant text ("final-text") whenever a
  transcript has no handback. Both misfiled records (r-f11676a4, the compaction summary; r-a5f08c93, "yes purge it")
  have no meta file: empty description and model.
- Fix: file a final-text only for a transcript with its `.meta.json` (a spawned agent), or drop the fallback now that
  agents have SubagentHandback (the code says interactive sessions lacked it "as late as 2026-11-16").
- Rework: no overlap.

Value: each misfiled record wakes the orchestrator for nothing. About 30-45 minutes, `tools/reports.py` only; collides
with nothing.

## T-2374 Split four large app files

**Verdict: rewrite (its waits are done, the files have grown a lot, and a fifth file belongs in it).**

- T-2371 and T-2372 landed 11-05, so "after T-2371 and T-2372" is stale. T-3734 already moved the save slot code out of
  the clock worker into `slots.ts` (723 lines).
- Sizes now, at the 11-04 survey in brackets: demandClient.ts 5,555 (4,102); clock.worker.ts 5,113 (3,985, even after
  slots.ts left); demand.worker.ts 2,870 (2,462); renderer.ts 2,507 (2,190; its `draw` is 657 lines); tools.ts 2,416
  (1,914), not in the task but the track and station tools. None of the five fits one read.
- The clock worker is the most edited code file: 235 commits since 10-21, level with style.css (only the task files,
  code-map.md and some docs see more). demandClient 131, tools 105, demand.worker 96, renderer 91.
- The rework touches three of them: the clock worker (edits, the network posting, the message handler), tools.ts (the
  track and station tools) and renderer.ts (drawing track and platforms). Splitting those first means the rework's
  agents read only the part they change, and the moves don't collide with rework edits. demandClient.ts and
  demand.worker.ts don't depend on the rework: do them whenever no tree holds them.
- The clock worker's seams, re-checked today: fares (68-220), ground fetches (water, terrain, streets, world water,
  roads; 705-1220), an edit's follow-up queue (1222-1567), the state built for the main thread (1567-1842), money and
  ledger (1842-2404), the diagram graph and posting what changed (2873-3358), the demand snapshot (3358-3760), edits
  (`applyInner` and its helpers, 3760-4253), messages (4253 to the end, `handleMessage` 450 lines). Moving all but the
  messages leaves about 1,900 lines: setup, panic recovery, saves' glue and the messages. The seams for demandClient.ts and demand.worker.ts come from the 11-04 survey (notes/T-2360.md
  stage 7) and were not re-checked; finding them again is the first step of those pieces.

Value: the highest on the line: the most edited code files, read by most app agents, and the most merge collisions
between trees. About 1 hour a piece, 11-13 hours over the pieces in the table below.

New text:

    - T-2374 [P3] {L:cleanup} Split the five biggest app files
      - Context: demandClient.ts 5,555 lines, clock.worker.ts 5,113, demand.worker.ts 2,870, renderer.ts 2,507 (`draw` 657), tools.ts 2,416; all grew 14-35% since the 11-04 survey and none fits one read. The clock worker is the most edited code file (235 commits since 10-21). The clock worker, the track tools and the renderer are what the track rework changes, so those go first.
      - Goal: Moves only, one agent per file: every new part under 1,500 lines, the clock worker under 2,000, demandClient under 3,000, the other three under 1,500, no function over 400 lines; typecheck and smoke pass.
      - Other: Seams in notes/review-cleanup-2026-11-20.md; demandClient and demand.worker from notes/T-2360.md stage 7.

## T-2375 Clean up main.ts and CSS

**Verdict: rewrite (about half done).**

- Done: T-3443 (b22b9a13, 11-16) turned on `noUnusedLocals` and `noUnusedParameters`, and T-2369's leftovers are gone
  (no `dayKindAt`, `PLATFORM_BLUE`, `tracksNear`, `isBackupOf`; research no longer names track.wav).
- Still there: the `?debug` probes (main.ts 822-947, about 125 of its 948 lines); URL parameters read in 28 files (41
  places; `?debug` in 7 files); ten unused CSS classes (style.css `.w-mode`, `.group-chips`, `.related-bullet`,
  `.blueprint-panel`, `.track-modes`, `.blueprint-corners`, `.add-corner`, `.corner-row`, `.split-key`; account.css
  `.acct-signout`; none named in app code); `startDemand`'s unused city argument.
- Rework: no overlap.

Value: small: main.ts easier to read, about 25 lines of CSS. About 1.5 hours. Waits on savefail (main.ts); the URL
module touches 28 files, so it goes last among the app pieces.

New text:

    - T-2375 [P4] {L:cleanup} Move debug probes out of main.ts
      - Context: The tsconfig part is done (T-3443). Left: about 125 lines of ?debug probes in main.ts, URL parameters read in 28 files, ten unused CSS classes.
      - Goal: The probes in their own module loaded only under ?debug, one module reads the URL parameters, the unused CSS gone.
      - Other: notes/T-2375.md, notes/T-2360.md stage 9.

## T-2382 Pipeline and pack_gate small items

**Verdict: low value; close it, and do the 15-minute deletions the next time an agent is in those files.**

Item by item (notes/T-2361.md stage 6):
1. One streaming download: still not shared (`routes.download`, `osm_water._download`, world_run's own copy;
   `netget.py` has no download). The gain would be a retry where there is none. ARCHIVE shows no pipeline download
   failure since T-1979 (11-03) added retries for requests (a grep, not an exhaustive search).
2. Frame helpers for the readers (look_pack, look_water, pack_gate's own `lonlat` and `to_xy`): still copies; they
   have not drifted. Not worth it.
3. Dead functions `osm_water.mask_on_grid` and `zone_outlines.segkey`: still there, no caller. 5 minutes.
4. pack_gate: the browser check count is still written in (`pack_gate.py` 1051, 8 or 7 by water), but since T-2298 and
   T-2399 (11-16) smoke runs the gate's browser check on Macau, so a ninth check would fail smoke rather than quietly
   park rows. The pack is still read twice (`fig_read_pack`, then `load`), the water mask reader is still its own copy,
   `registry()` is still re-read per call, `fleet()` still writes the registry's size as "packs"; T-930/T-3656 already
   cached the slow part of a re-gate. Not worth it.
5. `tools/aws/userdata/proof.sh` and its name in `launch.py`'s usage: still there. 5 minutes.

Rework: no overlap. The pipeline is busy (gapfill) anyway.

## T-2383 Sim refactors

**Verdict: low value; close it, and split a function when a task finds it in its way.**

- Item 20: `DestWalk` (kernel/assign.rs) and `LaneWalk` (everyday.rs) are one walk written twice, but a test
  (`lanes_match_the_kernel_assignment`) fails the moment they drift, and merging them risks the hot loop's speed.
- Items 21 and 23 and the two added later: `mode_choice_part` is about 830 lines (mode.rs 1,027), `pass_window` about
  710 (pass.rs 797), joint_gravity's `main` about 855, `realcorridors.build` about 590. Each file fits one read, each
  is edited 3-10 times since 10-21, and all are bit-identical hot paths (medium risk, 2-3 hours each).
- Rework: none is touched by it.

## T-1183 Remove the fixed-rate line path

**Verdict: rewrite (still true; the text undercounts the tests).**

- All still there: `LineData.trains` as an Option (net.rs 254), the `continuous` rounding (capacity.rs 1161-1202),
  `Op::Schedule` (world.rs 28, now also carrying T-3542's service rows; T-3542 is closed), `set_line` and
  `set_schedule` (wasm.rs 2241-2250, crate-only). The shipped real saves were regenerated by T-1751.
- Tests: 69 call sites in 13 test files (wasm/tests.rs 23, quad_edit 8, capacity 7, turn 6, stop_edit 5, timing 5, and
  more), each calling `set_line` or `set_schedule` directly, not through one helper, so each is a hand rewrite.
- Rework: it won't remove this (it is line service, not track), but it will rewrite the stop tests that hold about 20
  of those sites. Removing it first means the rework's `LineData` has one rate mode and its new tests start from train
  counts.

Value: medium-low: about 150 lines of engine code, and one less mode for anyone reading `LineData`. About 3-4 hours,
best on Opus (a rewritten test can quietly test less). Its first step: see whether one helper (train counts from a
rate) converts most sites. Order: after the net.rs split, before the rework's build stages.

New text:

    - T-1183 [P3] {L:cleanup} Remove the fixed-rate line path
      - Context: Every line in a game has train counts since T-1146, but the engine still keeps lines set by trains per hour (`LineData.trains` as an Option, capacity's continuous rounding, `Op::Schedule`, crate-only `set_line` and `set_schedule`), because 69 test call sites in 13 files use it.
      - Goal: The path is gone and those tests use train counts with the same assertions; best before the track rework's build stages.
      - Other: notes/T-1183.md

---

## Missed by the line (three)

1. **The track model split.** T-3444's Goal asks for "the track model split into parts an agent can read whole", and
   nothing on this line does it (T-2383's items are all demand or pipeline). `sim/src/track/net.rs` is 4,086 lines
   (about 54,000 tokens, two reads' worth) with `impl Network` running from line 930 to the end, 87 commits since
   10-21; `wasm/tests.rs` is 4,017 (the biggest Rust file), `wasm.rs` 2,357, `service.rs` 1,991, `world.rs` 1,905.
   Seams for net.rs by method group, each 180-520 lines, so each fits one read: service rows and train slots
   (930-1310); storage and ids (1318-1840); stations and one-track queries together with `set_station`,
   `check_station` and `platform_reach` (the part the rework rewrites, kept in one file so it can be replaced whole);
   setters and ground (2031-2555); derive and crossings (2555-2843); validation (2843-3244); costs (3244-3540); paths
   and routes (3540-4086). The types at the top (930 lines) stay in net.rs. Moves only, about 2 hours; wasm/tests.rs
   by area about 1 hour. Waits on rampspeed.
2. **code-map.md is too long to open.** 707 lines, 14,200 words, about 100 KB and 700 task ids, edited in 374 commits
   since 10-21: over the Read tool's cap, so an agent cannot read it whole, yet CLAUDE.md sends every agent there. One
   line per file saying what it holds, task ids dropped, would be about a third of the size. About 2-3 hours, after
   the splits (each split edits it).
3. **The `?perfOff` switches have grown back.** 49 after T-2370 (11-05), 68 now: 19 added, 18 of them for tasks now
   closed. DECISIONS 2026-11-05 keeps the measurement switches, so the candidates are the added ones that switch a
   decided change back off and that no check passes: `heldLocal` ("before T-3616"), `realZoom` ("as before T-2748"),
   `keepFigures` (T-2793), `settledHold` (T-2773), `strays` (T-2579), `allSums` (T-2596), `cacheFocus` (T-2574),
   `refineAside` (T-2605), `regional` (T-3503), `localKeep` and `localSkip` (T-3512; these two and `fareHold` are also
   missing from perfFlags.ts's list). Keep `bays` and `fareHold` (a check passes them) and the measurement ones
   (`msaa`, `grow`, `farChunks`, `overviewCache`, `singleWait`). Each removed name is a dead branch, mostly in
   demandClient.ts. About 1.5 hours; do it before the demandClient split so the split moves less.

---

## Worth doing next, ranked, in agent-sized pieces

Order for the rework: split what it touches first (net.rs, the clock worker, tools.ts, renderer.ts), the rest before
its build stages; T-1183 after the net.rs split. Everything else is independent of the rework and goes wherever trees
allow. Pieces in the same group own different files and can run side by side; pieces on the same file run one after
another.

For Anita: T-3444 says nothing is built before the design talk, and its Goal puts the track model split inside the
rework's plan. Splitting net.rs (piece 3a, moves only, no behaviour change) before the talk would let the design work
read the model whole. My recommendation is to do it first, as soon as rampspeed lands; it is her call whether that
counts as building before the talk.

| # | Piece | Owns | Waits on | Hours |
|---|---|---|---|---|
| 1 | T-3841 report inbox | `tools/reports.py` | nothing | 0.5-0.75 |
| 2 | T-3460 street speeds written once | `demand/kernel.rs` `Params` defaults, `track/streets.rs` and its test, code-tour | nothing | 0.5 |
| 3a | net.rs into `net/` by method group (missed item 1) | `sim/src/track/net.rs`, new `net/*.rs`, `track/mod.rs`, code-map's track lines | rampspeed; Anita's word on timing | 2 |
| 3b | `wasm/tests.rs` into area test files | `sim/src/track/wasm/tests.rs`, new `wasm/*_tests.rs` | rampspeed | 1 |
| 4 | renderer.ts `draw` into one method per pass | `map/renderer.ts` (new `rendererDraw.ts` if needed) | free cores for screenshots, both themes | 1.5 |
| 5 | Clock worker, piece 1: fares and ground fetches out | `workers/clock.worker.ts`, new files | savefail, rampspeed | 1 |
| 6 | Clock worker, piece 2: money and ledger out | same file | 5 | 1 |
| 7 | Clock worker, piece 3: diagram graph and posting out | same file | 6 | 1 |
| 8 | Clock worker, piece 4: demand snapshot and the main thread's state out | same file | 7 | 1 |
| 9 | Clock worker, piece 5: edits and the follow-up queue out | same file | 8 | 1 |
| 10 | tools.ts: the station tool out | `map/tools.ts`, new `stationTool.ts` | rampspeed | 1.5 |
| 11 | T-1183 fixed-rate path (Opus) | `capacity.rs`, net's `LineData`, `world.rs`, `wasm.rs`, the 13 test files | 3a, 3b | 3-4 |
| 12 | perfOff sweep (missed item 3) | `perfFlags.ts` and the call sites of the removed names (mostly demandClient.ts) | bigload, savefail | 1.5 |
| 13 | demandClient.ts split (seams found again first) | `workers/demandClient.ts`, new files | 12 | 2-3 |
| 14 | demand.worker.ts split (`run` by message kind, queries out) | `workers/demand.worker.ts`, new files | islands1 | 1.5 |
| 15 | T-2843 demand health | `game/loading.ts`, `game/lineFigures.ts`, demandClient's `fail()` | 13 | 1.5 |
| 16 | T-2843 four small bugs | `map/tools.ts`, `map/nodeDrag.ts`, `clockClient.ts` `fileChoice`, `slots.ts` | 10, savefail | 1.5 |
| 17 | T-3458 comment pass, one file after each split | the file just split | its split | 0.25-0.33 a file, about 3 in all |
| 18 | T-2375 debug probes, URL module, dead CSS | `main.ts`, new `debugProbes.ts` and `urlFlags.ts`, the 28 files reading URL parameters, `style.css`, `account.css` | savefail; last of the app pieces | 1.5 |
| 19 | code-map.md cut (missed item 2) | `docs/code-map.md` | the splits above | 2-3 |

Total: about 30 hours of agent time. Pieces 1 and 2 can start today. Piece 4 can too once the core ledger has room for
its screenshots. Pieces 3a, 3b and 10 are the ones the rework wants first.

Not worth doing: T-2382 and T-2383 (close both; T-2382's two 5-minute deletions can ride along with any pipeline task
that opens those files).
