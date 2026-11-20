# T-3550 check (trainratechk)

Setup: London real save (trainworld/data/packs/real/lon.save, 171 lines), tree Vite 6157 vs master Vite 6158,
each a play.mjs game; edits through tw.client.edit, worker times from tw.client.editLog (worker.total / apply).
Tree is 12+ commits behind master (master has T-3883 ramp speed and T-3735 weekends in sim), no control build
(tree HEAD without the diff) was made: the control build script failed (robocopy), skipped for time.
Machine was busy (several other games, ~11 of 14 cores claimed), master's demand final never finished during
the run, so times are noisy. Scratch: scratchpad/trainratechk (counts.js, hw.js, undo.js, outputs c*/h*/u*.json).

## Round trips vs master after 12 train-count edits (counts.js)
Edits: Metropolitan 31 [1,1,1] then [2,2,2] (1 -> 2), 30 [20,1,1], Central 4 [60,40,10] and [120,80,20]
(very high on full track), District 10 [30,20,5], 9 [1,1,1] / [2,2,2], windrush 18 [1,1,1] / [2,3,2], 17 [12,8,3], Central 3 [40,30,12].
After each edit all 171 lines x 3 levels compared: max difference 1.1 s (typically at line 170/156), trains
needed identical everywhere, no NaN/inf, all edits ok, no page error, no missing lines.
Before the first edit the two loaded states differed by up to 90 s on 14 lines (high level), but master's demand
had not finished then (I waited for final on the tree only); gone after the first edit.

## Worker time per edit (ms, worker total / apply), 12 train-count edits
tree: total median 554, max 2282 (windrush 18 -> [2,3,2]); apply median 175, max 567
master: total median 939, max 6337; apply median 342, max 4707
Windrush edits (lines 17, 18): tree apply 0.53-0.57 s vs master 4.4-4.7 s. Tree total on windrush 1.2-2.3 s
(apply 0.55 plus state/board ~0.1-0.5 plus other), above the 0.5 s goal; London lines without windrush mostly
0.2-0.7 s total.

## Headway edits (hw.js, 8 setHeadway edits, then 8 undo, 8 redo)
Headways: 31 [600,900,1200], 31 [300]*3, 28 [150,400,900], 10 [90,200,400], 4 [60,120,600], 18 [600,800,3000],
17 [200,300,1200], 9 [7000]*3. All ok both. Round trips within 1.1 s of master on edits 0-6 (every line).
Edit 7 (line 9 to 7000 s): line 28 mid level 7277 (tree) vs 7325 (master), trains needed 13 vs 14. The same two
values then appear swapped in the undo step in the other build, so it is a history-dependent state, not a
tree-only error. Worker total median 633 (max 4358, windrush 17) tree vs 1094 (max 7380) master.

## Undo/redo and save (undo.js, fresh reload, 3 edits then 3 undo, 3 redo, export, import)
Neither build restores exactly: undo steps differ from the earlier state by up to ~21 s (line 125) and redo by up
to 105 s (line 32) in BOTH tree (20.8, 105.1) and master (21.9, 104.9), same lines. So not caused by the change.
Export then import reproduces the figures exactly (diff 0.000 s, trains the same) in both.
(The tph comparison in the script is unreliable, ignore it.)

## Not done
agent_cargo.py test (builder's pass assumed), control build, junction with many lines tried beyond Metropolitan/District/Central,
single-track loops specifically, checks/t1071 not rerun, master's demand final never reached.
