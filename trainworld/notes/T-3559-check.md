# T-3559 / T-3528 capcheck (worktree capload), PASS

Method: Vite on the capload tree (port 6189) with a wasm built from the tree; clean HEAD baseline built from `git archive HEAD` (app + sim) in the scratchpad, own wasm, Vite on 6190. Fixtures made with edits on the real NYC save and exported: over.save (line 0 [30,25,20], line 1 [30,30,30]) and over2.save (lines 0-7 set to [90,90,90]). Scripts: scratchpad capcheck/run.mjs.

Results (all on the capload tree unless noted):
- over2 loaded: lines 0..3 lowered to [27x3], [40x3], [29x3], [26x3]; identical counts and cash (897.5) to the HEAD baseline. canUndo false on capload, true on baseline (baseline records the lowering).
- Undo with no edits right after load: nothing changes, canUndo and canRedo false.
- Edit on another line, undo, redo, extra redo, two undos: restores exact load state every time; redo flag right.
- Edit on a lowered line itself, undo: back to the lowered count (never the stored over-cap count); redo and extra undo fine.
- Three edits (two on lowered lines) then undo x3 / redo x2: each step equals the state after the matching edit; new edit after an undo clears redo.
- Save right after load and reload: counts stay at the cap, undo stack empty, undo is a no-op.
- Real NYC save (no over-cap lines beyond what it already has): counts for all 109 lines equal the HEAD baseline's; canUndo false (baseline true, since it records its lowering); edit then undo restores, stack empty after.
- Money: lowering costs 0 on both trees (cash unchanged by the load; over2 cash equal on both).
- Page errors: none.
Not tried: the race of a player edit landing before the async lowering has finished (an undo would then restore the stored over-cap count of that line); bus lines (the fixture had no running bus); the Rust unit test and checks/t3165-cap.mjs were not rerun.
Flake seen twice: a first import into a fresh Chrome sometimes never reached load=file within 120 s (cap2 first attempt, and cap B5 reload); a rerun passed.
