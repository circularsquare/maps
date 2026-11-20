# T-3646 check (fullratechk)

London real save (data/packs/real/lon.save), tree Vite 6172 vs master Vite 6173, play.mjs games, 2 cores each.
Probe scripts: scratchpad/fullratechk (init.js, f1/f2.js chain, fb.js fallback, fm.js tree vs master, sv.js reload).
Master is 18 commits ahead of the tree's base; forward figures still matched it exactly (case 5).

1. Chain (9 forward edits: counts, headways, addLine, removeLine, addStation, counts): undo all, redo all, 4 undos + new edit + 4 undos.
   Round trips, trains needed and rates equal the forward state to the bit at every step (rtMax 0, tph bits 0, tn 0),
   no NaN, no page errors. One miss: the first undo of the session (back to the loaded state) differed from S0 (rtMax 13.6 s,
   tn 2, tph 8 lines): lines 28,30,31,32,40,41,42,54 stayed at the edited rates (U0 == S1). That run waited for the demand
   final (66 s) before S0 and edit 1; the later run (fm.js, edits right after load, no wait) undid to S0 exactly. Cause not
   pinned: likely an off-history change (load cap lowering) or `found` refusing because rest_solved flipped on the first edit
   (found() needs rest_solved before == after == now). Master drifts on the same chain by up to 546 s round trip.
   Demand riders/money after undo: cash did not move on undo in the money readout (5535 after undoing a purchase) and the demand
   version did not advance during the quick chain (demand was still solving, 2 cores): not verified.
2. Fallback: unrecorded setTrainCount (cap lowering) between an edit and its undo: undo settled, no NaN, statuses running,
   figures differ from before by up to 0.1 tph (a settle). The undone line returned to its pre-edit counts (as master does).
   Weekend share edit: undo and redo exact.
3. Money refusal (Money:50925.0:5142.5): ok false, all lines bit-equal, cash unchanged.
4. 50 edits: Chrome working set 2366 -> 2445 MB (all processes, demand workers included), page heap 393 -> 629 MB, no errors,
   edits 220-1586 ms. Not compared with master.
5. Forward-only 8 edits: tree == master to the bit on every line and cash.
Reload: 3 redos, 2 undos, export, import: all lines bit-equal, 172 lines.
Skipped: Tokyo, calendar change, master memory comparison, demand figures after each undo.

## Second round (fullratechk2)

London real save, tree Vite 6155, play.mjs, 1 core (ledger full: the demand final took ~10 min). Probes in scratchpad/fullratechk2 (c1, c2, c2c, c2d, c3, sv).
The lowering is simulated after an edit with `setTrainCount ... unrecorded: true` (what capLoad.ts sends), since the real one's timing can't be forced.

1. After wait for the final (demand version 4), S0, edit setTrainCount line 4 [60,40,10], undo: every line bit-equal to S0 (rtMax 0, tph bits 0, tn 0);
   redo bit-equal to the edited state; undo again equal. Then edit on line 7, lowering lines 3 and 4 by 2 trains after it, undo: lines 3 and 4 keep the
   lowered counts, line 7 back to base, all other lines equal base except 5 and 6, which share track with 3 and 4 (lowering 3 and 4 by one more moves
   exactly lines 3,4,5,6 and the undo state differs from base on exactly 3,4,5,6). Redo equals the pre-undo state to the bit; undo again equal.
2. Edit sets line 9's trains (+3), lowering of line 9 (-2) after, undo: line 9 back to the count before the edit, lowering not made again; with a
   lowering of line 10 as well, line 10 stays lowered. Running, no NaN. Redo of that: all lines bit-equal to the state before the undo.
3. Chain of 5 forward edits (counts, headway, removeLine, addStation, counts) from the S0 after the wait: undo all, redo all, 4 undos, new edit,
   2 undos: all bit-equal (rtMax 0, tn 0, tph bits 0). The script's last comparison read past its snapshots (script bug, not game); 5 edits, not 6-8.
4. Export then import mid-chain: all 171 lines bit-equal. After the import the history is empty (undo and redo answer ok false, as before);
   a new edit then undo is exact. No page errors anywhere.
Cash: undoing a train-count edit did not move cash (-5.5% vs S0 after undo, equal to the edited state): same as the first round, not judged.
Skipped: Tokyo, real timing of the lowering arriving after an edit (simulated), 6-8 edit chain, demand figures after each undo.
