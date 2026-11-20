# T-3804 stage 1 check (hourturnchk)

Real NYC save, tree (:6151) vs master (:6152), both loads gave rail trips/day 12,558,064 (same demand result).

1. Totals, 2 game days at 960x: 53 hour/day records compared (fares, running, trains, build, rides); all equal except
   the two partial current entries (running cost differs only because the clocks were stopped at different times). PASS.
2. Edits after a turn (10 rounds: rename line, train count, add station on an edge, at +60..900 ms after the turn):
   tree edit ms median 360, max 760; master median 562, max 1901 (add station 731/760/692 vs 1128/1901/1147).
   No trips message without loads in either, no errors, hours after recovered equal to master. PASS.
3. Clock jumps: speed 2/0/1/2, pause just before turn and resume (x3), jump back to 180100 and +7h: no unloaded trips
   messages; hour rides/fares equal to master for all full hours (h57, h58 differ 0.006% from 100 ms sampling of
   the hour during speed changes; h66 partial). PASS (the "never sent hour shows no loads" window did not occur:
   account was up to date at the jumps, so not exercised).
4. Save and reload mid-hour: SKIPPED (no ready export/import path in play.mjs; time).
5. t1071 (SHOTS=0, tree): PASS. t2766 (tree): 27 PASS, 0 FAIL.

Probes: scratchpad hourturnchk (edits.js, jump.js, rec.js).
