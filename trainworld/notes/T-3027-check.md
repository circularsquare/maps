# T-3027 check (rampchk, 2026-11-20): PASS on all five cases

Tree rampspeed (:6173) against the main tree on master (:6174). Scratch scripts in the rampchk scratchpad
(rt.js round trips, hash.js track-bytes hash, p2 preview cases, p3/p4 lines, p5 save export, p8/p9 old-ramp edits).

## 1 Existing saves unchanged: PASS
- T-007-real-nyc: 109 lines, every round trip equal to the 0.01 s; riders per line equal; rail trips/day 12,558,064 on
  both. Total trips/day differs (848,081,216 tree, 895,752,175 master): master carries later commits (T-2216 island
  pairs, 3ae030c8), not this change; rail trips and riders are the same. Not compared against the branch base.
- na-day-491: 175 lines, round trips equal, riders equal (sum 22,879,379), cash equal; SHA-256 prefix of the saved
  network bytes identical (0c844f41eb4f70c7, 352,556 bytes), no LRA1 block. Header level 31 vs 30 only.

## 2 New ramp limit (sandbox new game, one 1-track piece level 0 to 1, drawn by pointer)
| track | room m | ghost tag | placed | edge min speed |
|---|---|---|---|---|
| regular | 400 | 160 | yes | 160 |
| regular | 250 | 160 | yes | 160 |
| regular | 170 | 136 | yes | 136 |
| regular | 150 | 120 | refused ("at least 160 m of ramp") | |
| high speed | 400 | 319 | yes | 319 |
| high speed | 250 | 199 | yes | 199 |
| high speed | 170 | 136 | yes | 136 |
| high speed | 150 | 120 | refused | |
| tram | 110 | 80 | yes | 80 |
| tram | 95 | 76 | refused ("at least 100 m of ramp") | |
Lines over 6 km of double high-speed (A flat, B level 1 after the ramp), round trip s:
flat 711.62; 400 m ramp 724.70 (master 724.70, same); 250 m 720.57 (master, 200 m ramp no limit, 719.77: +0.8 s);
170 m 739.04 (master refuses a 170 m ramp). Limited ramps are slower in the right direction, by less than a
whole-piece estimate (accel keeps the train from reaching top speed anyway). Note: the `.pi-label` elements in the
DOM listed stale labels from earlier previews (e.g. "120 km/h" beside a 400 m regular ghost whose tag read 160);
the visible tag was right every time.

## 3 Save round trip: PASS
Exported a game with 15 ramp pieces (LRA1 present, level 31), reloaded it: ramp pieces' lengths and min speeds all
equal (399/319, 249/199, 170/136, 110/80 tram ...), the four line round trips equal to the 0.01 s. Games with only old
ramps (na) write no LRA1 and save the same bytes as master (case 1).

## 4 Editing old ramps: PASS
Old pieces made on master (high speed, level 0 to 1, 600 m and 3 x 249 m), loaded in the tree: all 320 (no limit).
- Split at a flat part of the 600 m piece (x 150 pack units) and again 700 units along: pieces 114 m flat, 417 m
  with the ramp, 70 m flat on level 1; the ramp piece stays at 320 (no limit) and keeps the ramp. A split at the
  piece's middle (inside the ramp) is refused (SplitPoint), as designed.
- Reshape (moveNode): 249 to 230 m keeps no limit (320); to 196 m takes the rule (157 = 0.8 x 196); to 174 m, 139;
  a further squeeze to about 150 m is refused (Geom:RampTooShort).

## 5 Undo/redo: PASS
Placing a 170 m high-speed ramp: blueprint $80.1M to $91.5M, piece 136 km/h, $11.46M; undo back to $80.1M and the
piece gone; redo $91.5M, 136 km/h again; cash unchanged throughout (blueprint).

## Skipped
- Not compared with the branch base for total trips/day (master drift explains it).
- No tram line round trip; tram only checked by edge min speed and the ghost.
- Construct-time reshape of built old ramps (rebuild op) not tried.
- Screenshot: artifacts/trainworld/t3027chk-prev400.png (ghost tag 160 km/h on a 400 m ordinary ramp).
