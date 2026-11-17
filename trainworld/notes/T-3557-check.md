# T-3557 check (weekendcheck, 2026-11-17): PASS

Tree weekendstep (e85e761c plus uncommitted). WASM rebuilt first.

- cargo test --release: 606 passed. cargo test --release --features demand --lib: 797 passed.
- checks/t3557-weekend-headway.mjs: PASSED (32 PASS). checks/t3521-weekends.mjs: all passed (34 PASS).
- Own probe (fresh 3-stop headway line, real schedule-window buttons, Saturday and Sunday departures from lineTrips):
  weekday 4/4/5 min baseline. Stepping High to 6, Medium to 6, Low to 9 min: Saturday and Sunday both measured 6/6/9;
  stepping one level left the others at what they ran; weekdays stayed 4/4/5 throughout.
  Past the limit (shorter x8 on High, stored 180 s, track holds 240 s): runs 4 min, Medium and Low unchanged.
  Count edit at Medium: only Medium's headway dropped (180/0/540), High and Low keep.
  Percent edit after stepped: counts and headways dropped (80%, 0 0 0). Undo restores them.
  Save and reload: weekend block [100,1,5,4,2,180,0,540] identical, Saturday and Sunday departures identical.
  Weekday gaps move 4.0 -> 4.14 min after any reload; the same happens with no weekend edits at all (control run), so it predates this change.
  Line deleted and undone (removeLine, undo): headways and departures back.
- Old saves: T-007-real-nyc.save, T-125, tokyo day60 have no weekend settings and re-export identical bytes. A save made on
  main's build with 3 hand-count lines and 3 percent lines (wk_old.save, in the scratchpad) loads on both trees with the
  same re-exported bytes (sha equal, 91391) and the same rates; the only difference is lineStats.weekend has 8 numbers, not 5.
  The real save on both trees: identical bytes and line stats (weekday rates unchanged).
- Quad twins: not exercised in the browser (the twin needs stations on both pairs; my 4-track fixture ran on one pair,
  pairs=2). Only read: halve/double carry headways x2 / /2, line_weekend divides by the split. No sim test covers quad with headways.
- My own probe's "High still same after Medium count edit" failure was my script's mistake (High had been stepped to 3 min earlier).
