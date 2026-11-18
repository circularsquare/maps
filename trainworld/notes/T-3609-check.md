# T-3609 check (fcheck, 2026-11-17): PASS with gaps

Tree .tw-trees/splitflat, wasm rebuilt (was older than draw.rs), own Vite 6188; master on 8800 as the control. Scratch: scratchpad/fcheck/run.mjs, run2.mjs.
- cargo test --release: 620 passed, 0 failed.
- Split (NYC sandbox, double from north, S straight south, B south-west; orders SB/BS x press W/E x open/terminus, 8 cases):
  fix: 0 flat marks in all; the two singles always on different tracks (j1/j2) with opposite ways.
  master: SB-E-open both singles j1 (same way), SB-E-term 1 mark, BS-W both j2. Bug reproduced, fixed.
- Carry-on double (no split), single branch pressed W/E: join and mark identical to master (W: j2 no mark; E: j1, 1 mark = deliberate crossing).
- Saves: T-007-real-nyc 109 lines, 0 marks; T-125-northeast 110 lines, 0 marks (marksLen 2040): identical on fix and master.
- Gaps: no train run watched (judged from join/oneWay); the "both singles two-way still marks" fixture failed (B came out one way, 0 marks on fix and master), so case 3 rests on the carry-E mark and the unit tests; BS-term station at S end did not make S two-way (same on master).
