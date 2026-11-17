# T-3388 check (londoncheck, 2026-11-17)

PASS with notes. Probes: scratchpad londoncheck/probe (lcheck, rm), outputs in artifacts/trainworld/londoncheck_*.txt.
- cargo test --release: 580 passed; --features demand --lib: 771 passed, 0 failed.
- +1/-1 medium x6 on London lines 2, 20, 72: train counts never change; after the first cycle pacing is stable
  (line 2 tph differs 0.006% from cycle 0, line 72 identical, line 20 one 3.07% edge flip between cycle 0 and 1, then stable).
  Returning to the loaded save's exact pacing does not happen (19-50 paces differ, tph up to 3.1%); master is the same (264 paces, 3.16%).
- remove line + undo x4 on 20, 72, 34: 171->170->171, identical result every repeat; 0.5-3 s each (join/leave repaces by design).
- save/reload after edits: tph and pace identical, bytes identical; edit after reload 0.47 s.
- NYC (lines 16, 0): output identical to master, every tph/pace/train.
- Edit refused for money: covered by the new unit test (passes). Not tested: browser time, a save with no settled rates, a full merge.
