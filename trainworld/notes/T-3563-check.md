# T-3563 check (tramcheck, one round)

Result: PASS. Tree .tw-trees/tramloops, WASM rebuilt, Vite 6155 (stopped).

- Their check checks/t3563-tram-loops.mjs: ALL PASS. tram loop 905 s cap 3 (3.98/h); tram noloop 2140 s cap 2; rail loop 935 s cap 3; rail noloop 2230 s cap 2.
- My script (scratchpad tramcheck/mixed.mjs), 12 km single line, stops every 3 km, 60 km/h, 1/2/3/4/8 trains:
  - mixed (one passing stop, one one-track stop): tram 1330 s cap 2 (2.71/h), rail 1390 s cap 2 (2.59/h).
  - one-track marks on the two end stops: same as all-loop for tram and rail alike (3.98/h cap 3 tram, 3.85/h cap 3 rail).
  - 8 trains against a cap of 2-3: throughput holds (no collapse) for tram and rail.
  - export and reimport into a fresh page: every line's held/limit/cap figures identical for all six lines.
  - no page errors.
- cargo test --release --lib tram: 23 passed incl. trams_pass_and_turn_at_a_stop_as_trains_do.
- Diff read: no_passing() now identical for rail (old extra clause only fired for family != 0); its only caller is capacity.rs build_sections (line 443), so rail lines cannot change. Real-save rail figures not re-run in browser; follows from the code.
- Not done: an old save with tram lines (no fixture with tram track found; checks/lib/play.mjs does not exist in this tree). Deadlock facing-each-other on single track: no stall seen at 8 trains (throughput constant).
