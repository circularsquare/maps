# T-3565 / T-3595 check, round 2 (owcheck, 2026-11-17)

Tree `.tw-trees/onewaystops2` (uncommitted, branch tw/onewaystops2) against master (main tree Vite on 6195, its wasm
built 17:09 after the last sim commit). Tree wasm rebuilt here; tree Vite on 6196 (6194 belonged to another agent and
was never used). Scripts and raw output: scratchpad `owcheck\` (sc.mjs scenarios, saves.mjs, cara.mjs, cmp.py,
out\*.json, *.log). Screenshots `artifacts/trainworld/owc_*.png`. Cut short at the orchestrator's word (~2 h in).

**Verdict: PASS.** No regression against master found: everything tried on ordinary lines matches master exactly,
stops, path lengths, round trips and riders. Every difference is one the goals ask for.

## Tests

- cargo test --release 618 passed; --features demand --lib 809 passed.
- Tree's checks/t3595-extend-to-station.mjs: mid and end PASS; its third part (San Martin save) hit a page-load
  timeout in the harness (Vite under load), covered by the save load below instead.
- The earlier FAIL (onewaycheck, scen.mjs multi STEP=1: three leg stations per leg clicked in natural order): now a
  clean line, 10 stops, no turn back, out/back 4047/4047 m.

## Regression hunt, tree vs master (identical unless said)

- Ordinary lines (`two`): double, single and tram lines clicked out of order (D1 D3 D2 D5 D4 sorted to D1..D5);
  dropStop, undo, redo; hand order D1 D3 D2 D4 D5 (turns back at D3 and D2, kept); sortStops; setStops; stopPassed;
  insertStop; single track end dropped and re-added; closed circle line, drop a stop and undo. All identical, riders
  per stop identical (Dbl 36238/day, Sgl 25854, Ring 33385, Thru 19649).
- Extending (`ext`): line ending at a station at the track end, at a stop with 300 m loose end past it (station at the
  far end, or mid-way), single track, from the first stop's end; as blueprint, undo/redo of the station and track,
  construct, then the new station added. All identical and running on both.
- Deleting constructed track beyond a line's last stop: identical.
- Turn back at a stop that is a track end (`spur`): hand orders A T B, B T A B, a t b (single), b a t: identical.
- Old saves: T-007-real-nyc 109/109 lines identical, riders identical (23,802,766/day). t3541-caracas: Lines 1, 1a, 4,
  5 identical; Line 2 and Line 3 reordered on load (straighten_legs): Line 2 no longer turns back at Avenida
  Arismendi, round trip 4082 -> 1204 s, out/back 5279/14912 -> 3017/3468 m, riders 7,684 -> 44,236; Line 3 2075 ->
  729 s. t3595-caracas-sanmartin-broken: Line 2 broken on master, running on the tree (1156 s); Line 3 as above.
  Lines 4 and 5 lose about 40% of riders in both Caracas saves (2692 -> 1510, 3078 -> 1773) with their own stats
  unchanged: riders moving to the now much faster Lines 2 and 3; total riders up (421k -> 458k).

## The goals

- One-way stations 88-223 m apart, both listed (`pair`): master loops (out/back about 5.4/17.4 km, turns back at W);
  tree runs clean (3.6/3.6 km, rt 1049 s), both shown one-way. Only N listed: master serves it both ways via S, tree
  serves it once (rt 983 s). Riders board at N and S on the tree (p100: N 468, S 115/day).
- Pairs at 45 m and 34 m (`p40`, legs checked one way at the stations): click orders W w N S e E, W E N S, N S W E
  and setStops all give the same clean line on the tree (rt 1049 / 792 s, no turn back); master loops in every
  order (rt 2303-2573 s). Clicking S then N toggles the stop off again: same on master (T-3348's click rule).
- Hand order with a leg station listed against its way (W w S N e E): tree keeps it, served once, clean; master ran
  11.6 km with two end turns. drop, undo, undo back to the game's order: fine.
- A one-way station clicked as a line's end (W w N; N e E): the way back loops round the pair (1.8 / 5.8 km); same on
  master.
- T-3595 (`hs`, a line turning back at a headshunt, track drawn on from there): station at the new far end: master
  BROKEN once built, tree running (rt 1114 s, turns at that station without stopping); station mid-way and bare
  track: running on both; undo/redo of the draw fine; adding the new station as a stop then undo fine.

## Worth knowing (not regressions)

- New and allowed now, refused on master: dropping the middle stop of a single-track hand line b a t (a at the track
  end) keeps the path, so run 0 still goes out to a and turns there without stopping (2896 m) while run 1 goes
  straight back (999 m); round trip 762 s for what is a 1 km line. Odd for a player who removed the stop; follows from
  the T-3595 rule plus dropStop keeping the path.
- Unchanged and still poor: a closed circle over a one-way pair (W N E S W, setStops) loops and turns back (rt 4334 s,
  7.6/23.6 km), and the same circle against the ways becomes a 13-stop line (rt 12765 s); identical on master, as
  loops are outside the change (legs() skips them).
- A player whose 51-300 m pair served one listed station both ways now gets it served once (Anita's 50 m rule); a pair
  linked by a passage further than 50 m apart no longer counts as one stop either.

## Not covered

Live v0.3.5 (play.mjs cannot drive its Build tab: "can't press Build"), buses (code reading only: legs() and
straighten_legs skip them), tram riders (setTrains could not reach the tram panel on either build), half dots and
the departures board, extensions on her Caracas save past the end-case (mid/single/bare variants there).
