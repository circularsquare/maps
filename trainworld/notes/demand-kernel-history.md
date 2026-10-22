# docs/demand-kernel.md: reasons, measurements and options

Moved out of docs/demand-kernel.md 4.4 in the doc tidy (T-940); the doc keeps the rules. Most
figures were measured on the real New York pack (21.7M people, 10.2M jobs; 74,920 cells, 202,337
once blocks are spread over their area, T-013) and a synthetic city of 153k cells, with a
hand-made 434-station, 25-line network (method in notes/T-005.md, T-019.md, T-020.md).

## Why two levels, not cell pairs

75k-150k cells make 5.6-23 billion pairs. Truncating by distance does not save it: commutes
average 15-20 km, and a cutoff that keeps them keeps billions of pairs. Even pairing only cells
within walk of a station is 90-180M pairs on the hand-made network (1.3-2.6 s native per period)
and grows with the square of the area the network covers (notes/T-005.md).

## Gravity (step 1)

- Border factor fits (T-162, T-198): New York's own k came to 0.235 and Philadelphia's 0.219 at
  T-162's curve, 0.228 and 0.207 at T-198's; Washington 0.454 at T-198's curve, 0.486 at T-162's.
  With 0.23, DC residents working in Maryland or Virginia were 3-4% of DC's commuters against
  LODES's 13.5% each (T-181). The crow-fly gravity without a factor sent 2-4x LODES across state
  lines, land borders and rivers alike, while water inside one state was about right (T-159).
  T-171 tested the alternatives on five packs' LODES (notes/T-171.md 2): per-state-pair factors
  tie it (they only separate DC from real states), a fade near the line is score-neutral, and a
  factor only between metros or a water or toll-crossing term in its place does clearly worse.
- The curve (T-198): T-006 fitted d^-1 on New York; T-162 kept that shape and moved its length
  from 29 to 40 km to give back the trip length the border factor removes. T-185 found the steep
  near end kept residents of job-rich places too close to home: in LODES they commute out to
  suburban jobs about as far down the list of nearest jobs as anyone else, and the model sent
  them 17-26% too short. d^-0.8 exp(-d / 30 km) lengthens job-centre residents' trips by 0.6-1.3
  km in all nine packs (still 9-19% short) and moves centre-to-suburb commuting from 0.73-0.87x
  LODES to 0.88-0.97x. Trips misplaced by length, county pair, nearest-job rank and centre/suburb
  class fall in eight packs (sum of the four 17-35% to 12-27%); New York is about neutral (16.2% to
  16.7%: its own-county share falls from 42.0% to 40.1%, LODES 42.8%). Per pack: notes/T-198.md.
- Timings as the doc gave them: one pack's Furness ~3 s natively; opening a city rebuilds zone
  trips in ~2 ms in WASM instead of solving them for 8 s; the joint solve about 12 minutes (T-276;
  T-297's Anderson-mixed balance since brought it to about a minute, notes/T-297.md). Furness in
  the joint solve stopped at 500 iterations, max row error 2e-4 (one Boston zone) before T-297.
  Scaled to the union instead of per pack, the joint solve barely converged.

## Access and subzones (steps 2 and 3)

- Access per cell 7-13 ms native. Walk searches with routes (T-275): 0.6-1.2 ms a station; opening
  the real New York network's 867 stations 0.5-0.6 s native and 1.0 s in WASM, once per worker;
  an edit 31 ms in WASM with or without routes.
- Walk bands (T-720): without them one subzone on a short line spanned walks of 1 to 30 minutes and
  overstated rail 7-21%. 6,797 subzones on New York's real network (4,450 without bands), taking
  its commute pass from about 1.5 to 2.8 s native; the everyday pass barely changes. Access and
  subzones rebuilt in 16-27 ms on the hand-made network and ~57 ms on the real one in WASM (before
  the bands).

## Strategies and pairs (steps 4 and 5)

- Complexes on the real New York save (T-161): the 100 m merge already makes its 855 complexes 873
  demand stations, and the 20 pairs of demand stations sharing a complex are at most 269 m apart,
  inside the 300 m walk (Penn Station's 24 platforms are 2 demand stations, Grand Central's 16 are
  3, Times Sq's 10 are 2; `complexes_real_network`).
- Skim table: 32 MB for New York's real network searched towards all its 873 stations in a peak
  period (T-581).
- `COMMUTE_CHUNK` halves the pair visits on the real save (T-874).
- Pair counts (T-090): hand-made network 4.2M pairs (4.1M before T-090; with the 4 km reach and no
  pair bound 9.3M), real network 13.8M (13.6M; 31.8M).
- `flows` 10-20 ms in WASM for one cell or thousands (T-098).
- A click on a bubble on the real New York save showed its far end 80-170 ms after the click when no
  solve was running (first click of a session about 300 ms); it took 0.5-0.9 s (1.5 s first) when
  every sample of the city was weighed and every worker sent three numbers per cell per purpose
  (T-285).

## Crowding (step 7)

- The first round leaves the segment that takes the overflow at its worst: 963% of crush on the
  real New York network against 310% before it (T-022).
- With the early stop (T-434) the peaks keep all three rounds, quiet periods stop after one or two.
- Riders left behind on the real New York save (commutes and everyday purposes, native, three
  rounds; `left_behind_real_network`, T-188): rail trips a day fall 0.1% (commutes 3,463,591 to
  3,460,098, everyday 4,831,084 to 4,825,736); riders left behind in the evening peak fall from
  40,600 to 30,400 and in the morning from 5,600 to 4,500; the LIRR Hempstead Branch loses 4.2% of
  its boardings and the Babylon Branch 2.3%; no busiest line moves more than 0.8%. Why it runs
  inside the solve: feeding the account back would need a second solve after every solve (and a
  third, to settle), and a new game would open on waits from no account; run per round it settles
  with the crowding by the same averaging, costs nothing measurable and needs no state at startup.
  Weekdays only errs towards too few weekend riders on lines full on weekdays.

## Options if it gets slow, or for accuracy

If a bigger network makes it slow (pairs grow with the square of the subzone count): add workers
first. Skipping zone pairs under 0.2-0.5 trips cuts pairs 1.1-4x but drops long trips (segment
loads 3.5-5.4% off, T-020). With time to spare, 10-minute walk bands were the accuracy lever: 2.4x
the pairs, every error about a fifth smaller (T-020, before T-720's 15-minute bands).
