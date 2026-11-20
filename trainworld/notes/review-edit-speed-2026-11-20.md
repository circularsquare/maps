# Review of the Edit speed line, 2026-11-20 (agent speedreview)

Every open task on `{L:edit-speed}`, plus the contributors' speed PRs (T-3659), checked against today's code, the
archive and later tasks. No game runs or sweeps (the core ledger was full); figures are from earlier notes. Nothing
in TASKS.md was changed: the orchestrator applies the verdicts.

Counts: close 2, fold 5, rewrite 3, keep 1, low value 3.

## The tasks

### T-3550 London train-count edits still take 1-4 s
**Rewrite** (and take in T-1794 and T-1936).
- Still true: nothing since T-3388 (5eabd6d0, 11-17) touched the cost it names, `Service::update_scoped` working out
  100-190 line profiles a settle round at about 1 ms each. Since then weekend service (T-3521) settles a second service,
  five service slots (T-3542) and per-train timing (T-3346 stage 2, e40d878e) all landed in the same settle; T-3542's
  notes put the extra at a few ms an edit on average, but the slow London edits run several paced rounds, so these
  figures are probably worse now, not better.
- The lever is inside each profile: `profile::run_with` (sim/src/track/profile.rs) finds each spread hold's crawl
  speed by bisection, up to 50 rounds of `hop_time` over every hop the hold covers. That is T-1936's cost (its text
  predates T-2018, which turned holds into one lower speed over the stretch) and T-1794's "cheaper crawl bisection".
- New head line: `T-3550 [P3] {L:edit-speed} Train-count and headway edits on shared track are slow`
  - Context: London train-count edits inside its 130-line merge group take 0.5-1.3 s natively (up to 3.3 s on a
    3-train line), 1.4-3.7 s in the browser (T-3388); New York's busiest shared line took 150-400 ms (T-1794, before
    T-1863). Each settle round works out 100-190 profiles at about 1 ms, mostly the crawl-speed bisection in
    `run_with`.
  - Goal: London train-count edits under 0.5 s natively; figures within the pacing's own noise (T-3388's tolerance).
  - Other: notes/T-3388.md, notes/T-1779.md "T-1787", notes/T-1936.md.
- One number first: `t3388_probe` (notes/T-3388/) on master's London save, natively on one core, about 20 minutes.
- Value: London is a shipped real-network start, so this is common; every train-count edit in central London waits.
  Size: 3-5 h (a closed-form or secant crawl speed plus reusing the hold-free `Hops`; the figures will move in the
  last bits, so it needs a DECISIONS line).

### T-2858 Edit speed on other machines and browsers
**Rewrite**, and move to `{L:solve-speed}`.
- What it asks for is T-2830's unmeasured parts, which are all trip-calculation and loading times, not edits
  (notes/T-2830.md "Not measured").
- Two of them can be measured here: Playwright's Firefox and WebKit are installed (`ms-playwright/firefox-1532`,
  `webkit-2311`), and a laptop is a 2-worker pool plus CPU throttling. Real Safari, a phone and a real connection can't.
- New head line: `T-2858 [P4] {L:solve-speed} Trip calculation in Firefox and on a laptop`
  - Context: T-2830 timed trip calculation in Chrome on one desktop only. Firefox and WebKit run here through
    Playwright; a laptop is the 2-worker pool with the CPU slowed.
  - Goal: NYC and Tokyo cold load and three edits timed in each; anything over 1.5x Chrome is filed.
- Value: unknown until measured; no player has reported Firefox as slower (T-2867 was memory). Size: 2-3 h.

### T-2767 Station on built track takes 2-4 s
**Fold into T-3444** (and fix the head line if it stays open: 0.2-0.9 s, not 2-4 s).
- Still true: `map/lineBands.ts` is unchanged since T-2768 (acc208b8, 11-17). Placement already redoes only the 8-15
  lines over the split edge; the rest is `besideSamples` (60-350 ms in dense midtown) and the per-line band work.
  T-2768 found no small fix: it needs the band records to name the pieces they read.
- The rework changes exactly this: a station on track and the way parallel tracks are drawn beside each other. Make
  "a station that leaves the track's shape alone redraws no neighbours" a requirement of the new drawn model.
- Since T-3803 (5be37e4e) the next edit no longer waits for this, so a player feels it less than in the task's era.
- Value: New York-sized saves, each station placed on existing track. Size alone: 3-5 h of work the rework would
  redo.

### T-1211 Page DOM makes layouts slow
**Low value** until measured again.
- Every figure is from 2026-10-24 at 4-6x CPU slowdown (phone speed); the panels have changed a lot since.
- Station names are still page elements (about 875 on NYC); T-1505 (9d3cd02d) took the forced layout out of
  their placement, which was the main reader of layouts.
- One number would settle it: Lines tab, Stations tab and saves panel open times at normal CPU on the 2,647-station
  save, about 20 minutes. Under 200 ms: close. Over: keep with the new figures.
- Value: low on a desktop. Size: 0.5 h to measure, 2-4 h to fix whatever is still slow.

### T-3844 A paid stop click's train charge goes to no region
**Rewrite**, move to `{L:money}`, and take in its cousin.
- Still true: `chargeShares` (app/src/game/regionMoney.ts) finds the region from what changed in the world. A stop
  click changes no track or station, and its `insertStop` answer names no line, so new trains go to the unassigned
  region.
- T-3803's notes record a second case with the same cause, filed nowhere: a waiting room or other station upgrade
  bought on a built station also lands in the unassigned region.
- The fix that covers both: every paid edit's answer names what it paid for (its line, its station), and
  `chargeShares` uses that region when the world shows no change.
- New head line: `T-3844 [P3] {L:money} Some paid edits charge no region`
  - Context: A stop click that buys trains and a station upgrade (a waiting room) charge the unassigned region,
    since nothing on the map changed for `chargeShares` to place them by (found by T-3803). Shows as a wrong split
    in the money panel's regions.
  - Goal: Each paid edit's answer names its line or station, and both cases charge that region.
  - Other: notes/T-3803.md
- Value: anyone who reads the regions' money; a wrong figure, not a slow one. Size: 1-1.5 h.

### T-3804 Other worker work that holds edits up
**Keep** (and take in T-1842 as its `today` part).
- Fresh (T-3801's audit, 11-19); none of the four parts changed since. T-3803 dealt with edits waiting behind an
  edit's follow-up, not these.
- The hour turn is the biggest: 0.14-1.1 s every game hour, and a game hour is 3.75 s at speed 2, so on the
  2,647-station save the worker is busy with it up to about 30% of the time.
- Value: big saves at speed; an edit lands behind the hour turn often. Size: 4-8 h, best done as stages (the hour
  turn first).

### T-2701 Rebuild only near the drawn track
**Fold into T-3444.**
- The table in notes/T-2404.md is still right about what is whole-network (`networkFromState`'s pass over every
  edge, line and station). Two things since make it matter less: T-3803 option 3 (70845536) made the full state
  about 35% cheaper, and option 1 (5be37e4e) builds it in 0.1 s pieces with previews and edits going ahead between
  them, which was this task's player-visible cost ("a ghost asked right after the rest waits for it").
- What is left is total CPU, and the drawn network is what the rework will rebuild. Make "updates only near the
  change" a design requirement there.
- Value now: moderate on North America-sized saves. Size alone: 6-10 h, mostly redone by the rework.

### T-1794 Headway edit on busy New York line
**Fold into T-3550.**
- Figures from 2026-10-29. Since then T-1863 (51457285) settles only lines sharing capacity with the edit, and
  T-3388 holds pacing still; the Paris figure has not been measured since.
- Its open option, the cheaper crawl bisection in `run_with`, is the same lever as T-3550's.

### T-1842 past_day costs 40 ms on New York
**Fold into T-3804.**
- Still true: `Boarding::past_day` (sim/src/track/boarding.rs) tallies every line's whole day on each ask, and
  nothing caches per line. On the 2,647-station save it is 70-250 ms (T-3801), the "one today answer" in T-3804.
- Since T-3801 the asks are paced (at most a third of the worker), so the "4 times a second" no longer holds.
- A per-line tally needs care: the window moves with the clock, so it means running sums per departure, not a
  cached total. That also avoids the design call T-3804 names (splitting the answer).

### T-1040 Long piece ghost takes 100-300 ms
**Close.**
- Measured before two fixes: since T-1779 a preview skips the services (`TrackWorld::trial_light`, used by
  `preview_route`), and since T-2700 its rollback no longer prices the network (from-node previews 2.5 to 1.6 ms).
  What is left is a derive of the new piece and its rollback.
- Not measured again. If it should stay open: a 400 km piece's ghost time on the real save, about 20 minutes. The
  ghost path is part of the rework anyway.

### T-1031 First ghost of a run takes 0.5-0.7 s
**Low value.**
- The source (notes/T-1001.md) says once per page, a 0.4-0.6 s gap on the first preview of a session, not on each
  run as the title says. Most likely the first WASM preview path and shaders warming up.
- Not measured since 2026-10-23; T-2404 and T-1407 changed the paths around it since.
- Value: half a second once a session. Size: 1-2 h just to find what it waits on.

### T-758 Track derive takes 4.9 s on Japan
**Close.**
- The figure is from T-737 (2026-10-19), when a load derived twice and every mask re-derived the whole network.
  T-1039 made a mask re-derive only track in its box, and T-3234 (11-11) made a load derive once with the masks
  already set.
- Only an all-Japan 709-line save at load was ever affected. Whatever is left is a loading cost, and the rework
  replaces the derive.

### T-1936 Run profiles with many holds are slow
**Fold into T-3550.**
- Its text is stale: T-2018 (26528bb6) replaced the waits at stations with one lower speed over the shared stretch,
  so the goal ("waits sit at every station") no longer describes the code.
- The cost it points at is real and bigger now: each spread hold's bisection runs up to 50 rounds of `hop_time`
  (profile.rs `run_with`). That is the lever for T-3550.

### T-763 Penn Station to Emeryville preview takes 3 s
**Low value.**
- A 5,400 km snapped route across the continent, a stress case from T-727 (2026-10-19). The trial preview has got
  cheaper since (T-1779, T-2700); not measured again.
- Few players snap a route that long, and 3 s for it is not unreasonable. Close it, or keep it as a check case for
  the rework's preview.

## Contributors' speed PRs (T-3659)

On the public repo (circularsquare/trainsworld), checked with `gh` and `git apply --check` against master.

- **PR 18 and PR 19** (CinnamonVII, demand mode choice, SIMD): closed by their author on 11-18 after our timings
  showed no gain; their profiling agreed (1-3%). Nothing left.
- **PR 20** (Furness gravity pass): already in through PR 12, credited. Nothing left.
- **PR 24** (caelum418, waiting room, car park and building layout after edits): edit speed. On their 5,843-station,
  428-line save it claims the drawing buffers 6.3 s to 0.4 s and edit reply 7.1 s to 1.3 s, identical output.
  - Overlap: T-3803 option 3 (70845536, 11-19) already keeps each complex's placement and redoes only changed
    complexes, which covers much of the same edits. PR 24 makes each placement cheaper (spatial indexes, an early stop
    on scoring), which still helps a load, a draw-anchor move and edits at a complex with facilities (from scratch,
    0.2-0.4 s on the 2,647-station save, likely seconds on theirs).
  - Does not apply to master: `facilityPlace.ts`, `buildingNet.ts` and `network.ts` conflict with T-3803 (applies
    cleanly to the 11-19 export, which was synced before T-3803 option 3). A port by hand.
  - Related to T-2767 and T-2701 only by sitting in the same follow-up; different code.
  - Worth doing: yes, measured on the 2,647-station save first (or theirs, if the shared Day 5055 save can be
    found). 2-4 h.
- **PR 23** (caelum418, intercity passenger assignment): trip calculation, not edits. Intercity part alone
  39 s to 29 s, the whole calculation 140 s to 135 s on one worker (3%), byte-identical.
  - Does not apply: intercity `setup.rs`, `tests.rs` and kernel `search.rs` changed a lot since (T-2880, T-3718,
    T-3724, T-3730). A port by hand. No overlap with the edit-speed tasks; it belongs on solve-speed.
  - Worth doing: maybe; 3% on the whole is small, but the 25% on the intercity part matters when that part is the
    wait. 2-3 h.
- Suggested rewrite of T-3659's Context: "Left on the public repo: caelum418's 24 (facility and building layout
  after edits, overlaps T-3803 option 3) and 23 (intercity assignment, 3% of a whole calculation); neither applies to
  master, both need a port by hand. CinnamonVII's 18 and 19 were closed by their author (no gain), 20 is in via 12."

## Worth doing next, ranked

1. **T-3804 with T-1842**, the hour turn first: on big saves at speed 2 the worker is busy with it up to about 30%
   of the time, and every edit sent then waits. 4-8 h in stages.
2. **T-3550 rewritten (with T-1794 and T-1936)**: London is a shipped start and its central train-count edits take
   1.4-3.7 s in the browser. Measure on master first (20 minutes), then the crawl-speed fix, 3-5 h.
3. **PR 24 (T-3659)**: port onto T-3803's facility cache and measure on the 2,647-station save. 2-4 h.
4. **T-3844 rewritten**: a wrong number in the money panel, cheap. 1-1.5 h.
5. **PR 23**, with the next solve-speed round. 2-3 h.

Everything else waits for the rework (T-2767, T-2701), a measurement (T-1211), or is not worth the time (T-1031,
T-763, T-2858 until someone reports Firefox being slow).

## For Anita

- T-2767 (station placement redraws its neighbours) and T-2701 (rebuild only near drawn track) would become
  requirements of the track rework's design (T-3444) instead of tasks of their own. Recommended: yes, since the
  rework replaces the code both would change.
- Any reply on PR 18 and 19 is done (their author closed them). Replies on 23 and 24 wait until we've measured them,
  and you see each reply before it is posted.
