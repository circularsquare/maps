# T-4204 options: has this been tried, Gemini's ideas checked, what to build (agent passhist, 2026-11-24)

Read-only research for Anita's 11-24 ask ("we've already identified and tried to address this issue in the past ... have a gemini provide input"). No code changed. Problem and measurements: notes/T-4204.md. Gemini brief and answer: `artifacts/trainworld/gemini/T-4204-brief.md`, `T-4204-answer.md` (Gemini 4 Neon medium; the first run returned nothing because Gemini tried a shell command the sandbox denies, so the brief now says not to; one real run).

## For Anita, plainly

Yes, we have worked on this before, a lot. The 5-11 second figures you get for an edited line today are that work: since late October an edit first re-solves only the cities it touches, then re-solves the whole connected world to make every figure final. What was never done is the last step: letting most edits finish at that first quick result instead of waiting for the whole world. You were asked on 10-29 and chose to keep the whole re-solve for now (T-1734, T-2188); a speed survey on 11-09 (T-2830) also left it for later. On a world as big and connected as your Asia save, that whole re-solve is 165-300 s, and it is now the wait.

Also found: on a default desktop Chrome, the game gives all solving workers together 17.5% of 8 GB (Chrome never reports more than 8 GB). An Asia-size world needs about 1.4-2 GB a worker, so unless "Use more cores" is on in Settings, your Asia save calculates trips on **one** worker, and the whole re-solve is longer than the 2-worker figures measured in T-4204.

## 1. History: what was tried, and what still holds

| Task, date | What | Outcome | Holds today? |
|---|---|---|---|
| T-1650, 10-28 | Design: cut a big network into parts for demand | Geographic cuts rejected (the market's long trips cross every cut: 26-66% on Japan). Chosen: after an edit re-solve only the touched cities on frozen loads, show that, then the whole solve replaces it | Yes: the plan everything below built |
| T-1733, 10-28 | The local stage: touched cities solved against the rest's loads per round (`trail`) | Landed. Japan line shown 118 s after the edit instead of ~1,000; China 9 s. First version crowded on the rest's final loads and moved lines 6-16%; the per-round trail fixed it (worst 0.47% on Japan) | Yes: the 5-11 s figures in T-4204 |
| T-1609, 10-28 | A zone changing level solved alone on frozen loads | Landed (China 53 s to 4.7 s) | Yes |
| T-1735, 10-29 | The market's pairs near a long edit re-solved with the local stage (a window) | Landed, but a new express still shows about a quarter of its longer-trip riders until the whole solve (T-1760, open P3) | Yes; T-1760 open |
| T-1734, 10-29 / 11-04 | A running whole solve is never cancelled; edits wait for it and go locally on top | Landed 11-04. You, 10-29: "keep T-1733's behaviour for now", i.e. a whole solve after every edit | Replaced in part by T-3925 |
| T-2188, 10-29 (open P3) | A city line's edit ends at its local stage, no whole solve | Not built: "drift grows edit by edit, so it needs a rule for when a whole solve still runs". Tokyo: local 83 riders off the whole of 31.7M | Open; this is the missing piece |
| T-1850 / T-1862, 10-30 | A cluster only moved by float noise keeps its result | Landed | Yes; doesn't help one 140-zone cluster |
| T-2605, 11-06 | Zone in view gets a one-worker aside during a whole solve | Landed (SLC 37-118 s to 7 s) | Yes, only while an edit is held |
| T-2830, 11-09 | Survey of every wait. B4 "treat a small local result as final" listed, "not recommended unless A1 falls short"; per-pair caches, period splitting, WASM threads ruled out; two crowding rounds measured (lines up to 14%) | Survey only; A1 became T-3616/T-3925 | Rulings hold |
| T-3616, 11-17 | An edit during a quick local stage goes at its local result | Landed | Yes |
| T-3512, 11-17 | Skip the local stage where the touched cities are 35%+ of the work | Landed | Yes |
| T-2624, 11-21 | Edited lines move period by period | Landed (first move 10 s vs 40 on NA) | Yes |
| T-3925, 11-21 | An edit cuts a whole solve under 40% done | Landed (NA two edits 217-226 s vs 418-440); the restarted solve has no local stage | Yes; T-4204 point 2 |
| T-3513 / T-3938, 11-21 (open P3) | Warm-started crowding | Unmeasured; figures would depend on edit history | Open |
| T-1205, 10-26; T-3868, 11-20; T-2757 (open) | Idle workers restarted to give memory back, then the network solved again | Landed as designed; questions wait for the re-solve (T-2757 open, "needs a design call") | Yes; T-4204 point 4 |
| T-4093, 11-23 | Two crowding rounds refused: "final unless the swing numbers get a lot lower" | Decided | Yes |
| T-4057, 11-24 | Solve order and search-table reuse | Landed (Tokyo -4 to -17%) | Yes |
| T-3373 B (open, paused 11-22) | Split zones between workers (less memory, so more workers) | Stages 1-5 built behind `?split=1`, 6 in a tree | Paused |
| T-168 (P4), T-1760 (P3) | A viewed city without a whole re-solve; new express's riders | Someday / open | Open |

**Verdict.** The problem was found and worked on from 10-28 to 11-21. What came of it is the quick local figures, finals that keep landing while you build, and restarts. Not built: finishing an edit without the whole re-solve (T-2188, held back by your 10-29 call and by unmeasured drift), a quick stage for lines into new cities, and not re-solving after a memory restart (T-2757). The cost of the whole re-solve grows with how much of the map is joined up, and your Asia save is the first one joined across about 140 cities.

## 2. Gemini's ideas, checked against the code

Gemini ranked: (1) no re-solve after a memory restart, (2) a selective whole solve with a drift reset, (3) a local stage for lines into new zones, (4) a local stage after a T-3925 restart, (5) periods moved between workers on the fly, (6) yielding inside the whole solve's set-up, (7) warm start, (8) T-3373 B.

- **Memory restart (its 1): right.** `solveAgain` sets `freshParts` (demandClient.ts ~3231), so every worker, including ones not restarted, recomposes its market (demand.worker.ts 1931-1939) and the unchanged network is solved again. Its "lazy" version (re-solve only when a question comes) would make the first click wait minutes (T-2757's problem). Its other version is the good one: restart only when memory is actually short against the budget.
- **Selective whole solve (its 2): sound, and it is T-2188 / T-2830 B4 made general.** Checked: the whole solve's `setUp` puts every population of the cluster on the new network (`setUpPops`, 2235) and solves them all; the local stage already solves a set against frozen per-round loads (`trail`), and local-on-local already happens in the 8 s gap. Gaps in Gemini's version: (a) a whole solve cut by an edit leaves nothing for a local stage (below), so the "idle drift reset" needs a rule for edits made during it; (b) the market stays frozen outside T-1735's window, so lines with a long run (intercity) still need the whole solve (T-1760); (c) drift over many edits is unmeasured (T-1764's 0.16% after 17 lines was the whole solve's own carried state, not local drift). Its 12-20 h is on the high side; a first version behind a flag plus the drift measurement is about 8-12 h.
- **New zones (its 3): right about the cause, unsure about the cost.** `assignParts` frees the old market part when the market's composition changes (1885-1889), so `editOf` finds no intact market and skips the local stage. Keeping the old part until the local stage ends is plausible, but Gemini's "12-25 MB" for a market part is not checked; on a world this size a part with its pair tables could be hundreds of MB a worker for that moment. 4-8 h.
- **Local stage after a restart (its 4): the mechanism doesn't work.** Gemini proposes setting populations up lazily, just before their first pass. But every population of the cluster crowds every round of period 0 together, so all are set up before the first pass anyway (`setUp` at 2906, ahead of the period-by-period loop). The real fix needs the last final's per-round loads and each figure kind kept apart and spliced (T-1734's "new bookkeeping for each figure kind"): 6-10 h, risky. A cheap alternative is a tuning call: on clusters whose whole solve takes over about 2 minutes, hold the edit instead of restarting it. The edited line then shows at the end of the running solve plus its local stage (on average about half a solve sooner), and the final comes later. 0.5-1 h.
- **Periods moved between workers (its 5): numbers right, small gain.** `PERIOD_COST` [126,125,120,120,67] gives 2 workers 313 and 245, so about 11% at best, only on 2 workers, and a moved period loses its kept state on the other worker. Not worth it now.
- **Yield inside set-up (its 6): little gain.** Point 5's long waits are level changes held on purpose until the solve ends (`reopenLater`, 717); new zones already open between units (3-6 s seen). Not worth it.
- **Warm start (its 7), T-3373 B (its 8):** they match the history. Gemini also confirms the 8 GB Chrome cap with `SOLVING_SHARE` 0.175 (demandClient.ts 609-612: "on a reported 8 GB, New York 2, all of China 1").

## 3. Options (hours of agent time; waits on her Asia save from T-4204's 2-worker runs)

Today: an edited line in 5-11 s, "calculating trips" until 165-300 s; an edit during a whole solve: its line at 160-435 s; a line into new cities: nothing until 310 s; now and then a 4-5 minute re-solve with no edit.

**A. Stop the needless re-solve, and check the worker count (1-2 h).** Restart an idle solving worker for memory only when the pool is actually short against its budget, so a final stays final. Check whether "Use more cores" is on: if not, Asia runs on one solving worker, and turning it on (about 3 on her machine, roughly 5-6 GB in all) should cut the whole re-solve by about half (estimate, not measured). Effect: the 4-5 minute re-solve with no edit goes (about once in 40 minutes of play); no figure changes. The cost: workers keep up to a few hundred MB more after a big solve.

**B. Most edits finish at their quick result (T-2188, widened; 8-12 h, then her call on the numbers).** After the local stage, also re-solve the cities whose trains the change made busier or emptier past a threshold, then call that final. The whole re-solve runs only for edits with a long changed run (the intercity case, T-1735's 20 minutes) or into new cities, and as a refresh once you have stopped editing for a few minutes. Effect: a city edit's "calculating trips" ends in about 5-15 s instead of 165-300 s (about 95% off). Edits made one after another each finish that fast. Intercity edits wait as today. The lie: until a refresh, far cities don't react to the change's crowding, and the market keeps its old routes. Per edit that was within 0.5% on Japan's lines and 1.6% on the line beside a China edit (T-1733). How it adds up over many edits is the first thing to measure: 15 edits in a row on the NYC save and on Asia, against a fresh solve, before it ships. The weak spot: an edit made during the refresh restarts it with no quick result (the same as point 2 today), unless the refresh is held off until you are clearly idle.

**C. Quick figures for a line into new cities (4-8 h).** Keep the old market part through the local stage and solve the new cities and the touched ones against it. Effect: E2's line shows its city riders at about 15-40 s instead of 310 s. Longer-trip riders still come with the whole solve. Needs a memory check first (two market parts per worker for a moment).

**D. A faster whole re-solve (bigger, already filed).** Turning on "Use more cores" is the free part of this (in A). After that: T-3938 warm start (2-3 h prototype; figures then depend on edit history, your call once measured) and T-3373 B (24-40 h left, paused; more workers in the same memory). Both help loads too. Neither gets an edit under a minute on Asia by itself.

My order: A now. Then B, if you're OK with figures shifting a little until the refresh (it reverses your 10-29 call to keep the whole re-solve after every edit). Then C. D stays where it is on the board.

Advisor: not called. Gemini: one run (plus one empty run that tried a shell command).
