# Solve-speed triage, 2026-11-23 (agent solvetriage)

## Summary

- 24 open solve-speed tasks (T-4055 is consistency, not speed) plus the memory and loading ones that touch solving. Read from the board and notes, nothing measured.
- Real on a big Tokyo world: 6. Small: 17. Stale: 1 (T-414, closed).
- Tokyo's period 0 is 72% search, and 41% of that period is one-off searches (zones with their own frame searched for one call and thrown away). Nothing on the board names that. Both T-4057 fixes are NOT filed.
- Nearly all the open tasks are NYC-shaped or someday; only T-4054, the two T-4057 fixes, T-875 and T-921 reach Tokyo's cost.
- Order, first to last: keep-last-table (medium), one graph per cluster (medium, shifts answers), T-4054 left parts (small), T-3938 warm start (medium), T-875 (medium), T-921 (small).
- Detroit-style bus saves are already handled by T-4094 (-40% CPU).

## Tokyo cost (T-4057, period 0, 5 zones, natively)

Search 71.9% (one-off searches 41.2%), commutes' mode choice 13.3%, commutes' assignment 4.3%, everyday 4.4%, drops of big tables 2.2%, market 1.4%. Today already landed: T-4054 stage 1-2 (search -18% on the real NYC save), T-4093 (settled test 3%, busiest worker -9% on NYC), T-4094 (bus stop merging), T-4117 (coarse setting: Tokyo load-to-final about 23% faster, an edit about 18%, on by default on phones), T-4103 (phone memory).

## The two T-4057 fixes: filed? No (no task mentions them). Head lines for you to give ids:

- [P1] {L:solve-speed} Keep the last one-off search table for the next caller. Same answers; about 18% of Tokyo's period natively (less in game, which releases skims every step; not yet measured there). Or more SKIM_TABLES where memory allows (memory trade: check with perf_bench).
- [P1] {L:solve-speed} Build every zone's graph in one frame. Most of Tokyo's 41%; shifts riders: zone walks between stations 1-2% off for zones a degree of latitude apart. Needs the rail-trips and lines-that-move numbers on the real NYC save and the network shape that shows it worst (CLAUDE.md performance rule), then her call.

## Each task

Real = would save something on a big Tokyo now. Small = real but little or not on Tokyo. Stale = done or measures something gone.

| Task | Verdict | Why |
|---|---|---|
| T-4054 first-arrival search, left parts | real | search is 72% of Tokyo; change nodes' settling, entry_costs, u16 shares still open (size small) |
| T-4057 profile and speedups | real | the profile is done; stays open for the two fixes above, then close |
| T-3938 warm-start crowding | real | crowding rounds are two thirds of a solve; figures depend on edit history, needs her call once measured; measure on Tokyo too |
| T-875 evening from midday choices | real | up to 13% of a solve (T-869, 10-21, NYC); shifts riders |
| T-921 tiny commute pairs in chunks | real, small on Tokyo | NYC -14%; commutes' mode choice is 13% of Tokyo's period; give to the T-4057 agent |
| T-3659 PR 23 intercity assignment | small | 3% of a whole calculation; also post PR 24's reply |
| T-2879 two rounds then three | small | helps first figure on NYC; T-4093 already skips settled rounds; Tokyo periods take minutes so it would matter there, but total time may grow |
| T-2762 bubble answer on North America | small | 300 ms a worker, a question not the solve |
| T-2757 questions after a worker restart | small | needs a design call; restarts, not the solve |
| T-1610 clicked zone waits for refine | small | interaction wait; the coarse setting changes what refines |
| T-2458 zone picked during a load solves coarse | small | one extra solve for that zone |
| T-2188 city line edit ends with local solve | small | Tokyo 0.1 s local against 8 s whole on its old figures; re-measure |
| T-1760 new express shows a quarter of its riders | small | figure from 11-04, before T-4041; re-measure first |
| T-1629 coarse zone bias with the fine set moving | small | measurement from T-1206; the coarse setting (T-4117) makes it more relevant |
| T-1730 browser check after a local refine | small | needs a new driver, 723-zone numbers old |
| T-2046 slow first Homes/Destinations switch | small | maybe a headless stall; cheap to check |
| T-2858 Firefox and a laptop | small | no player report; fold into perf_bench cases (T-4025) |
| T-3699 goals wait if intercity result never comes | small | a failure path, not speed |
| T-1240 leaner sample index | small | someday, memory |
| T-876 fold tiny far-zone station rows | small | someday |
| T-877 market air lists once per airport | small | someday |
| T-777 market rows per country | small | someday |
| T-168 add a viewed city without whole re-solve | small | someday, but it is a whole solve on a big world; worth a look after the above |
| T-414 first worker solve slower in the browser | stale, closed | T-4057 measured WASM at about twice native; nothing to fix short of threaded WASM (T-3373 C) |

Memory and loading tasks that are really about solving: T-4051 (zone open sits at 90%, race between quick figures and the whole calculation, about 100 s off the long path; real on big saves, medium), T-3373 B and T-3932 (memory for time; paused), T-3160/T-3939 (phone memory; not Tokyo desktop speed), T-3882 (bar while calculating, small), T-281, T-898 (P4 loads). T-4055 is a stale-final bug (1-2 h), not speed.

## Order on Tokyo

1. Keep the last one-off search table (medium, same answers, about 18% of Tokyo's period natively; measure in game).
2. One graph per cluster (medium; most of the 41%; shifts walks 1-2%; her call with the numbers).
3. T-4054 left parts: change nodes, entry_costs, u16 shares (small).
4. T-3938 warm-start crowding (medium; measure on Tokyo and NA; her call on edit-history dependence).
5. T-875 evening from midday (medium; up to 13%, shifts riders).
6. T-921 chunked tiny commute pairs (small; 13% of Tokyo is commutes' mode choice, NYC saw -14%).
Then T-4051 for opening zones, and re-measure T-2188 and T-1760 on Tokyo before touching either.

Watch: one and two together may overlap (a shared graph already removes much of what a kept table saves), so do 1, measure, then decide on 2. Tokyo also takes 5 GB for a period; fixes that raise table counts trade memory.
