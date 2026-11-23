# Performance task sweep, 2026-11-23 (agent perfsweep)

## Summary for Anita

- Went through the 54 open performance tasks (lines memory, solve-speed, loading, edit-speed, schematic-speed, world-tooling, plus T-4045); the five being worked on today were left alone.
- Closed 6: T-060 and T-427 were already done by later work, T-1560 and T-354 are covered by the split-workers plan (T-3373 B), T-1230 by T-3804, and T-3937 is overtaken by today's solve profiling. The rest stay open. Many have figures from October that should be measured again before anyone starts on them.
- The biggest likely win not yet being worked on: T-921 (merge the tiniest commute trips into bundles), measured at 14% off the New York calculation, stations moving under 0.8%. It is in the same code as today's solve work, so it should go to that agent or wait for it to finish.
- Worth knowing before the next publish: T-1354 (landed 11-22, not live yet) makes every demand worker rebuild its 11.7-million-row corridor table whenever the open markets change, so 5-6 s per worker each time a new zone opens. That is the same wait as your T-4051 report (opening a zone on a big save), and it would make it worse. T-4051 should compare master with v0.4.5 on your North America save before the publish.
- Can start today without getting in anyone's way: T-4025 (the load-time and memory ledger), T-2589 with T-2247 (diagram speed), T-3932 (the cores slider, half built), T-2868 (memory cap for 32-bit browsers) and T-3882 (a bar while calculating trips here).
- T-4025 is the answer to the back-and-forth between memory and load time: no memory change or speed change lands without a bench run that shows both numbers. Several open tasks swap one for the other: T-3373 B, T-3160's parked patch, T-3932, T-876, T-1240.
- The perfledger tree is T-4025's paused start: one half-written script, nothing run, nothing landed.

## Closed today

| Task | Why |
|---|---|
| T-060 City packs on R2, compressed copy | Done by T-2486 (11-20): packs' .bin published as gzip members, headers in the index |
| T-427 Longer-trip answers only for zones with trips | Done by T-2747 (11-07): answers list only the far zones reached |
| T-1560 Deal city zones between solving workers | Superseded: T-3373 stage B is this design, built in stages (its plan names T-1560 as its ancestor) |
| T-354 Pack copies per worker | Folded into T-3373 B: each pack opens only on the workers holding it |
| T-1230 Clock worker first boarding hour is slow | Folded into T-3804: stage 1 works out an out-of-date boarding account in steps between edits; the 1.3 s was 10-24 |
| T-3937 High-demand edits a third slower since 11-17 | Superseded: a busy-machine comparison with rail trips up 4.5%, and the solve has changed since (T-4041, T-4054); T-4057 profiles it now, T-4025 catches drift |

Kept although old (unsure, or the idea still applies): T-414, T-168, T-281, T-344, T-346, T-875-T-877, T-777, T-898, T-1031,
T-1173, T-1240, T-361 (all P4, October figures); T-1610, T-1629, T-1730, T-2458 (the coarse/fine zone system still exists);
T-1760 (more relevant after T-4041, its 26% is from 11-04); T-2046 (possibly a headless stall, cheap to check).

Not performance despite the line or word match, left out: T-4074 (Spain blocks), T-4030 (track records), T-2198 (passing).

## What today's solve work covers (so nothing below duplicates it)

solveprof's profile (tree solveprof, notes/T-4057.md, real save natively): first-arrival search 40% (T-4054's), commutes'
mode choice 35.5%, everyday pass 11%, market 5%, assignment 6%. In the game, nearly all of an edit's 100-113 s wait is
the native solve, plus load balance (3 workers, 4 heavy periods). After a load or any change to the open markets, each
worker builds the same corridor book: 4.7-6.2 s each.

## Ranked list

Collides: S = solveprof/solvespeed (sim/src/demand, demand.worker.ts, demandClient.ts); T = track rework on tw/track
(sim/src/track, app/src/map, the clock worker files); B = split6 tree (T-3373 B, paused); P = phonedesign (T-4086).
Trade: whether it risks load time against memory or the other way round. Hours are agent hours.

| # | Task | Saves (on what, source) | Touches | Collides | Hours | Trade | Today |
|---|---|---|---|---|---|---|---|
| 1 | T-4025 load time and memory ledger | Nothing by itself; stops the back-and-forth. Draft exists | tools/perf_bench.mjs, publish skill | none (needs claimed cores; timings noisy while S runs) | 3-5 | it is the guard | yes |
| 2 | T-4051 zone open sits at 90% on a big save | 2-3 min per new zone on her North America save (her report). Likely includes the corridor book rebuild (5-6 s per worker per change on the real save, more on NA) that T-1354 added on master | demandClient, demand.worker, pack open | S (corridor book is in its profile) | 2-4 to name and time | memory: fewer workers would be slower | measure yes, fix with S |
| 3 | T-921 tiny commute pairs dealt at 1 a day | NYC solve -14% (34.3 to 29.6 s natively, notes/T-909.md, 10-22); stations move under 0.8% | sim/src/demand commutes | S | 2-3 | none; shifts riders, so her rule applies | give to solveprof |
| 4 | T-2589 + T-2247 diagram speed | All Japan 38 s to under 15, NYC 9.3 to under 4 (11-06 figures); names 0.5-1 s of NYC | app/src/diagram, diagram.worker | none | 3-6 | none | yes |
| 5 | T-3932 cores slider | No saving; the player picks memory against speed, shown before they choose | Settings, demandClient (tree coreslider, half built) | B (worker count), light S on demandClient | 2-3 | it is the trade, made visible | yes (tagged after usage) |
| 6 | T-875 evening from midday | Up to 13% of a solve (notes/T-869.md, 10-21) | sim/src/demand | S | 4-6 | none; shifts riders | after S |
| 7 | T-3373 stage B split workers | Memory: all-China 2.5 to 1.3 GB, NYC 0.67 to 0.40 GB; NYC edits 1.7-3.4x slower until stage 8 | demand workers, sim/src/demand/intercity | S, B | 15-25 left | yes, the big one: memory for time until stage 8 | no; after T-4025 |
| 8 | T-3939 clock worker memory on phones | 359-493 MB of a 1.6-1.7 GB iPhone tab (T-3781) | clock worker | T, P | 2-4 | memory, maybe load | after P's design |
| 9 | T-3160 phone memory | Solving worker 280 MB on a Tokyo phone start; parked patch 7 MB for maybe 6% slower | sim/src/demand | S, P | 3-5 | yes (the parked patch) | after P |
| 10 | T-2879 two rounds, then three | Figures one crowding round sooner (a round is 6-7 s a period natively) | demandClient ROUNDS, publish | S | 3-5 | total time may grow | after S |
| 11 | T-2188 city line edit ends with local solve | The whole solve after a city-line edit (Tokyo 8 s against 0.1 s) | demand runtime | S | 3-5 | none; drift needs her rule (T-1734) | no |
| 12 | T-1760 new express shows a quarter of its riders | First figure for a new express 26% against 72-85% (Japan, 11-04) | intercity window | S | 3-5 | none | after S, re-measure |
| 13 | T-1367 closed zones kept in heap | All-China 2,041 MB at rest after closes | demand worker, sim | S, B | 2-4 | memory only | after B |
| 14 | T-3659 PR 23 intercity assignment | 3% of a whole calculation | sim/src/demand/intercity | S | 2-3 | none | after S |
| 15 | T-2762 bubble answer on North America | 300 ms per worker to near NYC's 60 | intercity query | S (light) | 2-3 | memory: a zone row cache | after S |
| 16 | T-2458 zone picked during a load solves coarse | A second whole solve for that zone | demandLod, demandClient | S (light) | 1-2 | none | maybe |
| 17 | T-1610 clicked zone waits for refine | The refine wait on a click | demandLod | S (light) | 2-3 | none | maybe |
| 18 | T-2868 memory cap for 32-bit browsers | 32-bit Firefox fits about 1,008 MB against 688 | demand worker memory setup | none | 1-2 | none | yes |
| 19 | T-3882 bar while calculating trips here | A 40-60 s wait with no bar on her NA save | load bar UI | none | 1-2 | none | yes |
| 20 | T-3493 all-China peaks | Up to 1.2 GB between collections, not traced | measuring | machine only | 2-3 | memory only | yes, if cores free |
| 21 | T-2757 questions after a worker restart | Minutes on a big save | demandClient | S, B | design first | none | no |
| 22 | T-3708, T-3699 lost requests after a restart | A 20 s panel wait; goals waiting for good | demandClient | S (light) | 1-2 each | none | maybe |
| 23 | T-2858 Firefox and a laptop | Unknown; no player report | measuring | none | 2-3 | none | better as a T-4025 case |
| 24 | T-1211, T-2046 measure-once checks | Panels 0.4-0.7 s (10-24); a 6 s first switch | measuring | none | 0.5-1 each | none | yes |
| 25 | Edit speed: T-3550, T-3804, T-4045, T-1031 | London/windrush edits 1.2-2.3 s; edit waits 70-250 ms; one-way stops | sim/src/track, clock worker | T | 2-6 each | none | no (paused, after the rework) |
| 26 | P4 loading and memory: T-898, T-1173, T-281, T-414, T-168, T-344, T-346, T-777, T-877, T-876, T-1240, T-361 | 1-7 s of loads (October), 2-12 MB per worker, a 5-15 MB world market file | mostly sim/src/demand | S | 1-6 each | T-876, T-1240, T-361 trade memory for complexity | no |
| 27 | World tooling: T-2095, T-1992 | 159 s first register; a cold outlines run 4.4-5.1 min against 3 | pipeline | none | 1-3 each | none | yes, low value |

## The perfledger tree

`.tw-trees/perfledger` (branch tw/perfledger from master ed955687, last touched 2026-11-22 10:46) is T-4025's first
agent, paused for the week's usage before anything ran: one untracked half-written `tools/perf_bench.mjs` and the
tree's `notes/T-4025.md` (what is left: finish `runOnce` and `run`, build and bench v0.4.3, v0.4.4, v0.4.5 and master,
add the publish step). Nothing landed and nothing was measured. A new T-4025 agent can start from that tree.
