# Orchestrator handoff, trainworld

Orchestrator: session f9a7f88d-0531-4963-9df6-6771326e513f, name maps-45, since 2026-10-28 evening
Updated: 2026-10-28 evening

Read trainworld/CLAUDE.md and TASKS.md first. History before this session: notes/handoff-2026-10-26.md;
this session's landings are in ARCHIVE.md (T-1561 to T-1691) and `git log`.

## State
- master 15c27068; 8800 and live 2026-10-28f run it (see In flight). shonei-server b0ea1ec
  deployed (signed-in feedback names the account). Everything through T-1729 is published.
- Bus running cost US$6 a bus-km (Anita, T-1718 closed). Deploy recipe: deploy.py --version <v>
  --smoked from .tw-snap's build, then website add/commit "trainsworld"/push, wait for the
  Pages run (`gh run watch`), `deploy.py --verify --page --version <v>`, then purge the old R2
  prefix. Anita approves each deploy. Website's Pages build needs Ruby 3.3 (fixed bf68979).
- shonei-server 6051eed deployed 2026-10-28 (trainworld feedback route; config shows
  trainworldFeedback:true). Whether feedback mails her is untested: asked her to send one.
- Uncommitted: Anita's TODO.md (never commit), checks/t1223-build.mjs (untracked, T-1223's).
- 8800 rebuild: scratchpad snapbuild.cmd <commit>
  (C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\f28006bc-4241-4116-9970-a65fc772bb95\scratchpad\snapbuild.cmd),
  then smoke light and `SMOKE_SHADE=dark`. Never rebuild .tw-snap while a deploy upload runs.
  Pass an explicit master hash: "HEAD" means .tw-snap's own HEAD, which left 8800 and the
  2026-10-28e deploy at 287cc550 (found 2026-10-28 evening; maps-a2 rebuilding at 15c27068 and
  redeploying as 2026-10-28f).
- Commit messages: single-quote them in PowerShell ("$1" and inner double quotes break).
- Core cap 12 while Anita is up (keep 4 free), 15 overnight.

## In flight (maps-45)
- 8800 rebuilt at 15c27068, smoke 20/20 light and dark. LIVE: 2026-10-28f (= 15c27068),
  website da0cfe7 pushed (Anita's go), verify --page all pass; R2 holds only trainworld/2026-10-28f
  (rclone purge prints a harmless GetBucketVersioning 403).
- T-1609 landed db77a72b (refine 4.7 s vs 53 s on all-China); 8800 rebuilt at f5961ca2, smoke 20/20 both shades. Not deployed.
- Vietnam committed 2b955e0c (93/95 shipped; next SE Asia: Thailand easiest). T-1733 landed ff76b15e;
  8800 rebuilt there, smoke 20/20 both shades. Deploy 2026-10-28g refused by the permission
  check: asked Anita to approve it explicitly. Then website add/commit/push, gh run watch, verify --page, purge 28f.
- T-1734: second T-1733 agent resumed, tree .tw-trees/t1734 from ff76b15e; Vite 6122; 4 cores; may file T-1736-1737.
- T-1751 committed 1550be70 (EdgeData.sep: imported crossings separated; Tokyo peak 27; real saves regenerated);
  8800 rebuilt there, smoke 20/20 both shades.
- LIVE: 2026-10-28g (= 1550be70, before T-1755), website e74fa0e pushed (Anita allowed pushes), verify --page
  all pass; R2 holds only trainworld/2026-10-28g.
- T-1757 agent t1757 (Opus), main tree: T-1743 (+T-891, T-1231 if covered), then Thailand, then Singapore
  (Anita 2026-10-28); asks me before each join; up to 6 cores; Vite 6182, runner 6382; ids T-1758-1765.
- T-1755 car ownership +5% (Anita) committed 669e81a3; 8800 rebuilt there, smoke 20/20, real save 12,564,193 (+0.65%).
  Not in 28g. T-1752 abandoned.
- T-1756 committed e833b0cf; 8800 rebuilt there, smoke 20/20 both shades. Triangles = opening-day bunting
  (T-1434), kept; asked Anita.
- T-1758 (found by t1757): reused ids left 1.27M commute trips a day in market packs going 700+ km (VN, CA);
  pair rule fixed; repair of ~250 market packs given the go 2026-10-28 late. Live 28g has the bad pairs.
- T-1752 (car side of mode choice) asked Anita. T-1738 open for the next SE Asian country (Thailand easiest).
- (was) T-1733 DONE in .tw-trees/t1733, not landed: Japan 118 s after the line is built, big lines within 0.47%;
  China 9.3 s. Fix: local rounds crowd on the rest's per-round loads (trail). Smoke 20/20 in the tree.
  Land after Vietnam's join reports (the join's browser checks use the main tree's app), then 8800, then
  deploy (Anita: deploy after T-1733). Then T-1734 to the same agent (second T-1733 agent).
- T-1654 agent t1654 (Sonnet) in main tree, loadBar.ts; 4 cores; Vite 6131.
- T-1374 landed c722abb3 (Kyoto, Nara +39-57% going out; Hangzhou misses on attraction data, T-1739 filed;
  "up to twice" became 1.5x, in DECISIONS). master ff5fd7a0; 8800 rebuilt there, smoke 20/20 both shades. Not deployed.
- T-1654 committed f09936a8 with the Email (optional) placeholder; 8800 rebuilt there, smoke 20/20 both shades.
  It edited demandClient.ts (solveStage moved to loadBar.ts, job.firstLand added): t1733's tree also edits
  demandClient.ts, so its landing needs a merge onto f09936a8.
- T-1520 landed 88cc568c (groups +6-15% instead of +49-78%; rail within 0.01%; market pass interface unchanged). Tree removed.
- T-1738 Vietnam: agent t1738 (Opus), main tree; 95 rows built, 93 gated (8 relaxed), muo/cxx parked (T-1744);
  join given the go 2026-10-28 late: nothing else may write data/packs until it reports. notes/T-1738.md.
- T-1742 done (caf84b64): game rail per resident ~flat, Tokyo 0.4x real, NY 2.5x. Anita: no per-place factor;
  lean toward Tokyo correct, game easier. T-1750 (agent t1750, Sonnet, written answer; 2 cores, Vite 6196):
  Tokyo save's speeds, crowding and frequency limits against real, fixes proposed.
- 8800 at 88cc568c (T-1520), smoke 20/20 both shades. Not deployed: asked Anita.
- Anita 2026-10-28 late: deploy waits for T-1733; superholiday 1.5x fine; 0.48 of MLIT "pretty low", asked how
  other countries fit: T-1742 (agent t1742, Sonnet, written answer only, 3 cores, Vite 6195).
- T-1641 committed c2d883b6 (0 rAF callbacks paused, was ~60/s); 8800 rebuilt there, smoke 20/20 both shades.
- T-1625 committed (807 lines, 0.48 of MLIT pkm; save at artifacts/trainworld/t737-japan-trains.save, untracked).
  Its calls (89 thin lines left out, Shinkansen not cut) I accepted: a stress save. Old notes still say 709 lines.
- T-1650 design done (notes/T-1650.md); Anita approved, stages T-1733-1735 filed. T-1732 unassigned.

## Landed by maps-a2 (all committed and deployed as 2026-10-28e; detail in ARCHIVE.md)
- T-1561 to T-1729 except those still open in TASKS.md. Key calls are in DECISIONS.md.
- Agents of maps-a2 are all finished; nothing in flight. Resume one by its id only for a
  follow-up on its own work (ids in this file's git history).

## Waiting on Anita
- Feedback confirmed arriving (Anita 2026-10-28). Email placeholder now "Email (optional)",
  uncommitted in FeedbackPanel.tsx and docs/ui-panels.md (rides with the next commit).
- T-1609 on 8800: "figures load quite fast! nice" (Anita 2026-10-28).
- Opacity slider stuck at 100% after a new game: not reproduced; one zone-selection cause fixed
  (T-1726). Asked her to note what was selected if it recurs.
- Her goal: a public, share-worthy state this week (friends first, not posted yet).

## Anita's calls this session (all in DECISIONS.md or task lines)
- Load-time approximations: lines 10-20% off acceptable for ~50% faster loads, ideally with the
  city in view exact (T-1609); T-782's -23% round skip alone not worth it. Order: T-1609 next,
  then T-1650 (cut big networks into parts; also lets "constructing..." clear per city).
- No speed bought with memory. Deletes undoable. Corners section gone from the blueprint panel.
  Straight-line default fares (US $1 + $0.08/km); Japan's lower long fares fine. Loans 1%/game
  day. Clock speed not saved. No automatic crossovers; stations placed by the player. Weekends
  run the medium timetable. Bus running cost US$6 a bus-km. Select on key 4; Shift+1-7 elevation.

## Next (Anita 2026-10-28 evening: "this order is fine"; she is away)
1. Land T-1733 and T-1654 (smoke, commit, 8800). Deploy: asked her, no answer yet; don't publish without it.
2. T-1734 then T-1735 (same agent as T-1733). T-1374 beside them (agent t1374, own tree).
3. T-1520 (income groups cheaper), T-1625 (Japan save rebuild, Sonnet). Small perf: T-1634, T-1647, T-1644, T-1641.
4. Then, if nothing pressing: T-1738, Southeast Asia from Vietnam, as a compute-heavy run.
Ids handed out through T-1738 (T-1732 unassigned; T-1736-1737 the t1733 agent's).
Core cap 12 (Anita 2026-10-28 late: keep 4 free).

## Anita's standing asks
- Every demand task (loosely perf) has a "done when"; stop once met. No single-pair demand tuning.
- Performance: approximations welcome when a player can't tell; name them in DECISIONS.md.
- Buses settled: no more tuning unless she asks.
- Thanks to agents in my own reply, not by message; ask after agents quiet 45+ min; final checks
  in the foreground; browser loads count their demand workers in core claims.
- Tokens tight this week (to ~11-01): Sonnet for settled work, Opus for design, data correctness,
  unknown bugs; compute-heavy token-light approaches; investigations stop at a written answer.
- Someday tasks are real open tasks.
