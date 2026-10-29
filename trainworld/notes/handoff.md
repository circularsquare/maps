# Orchestrator handoff, trainworld

Orchestrator: session f9a7f88d-0531-4963-9df6-6771326e513f, name maps-45, since 2026-10-28 evening
Updated: 2026-10-29 night

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
- T-1734: built in .tw-trees/t1734, NOT landed: whole solves land every 11-14 min while building on Japan, last
  one bit-equal except Okayama 2e-9 (T-1736); but an edit during a whole solve waits up to 14 min (T-1733 alone: 2 min).
  In-between local solves: can't meet both targets at a line a minute (~4.5 min typical, big change). Tree kept;
  Anita 2026-10-29: keep T-1733's behaviour, return to it later for more thought. Tree kept. Agent idle (~357k).
- T-1759 agent t1759 (Sonnet): drop the opening-day bunting (Anita); Vite 6199; 2 cores.
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
  repaired (140 packs, 145 pairs, 1.26M trips gone); code committed with T-1743. The repaired market
  packs and registry market rows are data/uncommitted json: commit them with Thailand's after its join.
  Live 28g has the bad pairs until the next deploy.
- Thailand run started 23:33 (batch 20261028-233344, 67 rows, python 22764, log artifacts/trainworld/t1757-run-th.log);
  Singapore built too: 68 gated (ban, 66 zones, sin; knt mge relaxed). TH,SG join given the go 2026-10-29
  (~1 h, 6 jobs): nothing else may write data/packs until it reports.
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

## Overnight 2026-10-29 (Anita, off to bed): keep going with T-1735, think about the other perf tasks, and do
## or drop the small fixes ("up to you"). Core cap 15. Deploy only on her word.
- T-1759 committed e894d357 (bunting gone); 8800 rebuilt there, smoke 20/20 both shades.
- T-1762/T-1744 committed a6e9fc29 (VN 93/93; origins in towns). T-1735 landed afb89cbc (perfFlags conflict
  resolved by hand); main WASM rebuilt; 8800 at afb89cbc, smoke 20/20 both shades, real save 12,644,085.
- T-1765 Canada's five far origins: t1762 agent resumed (then maybe Kerkyra, Poland's two); only data/packs writer.
- T-1764 rule 1 committed 0ea6169e (smoke 20/20 on its Vite; couldn't reproduce the 9 span-only zones on master,
  kept as harmless). 8800 still at afb89cbc: rebuild with T-1766's landing. Its 0.16% whole-vs-fresh gap folded into T-1736.
- T-1766 agent t1766 (Opus): attraction visits from a city budget shared by Wikipedia page views (Anita 2026-10-29:
  "order of magnitude correctness" is enough; country term stays off). attractions_json.py + attractions.json; Vite 6141;
  3 cores. Review its numbers before committing.
- T-1770 committed 3edfe9a0; saves staged in artifacts/trainworld/t1770-realnet/. Follow-up running (t1770 resumed):
  Tokyo's default-blue matches back to palette, realcolour.py only. SHIP after T-1765's join, from trainworld/:
  `set REALNET_OUT=C:\Users\anita\projects\maps\artifacts\trainworld\t1770-realnet` then
  `..\venv\Scripts\python.exe pipeline\realnet.py ship`; then rebuild 8800.
- T-1305 stage 1 committed cebb5c03 (4 trunks right; Queens Blvd express 100.6% because game trains hit 160 km/h on
  the 4 km run). Stage 2 running (same agent resumed): realsave from corridors, staged in t1305-shared/; calls: trim
  Queens Blvd per 90% rule, W 4 St split by hand, fresh geom. Vite 6174, 3 cores. Not shipped until Anita looks.
- T-1305 design done (notes/T-1305.md); Anita: pairs of two (option a), go. Start stage 1 (NYC corridors, Python) once
  t1770 is done with realnet.py; then stage 2 (NYC save). One colour per trunk until she looks.
- T-1771 analytics: agent t1771 (Sonnet) building, reusing shonei-server d758c53 / website ce16e92 (notrack).
  /trainworld/ping + /trainworld/stats; commits shonei-server itself; no deploy.ps1 or push without Anita. Vite 6163.
- Thai zone names: Anita has no thoughts; dropped from asks.
- Anita 2026-10-29: no deploy yet.
- (was) T-1735 DONE in .tw-trees/t1735, not landed: measured express 85% of market riders for 18% of a pass; in play a
  new Japan express shows ~26% until the whole solve (T-1760 filed). WASM interface changed (IntercityApi window fns).
  Land after T-1762's VN run ends (its gate uses the main tree's app/WASM), then rebuild main WASM, 8800, smoke.
  Then T-1764 rule 1.
- T-1644 committed (z15 pan GPU 36% to 23%, overlay -70%, pixel-checked); T-1763 (cull stations/trains) filed P3.
- T-1739 agent t1739 (Sonnet), main tree attractions.json; Vite 6125; 2 cores.
- TH,SG join done 2026-10-29 (68/68 shipped, registry 1,960). Chiang Mai's origin is a border village (densest GHSL
  pixel), same flaw in mnm knt swa mge: t1757 rebuilding those 5 + second TH join (~90 min, writes data/packs; t1739
  told to keep off packs). Planner rule filed as T-1762. Uncommitted: all TH/SG/T-1758 json until it reports.
- T-1739 committed (no country term, COUNTRY_EFFECT=False; closed palaces need a figure; figured places win merges):
  Hangzhou +45%/+31% on rush days, Kyoto +58-60%; real save 12,644,085 (+0.6%). Asked Anita: country term?
  page views task? Not yet on 8800 (rebuild after TH/SG lands).
- t263-attractions' Tokyo National Museum hover check fails on old data too (a station card covers the mark): look
  (likely the check's view, not a game bug; T-1644 was not the cause, it fails on the old code).
- Thailand (67/67) and Singapore committed aa523771 with T-1758's market rows; 8800 rebuilt there (includes T-1739,
  T-1644), smoke 20/20 both shades, real save 12,644,085. Not deployed.
- T-1762 + T-1744 agent t1762 (Sonnet), main tree: origin rule, Vietnam's 10 far origins rebuilt, muo/cxx, VN rejoin;
  writes data/packs (only writer); 6 cores; Vite 6183, runner 6383; may file T-1765.
- T-1737 answered and archived (span steps of 0.05 touch far zones); t263 check fixed (its view, not a bug).
  T-1764 rule 1 (span tolerance 0.06) to do after T-1735 lands (both edit demand.worker.ts); rule 2 only measured first.
- Was "still to decide": T-1744 (after TH/SG join; pack writes), T-1737 (investigate), T-1736 (P3, keep), T-1634/T-1647
  (collide with T-1735's demand/market files: after it).

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
