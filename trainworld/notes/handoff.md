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
- T-1733 (T-1650 stage 1): first agent handed back near its token limit (state in the tree's notes/T-1733.md):
  China met (8.8 s, worst line 0.9%), Japan not (156-182 s, six big lines 5-16% off in the local result).
  A fresh Opus agent continues in .tw-trees/t1733 (merged to 5f6293ab, WASM to rebuild): find the line error,
  then Japan's time. Old run PID 40548 / Vite PID 31284 / claim t1733 handed to it. Then T-1734, T-1735.
- T-1654 agent t1654 (Sonnet) in main tree, loadBar.ts; 4 cores; Vite 6131.
- T-1374 landed c722abb3 (Kyoto, Nara +39-57% going out; Hangzhou misses on attraction data, T-1739 filed;
  "up to twice" became 1.5x, in DECISIONS). master ff5fd7a0; 8800 rebuilt there, smoke 20/20 both shades. Not deployed.
- T-1654 committed f09936a8 with the Email (optional) placeholder; 8800 rebuilt there, smoke 20/20 both shades.
  It edited demandClient.ts (solveStage moved to loadBar.ts, job.firstLand added): t1733's tree also edits
  demandClient.ts, so its landing needs a merge onto f09936a8.
- T-1520 landed 88cc568c (groups +6-15% instead of +49-78%; rail within 0.01%; market pass interface unchanged). Tree removed.
- T-1738 Vietnam: agent t1738 (Opus), main tree, world_plan/world_run; up to 8 cores; Vite 6181, runner 6380;
  ids T-1743-1749; asks me before its join (t1625 may write data/packs/real). notes/T-1738.md.
- 8800 at 88cc568c (T-1520), smoke 20/20 both shades. Not deployed: asked Anita.
- Anita 2026-10-28 late: deploy waits for T-1733; superholiday 1.5x fine; 0.48 of MLIT "pretty low", asked how
  other countries fit: T-1742 (agent t1742, Sonnet, written answer only, 3 cores, Vite 6195).
- T-1641 agent t1641 (Sonnet), SkyBanner.tsx; Vite 6191; 2 cores.
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
Anita away: core cap 15 overnight.

## Anita's standing asks
- Every demand task (loosely perf) has a "done when"; stop once met. No single-pair demand tuning.
- Performance: approximations welcome when a player can't tell; name them in DECISIONS.md.
- Buses settled: no more tuning unless she asks.
- Thanks to agents in my own reply, not by message; ask after agents quiet 45+ min; final checks
  in the foreground; browser loads count their demand workers in core claims.
- Tokens tight this week (to ~11-01): Sonnet for settled work, Opus for design, data correctness,
  unknown bugs; compute-heavy token-light approaches; investigations stop at a written answer.
- Someday tasks are real open tasks.
