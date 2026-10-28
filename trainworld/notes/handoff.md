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
- T-1609 agent t1609 (Opus) in .tw-trees/t1609; ids T-1730-1731; 5 cores; Vite 6121.
- T-1654 agent t1654 (Sonnet) in main tree, loadBar.ts; 4 cores; Vite 6131.
- T-1650 design done (notes/T-1650.md), waits on Anita's two questions; stages T-1733-1735 reserved, T-1736-1737 unused.
  T-1732 unassigned.

## Landed by maps-a2 (all committed and deployed as 2026-10-28e; detail in ARCHIVE.md)
- T-1561 to T-1729 except those still open in TASKS.md. Key calls are in DECISIONS.md.
- Agents of maps-a2 are all finished; nothing in flight. Resume one by its id only for a
  follow-up on its own work (ids in this file's git history).

## Waiting on Anita
- Feedback while signed in should now name her account (shonei-server b0ea1ec live): asked her
  to send one.
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

## Next
1. T-1609 (refine only the zone in view against frozen loads; exact; base for T-1650), then
   T-1650 design. T-1654 (real load-time estimate), T-1634, T-1647, T-1644, T-1520.
2. T-1374 (superholidays stage 3), T-1625 (rebuild the Japan stress save).
3. Rest of the world after calibration (Anita 2026-10-27).
Ids handed out through T-1729.

## Anita's standing asks
- Every demand task (loosely perf) has a "done when"; stop once met. No single-pair demand tuning.
- Performance: approximations welcome when a player can't tell; name them in DECISIONS.md.
- Buses settled: no more tuning unless she asks.
- Thanks to agents in my own reply, not by message; ask after agents quiet 45+ min; final checks
  in the foreground; browser loads count their demand workers in core claims.
- Tokens tight this week (to ~11-01): Sonnet for settled work, Opus for design, data correctness,
  unknown bugs; compute-heavy token-light approaches; investigations stop at a written answer.
- Someday tasks are real open tasks.
