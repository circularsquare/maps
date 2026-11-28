# Orchestrator handoff, trainworld

Orchestrator: session bae761ca-c85c-416e-b4e2-0a931ae6b8a6, name maps-dd, since 2026-11-27 23:10 UTC (Windows 11 now; maps-6c gone, nothing in flight)
Updated: 2026-11-28 03:00 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5 (write dates to her in her time). Core cap 14 (her word 11-27 18:33 her time: "we can rise core cap to 14").

## Next
1. Work order (her 11-27): real bug fixes and regular chores first; solve speed and memory where possible; track rework toward the release 1 merge (before new modes); modes (T-4479-T-4498) after. Edit-speed work stops at 50 ms an action.
2. Track rework: tracksync is merging master into tw/track (_track, levels to 44+, checklist in notes/T-3444-design/10-build-plan.md "The 11-28 sync"). Then T-4037 g3b (with T-4518: a click still picks between stations, her yes), then the release 1 list; crash round T-4519 and Asia edit bench T-4521 are gates before the merge.
3. Running: realgaps (T-4514, NYC figure diff goes to her before landing), busundo (T-4091), labelstay (T-4229), nosound (T-4376), edgecost2 (T-4534: the patch check still trips on cost; smoke failed on it once). Easy list: notes/task-sweep-2026-11-28.md (T-2165, T-3457, T-4166 left).
4. Server deploy waits for "more things in": shonei-server 07f5a75 + 3a8e65c (save level 43 with weekend offsets), unpushed; ask when a publish nears. Owed: 1 edit_bench + 1 perf_bench NYC on master, batch Gemini review before release.
5. _track was at level 42 (LBN1); master is 43. Redrawn zones on 8800: undo per notes/T-4364.md "The job"; data/work/pre-T4364* (19 GB) go 11-28 if she finds nothing.

## Her words
- 11-27 21:16-21:33 her time: track rework before new modes (DECISIONS); catch up tw/track, Gemini on it and the plan; crash testing all at once near release, not before; easy tasks to cut the count (notes/task-sweep-2026-11-28.md).
- 11-27 20:55 her time, nudges: players shift trains to space interlined lines evenly ("not one train A then a train B 2 minutes later, then like 20 minutes until the next train A"); "it would be cool if [ridership responded] ... but if it introduces lots of cost we dont have to do it in full weight way". T-4511: weekday/weekend switch yes, build now yes.
- 11-27 20:14: pause edit-speed work under 50 ms an action (DECISIONS); server deploy (07f5a75, 3a8e65c) once more is in.
- 11-27 18:30 her time: a hidden mode that was never built "dont show up anywhere, like in build menus" (T-4334 Goal); nothing to do for Resend or R2; track rework not started while there is enough other work ("shouldnt spread ourselves too thin").
- 11-27 18:25 her time, priorities: demand solve speed (if improvements are possible), memory (if actually possible), schedule nudging, monorail and cable cars, track rework. Real bug fixes and regular chores come before these.
- 11-27 18:10 her time: Windows 11 done, "can resume work as normal"; 3-10 agents at once; "stay lean as an orchestrator... try to not do that [work yourself]"; a code guard for it, yes (T-4428).
- 11-27 17:08 her time: "lets just ideally remove all references to other games. i think that'd be best"; "its harmless to make this change and probably better to state the game in its own terms". Current docs done (743e8a373); the rest is T-4427.
- 11-27 ~13:50 her time (to maps-2c, account 2): pushed master on her yes; Windows 11 upgrade coming (machine restarts; agents diskclean and win11guide running for it). Gemini for adversarial review only now that there are two Claude subs (DECISIONS). AWS raise to ~5 vCPU refused, wants usage history: use it now and then, e.g. country adds (docs/aws.md).
- 11-27 11:26 her time: Gemini quota out; "oki we can move bus fix to sonnet. but mostly just pause starting new work". Core cap 10 "for a while".
- 11-27 ~10:00: Claude runs out before Gemini, "so lets just use gemini for pretty much all new tasks. lets see how it goes."
- 11-27 ~09:45: agents 2-8, Gemini agents counted; Gemini 5x plan, Neon as Opus, Hydrogen as Sonnet (DECISIONS). Load
  proposals "3 then 1", "just proposals so no need to treat them as gospel"; Discord 30-minute loads: assume low memory or big saves.
- 11-27 ~10:00: core slider is high leverage now (T-3932, P1); light mode wouldn't help ("drawing is already very
  minimal"); T-4370/T-3160 stay high P2 unless an easy clear path; "buses are settled" dropped (DECISIONS); loads felt
  instant sometimes, 10 min others, 30 min for players: is that expected? Her guess: low memory / resources. She asked
  about reading Discord directly (T-4383). With Gemini back she's started work again (agents running 11-27).
- 11-25 ~12:50: "ok, ship." 0.4.7 (via standby maps-d6, after Belgrade setup +23% over her one-time 20% threshold).
- 11-25 ~12:15: (1-4 agents that week, lifted 11-27); usage at 83%, "dont worry about running out... dont want you
  to perform worse cuz u feel limited"; after the release, mostly quiet-machine work. T-4346 is P3.
- 11-25 ~12:10: at her parents' 1.5 days on her laptop, connecting by remote control; mostly urgent bug fixes, other
  work if there's room; keep the up-next items.
- 11-25 ~12:00: waiting rooms back to US$10k, tram stop parking smaller (T-4344); parking fee filed (T-4345); time left
  from a load's start (T-4347): "10 minutes left" then instant is "kinda funny but not a problem".
- 11-25 10:0x her time: buy trains after the confirm, cash may go negative; proxy "only in cases we actually need to";
  weekend solve after loads are faster; plays with Use more cores off; account cap 128 MB; PRs 21, 24 closed.
- 11-24 evening: edit speed's root cause is the main focus ("we should *only be targeting the root cause*"). Use more
  cores: she plays off; test off, both after big loading changes. Bars: past half only on a bounded time left. Don't
  ask her to test edit speed until it feels right on 8800. Edit-speed memory budget 30 MB. Game stays at anita.garden.
- 11-25 00:51: "the game reworking the whole network after each edit is really the target here and everything else is
  kinda just flavor." (DECISIONS T-4097.) Asia: station adds 1-3 s, removal about 4 s.
- 11-19: player complaints before new features. 11-25: player region requests are leads, not orders ("we gotta judge").

## Only a person knows
- Undoing a redrawn country, if she finds one wrong: notes/T-4364.md "The job" (restore.py T4364<cc> pre).
- Uncommitted, not ours (an 11-23 checks reorganisation): SPEC.md "What is covered", docs/code-tour.md,
  docs/interface.md, checks/old moves, checks/INDEX.md, tools/checks_index.py. Landings touching them: patch dance
  (`git diff --output=<f> -- <files>`, checkout, land, `git apply --3way <f>`, `git reset -- <files>`).
- maps .claude/settings.json: T-4428's guard entry sits beside the uncommitted autoMemoryDirectory (keep_hunks.py to commit only the hook).
- Uncommitted on purpose: TODO.md (hers), notes/asks-log.jsonl, notes/board-live.json, changelog/roadmap.json,
  maps .claude/settings.json, notes/T-3397.md, tools/ops/archive_bad_saves.py.
- Trees kept: _track (tw/track), tracksnap (snaptrack.py), _seed (Rust cache seed). hotfix461 removed 11-25 (branch on
  origin).
- notes/T-4214.md: stage A's section (stationclick, 11-25) sits uncommitted, not ours; keep it.
- `land` (not `finish`) conflicts on changelog/next.json when the tree has items: use `finish` (it moves them).
  `finish` drops a tree's edits to existing changelog items: redo them in the main tree.
