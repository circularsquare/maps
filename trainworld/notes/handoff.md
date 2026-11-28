# Orchestrator handoff, trainworld

Orchestrator: session e4482553-ed27-4817-abf8-a654e5f6c3ac, name maps-f0, account 2 (claude2), since 2026-11-28 12:44 UTC
Updated: 2026-11-28 15:24 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5 (write dates to her in her time). Core cap 14 (her word 11-28 00:05 her time: "we can raise core cap back to 14, and use a few more for country run if helpful").

## Next
1. Daytime is short-term work; country adds overnight only (her 11-28). Main features: track rework, then train shifting. 3-10 agents; reviews by Gemini when its quota allows (DECISIONS). Edit-speed P1s are T-4431 and T-4442 only (sweep, her yes).
2. Land staleworker (T-4618: 8800 and deploy keep 2 days of builds) before the next 8800 rebuild, then snap8800 at master (8800 is at ef76965ae; zonebubble, cityload2, phonemem3 stage 9 landed after). Smoke after.
3. Track rework (tw/track): pick4518 on T-4518. Next: T-4593 (join drawn by pointer, not shown in game) and a fix agent for release-findings' T-4520 store.rs:695 (a second Add a track charges the station upgrade before Construct: money); the rest of T-4037 (notes/T-4037.md groups) after T-4518.
4. Train shifting (switched off until T-4559): shift8 on T-4550 (save block: Gemini saves review before landing), shiftarrows on T-4597 then T-4556; next T-4551.
5. Running bugs: linefare (T-4615, Sichuan profit before the solve settles), tokyodrop (T-4605, Tokyo -9.4% rail trips on the iPhone path), benchhold (T-4603 core hold, then the owed master timing runs incl. Asia edit_bench --steps for the edit-speed sweep), geminidesign (T-4617 design review), boardq (T-3002, T-3003 "already done?").
6. Her open asks: a-5 countries and borders design calls (build T-4617 stages after her answer and the Gemini review), a-6 updated-page text. AWS: 1-vCPU jobs now and then for usage history; remind her to re-ask the raise about 12-05 (notes/aws-usage.md). Country batch 3 (TJ, KG, TM, MN) tonight from notes/country-plan-2026-11-28.md.

## Her words
- 11-28 10:20 her time: reload/quit prompts save first ("Save and reload"; CLAUDE.md Player text, T-4622); T-4617 shows long trips' factor only, and she asks how hard one ratio for every trip kind would be (borders agent answering); Gemini review of the T-4501 bubbles fix, she is skeptical it covers every case (geminibubbles); text acknowledged.
- 11-28 10:15 her time: country adds overnight only ("lets focus on more short term work for daytime"); batch 3 (TJ, KG, TM, MN) starts tonight from the plan notes.
- 11-28 08:45 her time: edit-speed priorities as the sweep proposed, yes; AWS plan yes: 1-vCPU jobs now and then to build a usage history, re-ask the raise about 12-05 (hers), timing runs here meanwhile with cores held for the run (benchhold agent queued until an agent slot frees; >10 asks first). Sizing: 200-500k tokens, note at 400k (T-4600). T-4586 yes. Timetables are a feature priority too.
- 11-28 08:00-08:35 her time: delegate, don't work ("please spawn subagents rather than just working"; "please respond rather than starting work!"); keep 3-10 agents busy; track rework is "the main new feature priority, lets please keep working on that"; timing runs on AWS (DECISIONS; AWS refused 4 vCPU today, her call on 1-vCPU runs pending); reminders not blocks (DECISIONS); "why are there 42 open edit speed tasks?" (sweep running); T-3938 warm start dropped; redrawn zones fine; board auto-refresh (T-4595, board.js).
- 11-28 07:50 her time: an hourly watchdog that restarts a stalled orchestrator, yes ("maybe hourly", T-4582); T-4230 packs after the first state, yes (T-4583); PR 23 reply "ok to send yes" after the next open-source sync; show her the text itself, never a notes file to search. Asked what T-3938 is (warm start was never on 8800; her fast loads were the T-328 cache).
- 11-27 23:00-23:37 her time: chores' last run is stamped on their lines in docs/orchestrator.md (her idea); smooth fade for T-4250 yes (DECISIONS); hold PR 42 and 41 (one-way pairing was "not a priority", never a rule: DECISIONS reworded); PR 45 "can do it eventually".
- 11-27 21:16-21:33 her time: track rework before new modes (DECISIONS); catch up tw/track, Gemini on it and the plan; crash testing all at once near release, not before; easy tasks to cut the count (notes/task-sweep-2026-11-28.md).
- 11-27 20:55 her time, nudges: players shift trains to space interlined lines evenly ("not one train A then a train B 2 minutes later, then like 20 minutes until the next train A"); "it would be cool if [ridership responded] ... but if it introduces lots of cost we dont have to do it in full weight way". T-4511: weekday/weekend switch yes, build now yes.
- 11-27 20:14: pause edit-speed work under 50 ms an action (DECISIONS); server deploy (07f5a75, 3a8e65c) once more is in.
- 11-27 18:30 her time: a hidden mode that was never built "dont show up anywhere, like in build menus" (T-4334 Goal); nothing to do for Resend or R2; track rework not started while there is enough other work ("shouldnt spread ourselves too thin").
- 11-27 18:25 her time, priorities: demand solve speed (if improvements are possible), memory (if actually possible), schedule nudging, monorail and cable cars, track rework. Real bug fixes and regular chores come before these.
- 11-27 18:10 her time: Windows 11 done, "can resume work as normal"; 3-10 agents at once; "stay lean as an orchestrator... try to not do that [work yourself]"; a code guard for it, yes (T-4428).
- 11-27 17:08 her time: "lets just ideally remove all references to other games. i think that'd be best"; "its harmless to make this change and probably better to state the game in its own terms". Current docs done (743e8a373); the rest is T-4427.
- 11-27 ~13:50 her time (to maps-2c, account 2): pushed master on her yes; Windows 11 upgrade coming (machine restarts; agents diskclean and win11guide running for it). Gemini for adversarial review only now that there are two Claude subs (DECISIONS). AWS raise to ~5 vCPU refused, wants usage history: use it now and then, e.g. country adds (docs/aws.md).
- 11-27 11:26 her time: Gemini quota out; "oki we can move bus fix to sonnet. but mostly just pause starting new work". Core cap 10 "for a while".
- 11-27 ~09:45: agents 2-8, Gemini agents counted; Gemini 5x plan, Neon as Opus, Hydrogen as Sonnet (DECISIONS). Load
  proposals "3 then 1", "just proposals so no need to treat them as gospel"; Discord 30-minute loads: assume low memory or big saves.
- 11-25 ~12:15: (1-4 agents that week, lifted 11-27); usage at 83%, "dont worry about running out... dont want you
  to perform worse cuz u feel limited"; after the release, mostly quiet-machine work. T-4346 is P3.
- 11-19: player complaints before new features. 11-25: player region requests are leads, not orders ("we gotta judge").
- 11-25 ~12:10: at her parents' 1.5 days on her laptop, connecting by remote control; mostly urgent bug fixes, other
  work if there's room; keep the up-next items.
  from a load's start (T-4347): "10 minutes left" then instant is "kinda funny but not a problem".
- 11-25 10:0x her time: buy trains after the confirm, cash may go negative; proxy "only in cases we actually need to";
  weekend solve after loads are faster; plays with Use more cores off; account cap 128 MB; PRs 21, 24 closed.
- 11-24 evening: edit speed's root cause is the main focus ("we should *only be targeting the root cause*"). Use more
  cores: she plays off; test off, both after big loading changes. Bars: past half only on a bounded time left. Don't
  ask her to test edit speed until it feels right on 8800. Edit-speed memory budget 30 MB. Game stays at anita.garden.
- 11-25 00:51: "the game reworking the whole network after each edit is really the target here and everything else is
  kinda just flavor." (DECISIONS T-4097.) Asia: station adds 1-3 s, removal about 4 s.

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
