# Orchestrator handoff, trainworld

Orchestrator: session 6ed48fec-8b16-41be-8883-aad8f899bcb0, name maps-e0 [81f3a0], since 2026-11-27 14:30 UTC
Updated: 2026-11-27 15:44 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5 (write dates to her in her time). Core cap 14 (her word, 03:32 UTC).

## Next
1. Running (6): busstops (Sonnet, T-4380 + T-4382, 6153: her lead-in answer and busstops-review.md findings 1-3; adds
   save block LAF1, level 42); loopfix2 (Gemini Hydrogen on tree loopfix: ferry port finding, then land T-4374, texts
   acknowledged); corescheck (Gemini Hydrogen, check only of the slider fixes); oomretry (Gemini Neon, T-4396);
   showblock (Gemini Hydrogen, T-4406: her one Show block); reachfill (Gemini Neon, T-4314). Gemini batch review of
   b59a6e59c, 3d32ddb43, 043eb3c71, 8bceef952, 85df2231a running: batch-1127c-review.md DONE: T-4381, T-4390, slider PASS; T-4400 and T-4346 FAIL (T-4407, T-4408); 02a97d401 next.
   gemini_agent's claim fails fast when the cap is full ("waiting ... exit 0"): start Gemini agents when cores are free.
2. Landed today: T-4377, T-4383, T-3932 + fixes, T-4381, T-4390, T-4261, T-4404, T-4400 (c012), T-4346 (c013), T-4153
   s10 (02a97d401; by parts ~92-96 ms, untimed on a quiet machine); notes only: T-4395 (closed), T-4373 (P4). 8800 =
   1f207a7eb, smoke PASSED; rebuild after busstops and loopfix land. perf_bench NYC and a quiet edit_bench on Asia owed.
3. She is checking the redrawn zones on 8800; undo: notes/T-4364.md "The job". Backups data/work/pre-T4364* (19 GB) go
   11-28 if she finds nothing; disk 37.6 GB free now, about 60 after.
4. Queued: T-4408 (after oomretry lands: same files; batch-1127c finding 3 + 6); hidepick (Gemini Neon, T-4407) runs
   for findings 1, 2, 4, 5. T-4153 stage 11 (train speeds, after reachfill: both in the network patch); then T-4309, T-4315, T-3160,
   T-4370, T-3838. New tasks go to Gemini agents (her 11-27), --game for a game run; Claude only for what Gemini can't.

## Her words
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
- Uncommitted on purpose: TODO.md (hers), notes/asks-log.jsonl, notes/board-live.json, changelog/roadmap.json,
  maps .claude/settings.json, notes/T-3397.md, tools/ops/archive_bad_saves.py.
- Trees kept: _track (tw/track), tracksnap (snaptrack.py), _seed (Rust cache seed). hotfix461 removed 11-25 (branch on
  origin).
- notes/T-4214.md: stage A's section (stationclick, 11-25) sits uncommitted, not ours; keep it.
- `land` (not `finish`) conflicts on changelog/next.json when the tree has items: use `finish` (it moves them).
  `finish` drops a tree's edits to existing changelog items: redo them in the main tree.
