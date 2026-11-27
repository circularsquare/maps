# Orchestrator handoff, trainworld

Orchestrator: session b9171a67-c849-4831-91d5-3e9c74c64f32, name maps-2c (account 2, claude2), since 2026-11-27 18:53 UTC
Updated: 2026-11-27 18:25 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5 (write dates to her in her time). Core cap 10 (her word, 16:16 UTC: "lets drop core cap to 10 for a while").

## Next
0. 11-27 ~14:45 her time: she is upgrading the PC to Windows 11 (BIOS fTPM + Secure Boot first). After the reboot: 8800 (serve.py), the board watch and reserve_cores need starting again; Claude sessions start fresh (/takeover trainworld).
1. Paused by her (11:26 her time): no new work. Parked: loadsave (T-4408 + Sonnet findings 4-5 on T-4396, tree
   loadsave empty) and pagestage5 (T-4284 s5, tree has partial work); restart with --tree when she lifts the pause.
   Gemini quota was out 16:24 UTC, back ~19:45 UTC. Gemini agents can't run tools/edit_bench.mjs: it's mine on master.
2. Batch 1127d reviewed; T-4380's findings fixed (b5697efed); smoke PASSED after it. Gemini finding 2 (putEntries,
   sampleStroke on an empty path) goes into T-4153 stage 11. Batch 1127e (hidepick, reachfill, crashcount, busfix,
   later pagestage5 and loadsave): one Gemini review, then 1 edit_bench NYC on master. perf_bench NYC owed.
3. Asks open: T-4396 notices (a-2026-11-27-13); shonei-server deploy 54c6241 + ff4e198 (a-2026-11-27-14; save
   check level 42 must be live before the next publish). _track also sits at level 42 (LBN1): renumber at sync.
4. T-4424 second Claude account: set up and committed; left her login (claude2) and the test in notes/T-4424.md.
   Uncommitted from it: maps .claude/settings.json autoMemoryDirectory, tools/checks_index.py (has others' edits).
5. Then T-4153 stage 11, T-4309, T-4315, T-3160, T-4370, T-3838; T-4420-4422 (tooling). Core cap 10.
6. Redrawn zones on 8800: undo per notes/T-4364.md "The job"; data/work/pre-T4364* (19 GB) go 11-28 if she finds nothing.

## Her words
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
- Uncommitted on purpose: TODO.md (hers), notes/asks-log.jsonl, notes/board-live.json, changelog/roadmap.json,
  maps .claude/settings.json, notes/T-3397.md, tools/ops/archive_bad_saves.py.
- Trees kept: _track (tw/track), tracksnap (snaptrack.py), _seed (Rust cache seed). hotfix461 removed 11-25 (branch on
  origin).
- notes/T-4214.md: stage A's section (stationclick, 11-25) sits uncommitted, not ours; keep it.
- `land` (not `finish`) conflicts on changelog/next.json when the tree has items: use `finish` (it moves them).
  `finish` drops a tree's edits to existing changelog items: redo them in the main tree.
