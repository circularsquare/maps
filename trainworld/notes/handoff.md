# Orchestrator handoff, trainworld

Orchestrator: session b48f5ab5-f4f4-40bd-8d9d-48e27f778e65, name maps-48 [99e2b4], since 2026-11-27 15:48 UTC
Updated: 2026-11-27 16:06 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5 (write dates to her in her time). Core cap 14 (her word, 03:32 UTC).

## Next
1. Running: hidepick (Gemini Neon, T-4407), reachfill (Gemini Neon, T-4314), modeicons (Gemini Hydrogen, T-4423, her
   mode icons: look at artifacts/trainworld/review/modeicons.png before landing), phonesmoke (Sonnet, T-4372),
   crashcount (Sonnet, T-4409; may commit in shonei-server, deploy waits for her), saverev (Sonnet, read-only save
   review of batch 1127d). Gemini review of batch 1127d running (artifacts/trainworld/gemini/batch-1127d-review.md).
   8800 rebuild to 2ed96ddf8 running; smoke on master after it.
2. Batch 1127d: 02a97d401 (T-4153 s10), 6e8ba815d (T-4374), 3add0b644 (T-4396), 2ed96ddf8 (T-4380/82, save level 42).
   After both reviews: a fix agent for findings; then T-4408 with any T-4396 findings folded in (brief in maps-48's
   scratchpad briefs/t4408.md, Gemini Neon --game nyc). perf_bench NYC and a quiet edit_bench on Asia owed. Publish
   waits for her word and T-4407/T-4408. Ask open: T-4396's two notices (a-2026-11-27-13).
3. Then T-4153 stage 11 (after reachfill), T-4411 (her stations list idea, P3), T-4309, T-4315, T-3160, T-4370,
   T-3838; maps-e0's T-4420-4422 (tooling). New tasks to Gemini agents; Claude only for what Gemini can't.
4. Redrawn zones on 8800: undo per notes/T-4364.md "The job"; data/work/pre-T4364* (19 GB) go 11-28 if she finds nothing.

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
