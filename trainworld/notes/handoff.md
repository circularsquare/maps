# Orchestrator handoff, trainworld

Orchestrator: session caa91e48-7288-4753-9895-c2b79ad5e3d9, name maps-c5 [fde5f5], since 2026-11-25 02:20 UTC
Updated: 2026-11-25 07:17 UTC (cycling at 500k). Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5 (write dates to her in her time). Core cap 14 (her word, 03:32 UTC).

## Next
0. Usage: Claude runs out before Gemini; Gemini for reviews (gemini_audit.py) and small clear code. Stay light: briefs
   in files, report heads only, review findings go to fix agents unread. Overnight (her): calm pace, 2-4 agents.
1. Edit speed, her main focus: ROOT CAUSE ONLY (DECISIONS T-4097, notes/repeats.md). Running: followcut (Opus, T-4281:
   follow-up steps by record so later clicks don't wait; brief in maps-c5 scratch followcut/brief.md). Asia click run on
   8800 ff532d4e9: first 558 ms, later 2.2 s (edit_bench --save asia --only clickrun). Landed tonight: T-4149, T-4249,
   T-4153 s6, T-4262 s1-2, T-4274, T-4280, T-4272 stage. Her answer pending a-2026-11-25-4 (buy trains after the confirm):
   on yes, line-track stations confirm locally (T-4272 rest). Then measure the click run again and report yes/no per edit.
2. fixes16a (Opus, T-4282: batch16a review findings, avoiding followcut's files; findings there go under "For followcut"
   in its notes: pass them on). Batch review due over T-4272, T-4274, T-4280, T-4281 when landed.
3. 8800 = ff532d4e9, smoke passed 07:06 UTC; T-4274/T-4280 landed after (rebuild next round). T-4275 roads fix is on
   8800: tell her it's ready for the hotfix publish when she's back (publish skill; her word).
4. Waiting on her: asks.py list (weekend solve T-4242, OpenFreeMap proxy, buy trains after confirm, rename and
   speedometer text, Asia load on fewer workers, PR #21/#24 comments). T-3742 after she asks; T-4250; T-4248.
5. Next starts (not edit speed): T-4276 group timetable, T-4080 s3-6, T-4199 s2, T-3160, T-4187.
6. reserve_cores: maps-3e's loop (its tab) holds the mutex; start one (`-All -Loop 30`) if it dies.

## Her words
- 11-24 evening her time: edit speed's root cause is the main focus ("we should *only be targeting the root cause*",
  "please dont overindex on what i've said"); away overnight, keep going calmly; usage not high this week. Speed dial
  smaller (landed). 35 MB GPU copy OK. Use more cores: she plays off; test off, both after big loading changes. Bars:
  past half only on a bounded time left. Don't ask her to test the bar or edit speed until it feels right on 8800.
  Repeats tracked in notes/repeats.md. Edit-speed memory budget 30 MB. Game stays at anita.garden.
- 11-25 00:51: "the game reworking the whole network after each edit is really the target here and everything else is
- 11-25 00:51: "the game reworking the whole network after each edit is really the target here and everything else is
  kinda just flavor. we should mostly be focusing on isolating the changes and cutting out the impacts on the broader
  map." (DECISIONS T-4097; T-4149, T-4150, T-4262 at P1.) Asia: station adds 1-3 s, removal about 4 s.
- 11-25: PR comments: never close without one; the reason, thankful not apologetic, normal capitalization; her OK
  first. Gemini code agents may run commands unattended in their tree; first 3 rounds asked about permissions (done).
  No terminal windows from anything we start (CLAUDE.md Shell). Airports later, P3, "dont reinvent the wheel". 180th
  meridian: "if people want to build it, they should ideally be able to". T-3742 after settlestop (my call).
- 11-24 22:4x: "you guys get a lot more mileage by thinking about actual code than by trying to test" (CLAUDE.md).
- 11-24 18:5x: shipped 0.4.6; root causes over surface fixes (DECISIONS T-4097). She says when to publish (never ask);
  asianita save is a stress test only; no "less memory" items when memory is up overall; keep cores 0-1 free.
- 11-23: answer her first; agents 2-8 by load; tasks about an hour; you don't investigate, agents do; cycle at 500k;
  no drafted player replies; times as percent of the wait.
- 11-19: "player complaints take precedent over other new features"; full playtest list before every publish.
- Standing: every publish, deploy and website push needs her yes, none overnight; advisor only for big design calls.

## Only a person knows
- Uncommitted, not ours (an 11-23 checks reorganisation): SPEC.md "What is covered", docs/code-tour.md,
  docs/interface.md, checks/old moves, checks/INDEX.md, tools/checks_index.py. Landings touching them: patch dance
  (`git diff --output=<f> -- <files>`, checkout, land, `git apply --3way <f>`, `git reset -- <files>`).
- Uncommitted on purpose: TODO.md (hers), notes/asks-log.jsonl, notes/board-live.json, changelog/roadmap.json,
  maps .claude/settings.json, notes/T-3397.md, tools/ops/archive_bad_saves.py.
- Trees kept: _track (tw/track), tracksnap (snaptrack.py), _seed (Rust cache seed), corridors (T-4199 stage 1),
  settlestop (settlefix working in it).
- reserve_cores: maps-3e's retry loop (task b19u3mylb; powershell PID 29592) follows the ledger's cap and pins Gemini
  runners; it dies with maps-3e's tab: the successor starts its own (a second copy waits on the mutex).
- `land` (not `finish`) conflicts on changelog/next.json when the tree has items: use `finish` (it moves them).
