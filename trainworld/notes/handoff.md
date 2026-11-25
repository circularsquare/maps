# Orchestrator handoff, trainworld

Orchestrator: session 74ece160-7d1e-4a7f-a351-88b3554e0a77, name maps-13 [f879aa], since 2026-11-25 07:18 UTC
Updated: 2026-11-25 08:31 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5 (write dates to her in her time). Core cap 14 (her word, 03:32 UTC).

## Next
0. Usage: Claude runs out before Gemini; Gemini for reviews (gemini_audit.py) and small clear code. Stay light: briefs
   in files, report heads only, review findings go to fix agents unread. Overnight (her): calm pace, 2-4 agents.
1. Edit speed, her main focus: ROOT CAUSE ONLY (DECISIONS T-4097, notes/repeats.md). Running: followcut (Opus, T-4281,
   maps-c5's tab: later clicks wait on the previous click's whole-save follow-up; Asia click run on ff532d4e9: first
   558 ms, later 2.2 s), pagestage3 (Opus, T-4284, page work after an edit, stage 2 left 27 ms placed / 49 removed).
   On her yes to a-2026-11-25-4 (buy trains after the confirm): line-track stations confirm locally (T-4272 rest).
2. aside4 (Opus, T-4220 stage 4, side solve behind a dev call). Briefs: my scratch <agent>/brief.md.
3. Landed this round: T-4282 (2a6a45584), T-4276 (cb79c1bbd, look A; ask a-2026-11-25-5), T-4150 s2 (3d899ba02,
   money.ts and readers: risky). When followcut lands: Gemini batch review over T-4272, T-4274, T-4280, T-4281, T-4282,
   T-4150 s2; T-4283 after it (a Gemini code agent fits); 1 edit_bench NYC on master (pageedit2 saw undo 133 to 612 ms,
   stop reorder 168 to 457 against an older baseline, worker side: check it there). 8800 rebuilt at 08:3x with smoke.
4. Waiting on her: asks.py list. T-4275 roads hotfix on 8800, ready when she says publish. T-3742 after she asks.
5. Next: T-4080 s3 (Sonnet, now pageedit2 is done), T-4153 rest and T-4149 rest after followcut's step table, T-4199
   s2, T-3160, T-4187.
6. reserve_cores: maps-3e's loop (its tab) holds the mutex; start one (`-All -Loop 30`) if it dies.

## Her words
- 11-24 evening her time: edit speed's root cause is the main focus ("we should *only be targeting the root cause*",
  "please dont overindex on what i've said"); away overnight, keep going calmly; usage not high this week. Speed dial
  smaller (landed). 35 MB GPU copy OK. Use more cores: she plays off; test off, both after big loading changes. Bars:
  past half only on a bounded time left. Don't ask her to test the bar or edit speed until it feels right on 8800.
  Repeats tracked in notes/repeats.md. Edit-speed memory budget 30 MB. Game stays at anita.garden.
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
