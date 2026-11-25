# Orchestrator handoff, trainworld

Orchestrator: session caa91e48-7288-4753-9895-c2b79ad5e3d9, name maps-c5 [fde5f5], since 2026-11-25 02:20 UTC
Updated: 2026-11-25 04:46 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5 (write dates to her in her time). Core cap 14 (her word, 03:32 UTC).

## Next
0. Usage: Claude runs out before Gemini; Gemini for reading and reviews (gemini_audit.py), small clear code to Gemini
   agents. Gemini quota hit 03:43 UTC, resets about 04:57: then batch review (template gemini/batch15/brief.md) over
   T-4149, T-4239, T-4243, T-4249, T-4251, T-4255-T-4259, T-4262-T-4265, T-4266, T-4270, T-4271; rerun gemini_agent
   t3303fix (T-4233, tree kept); T-4242 design draft (gemini/t4242/brief.md). Stay light: briefs in files.
1. Edit speed, her main focus: ROOT CAUSE ONLY (DECISIONS T-4097). Landed: T-4272 stage (answer from own checks for
   edits that move no running line's trains). 8800 ff532d4e9 click run on Asia: first 558 ms, later 2.2 s (p90 2.9).
   Running: followcut (Opus, T-4281: follow-up steps by record so later clicks don't wait), linenames (Opus, T-4280
   in tree trainslots: renderer told what changed; then land trainslots T-4274+T-4280 together). Her answer pending:
   buy trains after the confirm (a-2026-11-25-4); on yes, line-track stations confirm locally (T-4272 rest).
1b. batch16a review (gemini/batch16a/review.md, unread): its Opus fix agent starts once answerfirst and trainslots land.
2. Landed: roadsblock (Opus, T-4275 P1: roads and water missing behind tracker blockers since 0.4.6; the hotfix
   publish waits for her morning, publish skill).
3. 8800 = 93412fc20, smoke PASSED 05:43 UTC (has the T-4275 roads fix: tell her it's ready for the hotfix publish when
   she's back; publish skill). Landed overnight: T-4262 s2, T-4273, T-4233, T-4277, T-4279, T-4275, T-4243 dial.
   Next starts: T-4276 group timetable, T-4080 s3-6, T-4199 s2, T-3160, T-4187. Overnight: calm pace, 2-4 agents.
   T-3742 after she asks; T-4250; T-4248. Commit bookkeeping and push now and then.
5. reserve_cores: maps-3e's loop holds the mutex; start mine (`-All -Loop 30`) once its tab closes.

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
