# Orchestrator handoff, trainworld

Orchestrator: session 5930d0ed-8c43-4146-931e-316bd8ffe75e, name maps-76 [78c9fa], since 2026-11-25 13:21 UTC
Updated: 2026-11-25 16:42 UTC (cycling at 500k). Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5 (write dates to her in her time). Core cap 14 (her word, 03:32 UTC).

## Next
1. Edit speed, her main focus, T-4267's plan ("a big improvement... keep going and finish the plan"). Running: buytrains
   (Opus, T-4272 rest: trains bought after the confirm, cash may go negative; money: review before landing). Left:
   T-4309 (track limits), T-4314/T-4315 (page), T-4153 rest (~50 ms), T-4149 s3, T-4281 rest, T-4339 (rowsOwed, P3).
2. 0.4.7 release prep from about 18:00 UTC (1pm her time); she leaves about 20:00 UTC for about 2 days: no publish or
   deploy until she's back; "nothing absolutely needs to be in". Rebuild 8800 at master + smoke first (8800 2398650f9 is
   behind). Publish skill: changelog review agent, sort, highlights, her board edit, playtest list, smoke8800, perf_bench
   3 saves. Her OK on new text: selectmodes' 3 labels, linestab's 2 hovers + "Profit % {percent}", helpapp's Q and A,
   waitroom's Money hover; renameui's she gave. Server deploy on her yes BEFORE the publish: shonei-server ea0720a (save
   level 41: master writes 41) + 493e582 (128 MB cap). buytrains only if reviewed in time; complexfix (T-4300 load
   shortcut, review29's lows) and loadeta (T-4343, players' 3-hour loads, "Use more cores" maybe not working) if ready.
3. Owed on a quiet machine: `edit_bench.mjs --save asia,nyc --baseline` (T-4293).
4. Then: T-4220 s9 (Asia, review24's findings), T-3160 phone memory, T-4230; P3 pad tasks T-4332-T-4337; T-4328 proxy
   only if players stay blocked after 0.4.6.1.
5. reserve_cores: maps-3e's loop holds the mutex; start one (`-All -Loop 30`) if it dies.

## Her words
- 11-25 10:5x: 0.4.7 release prep at about 1pm her time (18:00 UTC), small features first (linestab T-4329/T-4331,
  selectmodes T-4330, savecheck41 for the server deploy); she leaves about 3pm (20:00 UTC) for about 2 days: no
  publish or deploy after that until she's back. "if we dont get it out its also not the end of the world."; 11:3x: "nothing absolutely needs to be in by any point."
- 11-25 10:0x her time: edit speed "a big improvement... keep going and finish the plan"; buy trains after the confirm,
  cash may go negative; proxy "only in cases we actually need to"; weekend solve after loads are faster; plays with
  Use more cores off; Resend: nothing; account cap 128 MB yes; Gemini waits; PRs 21, 24 closed with comments.
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
- 11-24 18:5x: shipped 0.4.6. She says when to publish (never ask); no "less memory" items when memory is up overall.
- 11-23: answer her first; agents 2-8 by load; tasks about an hour; you don't investigate, agents do; cycle at 500k;
  no drafted player replies; times as percent of the wait.
- 11-19: player complaints before new features; full playtest list before every publish (not the 0.4.6.1 hotfix).

## Only a person knows
- Uncommitted, not ours (an 11-23 checks reorganisation): SPEC.md "What is covered", docs/code-tour.md,
  docs/interface.md, checks/old moves, checks/INDEX.md, tools/checks_index.py. Landings touching them: patch dance
  (`git diff --output=<f> -- <files>`, checkout, land, `git apply --3way <f>`, `git reset -- <files>`).
- Uncommitted on purpose: TODO.md (hers), notes/asks-log.jsonl, notes/board-live.json, changelog/roadmap.json,
  maps .claude/settings.json, notes/T-3397.md, tools/ops/archive_bad_saves.py.
- Trees kept: _track (tw/track), tracksnap (snaptrack.py), _seed (Rust cache seed), hotfix461 (0.4.6.1 as shipped,
  branch pushed; remove with agent-tree.mjs remove --force when done with it).
- reserve_cores: maps-3e's retry loop (task b19u3mylb; powershell PID 29592) follows the ledger's cap and pins Gemini
  runners; it dies with maps-3e's tab: the successor starts its own (a second copy waits on the mutex).
- notes/T-4214.md: stage A's section (stationclick, 11-25) sits uncommitted, not ours; keep it.
- `land` (not `finish`) conflicts on changelog/next.json when the tree has items: use `finish` (it moves them).
