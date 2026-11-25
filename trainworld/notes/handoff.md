# Orchestrator handoff, trainworld

Orchestrator: session 5930d0ed-8c43-4146-931e-316bd8ffe75e, name maps-76 [78c9fa], since 2026-11-25 13:21 UTC
Updated: 2026-11-25 15:13 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5 (write dates to her in her time). Core cap 14 (her word, 03:32 UTC).

## Next
0. Usage: Gemini out until about 12-01 03:30 UTC (she: "gemini will wait"); reviews go to a fresh Sonnet reviewer per
   risky landing, findings to a fix agent unread. Briefs: maps-76's scratch <agent>/brief.md.
1. Edit speed, her main focus, T-4267's plan (her 11-25 10:0x: "a big improvement... keep going and finish the plan").
   Running: buytrains (Opus, T-4272 rest, her yes to buying trains after the confirm), complexnum (Opus, T-4300),
   netfix8 (Sonnet, review27's fixes on T-4153 s8: buildings ignore new track). Left after: T-4309 (track limits),
   T-4314/T-4315 (page), T-4153 rest (~50 ms), T-4149 s3, T-4281 rest; review28's lows go with T-4284's next stage.
2. Owed: edit_bench NYC on master (batch: modelpass's lone samples read slower) once corrfix lands; Asia baseline on a
   quiet machine (`edit_bench.mjs --save asia,nyc --baseline`, T-4293).
3. 0.4.6.1 hotfix: her yes and text given (asks.jsonl); hotfix461 (Opus) building it from 0.4.6 + 9b4573876 + the
   dismissable notice; I judge the notice screenshot (her 11-25: get it out faster), then deploy (no changelog
   review, playtest list or perf_bench: one fallback fix). Master then gets the notice and the v0.4.6.1 updates.ts entry; c030 marked shipped.
4. Also running: renameui (Opus: (R) on the rename tooltip, bulk-rename hover text, next stop name below with the speed
   dial right of Next stop; screenshots to her), ofmproxy (Sonnet, research only: OpenFreeMap terms, a Worker draft;
   her "only in cases we actually need to"), caprise (Sonnet: shonei-server MaxTwAccountBytes 64 to 128 MiB,
   committed 493e582, deploy waits for her word; shonei-server's save check is level 39 = live 0.4.6, master is 41
   (LBK1, LAP1): bump it before the next full release). ofmproxy done: notes/T-4328.md. corrfix landing.
5. Then: T-4220 s9 (Asia measuring + review24's findings, demand worker now free), T-3160 phone memory, T-4230.
6. reserve_cores: maps-3e's loop holds the mutex; start one (`-All -Loop 30`) if it dies.

## Her words
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
- 11-19: "player complaints take precedent over other new features"; full playtest list before every publish.
- Standing: every publish, deploy and website push needs her yes, none overnight; advisor only for big design calls.

## Only a person knows
- Uncommitted, not ours (an 11-23 checks reorganisation): SPEC.md "What is covered", docs/code-tour.md,
  docs/interface.md, checks/old moves, checks/INDEX.md, tools/checks_index.py. Landings touching them: patch dance
  (`git diff --output=<f> -- <files>`, checkout, land, `git apply --3way <f>`, `git reset -- <files>`).
- Uncommitted on purpose: TODO.md (hers), notes/asks-log.jsonl, notes/board-live.json, changelog/roadmap.json,
  maps .claude/settings.json, notes/T-3397.md, tools/ops/archive_bad_saves.py.
- Trees kept: _track (tw/track), tracksnap (snaptrack.py), _seed (Rust cache seed), corridors (corridors3),
  settlestop (settlefix working in it).
- reserve_cores: maps-3e's retry loop (task b19u3mylb; powershell PID 29592) follows the ledger's cap and pins Gemini
  runners; it dies with maps-3e's tab: the successor starts its own (a second copy waits on the mutex).
- notes/T-4214.md: stage A's section (stationclick, 11-25) sits uncommitted, not ours; keep it.
- `land` (not `finish`) conflicts on changelog/next.json when the tree has items: use `finish` (it moves them).
