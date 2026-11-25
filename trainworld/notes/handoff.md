# Orchestrator handoff, trainworld

Orchestrator: session 74ece160-7d1e-4a7f-a351-88b3554e0a77, name maps-13 [f879aa], since 2026-11-25 07:18 UTC
Updated: 2026-11-25 13:20 UTC (cycling at 500k). Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5 (write dates to her in her time). Core cap 14 (her word, 03:32 UTC).

## Next
0. Usage: Gemini's weekly quota ran out 10:24 UTC 11-25, back about 12-01 03:30 UTC (ask a-2026-11-25-6: upgrade?);
   meanwhile reviews (a fresh Sonnet reviewer per risky landing, read-only) and small code go to Sonnet. Review
   findings go to a fix agent unread. Briefs: maps-13's scratch <agent>/brief.md (copy a recent one as a template).
1. Edit speed, her main focus: ROOT CAUSE ONLY (DECISIONS T-4097, notes/repeats.md). Asia now: later clicks 144-360 ms
   (T-4281), network build 190 ms (T-4153 s7), page 27 ms placed / 45 removed, worker rows 59 ms (T-4291). Running:
   modelpass (Opus, T-4299: the track model's whole passes per state, about 15 ms, plus review21's three lows on
   T-4291). Then T-4300 (complex numbering, about 7 ms; shares clockDemand.ts with modelpass, so after it). On her yes
   to a-2026-11-25-4: T-4272 rest (train work before each answer). Don't ask her to test edit speed yet (her 11-24).
2. Owed on a quiet machine: `node trainworld/tools/edit_bench.mjs --save asia,nyc --baseline` on master (11-24's
   baseline has no stop rows; trees' lone samples are cold first edits, T-4293).
3. Running, not edit speed: aside8 (Opus, T-4220 stage 8: quick figures with Use more cores on; stage 9, measuring on
   her Asia save, after); drawmem (Opus, T-4187: draw caches on phones, exact). Each landing gets a Sonnet review.
4. 8800 = d56bb9594 = master, smoke passed 13:1x; every review finding from tonight is fixed on master. T-4275 roads
   hotfix ready when she says publish (never ask). Waiting on her: asks.py list.
5. Next starts: T-4199 s2 (corridors tree, stage 1 unlanded), T-3160 (phone memory), T-4306 (P3), T-3742 after she asks.
6. reserve_cores: maps-3e's loop holds the mutex; start one (`-All -Loop 30`) if it dies.

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
- notes/T-4214.md: stage A's section (stationclick, 11-25) sits uncommitted, not ours; keep it.
- `land` (not `finish`) conflicts on changelog/next.json when the tree has items: use `finish` (it moves them).
