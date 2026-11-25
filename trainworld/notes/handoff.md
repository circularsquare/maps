# Orchestrator handoff, trainworld

Orchestrator: session 74ece160-7d1e-4a7f-a351-88b3554e0a77, name maps-13 [f879aa], since 2026-11-25 07:18 UTC
Updated: 2026-11-25 12:23 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5 (write dates to her in her time). Core cap 14 (her word, 03:32 UTC).

## Next
0. Usage: Gemini's weekly quota ran out 10:24 UTC 11-25, back about 12-01 03:30 UTC (ask a-2026-11-25-6: upgrade?);
   meanwhile reviews and small code go to Sonnet. Stay light; review findings go to fix agents unread. Overnight: 2-4 agents.
1. Edit speed, her main focus: ROOT CAUSE ONLY (DECISIONS T-4097, notes/repeats.md). Asia: later clicks 144-360 ms
   (T-4281 s2), network build 190 ms (T-4153 s7), page 27 ms placed / 45 removed (T-4150 s2, T-4284 s3, all landed).
   NYC "slower" was noise (T-4293: trees' lone samples are cold first edits; edit_bench now compares first with first).
   editbench/baseline.json is 11-24's: take a fresh --baseline on master on a quiet machine. Next: T-4291 (rows and
   post: landed 755b1278f, Asia 144 to 59 ms; review21: 3 lows; modelpass (Opus,
   T-4299 + those lows) running; T-4300 next); T-4214 done (D-E landed 4a2ab2d37, review22 Sonnet on it); T-4294 (Edit stops clicks
   sending a question and no edit, P2). On her yes to a-2026-11-25-4: T-4272 rest (train work before each answer).
   Not edit speed: T-4080 bus stop names done and archived (081dcb203); connections are T-4083.
2. T-4220 stages 6-7 landed together 4ffd70e86 (NYC edited line's figures 17 s instead of 53); review20 found 6 (stale quick
   figures over a newer network, no early figures when refused, side figures read as final); fixes20 (Opus, T-4307). Until it lands:
   no publish from master and 8800 stays on 6c1cfb8ac (before T-4220 s6-7). Left: stage 8 (Use more cores on), 9.
   rowsdelta (Opus, T-4291). Briefs: my scratch <agent>/brief.md.
3. Landed tonight (mine): T-4282, T-4276 (ask a-2026-11-25-5), T-4150 s2, T-4281 s1-2, T-4283, T-4220 s4-5, T-4285,
   T-4153 s7, T-4080 s3 + review fixes, T-4293, T-4284 s3, T-4297. 8800 = 6c1cfb8ac, smoke passed 11:3x.
4. Waiting on her: asks.py list. T-4275 roads hotfix on 8800, ready when she says publish. T-3742 after she asks.
5. Next (not edit speed): T-4199 s2, T-3160, T-4187, T-4294 (Edit stops clicks lost).
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
- `land` (not `finish`) conflicts on changelog/next.json when the tree has items: use `finish` (it moves them).
