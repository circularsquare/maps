# Orchestrator handoff, trainworld

Orchestrator: session 74ece160-7d1e-4a7f-a351-88b3554e0a77, name maps-13 [f879aa], since 2026-11-25 07:18 UTC
Updated: 2026-11-25 10:11 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5 (write dates to her in her time). Core cap 14 (her word, 03:32 UTC).

## Next
0. Usage: Claude runs out before Gemini; Gemini for reviews (gemini_audit.py) and small clear code. Stay light: briefs
   in files, report heads only, review findings go to fix agents unread. Overnight (her): calm pace, 2-4 agents.
1. Edit speed, her main focus: ROOT CAUSE ONLY (DECISIONS T-4097, notes/repeats.md). T-4281 stage 2 landed 800843cf0:
   Asia later clicks 144-360 ms (were 618-2,156; first 212); questions no longer force a pending follow-up whole.
   Running: benchwhy (Opus, T-4293: NYC edit_bench rows rose over tonight's landings, station built 188 to 587-910 ms;
   real, which landing, or noise; and whether pagestage3's held tree, T-4284 stage 3, is fit to land). Next, after
   pagestage3 is settled (clockClient overlap): T-4291 (rows and post by what an edit touched, about 144 ms on Asia),
   which also covers T-4149's rest. On her yes to a-2026-11-25-4: T-4272 rest (the train work before each answer).
   Also running, not edit speed: busnames3 (Sonnet, T-4080 stage 3, bus stop names).
2. aside6 (Opus, T-4220 stage 6, start and batch side jobs in play; results shown at stage 7). Stage 5 landed 4f27fcda6
   (checks/t4220-aside-equal.mjs: exact). Gemini batch18 review running (artifacts/trainworld/gemini/batch18/review.md:
   T-4283, T-4220 s4, T-4285, T-4153 s7, T-4281 s2); findings to a fix agent. Briefs: my scratch <agent>/brief.md.
3. Landed tonight (mine): T-4282, T-4276 (ask a-2026-11-25-5), T-4150 s2, T-4281 s1, T-4283, T-4220 s4, T-4285,
   T-4153 s7, T-4281 s2 (c034). 8800 = 73160a08f, smoke passed 08:37; rebuild + smoke once benchwhy answers. pagestage3 gets its own review once
   landed.
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
