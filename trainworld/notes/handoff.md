# Orchestrator handoff, trainworld

Orchestrator: session 554327b9-cb26-4c02-abcf-41d74d42cf79, name maps-d6 [79008c], since 2026-11-25 18:52 UTC
Updated: 2026-11-25 19:19 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5 (write dates to her in her time). Core cap 14 (her word, 03:32 UTC).

## Next
1. 0.4.7 published 11-25 18:42 UTC (R2 2026-11-25, website 0497af6); no new player reports on it (19:15 UTC).
2. Landed 11-25: T-4348, T-4351, T-4337, T-4349 (labels only; T-4356 filed), T-4354 (all 12 languages at 3303/3303;
   one pass plus own back-translation, my call; publish skill step 4 now translates missing and stale keys), T-4357
   (her "slowest limit"), T-4272 stage 2 and T-4339 (566aa52a5; review31 passed; T-4358 filed; T-4272 stays open
   for its 100 ms goal). 8800 at 566aa52a; smoke passed there (30 checks). No agents running.
3. Edit bench baseline on master 65f0134b (11-25 20:31 UTC, quiet): Asia station built 341 ms, click run later 100;
   NYC station 76. Landed: T-4149 closed (d441fb19f; rows 11-53 ms on Asia; T-4360), T-4359 (914f564e6: her "stuck"
   load was a mid-game load of Haeju waiting 4.7 min on an edit's re-solve, bar full; T-4361, T-4362), T-4153 s9
   (b6aa1b90f: bands 27 to 15 ms in Node; T-4153 open for 100 ms; T-4363 filed). netcut9's page bench was worse (NYC
   station built 147 -> 249 not noisy, n 1; Asia noisy): after today3 lands, on master quietly: edit_bench NYC --runs
   3 --compare 65f0134b@20261125203108-11820 done 11-25: no regression (station 76 -> 80 within its range; click
   run later 122 -> 34, undo 143 -> 95, delete a line 254 -> 152). 8800 at 98aedf0d0, smoke passed. Wound down 11-25 22:40 UTC: no agents, nothing running. T-4281
   s3 landed (67426dfa8: `today` ~107 to 3-6 ms on Asia with the clock running; page click hold left as is, see
   notes). Before the next publish: a Claude batch review of d441fb19f, 914f564e6, b6aa1b90f, 67426dfa8. Then
   T-4153 s10, T-4309, T-4314/T-4315. Up next: T-3160, T-3838.
5. Standby: maps-3a (notes/standby.txt). Gemini out until about 11-30: Claude reviewers meanwhile.
6. Resources 11-25: Resend over (her: nothing); R2 11-18b, 11-19 purged (her yes); Class A 91% of the free million
   projected; Class B, repo size known to her. Cloudflare stats: her Configuration Rule leaves /trainsworld out (checked).

## Her words
- 11-25 ~17:20: usage at 88%: finish the running tasks and wind down; no new agents. Batch review waits for later.
- 11-25 ~15:00: quiet-machine work (baseline, edit-speed stages) "would make sense to start after" her League game.
- 11-25 ~12:50: "ok, ship." 0.4.7 (via standby maps-d6, after Belgrade setup +23% over her one-time 20% threshold).
- 11-25 ~12:15: rest of the week 1-4 agents (not 2-8); usage at 83%, "dont worry about running out... dont want you
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
- 11-25: Airports later, P3. 180th meridian: "if people want to build it, they should ideally be able to".
- 11-19: player complaints before new features.

## Only a person knows
- Uncommitted, not ours (an 11-23 checks reorganisation): SPEC.md "What is covered", docs/code-tour.md,
  docs/interface.md, checks/old moves, checks/INDEX.md, tools/checks_index.py. Landings touching them: patch dance
  (`git diff --output=<f> -- <files>`, checkout, land, `git apply --3way <f>`, `git reset -- <files>`).
- DECISIONS.md holds buytrains' two T-4272 entries uncommitted: commit them with its landing.
- Uncommitted on purpose: TODO.md (hers), notes/asks-log.jsonl, notes/board-live.json, changelog/roadmap.json,
  maps .claude/settings.json, notes/T-3397.md, tools/ops/archive_bad_saves.py.
- Trees kept: _track (tw/track), tracksnap (snaptrack.py), _seed (Rust cache seed). hotfix461 removed 11-25 (branch on
  origin).
- reserve_cores: maps-96's loop holds the mutex while its tab stays open; a successor starts its own (a second exits).
- notes/T-4214.md: stage A's section (stationclick, 11-25) sits uncommitted, not ours; keep it.
- `land` (not `finish`) conflicts on changelog/next.json when the tree has items: use `finish` (it moves them).
  `finish` drops a tree's edits to existing changelog items: redo them in the main tree.
