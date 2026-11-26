# Orchestrator handoff, trainworld

Orchestrator: session 72757b8d-c220-47a4-ac30-041ad0a7ac39, name maps-3a [8d9b93], since 2026-11-26 14:17 UTC
Updated: 2026-11-26 15:14 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5 (write dates to her in her time). Core cap 14 (her word, 03:32 UTC).

## Next
1. 0.4.7 published 11-25 18:42 UTC (R2 2026-11-25, website 0497af6). Player reports triaged 11-26 14:40 UTC: filed
   T-4367 (P1, demand crash 'o.beside' after a new line), T-4368, T-4369, T-4370; brown-map lead added to T-4328.
   Running: besidefix (Opus, T-4367, tree besidefix, port 6147, her "can investigate 4367" 11-26 ~10:10). Her
   no-new-agents wind-down (11-25) otherwise still unanswered: start nothing else without asking.
2. Landed 11-25 (all on 8800 98aedf0d0, smoke passed): T-4348, T-4351, T-4354, T-4357, T-4337, T-4349, T-4272 s2
   (open for 100 ms), T-4339, T-4149, T-4359, T-4153 s9 (open for 100 ms), T-4281 s3. Edit bench baseline on master
   65f0134b@20261125203108-11820; batch check after: no regression (NYC click run later 122 -> 34 ms).
   Before the next publish: a Claude batch review of d441fb19f, 914f564e6, b6aa1b90f, 67426dfa8.
3. Running: the redraws3 job (T-4364, T-4365; agent done, report r-b2d0784b): her 11-26 yes to the redraws except
   Poland, plus both new (Mansoura, Chiayi/Tainan: her yes 11-26, written to notes/T-4364/answer-1126.txt), so the
   job runs through EG/TW and the after steps without stopping (19-20 h from 14:47 UTC). Owns data/packs: snap8800 with the freeze, hold
   main-tree app/ writes during its gates. All of Macau's land goes
   in Macau's zone; it reports the Macau-Zhuhai border factor after the join.
4. Then T-4153 s10, T-4309, T-4314/T-4315. Up next: T-3160, T-3838.
6. Resources 11-25: Resend over (her: nothing); R2 11-18b, 11-19 purged (her yes); Class A 91% of the free million
   projected; Class B, repo size known to her. Cloudflare stats: her Configuration Rule leaves /trainsworld out (checked).

Standby: none listed. Gemini out until about 11-30: Claude reviewers meanwhile.

## Her words
- 11-26 ~09:10: Skierniewice: "ok i guess we can leave out poland. everything else fine to do." Asked whether Macau
  is part of Zhuhai ("shouldnt it be separate so that there can be a border factor?"): answered as item 3.
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
- 11-19: player complaints before new features. 11-25: player region requests are leads, not orders ("we gotta judge").

## Only a person knows
- The redraws3 job: cmd 13292, script PID 20708, 8 cores claimed as `redraws3`; log artifacts/trainworld/redraws3-run.log
  and notes/T-4364/kit/redraws3.state (check every couple of hours); stops at the first `FAILED:` line, which names
  the undo. Restart, undo and the after steps (changelog item, place names, smoke, snap8800, commit pipeline/
  world_plan.py, osm_units.py, build_city.py, docs zones.md, data-register.md): notes/T-4364.md "The job".
- Uncommitted, not ours (an 11-23 checks reorganisation): SPEC.md "What is covered", docs/code-tour.md,
  docs/interface.md, checks/old moves, checks/INDEX.md, tools/checks_index.py. Landings touching them: patch dance
  (`git diff --output=<f> -- <files>`, checkout, land, `git apply --3way <f>`, `git reset -- <files>`).
- DECISIONS.md holds buytrains' two T-4272 entries uncommitted: commit them with its landing.
- Uncommitted on purpose: TODO.md (hers), notes/asks-log.jsonl, notes/board-live.json, changelog/roadmap.json,
  maps .claude/settings.json, notes/T-3397.md, tools/ops/archive_bad_saves.py.
- Trees kept: _track (tw/track), tracksnap (snaptrack.py), _seed (Rust cache seed). hotfix461 removed 11-25 (branch on
  origin).
- notes/T-4214.md: stage A's section (stationclick, 11-25) sits uncommitted, not ours; keep it.
- `land` (not `finish`) conflicts on changelog/next.json when the tree has items: use `finish` (it moves them).
  `finish` drops a tree's edits to existing changelog items: redo them in the main tree.
