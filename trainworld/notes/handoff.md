# Orchestrator handoff, trainworld

Orchestrator: session 554327b9-cb26-4c02-abcf-41d74d42cf79, name maps-d6 [79008c], since 2026-11-25 18:52 UTC
Updated: 2026-11-26 14:16 UTC (maps-d6 cycling at 500k to standby maps-3a). Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5 (write dates to her in her time). Core cap 14 (her word, 03:32 UTC).

## Next
1. 0.4.7 published 11-25 18:42 UTC (R2 2026-11-25, website 0497af6); no new player reports on it (19:15 UTC).
2. Landed 11-25 (all on 8800 98aedf0d0, smoke passed): T-4348, T-4351, T-4354, T-4357, T-4337, T-4349, T-4272 s2
   (open for 100 ms), T-4339, T-4149, T-4359, T-4153 s9 (open for 100 ms), T-4281 s3. Edit bench baseline on master
   65f0134b@20261125203108-11820; batch check after: no regression (NYC click run later 122 -> 34 ms).
   Before the next publish: a Claude batch review of d441fb19f, 914f564e6, b6aa1b90f, 67426dfa8.
3. Running: redraws3 (Opus, T-4364, T-4365): her 11-26 yes to all 17 redraws (Skierniewice split included) plus two
   new (El Mansura, Chiayi/Tainan) whose sheets it makes first (send them with SendUserFile: she's on remote
   control); one unattended job (~16-17 h), Egypt and Taiwan gated on her yes to the two.
4. Then T-4153 s10, T-4309, T-4314/T-4315. Up next: T-3160, T-3838.
6. Resources 11-25: Resend over (her: nothing); R2 11-18b, 11-19 purged (her yes); Class A 91% of the free million
   projected; Class B, repo size known to her. Cloudflare stats: her Configuration Rule leaves /trainsworld out (checked).

Standby: maps-3a (notes/standby.txt). Gemini out until about 11-30: Claude reviewers meanwhile.

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
- 11-19: player complaints before new features. 11-25: player region requests are leads, not orders ("we gotta judge").

## Only a person knows
- redraws3 runs inside maps-d6's tab; its interim report (the two new sheets, artifacts/trainworld/review/
  redraws-1126-*.png) may reach only that tab: watch for those files and notes/T-4364.md "The job"; send her the sheets.
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
