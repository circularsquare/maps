# Orchestrator handoff, trainworld

Orchestrator: session 554327b9-cb26-4c02-abcf-41d74d42cf79, name maps-d6 [79008c], since 2026-11-25 18:52 UTC
Updated: 2026-11-25 19:19 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5 (write dates to her in her time). Core cap 14 (her word, 03:32 UTC).

## Next
1. 0.4.7 published 18:42 UTC (R2 version 2026-11-25, website 0497af6, export b89a791; bench marked published). Live
   smoke passed. Player reports since: none new on 0.4.7 (checked 19:15 UTC).
2. Landed 11-25: T-4348, T-4351, T-4337, T-4349 (labels only; T-4356, T-4357 filed), T-4354 stages 1-2 (ja, zh, ko,
   de, fr, es, pt). 8800 rebuild at 22a3efd23 and smoke on master running (scratchpad snapbuild.log, smoke.log).
3. Running: tr117c (Sonnet, tree tr117c, 6147, T-4354 stage 3: tr, ru, ca, id, glossary excepts for "boardings");
   then I edit the publish skill's step 4 to translate missing and stale keys, and close T-4354 (one pass plus own
   back-translation, my call). review31 (Opus, read-only) on buyfix (tree buytrains, unlanded: review30's 1, 2, 4, 7
   fixed, notice "More trains bought for {price}.", her "text ack"). review31 says land (artifacts/trainworld/reviews/
   review31-buyfix.md); buyfix resumed for its findings 1, 2, 4 (3 filed, T-4358); then land T-4272, T-4339, 8800.
4. Then on the quiet machine `edit_bench.mjs --save asia,nyc --baseline` (T-4293). Edit-speed stages (T-4309, T-4314/T-4315, T-4153, T-4149 s3, T-4281), T-4220 s9, T-4230, T-3160 wait for
   her return. Up next: T-3160, T-3838.
5. Standby: maps-3a is the next cycle's (notes/standby.txt); none left: ask her relay session for one. Gemini's quota
   is out until about 11-30: Claude reviewers meanwhile.
6. Resources 11-25: Resend over (her: nothing); R2 11-18b, 11-19 purged (her yes); Class A 91% of the free million
   projected; Class B, repo size known to her. Cloudflare stats: her Configuration Rule leaves /trainsworld out (checked).

## Her words
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
- reserve_cores: maps-96's loop (task bqimpnxwn) still holds the mutex while its tab runs year44, so maps-d6's exited;
  start it again once year44 reports and that tab closes.
- notes/T-4214.md: stage A's section (stationclick, 11-25) sits uncommitted, not ours; keep it.
- `land` (not `finish`) conflicts on changelog/next.json when the tree has items: use `finish` (it moves them).
  `finish` drops a tree's edits to existing changelog items: redo them in the main tree.
