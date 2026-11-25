# Orchestrator handoff, trainworld

Orchestrator: session a79cb0e8-7f65-4078-93d4-689df5d0c8b6, name maps-96 [c6b101], since 2026-11-25 16:42 UTC
Updated: 2026-11-25 18:51 UTC (handing over to standby maps-d6 for her remote access). Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5 (write dates to her in her time). Core cap 14 (her word, 03:32 UTC).

## Next
1. 0.4.7 published 18:42 UTC (R2 version 2026-11-25, website 0497af6, export b89a791; bench marked published). Live
   smoke: every game check passes; its one console error is Cloudflare's analytics beacon blocked by the page's CSP
   (asked her). Old trees: hotfix461 can go.
2. Running: year44 (Opus, T-4348): a player's slow edits on a 44-year save (Downloads/trainsworld-polska-day-15922.save,
   player's: stays on this machine); tree year44; time-boxed about 1.5 h; "if nothing pops up no need to chase forever".
3. buyfix: a fresh agent on the buytrains tree: review30's finding 1 (edits.rs:84, a shortfall's cars free), findings
   2, 4, 7, and a short notice for trains bought after a click, like other money notices (her yes). Then land T-4272
   (also does T-4339). Review: artifacts/trainworld/reviews/review30-buytrains.md.
4. Then T-4337; on the quiet machine `edit_bench.mjs --save asia,nyc --baseline` (T-4293). Player reports on 0.4.7
   first. Edit-speed stages (T-4309, T-4314/T-4315, T-4153, T-4149 s3, T-4281), T-4220 s9, T-4230, T-3160 wait for
   her return. Up next: T-3160, T-3838.
5. Standby: maps-3a is the next cycle's (notes/standby.txt); none left: ask her relay session for one. Gemini's quota
   is out until about 11-30: Claude reviewers meanwhile.
6. Resources 11-25: Resend over (her: nothing), R2 storage 24.4 GB (purge of 2026-11-18b, 2026-11-19 asked), Class A
   91% of the free million projected. Class B and the repo size are known to her.

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
- Trees kept: _track (tw/track), tracksnap (snaptrack.py), _seed (Rust cache seed), hotfix461 (0.4.6.1 as shipped,
  branch pushed; remove with agent-tree.mjs remove --force when done with it), buytrains (unlanded).
- reserve_cores: maps-96's loop (task bqimpnxwn); it dies with this tab: the successor starts its own.
- notes/T-4214.md: stage A's section (stationclick, 11-25) sits uncommitted, not ours; keep it.
- `land` (not `finish`) conflicts on changelog/next.json when the tree has items: use `finish` (it moves them).
  `finish` drops a tree's edits to existing changelog items: redo them in the main tree.
