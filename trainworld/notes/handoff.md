# Orchestrator handoff, trainworld

Orchestrator: session 72757b8d-c220-47a4-ac30-041ad0a7ac39, name maps-3a [8d9b93], since 2026-11-26 14:17 UTC
Updated: 2026-11-27 04:35 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5 (write dates to her in her time). Core cap 14 (her word, 03:32 UTC).

## Next
1. 0.4.7.1 live 11-26 17:07 UTC (her yes): 0.4.7 (058b348af) plus T-4367's fix, app only, data R2 2026-11-25,
   build mwxs0pfz, website fc62297, verify --page all PASS; branch tw/hotfix471 (a71808780) on origin, open-source
   synced; master has its entry (6692949a3). Filed T-4372 (smoke with phone settings, her "should have caught").
   Open asks: crash counts (a-2026-11-26-6); T-4372's start and the 11-25 wind-down unanswered.
2. Landed 11-25 (8800 98aedf0d0, smoke passed): T-4348 .. T-4281 s3 (ARCHIVE). Before the next publish: a Claude batch
   review of d441fb19f, 914f564e6, b6aa1b90f, 67426dfa8, 6a0cab532 (Gemini out until about 11-30).
3. Region redraws done (T-4364, T-4365 archived; 5b1fab73d): all her picks but Poland, IL rejoined for seams, place
   names refreshed, c007. 8800 at 5b1fab73, smoke passed (12 checks). The undo backups data/work/pre-T4364* (19 GB;
   49.9 GB free) stay until she has seen the zones on 8800; delete them 11-28 if nothing comes up.
4. Then T-4153 s10, T-4309, T-4314/T-4315. Up next: T-3160, T-3838.

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
- Undoing a redrawn country, if she finds one wrong: notes/T-4364.md "The job" (restore.py T4364<cc> pre).
- Uncommitted, not ours (an 11-23 checks reorganisation): SPEC.md "What is covered", docs/code-tour.md,
  docs/interface.md, checks/old moves, checks/INDEX.md, tools/checks_index.py. Landings touching them: patch dance
  (`git diff --output=<f> -- <files>`, checkout, land, `git apply --3way <f>`, `git reset -- <files>`).
- Uncommitted on purpose: TODO.md (hers), notes/asks-log.jsonl, notes/board-live.json, changelog/roadmap.json,
  maps .claude/settings.json, notes/T-3397.md, tools/ops/archive_bad_saves.py.
- Trees kept: _track (tw/track), tracksnap (snaptrack.py), _seed (Rust cache seed). hotfix461 removed 11-25 (branch on
  origin).
- notes/T-4214.md: stage A's section (stationclick, 11-25) sits uncommitted, not ours; keep it.
- `land` (not `finish`) conflicts on changelog/next.json when the tree has items: use `finish` (it moves them).
  `finish` drops a tree's edits to existing changelog items: redo them in the main tree.
