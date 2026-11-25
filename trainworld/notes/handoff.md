# Orchestrator handoff, trainworld

Orchestrator: session caa91e48-7288-4753-9895-c2b79ad5e3d9, name maps-c5 [fde5f5], since 2026-11-25 02:20 UTC
Updated: 2026-11-25 02:50 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5. Core cap 10 since 00:52 UTC (her: "for a few hours", "be fairly aggressive
about 10 core cap"); put it back to 14 around 04:00 UTC and tell her. Start more when agents + claimed cores / 2 < 8.

## Next
0. Usage (her 11-25): Claude runs out before Gemini. Claude agents take P1/P2 code work; small clear tasks and PR ports
   go to Gemini code agents (`tools/gemini_agent.py start <name> --brief <f> --task T-x --new-tree --cores 1`); reading
   and batch or design review go to `gemini_audit.py`. Stay light (notes/repeats.md): briefs in files, report heads only.
1. Running (briefs in my scratch <agent>/brief.md): slowfocus (Opus, T-4263 why a focused city waits minutes; then T-4264
   the bars, as a follow-up), fixes15 (Opus, T-4266 batch15 review findings), editdesign (Opus, T-4267 the edit design
   for her; also her new-line/delete-line and PR 29 questions), tramparking (Sonnet, T-4247), Gemini pr34port (T-4258).
   Adopted from maps-3e: cutstretch3 (T-4153 s6), uploadedit (T-4262), busrebuild (T-4249); multi-stage, `land` +
   `remove`. No new edit-speed stages until her answer on T-4267's design.
2. Landing queue: aside3 (T-4220 s3, `land`), pr29port (T-4255 finish --coauthor Leni-Vienne
   <71212417+Leni-Vienne@users.noreply.github.com>), then PRs #29/#30 closed with her OK'd comments (chained in REPL).
   rowsedit (T-4149 stage, `land` + `remove`) waits on Gemini's pre-landing review artifacts/trainworld/gemini/t4149/.
   Landed today: T-4239 (3cc8c5a0d), T-4256 (b6b8bcfdd). Then rebuild 8800 (has a17111304; T-4150 s1 onward missing).
3. Batch review over T-4239, T-4256, T-4255, T-4220 s3, T-4149 when landed; smoke on master after the batch.
4. Next starts (not edit speed): T-4257 (bus routes by speed), T-4251 (busways), T-4243 speedometer, corridors3
   (T-4199 s2), T-4080 s3-6, T-4230, T-4187, T-3160, T-4233, T-4242 design. Gemini: T-4259 (UI, needs a screenshot).
5. Waiting on her: a-2026-11-24-16 (PR #21/#24 comments); T-3742 after she asks; T-4250 (smooth fade); T-4248.
6. Core cap 10 until about 04:00 UTC, then back to 14 (her "for a few hours") and tell her. reserve_cores: maps-3e's
   loop still holds the mutex; start mine (`-All -Loop 30`) once its tab closes.

## Her words
- 11-25 02:2x-02:4x: the 90% bar and ~1 s edits are repeats (notes/repeats.md); don't ask her to test either until we think
  it feels responsive on 8800, mention progress only. New line and line delete should be instant, from the general
  design. Edit-speed memory budget 30 MB (DECISIONS T-4255). Game stays at anita.garden (T-4269).
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
