# Orchestrator handoff, trainworld

Orchestrator: session caa91e48-7288-4753-9895-c2b79ad5e3d9, name maps-c5 [fde5f5], since 2026-11-25 02:20 UTC
Updated: 2026-11-25 03:19 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5. Core cap 14 since 03:32 UTC (her word). Start more when agents + claimed cores / 2 < 8.

## Next
- Gemini quota ran out 03:43 UTC, resets about 04:57: then rerun gemini_agent t3303fix (T-4233, tree kept) and the
  T-4242 design draft (artifacts/trainworld/gemini/t4242/brief.md). Claude agents meanwhile.
0. Usage (her 11-25): Claude runs out before Gemini. Claude agents take P1/P2 code work; small clear tasks and PR ports
   go to Gemini code agents (`tools/gemini_agent.py start <name> --brief <f> --task T-x --new-tree --cores 1`); reading
   and batch or design review go to `gemini_audit.py`. Stay light (notes/repeats.md): briefs in files, report heads only.
1. Running (briefs in my scratch <agent>/brief.md): slowfocus (Opus, T-4263; then T-4264 the bars as its follow-up),
   fablereview (Fable at her ask, T-4267 design review to artifacts/trainworld/t4267-fable-review.md; its findings then go
   to a fresh Opus agent: editdesign is at 411k), rowsmerge (Opus, merges master into tree rowsedit + Gemini's 3
   findings, then `land rowsedit` + `remove`), smokefix (Sonnet, T-4270: t133, t144, t3192 fail on master), Gemini
   pr34port (T-4258, NYC figure diff to her before landing). Adopted: cutstretch3 (T-4153 s6), uploadedit (T-4262),
   busrebuild (T-4249): `land` + `remove`; they become T-4267's first steps.
2. T-4267 design approved by her pending both reviews (Gemini's answered in its section 10; about 38 h, up to 55). After
   Fable's is answered: tell her what changed, then start part 1's first stage (P1). No other edit-speed starts.
3. 8800 = 57eb2dbdd (smoke failed: T-4270 and the wiki, rebuilt). Master fa9105102 + bookkeeping 140252c17, pushed.
   Smoke again after T-4270 lands. Batch review due over T-4239, T-4256, T-4255, T-4220 s3, T-4266, T-4149 when landed.
4. Next starts (not edit speed): T-4257 (bus routes by speed), T-4251 (busways), T-4243 speedometer, corridors3
   (T-4199 s2), T-4080 s3-6, T-4230, T-4187, T-3160, T-4233, T-4242 design. Gemini: T-4259 (UI, needs a screenshot).
5. Waiting on her: a-2026-11-24-16 (PR #21/#24 comments); T-3742 after she asks; T-4250 (smooth fade); T-4248.
6. Core cap 14 (her 11-25 03:32). reserve_cores: maps-3e's
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
