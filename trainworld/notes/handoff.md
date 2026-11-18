# Orchestrator handoff, trainworld

Orchestrator: session a3a433e6-4dfe-47ce-ad1a-12e81c692557, name maps-56, since 2026-11-18 16:08 UTC
Updated: 2026-11-18 16:20 UTC

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status` (T-3667). This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
1. Adopted from maps-4f (reports arrive through the inbox): sideplat2 T-3672 (P1, her Caracas save
   Downloads/trainsworld-caracas-day-53.save; checker before landing, WASM rebuild), tripswithin T-3509 (demand
   change: checker before landing), corridor T-2136 (smoke + checker), regate T-930/T-3656 (land after dayrun's
   join), dayrun T-3671/T-3660/T-3517 (main tree, 6 cores).
2. Mine, started 16:20 UTC (Sonnet, trees): onewaytext T-3676/3677/3678 (one-way warning and question strings,
   3 untranslated keys; Vite 6150); checkfix T-3663/3679/3617 (three stale or failing checks; Vite 6151);
   commload T-3680 (P1, her pad: community cards say Could not load too soon; Vite 6152; app-only, no server deploy).
3. dayrun: until kit dayrun.state (notes/T-3671/kit) says "PACKS FREE" (about 19:15 UTC) hold anything writing
   data/packs or data/terrain, including 8800 rebuilds. Gates serve a frozen copy, so app/ and WASM landings are free.
4. 8800 rebuild once PACKS FREE: 8800 is 866e3212, lacks T-3673's one-way arrows (3210be41) and T-3666; the freeze
   20261118-065950 predates T-3623's Cairo places, so no frozen build. Fold in whatever lands before then.
5. T-3395: close after a day of clean daily scans (deployed 11-18 13:30 UTC; she reads the scan's emails).
6. Then: T-1080 (waits for a world run hitting the commit limit), T-3160 phone memory once corridor (T-2136) lands
   (both memory; avoid measuring under each other), T-3620 (Caribbean, Bolivia, Paraguay) only when she says.
   T-2216 islands needs a design to her first (a point past which transit isn't taken is a new demand rule).
7. Play/wait tools (T-3666) landed: CLAUDE.md now says smoke is waited on through its log (an agent saw shell
   calls moved to the background at 180 s). Watch whether agents' smoke runs still behave.

## Her words
- 11-18 16:25 UTC: close easy high-priority wins and old tasks, small over design-heavy, to get the open count
  down; retriage clear miscategorisations; "prio should take into account easiness as well as impact". Triage
  agents triageA/triageB write notes/triage-2026-11-18-{a,b}.md (Close, Small wins, Needs her, Reprioritise); I apply
  closes and priorities in one batch, queue small wins in Up next and start them. She is out walking, on remote control.
- "lets do them all, can start in whatever order you think makes sense" (workflow, 11-18; all seven done).
- 11-18 ~15:20 UTC: mostly hands off today (headache): ask her nothing unless it truly needs her; short messages.
- Daytime builds on part of the cores when compute is free, thinking-heavy tasks beside them (11-18).
- Caribbean+: "we can worruy about caribbean+ later ... i will lyk" (11-18).
- Free-fare leaderboard cheese (T-3657): "needs more thought, lets put it off til later".
- caelum418's speed PRs (T-3659): "look at the next time we look at this area of performance".
- She sends dattera's reply herself (T-3647). Server log: left to the daily scan's emails.
- Standing this week: small isolated fixes, count down, merge duplicates; she edits each update's changelog on the
  board before publish (highlights over ~20 items); lean testing, smoke per batch; one phrase per wait (loads /
  shows up / calculating trips / memory); long jobs in the background, answer her first, no windows on her screen;
  every publish, deploy and push needs her yes to that action, none overnight; advisor trial (successors with
  `next_orchestrator.py trainworld --advisor fable`, log calls in handoff-log).

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json; maps .claude/settings.json (her edits + T-3670 keys).
- Old trees shenzhen (dirty pipeline/osm_units.py, 11-16, owner unknown) and tt2 (stale, has a stray cores.json):
  ask before removing.
- Overnight undo points data/work/pre-T3455/{pre,prejoin} and pre-T3455my/prejoin: delete once she has played the
  new zones on 8800.
- Changelog has 26 unpublished items; a publish needs her edit on the board first.
