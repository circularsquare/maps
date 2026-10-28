# Orchestrator handoff, trainworld

Orchestrator: session f28006bc-4241-4116-9970-a65fc772bb95, name maps-a2, since 2026-10-27 23:08
Updated: 2026-10-28 evening (past 500k tokens; ready for a takeover)

Read trainworld/CLAUDE.md and TASKS.md first. History before this session: notes/handoff-2026-10-26.md;
this session's landings are in ARCHIVE.md (T-1561 to T-1691) and `git log`.

## State
- master 99ce4d5e; 8800 runs it (smoke 20/20 light and dark, real save 12,483,326 rail trips).
- LIVE: 2026-10-28c (= 62540fe3), website 8ee048d; R2 holds only trainworld/2026-10-28c. Not
  published: e588aaf2 onward (T-1607, T-1676-1691). Deploy recipe: deploy.py --version <v>
  --smoked from .tw-snap's build, then website add/commit "trainsworld"/push, wait for the
  Pages run (`gh run watch`), `deploy.py --verify --page --version <v>`, then purge the old R2
  prefix. Anita approves each deploy. Website's Pages build needs Ruby 3.3 (fixed bf68979).
- shonei-server 6051eed deployed 2026-10-28 (trainworld feedback route; config shows
  trainworldFeedback:true). Whether feedback mails her is untested: asked her to send one.
- Uncommitted: Anita's TODO.md (never commit), checks/t1223-build.mjs (untracked, T-1223's).
- 8800 rebuild: scratchpad snapbuild.cmd <commit>
  (C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\f28006bc-4241-4116-9970-a65fc772bb95\scratchpad\snapbuild.cmd),
  then smoke light and `SMOKE_SHADE=dark`. Never rebuild .tw-snap while a deploy upload runs.
- Commit messages: single-quote them in PowerShell ("$1" and inner double quotes break).
- Core cap 12 while Anita is up (keep 4 free), 15 overnight.

## In flight (maps-a2's agents, main tree; reports come here; nothing committed yet)
- T-1689 doubling up adds no twin stations + T-1699 station keeps orientation and stays on
  track when its track is removed: Opus, Vite 6157 (agent a1fafa3b0be586fae).
- T-1693-1698, T-1700 UI polish (rides text colour, bubbles #bbb, slider/toggle heights, money
  refusal wording, holiday dates, station sound pitch, loading logo boxes and tagline
  "~welcome to trains world~"): Sonnet, Vite 6152 (a7f3474feb496ec5b).
- T-1692 base map labels without halo and about half as many: Sonnet, Vite 6154 (a1e19845eedcd0994).
- T-1703-1706 mobile (undo cancels the segment being drawn, compact controls, search button off
  the visibility button, smaller fare axis labels, placing popup only check green / undo yellow
  / x red): Sonnet, Vite 6155 (ac4e565578ccbb97d).
- T-1702 bunny curtains, bunny further back, ears half; T-1701 elevation-follow button removed
  (hotkeys wait on Anita): Sonnet, Vite 6156 (abc7a220b91fe1f04).
- When each reports: typecheck, commit its explicit files (shared files: style.css,
  clock.worker.ts, renderer.ts; commit after the sharers finish or check the diff), cargo test if
  sim changed, 8800 rebuild, smoke both shades.

- T-1707 ready (uncommitted: tools/deploy.py, notes/T-924.md; no website or server change
  needed beforehand). Publish: deploy.py --version 2026-10-28d --smoked; git -C ..\website add
  trainsworld/ trainworld/; commit "trainsworld"; push; wait for Pages; deploy.py --verify
  --page --version 2026-10-28d (also checks the redirect); purge r2 trainworld/2026-10-28c.
- T-1703-1706 mobile done (uncommitted: TouchBar.tsx, FareGraph.tsx, style.css, map/tools.ts,
  touch.ts, busDraw.ts, docs). Smoke node checks t133/t144 fail on the polish agent's setDpr in
  game/ui.ts (window at module load); polish agent told to guard it.
- T-1701 (button gone, Shift+1-7 elevation) + T-1702 bunny done (uncommitted: Spinner.tsx,
  style.css, main.ts, toolKeys.ts, BuildTab.tsx, HelpPanel.tsx, game/ui.ts, map/tools.ts, docs).
  T-1692 labels done (uncommitted: map/basemap.ts, docs/ui-map.md; 170 -> 74 labels).
  T-1689/1699 done (uncommitted: sim crossings.rs, links.rs, tests, docs, DECISIONS; a station
  losing its track keeps a straight piece under it). Waiting on: polish (setDpr guard). T-1708
  after the deploy. Then commit all, 8800, smoke, deploy.
- T-1708 filed (auto bridge leaves room for a station beside a station on the crossed line):
  start after T-1699's agent reports (same track code).

- DEPLOY APPROVED (Anita 2026-10-28 evening, then away): once the in-flight agents and T-1707
  land, deploy everything (incl. the trainsworld URL) and leave it. Core cap 15 while she's away.
  Her goal: a public, share-worthy state this week (friends first, not posted yet).

## Waiting on Anita
- Next deploy (everything after 62540fe3) when she says.
- Feedback form test on the live site (does the mail arrive?).

## Anita's calls this session (all in DECISIONS.md or task lines)
- Load-time approximations: lines 10-20% off acceptable for ~50% faster loads, ideally with the
  city in view exact (T-1609); T-782's -23% round skip alone not worth it. Order: T-1609 next,
  then T-1650 (cut big networks into parts; also lets "constructing..." clear per city).
- No speed bought with memory. Deletes undoable. Corners section gone from the blueprint panel.
  Straight-line default fares (US $1 + $0.08/km); Japan's lower long fares fine. Loans 1%/game
  day. Clock speed not saved. No automatic crossovers; stations placed by the player.

## Next
1. Land the five agents above; deploy when Anita says.
2. T-1609 (refine only the zone in view against frozen loads; exact; base for T-1650), then
   T-1650 design. T-1654 (real load-time estimate), T-1634, T-1647, T-1644, T-1520.
3. T-1374 (superholidays stage 3), T-1625 (rebuild the Japan stress save).
4. Rest of the world after calibration (Anita 2026-10-27).
Ids handed out through T-1708.

## Anita's standing asks
- Every demand task (loosely perf) has a "done when"; stop once met. No single-pair demand tuning.
- Performance: approximations welcome when a player can't tell; name them in DECISIONS.md.
- Buses settled: no more tuning unless she asks.
- Thanks to agents in my own reply, not by message; ask after agents quiet 45+ min; final checks
  in the foreground; browser loads count their demand workers in core claims.
- Tokens tight this week (to ~11-01): Sonnet for settled work, Opus for design, data correctness,
  unknown bugs; compute-heavy token-light approaches; investigations stop at a written answer.
- Someday tasks are real open tasks.
