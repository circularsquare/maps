# Orchestrator handoff, trainworld

Orchestrator: session 1a8ad997-acf4-4a89-963d-166c67fedf88, name maps-65 [fd56e0], since 2026-11-20 18:52 UTC
Updated: 2026-11-20 20:30 UTC

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`. This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
0. Live v0.4.4. Cap 14. 8800 = 78c7f12f (rebuild 20:55 UTC; smoke passed). Master pushed through 78c7f12f.
1. Evening update, maybe tomorrow (her call, needs her playtest). Publish skill (sort, highlights, playtest list,
   translations incl. T-3822/3823 hovers + addCity + today's strings, zones image for PK/BD, smoke8800). Server
   first, her yes: shonei-server 2754ba1, fd294e7, 43360be, 39e1bd2 (not pushed; T-3735 landed so 39e1bd2's test
   passes). After publish: PR 24 reply (notes/T-3659.md, her OK), open-source sync, then land binzip.
2. Agents (mine unless noted): tramrail T-3889 done (strings to her), checker tramrailchk running; its two
   clock.worker.ts one-liners will conflict with the clock split at landing (addLine/copyLine moved); walkab T-3888 stopped on the rule: Houston test A 79 with all
   three; ask a-2026-11-20-8 (options 1-4); tree walkab holds all three as --set switches, defaults = her three, don't
   land until she picks (then docs, WASM, smoke, checker); fullrate T-3655+T-3646 (checker before landing);
   drawsplit T-2374 renderer piece; binzip T-2486 done + checker's 3 fixes done (tree binzip, ready), land AFTER the evening publish;
   frontier T-3887 (maps-b4's, reports via inbox; her new axes relayed via maps-b4).
3. Then: T-3856 plain buses (after frontier's chart), T-3378 BRT design to her, ferry ideas (her 20:25 ask);
   cleanup table notes/review-cleanup-2026-11-20.md: T-1183 and tools.ts station split after tramrail,
   demandClient/demand.worker splits after binzip, T-2843 demand-health piece after the demandClient split.
4. T-3395: run `tools/ops/save_health.py` after the 11-21 04:00 UTC scan; clean = close it.

## Her words
- 11-20 ~21:00 UTC: ferry types yes, a fare multiplier per type in advanced fares; fans slightly up, all purposes,
  half for work/medical/airport; check her Okinawa line after the islands fix first; wants revenue/cost of a full
  ferry vs train/bus/tram on long trips at China fares (all sent to ferrydesign). T-3889 string -> "Trains can't run
  on tram track." (done in the tree).
- 11-20 ~20:25 UTC: ferries with buses: long-distance ferries more traffic, fans stronger/more common, small and
  large boats (current = medium).
- 11-20 ~20:15 UTC: frontier chart axes: x trip length, z stop spacing, y share by mode (T-3887 Goal).
- 11-20 ~20:10 UTC: likes the bus stop look. Edit speed paused (T-3550, T-3804 P4) "until people complain again";
  board Ready columns merged (done).
- 11-20 ~19:45 UTC: out ~5 h (back ~00:45 UTC), remote control here only, can't playtest; after a handoff the old
  tab relays her messages. "if you find idle resources / compute ... please pick up additional tasks ... mostly
  self contained and can be done without too much input."
- 11-20 ~19:35 UTC: walking bonus 0, 2 min entering the first train station + 2 leaving the last (T-3888).
- 11-20 ~19:15 UTC: tram joins rail, "trams can go on train track but trains cannot path onto tram track" (T-3889).
- 11-20 ~19:10 UTC buses: plain bus no boarding cost/85%/near-free/0.8x fare (T-3856, after the chart), BRT hybrid
  (T-3378). Her words in those tasks.
- 11-20 15:20 UTC: cleanup tasks, then "full track rework" (T-3444 design talk).
- 11-20 ~17:45 UTC: PR 24 reply (notes/T-3659.md) "looks good": post it after the open-source sync.
- 11-20 ~14:00 UTC: no minor track work (T-3662, T-3353 wait on the rework design). 400 km per ferry leg stays.
- 11-19: "player complaints take precedent over other new features"; full playtest list before every publish.
- When she plays games: no uploads until she says she's done; she may ask for a lower core cap.
- Standing: lean testing, smoke per batch; long jobs in the background, answer her first; every publish, deploy and
  push to the website needs her yes in this tab, none overnight; advisor trial (log calls in handoff-log).

## Waiting on her (asks.py list has the open ones)
- She sends the T-3635 player reply herself. Text ack for T-3890's four strings (asked 20:00 UTC).

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine: notes/T-2380.md, notes/T-3382.md.
- DECISIONS.md holds binzip's T-2486 line uncommitted until binzip lands.
- A yes she gives in another session's tab doesn't pass the permission check here: ask again in this tab.
- Never delete a tree folder by hand (its app/node_modules junction deletes the main tree's).
