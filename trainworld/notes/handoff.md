# Orchestrator handoff, trainworld

Orchestrator: session 9036b094-ff61-4bec-9dce-e9d9c5d30c00, name maps-cb [60eeab], since 2026-11-22 16:42 UTC
Updated: 2026-11-23 02:50 UTC (ready for 11-23; nothing running, usage exhausted until the reset ~10:00 UTC)

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`; diary notes/handoff-log.md. Her time UTC-5.

## Next (11-23, after the usage reset)
1. PUBLISH her update "probably tomorrow evening" (11-23): run the `publish` skill early in the day. Landed since
   v0.4.5 and checked (batch 13 on 547389c2; batch 14 and 15 on master 5e4ac09f): T-4041 riders take the line that
   arrives first (0.75 softening, late joins), T-4050 evenly spaced shared platforms, T-4054 stage 1, T-1354, T-4040,
   load bars (skew + fill at end). Before it: fresh freeze, translations (T-3986, T-3991, T-3999, T-4003, T-4010,
   T-2264, T-3933, T-3378 s4, T-4013, T-4005, T-4040, the new zones' names: place_names.mjs --zones), smoke8800 on a
   quiet machine, the changelog review (46 unpublished; c051 is T-4041's). Then her text OK and her yes, in this tab.
2. T-4054 (P1, Opus): the solve is ~+40% since T-4041 (real save ~41 s -> ~58 s native), accepted for the update;
   her: "we can look for further speedups elsewhere, tomorrow". Notes/T-4054.md: no-late-joins floor is +23%
   (T-4041 said +13%, gap unexplained), pools of 3+ lines ~600 ms/search, a faster `pool_cost` 2-3 h, u16 shares.
   "Elsewhere" = anywhere in the solve, not only the new rule: measure first.
3. T-4051 (P2): opening a big city on her large North America save waits ~3 min at 90% (Medellin, Managua; Puebla
   fast). Her guess: "something where we're failing and retrying". After T-4041, her word.
4. T-4055 (P2, 1-2 h): a two-zone edit's first final is stale (up to 6% low until the next edit); older bug.
5. New usage week: back to 4-12 agents unless she says otherwise. Resume the paused trees (plans in each tree's
   notes; tags {after:next week's usage}): hokkaido T-3688 (join done, after-steps left in its task; data/packs lacks
   sor/ibr/okh/ksr until it lands), clockmem T-3939, onewaylag T-4045 (memo fix wants a checker), addtrack T-4024,
   stationids T-4043, timetables T-3346, coreslider T-3932, perfledger T-4025, split6 T-3373.
6. After hokkaido lands: sync-track, 8800 = tw/track tip as her preview (her yes 11-22), a batch checker on master.
7. TRACK (her top priority): plan notes/T-4030.md; g (T-4037) next with her calls in the task, then 4d, 4f, 5a/5b.

LIVE v0.4.5. 8800 = fa196b91. Cap 14 (her daytime pick). Cores 0-1 hers (`reserve_cores.ps1 -Loop 30` at
takeover). Server deployed 3b14158. Save level 38 on master. Filed 11-22 from her pad, not started: T-4046-T-4049,
T-4052, T-4053 (a new demand factor: design to her first).

## Her words
- 11-22 evening, T-4041: lines independent ("every line running at its own random times is probably correct");
  softening "75% of the way" (a quarter take the first train that can still make it); keep the cliff fix ("i do
  want the fix actually"); the first-train quarter stays agnostic about changes; the slowdown is "okay". T-4050:
  same track + same stops + no passing = evenly spaced, per shared stretch ("players dont have any tools to
  manually space out lines"). Load bars: what read 90% now reads half full (her idea). Her hope: "buses get more
  reasonable values relative to trams ... more uniform ... cuz everything is parallel".
- 11-22: China shape fix next week (T-4044); Hokkaido via the old Japan tool; a yes in another tab doesn't count here.
- 11-21: main priorities the track rework, finish and polish BRT; T-3742 planned, start later. Standby
  orchestrator tabs (notes/standby.txt).
- 11-19: "player complaints take precedent over other new features"; full playtest list before every publish. When
  she plays: no uploads until she's done; she may ask for a lower core cap.
- Standing: lean testing, smoke per batch; long jobs in the background, answer her first; every publish, deploy and
  push to the website needs her yes in this tab, none overnight; advisor trial (log calls in handoff-log); BRT
  stations for BRT lines only; never ask again whether big saves get more workers (11-21).

## Waiting on her
- No open asks. She sends the T-3635 player reply herself. Quote text in the message, never send her a notes file.

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine: notes/T-2380.md, notes/T-3382.md,
  tools/feedback-inbox/builds.json, tools/release/new_zones_map.py, notes/T-3821.md, notes/T-3896-report.png.
- .tw-trees/arrivefirst is a leftover folder, not a worktree; a node_modules junction may be inside: never delete it
  by hand (agent-tree.mjs remove refused it; check for the junction first).
- Untracked working files: notes/T-3688/kit (the redraw scripts), notes/T-3444-design/tasks-to-file.md.
- A task with stages left: `agent-tree.mjs land` + `remove`, not `finish` (it archives); swap the taking mark by hand.
