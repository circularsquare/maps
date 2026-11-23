# Orchestrator handoff, trainworld

Orchestrator: session 437c6600-7ce7-4abc-98f9-fc664146729b, name maps-65 [c4fa02], since 2026-11-23 14:09 UTC
Updated: 2026-11-23 14:15 UTC. Live state: `venv\Scripts\python.exe
trainworld/tools/handoff.py status`; diary notes/handoff-log.md. Her time UTC-5.

## In flight (adopt from the board; reports arrive through the watch)
- solvespeed T-4054 (tree solvespeed): same-answer speedups + her yes 11-23 to both shifting ones (no late joins at
  change nodes; pool integrals at 2 points; DECISIONS each), then T-4064 (branch riders, her yes). Land on report.
- solveprof T-4057 (tree solveprof) + T-4082 Detroit bus save (community share iemjwzum): profile, then cuts.
- trackg1 T-4037 g1 (tw/track): rebase over Add a track, land --into tw/track; g2 to a fresh agent. addbp T-4085
  (tw/track). busconn T-4079 (master; T-4083 split off). safariwarn T-3940. checkbns: checker of tree busnosnap
  (T-4069, save level 39): land on PASS; shonei-server's save check must accept 39 before any publish (reads 38).
- tracksync: merges master into tree tracksync (copy of tw/track 3c62751b), leaves it staged, no commit. On its
  report: commit the merge in that tree, `git -C .tw-trees/_track merge --ff-only tw/tracksync`, remove the tree;
  then 8800 = tw/track tip (her yes 11-22), and a batch checker on master (crash fixes 840bfa8f, Hokkaido
  522862ff, touch d0a09297, community). The permission check refused editing and committing in _track itself.
- tracksync2 (queued, next slot): merging master into tw/track conflicts in saveFile.ts, save.rs (busnosnap level 39 / LBE1 vs track blocks) and checks/INDEX.md; aborted in _track. Same route as before: a tree copy of tw/track, agent resolves, I commit and ff _track, then snap8800 tw/track.

## Next (11-23)
0. Her goals 11-23: solve speedups (main); loading and crashes (main; T-4071 triage closed 11-23, T-3940 next, phone load is her ask); track rework all week. Resend over its cap is fine. Publish: don't ask when.
1. 0.4.6 plan (her 11-23: "we're not in a huge rush"): wait for T-4064, T-4093, T-4051's master-vs-v0.4.5 zone-open timing (T-1354 corridor book: a blocker if worse), and most of busmerge T-4094, edit speed T-4097 and the lighter phone load T-4086; not the track rework (stays on tw/track). Then 8800 = master for her playtest list, T-4025 bench v0.4.5 vs candidate as the net check. PUBLISH when ready (never ask her when; `publish` skill). Before it: a batch checker on everything since 11-22,
   fresh freeze, translations of every new key (`i18n.mjs`; zone names place_names.mjs --zones), the server's level-39
   save check, smoke8800, the changelog review. Then her text OK and her yes, in this tab.
2. Speed: wins land as they report (real-save numbers; orchestrator.md "Speedups that shift riders"); checker after.
3. Next: TRACK g2-g4 (notes/T-4037-design.md), then 4d, 4f, 5a/5b; T-4080 + T-4083 (one design: bus stops keeping
   identity across edits/saves, after busnosnap); T-4075, T-4051, T-4055; paused trees (onewaylag after T-4054).

LIVE v0.4.5. 8800 = 73299e61. Cap 14 (her daytime pick). Cores 0-1 hers (`reserve_cores.ps1 -Loop 30` at
takeover). Server 79777f5. Save level 38. Pad 11-22, not started: T-4046-49, T-4052, T-4053 (design to her first).

## Her words
- 11-23: 2 crowding rounds refused, final (DECISIONS); stop rounds under 3% yes; merge bus stops within 300 m for
  demand yes; times as percent of the wait. Speed tests on real saves only ("manually built saves for correctness testing"); bus passages save no time,
  connections without one +1 min ("say that building a passage saves one minute"); Add a track as a blueprint, no
  track limit; hold a server deploy while a follow-up to it is running ("we should have just waited for that check").
- 11-22, T-4041: lines independent; softening "75% of the way"; keep the cliff fix; slowdown "okay"; T-4050 evenly
  spaced shared stretches.
- 11-22: China shape fix next week (T-4044); Hokkaido via the old Japan tool; a yes in another tab doesn't count here.
- 11-21: track rework, finish and polish BRT; T-3742 planned, later. Standby tabs: notes/standby.txt.
- 11-19: "player complaints take precedent over other new features"; full playtest list before every publish. When
  she plays: no uploads until she's done; she may ask for a lower core cap.
- Standing: lean testing, smoke per batch; long jobs in the background, answer her first; every publish, deploy and
  push to the website needs her yes in this tab, none overnight; advisor only for big design decisions (11-23); BRT
  stations for BRT lines only; never ask again whether big saves get more workers (11-21).

## Waiting on her
- `asks.py list`: lighter phone load (a-2026-11-23-3), Spain block (-4). All strings so far acknowledged (11-23,
  busnosnap's included). T-3635 reply is hers.

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine: notes/T-2380.md, notes/T-3382.md,
  tools/feedback-inbox/builds.json, tools/release/new_zones_map.py, notes/T-3821.md, notes/T-3896-report.png.
- .tw-trees/arrivefirst is a leftover folder, not a worktree; a node_modules junction may be inside: never delete it
  by hand (agent-tree.mjs remove refused it; check for the junction first).
- Untracked working files: notes/T-3688/kit (the redraw scripts), notes/T-3444-design/tasks-to-file.md.
- A task with stages left: `agent-tree.mjs land` + `remove`, not `finish` (it archives); swap the taking mark by hand.
