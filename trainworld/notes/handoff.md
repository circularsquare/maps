# Orchestrator handoff, trainworld

Orchestrator: session ca3a5e8b-220b-4b3d-a5e9-87f500e687dc, name maps-98 [60f1c2], since 2026-11-21 12:25 UTC
Updated: 2026-11-21 16:15 UTC (cycling to a successor at 500k; she is up and at this tab)

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`; diary notes/handoff-log.md. Her time UTC-5.

## Next
0. LIVE v0.4.5 (11-21 ~04:15 UTC). Cap 14 by day. Cores 0-1 are hers: start `reserve_cores.ps1 -Loop 30` at takeover
   (docs/orchestrator.md). 8800 = 7e04ea1f. Next update: translations nearer publish (T-3986's new keys too).
1. Batch 9 on master, not on 8800, unchecked: phonemem2 49a4142e, autostops 94832b99, dropindent d80eaa39, splitb1
   2382b9b8, brtstations 49fa812e (save level 35: tell her a save made on 8800 may not load if reverted). maps-98's
   land3.ps1 (scratch) lands editfail T-3945, islandfix (check only), viewonly T-3851 in that order; check `git log`.
   Then: batchchk9 (Sonnet; brief like batchchk8, notes/batch-2026-11-21.md "Batch 8") + smoke + snap8800.
2. Running (adopt from the board): ownwaits T-3931, fullrate4 T-3655 (+T-3941), clockmem T-3939, keihan T-3935,
   brtblueprint T-3944, busreach T-3934, splitb2 (T-3373 B stage 2), modeldown T-3986.
3. Start next: one Sonnet agent on T-3990 + T-3991 (Community: Discord and leaderboards links; Report as a flag icon
   with a confirm popup) once viewonly has landed; T-3948 (track stage 1, T-1183 folded in) once ownwaits and
   fullrate4 land; T-3378 stage 4 after T-3944; T-3933/T-3932 later; T-3937 needs a quiet machine. Keep 4-12 agents.
4. T-3395: save_health.py after the next scan; clean = close it.
## Her words
- 11-21 ~16:10 UTC: Report on a community save is "a bit of a killjoy": a flag icon, and a confirm popup "Report this
  save? [reason] cancel submit" (T-3991). Discord and leaderboards links in Community (T-3990). itch.io $0-or-donate,
  $5 suggested (T-3988); Ko-fi hiding dropped. Light view: money frozen (built), "text ack" for viewonly, BRT
  station text ack.
- 11-21 ~15:55 UTC: track rework all calls made (notes/T-3444-design/answers-2026-11-21.md): no passing-track
  purpose; added tracks stay unconnected with a "Build crossovers?" offer. 38 stages T-3948..T-3985 on tw/track.
- 11-21 ~14:05 UTC: her "no more compute" at 13:20 was a moment, not a standing hold; keep 4-12 agents. Count every
  Auto stop ("if people are seeing bus stops the bus stops have to be real"; optimizations elsewhere: T-3934).
- 11-21 ~12:50 UTC: BRT design yes, BRT stations for BRT lines only; upkeep 0.2%; tram brake 0.9; T-3373 B go, the
  memory limit rises (players set it: T-3932; a load out of memory retries lower: T-3933). Never ask her again
  whether big saves should get more workers ("on the player to decide").
- 11-21 ~02:20 UTC, playtesting plain buses: "buses feel really easy ... 10% ... taking the bus ... 90% driving.
  which feels pretty realistic". Bus speeds set 11-21: plain floor 25, BRT +10, fast roads 1.4x street.
- 11-19: "player complaints take precedent over other new features"; full playtest list before every publish.
- When she plays games: no uploads until she says she's done; she may ask for a lower core cap.
- Standing: lean testing, smoke per batch; long jobs in the background, answer her first; every publish, deploy and
  push to the website needs her yes in this tab, none overnight; advisor trial (log calls in handoff-log).

## Waiting on her (asks.py list has the open ones)
- She sends the T-3635 player reply and the T-3945 reply herself (draft in notes/T-3945.md).

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine: notes/T-2380.md, notes/T-3382.md.
- Tree .tw-trees/autostops is gone; .tw-trees/busreach changes only a constant for measuring: never land its code,
  only its notes.
- A yes she gives in another session's tab doesn't pass the permission check here: ask again in this tab.
- Never delete a tree folder by hand (its app/node_modules junction deletes the main tree's).
- Commit messages through PowerShell: single quotes, or "$2M" becomes empty (49fa812e lost "US$2M").
