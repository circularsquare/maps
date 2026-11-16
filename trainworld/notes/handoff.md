# Orchestrator handoff, trainworld

Orchestrator: session f7c8626c-6b89-4b09-983c-732243e787e9, name maps-d7 [e8137b], since 2026-11-16 15:36 UTC
Updated: 2026-11-16 18:05 UTC

A short state page, rewritten (not appended) as things change. The diary is notes/handoff-log.md; older detail is in
git history. Read trainworld/CLAUDE.md and docs/orchestrator.md first.

## State
- Live = v0.3.5 (maps bfe58c54, published 11-15). Data 2026-11-13 freeze 20261113-013559 on R2.
- master 516d0dd5, pushed to cefc5ac1 (push after the next batch). 8800 = cefc5ac1 on live data/packs (Shenzhen
  rebuilt 11-16; no newer freeze: the Guangdong join's market trips register timed out, not redone).
- Uncommitted on purpose in the main tree: T-3399's country plan edits (pipeline/world_plan.py/.json, realms.json,
  zone-names.json, docs/data-register.md, notes/T-3399.md): commit when T-3399 starts. Also not ours: TODO.md (hers),
  notes/T-3395.md, T-3397.md, jordan-save-investigation-2026-11-15.md, tools/ops/archive_bad_saves.py (saves session).
- Board inbox watch: Monitor on `venv/Scripts/python.exe -u trainworld/board/watch.py` (30 min, re-arm).
- Server shonei-server: T-3402 + T-3440 deployed and pushed 11-16.

## Running agents (maps-d7)
- names acde25cc42203d034 (Opus, main tree, i18n place-name files): T-3450 Shiga and a Wikidata match guard; told
  to hand-check only prominent places.
- regions ac47cdc31cb8469fc (Opus, .tw-trees/regions, 6181): T-3448 one schematic per region.
- feedsample aca180a7fbc3e6ae7 (Opus, .tw-trees/t3454-base reverts c9db1359 for "before", 6196/6197): T-3454.
- flow-board a6ef6e2aa7e353ca5 (Sonnet): asks on the board (notes/asks.json), Someday -> P4, P4 not counted.
- flow-tools afb0d935b17ce763c (Sonnet): tools/tasks.py start/done/close/check; agent-tree JSON handling; T-3395 format.
- Landing: restore app/src/game/*.json in a tree before agent-tree land (until flow-tools automates it).

## Waiting on Anita
- Auto-starting the next orchestrator (T-3451 rec 6 replaced by her question): see the latest chat; nothing built.
- Player-facing explanation of the feeders fix: drafted in chat 18:05, her wording call.
- T-3399 Central America + Venezuela: starts only on her word ("i will let you know when its time").
- T-3455 coastal gap scan: queued for an overnight run on her word.

## Her standing asks this week
- Small isolated fixes, open-task count down, merge duplicates; bug investigations give up if hard.
- Quad paused (waits on T-3444). Pack rebuilds (region redraws, data fixes) batch into overnight runs.
- No windows on her screen: background processes hidden (start_hidden.ps1, CREATE_NO_WINDOW).
- Manual agent checking only for prominent cases; prefer automatic checks (11-16, on T-3450).
- Every publish, deploy and server push needs her yes to that exact action; none overnight.

## Next
- Land the five agents above as they report; smoke after each batch; 8800 rebuild; push.
- Then more small fixes from P2/P3 bugs (ready, no waits); her review calls on T-3451 recs as they come.
