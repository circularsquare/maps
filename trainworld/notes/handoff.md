# Orchestrator handoff, trainworld

Orchestrator: session 79e26413-4648-481b-ba5a-8ddcd716a0f0, name maps-60 [6ba449], since 2026-11-20 05:20 UTC
Updated: 2026-11-20 13:20 UTC (her computer restarted ~13:00: every process died; 8800 serve restarted)

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`. This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
0. Live v0.4.4. Cap 14. 8800 = 7637e0e0 (smoke passed). Since: T-3846 landed (b9874d89, not on 8800). Open in
   chat: 400 km per leg (T-3809's strings change with T-3845).
1. Running, mine: gapfill (Opus, T-3764, main tree: JOIN of 338 PK+BD rows from ~13:20, pid 9092, log
   artifacts/trainworld/t3764-join.log; no data/packs writes or plain 8800 rebuild until it ends; its uncommitted
   pipeline, checks, game json, notes files are its own), riverfix (Opus, T-3839, 6152), ferryorder (Sonnet, T-3845,
   6153), islands1 (Opus, T-2216 stage 1, 6155; fresh Sonnet checker before landing), legacytime (Sonnet, T-3847,
   6156), todoreview (Sonnet, main tree TASKS.md only). One tree each.
2. T-3395: 11-20 scan (pass 1, not done) found 0 new damage. Run `tools/ops/save_health.py` after the 11-21
   04:00 UTC scan; clean = close it (the two rev-1 games of ue83342ed78f8fc4c are old 11-19 damage).
3. Up next: T-3804; T-2216 stages 2-3 after stage 1 (stage 0 is T-3838); after gapfill: T-3838, T-1376 retry, T-3843, rest of T-3764;
   T-1738 Bhamo; T-3836 (design for her).
4. Next update: unpublished changelog c001-c015 plus tonight's; untranslated: T-3822's and T-3823's hover strings,
   diagramView.addCity; use the publish skill (review agent, sort, playtest list, translations, new zones image).
5. Leftover trees ttchk2base, touchrule, timekeys. T-3781 waits for the next report's diagnosis. Steam (T-3729): her screenshots in artifacts/trainworld/steam.

## Her words
- 11-20 ~12:45 UTC, morning: new countries "look pretty cool"; rivers still break (T-3839, her words there); ferry
  order port, waypoints, port (T-3845); quieter related lines (T-3846); islands "3x, A", stage 2 yes.
- 11-20 ~13:40 UTC: T-3835 text ack. TODO.md review: file everything in "for anita" and "other player
  suggestions" not done or filed ("basically want all of this eventually"; agent todoreview, ids T-3850-T-3899,
  table in notes/todo-review-2026-11-20.md). New players default to legacy time buttons (T-3847, agent
  legacytime). Asked which Ready (low) tasks to raise to high: my picks went to her in chat.
- 11-20 ~05:20 UTC, to bed: "lets try to keep going through the night, continuing to run stuff if resources free
  up". Picked rivers E (+ min-straighten, round caps; coastline T-3833 after); OK'd both global CLAUDE.md wordings.
- 11-20 ~04:20 UTC: rivers: likes B's look, wants C's correctness: "C but just do like microstraightening ... and we
  could also round corners"; wants a zoomed-out Hong Kong Island frame. CLAUDE.md review: all changes yes; board
  changelog review is NOT the publish text OK; "next up" bullets occasionally and always after shipping an update.
  Countries overnight ("do we have new countries prepared for overnight run? lets maybe start work on that now").
- 11-20 ~03:10: T-3822 text ack; intercity effective time "keep it as is"; packing backups filed for later (T-3821).
- 11-20 ~01:55: R2 11-13 purged; Resend stays free.
- 11-20 ~00:55: real-network fixes and additions P4, Japan P3, one-city changes low (LINES.md).
- 11-19: "player complaints take precedent over other new features" (actual bugs). Before every publish, a full
  playtest list. Fares by city: country or city multiples, never both.
- 11-18: close easy high-priority wins, small over design-heavy. "less willing to put stuff on the changelog".
- Share previews: free Worker, careful with the Cloudflare account. Mod support (T-3377): never say publicly it's
  aimed at vibe-coders. caelum418's speed PRs (T-3659): next time we look at performance.
- When she plays games: keep 4 cores free (cap 12) and no uploads until she says she's done.
- Standing: lean testing, smoke per batch; long jobs in the background, answer her first; every publish, deploy and
  push to the website needs her yes in this tab, none overnight; advisor trial (log calls in handoff-log).

## Waiting on her (asks.py list has the open ones)
- a-25 rivers pick (now superseded by prototype 4's sheet). She sends the T-3635 player reply herself.

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine: notes/T-2380.md, notes/T-3382.md.
- A yes she gives in another session's tab doesn't pass the permission check here: ask again in this tab.
- Never delete a tree folder by hand (its app/node_modules junction deletes the main tree's).
- A stale .git/index.lock with no git.exe running can be removed (one appeared 11-20 04:12 UTC).
