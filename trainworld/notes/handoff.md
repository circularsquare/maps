# Orchestrator handoff, trainworld

Orchestrator: session a6a6b4a3-3cb7-4e31-ac63-f1cd25a92ff1, name maps-60, since 2026-11-20 01:43 UTC
Updated: 2026-11-20 04:35 UTC (all running agents are maps-60's; guianas' session ended, carwrap finishes its work)

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`. This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
0. Live v0.4.4. Cap 14. 8800 = master b60cbea7 era (check status); master pushed. Landed today: T-3812, T-3820,
   T-3805, T-3822, T-3803 option 3, T-3817, T-3819, T-3818, T-3825, T-3826; T-3790 shipped (mos, ber, hon; sai kept).
1. Running: editfirst (Opus, T-3803 option 1, tree editfirst; checker before landing; also the player's stop-click
   burst), riverroute (Opus, T-3358 prototype 4 "micro-straightening" sheet + Hong Kong Island frame, scratchpad
   only), mdreview (Fable, applying her CLAUDE.md picks; then a reviewer agent on its rewords, rewords.md in its
   scratchpad), weekchk (Sonnet, T-3827), firstwait (Sonnet, T-3823), ferrywp (Opus, T-3809, smoke at end),
   carwrap (Sonnet, T-3762 checks; then commit its file list), seasia (Opus, T-1738 overnight; message it
   "T-3762 committed" before it touches game json or data/packs).
2. T-3395: after 04:00 UTC run `tools/ops/save_health.py`; the two rev-1 games of ue83342ed78f8fc4c still corrupt =
   old damage, close after a clean 11-21; intact = the scan's check lagged (notes/T-3395.md).
3. Up next: T-3763 (Iraq, Iran) after seasia's join; T-3804 after editfirst; T-3824 and the river build after her
   pick; T-3632 islands; T-3828, T-3829-T-3832 (P3).
4. Next update: unpublished changelog c001-c015; untranslated: T-3822's hover strings, diagramView.addCity; before
   publishing: changelog review agent, sort, playtest list, translations, new zones image (Caribbean, SE Asia).
   After publishing: a few bullets of what next, and start it.
5. Leftover: stale Vite pid 33736 on 6181 (leave), trees ttchk2base, touchrule, timekeys.
6. T-3781 waits for the next report's diagnosis. Steam (T-3729): her screenshots in artifacts/trainworld/steam.

## Her words
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
- CLAUDE.md changes staged (reviewed): on her OK to the For you and push wordings, run
  `python <maps-60 scratchpad>\mdreview\apply.py --yes` (dry run without --yes), then commit the maps paths.
- a-25 rivers pick (now superseded by prototype 4's sheet). She sends the T-3635 player reply herself.

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine: notes/T-2380.md, notes/T-3382.md.
- A yes she gives in another session's tab doesn't pass the permission check here: ask again in this tab.
- Never delete a tree folder by hand (its app/node_modules junction deletes the main tree's).
- A stale .git/index.lock with no git.exe running can be removed (one appeared 11-20 04:12 UTC).
