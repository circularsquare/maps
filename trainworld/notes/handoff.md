# Orchestrator handoff, trainworld

Orchestrator: session 79e26413-4648-481b-ba5a-8ddcd716a0f0, name maps-60 [6ba449], since 2026-11-20 05:20 UTC
Updated: 2026-11-20 08:35 UTC (both sessions are named maps-60: message by ref)

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`. This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
0. Live v0.4.4. Cap 14. 8800 = master = 7637e0e0 (~11:15, pushed). Landed
   tonight by maps-60 [6ba449]: T-3809 ferry waypoints, T-3834, T-3832, T-3828, T-3358 rivers E (her look call),
   T-3835 loops (loopchk passed), T-1738 Myanmar/Laos/Timor-Leste (Bhamo left), T-3833 + T-3824 coast and ferries
   (07:30). Filed T-3837-T-3842. Review: review/rivers-e-london.png, review/coast-round-hk.png. Asked in chat,
   unanswered: T-3809 text ack, 400 km per leg, T-3835 text ack. Ask a-2026-11-20-1: T-2216 island design.
1. ~10:50: T-3763 Iraq (38) and Iran (84) shipped and committed (3990153a; its smoke 29/29 on main incl. T-3842);
   T-3842 Burmese labels landed, mki/pgx relax dropped; T-3843 filed. ~11:15: T-3803 option 1 landed after
   editdiag (its fix for a charge-split race; counts and money match HEAD); T-3844 filed; 8800 = 7637e0e0, smoke
   passed on it.
   Running, mine: gapfill (Opus, T-3764: the missing-country list, then Pakistan and Bangladesh; main tree). Runs
   started ~11:25 (PK 196 rows port 6395, BD 142 rows port 6396, --no-join; gates serve a frozen app copy), gating
   ~2-3 h, then it messages before the join (no data/packs writes during it). Its uncommitted files so far: pipeline
   world_plan.py, admin_totals.py, borders.json (IN|PK 0), checks/country_preflight.py, checks/state_gaps.py, game
   zone-names.json and realms.json, notes/T-3764.md.
  
2. T-3395: 11-20 scan (pass 1, not done) found 0 new damage. Run `tools/ops/save_health.py` after the 11-21
   04:00 UTC scan; clean = close it (the two rev-1 games of ue83342ed78f8fc4c are old 11-19 damage).
3. Up next: T-3804 after editfirst; T-2216 stages once she picks; T-3838, T-1376 retry, T-3764 (Pakistan, Bangladesh, Central Asia; Iran's plan_people rule touches its tiles);
   T-1738 Bhamo; T-3836 (design for her).
4. Next update: unpublished changelog c001-c015 plus tonight's; untranslated: T-3822's and T-3823's hover strings,
   diagramView.addCity; use the publish skill (review agent, sort, playtest list, translations, new zones image).
5. Leftover: stale Vite pid 33736 on 6181 (leave), trees ttchk2base, touchrule, timekeys.
6. T-3781 waits for the next report's diagnosis. Steam (T-3729): her screenshots in artifacts/trainworld/steam.

## Her words
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
