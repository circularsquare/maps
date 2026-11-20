# Orchestrator handoff, trainworld

Orchestrator: session a6a6b4a3-3cb7-4e31-ac63-f1cd25a92ff1, name maps-60, since 2026-11-20 01:43 UTC
Updated: 2026-11-20 02:55 UTC (ttzoom runs in maps-c2's tab, guianas in maps-e8's; both report to the inbox)

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status` (T-3667). This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
0. Live v0.4.4 (published ~01:00 UTC 11-20, silent centering redeploy ~01:30). shonei-server deployed + pushed
   fdaa596. Cap 10 (her ask, 02:20). Master fcc45585; 8800 rebuilding to it (then smoke: money/edit path changed).
   Landed by maps-60: T-3812, T-3820, T-3805, T-3822 (b23a2149, hover strings await her ack), T-3803 option 3
   (70845536, fullchk PASS).
1. Running: ttzoom (Opus, T-3817 + T-3819, tree ttzoom, 6200; may stop before removing the Marey button and ask;
   UI only, no checker), guianas (T-3762, writes data/packs; report not yet in), editfirst (Opus, T-3803 option 1,
   tree editfirst, 6207; checker before landing: edit path and money).
   Also: riverroute (Opus, T-3358 prototype, scratchpad only; sheet A/B/C incl. Hong Kong), schemcity (Opus,
   T-3825 + T-3826, tree schemcity, 6209), comments A/B (T-3726) pair 1 arms ab-1-c / ab-1-s running (Sonnet,
   ids a592513c82e673c22 / ac4fb532666d58ca0); then score per maps-6b's commentab scoring.md (runstats, audit,
   score-t3704.py), then pair 3 (ab-3-c/s), then notes/T-3726.md.
   Waiting on her: T-3821 games store (pack or drop backups, + cap 24 GB; asks a-...-23), intercity effective time by
   trips or riders (a-...-24), T-3822 strings. T-3395: after 04:00 UTC 11-20 run `tools/ops/save_health.py`; the
   two rev-1 games of ue83342ed78f8fc4c still corrupt = old damage, close after a clean 11-21; intact = the scan's
   check lagged save levels (notes/T-3395.md "Scan finding of 2026-11-19").
2. T-3790: pipeline fix landed e9a6f91b; ship mos (--force), ber, hon (--force) from
   artifacts/trainworld/t3790/moscowfix once guianas stops writing data/packs; keep sai at 123 with
   `realauto.py keep sai trainworld\data\packs\real\sai.save "..."` first (my call, same rule as Moscow).
3. Up next: T-1738, T-3763 after guianas; T-3818 after T-3817; T-3804 after editfirst (same worker code); T-3809
   (ferry waypoints), T-3823 (intercity hover's first wait), T-3632 islands.
4. Next full update (master): unpublished changelog c001-c013; untranslated: T-3822's three hover strings; before publishing: changelog review agent, sort,
   playtest list, translations of new keys (advancedFares.byCountry to drop).
5. Silent hotfix 11-20 ~01:30: website 2a040d9 (0f9f1f7d on hotfix-v0.4.4; master d426a3d5). Leftover:
   .tw-trees\gearfix (held, probably by this session's shell; `agent-tree.mjs remove gearfix` later), .tw-trees\ringfix folder (worktree unregistered, held; never delete by hand), stale Vite pid 33736 on
   6181 (from 11-18; stopping it was refused, leave), trees ab-1-*, ab-3-*, ttchk2base, touchrule, timekeys.
6. T-3781 open for the next report's diagnosis. Steam (T-3729): her screenshots in artifacts/trainworld/steam.

## Her words
- 11-20 ~03:10 UTC: cap back to 14. Rivers again: "where exactly did we leave this off, and what might we try now
  ... maybe lets consult fable" (riverroute prototype, T-3358). T-3822 text ack; intercity effective time "keep it
  as is"; packing backups: "if you think its pretty riskless lets file it ... til we get cloesr to the limit" (T-3821).
- 11-20 ~02:20 UTC: "could we lower core cap to 10 for a few hours?" (done; back to 14 at 03:10).
- 11-20 ~01:55: "yes can purge r2" (11-13 version purged), "no sta for resend" (stay on the free plan). Games store
  at 80% "a little concerning": compression or a cheaper store? (T-3821, agent gamestore measuring).
- 11-20 ~01:15: start page gear and Settings "- 90% +" not centered: "do want to get this out as a silent hotfix"
  (gearfix, tree from hotfix-v0.4.4; then snap8800 it, smoke8800, deploy.py --skip-data --version 2026-11-19
  --smoked, website push, verify, land onto master, 8800 back to master). No screenshot-compare check: "lets not
  implement it".
- 11-20 ~00:55: v0.4.4 changelog in her words ("added tram stops to real world reference network, and stations show
  names on hover"; naming default item dropped). "yes okay to publish 4.4". Naming: questions both ways, no
  "Confirm?". Change name as a pencil icon. Real-network fixes and additions P4, Japan P3, reference map a bit
  higher, one-city changes low (LINES.md). Up next: Lines tab gear "top right", filters on top of this city / whole
  world; PWA "if super easy to do and maintain".
- 11-20 ~00:10: panel refresh throttle OK "as long as it doesnt creep like way higher than 4s"; Z3 inheritance OK;
  Moscow: only in the real-network build, keep near 106 lines; fares panel x icons, rings icon 25% bigger.
- 11-19 ~23:45: hidden names stay hidden across renames; schematic buses "look great" (A, under); fares by city never
  uses the country multiple; text acknowledged for Z3, display name and schematic hovers.
- 11-19 ~23:30: T-3409 B, chosen cities laid out "as if they were the only things in the world"; Type a column,
  not the default sort; ferry waypoints filed (T-3809); help entry for grouping and no Leave group (done).
- 11-19: "player complaints take precedent over other new features" (actual bugs). Before every publish, a full
  playtest list. Fares by city: country or city multiples, never both.
- 11-18: close easy high-priority wins, small over design-heavy. "less willing to put stuff on the changelog".
  "manually", never "by hand".
- Share previews: free Worker, "please be careful" with the Cloudflare account. Mod support (T-3377): never say
  publicly it's aimed at vibe-coders. caelum418's speed PRs (T-3659): next time we look at performance.
- When she plays games: keep 4 cores free (cap 12) and no uploads until she says she's done.
- Standing: lean testing, smoke per batch; long jobs in the background, answer her first; every publish, deploy and
  push needs her yes to that action in this tab, none overnight; advisor trial (log calls in handoff-log).

## Waiting on her (asks.py list has the open ones)
- a-1 (T-3746 text), a-14 (drop the load card once the city in view is ready). She sends the T-3635 player reply
  herself (draft in notes/T-3635.md).

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine: notes/T-2380.md, notes/T-3382.md; guianas' game json.
- A yes she gives in another session's tab doesn't pass the permission check here: ask again in this tab.
- Never delete a tree folder by hand (its app/node_modules junction deletes the main tree's).
- A tree made from a non-master commit and landed with `finish` squashes its whole branch onto master (dispname,
  1fab7f4a): land hotfix work from the one hotfix tree only.
