# Orchestrator handoff, trainworld

Orchestrator: session 49d5e3c4-3205-457a-b910-06fb8268277a, name maps-76, since 2026-11-18 19:20 UTC
Updated: 2026-11-18 22:30 UTC

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status` (T-3667). This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
0. v0.4.2 PUBLISHED 22:22 UTC (her "ok done. lets send :)"): data 2026-11-18b on R2, website 18c0dc9, --verify
   --page all PASS, open-source copy synced (7e55ee8). Master = 05eea22c pushed; 8800 = 05eea22c. Server deployed
   20:12 UTC (12d58a0, 662ed86, c960adc), pushed 22:35 UTC; from now on a server deploy's yes covers its push. Five
   wrongly latched normal games unlatched after publish (latched.py --wrong now 0). Old R2 versions: purge only on
   her word (r2_purge.py).
1. Her go 22:40 UTC: "lets start with all of group 1 and feel free to continue with group 2 and 3 as well. and also
   continue with whatever else if you finish with that." Keep 4 cores free ("no need to be super strict"): cap 12.
   T-3731 landed df82464f (next update's c001). Running (started ~22:45 UTC):
   - walkstart (Opus) T-3718 trips start with a walk; panelstops (Sonnet) T-3732; localroute (Sonnet) T-3730;
     savecheck22 (Sonnet) T-3733 in shonei-server (commits there, no push/deploy); onewaychk (Sonnet) checker on
     T-3711 tree oneway; refactorlist (Opus) T-3725 survey -> notes/refactor-survey-2026-11-18.md for her.
   - T-3733 done: shonei-server 0846bef (save check level 22), not pushed/deployed; its TestTwSaveCheckFollowsTheGame
     fails until T-3711 lands in maps, and deploy.ps1 runs go test first, so land T-3711 before any server deploy.
   - copylabel (Sonnet) T-3721 started 22:55 (server commit + game tree copylabel).
   - Refactor: she leaves the pick to us ("i am not suited to make a call"), skip track (rework coming). I pick
     from the survey and run it; tell her what and why.
   - Next: T-3726 comments A/B tonight after
     group 1 lands (advisor's design in handoff-log 20:47); then the up-next list.
   - T-3711: on PASS land (DECISIONS line from its notes; WASM). Shift meaning different things on the map and
     in the stop list: OK by her (22:50 UTC). Ships with a server deploy of T-3733 on her word.
3. Up next: T-2880, T-3495, T-2054, T-2901, T-2380, T-3354, T-3525, T-3584, T-3518, T-2598, T-3461; filed today T-3709,
   T-3710, T-3712, T-3713, T-3720.
4. T-3729 Steam: she picks screenshots this evening and says where; then agents make the store images from the
   landing page's "trains world!" logo (research/steam-page.md has sizes and rules).
5. T-3395: close after a day of clean daily scans (deployed 11-18 13:30 UTC).

## Her words
- 11-18: close easy high-priority wins and old tasks, small over design-heavy; "prio should take into account
  easiness as well as impact" (notes/triage-2026-11-18-{a,b}.md, todo-sweep-2026-11-18.md).
- "we need to be less willing to put stuff on the changelog": a player-view review agent hides noes before she edits
  (docs/orchestrator.md, notes/changelog-review.md). Player text says "manually", never "by hand" (CLAUDE.md).
- No route: "showing anything, even if its not eprfect, is better than showing nothing at all" (T-3724).
- Track look: no halos on the map; only the count icon beside the level in the hovers (T-3714, T-3727).
- Points scale with each line's cost recovery capped at 1 (T-3657, shipped in v0.4.2).
- Mod support (T-3377) serves hand-coders and assistant users alike; never say publicly it's aimed at vibe-coders.
- Venezuela intercity demand: "fine tuning we can do later" (T-3706 P4). Caribbean+: later, she'll say.
- caelum418's speed PRs (T-3659): "look at the next time we look at this area of performance".
- When she plays games: keep 4 cores free (cap 12) and no uploads until she says she's done.
- Standing this week: small isolated fixes, count down, merge duplicates; she edits each update's changelog on the
  board before publish; lean testing, smoke per batch; long jobs in the background, answer her first; every
  publish, deploy and push needs her yes to that action, none overnight; advisor trial (log calls in handoff-log).

## Waiting on her (asks.py list has the open ones)
- None open beyond text acks. Steam screenshots.
- Text not yet acknowledged: "Saving stops" (T-3731).
- She sends the T-3635 player reply herself (draft in notes/T-3635.md).

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, notes/asks-log.jsonl; maps .claude/settings.json.
- Old trees shenzhen (dirty pipeline/osm_units.py, owner unknown) and tt2 (stale, stray cores.json): ask before
  removing. Empty locked folders .tw-trees/touchrule and .tw-trees/timekeys: remove once whatever holds them exits.
- Never delete a tree folder by hand: its app/node_modules junction deletes the main tree's (npm install from
  trainworld/app restores it).
