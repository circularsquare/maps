# Orchestrator handoff, trainworld

Orchestrator: session e152feed-f0d3-4dcd-84f4-e7cb5c5a0a5b, name maps-3f, since 2026-11-18 22:56 UTC
Updated: 2026-11-18 23:27 UTC

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
   T-3731 landed df82464f (next update's c001).
   - Landed 23:00-23:26 UTC: T-3711 (c002), T-3709, T-3710, T-3730, T-3721 (c003), T-2380 (intercity.json deleted),
     T-3461 (docs/code-tour.md), T-3732. Master 28e475a2 pushed. 8800 = 94a62a96 (needs a rebuild for T-3732 etc.
     after the next landings). Batch smoke on 94a62a96 PASSED (29).
   - Server deployed 23:08 UTC (0846bef + fa61219) and pushed, on her "yes can deploy server now".
   - Running: walkstart (Opus, adopted) T-3718, needs a checker; slotown (Opus, adopted) T-3734 stage 1, stage 2
     can start now (copylabel and panelstops landed); weekpace (Sonnet) T-3584; speedsel (Opus) T-3525 port 6186;
     rampend (Sonnet) T-3712+T-3713 port 6191; pairschk (Sonnet) checker on tree servedpairs (T-2880 done there).
   - Refactor: she leaves the pick to us ("i am not suited to make a call"), skip track (rework coming). T-3734 picked.
   - Next: T-3726 comments A/B after group 1 lands (advisor's design in handoff-log 20:47); then the up-next list.
3. Up next: T-3495, T-2054, T-2901, T-3354, T-3518, T-2598; T-3720 after T-3734 (same slot code).
4. T-3729 Steam: on hold (Anita 23:15 UTC: "dont worry about steam now"). She is thinking about screenshots (hers
   or agents'); capsule art from the "trains world!" logo after (research/steam-page.md). Don't raise it.
5. T-3395: close after a day of clean daily scans (deployed 11-18 13:30 UTC).

## Her words
- 11-18: close easy high-priority wins and old tasks, small over design-heavy; "prio should take into account
  easiness as well as impact" (notes/triage-2026-11-18-{a,b}.md, todo-sweep-2026-11-18.md).
- "we need to be less willing to put stuff on the changelog": a player-view review agent hides noes before she edits
  (docs/orchestrator.md, notes/changelog-review.md). Player text says "manually", never "by hand" (CLAUDE.md).
- 11-18 23:35: a task kept open after its main work gets its title/Context/Goal rewritten to what's left; Goal
  dropped when the title says it (CLAUDE.md Tasks; tasks.py check no longer requires Goal). Sweep: tasksweep agent.
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
- Text not yet acknowledged: "Saving stops" (T-3731). T-3711's shift-click hovers and T-3710's roundTip: acked 23:16 UTC.
- She sends the T-3635 player reply herself (draft in notes/T-3635.md).

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, notes/asks-log.jsonl; maps .claude/settings.json.
- Old trees shenzhen (dirty pipeline/osm_units.py, owner unknown) and tt2 (stale, stray cores.json): ask before
  removing. Empty locked folders .tw-trees/touchrule and .tw-trees/timekeys: remove once whatever holds them exits.
- Never delete a tree folder by hand: its app/node_modules junction deletes the main tree's (npm install from
  trainworld/app restores it).
