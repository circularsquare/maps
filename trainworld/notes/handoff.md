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
   - Later landed (to 00:10 UTC): T-2880, T-3584, T-3734 stage 1 (checks), T-3736, T-3737, T-3741, T-3712, T-3713
     (c004). 8800 = 24fbf73c. No smoke on master since 94a62a96: run one after walkstart lands.
   - Running: walkchk (checker, T-3718 tree walkstart, rebased on 96575add); speedsel (Opus) fixing the Mixed
     stepper stuck bug limitchk found, then land T-3525; slotown2 (Opus) T-3734 stage 2 port 6193; freightsnap
     (Sonnet) T-3495 port 6195 (graph build, ~100 MB); faredesign (Opus) T-3739+T-3740 design only;
     periods0 (Opus) T-3542 stage 0 port 6199 (T-3542 before T-3346 stage 2: DECISIONS 11-18; tt2 untouched).
   - T-3738 share previews: her yes 00:12 UTC ("built as described yes, as long as theres no substantial new costs";
     schematic with no names OK). Built: shonei-server 5c0a276 + debd1e1 (not pushed/deployed), worker in
     trainworld/tools/sharepreview/ (uncommitted). Ask 29: free worker (35-51% of 100k/day, shared with graffiti)
     or $5/mo or short link. Then server deploy on her word, her `wrangler login`, steps at end of notes/T-3738.md.
   - T-3495 freight snap: checker found +2.2 s and +200 MB for US players; freightsnap resumed to split kinds 8/9
     into a per-country file loaded only under All, router ignores them otherwise, fix old-app `< 0` tests.
   - 8800 = 4bc06cc9 (T-3718, T-3734 stage 2). Smoke on 59544550+T-3718 master PASSED.
   - "Train adjustments": she doesn't recall; dropped. T-3346 stage 2's failing 30/10 case: she's open to fixing it
     "if we think its worthwhile" (part of porting stage 2 after T-3542).
   - Text acked: "Can't branch off a ramp. Use a flat stretch of track."; T-3698's "This game was saved in another tab
     first. Save again to overwrite it."
   - T-3742 nudge a line's departures (her "train adjustments"): "can do nudges later, not in the same round"
     ({after:v0.4.3}); design done (notes/T-3742.md, 4-6.5 h, save level bump, after T-3542 stage 0). Its three
     questions and draft strings go to her when that round starts, not before.
   - Fares: design in notes/T-3739.md; ask 28 open (four calls + build T-3739 now?).
   - T-3718 landed b191eede (c006). slotchk checking T-3734 stage 2 (tree slotown2; closes T-3698 at landing).
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
- 11-18 23:55: "editing stops feels good!"; text reused word for word from the game needs no ack (CLAUDE.md);
  T-3736 (ferry/bus icons in line hover), T-3737 (drop the calculating-trips tooltip), T-3738 (share links preview
  the map: design to her first, sharepreview agent) filed from her message. "Speed limit"/"Mixed" acked.
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
