# Orchestrator handoff, trainworld

Orchestrator: session aa626a2e-dc8d-4ec4-888a-66763905c165, name maps-6b, since 2026-11-19 03:22 UTC
Updated: 2026-11-19 13:25 UTC

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status` (T-3667). This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
0. Live: v0.4.2 (data 2026-11-18b). Unpublished changelog c001-c015 (v0.4.3). Cap 14. Server deployed 13:14 UTC
   11-19 (shonei-server c19200d, save levels 23-26, pushed). 8800 = 38075a1a (all overnight work); master has
   fare zones Z1 on top (invisible).
1. Running (her asks of 13:00 UTC 11-19), each in its own tree, each needs a checker before landing:
   - savesize (Opus, T-3757 P1): players' saves of 6-14 GB crashing; she loaded a 14 GB one at 6 GB RAM. Port 6154.
   - zonefares2 (Opus, T-3740 Z2): paint tool, tint (3-version sheet), panels. Port 6153.
   - zonerule (Opus, T-3758): zones counted lowest-to-highest the route touches; per-line zone/distance mode, kept
     on copy; may take save level 27. Port 6155.
   - namebind (Opus, T-3759): per-station bound flag, setting for new stations (default on), bind-all; may need a
     save level (renumber at landing; server change then). Port 6156.
   - irlstations (Sonnet, T-3760): real stations as rings above blueprints and track. Port 6157.
   - T-3761 and T-3760 (rings) LANDED; 8800 rebuild 5e5f6e30 + smoke started ~13:25 UTC.
   - The "GB" is the estimated memory usage readout, a general pattern (her 13:25); relayed to savesize.
   - namebind: "ideally just one button for binding": one toggle in the station panel; bind-all folded into the
     setting's confirm.
   - ttport (Opus, T-3346 stage 2: compare tt2 with T-3542 first, port only what's left; port 6159) and
     focusload (Opus, T-3686; port 6160), her picks 13:25.
   - commentab (Opus, T-3726): sets up the comments A/B (3 tasks, -c/-s trees ab-<n>-*, blind briefs in scratchpad
     commentab); then I launch the arms (same model both), score per scoring.md, report to her.
   - 8800 = 5e5f6e30 (T-3760 rings, T-3761, Z1), smoke passed, pushed ~13:40 UTC.
   - Steam: she is shooting screenshots now into artifacts/trainworld/steam/screenshots; guide research/steam-page.md.
2. Her country order (11-19): Guyana/Suriname/French Guiana (T-3762), then Laos and Myanmar (T-1738, players
   asked), Iraq and Iran (T-3763, players asked); then gaps (T-3764: Pakistan, Bangladesh, Central Asia). Up next
   queue: T-3762, T-1738, T-3763, T-3409 + T-3360 (schematic: chosen cities, buses and ferries; "would feel quite
   nice for players"). wherestats (Sonnet, T-3382, her pick; game tree + shonei-server commit, deploy her yes).
   Tonight: T-3762 Guyana, Suriname, French Guiana world run (French Guiana needs planner work: inside France's
   Natural Earth feature). Start it at her bedtime, as T-3620 (agent keeps GATES/JOIN lines; hold app/ landings).
3. Isolated work offered to her 11-19 (pick or I start): T-3346 stage 2 port (timetables, tree tt2, unblocked),
   T-3060 rebuild preview before spending, T-3686 focused city slow on big saves, T-2268 station placement over
   ramps, T-3660 two Egyptian units missing, T-3079 exact stop ordering. Open ask: a-2026-11-19-1 (T-3746 text).
4. Fares later: T-3744 per-stop fare (design for her first); Z3 (rings, new-station zones, bus stops).
5. T-3738 share previews: watch Workers requests daily for a week, then close. T-3395: close after clean scans.
   T-3354 needs her exact drag. T-3729 Steam: she is making the store page 11-19 (screenshots can change later).

## Her words
- 11-19: "the schedule periods look really nice!", "fare fixes also look good"; no changelog text acks needed
  (she reviews the changelog separately; CLAUDE.md); "caribbean" covers the whole Caribbean in the changelog.
- 11-19 zones: "if a trip from zone 2 to 2 counts as 1 that does feel pretty weird to me"; lines choose zone or
  distance fares, "you can exempt metro lines from zones", copies keep it (T-3758).
- 11-19 station names: a bind setting, default on, for new stations; "if you unbind you can set whatever" (T-3759).
- 11-18: close easy high-priority wins and old tasks, small over design-heavy; "prio should take into account
  easiness as well as impact".
- "we need to be less willing to put stuff on the changelog". Player text says "manually", never "by hand".
- 11-18: a task kept open after its main work is retitled to what's left; Goal dropped when the title says it
  (CLAUDE.md Tasks). Text reused word for word from the game needs no ack (CLAUDE.md).
- v0.4.3 wishes (pad 11-18): fuller fare setting (T-3739, T-3740, T-3744), five schedule periods (T-3542); nudges
  "later, not in the same round".
- 11-19 playtest: "the shift-click works as intended!", "icons cute!", freight snap works; unify the two track
  selections (T-3746); hide flat marks no line uses (T-3745, landed).
- Share previews: "thatll make people feel a lot better about sharing"; free Worker, upgrade if we hit it
  regularly; "please be careful" with the Cloudflare account.
- No route: "showing anything, even if its not eprfect, is better than showing nothing at all" (T-3724).
- Mod support (T-3377) serves hand-coders and assistant users alike; never say publicly it's aimed at vibe-coders.
- Venezuela intercity demand: "fine tuning we can do later" (T-3706 P4).
- caelum418's speed PRs (T-3659): "look at the next time we look at this area of performance".
- When she plays games: keep 4 cores free (cap 12) and no uploads until she says she's done.
- Standing this week: small isolated fixes, count down, merge duplicates; she edits each update's changelog on the
  board before publish; lean testing, smoke per batch; long jobs in the background, answer her first; every
  publish, deploy and push needs her yes to that action, none overnight; advisor trial (log calls in handoff-log).

## Waiting on her (asks.py list has the open ones)
- She sends the T-3635 player reply herself (draft in notes/T-3635.md).

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json.
- Old trees shenzhen (dirty pipeline/osm_units.py, owner unknown) and tt2 (T-3346 stage 2, stale since 11-12, keep
  for the port): ask before removing. Empty locked folders .tw-trees/touchrule and .tw-trees/timekeys: remove once
  whatever holds them exits.
- Never delete a tree folder by hand: its app/node_modules junction deletes the main tree's (npm install from
  trainworld/app restores it).
