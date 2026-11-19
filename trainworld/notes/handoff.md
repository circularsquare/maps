# Orchestrator handoff, trainworld

Orchestrator: session e152feed-f0d3-4dcd-84f4-e7cb5c5a0a5b, name maps-3f, since 2026-11-18 22:56 UTC
Updated: 2026-11-19 01:58 UTC

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status` (T-3667). This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
0. Live: v0.4.2 (data 2026-11-18b). Unpublished changelog c001-c009 (next update). Server deployed 01:03 UTC
   (shonei-server 962fcae, pushed). Cap 12 cores ("keep 4 cores free... no need to be super strict").
1. Running (each needs checker if risky, then `agent-tree.mjs finish`):
   - periods0 (Opus) T-3542 stage 0 (five period slots, to the bit on NYC/London, +0.7-1 ms per edit accepted):
     finishing the 10 stride-3 checks, trying the Capacity-clone cure, full smoke. Then a checker, land, and
     stage 1 (3/4/5 rows per line) as a fresh agent from notes/T-3542.md "Stage 0". T-3542 goes before T-3346
     stage 2 (DECISIONS 11-18; tree tt2 untouched; its 30/10 fix after, she's fine with it "if worthwhile").
   - slotown3 T-3734 stage 3 (one game stamp) built; checker stampchk running (scores must still be sent after
     every kind of load). Stage 4 (save code out of clock.worker.ts) after; clear the taking mark if not started.
   - jointerrain (Opus) T-2054: first answers whether new zones still need per-zone terrain after T-2901.
2. T-3738 share previews LIVE (worker trainsworld-sharepreview 772a7d85 on anita.garden/trainsworld/* + 13
   no-worker routes; free plan, her pick). Left: she clicks fail open in the dashboard (no API field), pastes a link
   in Discord; watch daily Workers requests vs 100k for a week; then close T-3738 (c009). Undo = delete the route.
   Seen: the page's CSP blocks Cloudflare Web Analytics' injected script (pre-existing; told her).
3. Fares: her answers 03:00 UTC in notes/T-3739.md "Her answers" (no Steps, Add point + click snapped monotonic;
   zones A+B, new stations take neighbours' zone). advfares (Opus) building T-3739 port 6108. T-3740 zones and
   T-3744 per-stop fare (new, design first) after it. "Saving stops" acked. T-3742 nudges: {after:v0.4.3}.
   Cloudflare fail-open: no field on the zone's route edit; docs say only "toggle the route"; she can try Workers &
   Pages > trainsworld-sharepreview > Settings > Domains & Routes; else leave it (35-51k/day vs 100k).
4. Landed since: T-2054, T-3734 stages 3-4 (closed), T-3518, T-2901 stage 4 (70eec98a). Smoke on 6af04bd4 PASSED.
   8800 = 6af04bd4 (next snapbuild proves no dist/packs/terrain). After periods0 lands: delete
   data/terrain/*.terrain (4,631 files, 704 MB; keep world.water). Running: periods0, autolatch (T-3720).
   Up next: T-3354, T-2598, T-3743; T-3726 comments A/B (design in handoff-log 20:47).
5. T-3729 Steam: on hold (Anita 23:15 UTC: "dont worry about steam now"). Don't raise it.
6. T-3395: close after a day of clean daily scans (deployed 11-18 13:30 UTC).

## Her words
- 11-18: close easy high-priority wins and old tasks, small over design-heavy; "prio should take into account
  easiness as well as impact" (notes/triage-2026-11-18-{a,b}.md, todo-sweep-2026-11-18.md).
- "we need to be less willing to put stuff on the changelog": a player-view review agent hides noes before she edits
  (docs/orchestrator.md, notes/changelog-review.md). Player text says "manually", never "by hand" (CLAUDE.md).
- 11-18: a task kept open after its main work gets its title/Context/Goal rewritten to what's left; Goal dropped
  when the title says it (CLAUDE.md Tasks). Text reused word for word from the game needs no ack (CLAUDE.md).
- v0.4.3 wishes (pad 11-18): fuller fare setting (T-3739, T-3740), five schedule periods (T-3542); nudges later.
- Share previews: "thatll make people feel a lot better about sharing"; free Worker, "upgrade later if we hit it
  regularly"; "please be careful" with the Cloudflare account.
- No route: "showing anything, even if its not eprfect, is better than showing nothing at all" (T-3724).
- Track look: no halos on the map; only the count icon beside the level in the hovers (T-3714, T-3727).
- Mod support (T-3377) serves hand-coders and assistant users alike; never say publicly it's aimed at vibe-coders.
- Venezuela intercity demand: "fine tuning we can do later" (T-3706 P4). Caribbean+: later, she'll say.
- caelum418's speed PRs (T-3659): "look at the next time we look at this area of performance".
- When she plays games: keep 4 cores free (cap 12) and no uploads until she says she's done.
- Standing this week: small isolated fixes, count down, merge duplicates; she edits each update's changelog on the
  board before publish; lean testing, smoke per batch; long jobs in the background, answer her first; every
  publish, deploy and push needs her yes to that action, none overnight; advisor trial (log calls in handoff-log).

## Waiting on her (asks.py list has the open ones)
- Ask 28 (fares). Text not yet acknowledged: "Saving stops" (T-3731). Fail open click + Discord test (T-3738).
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
