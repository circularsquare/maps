# Orchestrator handoff, trainworld

Orchestrator: session 96357f14-1979-4190-bef7-751a0544778a, name maps-6f, since 2026-11-19 14:21 UTC
Updated: 2026-11-19 14:21 UTC

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status` (T-3667). This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
0. Live: v0.4.2 (data 2026-11-18b). Unpublished changelog c001-c016 + tree items (v0.4.3). Cap 14. Server deployed
   13:14 UTC 11-19 (shonei-server c19200d, save levels 23-26, pushed). 8800 = 5e5f6e30; master since adds T-3382
   (d4195ede, game side). maps-6b handed over ~14:25 UTC at 500k.
1. Agents running (maps-6b's; reports reach you through the inbox; follow-ups go to fresh agents on their trees):
   - savesize (Opus, T-3757 P1, tree savesize, 6154): players' "estimated memory usage" readout shows 6-14 GB on
     some machines, far above ours ("a general pattern"); she saw 6 GB RAM loading a 14 GB-readout save. Checker.
   - zonerule (Opus, T-3758, tree zonerule, 6155): zones counted lowest-to-highest the route touches; per-line
     zone/distance mode kept on copy; may take a save level. Checker. Lands AFTER Z2; fix the wiki rule line then.
   - namebind (T-3759, tree namebind): done incl. her two changes (setting per player, far out shows the complex's
     name); level 27 LSN1, server needs it. Checker namechk running; land after it passes.
   - zonechk2 checking T-3740 Z2 (tree zonefares2, 6162/6163). Her look 14:20: keep B's look but crisp vector
     Voronoi (not pixelled cells), no overlaps, tint only in painting mode (already), every station/complex shows
     its zone number in painting mode. After the checker, a FRESH agent on tree zonefares2 does that (zonefares2
     itself is at 480k), then land Z2 (strings asked, a-? not filed: she reviews on landing).
   - ttport (Opus, T-3346 stage 2, 6159): compares tree tt2 with T-3542 first; may stop and recommend closing.
   - focusload (Opus, T-3686, tree focusload, 6160): focused city slow on big saves; measure, cut save-size steps.
   - commentab (T-3726): setup done (r-c119e957; <S> = maps-6b's scratchpad\commentab, scoring.md there). Arms run
     one pair at a time so both sides share the same load; all Sonnet. Pair 2 (T-3603) ran 14:24 UTC (maps-6f):
     ab-2-c agent afd9acbe90af31f97, ab-2-s a31fb76712767f2c3; both the same partial fix, both FAIL, no difference (notes/T-3726.md).
     Pair 1 (Rust) waits for free cores (ledger over cap 14:40).
     Then pairs 1 and 3, then pair 2 again on fresh trees (setup.md). Disk: C: 10.7-23 GB free and swinging; trees
     hold 36 GB; remove each ab tree once scored.
   - maps-6f's: namechk (checker on tree namebind, 6166; both of her changes made, r-d4dfa211); busverify (6167,
     master: do bus/ferry fleet edits recalculate? all reports were v0.4.1, T-3704 fixed it in v0.4.2);
     periods5 (Opus, T-3767 P1, tree periods5, 6169: new lines' auto headways for every period, buses and ferries
     get 3/4/5 periods; same schedule code as ttport); zonerulechk (checker on tree zonerule, 6170/6171;
     zonerule r-8206de31 done, level 27 LFD1, still lands after Z2).
   - T-956 Safari bubbles landed 11-19 (blank overlay canvases hidden); not reproducible here, real Safari check
     at publish.
   - Landed 1e793d04 T-3765 legacy speed setting (c017); 8800 = 1e793d04 since ~14:36.
2. Save levels: master 26 (LFZ1). namebind wants 27 (LSN1), zonerule may want one, ttport may: renumber at landing
   in landing order; then shonei-server needs each (srvlevel's pattern: c19200d; deploy on her yes, not overnight).
   Open server ask a-2026-11-19-10: deploy 5bf10c7 (twstats minutes per city, T-3382).
3. Tonight: T-3762 world run: Guyana, Suriname, French Guiana and every Caribbean island group left out (Guadeloupe,
   Martinique, USVI, Saint Kitts, Sint Maarten/Saint-Martin, Caribbean Netherlands, Montserrat, Anguilla, BVI,
   Turks and Caicos), "can't really island hop" otherwise; small islands included despite the zone floor. As
   T-3620 (agent keeps GATES/JOIN lines in its notes; hold app/ landings then). Then T-1738 Laos/Myanmar, T-3763
   Iraq/Iran (players asked), T-3764 gaps. Up next also: T-3409, T-3360 (schematic: chosen cities, buses/ferries).
4. Steam (T-3729): fee paid, tax form in (Valve up to ~45 days); her screenshots go in
   artifacts/trainworld/steam/screenshots, then an agent makes store/library art (research/steam-page.md).
5. Later: T-3744 per-stop fare (design first), T-3740 Z3, T-3754, T-3747. T-3738 watch then close; T-3395 close
   after clean scans; T-3354 needs her drag. Open ask a-2026-11-19-1 (T-3746 text).

## Her words
- 11-19 afternoon: "player complaints take precedent over other new features"; v0.4.3 still aims to ship with the
  five-period schedules and advanced fares. Fares by city (T-3766): country or city multiples, never both.
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
- Old tree tt2 (T-3346 stage 2, stale since 11-12): keep for ttport's port, remove after. shenzhen removed 11-19. Empty locked folders .tw-trees/touchrule and .tw-trees/timekeys: remove once
  whatever holds them exits.
- Never delete a tree folder by hand: its app/node_modules junction deletes the main tree's (npm install from
  trainworld/app restores it).
