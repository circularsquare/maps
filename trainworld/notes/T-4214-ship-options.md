# T-4214 ship options: quick edit sequences in 2-4 hours of agent work (agent shipsafe, 2026-11-24)

Her ask (11-24): "please consult gemini also for what we should do if we want to ship a workable, nonbuggy version in
like the next 2-4 hours. maybe just disable all of the stuff we've added?" Read-only research; no code changed, no
game runs, no benchmarks. Design background: notes/T-4214.md.

## Summary for Anita

- Turning off what we added would not fix your three sequences. Placed track was already drawn before it was clickable
  in v0.4.5, so "track then delete" and "track then station" wait exactly as they did then. "Station then delete" gets
  rarer (the station isn't drawn until it can be clicked) but a quick delete still takes the wrong half, as in v0.4.5.
  And every edit would show up 0.3-1.1 s later again on a big save.
- What does fit in 3-4 hours and fixes all three: a click on the map while the last edit hasn't reached the clickable
  map waits (the tool's spinner) and is carried out at the same spot as soon as it has. The drawing at the click stays.
  The cost: a second click right after the first takes effect when the first edit is built: about 0.3-0.5 s on NYC,
  0.7-1.1 s on your Asia save, up to about 2 s while trips are calculated. Drawing a run of track and clicking stops
  onto a line don't wait, as today.
- Gemini read the code and recommends the same, plus stamping every edit with its game. It agrees turning things off
  fixes none of the three bugs.

## 1. What was added since v0.4.5, and what turning it off gives back

No `perfOff` flag covers any of it (app/src/perfFlags.ts has none for edits shown at the click). But since T-4170
every picture drawn at the click goes through one place, game/pending.ts: `ahead()` (what a tool draws) and
`pendingSent()` (the guess read from the op, game/editGuess.ts). Two early returns there (`ahead()` hands back a
handle whose `send(f)` just calls `f()`, `pendingSent()` makes no entry) turn off every row marked "switch" below at
once: about 1 h with a typecheck, the t4170-one-path check retired or skipped, and one run.

"Shows" is when the click is drawn, "built" when the map you click on has it (edit bench). Asia: 11-24 16:22 quiet
run on master (settle1-before); NYC: 11-24 17:31 (t4210 tree, master plus the ghost work); "while calculating": 11-23
22:50 (older build). Older "before" figures are from ARCHIVE.md.

| Task | Drawn at the click | Off cheaply? | What comes back if off |
|---|---|---|---|
| T-4145 base, T-4170 one path, T-4181 fixes, T-4185 undo of copy/new line | every edit's picture, undo at its click, panel values and cash at the click | switch | undo: 19 ms to about 1.1 s on Asia (archive 1,094 ms), NYC 29 to 245 ms; delete a line 9 ms to 1.7 s |
| T-4146 stage 1 | placing and deleting a station, deleting track, platform length | switch | Asia station 23 ms to 1,067 ms (p90 1,611), 98% of the wait; NYC 15 to 263 ms |
| T-4148 stages 1-2 | construct, blueprint, level, track count and type, double up, rebuild, speed limit, one way, station off | switch | Asia construct 30 to 853 ms; station off 61 to 697; track type 43 to 425, double up 74 to 510, one way 52 to 516, rebuild 48 to 399 (archive) |
| T-4147 stages 1-3 | New line, copy, colour, line settings, an added stop's path, groups, end line here, bus routes, ferries | switch | groups 12-29 to 175-191 ms, end line 18 to 185, bus route 11 to 959, spacing 12 to 1,432 (Asia, archive) |
| T-4114 | a line's path at a stop click, a deleted line gone | switch (pictures only; its faster stop-click sends stay) | Asia end stop added 0.10-0.18 s to 2.7-3.5 s; new line's second stop 0.1 to 1.7-2.8 s |
| T-4097 stop take-out | a removed stop's path | switch | 17 to 349 ms (Asia) |
| T-4143, T-4136, T-4208 | the track ghost while drawing; T-4208 draws the placed segment at the click instead of at the worker's early answer | no: own code in map/tools.ts, reverting commits | first ghost 8-95 ms to up to 2-3.6 s after a pan; ghost trailing 0 to 40-86 px (Asia) |
| T-4113, T-4125, T-4132, T-4133, T-4149 stage 1, T-4153 stage 1, T-4155 | nothing: worker and page work after an edit | no switch | they shorten the real wait; off, every "built" time above grows and the window for the bugs widens |
| T-3999 (a feature, not edit speed) | a station click on open ground starts a two-click station | no | changes sequence 2's outcome: v0.4.5 said "stations go on track only" |

The 0.4.6 changelog items that would go or need rewording with the switch: c080 ("faster / more responsive edit
speed", partly) and c101 (groups, end line, bus and ferry routes at once).

## 2. Does turning them off remove her bugs?

No. Checked against v0.4.5 (b0899150e):
1. **Track, then delete it**: unchanged. v0.4.5's `segmentClick` already drew the placed segment as a stand-in at the
   worker's early answer (`keepPreviewAsPlaced` / `addPlaced`, T-1407, T-2404), and the map tools click on got it
   with the network then as now. The switch doesn't touch it.
2. **Track, then a station on it**: same wait. Outcome differs from v0.4.5 because of T-3999, not edit speed.
3. **Station, then delete the track past it**: rarer, not fixed. v0.4.5 drew a station marker at the click until the
   worker's answer (`addPending`/`dropPending` in its stationTool.ts), then nothing until the network: the same
   window, with less on screen inviting the click. The split keeping the old piece number and the name lookup before the
   send are both in v0.4.5. A delete in the window removes the wrong half there too.

## 3. A general option that fits: clicks wait for the map (B)

**Rule**: a click on the map with an edit tool is carried out against a map that holds every earlier edit. One made
while an earlier edit has not reached the clickable map is held, the tool tag shows its spinner (`waiting()`,
tools.ts), and it is carried out at the same map spot once the map has caught up.

How, from the code:
- **When the map has an edit**: beside the existing message-order rule in game/pending.ts, a small list of edits out:
  added in `pendingSent` (every edit passes there, panels and keys too), gone when refused (`pendingAnswer` with ok
  false), when its word says no network follows, or at the first network after its word. The signal fires in main.ts
  `client.onState`'s `put` after `renderer.setNetwork` and the grid carries, not inside `pendingTaken`, which runs
  just before them. Cleared on a load, a worker restart (`pendingClear`) and a panic.
- **The worker says "no network follows" for every edit that has none**: today fares, fare reference, borrow and repay
  send a plain word (`plainWord`, clock.worker.ts about line 905-935) with no `noNet` and no state, so a held click
  would sit until the backstop. One line there.
- **Held at the click, before any await**: the station name lookup (`nameFor`), BRT roads, real and street routes,
  `lineEffects` in `removeThing` all await before `edit()`, so the click counts as out from the moment it is made
  (map/tools.ts `click` and `tap`, plus `touchConfirm`); a held click also posts `placedNow()` so the T-2404 hold
  (up to 5 s while the pointer moves) and the T-648 trips hold (250 ms) don't add to the wait.
- **Carried out**: `move(e)` then `click(e)` with the click's map spot projected again and its Shift, Ctrl, Meta and
  double-click kept, as `cameraMoved` and `heldLanded` already do. The station tool's worker verdict for the spot
  isn't back yet then, so it places from its candidates (`alts`) as a quick click does today. Undo and redo wait for
  held clicks, as they already wait for stop clicks (`afterStopEdits`).
- **Two exemptions keep today's speed**: the next segment of a track run doesn't wait for the run's own segments (it
  starts from the node the worker answered; `nearHeld`/`heldLanded` stays for clicks back onto the run), and edits that
  only change a line's stops or service never hold a click (stop-click bursts, T-4114, stay instant).
- **Dropped** on Escape, a tool change, a load or a panic, or if the spot has left the screen. **Backstop** 5 s: the
  click is then carried out as it would be today, so the worst case is today's behaviour.

What the player sees: the first click as now; a second click right after waits for the first edit's network
(NYC 0.26-0.54 s, Asia 0.7-1.1 s median, about 1-2 s while trips calculate), counted from the first click, then acts on
the right piece. Clicks made while a track segment is out are held instead of dropped (they are dropped today).

Not covered (left for the long-term design's stage B): a panel button or a node drag acting on something selected
before a quick split; their edits still count as out, so the next map click waits for them.

No new player text: the spinner exists. A changelog line for the implementer to propose, e.g. "fixed quick clicks
acting on the map before the last edit showed up" (--fix, area track).

## 4. The options

| | What the player sees change | Hours | Risk | Seq 1 track, delete | Seq 2 track, station | Seq 3 station, delete past it |
|---|---|---|---|---|---|---|
| **A. Pictures off** | every edit shows up when built: on Asia stations about 1 s, construct 0.85 s, undo 1.1 s, a line's added stop 2.7-3.5 s | 1 | low in code; undoes this week's speed work and two changelog items | not fixed (same as v0.4.5) | not fixed | rarer, not fixed |
| **B. Clicks wait for the map** (with the game stamp below) | the same instant drawing; a quick second click waits 0.3-1.1 s and then does the right thing | 3-4 | moderate: changes the click path of every tool just before a release; worst case of the backstop is today's behaviour | fixed | fixed (station lands on the new track) | fixed (right half, station sent first) |
| **C. Ship as is**, long-term design after | nothing | 0 | the data bug ships (wrong half deleted, stations moved) | not fixed | not fixed | not fixed |

Small items, judged:
- **Game stamp on every edit** (`ClockClient.edit`: `game ?? gameStamp.peek()`; the worker already refuses an edit
  stamped with another game, clock.worker.ts about line 743): sound and small. Checked that a restart, a load and a side
  switch send `loaded` with the new game (clockMain.ts `restart`, lines 415-442), which sets the page's stamp, so no edit
  is wrongly refused after one. About 0.5 h with a load and a side switch tried. Guards edits queued across a panic.
- **Piece numbers not reused at once** (sim/src/track/net/store.rs `alloc_edge`, `alloc_node` pop the newest freed id):
  sound as a safety net, but it fixes none of the three sequences (the split keeps a live id), and it is a sim change
  that needs Rust tests and a NYC figure diff. Not in this ship; with the long-term stage B.
- **A slice of the long-term stage C** (patch the clickable map from the worker's word): does not fit. The page merges
  states as changes to the last one (`takeRows`, `takeNet`/`mergeNet`, T-1029), and the snap grids are kept per
  object (snapIndex.ts lines 33 and 89, WeakMaps), so patching in place breaks both. 6-9 h, as in the design.

## 5. Gemini

Brief: artifacts/trainworld/gemini/T-4214-ship-brief.md; answer: artifacts/trainworld/gemini/T-4214-ship-answer.md (one
run, answered first time). It recommends B hardened, plus the game stamp, about 3-3.5 h; A and C no.

Agreed, checked in code:
- A fixes none of the three; the placed track isn't drawn through pending.ts (tools.ts `clickAhead`, `addPlaced`).
- `removeThing` sends a blueprint delete with no question at once, from the old map's figures (remove.ts line 167).
- T-3999 turns sequence 2 into a two-click open-ground station (stationTool.ts lines 163-164).
- Fire the replay after `setNetwork` and the grid carries (main.ts lines 389-394), not in `pendingTaken`.
- Money-only edits send a plain word with no `noNet` and no network (clock.worker.ts `plainWord`, fares, fareRef,
  borrow, repay): a click held after one would wait for the backstop. Taken into B as a worker fix.
- Mark the click out before the tools' awaits; release the T-2404 hold; gate `touchConfirm`; the stop-click exemption
  only while every edit out is a stop edit; panels, drags and keys arm it but aren't held.
- No stage C slice in this window (delta merges, WeakMap grids).

Not taken, or taken with a change:
- "T-3803's `standAside` is a change since v0.4.5 that makes it worse": T-3803 landed 11-20, before v0.4.5 (no commit in
  b0899150e..master). A new edit does restart the network build behind it, which lengthens the wait in bursts, but that
  is old.
- "T-4208 widens the unclickable window": only the picture comes earlier (at the click, not the early answer); when the
  map can be clicked is unchanged.
- "The game stamp is five minutes": the line is, but every edit gets a new way to be refused; checked that the page's
  stamp follows every restart (above). Half an hour with a load and a side switch.
- "Drop the held click if the player pans or zooms much": B carries it out at its map spot and drops it only if the spot
  has left the screen; dropping on a zoom would lose a click she meant.

## 6. Recommendation

B with the game stamp and the worker's "no network" word for money edits, 3-4 h, keeping every picture at the click.
It is the one option of the three that makes all three sequences correct, it needs no new player text, and it is the
rule the long-term design needs anyway: later stages only make the wait shorter (stage C, the clickable map patched
after each edit) and rarer (stage B, stale names resolved in the worker). After it lands: the nearest checks
(t4170-one-path, any track-run check near `heldLanded`) and one short run of her three sequences on a small new game
in its tree; the release smoke on master then covers the rest.

Budget: 0 benchmark runs, 0 minutes of game checks; one Gemini run.
