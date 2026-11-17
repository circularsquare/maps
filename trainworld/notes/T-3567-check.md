# T-3567 / T-3566 check (farecheck, 2026-11-17)

Tree .tw-trees/lineprofit, own Vite 6159 (stopped), real save T-007-real-nyc.save, line 14.

## Passed
- `npm run check --prefix app` (tsc): clean. Tree's own check checks/t3567-line-fare.mjs: all passed (1.2 / 5 held to 3 / 0 /
  undo button / exported save carries it and reopens at 1.20x / no page errors).
- Row sits right above the chart at the panel bottom; chart is thin blue and red bars (shot: artifacts/trainworld/t3567_line_light.png in the tree).
- Old save (no `lines`, empty lineDays) at 1.00: every line's riders and revenue, rail trips, and total revenue are
  byte-identical (JSON hash) to a master Vite (main tree HEAD 2066814c, same save, same wait). Note master is 8 app/sim commits ahead of the tree's base.
- Typed values (panel typing): -1 -> 0.00x; 3.5 -> 3.00x; "abc", empty: no change; 0.123456789 -> 0.12x; 1.005 -> 1.00x and the
  entry is dropped; 1e1 -> 3.00x. Via world: unknown line 9999, NaN, Infinity refused, nothing stored.
- Only the edited line gets an entry (`{"14":1.5}`); 1.00 leaves no entry.
- Sim line ids are appended (line_alive.push), never reused, so a new line cannot inherit a deleted line's multiple.
- 0x: fare a ride $0.820 vs $1.491 at 1.00 (base still paid), revenue still > 0 (the recorded caveat).

## Findings (minor)
1. Typing "1,5" reads as 15 (the widget strips non-digits except "."), so 3.00x. Same widget as other typed multiples.
2. Deleting a line then pressing undo brings the line back but its multiple is gone (tidyLineFares dropped it after the
   snapshot). Entry also lingers in moneyState.fares.lines for a moment after delete.
3. The undo op does not undo a fare multiple change (fare ops bypass the undo stack, as for the other fare ops).

## Not tried
Quad-track twins, bus and ferry lines (the real save has none; building them through the harness was not done), save size growth
(figures from the notes: ~20 bytes a line a day), finances-tab total vs sum of line charts (only sums of line revenue read: 3811.3 identical to master),
30-day chart on an old save with 8 days (the real save has 0 line days; chart drew 6 days right-aligned after a clock run).
Demand re-solving after a change in my break script was not confirmed (version did not move in 60 s under load); the tree's own check does confirm it (riders move each way).

## Environment
Shared trainworld/app/node_modules was broken at 13:48 (an interrupted npm op: @preact/signals, .bin and the rolldown binding's package.json missing).
I restored it with `npm install` in trainworld/app (added 15 packages, package.json and lock unchanged) and one hand-written package.json for the binding.
