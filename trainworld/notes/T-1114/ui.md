# T-1114 review: app/src/ui/

Read-only review of `app/src/ui/` (components, `inspector/`, `style.css`, `account.css`), 2026-10-24.
Every "unused" claim below was checked with a search over `app/src` and `checks/`. Line numbers are
as of commit 54005ec plus the working tree.

## Summary

- **34 cleanup items and 4 bugs**, in 7 batches.
- **Lines that could go:** about 250 to 300 net (duplicated helpers about 120, history comments
  about 40, dead code and dead CSS about 50, merged CSS rules and dialog copies about 50). The two
  file splits (items 6 and 7) move about 1,500 lines without cutting them.
- **The five best:**
  1. Bug: the "refused for cash" shake never plays. The global `animation: none !important` beats
     `.shake` (B1, a one-word fix).
  2. Bug: two different bars share the class `.load-bar`, so the train's "On board" bar picks up the
     loading bar's height and margins (B2).
  3. One shared set of trip text helpers and route-row parts for the trip panels. `pct`, `exact`,
     `min`, `num` and `roundTo` exist in 2 to 6 copies each, and the route row is written three
     times (item 1).
  4. Remove the dead `cell` and `zone` selections and their placeholder inspector (item 2).
  5. One `useClockQuery` hook in place of 7 copies of the same "ask the clock, guard with `on`"
     effect (item 4).

Comment slop is not listed as a sweep, because CLAUDE.md says "trim a function's comments when you
edit it; no sweeping rewrites". Each batch lists the history and stale comments in its own files
(item 5 has the full list), so they go as those files are touched.

---

## Bugs (fix first, batch 0)

**B1. The refused-for-cash shake never plays.** `style.css:44` sets
`*, *::before, *::after { animation: none !important }`, and `.shake` at `style.css:329` sets
`animation: shake 300ms linear` without `!important`, so the important rule wins.
`LineInspector.tsx:413-427` (SHAKE_MS, `wrap`, the `key` swap) runs, but nothing moves. Anita asked
for the shake (DECISIONS T-625, notes/T-679.md). Fix: `animation: shake 300ms linear !important`,
as `.spin` and the `bt-` rules already do. The reduced-motion rule at `:331` then needs
`!important` too.
Risk: none (it restores agreed behaviour). Effort: small. Mechanical: yes.
*Fixed* (appbugs): both rules `!important`.

**B2. Two bars share `.load-bar`.** (Fixed in appclean, 2026-10-24: the train's bar is `.board-bar`, with the look it had in use.) The train's "On board" bar (`others.tsx:533`, styled at
`style.css:353-355`) and the loading progress bar `LoadBar` (`LoadingScreen.tsx:40`, styled at
`style.css:578-580`) use the same class. The later rule wins, so the train bar is 8 px tall, not
10 px, with `margin: 8px 0 6px` (its row grows) and loses its `margin-right: 8px`. Fix: rename the
train one to `.aboard-bar` in `others.tsx:533` and `style.css:353-355`.
Risk: low (the train bar goes back to how it was written). Effort: small. Mechanical: yes.
*Not fixed* (appbugs): handed to the app cleanup agent, which moves the train bar into
TrainInspector.tsx and renames the class there.

**B3. A locale-dependent figure.** `Blueprint.tsx:124`:
``data-tip={`$${Math.round(r.cost).toLocaleString()}`}`` has no locale, so a German browser shows
"$1.234.567". Every other figure is `en-US`. Fix: `moneyExact(r.cost)` from game/format.ts.
Risk: none. Effort: small. Mechanical: yes.
*Fixed* (appbugs): `moneyExact(r.cost)`.

**B4. Settled, no change.** `StartMenu.tsx:53` "Welcome to anitabuilder!" keeps its exclamation
mark (Anita 2026-10-24).

Lesser layout collisions (not bugs anyone has reported): `.track-row` is defined twice for two
different rows (`style.css:521` for the map's track display, `:634` for Build's Track row), and the
second rule's `gap: 8px` overrides the first's `6px 8px`. `.speed-key` (`:674`) is mostly
overridden by `.track-row .speed-key` (`:523`) and `.track-row .track-key` (`:522`); only its
`font-size` still applies. Both are fixed in item 14.

---

## Ranked cleanup items

### 1. Shared trip text helpers and route parts (high value, low risk)

The same small formatters are pasted into each trip and demand panel:

- `pct(n, of)`, identical in 4 places: `CityInspector.tsx:16-19`, `LongerTrips.tsx:26-29`,
  `StationRiders.tsx:14-17`, `PurposeSplit.tsx:34-37`.
- `pct(v)` for a share: `TripInspector.tsx:21` and `MarketTrip.tsx:18`.
- `exact` (rounded, with thousands commas): `TripInspector.tsx:54`, `MarketTrip.tsx:20`,
  `LongerTrips.tsx:25`, `CityInspector.tsx:15`, `StationRiders.tsx:29`, and `TrainStops.tsx:12`,
  which is named `count`, shadows format.ts's `count` and sits between the imports. `others.tsx`
  writes `Math.round(x).toLocaleString("en-US")` six times (`:529-541`) and `tabs.tsx` three times.
- `min`, `num` and `roundTo`: `TripInspector.tsx:20, 49, 53` and `MarketTrip.tsx:17, 19, 22`.

The same rows are rebuilt as well:

- The route row (ride legs, then from, to and line names, then chips and a share) is written three
  times: `TripInspector.tsx:300-313` (OtherRoutes), `TripInspector.tsx:372-383` (BusiestRoutes) and
  `MarketTrip.tsx:116-126`.
- The leg rows (the walk-change row and the ride row) appear twice: `TripInspector.tsx:161-182` and
  `MarketTrip.tsx:74-91`. Only the access and egress rows differ.
- "Train by period": `TripInspector.tsx:401-421` and `MarketTrip.tsx:220-235` share the same
  markup.

What to change: add `inspector/tripText.ts` (`pctOf`, `pctShare`, `exact`, `minText`, `num1`,
`roundTo`) and `inspector/RouteParts.tsx` (`routeSummary(r, stations, lines)` returning rides, from,
to and names; `<RouteChips>`; `<LegRows>`; `<PeriodRows>`). If the game/ reviewer agrees, `exact`
goes in game/format.ts next to `count`, so `points.ts` `pointsExact` and `networkTips.ts` `n0` can
share it.
Risk: low. The hovers must keep their exact wording; the copies differ only in the hover text, which
can be passed in. Effort: medium. Mechanical: mostly, but the hover strings need care.

### 2. Dead `cell` and `zone` selections (high value, none to low risk)

`AreaInspector` and `Placeholder` (`others.tsx:926-942`), the fallback branch at
`Inspector.tsx:22`, and the `{ kind: "cell" }` and `{ kind: "zone" }` variants of `Selection`
(`game/types.ts:246-247`) are dead. A search for `kind: "cell"`, `kind: "zone"`, `kind === "cell"`
and `kind === "zone"` over app/src and checks finds only these lines: nothing creates either
selection. Their text ("once the city pack is in the game") is from before city packs existed. The
`others.tsx:1-2` header ("placeholders for cell and zone (T-026's)") goes with them.
Risk: none. TypeScript will flag any reader of the two kinds. The types.ts lines belong to the
game/ reviewer's area, so coordinate. Effort: small. Mechanical: yes.

### 3. One modal dialog (medium value, low risk)

`Modal` lives in `AccountPanel.tsx:134-160`. `SavesPanel.tsx:172-207` re-implements the same thing
line for line: the native dialog, showModal with focus restore, onCancel, a backdrop click, an Esc
key handler, `saves-inner` and `saves-head` with a close button. The only extras are the heading id
and `manageSaveSlot("refresh")`. `Ask` in `Toolbar.tsx:26-36` repeats the showModal and
focus-restore effect a third time.

What to change: move `Modal` to its own `ui/Modal.tsx` with a `useModalDialog(ref, open)` hook. Use
it in SavesPanel (the refresh goes in its own `useEffect` on `open`), AccountPanel and Ask. The
Leaderboard.tsx import changes from `./AccountPanel` to `./Modal`; since Leaderboard is being
rewritten, do this after that lands, or leave a re-export in AccountPanel for now.
Risk: low. Watch that Esc in SavesPanel still stops at the dialog, since main.ts clears the selection
on Esc. Effort: small. Mechanical: mostly.

### 4. A `useClockQuery` hook (medium value, low risk)

The same effect appears 7 times: `let on = true; void queryClock(...).then((d) => on && d && set(parse(d)));
return () => { on = false; }`. The copies are at `LineInspector.tsx:509-516, 517-527, 533-538`,
`others.tsx:557-564` (Shape), `others.tsx:643-653` (DoubleUp) and `others.tsx:828-835` (Junction,
two queries), and, the same shape around `complexTransfers`, `ComplexMap.tsx:216-221`.

What to change: `useClockQuery(req, deps, parse)` returns the parsed value or null, in a small
`ui/useClockQuery.ts`. lineHeld parses into `{ held, limits }`, which removes the two parallel
`useState`s at `LineInspector.tsx:481-482`. Note that Shape and DoubleUp reset to null first and
the line queries keep their last value. Keep that difference with an option, or accept the reset
(a one-frame blank).
Risk: low. Effort: small. Mechanical: mostly.

### 5. History and stale comments (medium value, none risk; fold into each batch)

These narrate history or no longer match the code, against CLAUDE.md's comment rule:

| File:line | What it says | Why it goes |
|---|---|---|
| `tabs.tsx:1-3` | "The City tab is gone (T-220)..." | history |
| `tabs.tsx:69-70` | "a letter since T-444: the numbers are for building" | history; keep "unlike the build tools it leaves the tab as it is" |
| `tabs.tsx:438-439` | T-213, T-482 JSX comments | history |
| `SettingsTab.tsx:1-6` | a T-id tour of the file, "T-253: no explanatory notes" | what plus history |
| `SettingsTab.tsx:32, 81, 108-110, 116` | "moved here from the dock's tab row", "their setting is gone", "the bands and dots options are gone" | history |
| `SettingsTab.tsx:144` | "a trial of other rounded fonts" | check whether it is still a trial |
| `CityInspector.tsx:127` | "here now that the tab is gone" | history |
| `others.tsx:256-257` | "the Station / Complex switch is gone" | history |
| `StationRiders.tsx:2` | "the map stays as it was (T-519)" | history |
| `PurposeBreakdown.tsx:75, 90` | "T-253: these were notes under the chart" | history |
| `LoadingScreen.tsx:55, 58` | "T-253: no ... line under ..." | history |
| `tip.ts:7-8` | "write `data-tip` where they once wrote `title`" | history |
| `Toolbar.tsx:2` | "Construction controls and the blueprint quote live together in Build (T-100)" | not this file's business |
| `Search.tsx:62-63` | "the Station / Complex setting stays as the player had it" | stale: no such setting exists now |
| `Search.tsx:1-3` | broken line wrap ("Most of the" alone on a line) | tidy |
| `HelpPanel.tsx:1` | "opened by the "?" in the dock's tab row" | stale: since T-1099 the button is at the top of Settings (`SettingsTab.tsx:33-36`) |
| `widgets.tsx:56` | "T-1006: show a realm, four arrows out to the corners" | orphan: no icon follows it |
| `widgets.tsx:17-69` | T-ids on most icons ("T-252: the dock's tabs, and T-251's save button") | keep the picture words ("a magnet over a curving line"), drop the ids |
| `style.css:641` | "five tools: three on the first row, two on the second" | stale: `TOOLS` has 4 |
| `style.css:537-538` | "no transition or animation, the percentage is the motion" | stale: the bunny animates and the bar replaced the percentage |
| `style.css:229` | "the settings pane's bubble size, T-099" | stale: five sliders use it |
| `style.css:921` | "Focused-city demand and purpose charts (T-143)" heading | misplaced over the demand-pending rule |

Risk: none. Effort: small per file. Mechanical: yes, given the list.

### 6. Split `inspector/others.tsx` (942 lines, five inspectors) (medium value, low risk)

What to change, one file per selection kind:

- `StationInspector.tsx`: `others.tsx:42-436`, with ComplexRow, `useComplexNow`, ComplexSummary,
  ComplexLines, BoardingsToday, StationLinks, StationDetails and Waiting.
- `TrainInspector.tsx`: `:438-546`, with OnBoard.
- `TrackInspector.tsx`: `:548-739`, with Shape, RADII and DoubleUp.
- `JunctionInspector.tsx`: `:741-924`, with parseJunction and JunctionMap.

AreaInspector is deleted (item 2). ComplexMap's header (`ComplexMap.tsx:4`) and `useComplexNow`'s
cross-reference (`ComplexMap.tsx:227-228`) name `others.tsx`; update them, and update code-map.md's
`inspector/` line.
Small things on the way:

- Inline `StationGlyph` (`:42-44`, used once).
- Drop the needless fragment in BoardingsToday (`:199-206`).
- Use `LEVELS` for the literal `[3, 2, 1, 0, -1, -2, -3]` at `:629`.
- `clock.minute.value;` at `:390` and `:440` becomes `void clock.minute.value;` as elsewhere.
- The inline flex style at `:314` becomes a class, shared with `ClosestStations.tsx:44`, which has
  the identical style.
- DoubleUp's inline SVG (`:658`) becomes an `Icon.doubleUp`.

Risk: low (moves only). Effort: medium. Mechanical: yes.

### 7. Take keys and `pickTool` out of `tabs.tsx`, then split it (medium value, low risk)

`TOOL_KEYS`, `TRIP_KEY`, `ACTION_KEYS` and `pickTool` (`tabs.tsx:67-80`) are imported by `main.ts`,
`HelpPanel.tsx`, `SettingsTab.tsx` and `inspector/others.tsx` (`others.tsx:20` imports the whole
tabs module for `pickTool`). `TOOL_KEYS` repeats the keys already in `TOOLS` (`:60-65`).

What to change: a `ui/toolKeys.ts` holding TOOLS, TOOL_KEYS derived from it
(`Object.fromEntries(TOOLS.map(...))`), TRIP_KEY, ACTION_KEYS and pickTool. Then split tabs.tsx (602
lines) into `BuildTab.tsx`, `LinesTab.tsx`, `StationsTab.tsx` and `MoneyTab.tsx`. The shared list
parts (ScopeToggle, Near, byNear, `natural`) go in `ui/listParts.tsx`. `CitiesTab` (`:599-601`) is a
one-line wrapper over Cities.tsx; it can become a PANES entry in Dock once Cities.tsx's rewrite
lands.
Risk: low (moves, and main.ts imports change). Effort: medium. Mechanical: yes.

### 8. Shared plain-text helpers (medium value, low risk; coordinate with game/)

- The upkeep rule sentence ("Upkeep is ${upkeepRule(m)}, whether trains run or not. High-speed track
  costs ${m.upkeepHsr} times as much and single track ${m.upkeepSingle} times.") is written 3 times:
  `tabs.tsx:486`, `others.tsx:701` and `Blueprint.tsx:46`. Add `upkeepText(m, lead)` next to
  `upkeepRule` in game/format.ts.
- "Buys N cars for $X" or "Uses N cars from the depot" is written 3 times in LineInspector
  (`:320-321`, `:342-343`, `:373-374`). Add a `carsBoughtText(k, n)` helper in LineInspector.
- `LEVEL_NAME` (High viaduct ... Deep tunnel) is pasted in `tabs.tsx:49-57` and `Blueprint.tsx:12`.
  Add `levelName(l)` in game/format.ts next to `levelLabel`.
- Plurals ("n thing" or "n things") are hand-rolled at least 8 times: `carsText` and `stationsText`
  (LineInspector `:304, :598`), `plural` (SavesPanel `:41`), `pieces` and `description` (Blueprint
  `:18-19, :55`), `Toolbar.tsx:93`, `StationsFull.tsx:99` and `LoadingScreen.tsx:86`. Add one
  `plural(n, one, many?)`.
- "a, b and c" lists appear 3 times: `andList` (`LineInspector.tsx:599`), `PurposeBreakdown.tsx:87`
  and, as a variant, `SavesPanel.tsx:34-39`. Add one `andList`.
- "Capitalise the first letter" appears 5 times: `TripInspector.tsx:74, 97, 241`,
  `PurposeBreakdown.tsx:91` and `CommutersInspector.tsx:61`.

Risk: low. The output strings must not change. Effort: small. Mechanical: yes.

### 9. Constants copied by hand (medium value, low risk; coordinate with game/ and map/)

- `SEATS_PER_CAR = 44` and `CRUSH_PER_CAR = 160` (sim `api.rs`) are pasted in 5 places:
  `others.tsx:478-479`, `LineInspector.tsx:306`, `map/demandLayers.ts:63`, `map/networkTips.ts:33`
  and `map/renderer.ts:40`. Put them once in game/, for example beside `platformRoom` in boarding.ts.
- The platform limits (60 to 400 m, steps of 20) appear in `tabs.tsx:130` and `others.tsx:351`, and
  the 20 m car in `LineInspector.tsx:333`. Name them `PLATFORM_MIN`, `PLATFORM_MAX` and `CAR_M` in
  one place.

Risk: low. Effort: small. Mechanical: yes.

### 10. Station and line lookups rebuilt per render (medium value, low risk; game/world.ts)

`new Map(w.save.stations.map(...))` is built about 17 times in ui/. TripInspector alone builds the
station-name map 5 times a render: RouteLegs `:145`, routeRows `:268` (called by both OtherRoutes
and mainNote), mainShare `:333` (called inside routeRows), OtherRoutes `:288` and BusiestRoutes
`:363`. `linesAt` (game/world.ts:241) filters every line on every call, and the stops list calls it
for every complex member of every stop (`LineInspector.tsx:650`). StopTransfers (`:281-296`)
filters all lines per stop, and TrainStops (`:91-92`) builds its own lines-at map.

What to change: memoised `stationsById(w)`, `stationNames(w)` and `linesAtMap(w)`, cached in a
WeakMap on `w.save`, in game/world.ts. Then use them in ui/. The map/ copies (`networkTips`,
`tripView`, `stopMarkers`) can follow in the other reviewer's batch.
Risk: low (the save object is replaced on every edit, so a WeakMap on it stays correct).
Effort: small to medium. Mechanical: mostly.

### 11. LineInspector simplifications (medium value, low risk)

- `editingThis` (`:543`) and `editingStops` (`:552`) are the same expression. Keep one, computed
  before the early return.
- `addPassed` (`:592-597`) and `removeStop` (`:600-604`) end the same way (setStops, then update
  `lineDraft`). Add one `setStops(stops)`.
- The colour live-edit (`liveEdit` ref `:477`, `sendLive`, `dragColour`, `commitColour`
  `:560-586`, plus the preview state `:471-475`, `:487`) is about 40 lines of throttling inside the
  component. Move it to a `useLiveColour(line)` hook.
- A line's local time (`clock.hour.value * 3600 + (lineStats[id]?.shift ?? 0)`) is written 5 times:
  `LineInspector.tsx:506, 530`, `tabs.tsx:241, 301` and `others.tsx:490` (and TrainStops `:89-99`).
  Add a `lineSeconds(st)` helper in game/clock.ts or localTime.ts.
- The four Float64Array parsers (`spotsFrom`, `heldFrom`, `limitsFrom`, `marksFrom`, `:44-111`) are
  clock-protocol decoding, not UI. They could live in game/ beside `parseEdgePis`; optional.
- The component `LineToday` (`:195`) has the same name as the type `LineToday` in game/today.ts.
  Rename it `PastDay`.
- The header (`:1-5`) lists the panel's contents with task ids. It can go.

Risk: low. Effort: medium. Mechanical: partly (the hook needs care around the timer cleanup at
`:478`).

### 12. One elevation-buttons widget (low to medium value, low risk)

The level button row is built 3 times: `tabs.tsx:100-106` (Build), `Blueprint.tsx:83-90` (the
Select bar) and `others.tsx:628-634` (a blueprint stretch's Shape). The first two carry the level
colours (`btn lv`, `--lc`, `--lk`); Shape's does not.

What to change: `<LevelButtons value onPick tip disabled>` in widgets.tsx.
Risk: low. Shape's buttons gain the level colours, a small visible change toward consistency.
Effort: small. Mechanical: yes.

### 13. One split-bar component (low to medium value, low risk)

"A bar of coloured segments, each with a hover" is hand-built 5 times: `TripInspector.tsx:119-121`,
`MarketTrip.tsx:173-175`, `CityInspector.tsx:32-36`, `StationRiders.tsx:34-36` and
`PurposeSplit.tsx:51-57`.

What to change: `<SegBar parts={[{ n, cls | colour, tip }]} label>` beside the trip parts (item 1).
Risk: low. Effort: small. Mechanical: yes.

### 14. style.css: dead rules, merged duplicates (medium value, none to low risk)

Dead rules (no element ever gets them; checked over app/src and index.html):

- `.w-period` (`:225`).
- `#tool-notes.ask` (`:650`). Toolbar renders the Ask dialog separately and never adds `ask`.
- `.line-list-actions label` (`:285`), and `.line-list-actions select` in `:286`. The row holds only
  buttons and the scope toggle.
- `.complex-map p` (`:340`). ComplexMap has no `<p>`.
- `.alert.warn .alert-mark` (`:748`). No caller passes `level="warn"` (item 16).

Selectors defined twice or more, to merge:

- `#dock` (`:103` and `:136`) and `.sort-head` (`:213` and `:219`).
- `.blueprint-head` (`:610` and `:611`) and `.blueprint-bar` (`:623` and `:627`, the second
  overriding `gap`).
- `.tool-pick > .btn` (`:643` and `:651`).
- `.save-info` (`:559` scoped, `:895`, `:898`).
- `.track-row` (`:521` and `:634`, two different rows: give Build's its own class).
- `.speed-key` (`:523` and `:674-676`).

Style drift: `.purpose-scope`, `.purpose-item` and `.purpose-mirror` (`:907-919`) use hard-coded
`#8882`, `#8883` and `#8881` and `border-radius: 4px`, where the rest of the UI is square and built
from tokens. Use `--shade` and `--rule`. Better still, replace the hand-built All modes / Train only
pair (`PurposeBreakdown.tsx:92-95`) with `Segmented`, which is a small visible change.

Optional and larger: about 330 of the 1,010 lines style map DOM, not panels (`.tool-marks`,
`.cap-marks`, `.stop-marks`, `.rev-marks`, `.airport-marks`, `.station-ghost`, `.stn-labels` and
the tags, `:655-801`, `:943-957`). They could move to a `map/map.css` imported by main.ts. It is low
value; do it only if the CSS keeps growing.
Risk: none for dead rules and merges, as long as order-dependent overrides are kept. Effort: small.
Mechanical: yes (the merges need care to keep the cascade).

### 15. Needless exports (low value, none risk)

These are exported but used only in their own file:

- `ridersAt` (`ComplexMap.tsx:32`).
- `ModeHead`, `ModeCells`, `ElsewhereRow`, `bySize` (`LongerTrips.tsx:32, 47, 59, 71`).
- `purposeSplit` (`PurposeSplit.tsx:29`).
- `fullStations`, `FullStation` (`StationsFull.tsx:19, 31`).
- `isTextField` (`keys.ts:22`).
- `findStart`, `realNetwork`, `START_CITIES`, `REAL_NETWORKS` (`NewGame.tsx:15, 18, 38, 64`).

Drop the `export`s. code-map.md names `START_CITIES` and `REAL_NETWORKS`, which is still fine as a
description of the file.
Risk: none. Effort: small. Mechanical: yes.

### 16. Small dead bits (low value, none risk)

- `Icon.crossover` and `Icon.pointer` (`widgets.tsx:32-33`): no use anywhere, and the map code
  never imports widgets.
- `Segmented`'s `cls` prop (`widgets.tsx:174, 178`): no caller passes it.
- `Alert`'s `level` (`widgets.tsx:219`): never set to "warn", only used for the two line alerts. Keep
  it if held-back alerts are meant to turn yellow; otherwise drop it and its CSS.
- `key !== "Esc"` (`tabs.tsx:95`): no TOOLS entry has the key Esc.
- `cv.draws` (`ComplexMap.tsx:230, 276`): a frame counter no check or note reads.
- ComplexMap's `onOpen` fallback (`:199`): its only caller passes `onOpen` (`others.tsx:161`). Make
  it required and drop the fallback.
- `data-total`, `data-upkeep` and `data-cost` (`Blueprint.tsx:39, 46, 119`): no check reads them
  (searched for `data-total`, `dataset.total` and the rest over trainworld/).
- `CityInspector.tsx:68` calls `setBubbleColours(...)` during render, writing the `modeScheme`
  signal. `game/ui.ts:179` already does it in an effect. Keep `void display.value` so the swatch
  re-renders on a scheme switch.
- `StationsFull.tsx:74` has a needless alias (`const ZOOM = STATION_BOX_ZOOM`), and its jump at
  `:93-95` repeats `locateStation` (locate.ts:16-21). Use locateStation.
- `MoneyToday` (`tabs.tsx:178-187`) recomputes `lineNet` (`:177`) by hand. Use it.
- `TrainStops.tsx:12`: the local `count` between imports (item 1).

Risk: none. Effort: small. Mechanical: yes.

### 17. Smaller duplications (low value, low risk)

- Blocking keys until a choice: `StartMenu.tsx:78-86` and `LoadingScreen.tsx:109-116` hold the same
  window capture listener. Add a `blockKeysOutside(root, when)` helper.
- `SavesPanel.tsx:66` and `:112` repeat one date format. Add a `fullDate(at)` helper.
- `CITIES.some((c) => c.id === id)` is repeated at `SavesPanel.tsx:35, 48` and `StartMenu.tsx:24`.
  Add an `isCity(id)` helper in game/cities.ts, if it has none.
- Four range sliders in SettingsTab (`:42-58`, `:132-136`) repeat the same label, span and input.
  Add a `<Slider label value min max step tip onInput>`.
- `PurposeBreakdown.tsx:8-9` holds `LABELS` and `HOURS` as hand copies of `PERIODS` (game/periods.ts):
  "6–10" is `${p.from}–${p.to}` and 4 is `p.to - p.from`. Derive them, so a change to the periods
  cannot leave the chart behind.
- The search glass (`Search.tsx:157-160`) is an inline SVG outside the Icon set. Move it to
  `Icon.search`.

Risk: low. Effort: small. Mechanical: yes.

### 18. Smaller reorganisations (low value, low risk)

- `viewStart` (`NewGame.tsx:51-61`) is a camera move, and `locate.ts:13` imports the NewGame sheet
  just for it. It belongs in locate.ts.
- `Toolbar.tsx` holds the tool hint, the notice and the Ask dialog; no toolbar is left. Rename it
  `ToolNotes.tsx` (main.ts import).
- `DemandSelector.tsx` holds MapTopLeft, the demand buttons, the track display and the purpose
  filter. Rename it `MapTopLeft.tsx`; optional.

Risk: low. Effort: small. Mechanical: yes.

### 19. Stations list sort (low value, a small behaviour change)

`tabs.tsx:432` sorts station names with `localeCompare`, while lines use the numeric collator
`natural` (`:42`), whose comment explains the speed reason. Stations "10 Av" and "2 Av" sort as text
("10" first). Using `natural` gives the speed and number-aware order. `others.tsx:111` and
`StationsFull.tsx:70` also use plain localeCompare for tie-breaks, where it does not matter.
Risk: low. The visible order changes for numbered names, which counts as a behaviour change, so it
is listed for a yes. Effort: small. Mechanical: yes.

---

## Batches (one sitting each, owned files named)

Order: 0, then G, then the rest in any order. A to E can run side by side, since they own disjoint
files. F after B (both touch the train bar class).

- **Batch 0: bugs.** Owns style.css lines 44 and 329-331, plus `others.tsx:533` and
  `style.css:353-355` for the class rename, and `Blueprint.tsx:124`. B1, B2, B3. About 15 minutes.
- **Batch G: shared helpers in game/** (with the game/ reviewer's batch).
  **Done 2026-10-24 (appclean): net -7 lines in game/, main.ts and the three map imports. exact, plural, andList, capitalise, levelName, upkeepText and sig3 in game/format.ts; stationsById, stationNames, stationsByNum, linesById, linesByNum and linesAtMap in game/world.ts, one WeakMap on the save, used in game/ and the inspectors; SEATS_PER_CAR, CRUSH_PER_CAR, CAR_M, gauge and stepPlatform in game/boarding.ts (renderer.ts, demandLayers.ts and networkTips.ts import them; the shader string is the renderer agent's); lineSeconds in game/clock.ts; the cell and zone selections gone with AreaInspector. Not done: the hover-path lookups in map/ and holidays.ts.** Owns game/format.ts
  (`exact`, `plural`, `andList`, `capitalise`, `levelName`, `upkeepText`), game/world.ts
  (`stationsById`, `stationNames`, `linesAtMap`), the car and platform constants (item 9), a line
  local-time helper, and game/types.ts `Selection` (item 2's two lines). Items 8, 9 and 10 (the game
  parts only), and the item 11 helper.
- **Batch A: trip and demand panels.**
  **Done 2026-10-24 (appclean): net -23 lines (tripText.ts and RouteParts.tsx included). pctOf, pctShare, minText, num1, roundTo in inspector/tripText.ts; routeRow, legRows and PeriodRows in RouteParts.tsx; SegBar in widgets.tsx for the five split bars; exports and comments trimmed. Not done: the Segmented swap in PurposeBreakdown and the token colours of the purpose rules (both change the look); CityInspector keeps its setBubbleColours call in render (game/ui.ts says views call it again before building).** Owns inspector/TripInspector.tsx, MarketTrip.tsx,
  LongerTrips.tsx, CityInspector.tsx, StationRiders.tsx, PurposeSplit.tsx, PurposeBreakdown.tsx,
  CommutersInspector.tsx, and the new inspector/tripText.ts and RouteParts.tsx. Items 1 and 13, the
  PurposeBreakdown parts of 14 and 17, `setBubbleColours` (16), the item 15 exports in these files,
  and item 5's comments in them.
- **Batch B: station, train, track and junction inspectors.**
  **Done 2026-10-24 (appclean): others.tsx is now StationInspector, TrainInspector, TrackInspector and JunctionInspector (net -50 lines with the four new files), Built and ConstructRemove moved to widgets.tsx, the train's On board bar has its own class .board-bar (B2, today's look copied), ComplexMap's onOpen is required, ridersAt and cv.draws gone, a chip-line class replaces two inline styles, Icon.doubleUp, and useClockQuery or useLatestAnswer in Shape, DoubleUp, JunctionInspector and ComplexMap. TrainStops uses linesAtMap.** Owns inspector/others.tsx (split into
  four), Inspector.tsx, ComplexMap.tsx, TrainStops.tsx, ClosestStations.tsx, and the new
  useClockQuery.ts. Items 2 (the UI half), 4 (its copies here), 6, the ComplexMap bits of 15 and 16,
  and item 5's comments in them.
- **Batch C: line inspector.**
  **Done 2026-10-24 (appclean): LineInspector.tsx -21 lines, plus ui/useClockQuery.ts (34). useClockQuery for its four queries, useLiveColour, carsBought, one setStops, one editingStops, LineToday renamed PastDay, lookups from game/world.ts. Not done: moving the four Float64Array parsers to game/ (optional).** Owns inspector/LineInspector.tsx (and any file split out of it).
  Items 4 (its 3 copies, after B lands the hook, or C writes the hook and B reuses it), 8's cars
  text, and 11.
- **Batch D: tabs, keys and settings.**
  **Done 2026-10-24 (appclean): tabs.tsx is BuildTab, LinesTab, StationsTab and MoneyTab, with listParts.tsx and toolKeys.ts (TOOL_KEYS derived from TOOLS); net +27 lines for the split's headers and imports. LevelButtons widget (Build and the selection bar), the Stations list sorts with the number-aware collator (item 19), Slider in SettingsTab, dead icons, Segmented's cls and Alert's level gone, Dock mounts the Cities list itself. Not done: Icon.search (the glass would change size), Shape's level buttons keep their plain look, MoneyToday's lineNet.** Owns tabs.tsx (split), the new toolKeys.ts and listParts.tsx,
  HelpPanel.tsx, SettingsTab.tsx, Blueprint.tsx, widgets.tsx, Dock.tsx, StationsFull.tsx, and the
  main.ts import lines. Items 7, 12, 19, the widgets and tabs parts of 15 and 16, the slider and
  search icon of 17, and item 5's comments in them.
- **Batch E: dialogs, menus and startup.**
  **Done 2026-10-24 (appclean): net -14 lines with Modal.tsx. Modal and useModalDialog in ui/Modal.tsx (AccountPanel, SavesPanel, Leaderboard, and Ask in ToolNotes.tsx); blockKeysOutside in keys.ts; isCity in game/cities.ts; viewStart moved to locate.ts (NewGame no longer imports the map); Toolbar.tsx renamed ToolNotes.tsx. Not done: DemandSelector.tsx rename (optional).** Owns AccountPanel.tsx, the new Modal.tsx, SavesPanel.tsx,
  Toolbar.tsx, StartMenu.tsx, LoadingScreen.tsx, NewGame.tsx, locate.ts, Search.tsx, tip.ts and
  keys.ts. Items 3, 17's key blocking, dates and `isCity`, 18, NewGame's exports in 15, and item 5's
  comments in them. Run it after Leaderboard.tsx's rewrite, or keep a re-export of `Modal`.
- **Batch F: CSS.**
  **Done 2026-10-24 (appclean): net -10 lines. Dead rules (.w-period, .line-list-actions label and select, .complex-map p, #tool-notes.ask, .alert.warn) and the duplicates of #dock, .sort-head, .blueprint-head, .blueprint-bar, .save-info and .speed-key merged; Build's track row is .build-track-row. Left alone: .shake, .load-bar, the animation: none rule, the purpose rules' token colours, the map-DOM move to map.css.** Owns style.css (after batch 0) and account.css. Item 14, and the CSS comments of
  item 5.

## Bound to the files being rewritten (one line each, not reviewed further)

- TopBar.tsx: its CSS (`.city`, `.split*`, `.bar-box`, `.pct`, `.cash*`, `.clock*`, `.holiday-dot`,
  `style.css:462-504, 816-849`) should be pruned with the rewrite.
- SkyBanner.tsx: nothing noted.
- Cities.tsx: `goToCity` is imported by Search.tsx, and `CitiesTab` in tabs.tsx is a one-line wrapper
  over `CitiesList`. Its CSS is `style.css:187-208`.
- Leaderboard.tsx: imports `Modal` from AccountPanel.tsx (item 3 moves it).
- MoneyHistory.tsx: its own copy of `exact` (`:30`, item 1).
- FareGraph.tsx: nothing beyond `distText`'s `toLocaleString("en-US")`, which is fine.

## For the other reviewers

- game/world.ts `linesAt(station)` and map/pick.ts `linesAt(overlay, px, py, ...)` are two different
  functions with one name.
- game/types.ts lines 246-247 (the dead `cell` and `zone` Selection kinds) are item 2.

## For Anita

- Settled 2026-10-24: "Welcome to anitabuilder!" stays as it is (Anita); item 19's number-aware
  Stations order goes ahead (orchestrator's call).
