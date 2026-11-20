# T-2374 tools.ts split (toolsplit)

Moves only. `Tools` stays one class in `tools.ts`; the methods of its station, touch, delete and marks parts became functions in four new files, and its constants and types went to `toolParts.ts`. Nothing was rewritten.

## Files and lines (app/src/map/)

| File | Lines | Holds |
|---|---|---|
| `tools.ts` | 1,359 (was 2,406) | the class: fields, constructor, snapping, events, the track tool (`trackClick`, `segmentClick`, previews), the line tool, `place`, and `altsKey`, `verdict`, `stationSpot` (checks call them as `window.tw.tools.verdict()` etc.) |
| `stationTool.ts` | 450 | `stationClick`, the name (`nameKey`, `nameAhead`, `shownNameAt`, `nameFor`, `lastPlacedName`), the trips near (`askLegs`, `tripsNear`, `sumTripsNear`), the marks (`el`, `svgMark`), pending stations, the ghost (`stationGhost`, `boxGhost`, `ghostShape`, `ghostPath`, `outsideSide`, `stationRings`, `checkGhost`), `layout`, `roomSnap` |
| `toolMarks.ts` | 230 | `curveCircle`, `curveTangent`, `drawBand`, `placeMarks`, `showTag`, `helpLines` |
| `toolTouch.ts` | 178 | `nearTouch`, `evAt`, `touchConfirm`, `touchBack`, `touchStop`, `dragHandle`, `touchState`, `placeTouch` |
| `toolDelete.ts` | 157 | `deletePick`, `stationUnder`, `clearDelHover`, `deleteHover`, `deleteClick` |
| `toolParts.ts` | 114 | the constants, `Snap`, `Spot`, `Verdict`, `Placed`, `LngLatish`, `lengthOf`, `trimPolyline`, `placedAt`, `withName`, `tooFarText` |

The longest function is the `Tools` constructor, about 150 lines. Importers (`main.ts`, `live.ts`) are unchanged: only `Tools` is exported.

## What the move forced

- A moved method is `export function name(tk: Tools, ...)`; `this` became `tk` (`_tk` in `shownNameAt` and `sumTripsNear`, which never use it). `t` is taken by locals in these bodies.
- The `private` modifiers are gone from the class (TypeScript only), and the constructor's parameter properties are `readonly`.
- Calls read `name(this, ...)` in the class and `name(tk, ...)` in the parts. The parts import each other; the cycle is harmless since every cross-reference is a function declaration used at call time.
- Module constants and types gained `export`. Import lines were pruned per file; a one-line header comment in each new file. No other line was edited.
- `altsKey`, `verdict`, `stationSpot` were left as methods: three standing checks (t1870, t1964, t671-ghost-hover) call them on `window.tw.tools`.

## Checks

- `npm run check --prefix app`: clean.
- Smoke (`TW_CORES=4`, own Vite on 6168): 30 PASS, no FAIL.
- t1870-station-follow, t1811-track-tool, t1805-placement, t2430-station-tag, t1964-station-joints, t3153-trackmode, t3069-snap-bar, t3379-single-fixes: all pass.
- t671-ghost-hover: "ghost corrected after its first frame near track ends" fails (21 of 84) and fails the same on the original tools.ts (21 of 84); the overlapping-tracks line passes in both.
- The move was done by a script from the pristine file (not kept); `tools.ts` calls the moved functions as `name(this, ...)`.
