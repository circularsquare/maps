# T-2374 renderer.ts split (drawsplit)

Moves only. `draw` and its helpers, the stroke buffer code and the constants left `renderer.ts`; nothing was rewritten.

## Files and lines

| File | Lines | Holds |
|---|---|---|
| `app/src/map/renderer.ts` | 1,155 (was 2,507) | `NetworkRenderer`: fields, setters, picking, `setNetwork`, trips and staging. `draw` is a one-line call to `drawFrame`. Re-exports `EPOCH_S`, `laneState`, `passingState`, `CROWD_ZOOM`, `CONFLICT_ZOOM` and the types `StationGroups`, `DrawOptions`, `NetworkLayers`, so importers are unchanged. |
| `app/src/map/rendererDraw.ts` | 835 | `drawFrame` and its passes, below; `guardTrains`, `guardTrainsOn`, `useBridges`, `projector`, `drawGlow`, `syncGroups`, `syncSelection`, `bindTrainTextures` |
| `app/src/map/rendererStrokes.ts` | 366 | `texture`, `buffer`, `attrib`, `strokes`, `checkStrokes`, `checkFull`, `pickStrokes`, `shownOf`, `bandWmul`, `bandFull`, `workBand`, `workLod`, `bindAt`, `dropLod`, `rewrite` |
| `app/src/map/rendererParts.ts` | 253 | the constants, `DrawOptions`, `StrokeVao` and the other interfaces, `plainWmul`, `fullAttr`, `laneState`, `passingState` |

The longest function is `stationsPass` at 218 lines (`draw` was 592).

## The passes (in `rendererDraw.ts`, run in this order by `drawFrame`)

`frameSetup` (sizes, station style, projector, stroke cache, `redraw`), `beginStrokes` (cache layer 0, car parks, bus lines),
`strokeUniforms` (the stroke program's uniforms and lane, band and blueprint numbers), `strokeDraws` (the `stroke` and
`trackStrokes` closures), `strokesPass` (highlight, track, lines, ghost, marks, chevrons, the cache layers),
`stationsPass` (hall crowds, platforms, bridges, connectors, stencil, crowds, walkers, bus stops at stations),
`trainsPass`, then the unbinding at the end. Each pass reads the numbers earlier ones worked out from one object (`c1`, `c2`, `c3`
in `drawFrame`), destructured at the top of the pass; the code below it is the old text with `this` as `rnd`.

## What the move forced

- `this` became the first parameter `rnd` (not `r`: `stationsPass` has a local `r`). A function that never uses it names it `_rnd`.
- The `private` modifier is gone from the class's members, since the moved functions reach them (TypeScript only; no runtime change).
- Module-level constants and interfaces gained `export`.
- Calls to moved methods read `name(this, ...)` instead of `this.name(...)`.
- The unused `k` and `G` are left out of `stationsPass`'s destructuring (it declares its own).
- Whitespace: moved code is dedented two spaces. No comment text changed.

## Checks

- `npm run check --prefix app`: passes.
- Screenshots on T-007-real-nyc.save, trips final, clock held, light and dark theme, five views each (city 11.2, mid 13.2,
  street 15.6 at Midtown, street 15.2 in Brooklyn, far 6): before (original renderer.ts in this tree) and after are
  pixel-identical in all 10 (the debug readout's corner masked). The first run against master's Vite was not: master had moved on
  (bus stop work), so the baseline is this tree's own original file. The save has no bus or tram line and the schematic is drawn by the
  diagram, not renderer.ts, so those were not captured; the bus pass is `beginStrokes`'s unchanged call of `BusLayer.draw`.
- Frame time (200 jumpTo frames at 13.2, debug `drawMs` avg; the machine was shared): before 4.56 and 3.91 ms, after 4.60 and 4.18 ms;
  GPU 1.13-1.17 ms before, 1.15-1.16 after. No difference beyond the noise.
- `t1799-bands.mjs`: all PASS. `t1965-follow.mjs`: its "none drawn through each other" fails (63 frames) and fails the same on the
  original file (58 frames); the other four lines pass in both.
- Pair for Anita: `artifacts/trainworld/t2374-draw-before.png` and `t2374-draw-after.png` (dark, street view).
