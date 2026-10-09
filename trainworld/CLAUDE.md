# trainworld — notes for agents

Read this, then `SPEC.md`, then `TASKS.md`. Everything else on demand.

## What each file is for

| File | Holds | Who edits |
|---|---|---|
| `SPEC.md` | What the game is and how it works, **as of now** | agents, rewritten in place |
| `DECISIONS.md` | Dated log of decisions and why; the history the spec does not keep | agents, add only |
| `TASKS.md` | Open agent work, by priority | agents |
| `ARCHIVE.md` | Finished and abandoned tasks, newest first | agents, add only |
| `notes/T-###.md` | Working notes for a task too big for one line | the task's agent |
| `research/` | Background reports with sources, one file per topic | agents |
| `docs/` | Spec sections that outgrew SPEC.md (none yet) | agents |
| `TODO.md` | Anita's list: her decisions, tests, chores | Anita; agents add under "Added by agents" only |

## Rules

**The spec says what is true now.** When a decision changes, rewrite the affected text so it
states the new design, and add a dated line to `DECISIONS.md` saying what changed and why. Never
append "Update:" paragraphs, and never leave a section that a later one contradicts. (religiondots'
spec grew by appending and readers have to hunt for which section overturns which; do not repeat
that.) When a spec section passes ~150 lines, move it to `docs/<area>.md` and leave a short summary
and link in its place.

**Tasks.** One line each in `TASKS.md`:
`- T-012 [P2] short title. State, owner if taken, link to notes/T-012.md if any.`

- Ids are never reused. Take the next number after the highest in TASKS.md and ARCHIVE.md.
- P1 = blocking or doing now; P2 = this milestone; P3 = later or ideas.
- Taking a task: add `taking, <date>` to its line before starting, so parallel sessions do not
  collide.
- More than ~3 lines of context goes in `notes/T-###.md`, not in TASKS.md.

**A task is done when** the work runs, `SPEC.md` matches it (if design or behaviour changed),
`DECISIONS.md` has any new decision, and the line has moved from `TASKS.md` to the top of
`ARCHIVE.md` with the date and a one-line outcome. An abandoned task moves the same way, saying why.
Found new work along the way? Add it to `TASKS.md`; do not silently widen the current task.

**Each milestone ends with a doc tidy task**: check the spec against the code, prune and
re-prioritise TASKS.md, fold finished `notes/` into the spec or archive them, flag anything stale.

**Shared files, parallel sessions.** Several agents may work here at once. Re-read a doc right
before editing it, make small targeted edits, and never rewrite `TASKS.md`, `DECISIONS.md` or
`ARCHIVE.md` wholesale.

**Project knowledge lives here, not in Claude's memory.** Memory is for facts about Anita and
cross-project lessons; anything about this game goes in these files where the next agent will read
it.

## Layout and build

- `sim/`: Rust crate (`trainworld-sim`), compiled to WebAssembly with wasm-pack 0.15.
  `cargo test --manifest-path sim/Cargo.toml` runs its tests natively.
  - `src/track/`: the track model (T-040): geometry, costs, crossings, the network and its edits
    with undo (`world.rs`; blueprint and construct, T-055), run-time profiles, capacity, saves,
    and `wasm.rs` (`TrackApi`, the clock worker's API, with the drawing tool's routes). In the default build. `cargo run --release --bin track_bench` times edits
    on a synthetic 2,000 km network; feature `track-bench` exports it to WASM. notes/T-040.md.
  - `src/demand/`: the demand kernel, behind feature `demand` (on in the app's build), and
    `api.rs` (`DemandApi`, the demand workers' API). `cargo test --release --features demand
    nyc_day -- --ignored --nocapture` solves a day on the real pack natively. notes/T-026.md.
- `app/`: Vite 8 + TypeScript 7 + MapLibre GL 6.13 (ESM only: `import * as maplibregl`, worker
  URL passed via `setWorkerUrl`, no `map.transform`) + Preact 11 with signals (TSX, no Vite
  plugin). `src/main.ts` wires everything; `src/wasm/` is generated: `track/` for the clock
  worker, `demand/` for the demand workers (two modules from the one crate, notes/T-051.md).
  - `src/game/`: what the UI reads. `types.ts` (the save's inputs, worker results, `WorldView`,
    `Selection`), `world.ts` (the snapshot signal and the edit functions), `ui.ts` (tab,
    selection, display settings, theme), `clock.ts`, `periods.ts` (the five periods),
    `demand.ts` (demand results: mode split, riders), `palette.ts`, `format.ts`, `coords.ts`
    (pack metres to lng/lat), `issues.ts` (the track model's refusals in plain words),
    `money.ts` (cash and ledger from the worker, fare income from riders, T-028), `places.ts`
    (station and compass names for the inspectors), `remove.ts` (removing things, with the
    question asked before removing anything constructed, T-096). The `WorldView` is built from the clock
    worker's states in `workers/clockClient.ts`.
  - `src/map/`: `mapView.ts` (the MapLibre map), `basemap.ts` (the Positron style: faint
    repaint, labels in Zen Maru Gothic from `public/fonts/`), `overlay.ts` (the overlay canvas and frame loop, notes/T-015.md),
    `renderer.ts` (WebGL2 track, stations, trains), `network.ts` (the renderer's buffers),
    `stationLabels.ts`, `capacityMarkers.ts`, `pick.ts`, `tools.ts` (drawing track, placing
    stations, the line tool), `edgeEdit.ts` (dragging blueprint track's PIs, T-064, T-092), `nodeDrag.ts` (dragging
    blueprint nodes, T-093), `geo.ts`,
    `synth.ts` (stress network), and the demand views (T-084, T-097): `demandViews.ts` (wiring,
    a station's catchment), `demandLayers.ts` (line width, station size, train fill),
    `commuterView.ts` and `bubbles.ts` (commuter bubbles and bubble selections), `demandOverlay.ts`
    (their canvas, over the network's), `demandHover.ts` (the count tag by the pointer).
  - `src/ui/`: Preact components, `style.css` (theme tokens). `src/workers/`: `protocol.ts`
    (edit ops, clock messages, the network snapshot for demand), `clock.worker.ts` (owns the
    network: `TrackApi`, notes/T-025.md; money and autosave too), `saveFile.ts` (the save format
    and IndexedDB, T-029), `clockClient.ts`,
    `demand.worker.ts`, `demandClient.ts`, `demandProtocol.ts`, `demandTest.ts`,
    `commuters.worker.ts` and `commutersProtocol.ts` (the commuter bubbles' sums, T-097).
  - URL: `?debug=1` (readout, `window.tw` probes incl. `lagStart/lagStop`, `checkPrecision`;
    `window.twDemand` with demand timings, `window.twViews` with the demand views'), `?paused=1`, `?new=1` (a new game instead of the
    autosave), `?synth=1&lines=500&trains=10000`,
    `?demandTest=1` (demand on T-005's hand-made network), `?demandWorkers=N`,
    `?perfOff=...` / `?perfTry=...` (CPU A/B switches, `src/perfFlags.ts`, notes/T-045.md).
- `npm run build --prefix app` = wasm-pack twice (Cargo profile `wasm`) → type check → `app/dist/` (city packs copied into
  `dist/packs/` by `vite.config.ts`), which serve.py serves at `localhost:8800/trainworld/`. `npm run dev --prefix app` for an agent's own loop.
- `pipeline/`: Python city packs, run with `..\venv\Scripts\python.exe` (maps venv).
  `build_city.py nyc` → `data/packs/` (downloads cached in `data/raw/`; the zone gravity is
  solved by the sim crate's `pack_gravity` bin through cargo, built in `data/work/target`), `read_pack.py nyc`
  checks a pack, `look_pack.py nyc` draws `T-004.png`. City configs are in `build_city.py`.
- Basemap glyph tiles (`app/public/fonts/`, checked in): `pipeline\glyphs.py` rebuilds them, after a
  one-off `cargo install build_pbf_glyphs --locked` (on Windows set `CFLAGS` to a libz-sys zlib
  folder first, see notes/T-056.md). Only needed to change the font, weights or fallback.
- Shells started before Rust was installed lack it on PATH: prepend `%USERPROFILE%\.cargo\bin`.
  Set `CARGO_BUILD_JOBS=6` (CPU cap).

## Working here

- **open** in the spec means Anita decides. Otherwise decide, record why, and carry on; do not ask
  her to confirm rail facts.
- Asks for Anita go in the "For you" block of your final message and, if they will outlive the
  session, under "Added by agents" in `TODO.md`.
- Subagents are welcome on this project (Anita's standing go-ahead).
- Serve pages through `python serve.py add trainworld <folder with index.html> "rail building game"`
  at the maps root → `localhost:8800/trainworld/`. Relative URLs only. A Vite dev server is fine for
  an agent's own loop; what Anita opens goes through 8800.
- CPU: at most ~6 of 16 cores for builds and tests; she is using the machine.
- Never commit or push unless Anita says so in that message.
