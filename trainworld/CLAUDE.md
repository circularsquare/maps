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
| `docs/` | Current detailed spec: `demand.md` (4), `track-and-service.md` (6), `interface.md` (8); also `demand-worldwide.md` (demand rules to recheck abroad) and `adding-a-city.md` (the checklist) | agents |
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
  - `src/track/`: the track model (T-040): geometry, costs, crossings, world frames (`frame.rs`, T-125), the network and its edits
    with undo (`world.rs`; blueprint and construct, T-055), run-time profiles, capacity, the
    boarding account (`boarding.rs`: each train's load and each platform's crowd, T-164), saves,
    and `wasm.rs` (`TrackApi`, the clock worker's API, with the drawing tool's routes). In the default build. `cargo run --release --bin track_bench` times edits
    on a synthetic 2,000 km network; feature `track-bench` exports it to WASM. notes/T-040.md.
  - `src/demand/`: the demand kernel, behind feature `demand` (on in the app's build), and
    `api.rs` (`DemandApi`, the demand workers' API), `intercity.rs` (the regional market),
    `warm.rs` (the everyday warm-up demand worker 0 runs at startup, T-173), `days.rs` (the weekend's factors per purpose and period, T-260), `routes.rs` (routed walks round water and drives over OSM roads from the pack, T-275). `cargo test --release --features demand
    nyc_day -- --ignored --nocapture` solves a day on the real pack natively. notes/T-026.md.
- `app/`: Vite 8 + TypeScript 7 + MapLibre GL 6.13 (ESM only: `import * as maplibregl`, worker
  URL passed via `setWorkerUrl`, no `map.transform`) + Preact 11 with signals (TSX, no Vite
  plugin). `src/main.ts` wires everything; `src/wasm/` is generated: `track/` for the clock
  worker, `demand/` for the demand workers (two modules from the one crate, notes/T-051.md).
  - `src/game/`: what the UI reads. `types.ts` (the save's inputs, worker results, `WorldView`,
    `Selection`), `world.ts` (the snapshot signal and the edit functions), `ui.ts` (tab,
    selection, display settings, theme), `clock.ts`, `periods.ts` (the five periods), `days.ts` (weekday and weekend, the shown day, T-260),
    `demand.ts` (demand results: mode split, riders), `boarding.ts` (the boarding account's hour on
    the main thread: riders packed for the clock worker, platform crowds, T-164), `complexLive.ts` (a complex's platform crowds and trains now, for the complex map, T-246), `palette.ts`, `format.ts`, `coords.ts`
    (world metres to lng/lat), `frame.ts` and `registry.json` (the two world frames and the one city registry, `cities.ts` reads it, T-125), `issues.ts` (the track model's refusals in plain words),
    `money.ts` (cash and ledger from the worker, fare income from riders, T-028), `places.ts`
    (station and compass names for the inspectors), `remove.ts` (removing things, with the
    question asked before removing anything constructed, T-096), `loading.ts` (what is loading,
    the startup gate and the clock hold, T-149, T-172), `boundaries.ts` (each city's boundary
    file, fetched once, T-172), `search.ts` (the search box's matching, T-148), `placeList.ts` (each city's OSM place list, fetched on first search or station naming, T-177, T-065), `walks.ts` (walking minutes to the closest stations on the demand model's walking barrier, read from the pack on the main thread, T-312), `complexes.ts` (station complexes from platform geometry and the player's links, T-146), `complexView.ts` (a click selecting the complex, a station opened alone, the zoomed-out dot groups; T-217, T-249), `links.ts` (the player's links, T-197), `purposes.ts` (the trip purposes as shown, T-192), `lineActions.ts` (new and copied lines). The `WorldView` is built from the clock
    worker's states in `workers/clockClient.ts`.
  - `src/map/`: `mapView.ts` (the MapLibre map), `basemap.ts` (the Positron style: faint
    repaint, labels in Zen Maru Gothic from `public/fonts/`), `overlay.ts` (the overlay canvas and frame loop, notes/T-015.md),
    `renderer.ts` (WebGL2 track, stations, trains), `network.ts` (the renderer's buffers),
    `stationGeometry.ts` (platform shapes along the track for drawing, picking and the ghost, T-189, T-254),
    `tripView.ts` (the trip explorer's route on the map, T-195; its panel is
    `ui/inspector/TripInspector.tsx`, its state `game/trip.ts`, its query `sim/src/demand/trip.rs`),
    `stationLabels.ts`, `capacityMarkers.ts`, `stopMarkers.ts` (the toggle markers of stop editing, T-233), `trainFollow.ts` (a selected train followed onto its next trip, T-245), `pick.ts`, `tools.ts` (drawing track, placing
    stations, the line tool), `edgeEdit.ts` (dragging blueprint track's PIs, T-064, T-092), `nodeDrag.ts` (dragging
    blueprint nodes, T-093), `geo.ts`,
    `synth.ts` (stress network), and the demand views (T-084, T-097): `demandViews.ts` (wiring,
    a station's catchment), `demandLayers.ts` (line width, station size, train fill),
    `commuterView.ts` and `bubbles.ts` (commuter bubbles and bubble selections), `intercityView.ts`
    (longer trips of a selection or station, one bubble per far zone, T-118, T-120; one bubble per
    city zoomed far out, T-186), `catchUp.ts` (the bubbles catching up over a game day, T-156), `demandOverlay.ts`
    (their canvas, over the network's), `demandHover.ts` (the count tag by the pointer);
    `trackHover.ts` and `networkTips.ts` (the hover tags on track, trains and stations, T-157,
    T-194), `cityBoundaries.ts` and `boundaryGeometry.ts` (the metro outlines and outside shade,
    T-131, T-139), `labelLayout.ts` (shared spacing for names and tags), `live.ts` (the live map
    objects for UI code, set by main.ts).
  - `src/ui/`: Preact components (`Dock.tsx`: the dock, its icon tabs and resizing, T-252; `tabs.tsx`: the Build, Lines, Stations and Money tabs; `SettingsTab.tsx`; `SavesPanel.tsx`: the saved games panel, T-251; `widgets.tsx`: chips, steppers, checkboxes and the inline SVG icons, the tab icons included; `TopBar.tsx`; `Toolbar.tsx`: tool hints and the confirmation dialog; `DemandSelector.tsx`: None/Homes/Destinations and the purpose filter; `Blueprint.tsx`: Build's quote and Construct; `Debug.tsx`: the `?debug=1` readout; `inspector/`: one inspector per selection kind, `others.tsx` holding station, train, track, junction and cell or zone, `ComplexMap.tsx` the complex map; `LoadingScreen.tsx`: the clock hold at startup, the loading card over the map while it holds, and the screen when demand cannot load, T-149, T-172, T-180; `Search.tsx`: the map's search box, T-148; `StationsFull.tsx`: the "N stations full" button under the demand buttons, T-231; `keys.ts`: which keys a focused control keeps, for main.ts and search alike, T-183; `tip.ts`: the hover tag for every panel and control, from `data-tip`, never `title`, T-210), `style.css` (theme tokens). `src/workers/`: `protocol.ts`
    (edit ops, clock messages, the network snapshot for demand), `clock.worker.ts` (owns the
    network: `TrackApi`, notes/T-025.md; money and autosave too), `saveFile.ts` (the save format
    and IndexedDB, T-029), `clockClient.ts`,
    `demand.worker.ts`, `demandClient.ts`, `demandProtocol.ts`, `demandTest.ts`, `cityReach.ts`
    (which city packs a network's stations can reach, T-149),
    `commuters.worker.ts` and `commutersProtocol.ts` (the commuter bubbles' sums, T-097),
    `selectionMerge.ts` (a bubble selection's everyday far end added up over the workers, T-285).
  - URL: `?debug=1` (readout, `window.tw` probes incl. `lagStart/lagStop`, `checkPrecision`;
    `window.twDemand` with demand timings and `client.qlog` per query, `window.twViews` with the demand views'), `?paused=1`, `?new=1` (a new game instead of the
    autosave), `?synth=1&lines=500&trains=10000`,
    `?demandTest=1` (demand on T-005's hand-made network), `?demandWorkers=N`,
    `?perfOff=...` / `?perfTry=...` (CPU A/B switches, `src/perfFlags.ts`, notes/T-045.md).
- `npm run build --prefix app` = wasm-pack twice (Cargo profile `wasm`) → type check → `app/dist/` (city packs copied into
  `dist/packs/` by `vite.config.ts`), which serve.py serves at `localhost:8800/trainworld/`. `npm run dev --prefix app` for an agent's own loop.
- `pipeline/`: Python city packs, run with `..\venv\Scripts\python.exe` (maps venv).
  `build_city.py nyc` → `data/packs/` (downloads cached in `data/raw/`; the zone gravity is
  solved by the sim crate's `pack_gravity` bin through cargo, built in `data/work/target`), `read_pack.py nyc`
  checks a pack, `look_pack.py nyc` draws `../artifacts/trainworld/T-004.png`. City configs are in `build_city.py`.
  `regions.py nyc` adds each cell's state to a finished pack and re-solves its gravity in place
  (seconds; T-162), the way to re-solve after a decay or border factor change without a rebuild.
  `enrich_all.py phl bos` adds the non-work inputs (everyday sectors, NCES schools, IPEDS
  universities, airports from FAA/BTS/OurAirports, medical and parks from OSM) to finished packs in place,
  about 0.5-2 minutes a city plus Overpass time (T-142).
  `routes.py <ids>` adds routed walking and driving to finished packs (OSM from Geofabrik, cached in `data/raw/osm/`; the sim's `pack_walk` bin; 1-8 minutes a pack; rerun after anything that changes a pack's zones, `--check` lists stale ones; T-275).
  `places.py [ids]` writes each city's OSM place list, `data/packs/<id>.places.json` (towns and neighbourhoods inside the boundary, one Overpass request a city cached in `data/raw/`; T-177), beside the pack, not in it.
  `frame_fixture.py` writes PROJ's values for the world frame (`app/src/game/frame.fixture.json`),
  which `notes/t125-frame-check.mjs` and the Rust frame test compare with.
- Northeast runtime needs `build_city.py` outputs for all registered zones (cities `nyc` (with
  Mercer since T-276), `phl`, `bos`, `wdc`, `bal`, `pvd`, `nhv`, `hfd`, `spf`; non-city zones `wor`,
  `wnd`, `nlc`, `lit`, docs/zones.md), then in order: `pipeline/zones.py` (registers built
  non-city zones), `pipeline/outside.py` (the market's outside world, Kontur),
  `pipeline/intercity.py` (the market's sites, then the joint commute gravity `joint_gravity`,
  which rewrites every zone pack's gravity factors and the market's commute list, about 12
  minutes, then the `derived` block with `market_pack`; `--derive` does only that, seconds: rerun
  it after changing `FORMULA` in `intercity.rs`, T-174), `pipeline/zones.py --trips`, and last
  `pipeline/routes.py <ids>` for any pack whose zones changed (T-275).
  Vite serves/copies all registered packs. Zone packs stay separate; demand workers share a
  regional pool. Registry market places retain NYC0/PHL1/Mercer2 (no sites since T-276)/BOS3/WDC4
  and append Baltimore5/Providence6/NewHaven7/Hartford8/Springfield9 and the non-city zones. The
  regional builder reads this registry.
- Toronto (`tor`, T-277) is registered with `market: -1` (outside the regional market) and built by
  `pipeline/canada.py` (Canadian adapters; `canada.py tor --finish` reruns gravity, purposes and
  checks from the cached cells). Its OSM inputs come from the Geofabrik Ontario extract, not
  Overpass.
- Adding a city: the step-by-step checklist is `docs/adding-a-city.md` (followed end to end once,
  for Toronto; its "Outside the US" section covers other countries).
- Basemap glyph tiles (`app/public/fonts/`, checked in): `pipeline\glyphs.py` rebuilds them, after a
  one-off `cargo install build_pbf_glyphs --locked` (on Windows set `CFLAGS` to a libz-sys zlib
  folder first, see notes/T-056.md). Only needed to change the font, weights or fallback.
- Shells started before Rust was installed lack it on PATH: prepend `%USERPROFILE%\.cargo\bin`.
  Set `CARGO_BUILD_JOBS=6` (CPU cap).

## Working here

- **The goal is the whole world; demand realism is core** (Anita, 2026-10-13). New US cities stay
  paused. Instead a **sampler of other countries' cities** (Toronto done; London, Berlin, Paris,
  Shanghai, Mumbai next, not all at once) to learn which demand methods work everywhere and which
  need a substitute. Each is a **low-fidelity first pass**: if a country has no easy medical data,
  leave medical out; if medical jobs are only in the jobs total, leave them there. Pragmatic
  extents, basic checks, no deep calibration.
- **General sources and methods only** (Anita, 2026-10-12, stressed 2026-10-13). Do not spend time
  or tokens on solutions that work for one city, or even one small country. Prefer global sources
  (OSM, GHSL, Kontur, WorldPop and the like) or a method any national census can feed. A source
  covering one city is worth it only if very easy; one covering one minor city never is. Research
  briefs say this.
- **Intercity demand: retune plus commutes between cities** (Anita, 2026-10-13, option 2 of
  `research/intercity-review.md`), within a world of zones (T-276): city zones, non-city zones
  between cities, and unmodelled space. Plausible is still the bar, not close calibration.
- **Demand rules must hold up worldwide** (Anita, 2026-10-12). `docs/demand-worldwide.md` lists
  demand assumptions to recheck for cities with other density patterns, car ownership and
  incomes. Any task that adds or changes a demand rule adds or updates its entry there.
- **Rebuild from a snapshot** (Anita, 2026-10-13). 8800 serves a build of the last commit, made in
  the orchestrator's snapshot worktree `maps/.tw-snap/` (detached HEAD, kept out of Git by
  `.git/info/exclude`; `trainworld/data` and `trainworld/app/node_modules` are junctions to the main
  tree's; its own `CARGO_TARGET_DIR=maps/.tw-snap-target`, also excluded), registered as `serve.py add trainworld
  .tw-snap/trainworld/app/dist`. So a rebuild never waits for agents mid-edit: commit what is
  finished, check the snapshot out at that commit, build there. Agents still work in the one main
  tree (Anita does not want a worktree per agent).
- **Commits** (Anita, 2026-10-13): the orchestrator may commit trainworld work without asking, not
  very often: at the end of a round or when a snapshot build needs it. Explicit `trainworld/`
  paths only (the maps repo holds other projects' uncommitted work), one plain sentence per
  commit. Subagents do not commit. **Pushing still needs Anita's word.**
- **Task size** (Anita, 2026-10-13). Split big tasks into stages, each its own task id and report,
  so Anita sees progress and can redirect. A big task's design goes to Anita for review before it
  is built; small, isolated work that is quick to build needs no review. An agent that has run
  long enough to have its context compacted winds down: finishes the stage it is on, writes the
  rest as tasks, reports, and a fresh agent takes the next stage. An agent compacting means the
  orchestrator gave it too big a task: size the next briefs smaller. And be nice to subagents:
  kind, clear briefs, thanks for their work, problems relayed without blame (Anita, 2026-10-13).
- **Live feel** (Anita, 2026-10-13). Ask of every figure: if every train and passenger were
  simulated live, would this number change often? If so it should move smoothly, as trains arrive
  and fares are paid (money, riders, waiting counts), not jump once an hour.
- **Every drawn quantity has a tooltip** (Anita, 2026-10-12): anything drawn on the map or in a
  panel that encodes a number (widths, sizes, colours, bars) gets a hover that states the number.
  Always the game's own responsive tag, never a browser `title` tooltip (Anita, 2026-10-12).
- **Panels show what can change** (Anita, 2026-10-13): keep text to a minimum. No explanatory
  notes, how-it-works sentences or flavour text that is useful once ("Your game is saved in this
  browser every minute…"); put such facts in a hover if anywhere. Labels, numbers and states
  that change are what a panel is for. Prefer icons with hovers over word labels where the icon is
  clear.
- **Saves are Anita's only** (2026-10-12). Nobody else has saves yet, so keeping old saves loading
  is a nicety, not a constraint: don't spend real effort on migrations or conversion offers. She
  will say when that changes.
- **Content first** (Anita, 2026-10-10). Performance is adequate. Pick content and gameplay
  tasks; leave optimization work parked until Anita reports a performance issue. Keep the
  existing responsive architecture, but do not start profiling or optimizing proactively.
  Reported 2026-10-13: a click on a demand bubble can take ~3 s to show its other end (T-285:
  fixed when idle; during a re-solve it waits for the solve, T-122 for her review).
- **Keep diagnostic images outside this project.** Screenshots and diagnostic plots go in
  `maps/artifacts/trainworld/` (from this folder: `../artifacts/trainworld/`), ignored by Git.
  Never put captures in the trainworld root or its notes directory. Keep actual game assets
  in their normal source locations. Task notes may link to the external local artifacts.
- **notes/ is the agents'** (Anita, 2026-10-13): move, merge, archive or delete notes freely to
  keep it useful. Check outputs (.json, .log) belong in `../artifacts/trainworld/`, not notes/;
  the old ones and what the notes/ check scripts write now are in
  `../artifacts/trainworld/notes-outputs/` (T-278).
- **open** in the spec means Anita decides. Otherwise decide, record why, and carry on; do not ask
  her to confirm rail facts.
- Asks for Anita go in the "For you" block of your final message and, if they will outlive the
  session, under "Added by agents" in `TODO.md`.
- Subagents are welcome on this project (Anita's standing go-ahead).
- Serve pages through `python serve.py add trainworld <folder with index.html> "rail building game"`
  at the maps root → `localhost:8800/trainworld/`. Relative URLs only. A Vite dev server is fine for
  an agent's own loop; what Anita opens goes through 8800.
- CPU: at most ~6 of 16 cores for builds and tests; she is using the machine.
- Subagents never commit or push; the orchestrator commits as above and never pushes unasked.
