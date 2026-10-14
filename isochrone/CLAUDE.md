# isochrone — notes for agents

Read this, then `SPEC.md`, then `TASKS.md`. Everything else on demand.

An interactive world travel-time map: pick a start, choose which ways of travelling and which
infrastructure exist, see how long it takes to reach everywhere and how likely you are to survive
the trip. Started by Ashley (Anita's girlfriend) on 2026-10-13 from Anita's idea; Anita now leads it.

## What each file is for

| File | Holds | Who edits |
|---|---|---|
| `SPEC.md` | What the map is and how it works, **as of now** | agents, rewritten in place |
| `DECISIONS.md` | Dated log of decisions and why; the history the spec does not keep | agents, add only |
| `TASKS.md` | Open agent work, by priority | agents |
| `ARCHIVE.md` | Finished and abandoned tasks, newest first | agents, add only |
| `notes/T-###.md` | Working notes for a task too big for one line | the task's agent |
| `research/` | Background reports with sources, one file per topic | agents |
| `docs/` | Detailed spec for one area, when a SPEC section outgrows it (`resolution.md`) | agents |
| `TODO.md` | Anita's list: her decisions, tests, chores | Anita; agents add under "Added by agents" only |

## Rules

**The spec says what is true now.** When a decision changes, rewrite the affected text so it
states the new design, and add a dated line to `DECISIONS.md` saying what changed and why. Never
append "Update:" paragraphs, and never leave a section that a later one contradicts. When a spec
section passes ~150 lines, move it to `docs/<area>.md` and leave a short summary and link.

**Tasks.** One line each in `TASKS.md`:
`- T-012 [P2] short title. State, owner if taken, link to notes/T-012.md if any.`

- Ids are never reused. Take the next number after the highest in TASKS.md and ARCHIVE.md.
- P1 = blocking or doing now; P2 = this milestone; P3 = later or ideas.
- Taking a task: add `taking, <date>` to its line before starting, so parallel sessions do not
  collide.
- More than ~3 lines of context goes in `notes/T-###.md`, not in TASKS.md.

**A task is done when** the work runs, `SPEC.md` matches it (if design or behaviour changed),
`DECISIONS.md` has any new decision, and the line has moved from `TASKS.md` to the top of
`ARCHIVE.md` with the date and a one-line outcome. An abandoned task moves the same way, saying
why. Found new work along the way? Add it to `TASKS.md`; do not silently widen the current task.

**Shared files, parallel sessions.** Several agents may work here at once. Re-read a doc right
before editing it, make small targeted edits, and never rewrite `TASKS.md`, `DECISIONS.md` or
`ARCHIVE.md` wholesale.

**Project knowledge lives here, not in Claude's memory.**

## Layout and build

- `index.html` + `worker.js`: the page. Plain JS, MapLibre GL 4.7.1 from unpkg, no build step.
  The worker owns the grid and runs the searches; the main thread only paints and handles input.
- `tools/build_grid.py`: the coarse global grid, `data/grid.bin.gz` + `data/grid.json`
  (seconds). `tools/build_tiles.py`: the L3 and L1 tiles for the view refine, `data/tiles/`
  (gitignored, ~45 s). `tools/inspect_tif.py` prints facts about the raw tif. Run them with plain
  `python` (C:\Python39 has rasterio; the maps venv does not).
- `data/raw/` (gitignored): downloads. `data/raw/unzipped/2020_walking_only_friction_surface.geotiff`
  is the Malaria Atlas Project surface (726 MB).
- Served at `localhost:8800/isochrone/` by the maps-wide `serve.py` (already registered).
  Relative URLs only. Never start another server for a page Anita will open.

## Working here

- **Licences.** The MAP friction surfaces are CC BY 4.0: fine to use, credit them in the info
  panel. Check every new source's licence and record it in its research file.
- **Prefer global sources.** This is a whole-world map; a source covering one country is worth it
  only as a check or if trivially easy.
- **Task size.** Split big tasks into stages, each its own id and report, so Anita sees progress.
  A big task's design goes to Anita for review before it is built; small isolated work needs no
  review. An agent whose context gets compacted finishes its stage, writes the rest as tasks, and
  reports. Be kind to subagents: clear briefs, thanks, problems relayed without blame.
- **open** in the spec means Anita decides. Otherwise decide, record why, carry on.
- **UI** follows the maps house style (riders/nycriders): bare title, dark panels, few words,
  sentence-case labels, no transitions, every drawn number gets a hover.
- **Images.** Screenshots and diagnostic plots go in `maps/artifacts/isochrone/` (Git-ignored),
  never in this folder. Headless screenshots: copy `maps/neighborhoods/tools/screenshot.js`,
  make its port `process.env.CDP_PORT`, launch Chrome with `--headless=new
  --enable-unsafe-swiftshader --remote-debugging-address=127.0.0.1` on a private port and a
  `--user-data-dir` in your own scratchpad folder, add `?v=N` to bust the cache, and stop only
  the Chrome processes whose command line holds your profile path.
- **CPU**: at most ~6 of 16 cores; Anita is using the machine. Set `OMP_NUM_THREADS` before numpy.
- **Commits**: nobody commits or pushes here without Anita's word. Subagents never do.
- Asks for Anita go in the "For you" block of the final message and, if they will outlive the
  session, under "Added by agents" in `TODO.md`.
