# trainworld — notes for agents

Read this, then `SPEC.md`, then `TASKS.md`. Everything else on demand; where code lives is
`docs/code-map.md`.

## What each file is for

| File | Holds | Who edits |
|---|---|---|
| `SPEC.md` | What the game is and how it works, **as of now**, in summary | agents, rewritten in place |
| `docs/` | The spec in detail, one file per area: `demand.md` (index of the `demand-*.md` files), `regional.md`, `track-and-service.md`, `architecture.md`, `world.md`, `zones.md`, the interface in `ui-*.md` (index: `interface.md`); also `demand-worldwide.md`, `adding-a-city.md`, `code-map.md`, and `data-register.md` (each country's sources by input, their status, and its procedure) | agents |
| `TASKS.md` | Open agent work, by priority | agents |
| `DECISIONS.md` | Dated choices someone might revisit, and why | agents, add only |
| `ARCHIVE.md` | Finished and abandoned tasks, newest first | agents, add only |
| `notes/T-###.md` | Working notes for a task too big for one line; analysis scripts a note cites | the task's agent |
| `checks/` | Standing checks, `smoke.mjs` first (code-map.md lists them) | agents |
| `research/` | Background reports with sources, one file per topic | agents |
| `TODO.md` | Anita's own list | Anita only; agents never edit it |
| `docs/demand-plain.md` | How demand works in plain words, one page, for Anita | any agent that changes a demand rule it mentions |

## Doc rules

- **The spec says what is true now.** When a decision changes, rewrite the affected text to state
  the new design; never append "Update:" paragraphs or leave a section a later one contradicts.
  A spec section past ~150 lines moves to `docs/<area>.md`, leaving a summary and link.
- **Touch only the section your change touches**, in the spec, docs and code-map alike.
- **Docs hold current rules only** (Anita, 2026-10-14): the rule and the numbers that decide it, in
  the present tense. Measurements, how a figure was reached and before/after tables go in notes/.
  Read the sections you need, never whole files when one section will do.
- **Comments** (Anita, 2026-10-14): before writing one, check whether the code already says it; most
  code needs none. A comment says why, only where the why is not evident from the code (a non-obvious
  choice, a unit, an invariant), with a task id if the reasoning lives elsewhere. Never what the code
  does, never history ("before T-275 this..."): that is DECISIONS.md. Trim a function's comments when
  you edit it; no sweeping rewrites.
- **DECISIONS.md and ARCHIVE.md are searched, never read whole**: grep them for a task id or word.
- **A DECISIONS line** only for a choice someone might reasonably revisit: date, one or two
  sentences, the task id. Not for what the spec already says.
- **An ARCHIVE entry** is at most two lines (about 40 words): id, title, date, the outcome only,
  and a notes/ link for detail. Old long entries stay as they are.
- **Tasks.** One line each in `TASKS.md`:
  `- T-012 [P2] short title. State, owner if taken, link to notes/T-012.md if any.`
  Ids are never reused (next after the highest in TASKS.md and ARCHIVE.md). P1 = blocking or
  now, P2 = this milestone, P3 = later or ideas. Add `taking, <date>` before starting. More than
  ~3 lines of context goes in `notes/T-###.md`.
- **A task is done when** the work runs, the spec matches it, any decision is in DECISIONS.md, and
  its line has moved from TASKS.md to the top of ARCHIVE.md. Abandoned tasks move the same way,
  saying why. New work found on the way goes in TASKS.md; do not silently widen the task.
- **Each milestone ends with a doc tidy task**: spec against code, prune TASKS.md, fold finished
  notes into the spec or archive, flag anything stale.
- **Shared files.** Several agents work here at once: re-read a doc right before editing it, make
  small targeted edits, never rewrite TASKS.md, DECISIONS.md or ARCHIVE.md wholesale.
- **notes/ is the agents'** (Anita, 2026-10-13): move, merge or delete freely. One-off scripts go
  once their task is done; scripts worth keeping go to `checks/`. Check outputs (.json, .log) go in
  `../artifacts/trainworld/`, never notes/.
- **Project knowledge lives here**, not in Claude's memory. Adding a city: `docs/adding-a-city.md`.
- **The data register** (`docs/data-register.md`): a city or data task that uses, finds or rules
  out a source updates that country's cell, and files a P3 task for a source usable but not used.

## Agent working rules

Briefs carry only the task; these hold for every agent.

- **Your own Vite**: `node trainworld/checks/dev-nowatch.mjs <port>` (from the maps root; a port in
  6100-6199 nobody else uses), watching and HMR off so others' edits do not reload your page;
  restart it after editing the app. What Anita opens goes through 8800 (`serve.py`), never yours;
  run checks on your own Vite, and never long sweeps against 8800 while she may be playing.
- **Headless Chrome** through Playwright with its own temporary `--user-data-dir`. Stop only the
  PIDs you started; never kill browsers or node broadly.
- **The real save** is `T-007-real-nyc.save` (New York, 109 lines). `?new=1` never writes a slot.
- **Typecheck** with `npm run check --prefix app`. **No full build and no commits**: the
  orchestrator builds and commits. `npm run wasm --prefix app` is fine when Rust changed.
- **Rust**: `CARGO_BUILD_JOBS=3`; keep the crate compiling between edits, since others build it
  too. Shells started before Rust was installed lack it on PATH: prepend `%USERPROFILE%\.cargo\bin`.
- **CPU**: at most 6 of 16 cores in all, every agent together, while Anita is using the machine;
  up to 12 overnight (Anita, 2026-10-16).
- **Calibration of new cities** (Anita, 2026-10-16): cities added through the world plan need no
  calibration beyond the pipeline's gate; a deep calibration comes once everything is in.
- **Bash**: one simple command per call (no `&&`, pipes or loops); scripts go in a file in your
  scratchpad folder, not heredocs.
- **Screenshots and diagnostics** go in `../artifacts/trainworld/`, never in this folder.
- **Player-facing text** follows Anita's rules (her global CLAUDE.md "User-facing text"): no dashes
  between phrases, people act rather than interface nouns, facts not sales or vibes, no
  exclamation marks or softeners, sentence case, no all-caps labels.
- **Stop at a clean point** if the work outgrows one context: finish the stage you are on, write
  the rest as tasks, hand back. Compacting means the task was sized too big.
- **Keep your context lean** (T-288's retrospective): each step re-reads your whole conversation,
  and waking after more than about 5 idle minutes re-caches all of it (29% of London's cost). Wait
  for a long job with one background wait on the process ending (not on a log line, which a script
  may never print: T-292), not progress polling, and never wait in the foreground
  for more than about 4 minutes (T-289, T-291: each such wait cost about 270k); do other work meanwhile. Start
  independent long jobs together and wait once (T-360: six separate waits were still 36%). Write
  long logs to a file and read summaries; read doc sections, not whole files; grep TASKS.md.
- **Reports to the orchestrator are short**: what changed and where, check results, anything for
  Anita. Detail goes in notes/, not the report.
- **open** in the spec means Anita decides. Otherwise decide, record why, and carry on; never ask
  her to confirm rail facts.

## Before a commit (orchestrator)

`node trainworld/checks/smoke.mjs [url]` from the maps root: Node checks, then a new game and the
real save in headless Chrome (counts, demand final, station and train clicks, no console or GL
errors). About half a minute with its own Vite; one PASS or FAIL line per check. After rebuilding
8800's snapshot, run it once more against `http://localhost:8800/trainworld/`, also with
`SMOKE_SHADE=dark`: a production build copies only the pack files `vite.config.ts` lists, and the dev
server serves everything (T-377's market packs were missing there, 2026-10-15).

After each commit, in order:
1. Smoke on the dev server (above).
2. Commit explicit paths only; check the staged diff right before committing, since agents edit shared
   files while you stage.
3. Rebuild 8800: `git -C maps/.tw-snap checkout --detach <commit>`, then in `.tw-snap` with
   `CARGO_TARGET_DIR=maps/.tw-snap-target`, `CARGO_BUILD_JOBS=2` and `%USERPROFILE%\.cargo\bin` on PATH,
   `npm run build` through `cmd /c` (PowerShell treats wasm-pack's stderr as an error).
4. Smoke against `http://localhost:8800/trainworld/`, then again with `SMOKE_SHADE=dark`.

Several agents at once work well when each owns named files; give every brief the others' files to
leave alone.

## Build and run

`app/` (Vite, TypeScript, MapLibre, Preact): `npm run build --prefix app` makes `app/dist/`.
`sim/` (Rust, to WASM): `cargo test --manifest-path sim/Cargo.toml`. `pipeline/` (Python city
packs, `..\venv\Scripts\python.exe`). Details and build orders: `docs/code-map.md`.

## Working here (Anita's decisions)

- **The goal is the whole world; sensible demand is core** (2026-10-13; "sensible" over "realistic" since 2026-10-17, below). New US cities stay paused.
  Instead a sampler of other countries' cities (Toronto done; London, Paris, Shanghai, Mumbai, São Paulo, Lagos, Berlin, Seoul and
  Tokyo, Osaka, Nagoya, Fukuoka, Sapporo, Sendai and Hiroshima, with the rest of Japan as 38 prefecture zones, all at tier 0 with their country's inputs; 67 zones in all) to learn which demand methods work everywhere. Each is a
  low-fidelity first pass: missing data (medical, say) is left out, extents pragmatic, basic
  checks, no deep calibration. The global build is the baseline everywhere; the reach goal is
  tailoring country by country, first for likely-popular countries (China, Japan, the UK, Germany
  and the like) (2026-10-14). Whole-world estimate, 2026-10-14: about 40-60 agent hours (robust
  unattended tier 0, a few sampler cities, automatic city and zone choice, runtime scaling), then
  days of machine time for the build. In tokens (notes/city-costs.md): London's tier 0 city cost
  about 6.8M weighted with first-time tooling; the Japanese batches since came to 1.6-3.4M a pack
  with a country module, so 200 metros about 300-700M, a quarter to three quarters of one week's
  limit in all.
- **Demand is simple on purpose** (Anita, 2026-10-17, replacing T-509's fitted calibration): the
  model aims at sensible and fun, so building rail matters, over fitting real cities. Her two
  ceiling tests are the targets (rail of rail + car: A at least 85%, B at least 50%; notes/T-636.md);
  real cities are loose checks. Think hard before adding any complicating factor to demand (a
  constant per country or purpose, a fitted curve): propose it to Anita first.
- **General sources and methods only** (2026-10-12, stressed 2026-10-13). Prefer global sources
  (OSM, GHSL, Kontur, WorldPop) or a method any national census can feed. A one-city source is
  worth it only if very easy; one for a minor city never is. Research briefs say this.
- **Demand rules must hold up worldwide** (2026-10-12): any task that adds or changes a demand
  rule updates its entry in `docs/demand-worldwide.md`.
- **Intercity demand**: retune plus commutes between cities (2026-10-13, option 2 of
  `research/intercity-review.md`), in a world of zones (T-276). Plausible is the bar.
- **Performance is open work** (2026-10-15, replacing "content first"): quantify each item's impact
  and do the high-impact ones and those that let the game scale to hundreds of cities first.
- **Live feel** (2026-10-13): a figure that would change often if every train and passenger were
  simulated (money, riders, waiting counts) moves smoothly, not once an hour.
- **Every drawn quantity has a tooltip** (2026-10-12): anything encoding a number on the map or in
  a panel gets a hover stating it, always the game's own tag, never a browser `title`.
- **Panels show what can change** (2026-10-13): labels, numbers and states; no explanatory or
  one-time text (put such facts in a hover if anywhere); icons with hovers over word labels.
- **Saves are Anita's only** (2026-10-12): keeping old saves loading is a nicety; no real effort
  on migrations until she says.
- **Task size** (2026-10-13): split big tasks into stages, each its own id and report. A big
  task's design goes to Anita before it is built; small, quick, isolated work needs no review.
  Treat subagents well by substance: a brief they can act on (goal, decisions made, scope, done
  looks like), leave to stop or push back, problems relayed without blame; no formulaic thanks.
- **Token counts** (2026-10-14): hundreds of metros are planned, so cost per city matters. Every city
  task ends with a retrospective (docs/adding-a-city.md 12); the orchestrator records each task's
  tokens with `checks/token_usage.py` and judges the proposed changes, passing edge cases to Anita.
- **Asks for Anita** go in the "For you" block of the final message, never in `TODO.md` (Anita,
  2026-10-15). If there are many, or they outlive the session, put them in `notes/for-anita.md` and
  link it. Subagents are welcome (standing go-ahead).
- **Serving**: `python serve.py add trainworld <folder> "rail building game"` at the maps root, so
  `localhost:8800/trainworld/`; relative URLs only. 8800 serves a build of the last commit from the
  orchestrator's snapshot worktree `maps/.tw-snap/` (detached HEAD, excluded by
  `.git/info/exclude`; `data` and `app/node_modules` are junctions to the main tree's; its own
  `CARGO_TARGET_DIR=maps/.tw-snap-target`), so a rebuild never waits for agents mid-edit. Agents
  work in the one main tree (no worktree per agent).
- **Commits** (2026-10-13): the orchestrator commits trainworld work without asking, not often (end
  of a round, or when a snapshot build needs it); explicit `trainworld/` paths, one plain sentence.
  Subagents never commit. **Pushing needs Anita's word.**
