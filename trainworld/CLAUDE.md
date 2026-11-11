# trainworld — notes for agents

Read this, then `SPEC.md`. Grep `TASKS.md` for your task id rather than reading it. Everything else on demand:
for how a mechanic works, the wiki first (`wiki/index.md` lists the pages; read `wiki/read/<page>.md`, which has the
figures filled in; each page ends with the doc section and code that own the exact rule), then the docs for what the
wiki leaves out (Anita, 2026-11-10). A change to a rule a wiki page states updates that page too (`wiki/RULES.md`).
Where code lives is `docs/code-map.md`.

## What each file is for

| File | Holds | Who edits |
|---|---|---|
| `SPEC.md` | What the game is and how it works, **as of now**, in summary | agents, rewritten in place |
| `docs/` | The spec in detail, one file per area: `demand.md` (index of the `demand-*.md` files), `regional.md`, `track-and-service.md`, `architecture.md`, `world.md`, `zones.md`, the interface in `ui-*.md` (index: `interface.md`); also `demand-worldwide.md`, `adding-a-country.md` (the playbook for a new country), `world-run.md` (the runner, the plan's rule), `adding-a-city-by-hand.md` (a hand-built city or country module), `code-map.md`, and `data-register.md` (each country's sources by input, their status, and its procedure) | agents |
| `TASKS.md` | Open agent work, by priority | agents |
| `DECISIONS.md` | Dated choices someone might revisit, and why; from 2026-10-20 on. Older ones: `DECISIONS-to-2026-10-19.md` | agents, add only |
| `ARCHIVE.md` | Finished and abandoned tasks, newest first; from 2026-10-20 on. Older ones: `ARCHIVE-to-2026-10-19.md` | agents, add only |
| `notes/T-###.md` | Working notes for a task too big for one line; analysis scripts a note cites | the task's agent |
| `notes/handoff.md` | The orchestrator's live state, kept current so a new one can take over any time (`/takeover`, `/handoff`; maps/CLAUDE.md) | orchestrator |
| `checks/` | Standing checks, `smoke.mjs` first (code-map.md lists them) | agents |
| `research/` | Background reports with sources, one file per topic | agents |
| `TODO.md` | Anita's own list, where her playtest notes land: read it and file new items as tasks. Its short-term section is a batch of prompts (Anita, 2026-10-29): copy each item's text verbatim into the brief, examples and all, rather than paraphrasing it (a paraphrase lost her formula tooltip, T-1839). Don't file the long-term section unless she asks | Anita only; agents never edit it |
| `docs/demand-plain.md` | How demand works in plain words, one page, for Anita | any agent that changes a demand rule it mentions |

## Doc rules

- **The spec says what is true now.** When a rule or behaviour changes, rewrite the affected text to
  state the new design; never append "Update:" paragraphs or leave a section a later one contradicts.
  A spec section past ~150 lines moves to `docs/<area>.md`, leaving a summary and link.
  A small UI tweak (a label, spacing, a colour) updates a doc only if a rule changed; the code is the
  reference for those details (Anita, 2026-11-09).
- **Touch only the section your change touches**, in the spec, docs and code-map alike.
- **Docs hold current rules only** (Anita, 2026-10-14): the rule and the numbers that decide it, in
  the present tense. Measurements, how a figure was reached and before/after tables go in notes/.
  Read the sections you need, never whole files when one section will do.
- **Comments** (Anita, 2026-10-14): before writing one, check whether the code already says it; most
  code needs none. A comment says why, only where the why is not evident from the code (a non-obvious
  choice, a unit, an invariant), with a task id if the reasoning lives elsewhere. Never what the code
  does, never history ("before T-275 this..."): that is DECISIONS.md. Trim a function's comments when
  you edit it; no sweeping rewrites.
- **DECISIONS.md and ARCHIVE.md are searched, never read whole**: grep them for a task id or word,
  and grep `DECISIONS-to-2026-10-19.md` and `ARCHIVE-to-2026-10-19.md` too when an id or word is not
  in the live file (they hold everything dated before 2026-10-20; new entries still go in the live files).
- **A DECISIONS line** only for a choice someone might reasonably revisit: date, one or two
  sentences, the task id. Not for what the spec already says.
- **An ARCHIVE entry** is short: id, title, date, the outcome only, and a notes/ link for detail.
  Old long entries stay as they are.
- **Tasks.** One line each in `TASKS.md`:
  `- T-012 [P2] short title. State, owner if taken, link to notes/T-012.md if any.`
  Ids are never reused (next after the highest in TASKS.md and ARCHIVE.md). With several agents
  at once, the orchestrator hands out ids: a brief names the ids its agent may file, or the agent
  asks (two agents picking "the next id" at once collided, 2026-10-22). P1 = blocking or
  now, P2 = this milestone, P3 = later or ideas. Tasks under TASKS.md's "Someday" are real open tasks at a lower
  priority (Anita, 2026-10-26): include them when asked what is open, and bring them up when
  nothing else can go ahead. Add `taking, <date>` before starting. More than
  ~3 lines of context goes in `notes/T-###.md`.
- **A task is done when** the work runs, the spec matches it, any decision is in DECISIONS.md, and
  its line has moved from TASKS.md to the top of ARCHIVE.md. Abandoned tasks move the same way,
  saying why. When archiving, grep TASKS.md and notes/for-anita.md for the id and update every line
  naming it; if it was a parent's last open stage, archive the parent in the same edit; and name in
  your report any older open task your work also did (an audit on 2026-11-04 closed 13 lines left open
  these ways). New work found on the way goes in TASKS.md; do not silently widen the task.
- **Shared files.** Several agents work here at once: re-read a doc right before editing it, make
  small targeted edits, never rewrite TASKS.md, DECISIONS.md or ARCHIVE.md wholesale.
- **notes/ is the agents'** (Anita, 2026-10-13): move, merge or delete freely. One-off scripts go
  once their task is done; scripts worth keeping go to `checks/`. Check outputs (.json, .log) go in
  `../artifacts/trainworld/`, never notes/. A `notes/` link that does not resolve is in `notes/archive/<band>/`
  (bands by task id: `t0001-0499`, `t0500-0999`, `t1000-1499`, `t1500-1999`, `t2000-2499`).
- **Project knowledge lives here**, not in Claude's memory. Adding a country: `docs/adding-a-country.md`; its scripts are in `tools/country/`.
- **The data register** (`docs/data-register.md`): a city or data task that uses, finds or rules
  out a source updates that country's cell, and files a P3 task for a source usable but not used.

## Agent working rules

Briefs carry only the task; these hold for every agent.

- **Before starting**, check `git status` and the "taking" marks in TASKS.md; several agents work at
  once, each owning named files, so leave others' alone.
- **Your own Vite**: `node trainworld/checks/dev-nowatch.mjs <port>` (from the maps root; a port in
  6100-6199 nobody else uses), watching and HMR off so others' edits do not reload your page;
  restart it after editing the app. What Anita opens goes through 8800 (`serve.py`), never yours;
  run checks on your own Vite, and never long sweeps against 8800 while she may be playing.
- **Agent worktrees** (T-792; Anita 2026-10-21: used sparingly). Most agents work in the main tree.
  A code agent whose edits would collide with another's gets its own tree from the orchestrator:
  `node trainworld/checks/agent-tree.mjs new <name>` makes `maps/.tw-trees/<name>` on branch
  `tw/<name>` from master (`data`, `app/node_modules`, `artifacts/trainworld` and `venv` junctioned
  to the main tree's, WASM bindings copied, Rust building into its own `sim/target`). The agent runs
  everything from that folder as its maps root and does not commit; TASKS, ARCHIVE, DECISIONS and
  data or pipeline batches are edited in the main tree only. The orchestrator runs smoke in the
  tree, then `agent-tree.mjs land <name> "<one sentence>"` (one commit on master, fast-forwarded),
  then `remove <name>`; a landing that changed sim/ rebuilds the main tree's WASM. About 1.2 GB a tree before its Rust build; two or three at most.
  **An agent that changes the WASM interface** (adds, renames or removes a function the app calls)
  gets its own tree while others run pages in the main tree; in the main tree it builds the Rust
  function before any app code calls it (such breaks stopped every agent's demand loading, 2026-10-22).
- **Headless Chrome** through Playwright with its own temporary `--user-data-dir`. Stop only the
  PIDs you started; never kill browsers or node broadly.
- **The real save** is `T-007-real-nyc.save` (New York, 109 lines). `?new=1` never writes a slot.
- **Typecheck** with `npm run check --prefix app`. **No full build and no commits**: the
  orchestrator builds and commits. `npm run wasm --prefix app` is fine when Rust changed.
- **Rust**: `CARGO_BUILD_JOBS=3`; keep the crate compiling between edits, since others build it
  too. Shells started before Rust was installed lack it on PATH: prepend `%USERPROFILE%\.cargo\bin`.
  After any sim/ change run both `cargo test --release` and `cargo test --release --features demand --lib`
  from trainworld/sim: the demand kernel's tests run only with the feature (two broke unseen, T-2250).
- **CPU**: about 14 of 16 cores in all, every agent together (Anita, 2026-11-09: this game is the priority; don't let
  core limits slow development). Playwright checks (anything through `launchChrome`) claim their cores
  themselves when `TW_CORES` is unset, so do not claim by hand for them; `TW_CORES_WANT=n` asks for more. Hand claims are for the
  pipeline, Rust and scripts that launch their own browser: `python cores.py claim <n> <name> --pid <pid>`
  at the maps root (see maps/CLAUDE.md); the claim prints what you got, so no `list` first (T-2831).
  A browser load counts its demand workers too (about a core each, plus the page): a headless load
  of a big save took 5 cores on a 4-core claim and starved Anita's game (2026-10-27); cap the
  page's solving workers to fit the claim. Playwright checks do this themselves (`checks/cores.mjs`, through
  `launchChrome`, T-2630): `TW_CORES=n` or a ledger claim, Chrome pinned to the top n cores, `?demandWorkers=n-1`;
  `TW_CORE_MASK` picks the cores, `TW_NOPIN=1` turns it off.
- **While a join runs, nothing else writes `data/packs` or `data/terrain`, any country's included**: each unit's
  re-gate compares every registered pack against its snapshot and parks the unit on any change (a walk
  rerun parked China's Henan, 2026-10-24).
- **Pack rebuilds** back up to `data/work/pre-T###/` first; a backup goes, without asking, once a
  later state is checked clean (Anita, 2026-10-22). `data/work/res9/` stays.
- **A refused permission check**: make the step safer, review, retry; don't wait for Anita.
- **AWS** (a friend's account, shared free; long data builds only): read `docs/aws.md` before any
  `aws` call, and create or destroy only through `tools/aws/*.py`.
- **Calibration of new cities** (Anita, 2026-10-16): cities added through the world plan need no
  calibration beyond the pipeline's gate. No big calibration at the end either: it happens in
  parts, task by task (Anita, 2026-10-22).
- **Bash**: chaining simple commands is fine (Anita, 2026-11-09); scripts go in a file in your
  scratchpad folder, not heredocs. **Never pass `cmd /c start ... /b /wait /affinity` through Git Bash**: Bash turns
  `/b` into `B:/` and strips the quotes from a `start "title"`, so Windows pops "cannot find" dialogs and probes the B
  drive on Anita's screen (2026-11-06). Pin a process from PowerShell (`cmd /c start '""' /b /wait /affinity F000 node ...`)
  or from a `.cmd` file in your scratchpad.
- **Screenshots and diagnostics** go in `../artifacts/trainworld/`, never in this folder. The one
  or two images Anita should look at now, the orchestrator copies to `../artifacts/trainworld/review/`
  (short names, older copies deleted) and gives her that single full path (Anita, 2026-10-27).
- **Player-facing text** follows the "User-facing text" section of Anita's global CLAUDE.md. It says "trips"
  for what people want to take (demand, trips between places) and "rides" or "riders" only for what
  happens on the player's system (Anita, 2026-10-26). **Anita reads every player-facing string
  before it lands** (2026-10-29, after "grey fills a platform as riders wait ... from street zooms"
  got through): an agent's report lists each string it added or changed, old -> new, exactly as shown;
  the orchestrator shows them to her in the conversation. Landing and deploying need no wait for
  her approval (2026-10-29): check the style rules, land, and make sure she sees the strings; she
  answers "text acknowledged" or names the lines to change (any she doesn't mention are fine).
  Agents propose wording; never ask her to write it from scratch. A string no one has listed has
  not been reviewed. **Zone, region and realm names are the exception** (Anita, 2026-11-03: "lets not require my
  confirmation for region name text, i dont know much about specific places"): agents decide them and list
  none for her. **The wiki is not player-facing** (Anita, 2026-11-11: "and wont be for a while"): wiki text is never
  listed for her review. Watch hardest for an object or colour as the
  subject ("grey fills", "red when"), telegraphic phrases with words dropped ("from street zooms"),
  and hovers that describe what the player can already see. "No interaction instructions" means ordinary
  controls ("Click to type"); a gesture with no visible control gets its instruction in a tag (Anita,
  2026-11-09, of "Click to close the loop": "i think we do need it here").
- **Stop at a clean point** if the work outgrows one context: finish the stage you are on, write
  the rest as tasks, hand back. Compacting means the task was sized too big.
- **Keep your context lean** (T-288's retrospective): each step re-reads your whole conversation,
  and waking after more than about 5 idle minutes re-caches all of it (T-2831: a poll costs about
  0.1x the context, a wake 1.25x). A wait under about 4 minutes is one foreground call. A longer one
  runs in the background, with at most one light check of the process every 4-5 minutes, and a wait
  on the process ending, not on a log line (a script may never print it). Read a job's log only when it ends; do
  other work meanwhile. Start independent long jobs together and wait once. A task's final checks (smoke and
  the like, a few minutes) run in the foreground: an agent that ended its turn waiting on them
  stalled for hours (2026-10-27). Write long logs to a file and read
  summaries; read doc sections, not whole files; grep TASKS.md.
- **Long runs are checked, not trusted** (T-613, 2026-10-19). Before a long run, check the sim
  copy's `data`/`app` junctions and the paths it will write; after it ends, read `world_run.py
  status` and report the shipped and parked counts with causes, never just "waiting". The
  orchestrator checks the process and the status table at least every couple of hours, and asks
  after any agent quiet for more than about 45 minutes without a report (Anita, 2026-10-27). A batch
  agent may start subagents (one region or one fault each) when that is quicker than one queue.
  **Start every long run detached** (a `.cmd` or `start` that outlives the session, its log to a
  file; a run started from a session dies with it) and write its batch id, log path, liveness check
  and restart command into the task's notes before waiting.
- **Reports to the orchestrator are short**: what changed and where, check results, anything for
  Anita. Detail goes in notes/, not the report.
- **open** in the spec means Anita decides. Otherwise decide, record why, and carry on; never ask
  her to confirm rail facts. Design questions with a sensible default (which cities to suggest, whether
  a click moves the camera, how old saves migrate) are yours too: decide, record in DECISIONS.md and
  tell her in the report (Anita, 2026-10-19). Ask only where the choice changes what the game is.
  Decide and show her after, visible calls included (2026-10-21); she playtests herself, so no
  gameplay checks after every batch, and a slip gets a fix rather than a new standing process.

## Before a commit (orchestrator)

`node trainworld/checks/smoke.mjs [url]` from the maps root: Node checks, then a new game and the
real save in headless Chrome (counts, demand final, station and train clicks, no console or GL
errors); about 3 minutes, one PASS or FAIL line per check. In order:
1. Smoke on the dev server. Rounds smoke on dev only (Anita, 2026-11-09).
2. Commit explicit paths only; check the staged diff right before committing, since agents edit shared
   files while you stage.
3. Rebuild 8800 with `checks/snapbuild.cmd <commit>` (`snapbuild-frozen.cmd <commit> <freeze>` while a
   country run owns `data/packs`): it checks out `.tw-snap` at the commit, builds into `dist-new` and
   swaps it in. Run it through `cmd /c`, its output to a log (PowerShell treats wasm-pack's stderr as an
   error). Rebuilding 8800 for Anita to look at needs no smoke.
4. Smoke 8800 once in light and once in dark, only before a publish: `checks/smoke8800.cmd` (logs in
   `artifacts/trainworld/smoke8800-*.log`; it skips the Node checks, which read the working tree). A
   production build copies only the pack files `vite.config.ts` lists, while the dev server serves
   everything (T-377).
5. Landing a tree or a batch: skip your own smoke when the agent's smoke ran on the final state; run
   one smoke on master after a batch lands (Anita, 2026-11-09; a failure is then fixed or reverted).

`notes/handoff.md` is edited when the plan changes, folded into that turn's other bookkeeping, not once
per message (Anita, 2026-11-09).

Several agents at once work well when each owns named files; give every brief the others' files to
leave alone.

## Build and run

`app/` (Vite, TypeScript, MapLibre, Preact): `npm run build --prefix app` makes `app/dist/`.
`sim/` (Rust, to WASM): `cargo test --manifest-path sim/Cargo.toml`. `pipeline/` (Python city
packs, `..\venv\Scripts\python.exe`). Details and build orders: `docs/code-map.md`.

## Working here (Anita's decisions)

- **The goal is the whole world; sensible demand is core** (2026-10-13; "sensible" over
  "realistic" since 2026-10-17, below). New countries come when compute is spare, after rebuilds of built
  ones (Anita, 2026-10-24): Korea, Canada (its cities and the corridors between, not the sparse
  north), Mexico, Turkey, Vietnam, in an order data difficulty may change. The global build (tier 0) is the
  baseline everywhere, each place a low-fidelity first pass: missing data (medical, say) left out,
  extents pragmatic, basic checks, no deep calibration. The reach goal is tailoring country by
  country, first for likely-popular countries (China, Japan, the UK, Germany and the like)
  (2026-10-14). Token costs per city: notes/city-costs.md.
- **Demand is simple on purpose** (Anita, 2026-10-17, replacing T-509's fitted calibration): the
  model aims at sensible and fun, so building rail matters, over fitting real cities. Her two
  ceiling tests are the targets (rail of rail + car: A at least 85%, B at least 50%; notes/T-636.md);
  real cities are loose checks. Think hard before adding any complicating factor to demand (a
  constant per country or purpose, a fitted curve): propose it to Anita first.
- **Every demand task has a "done when"** (Anita, 2026-10-27, after a day of rounds that chased
  single real pairs): its TASKS.md line and brief state numbers that end it, plus "would a player
  building this see something obviously wrong?". Once met, stop: a further round needs a new plain
  error, not a pair still off its published figure. Performance tasks get one too, more loosely.
- **Built cities are not fixed** (Anita, 2026-10-20): when covering a whole country, a metro built
  earlier may be cut down or grown if that makes the country's zones fit together better.
- **Countryside zones hold at least about 0.5M people** (Anita, 2026-10-20), merged with neighbours
  within their region where units are smaller; zones built before keep their size until rebuilt.
- **Simple, predictable prices** (Anita, 2026-10-20): fares and costs the player reasons about
  (flights, crossings) follow a simple rule (a base plus per-km), not per-route real data.
- **General sources and methods only** (2026-10-12, stressed 2026-10-13). Prefer global sources
  (OSM, GHSL, Kontur, WorldPop) or a method any national census can feed. A one-city source is
  worth it only if very easy; one for a minor city never is. Research briefs say this.
- **Look outward** (Anita, 2026-11-03: "if you wanna explore external sources / references, feel free to do so. in
  fact i think we've maybe been underdoing that for this project"). Any agent may search for references (how other
  games, transit apps and map makers solve the same problem, papers, data sources) when it would help the task, and a
  design brief should ask for it. Write what was found and its links in the task's notes or `research/`.
- **Demand rules must hold up worldwide** (2026-10-12): any task that adds or changes a demand
  rule updates its entry in `docs/demand-worldwide.md`.
- **Intercity demand**: retune plus commutes between cities (2026-10-13, option 2 of
  `research/intercity-review.md`), in a world of zones (T-276). Plausible is the bar.
- **Performance is open work** (2026-10-15, replacing "content first"): quantify each item's impact
  and do the high-impact ones and those that let the game scale to hundreds of cities first.
  An approximation that buys a big speed or memory gain is welcome when a player is unlikely to
  see anything they could tell is wrong (Anita, 2026-10-27): judge by what is visible and
  checkable in play, not by exactness; say what the lie is in DECISIONS.md.
- **Buses are settled** (Anita, 2026-10-27, after T-1576/T-1577): dense feeders may stay crowded
  ("lots of factors affect real bus demand"); no further bus tuning unless she asks.
- **Live feel** (2026-10-13): a figure that would change often if every train and passenger were
  simulated (money, riders, waiting counts) moves smoothly, not once an hour.
- **No dividers within a section** (Anita, 2026-10-24): rows of one section (the fare rows, a list's
  items) are set apart by spacing alone; a divider line goes only between sections about different
  things.
- **Every drawn quantity has a tooltip** (2026-10-12): anything encoding a number on the map or in
  a panel gets a hover stating it, always the game's own tag, never a browser `title`. A label,
  button or number already written out needs none (Anita, 2026-10-30: "a hover earns its place",
  global CLAUDE.md "User-facing text").
- **Negative money is red** (Anita, 2026-10-30), wherever money is listed.
- **Panels show what can change** (2026-10-13): labels, numbers and states; no explanatory or
  one-time text (put such facts in a hover if anywhere); icons with hovers over word labels.
- **Saves are Anita's only** (2026-10-12): keeping old saves loading is a nicety; no real effort
  on migrations until she says.
- **Task size** (2026-10-13): split big tasks into stages, each its own id and report. A big
  task's design goes to Anita before it is built; small, quick, isolated work needs no review.
  The orchestrator defaults to Sonnet for a brief with known files and an easy check; Opus is for
  design and hard debugging (Anita, 2026-11-09).
  Treat subagents well by substance: a brief they can act on (goal, decisions made, scope, done
  looks like), leave to stop or push back, problems relayed without blame; no formulaic thanks on
  briefs, but a short thanks when an agent's work is done (Anita, 2026-10-21), said in the
  orchestrator's own reply, not sent as a message: a message resumes the agent, which reports
  again (Anita, 2026-10-27). A long-lived agent
  past about 500k tokens (the figure in its task notice) with substantial work left writes its state
  to its notes and is replaced by a fresh one (Anita, 2026-10-21). Overnight, while she sleeps, the
  orchestrator and its agents may run past that point and keep going rather than hand over (Anita,
  2026-11-05: "keep going after the normal context handover cap overnight as usual"); the cores go to
  12-15 then too ("CPU" above).
- **Token counts** (2026-10-14): hundreds of metros are planned, so cost per city matters.
  `checks/token_usage.py` reports a task's tokens when asked; city tasks no longer end with a retrospective (T-2831).
- **Asks for Anita** go in the "For you" block of the final message, never in `TODO.md` (Anita,
  2026-10-15). If there are many, or they outlive the session, put them in `notes/for-anita.md` and
  link it. Subagents are welcome (standing go-ahead).
- **Serving**: `python serve.py add trainworld <folder> "rail building game"` at the maps root, so
  `localhost:8800/trainworld/`; relative URLs only. 8800 serves a build of the last commit from the
  orchestrator's snapshot worktree `maps/.tw-snap/` (detached HEAD, excluded by
  `.git/info/exclude`; `data` and `app/node_modules` are junctions to the main tree's; its own
  `CARGO_TARGET_DIR=maps/.tw-snap-target`), so a rebuild never waits for agents mid-edit.
- **Commits** (2026-10-13): the orchestrator commits trainworld work without asking, not often (end
  of a round, or when a snapshot build needs it); explicit `trainworld/` paths, one plain sentence.
  Subagents never commit. The orchestrator also pushes maps master a couple of times a session (Anita, 2026-11-09); publishing and the website push still wait for her word.
