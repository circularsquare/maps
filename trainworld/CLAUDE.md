# trainworld — notes for agents

You are the owner and long-term maintainer of this codebase, and its design is up to you. Leave the code better
than you found it, fix the cause rather than the case in front of you, and keep it simple enough for the next agent
to change. What the game is, and the calls under "What waits for her word", stay Anita's.

Read this, then in `SPEC.md` the top and sections 1 and 2 (the game's two names, what **open** means, the
priorities, the core performance idea), the first lines of section 11 (what a save holds), and the sections
your task touches. Grep `TASKS.md` for your task id rather than reading it. Everything else on demand:
for how a mechanic works, the wiki first (`wiki/index.md`, then `wiki/read/<page>.md`, which has the figures filled
in and ends with the doc section and code that own the rule), then the docs. Where code lives: `docs/code-map.md`.
The orchestrator also reads `docs/orchestrator.md`.

## Talking to Anita

- **Reports go to the orchestrator** and are short, about 15 lines (each report stays in the orchestrator's context
  for the rest of its session): what changed and where, check results, strings, anything for Anita (marked as hers).
  Detail goes in notes/.
- **She doesn't read the code.** Quote the value or list in the message, or name the place on 8800, never a file
  to open.
- **Estimates in hours of real agent time**, checked against similar past tasks (T-3137 C1, guessed at days, took
  about 45 minutes). Never "agent-days". Don't call something impossible by a time: our guesses run pessimistic, and
  work often lands sooner. No deadlines unless she gives one; aim for the next release and see.
- **Ask when the answer is needed next**, not ahead. Test items only once the change is on 8800.

## What waits for her word

Publishing the game and the website push; every server deploy (never overnight); adding a complicating factor to
demand; a choice that changes what the game is; a change to what players can build or how fast or far things go,
even when a player asked or a rail fact backs it (a request alone is no reason, as with 350 km/h high speed); anything the spec marks **open**; a big task's design before it is built. Everything else you decide, record in DECISIONS.md if someone might revisit it, and tell her after: rail
facts, sensible defaults, visible calls, zone and region names. A slip gets a fix, not a new standing process.

## Player text

- **Voice**: the "User-facing text" section of her global CLAUDE.md, the wiki included. "Trips" for what people want
  to take (demand); "rides" and "riders" only for what happens on the player's system.
- **Speed and waiting**, in game text and changelog alike, one phrase per wait: opening a game or
  city "loads"; the map catching up to an edit "shows up" ("deleting a station shows up faster"); the numbers
  updating after a change are "calculating trips" ("trips are calculated 10x faster"), never "settle", "demand" or
  "rides" for that wait; memory "uses N MB less memory".
- **Review**: list every in-game string you add or change, old -> new, exactly as shown. Landing doesn't wait; she
  answers "text acknowledged" or names lines. Propose wording; never ask her to write it. Changelog items are not
  asked about when you land: she reviews them separately on the board. (A publish still waits for her OK on new
  player text: the `publish` skill.)
- **Changelog**: a change a player would notice (a new control, a changed number, a fixed bug they'd have met)
  gets one item before you report. When in doubt, no item: she wants us "less willing to put stuff on the
  changelog". The command:
  `venv\Scripts\python.exe trainworld/tools/changelog.py add "<line>" --area <area> [--fix] --task T-#### --by <you>`
  - **The line**: one plain line, lowercase. Past tense for the change ("added ...", "fixed ..."), present for
    how things are now ("the game uses 500 MB less memory").
  - **Area**: track, demand, scheduling, fares, schematic (the map and schematic), community (accounts and
    sharing), performance; other if none fits. A bug fix takes its area plus `--fix`; `bug` only when no area
    fits.
  - **Already there**: if the unpublished changelog (`changelog.py list`) already has an item for the thing
    you changed, including the one that added it, reword that item (`changelog.py reword <id> "<line>"`)
    instead of adding one.
  - **No item for**: internals, tools and docs; a change too minor or too fiddly to describe for a player (undo
    after a load's cap lowering); a change that is only a tooltip, an alert's wording or diagnostic text, since
    that text is the whole change and players can read it.
  - List the item with your strings. Anita rewords or hides items on the board's Changelog view.
- **Not reviewed**: the wiki, zone/region/realm names, docs, commit messages, text copied word for word from her
  own message, and text reused word for word from a string already in the game; still list it in the report, marked as reused.
- "Manually", never "by hand" ("by hand is a bit weird").
- "Segment" for a stretch of track, never "run" (her 11-21: "i dont like the terminology runs in general").
- American wording: "turn around", never the British "turn round" or "round" for "around" (a pet peeve
  of hers).
- Watch for an object or colour as the subject ("grey fills"), dropped words ("from street zooms"), and hovers that
  say what the player can see. A gesture with no visible control gets its instruction in a tag ("Click to close the
  loop"); ordinary controls get none.

## Files

| File | Holds | Who edits |
|---|---|---|
| `SPEC.md` | What the game is and how it works, **as of now**, in summary | agents, rewritten in place |
| `docs/` | The spec in detail, one file per area (indexes: `interface.md`, `demand.md`); `adding-a-country.md`, `world-run.md`, `data-register.md`, `code-map.md`, `code-tour.md` (the parts and how data moves) | agents |
| `wiki/` | Pages for a curious player; how to edit: `wiki/RULES.md` | agents |
| `TASKS.md` | Open work, by priority | agents |
| `LINES.md` | Anita's lines: the conceptual units tasks group into, by area | agents |
| `DECISIONS.md`, `ARCHIVE.md` | Dated choices and why; finished tasks, newest first. Older: `*-to-2026-10-19.md` | agents, add only |
| `notes/` | `T-###.md` task notes, `handoff.md` (orchestrator: plan, her words), `asks.json` (orchestrator, only through `tools/asks.py`) | agents |
| `checks/` | Standing checks, `smoke.mjs` first | agents |
| `research/` | Background reports with sources | agents |
| `TODO.md` | Anita's own list | Anita only; agents never edit it |
| `docs/demand-plain.md` | Demand in plain words, for Anita | whoever changes a demand rule it mentions |

## Doc rules

- **The spec says what is true now.** Rewrite the affected text; never append "Update:" or leave a section a later
  one contradicts. A section past ~150 lines moves to `docs/<area>.md`. A small UI tweak changes a doc only if a
  rule changed; the code is the reference for details. A rule a wiki page states changes that page too.
- **Touch only the section your change touches.** Read sections, not whole files.
- **Docs hold current rules**: the rule and its deciding numbers. Measurements and how a figure was reached go in
  notes/. Don't write down what the code says better and will change soon.
- **Comments** only for a why the code doesn't show (a non-obvious choice, a unit, an invariant). Never what the code
  does, never history.
- **DECISIONS.md and ARCHIVE.md are grepped, never read whole** (the `-to-2026-10-19` files too). A DECISIONS line is
  for a choice someone might revisit: date, one or two sentences, task id. An ARCHIVE entry: id, title, date,
  outcome, notes link.
- **Tasks**: written to her and very short; leave out what you're unsure of. Examples: notes/T-3394.md.
  - **Head line**: `- T-012 [P2] {L:slug} Title`. The title is 2-10 words, no full stop, no colon. Every open
    task carries `{L:<slug>}` from `LINES.md` right after its priority, plus `{waits:Anita}` or
    `{after:<task id or phrase>}` when it waits.
  - **Fields**: indented, one line each, in this order.
    - `  - Need from Anita:` exactly when the head has `{waits:Anita}`: one sentence she can act on, with the
      recommended answer if there is one.
    - `  - Context:` where things stand and why.
    - `  - Goal:` what done is, with the deciding number. Left out when the title already says it, and never
      "these items are done".
    - `  - Other:` optional; the notes link last.
  - **Priorities**: P1 now, P2 this milestone, P3 later, P4 lowest (set by her on the board). Someday tasks are
    P4 and don't count as open.
  - **Ids and lines**: use only the ids your brief gives you. A new task joins an existing line or adds one; set
    a line's `done:` when its last task closes.
  - **Commands** (agents may use both): mark a task taken before starting with
    `venv\Scripts\python.exe trainworld/tools/tasks.py start <ids> --agent <name>` and archive it when done with
    `done <id> --outcome "..."`. `tasks.py check` lists malformed lines.
  - **Updating**: rewrite a task's fields, never append to the head line. When a task's main work is done but it
    stays open for what's left, rewrite the title, Context and Goal to be the part that's left, so the board
    shows what is still open.
- **Done** means it runs, the spec matches, any decision is recorded, and the task has moved to the top of
  ARCHIVE.md (`tasks.py done` does this). Then grep TASKS.md for the id, close a parent whose last stage this was, and
  name any other open task your work also did. New work found on the way is a new task.
- **Shared files**: re-read right before editing, small edits, never rewrite TASKS, DECISIONS or ARCHIVE wholesale.
  In a tree, TASKS.md, DECISIONS.md, ARCHIVE.md and LINES.md are never edited (a hook refuses): edit the main tree's,
  and run tasks.py from the maps root. Changelog items made in a tree are `<tree>-1`, `<tree>-2` until landing
  numbers them.
- **notes/** is the agents': move, merge, delete. Check outputs go in `../artifacts/trainworld/`. A notes link that
  doesn't resolve is in `notes/archive/<band>/`.
- **Data register**: a task that uses, finds or rules out a source updates that country's cell in
  `docs/data-register.md`.

## Working as an agent

- **Before starting**, `git status` and the "taking" marks; leave other agents' files alone.
- **Your own Vite**: `node trainworld/checks/dev-nowatch.mjs <port>` from the maps root (a free port in 6100-6199),
  restarted after app edits. 8800 is Anita's; no long sweeps against it. If your brief gives you a worktree, run
  everything from it.
- **Headless Chrome** through Playwright with its own `--user-data-dir`; stop only PIDs you started.
- **Playing the game in a check or a probe**: probe with `node trainworld/checks/play.mjs` (how-to at its top).
  `start <city|save> --port <your Vite>` keeps a game open in a headless Chrome of its own (cores claimed; it stops
  after 30 unused minutes); then `do '<act/see calls>'`, `sim <read>` and `edits <file>` answer in a second or two
  with short JSON, waiting on conditions (`wait.final()`, `wait.panel()`, `wait.selection()`); `reload` after
  restarting Vite, `stop` when done. A standing check makes the same calls through `checks/lib/play.mjs`, which picks
  tools on the real toolbar, draws track and places stations by pointer at lon/lat points, and reads panels and money
  as shown. Build fixtures by the player's path where it matters; a new read goes in `checks/lib/reads.mjs`.
- **The real save** is `T-007-real-nyc.save`. `?new=1` never writes a slot.
- **Typecheck** with `npm run check --prefix app`; `python trainworld/tools/agent_cargo.py wasm` when Rust changed.
  No full builds, no commits: landing your work is the orchestrator's commit (a brief that touches shonei-server
  may say to commit there; pushing and deploying stay the orchestrator's).
- **Rust**: keep the crate compiling between edits. After a sim/ change run `python trainworld/tools/agent_cargo.py
  test` (both suites on the fast `agent` profile: no LTO, incremental, about 70 s warm against 180 s for `cargo test
  --release`; it claims its cores in the ledger and releases them). Build your tree's WASM with `agent_cargo.py
  wasm`, not `npm run wasm`, which is the shipped build and stays for landing. Your tests link differently from the
  shipped build (DECISIONS 2026-11-18). Add a function to the WASM interface in Rust before any app code calls it.
- **CPU**: the cap is the ledger's (`python cores.py list`). Playwright checks claim for themselves
  (`checks/cores.mjs`); claim by hand for the pipeline, Rust and scripts that launch their own browser. A page's
  demand workers take about a core each: cap them to your claim.
- **Data runs, packs, joins, AWS**: read the rules at the top of `docs/world-run.md` first.
- **Shell**: scripts in a file in your scratchpad. Pin processes from PowerShell or a `.cmd`, never
  `cmd /c start ... /b` through Git Bash (it pops dialogs on her screen). Nothing may open a window on her screen: a
  process you start from code gets `windowsHide: true` (Node) or `creationflags=subprocess.CREATE_NO_WINDOW`
  (Python), since a background process has no console and each child without it flashes a terminal.
- **Screenshots** go in `../artifacts/trainworld/`. For UI work, judge your own screenshots before reporting.
- **Advisor** (when your session has one): sparingly, only for a big design decision nobody has made (her 11-23:
  "only for like big design decisions"). Not when stuck, not to confirm a cause, not for routine steps, wording,
  small UI, or a choice already going to Anita. Say in your report whether you called it and what changed.
- **A refused permission check**: make the step safer and retry; don't wait for Anita.
- **Stop at a clean point** if the work outgrows one context: finish the stage, file the rest, hand back.
- **Waiting**: no `sleep`, no polling loops of your own: `node trainworld/checks/wait.mjs <file> <pattern>
  <timeout-s> [--pid N] [--fail <pattern>]` for a log or build, `play.mjs do 'wait.final()'` for the game. A shell call
  still running at about 180 s can be moved to the background, so keep each wait under 170 s and wait again if it
  times out; never end your turn while something you need is running (a watch on a background process does not hold
  the turn open). Work that needs no result from you may run in the background, with at most one light check every
  4-5 minutes. Final checks (smoke) write a log; wait on it with `wait.mjs` and read its result before you report.
- **Testing, kept lean** (she found the work "a bit too test heavy"): run the checks near your change, not every
  check. In the game, one run that shows the change working; breadth (edge cases, other cities, undo chains) is the
  batch checker's, which tests risky changes together on master after landing. Smoke only when your brief asks or the change touches sim, saves, demand or money; the orchestrator runs one
  smoke per landed batch. Add a standing check only for a bug that really happened and could quietly come back;
  otherwise prove the change with a scratch script and leave it in your scratch folder. Before writing a scratch
  script, grep `checks/INDEX.md` (made by `tools/checks_index.py`) and reuse a check that already proves it.

## Game principles (Anita's)

- **The whole world**, a low-fidelity global build (tier 0) everywhere first; tailoring country by country later,
  popular countries first.
- **Demand is simple on purpose**: sensible and fun over fitting real cities. Targets are her two ceiling tests (rail
  share of rail + car: A at least 85%, B at least 50%; notes/T-636.md). Propose any complicating factor to her first.
  A demand rule change updates `docs/demand-worldwide.md`.
- **Every demand task has a "done when"**: numbers that end it, plus "would a player see something obviously wrong?".
  Once met, stop. Performance tasks get a looser one.
- **General sources**: global datasets or a method any census can feed; a one-city source only if very easy.
  Local demand factors (by country, region or city kind) are allowed again where a worldwide rule misses a real gap
  and published figures show it; a place that already looks right (the US intercity) keeps the worldwide rule. Look
  outward for how other games and map makers solve it; write what you find in notes or research/.
- **Performance**: quantify impact, do the big ones first. An approximation is welcome when a player couldn't tell;
  say what the lie is in DECISIONS.md.
- **Simple prices**: fares and costs the player reasons about follow a simple rule (a base plus per km).
- **Buses are settled**; no more bus tuning unless she asks.
- **Saves are hers**: keeping old ones loading is a nicety, no real migration effort.
- **UI principles** live in `docs/ui-panels.md` "Look" and "Hover tags".

## Build and run

`app/` (Vite, TypeScript, MapLibre, Preact), `sim/` (Rust to WASM), `pipeline/` (Python city packs,
`..\venv\Scripts\python.exe`). Build orders: `docs/code-map.md`.
